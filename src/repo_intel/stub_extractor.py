"""Stub extractor — lightweight, non-LLM symbol stubs for mentioned files.

Goal: give small models a *structural map* (functions/classes/methods + signatures + docs)
without injecting full file contents.

This is used by the TaskRouter context prefetcher and is safe to run eagerly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import structlog

from src.repo_intel.parser import ASTParser

logger = structlog.get_logger(__name__)


StubKind = Literal["function", "class", "method", "type"]


@dataclass(frozen=True)
class SymbolStub:
    kind: StubKind
    name: str
    signature: str
    line: int
    doc: str | None = None


_SUPPORTED_EXTS: set[str] = {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java"}


def extract_stubs(path: Path, *, max_symbols: int = 200) -> list[SymbolStub]:
    """Extract symbol stubs from a single file.

    Returns stubs in source order and caps the number of returned symbols.

    Notes:
    - Best-effort: if Tree-sitter parsing fails, falls back to line-regex extraction.
    - Never returns full file content; only short stub strings.
    """
    if max_symbols <= 0:
        return []

    ext = path.suffix.lower()
    if ext not in _SUPPORTED_EXTS:
        return []

    try:
        source_text = path.read_text(encoding="utf-8", errors="replace")
        source_bytes = source_text.encode("utf-8", errors="replace")
        lines = source_text.splitlines()
    except OSError as e:
        logger.debug("stub_extractor.read_failed", file=str(path), error=str(e))
        return []

    parser = ASTParser()
    tree = None
    try:
        tree = parser.parse_file(path)
    except Exception as e:  # pragma: no cover
        logger.debug("stub_extractor.parse_failed", file=str(path), error=str(e))
        tree = None

    if tree is None:
        return _fallback_regex_stubs(lines, ext, max_symbols=max_symbols)

    # Tree-sitter path
    stubs: list[SymbolStub] = []

    # Query catalog per language. We capture the *declaration node* itself.
    queries: list[tuple[str, StubKind]] = []
    if ext == ".py":
        queries = [
            ("(function_definition) @def", "function"),
            ("(class_definition) @def", "class"),
        ]
    elif ext in {".js", ".jsx", ".ts", ".tsx"}:
        queries = [
            ("(function_declaration) @def", "function"),
            ("(class_declaration) @def", "class"),
            ("(method_definition) @def", "method"),
        ]
    elif ext == ".go":
        queries = [
            ("(function_declaration) @def", "function"),
            ("(method_declaration) @def", "method"),
            ("(type_spec name: (type_identifier) @name) @def", "type"),
        ]
    elif ext == ".rs":
        queries = [
            ("(function_item) @def", "function"),
            ("(struct_item) @def", "type"),
            ("(enum_item) @def", "type"),
            ("(trait_item) @def", "type"),
            ("(impl_item) @impl", "type"),
        ]
    elif ext == ".java":
        queries = [
            ("(class_declaration) @def", "class"),
            ("(interface_declaration) @def", "type"),
            ("(method_declaration) @def", "method"),
            ("(constructor_declaration) @def", "method"),
        ]

    def _safe_name(node) -> str:
        try:
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                return name_node.text.decode("utf-8", errors="replace")
        except Exception:
            pass
        # Some nodes capture name differently (Go type_spec query captures @name separately)
        try:
            # Heuristic: first identifier-like child
            for child in node.children:
                t = getattr(child, "type", "")
                if t in {"identifier", "type_identifier", "field_identifier", "scoped_identifier"}:
                    return child.text.decode("utf-8", errors="replace")
        except Exception:
            pass
        return ""

    # Collect decl nodes in source order (by start_byte)
    decls: list[tuple[int, object, StubKind]] = []
    for q, kind in queries:
        captures = parser.query(tree, ext, q)
        for cap_node, cap_tag in captures:
            # When query captures @name (Go), the @def capture is a type_spec node.
            # Prefer the @def node itself.
            node_obj = cap_node
            if cap_tag in {"def", "impl"} or cap_tag.endswith("def"):
                decls.append((getattr(node_obj, "start_byte", 0), node_obj, kind))

    decls.sort(key=lambda t: t[0])

    for _start_byte, node_obj, kind in decls:
        if len(stubs) >= max_symbols:
            break

        if ext == ".py" and kind == "function" and _is_inside_python_class(node_obj):
            kind = "method"

        name = _safe_name(node_obj)
        if not name:
            # Rust impl blocks can be huge; skip unnamed.
            continue

        start_line = int(getattr(node_obj, "start_point", (0, 0))[0]) + 1
        signature = _extract_signature(source_bytes, node_obj, ext)
        doc = None
        if ext == ".py":
            doc = _extract_python_docstring(source_text, node_obj)
            if not doc:
                doc = _extract_python_docstring_fallback(lines, start_line)
        if not doc:
            doc = _extract_doc(lines, start_line)

        stubs.append(SymbolStub(kind=kind, name=name, signature=signature, line=start_line, doc=doc))

    # Rust: pull methods from impl blocks in fallback mode (tree-sitter impl methods are non-trivial).
    if ext == ".rs" and len(stubs) < max_symbols:
        extra = _fallback_regex_stubs(lines, ext, max_symbols=max_symbols - len(stubs))
        stubs.extend(extra)

    return stubs[:max_symbols]


def render_stub_map(
    stubs_by_file: dict[Path, list[SymbolStub]],
    *,
    max_chars: int = 20_000,
) -> str:
    """Render a stable stub map string suitable for Planner injection."""
    if not stubs_by_file:
        return ""

    chunks: list[str] = ["Prefetched File Stubs (auto)"]
    for file_path in sorted(stubs_by_file.keys(), key=lambda p: str(p).lower()):
        stubs = stubs_by_file[file_path]
        rel = str(file_path).replace("\\", "/")
        chunks.append(f"\n## {rel}")
        for s in stubs:
            sig = s.signature.strip() or s.name
            line = f"- {s.kind} {s.name} @L{s.line}: {sig}"
            chunks.append(line)
            if s.doc:
                doc_line = _one_line(s.doc)
                if doc_line:
                    chunks.append(f"  doc: {doc_line}")

        if sum(len(c) + 1 for c in chunks) > max_chars:
            chunks.append("\n[Truncated: stub map exceeded size cap]")
            break

    rendered = "\n".join(chunks).strip()
    return rendered[:max_chars]


def _one_line(text: str) -> str:
    t = " ".join(text.strip().split())
    return t[:240]


def _extract_signature(source: bytes, node_obj: object, ext: str) -> str:
    """Extract a single-line signature from the declaration start."""
    start = int(getattr(node_obj, "start_byte", 0) or 0)
    if start < 0 or start >= len(source):
        return ""

    terminators = {"{", ";"}
    if ext == ".py":
        terminators = {":"}
    elif ext in {".go"}:
        terminators = {"{"}
    elif ext in {".rs"}:
        terminators = {"{", ";"}
    elif ext == ".java":
        terminators = {"{", ";"}

    # Scan forward while tracking paren depth (handles multiline python signatures).
    max_scan = min(len(source), start + 1200)
    depth = 0
    end = start
    for i in range(start, max_scan):
        b = source[i]
        ch = chr(b)
        if ch == "(":
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
        if depth == 0 and ch in terminators:
            end = i + 1
            break
        # If we hit a newline and have something and we're not mid-signature, stop.
        if ch == "\n" and i > start + 5 and ext != ".py" and depth == 0:
            end = i
            break
    else:
        end = max_scan

    sig = source[start:end].decode("utf-8", errors="replace")
    sig = " ".join(sig.replace("\r", "").replace("\n", " ").split())
    return sig[:320]


def _extract_doc(lines: list[str], start_line: int, *, lookback: int = 10) -> str | None:
    """Best-effort doc comment extraction from preceding comment block."""
    if start_line <= 1 or not lines:
        return None

    idx = start_line - 2
    doc_lines: list[str] = []

    def _is_comment(line: str) -> bool:
        s = line.strip()
        return (
            s.startswith("#")
            or s.startswith("//")
            or s.startswith("/*")
            or s.startswith("*")
        )

    scanned = 0
    while idx >= 0 and scanned < lookback:
        line = lines[idx]
        if not line.strip():
            # blank line breaks doc block once we started collecting
            if doc_lines:
                break
            idx -= 1
            scanned += 1
            continue
        if _is_comment(line):
            doc_lines.append(line.strip().lstrip("#/").strip())
            idx -= 1
            scanned += 1
            continue
        break

    if not doc_lines:
        return None

    doc_lines.reverse()
    doc = " ".join(d for d in doc_lines if d)
    return doc.strip() or None


def _is_inside_python_class(node_obj: Any) -> bool:
    """Return True if node has a class_definition ancestor (Tree-sitter Python)."""
    try:
        p = getattr(node_obj, "parent", None)
        while p is not None:
            if getattr(p, "type", None) == "class_definition":
                return True
            p = getattr(p, "parent", None)
    except Exception:
        return False
    return False


def _extract_python_docstring(source_text: str, node_obj: Any) -> str | None:
    """Best-effort extraction of Python docstring from the first statement in the body."""
    try:
        body = node_obj.child_by_field_name("body")
        if body is None:
            return None
        children = getattr(body, "children", None) or []
        if not children:
            return None

        first = children[0]
        if getattr(first, "type", None) != "expression_statement":
            return None
        expr_children = getattr(first, "children", None) or []
        string_node = None
        for c in expr_children:
            if getattr(c, "type", None) == "string":
                string_node = c
                break
        if string_node is None:
            return None

        start = int(getattr(string_node, "start_byte", -1))
        end = int(getattr(string_node, "end_byte", -1))
        if start < 0 or end <= start:
            return None

        raw = source_text.encode("utf-8", errors="replace")[start:end].decode("utf-8", errors="replace")

        # Safe evaluation of string literal to get the actual docstring text.
        import ast

        try:
            val = ast.literal_eval(raw)
        except Exception:
            return None
        if not isinstance(val, str):
            return None
        doc = " ".join(val.strip().split())
        return doc[:1000] if doc else None
    except Exception:
        return None


def _fallback_regex_stubs(lines: list[str], ext: str, *, max_symbols: int) -> list[SymbolStub]:
    import re

    stubs: list[SymbolStub] = []

    if ext == ".py":
        func_re = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
        cls_re = re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\b")
        for i, line in enumerate(lines, 1):
            m = func_re.match(line)
            if m:
                doc = _extract_python_docstring_fallback(lines, i) or _extract_doc(lines, i)
                stubs.append(SymbolStub("function", m.group(1), line.strip()[:320], i, doc))
            m = cls_re.match(line)
            if m:
                doc = _extract_python_docstring_fallback(lines, i) or _extract_doc(lines, i)
                stubs.append(SymbolStub("class", m.group(1), line.strip()[:320], i, doc))
            if len(stubs) >= max_symbols:
                break
        return stubs

    # Generic fallback for other langs
    patterns: list[tuple[re.Pattern, StubKind]] = []
    if ext in {".js", ".jsx", ".ts", ".tsx"}:
        patterns = [
            (re.compile(r"^\s*function\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\("), "function"),
            (re.compile(r"^\s*class\s+([A-Za-z_$][A-Za-z0-9_$]*)\b"), "class"),
        ]
    elif ext == ".go":
        patterns = [
            (re.compile(r"^\s*func\s+\(?([A-Za-z_][A-Za-z0-9_]*)?\)?\s*([A-Za-z_][A-Za-z0-9_]*)\s*\("), "function"),
            (re.compile(r"^\s*type\s+([A-Za-z_][A-Za-z0-9_]*)\s+"), "type"),
        ]
    elif ext == ".rs":
        patterns = [
            (re.compile(r"^\s*(pub\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\b"), "function"),
            (re.compile(r"^\s*(pub\s+)?struct\s+([A-Za-z_][A-Za-z0-9_]*)\b"), "type"),
            (re.compile(r"^\s*(pub\s+)?enum\s+([A-Za-z_][A-Za-z0-9_]*)\b"), "type"),
            (re.compile(r"^\s*trait\s+([A-Za-z_][A-Za-z0-9_]*)\b"), "type"),
            (re.compile(r"^\s*impl\s+([A-Za-z_][A-Za-z0-9_<>:]*)\s*\b"), "type"),
        ]
    elif ext == ".java":
        patterns = [
            (re.compile(r"^\s*(public|protected|private)?\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\b"), "class"),
            (re.compile(r"^\s*(public|protected|private)?\s*interface\s+([A-Za-z_][A-Za-z0-9_]*)\b"), "type"),
        ]

    for i, line in enumerate(lines, 1):
        for pat, kind in patterns:
            m = pat.match(line)
            if not m:
                continue
            name = m.group(m.lastindex or 1)
            stubs.append(SymbolStub(kind, name, line.strip()[:320], i, _extract_doc(lines, i)))
            break
        if len(stubs) >= max_symbols:
            break

    return stubs


def _extract_python_docstring_fallback(lines: list[str], decl_line: int, *, max_lookahead: int = 40) -> str | None:
    """Best-effort docstring extraction using line scanning.

    Looks for a triple-quoted string as the first statement in the declaration body.
    """
    if decl_line <= 0 or decl_line > len(lines):
        return None

    decl = lines[decl_line - 1]
    base_indent = len(decl) - len(decl.lstrip(" \t"))

    # Find first non-empty body line with greater indentation.
    for j in range(decl_line, min(len(lines), decl_line + max_lookahead)):
        body_line = lines[j]
        if not body_line.strip():
            continue
        indent = len(body_line) - len(body_line.lstrip(" \t"))
        if indent <= base_indent:
            return None

        s = body_line.lstrip()
        if not (s.startswith('"""') or s.startswith("'''")):
            return None

        quote = s[:3]
        rest = s[3:]
        if quote in rest:
            content = rest.split(quote, 1)[0]
            doc = " ".join(content.strip().split())
            return doc or None

        # Multi-line docstring
        parts: list[str] = [rest.rstrip("\r\n")]
        for k in range(j + 1, min(len(lines), decl_line + max_lookahead)):
            line_k = lines[k]
            if quote in line_k:
                before = line_k.split(quote, 1)[0]
                parts.append(before)
                break
            parts.append(line_k)
        doc = " ".join("\n".join(parts).strip().split())
        return doc or None

    return None

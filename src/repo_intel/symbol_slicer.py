"""Symbol slicer — extract *only* specific symbol definitions from a file.

Used to keep worker context small: provide the exact definition blocks referenced
by a plan node (e.g. a single function) instead of entire files or raw pasted
code blobs.

Python slicing uses the builtin `ast` module for robust lineno/end_lineno.
Other languages fall back to conservative regex signature blocks.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

import structlog

from src.core.schemas import CodeSnippet

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class SliceConfig:
    max_snippets_total: int = 8
    max_snippet_chars: int = 6000
    max_total_chars: int = 18_000


def slice_symbols(
    path: Path,
    symbols: list[str],
    *,
    cfg: SliceConfig | None = None,
) -> list[CodeSnippet]:
    cfg = cfg or SliceConfig()
    if not symbols or cfg.max_snippets_total <= 0:
        return []

    ext = path.suffix.lower()
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.debug("symbol_slicer.read_failed", file=str(path), error=str(e))
        return []

    if ext == ".py":
        return _slice_python(path, text, symbols, cfg=cfg)

    # Conservative fallback: keep only declaration lines (signature-ish).
    return _slice_signatures_fallback(path, text, symbols, cfg=cfg)


def _slice_python(path: Path, text: str, symbols: list[str], *, cfg: SliceConfig) -> list[CodeSnippet]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _slice_signatures_fallback(path, text, symbols, cfg=cfg)

    wanted = set(s for s in symbols if s)
    if not wanted:
        return []

    lines = text.splitlines()

    def _node_range(n: ast.AST) -> tuple[int, int] | None:
        lineno = getattr(n, "lineno", None)
        end_lineno = getattr(n, "end_lineno", None)
        if lineno is None:
            return None
        if end_lineno is None:
            # Best effort if Python < 3.8 or missing end_lineno
            return (int(lineno), int(lineno))
        return (int(lineno), int(end_lineno))

    snippets: list[CodeSnippet] = []
    used_chars = 0

    for node in ast.walk(tree):
        if len(snippets) >= cfg.max_snippets_total:
            break
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        name = getattr(node, "name", "")
        if name not in wanted:
            continue

        r = _node_range(node)
        if r is None:
            continue
        start, end = r
        start = max(1, start)
        end = max(start, min(end, len(lines)))

        content = "\n".join(lines[start - 1 : end]).rstrip() + "\n"
        if len(content) > cfg.max_snippet_chars:
            content = content[: cfg.max_snippet_chars] + "\n# [Truncated: slice exceeded size cap]\n"

        if used_chars + len(content) > cfg.max_total_chars:
            break

        snippets.append(
            CodeSnippet(
                file_path=str(path).replace("\\", "/"),
                start_line=start,
                end_line=end,
                content=content,
                language="python",
                summary=f"Sliced definition for symbol '{name}'",
            )
        )
        used_chars += len(content)

    return snippets


def _slice_signatures_fallback(
    path: Path,
    text: str,
    symbols: list[str],
    *,
    cfg: SliceConfig,
) -> list[CodeSnippet]:
    lines = text.splitlines()
    wanted = [s for s in symbols if s]
    if not wanted:
        return []

    # Match only the signature/declaration line.
    decls: list[tuple[int, str]] = []
    for i, line in enumerate(lines, 1):
        for sym in wanted:
            if re.search(rf"\b{re.escape(sym)}\b", line):
                if any(k in line for k in ("def ", "class ", "function ", "fn ", "type ")):
                    decls.append((i, line.strip()))
                    break
        if len(decls) >= cfg.max_snippets_total:
            break

    snippets: list[CodeSnippet] = []
    used_chars = 0
    for i, sig in decls:
        content = sig[: min(len(sig), cfg.max_snippet_chars)] + "\n"
        if used_chars + len(content) > cfg.max_total_chars:
            break
        snippets.append(
            CodeSnippet(
                file_path=str(path).replace("\\", "/"),
                start_line=i,
                end_line=i,
                content=content,
                language=_guess_language(path),
                summary="Sliced signature (fallback)",
            )
        )
        used_chars += len(content)
    return snippets


def _guess_language(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".py":
        return "python"
    if ext in {".js", ".jsx"}:
        return "javascript"
    if ext in {".ts", ".tsx"}:
        return "typescript"
    if ext == ".go":
        return "go"
    if ext == ".rs":
        return "rust"
    if ext == ".java":
        return "java"
    return "text"

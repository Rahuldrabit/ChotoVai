"""
Tree-sitter AST parser wrapper for multi-language syntax tree extraction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
import tree_sitter
import tree_sitter_python
import tree_sitter_javascript
import tree_sitter_typescript
import tree_sitter_go
import tree_sitter_rust
import tree_sitter_java

logger = structlog.get_logger(__name__)

# Map file extensions to the actual tree-sitter language object
LANGUAGE_EXTENSIONS = {
    ".py": tree_sitter_python.language(),
    ".js": tree_sitter_javascript.language(),
    ".jsx": tree_sitter_javascript.language(),
    ".ts": tree_sitter_typescript.language_typescript(),
    ".tsx": tree_sitter_typescript.language_tsx(),
    ".go": tree_sitter_go.language(),
    ".rs": tree_sitter_rust.language(),
    ".java": tree_sitter_java.language(),
}


class ASTParser:
    """Wrapper around tree-sitter to parse source files."""

    def __init__(self) -> None:
        self._parsers: dict[str, tree_sitter.Parser] = {}

    def _get_parser(self, extension: str) -> tree_sitter.Parser | None:
        """Lazy load and return a tree-sitter parser for the extension."""
        if extension not in LANGUAGE_EXTENSIONS:
            return None

        if extension not in self._parsers:
            parser = tree_sitter.Parser()
            lang = tree_sitter.Language(LANGUAGE_EXTENSIONS[extension])
            # tree_sitter Python bindings have changed over time:
            # - older: Parser.set_language(Language)
            # - newer: Parser.language = Language
            if hasattr(parser, "set_language"):
                parser.set_language(lang)  # type: ignore[attr-defined]
            else:
                parser.language = lang  # type: ignore[attr-defined]
            self._parsers[extension] = parser

        return self._parsers[extension]

    def parse_file(self, filepath: str | Path) -> tree_sitter.Tree | None:
        """Parse a source file and return its AST."""
        path = Path(filepath)
        if not path.is_file():
            return None

        parser = self._get_parser(path.suffix)
        if not parser:
            logger.debug("ast_parser.unsupported_extension", ext=path.suffix)
            return None

        try:
            content = path.read_bytes()
            return parser.parse(content)
        except Exception as e:
            logger.warning("ast_parser.parse_failed", file=str(path), error=str(e))
            return None

    def query(
        self, tree: tree_sitter.Tree, extension: str, query_string: str
    ) -> list[tuple[tree_sitter.Node, str]]:
        """Run a tree-sitter query against a parsed tree.

        This returns a stable list of (Node, capture_name) tuples across
        tree_sitter binding versions.
        """
        if extension not in LANGUAGE_EXTENSIONS:
            return []

        def _normalize_captures(caps: Any, query_obj: Any) -> list[tuple[tree_sitter.Node, str]]:
            # Newer QueryCursor.captures returns: {capture_name: [Node, ...]}
            if isinstance(caps, dict):
                out: list[tuple[tree_sitter.Node, str]] = []
                for cap_name, nodes in caps.items():
                    for n in nodes:
                        out.append((n, str(cap_name)))
                return out

            # Some bindings return an iterable of (Node, capture_id/name).
            out: list[tuple[tree_sitter.Node, str]] = []
            try:
                for item in caps:
                    if not isinstance(item, tuple) or len(item) != 2:
                        continue
                    n, cap = item
                    if isinstance(cap, int) and hasattr(query_obj, "capture_name"):
                        try:
                            cap = query_obj.capture_name(cap)
                        except Exception:
                            cap = str(cap)
                    out.append((n, str(cap)))
            except TypeError:
                return []
            return out

        try:
            lang = tree_sitter.Language(LANGUAGE_EXTENSIONS[extension])

            # Build a query object.
            query_obj: Any
            if hasattr(tree_sitter, "Query"):
                query_obj = tree_sitter.Query(lang, query_string)  # type: ignore[attr-defined]
            else:
                query_obj = lang.query(query_string)  # type: ignore[deprecated]

            root = tree.root_node

            # Prefer QueryCursor when available.
            if hasattr(tree_sitter, "QueryCursor"):
                cursor = tree_sitter.QueryCursor(query_obj)  # type: ignore[attr-defined]
                caps = cursor.captures(root)
                return _normalize_captures(caps, query_obj)

            # Fallback: older bindings use Query.captures(root)
            if hasattr(query_obj, "captures"):
                caps = query_obj.captures(root)
                return _normalize_captures(caps, query_obj)

            return []
        except Exception as e:
            logger.warning("ast_parser.query_failed", error=str(e))
            return []

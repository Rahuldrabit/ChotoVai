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
            parser.set_language(lang)
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

    def query(self, tree: tree_sitter.Tree, extension: str, query_string: str) -> list[tuple[tree_sitter.Node, str]]:
        """Run a tree-sitter query against a parsed tree."""
        if extension not in LANGUAGE_EXTENSIONS:
            return []
        
        try:
            lang = tree_sitter.Language(LANGUAGE_EXTENSIONS[extension])
            query = lang.query(query_string)
            return query.captures(tree.root_node)
        except Exception as e:
            logger.warning("ast_parser.query_failed", error=str(e))
            return []

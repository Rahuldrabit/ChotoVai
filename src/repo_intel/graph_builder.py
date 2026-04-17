"""
GraphBuilder — ingestor that runs AST extraction and translates it into Cypher queries for the GraphStore.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import structlog

from src.repo_intel.parser import ASTParser
from src.repo_intel.graph_store import GraphStore

logger = structlog.get_logger(__name__)


class GraphBuilder:
    def __init__(self, parser: ASTParser, store: GraphStore) -> None:
        self.parser = parser
        self.store = store

    def ingest_directory(self, root_dir: str | Path, exts: tuple[str, ...] = (".py",)) -> None:
        """Scan directory and ingest all supported files into the graph."""
        root = Path(root_dir)
        for filepath in root.rglob("*"):
            if filepath.is_file() and filepath.suffix in exts:
                self.ingest_file(str(filepath))

    def ingest_file(self, filepath: str) -> None:
        """Parse a single file and load its symbols into Kuzu."""
        tree = self.parser.parse_file(filepath)
        if not tree:
            return

        path = Path(filepath)
        ext = path.suffix

        # Create File node
        self.store.execute(
            "MERGE (f:File {filepath: $path}) ON CREATE SET f.extension = $ext",
            {"path": filepath, "ext": ext}
        )

        # Basic language queries (Python specific as fallback demo — in production, extract per language)
        if ext == ".py":
            class_query = """
            (class_definition
              name: (identifier) @class.name) @class.def
            """
            func_query = """
            (function_definition
              name: (identifier) @func.name
              body: (block . (expression_statement (string) @func.docstring))?) @func.def
            """

            class_captures = self.parser.query(tree, ext, class_query)
            # Process classes and functions, map them to DB...
            # In MVP, we do a basic structural loop to ingest nodes

            current_class_id = None
            
            for node, tag in class_captures:
                if tag == "class.name":
                    class_name = node.text.decode('utf8')
                    # Hashed ID for uniqueness
                    class_id = hashlib.md5(f"{filepath}::{class_name}".encode()).hexdigest()
                    
                    self.store.execute(
                        "MERGE (c:Class {id: $id}) "
                        "ON CREATE SET c.name = $name, c.filepath = $path",
                        {"id": class_id, "name": class_name, "path": filepath}
                    )
                    self.store.execute(
                        "MATCH (c:Class {id: $id}), (f:File {filepath: $path}) "
                        "MERGE (c)-[:DEFINED_IN_FILE]->(f)",
                        {"id": class_id, "path": filepath}
                    )

            # Ingest functions
            func_captures = self.parser.query(tree, ext, func_query)
            functions_in_file: list[tuple[str, str]] = []  # (func_id, func_name)
            for node, tag in func_captures:
                if tag == "func.name":
                    func_name = node.text.decode('utf8')
                    func_id = hashlib.md5(f"{filepath}::{func_name}".encode()).hexdigest()
                    functions_in_file.append((func_id, func_name))
                    
                    self.store.execute(
                        "MERGE (fn:Function {id: $id}) "
                        "ON CREATE SET fn.name = $name, fn.filepath = $path",
                        {"id": func_id, "name": func_name, "path": filepath}
                    )
                    self.store.execute(
                        "MATCH (fn:Function {id: $id}), (f:File {filepath: $path}) "
                        "MERGE (fn)-[:DEFINED_IN_FILE]->(f)",
                        {"id": func_id, "path": filepath}
                    )

            # Heuristic CALLS edges (MVP):
            # - We don't do full symbol resolution yet.
            # - We add CALLS edges when a function body contains a textual `callee(` that matches
            #   another function defined in the same file.
            #
            # This is good enough to power callers/callees queries for intra-file flows, which is
            # where most "find the endpoint handler" exploration starts.
            try:
                source = Path(filepath).read_text(encoding="utf-8", errors="replace")
            except Exception:
                source = ""

            if source and functions_in_file:
                import re

                names = [n for _, n in functions_in_file]
                name_to_id = {n: i for i, n in functions_in_file}

                # Very rough: locate "def <name>(" blocks and scan until next top-level def/class.
                # This will miss nested defs and methods; those are Phase 5+ work.
                def_spans: dict[str, tuple[int, int]] = {}
                for m in re.finditer(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", source, re.MULTILINE):
                    fn = m.group(1)
                    if fn in name_to_id:
                        def_spans[fn] = (m.start(), len(source))
                # End each span at the next def/class start.
                starts = sorted([(pos, fn) for fn, (pos, _) in def_spans.items()])
                for idx, (pos, fn) in enumerate(starts):
                    end = starts[idx + 1][0] if idx + 1 < len(starts) else len(source)
                    def_spans[fn] = (pos, end)

                for caller_name, (start, end) in def_spans.items():
                    body = source[start:end]
                    caller_id = name_to_id[caller_name]
                    for callee_name in names:
                        if callee_name == caller_name:
                            continue
                        # Count occurrences of direct calls like "foo(".
                        count = len(re.findall(rf"\\b{re.escape(callee_name)}\\s*\\(", body))
                        if count <= 0:
                            continue
                        callee_id = name_to_id[callee_name]
                        self.store.execute(
                            """
                            MATCH (a:Function {id: $caller}), (b:Function {id: $callee})
                            MERGE (a)-[c:CALLS]->(b)
                            ON CREATE SET c.count = $count
                            ON MATCH SET c.count = c.count + $count
                            """,
                            {"caller": caller_id, "callee": callee_id, "count": int(count)},
                        )
        
        logger.info("graph_builder.ingested", file=filepath)

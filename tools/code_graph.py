"""
Code graph tools (fixed query catalog).

These are intentionally *not* free-form Cypher. The Explorer chooses an operation
name; the runtime executes a bounded, parameterized query.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import structlog

from src.repo_intel.graph_store import GraphStore

logger = structlog.get_logger(__name__)


OpName = Literal[
    "find_function",
    "callers",
    "callees",
    "module_neighborhood",
    "top_modules",
]


_DEFAULT_LIMIT = 50


def _graph() -> GraphStore:
    # Lazily open embedded DB (Kuzu) via GraphStore.
    store = GraphStore()
    store.init_schema()
    return store


def code_graph(op: OpName, params: dict[str, Any]) -> dict[str, Any]:
    """
    Dispatch a code graph catalog query.

    Args:
        op:     Query name (catalog operation).
        params: Operation parameters.
    """
    store = _graph()

    limit = int(params.get("limit", _DEFAULT_LIMIT))
    if limit <= 0:
        limit = _DEFAULT_LIMIT
    limit = min(limit, 200)

    if op == "find_function":
        name = str(params.get("name", "")).strip()
        if not name:
            return {"items": []}
        rows = store.execute(
            """
            MATCH (fn:Function)
            WHERE fn.name = $name OR fn.id = $name
            RETURN fn.id AS id, fn.name AS name, fn.filepath AS filepath,
                   fn.start_line AS start_line, fn.end_line AS end_line
            LIMIT $limit
            """,
            {"name": name, "limit": limit},
        )
        return {"items": rows}

    if op == "callers":
        fn_id = str(params.get("id", "")).strip()
        if not fn_id:
            return {"items": []}
        rows = store.execute(
            """
            MATCH (caller:Function)-[c:CALLS]->(target:Function)
            WHERE target.id = $id
            RETURN caller.id AS caller_id, caller.name AS caller_name, caller.filepath AS caller_file,
                   c.count AS count
            LIMIT $limit
            """,
            {"id": fn_id, "limit": limit},
        )
        return {"items": rows}

    if op == "callees":
        fn_id = str(params.get("id", "")).strip()
        if not fn_id:
            return {"items": []}
        rows = store.execute(
            """
            MATCH (f:Function)-[c:CALLS]->(callee:Function)
            WHERE f.id = $id
            RETURN callee.id AS callee_id, callee.name AS callee_name, callee.filepath AS callee_file,
                   c.count AS count
            LIMIT $limit
            """,
            {"id": fn_id, "limit": limit},
        )
        return {"items": rows}

    if op == "module_neighborhood":
        filepath = str(params.get("filepath", "")).strip()
        if not filepath:
            return {"module": None, "contains": [], "imports": [], "imported_by": []}

        # Normalize path for keying; GraphBuilder stores absolute paths today.
        p = Path(filepath)
        filepath_norm = str(p)

        module = store.execute(
            """
            MATCH (f:File {filepath: $path})
            RETURN f.filepath AS filepath, f.extension AS extension
            LIMIT 1
            """,
            {"path": filepath_norm},
        )
        contains = store.execute(
            """
            MATCH (fn:Function)-[:DEFINED_IN_FILE]->(f:File {filepath: $path})
            RETURN fn.id AS id, fn.name AS name
            LIMIT $limit
            """,
            {"path": filepath_norm, "limit": limit},
        )
        imports = store.execute(
            """
            MATCH (f:File {filepath: $path})-[:IMPORTS]->(dep:File)
            RETURN dep.filepath AS filepath
            LIMIT $limit
            """,
            {"path": filepath_norm, "limit": limit},
        )
        imported_by = store.execute(
            """
            MATCH (src:File)-[:IMPORTS]->(f:File {filepath: $path})
            RETURN src.filepath AS filepath
            LIMIT $limit
            """,
            {"path": filepath_norm, "limit": limit},
        )
        return {
            "module": module[0] if module else None,
            "contains": contains,
            "imports": imports,
            "imported_by": imported_by,
        }

    if op == "top_modules":
        # Simple heuristic: return the most recently ingested files by filepath sort.
        rows = store.execute(
            """
            MATCH (f:File)
            RETURN f.filepath AS filepath, f.extension AS extension
            LIMIT $limit
            """,
            {"limit": limit},
        )
        return {"items": rows}

    logger.warning("code_graph.unknown_op", op=op)
    return {"error": f"unknown op: {op}"}


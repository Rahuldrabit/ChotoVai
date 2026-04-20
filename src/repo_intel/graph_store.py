"""
Interface for KuzuDB, an embedded property graph database for storing repository intelligence.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import kuzu
    _KUZU_AVAILABLE = True
except ImportError:
    kuzu = None  # type: ignore[assignment]
    _KUZU_AVAILABLE = False
import structlog

from src.core.config import get_config

logger = structlog.get_logger(__name__)


class GraphStore:
    """
    Manages the KuzuDB connection and Code Property Graph schema.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        cfg = get_config()
        # Default to a kuzu_db folder inside the data_dir
        self._db_path = Path(db_path or Path(cfg.data_dir) / "kuzu_db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._db = kuzu.Database(str(self._db_path))
        self._conn = kuzu.Connection(self._db)
        logger.info("graph_store.connected", path=str(self._db_path))

    def init_schema(self) -> None:
        """Idempotent creation of the code property graph schema."""
        schema_statements = [
            # Nodes
            "CREATE NODE TABLE IF NOT EXISTS File (filepath STRING, extension STRING, PRIMARY KEY (filepath))",
            "CREATE NODE TABLE IF NOT EXISTS Class (id STRING, name STRING, filepath STRING, start_line INT64, end_line INT64, PRIMARY KEY (id))",
            "CREATE NODE TABLE IF NOT EXISTS Function (id STRING, name STRING, filepath STRING, start_line INT64, end_line INT64, docstring STRING, PRIMARY KEY (id))",
            "CREATE NODE TABLE IF NOT EXISTS Module (name STRING, PRIMARY KEY (name))",
            
            # Edges
            "CREATE REL TABLE IF NOT EXISTS DEFINED_IN_FILE (FROM Class TO File, FROM Function TO File)",
            "CREATE REL TABLE IF NOT EXISTS DEFINED_IN_CLASS (FROM Function TO Class)",
            "CREATE REL TABLE IF NOT EXISTS CALLS (FROM Function TO Function, count INT64)",
            "CREATE REL TABLE IF NOT EXISTS IMPORTS (FROM File TO File)",
            "CREATE REL TABLE IF NOT EXISTS BELONGS_TO (FROM File TO Module)"
        ]

        for stmt in schema_statements:
            try:
                self._conn.execute(stmt)
            except RuntimeError as e:
                # Kuzu raises RuntimeError if table exists despite IF NOT EXISTS in some older versions
                if "already exists" not in str(e).lower():
                    logger.warning("graph_store.schema_error", stmt=stmt, error=str(e))
        
        logger.info("graph_store.schema_initialized")

    def execute(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """
        Execute a Cypher query against KuzuDB.
        """
        params = parameters or {}
        try:
            result = self._conn.execute(query, params)
            
            # Convert result to list of dicts based on column names
            # Kuzu's Python API provides get_column_names() and has_next()
            ret = []
            if result.has_next():
                columns = result.get_column_names()
                while result.has_next():
                    row = result.get_next()
                    ret.append(dict(zip(columns, row)))
            return ret
        except Exception as e:
            logger.error("graph_store.query_failed", query=query, error=str(e))
            return []

    def close(self) -> None:
        """Close connection (optional, Kuzu cleans up automatically, but good practice)."""
        pass

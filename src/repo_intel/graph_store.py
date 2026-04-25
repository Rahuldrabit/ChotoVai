"""
Interface for graph backends used to store repository intelligence.
"""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any

try:
    import kuzu
    _KUZU_AVAILABLE = True
except ImportError:
    kuzu = None  # type: ignore[assignment]
    _KUZU_AVAILABLE = False

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable
    _NEO4J_AVAILABLE = True
except ImportError:
    GraphDatabase = None  # type: ignore[assignment]
    ServiceUnavailable = Exception  # type: ignore[assignment]
    _NEO4J_AVAILABLE = False
import structlog

from src.core.config import get_config

logger = structlog.get_logger(__name__)


class GraphStore:
    """
    Manages the graph connection and Code Property Graph schema.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        cfg = get_config()
        repo_cfg = cfg.repo_intel

        self._backend = repo_cfg.graph_db_backend
        self._db_path: Path | None = None
        self._db = None
        self._conn = None
        self._driver = None

        if self._backend == "neo4j":
            self._init_neo4j(
                uri=repo_cfg.neo4j_uri,
                user=repo_cfg.neo4j_user,
                password=repo_cfg.neo4j_password,
            )
            return

        self._init_kuzu(
            db_path=db_path,
            configured_graph_path=repo_cfg.graph_db_path,
        )

    def _init_kuzu(self, db_path: str | Path | None, configured_graph_path: str) -> None:
        if not _KUZU_AVAILABLE:
            raise RuntimeError(
                "Kuzu backend selected but dependency is not installed. "
                "Install package 'kuzu'."
            )

        self._db_path = Path(db_path or configured_graph_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._db = kuzu.Database(str(self._db_path))
        self._conn = kuzu.Connection(self._db)
        logger.info("graph_store.connected", backend="kuzu", path=str(self._db_path))

    def _init_neo4j(self, uri: str, user: str, password: str) -> None:
        if not _NEO4J_AVAILABLE:
            raise RuntimeError(
                "Neo4j backend selected but dependency is not installed. "
                "Install package 'neo4j'."
            )

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        max_attempts = 10
        delay_s = 0.2
        max_delay_s = 2.0
        last_err: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                self._driver.verify_connectivity()
                last_err = None
                break
            except ServiceUnavailable as e:
                last_err = e
                if attempt >= max_attempts:
                    break

                logger.warning(
                    "graph_store.neo4j_connect_retry",
                    attempt=attempt,
                    max_attempts=max_attempts,
                    delay_s=round(delay_s, 3),
                    error=str(e),
                )
                time.sleep(delay_s)
                delay_s = min(delay_s * 2, max_delay_s)

        if last_err is not None:
            try:
                self._driver.close()
            except Exception:
                pass
            raise RuntimeError(
                f"Neo4j not ready after {max_attempts} attempts: {last_err}"
            ) from last_err

        logger.info("graph_store.connected", backend="neo4j", uri=uri, user=user)

    def init_schema(self) -> None:
        """Idempotent creation of the code property graph schema."""
        if self._backend == "neo4j":
            self._init_schema_neo4j()
            return

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
        
        logger.info("graph_store.schema_initialized", backend="kuzu")

    def _init_schema_neo4j(self) -> None:
        schema_statements = [
            "CREATE CONSTRAINT file_filepath_unique IF NOT EXISTS FOR (f:File) REQUIRE f.filepath IS UNIQUE",
            "CREATE CONSTRAINT class_id_unique IF NOT EXISTS FOR (c:Class) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT function_id_unique IF NOT EXISTS FOR (fn:Function) REQUIRE fn.id IS UNIQUE",
            "CREATE CONSTRAINT module_name_unique IF NOT EXISTS FOR (m:Module) REQUIRE m.name IS UNIQUE",
        ]

        with self._driver.session() as session:
            for stmt in schema_statements:
                try:
                    session.run(stmt).consume()
                except Exception as e:
                    logger.warning("graph_store.schema_error", stmt=stmt, error=str(e))

        logger.info("graph_store.schema_initialized", backend="neo4j")

    def execute(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """
        Execute a Cypher query against the configured backend.
        """
        params = parameters or {}
        try:
            if self._backend == "neo4j":
                with self._driver.session() as session:
                    result = session.run(query, params)
                    return [record.data() for record in result]

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
            logger.error("graph_store.query_failed", backend=self._backend, query=query, error=str(e))
            return []

    def close(self) -> None:
        """Close backend connections."""
        if self._backend == "neo4j" and self._driver is not None:
            self._driver.close()

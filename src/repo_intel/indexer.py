"""
Hierarchical Indexer — translates code graph structures into nested textual summaries.
"""
from __future__ import annotations

import structlog

from src.repo_intel.graph_store import GraphStore

logger = structlog.get_logger(__name__)


class RepoIndexer:
    """
    Summarizes Code Property Graph at different granularities.
    Uses SummarizerAgent to compress blocks of code into embeddings.
    """

    def __init__(self, store: GraphStore) -> None:
        self.store = store

    async def index_all(self) -> None:
        """
        Iterates over the database to build hierarchical summaries.
        In the MVP, this reads KuzuDB and calls the Agent.
        """
        logger.info("repo_indexer.start_indexing")
        
        # 1. Summarize Functions -> update DB
        # 2. Summarize Classes -> update DB
        # 3. Summarize Files -> update DB
        # 4. Push final embeddings to Qdrant semantic layer
        
        # Skeleton implementation for MVP:
        files = self.store.execute("MATCH (f:File) RETURN f.filepath AS path")
        for f in files:
             path = f.get("path")
             logger.debug("repo_indexer.summarizing_file", path=path)
             # Extract string content, send to Summarizer, push to Qdrant...

        logger.info("repo_indexer.indexing_complete")

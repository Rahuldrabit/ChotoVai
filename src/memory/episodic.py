"""
Episodic memory — append-only trajectory store backed by Qdrant.
Each entry captures a completed agent action with outcome metadata.
Retrieval: top-k by cosine similarity to the current task embedding.
"""
from __future__ import annotations

import asyncio
from typing import Sequence
from uuid import UUID

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    Filter,
    FieldCondition,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.core.config import get_config
from src.core.schemas import EpisodicEntry, ValidationOutcome

logger = structlog.get_logger(__name__)


class EpisodicStore:
    """
    Asynchronous episodic memory backed by Qdrant.

    Usage:
        store = EpisodicStore()
        await store.initialize()     # once
        await store.insert(entry)
        results = await store.retrieve(query_text, top_k=3)
    """

    def __init__(self) -> None:
        cfg = get_config().memory
        self._client = AsyncQdrantClient(url=cfg.qdrant_url)
        self._collection = cfg.qdrant_collection_episodic
        self._embedding_model_name = cfg.embedding_model
        self._dim = cfg.embedding_dim
        self._top_k = cfg.episodic_top_k
        self._embedder = None  # Lazy-loaded

    async def initialize(self) -> None:
        """Create the Qdrant collection if it doesn't exist."""
        try:
            existing = await self._client.get_collections()
            names = [c.name for c in existing.collections]
            if self._collection not in names:
                await self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
                )
                logger.info("episodic_store.created", collection=self._collection)
            else:
                logger.debug("episodic_store.connected", collection=self._collection)
        except Exception as e:
            logger.warning("episodic_store.init_failed", error=str(e))

    async def insert(self, entry: EpisodicEntry) -> None:
        """Embed and insert an episodic entry."""
        text = f"{entry.task_description} {entry.action_summary}"
        embedding = await self._embed(text)
        entry.embedding = embedding

        payload = entry.model_dump(exclude={"embedding"}, mode="json")
        # Qdrant needs a deterministic int or str id
        point_id = str(entry.id)
        point = PointStruct(id=point_id, vector=embedding, payload=payload)
        try:
            await self._client.upsert(collection_name=self._collection, points=[point])
        except Exception as e:
            logger.warning("episodic_store.insert_failed", error=str(e))

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        outcome_filter: ValidationOutcome | None = None,
    ) -> list[EpisodicEntry]:
        """Retrieve top-k most similar episodic entries."""
        embedding = await self._embed(query)
        k = top_k or self._top_k

        query_filter = None
        if outcome_filter:
            query_filter = Filter(
                must=[FieldCondition(key="outcome", match=MatchValue(value=outcome_filter.value))]
            )

        try:
            results = await self._client.search(
                collection_name=self._collection,
                query_vector=embedding,
                limit=k,
                query_filter=query_filter,
                with_payload=True,
            )
            entries = []
            for r in results:
                try:
                    entries.append(EpisodicEntry(**r.payload))
                except Exception:
                    pass
            return entries
        except Exception as e:
            logger.warning("episodic_store.retrieve_failed", error=str(e))
            return []

    async def _embed(self, text: str) -> list[float]:
        """Embed text using sentence-transformers (lazy-loaded)."""
        if self._embedder is None:
            await self._load_embedder()
        # sentence-transformers encode is synchronous — run in thread pool
        loop = asyncio.get_event_loop()
        vector = await loop.run_in_executor(None, self._embedder.encode, text)
        # Our fallback embedder returns list[float] already.
        return vector.tolist() if hasattr(vector, "tolist") else list(vector)

    async def _load_embedder(self) -> None:
        """
        Try to load SentenceTransformer. If it isn't installed (or pulls heavyweight deps like torch),
        fall back to a deterministic hash embedder so the system stays usable in "light" mode.
        """
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            loop = asyncio.get_event_loop()
            self._embedder = await loop.run_in_executor(
                None, SentenceTransformer, self._embedding_model_name
            )
            logger.info("episodic_store.embedder_loaded", model=self._embedding_model_name)
        except Exception as e:
            logger.warning("episodic_store.embedder_fallback", error=str(e))

            class _HashEmbedder:
                def __init__(self, dim: int) -> None:
                    self.dim = dim

                def encode(self, s: str) -> list[float]:
                    import hashlib

                    # Expand sha256 into dim floats in [-1, 1].
                    h = hashlib.sha256(s.encode("utf-8", errors="ignore")).digest()
                    out: list[float] = []
                    while len(out) < self.dim:
                        for b in h:
                            out.append((b / 127.5) - 1.0)
                            if len(out) >= self.dim:
                                break
                        h = hashlib.sha256(h).digest()
                    return out

            self._embedder = _HashEmbedder(self._dim)
            logger.info("episodic_store.embedder_loaded", model="hash_embedder", dim=self._dim)

"""Recursive summarization utility ("Tracker").

This is used to compress very large user inputs (goal + pasted code) before
intent reasoning and planning, so the Planner never needs raw large code blobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

import structlog

logger = structlog.get_logger(__name__)


SummarizeFn = Callable[[str], Awaitable[str]]


@dataclass(frozen=True)
class RecursiveSummarizerConfig:
    trigger_chars: int = 20_000
    chunk_chars: int = 6_000
    target_chars: int = 6_000
    max_rounds: int = 6


class RecursiveSummarizer:
    def __init__(self, summarize_fn: SummarizeFn, cfg: RecursiveSummarizerConfig | None = None):
        self._summarize_fn = summarize_fn
        self._cfg = cfg or RecursiveSummarizerConfig()

    @property
    def cfg(self) -> RecursiveSummarizerConfig:
        return self._cfg

    def should_summarize(self, text: str) -> bool:
        return len(text) >= self._cfg.trigger_chars

    async def summarize(self, text: str, *, hint: str | None = None) -> str:
        """Recursively summarize until under target size or rounds exhausted."""
        if not text:
            return ""

        if len(text) <= self._cfg.target_chars:
            return text

        current = text
        for round_idx in range(self._cfg.max_rounds):
            if len(current) <= self._cfg.target_chars:
                break

            chunks = _chunk(current, self._cfg.chunk_chars)
            summaries: list[str] = []
            for idx, chunk in enumerate(chunks):
                prompt = chunk
                if hint:
                    prompt = f"{hint}\n\n{chunk}"
                try:
                    s = await self._summarize_fn(prompt)
                except Exception as e:  # pragma: no cover
                    logger.debug(
                        "recursive_summarizer.summarize_failed",
                        error=str(e),
                        round=round_idx,
                        chunk=idx,
                    )
                    # Deterministic fallback: truncate chunk
                    s = chunk[: max(800, min(len(chunk), self._cfg.target_chars // max(1, len(chunks))))]
                summaries.append(s.strip())

            merged = "\n\n".join(summaries).strip()
            if not merged:
                break

            current = merged

        if len(current) > self._cfg.target_chars:
            current = current[: self._cfg.target_chars]
        return current


def _chunk(text: str, chunk_chars: int) -> list[str]:
    if chunk_chars <= 0:
        return [text]

    parts: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_chars)
        # Try to split on paragraph boundary
        split = text.rfind("\n\n", start, end)
        if split != -1 and split > start + 500:
            end = split
        parts.append(text[start:end].strip())
        start = end

    return [p for p in parts if p]

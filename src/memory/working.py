"""
Working memory — per-agent rolling context buffer with auto-compaction.
Compacts at ~92% of the configured token threshold by calling the Summarizer.
"""
from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import structlog

from src.core.config import get_config
from src.core.schemas import AgentMessage, AgentRole, ContextPack, PlanNode

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    pass


class WorkingMemory:
    """
    Rolling message buffer for a single agent invocation.
    Tracks token usage and triggers compaction when near the threshold.
    """

    def __init__(self, max_tokens: int | None = None) -> None:
        cfg = get_config().memory
        self._max_tokens = max_tokens or cfg.working_memory_max_tokens
        self._threshold = get_config().memory.working_memory_compaction_threshold
        self._messages: deque[AgentMessage] = deque()
        self._total_tokens: int = 0

    def add(self, message: AgentMessage) -> None:
        token_count = message.token_count or _estimate_tokens(message.content)
        message.token_count = token_count
        self._messages.append(message)
        self._total_tokens += token_count

    @property
    def messages(self) -> list[AgentMessage]:
        return list(self._messages)

    @property
    def utilization(self) -> float:
        return self._total_tokens / self._max_tokens if self._max_tokens else 0.0

    @property
    def should_compact(self) -> bool:
        return self.utilization >= self._threshold

    def compact_with_summary(self, summary: str) -> None:
        """
        Replace all but the last 2 messages with a single summary message.
        Called after a Summarizer agent produces a compressed summary.
        """
        recent = list(self._messages)[-2:]
        summary_msg = AgentMessage(
            role="system",
            content=f"[Compressed context summary]\n{summary}",
            token_count=_estimate_tokens(summary),
        )
        self._messages.clear()
        self._messages.append(summary_msg)
        for m in recent:
            self._messages.append(m)
        self._total_tokens = sum(m.token_count or 0 for m in self._messages)
        logger.debug("working_memory.compacted", remaining_tokens=self._total_tokens)

    def clear(self) -> None:
        self._messages.clear()
        self._total_tokens = 0


def _estimate_tokens(text: str) -> int:
    """Fast token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)

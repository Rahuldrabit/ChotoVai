"""Scratchpad compaction ("Janitor").

Hard-enforces bounded scratchpad growth by truncating in place:
- Keep file header (first lines up to the first separator)
- Summarize the older body into a single Janitor summary entry
- Keep the last N characters unmodified

This is intentionally conservative: it only triggers when the scratchpad exceeds
configured size caps.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import structlog

from src.memory.recursive_summarizer import RecursiveSummarizer
from src.memory.scratchpad import SessionScratchpad

logger = structlog.get_logger(__name__)


_SEPARATOR = "\n---\n"


@dataclass(frozen=True)
class JanitorConfig:
    max_chars_before_compact: int = 50_000
    keep_recent_chars: int = 12_000
    summary_target_chars: int = 3_500


class ScratchpadJanitor:
    def __init__(self, summarizer: RecursiveSummarizer, cfg: JanitorConfig | None = None):
        self._summarizer = summarizer
        self._cfg = cfg or JanitorConfig()

    async def maybe_compact(self, scratchpad: SessionScratchpad) -> bool:
        """Returns True if compaction happened."""
        try:
            path = scratchpad.path
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            logger.debug("janitor.read_failed", error=str(e))
            return False

        if len(text) < self._cfg.max_chars_before_compact:
            return False

        header, body = _split_header_body(text)
        if not body.strip():
            return False

        keep_tail = body[-self._cfg.keep_recent_chars :]
        old_part = body[: max(0, len(body) - len(keep_tail))].strip()
        if not old_part:
            return False

        summary = await self._summarizer.summarize(
            old_part,
            hint=(
                "Summarize the scratchpad history into compact bullet points. "
                "Preserve decisions, constraints, open questions, and TODOs. "
                "Do not include verbatim code." 
            ),
        )
        summary = summary[: self._cfg.summary_target_chars].strip()

        ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        janitor_entry = (
            f"[{ts}] [janitor]\n"
            f"Compacted scratchpad history summary:\n{summary}\n"
        )

        new_body = (janitor_entry + _SEPARATOR + keep_tail.lstrip("\n")).strip("\n")
        new_text = (header + _SEPARATOR + new_body).strip("\n") + "\n"

        try:
            path.write_text(new_text, encoding="utf-8")
        except OSError as e:
            logger.debug("janitor.write_failed", error=str(e))
            return False

        logger.info(
            "janitor.compacted",
            before_chars=len(text),
            after_chars=len(new_text),
            keep_recent_chars=self._cfg.keep_recent_chars,
        )
        return True


def _split_header_body(text: str) -> tuple[str, str]:
    """Split header and body at the first separator."""
    if _SEPARATOR.strip() not in text:
        return "", text

    parts = text.split(_SEPARATOR, 1)
    header = parts[0].strip("\n")
    body = parts[1].strip("\n") if len(parts) > 1 else ""
    return header, body

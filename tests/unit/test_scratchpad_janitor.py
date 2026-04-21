from __future__ import annotations

from pathlib import Path

import pytest

from src.memory.recursive_summarizer import RecursiveSummarizer, RecursiveSummarizerConfig
from src.memory.scratchpad import SessionScratchpad
from src.memory.scratchpad_janitor import JanitorConfig, ScratchpadJanitor


async def _fake_summarize(text: str) -> str:
    # Deterministic summary for tests
    t = " ".join(text.split())
    return ("SUMMARY: " + t)[:200]


@pytest.mark.asyncio
async def test_janitor_compacts_in_place(tmp_path: Path) -> None:
    sp = SessionScratchpad(tmp_path / "scratchpad.md")

    for i in range(40):
        sp.append("x" * 80 + f" entry={i}", role="tester", node_id=f"N{i}")

    before = sp.path.read_text(encoding="utf-8")
    assert len(before) > 800

    tracker = RecursiveSummarizer(
        summarize_fn=_fake_summarize,
        cfg=RecursiveSummarizerConfig(trigger_chars=0, chunk_chars=300, target_chars=200, max_rounds=2),
    )
    janitor = ScratchpadJanitor(
        tracker,
        cfg=JanitorConfig(max_chars_before_compact=700, keep_recent_chars=250, summary_target_chars=120),
    )

    did = await janitor.maybe_compact(sp)
    assert did is True

    after = sp.path.read_text(encoding="utf-8")
    assert "[janitor]" in after
    assert len(after) < len(before)
    # Should keep recent tail content
    assert "entry=39" in after

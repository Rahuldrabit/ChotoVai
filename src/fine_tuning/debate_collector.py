"""
Debate trace collector — records completed debate games as training data.

Implements the data-collection half of the IMAGINE framework:
  - Multi-agent debate runs in production (or simulation)
  - DebateTraceCollector serialises each completed DebateState
  - Traces are exported as multi-turn SFT data (ShareGPT / Alpaca format)
  - These traces are then consumed by ImagineTrainer for LoRA fine-tuning

Storage: local JSONL files under data/debate_traces/ (queryable, portable).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from src.core.schemas import DebateOutcome, DebateState

logger = structlog.get_logger(__name__)


class DebateTraceCollector:
    """
    Records completed debate interactions for IMAGINE-style distillation.

    Usage:
        collector = DebateTraceCollector()
        collector.record(debate_state)          # call after each debate
        n = collector.export_for_sft(out_path)  # export for fine-tuning
    """

    def __init__(self, traces_dir: str | Path | None = None) -> None:
        self._dir = Path(traces_dir or "./data/debate_traces")
        self._dir.mkdir(parents=True, exist_ok=True)
        self._buffer: list[dict[str, Any]] = []

    # ──────────────────────────────────────────────────────────────────────
    # Collection
    # ──────────────────────────────────────────────────────────────────────

    def record(self, debate: DebateState) -> None:
        """
        Serialise a completed DebateState into the trace buffer.
        Only records debates with a terminal outcome (not IN_PROGRESS).
        """
        if debate.outcome == DebateOutcome.IN_PROGRESS:
            logger.warning("debate_collector.skip_in_progress", node_id=debate.node_id)
            return

        trace = {
            "id": str(debate.id),
            "node_id": debate.node_id,
            "task": debate.task_description,
            "success_criteria": debate.success_criteria,
            "outcome": debate.outcome.value,
            "final_score": debate.critic_score,
            "turn_count": debate.turn_count,
            "tokens_used": debate.tokens_used,
            "moves": [m.model_dump(mode="json") for m in debate.moves],
            "final_code": debate.current_code,
            "recorded_at": datetime.utcnow().isoformat(),
        }
        self._buffer.append(trace)

        # Flush to disk every 10 traces to avoid data loss
        if len(self._buffer) >= 10:
            self._flush()

    def flush(self) -> None:
        """Force-flush the buffer to disk."""
        self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        out = self._dir / f"traces_{ts}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for trace in self._buffer:
                f.write(json.dumps(trace, ensure_ascii=False) + "\n")
        logger.info("debate_collector.flushed", path=str(out), count=len(self._buffer))
        self._buffer.clear()

    # ──────────────────────────────────────────────────────────────────────
    # Export — convert traces to fine-tuning format
    # ──────────────────────────────────────────────────────────────────────

    def export_for_sft(self, output_path: Path | str) -> int:
        """
        Convert all collected JSONL traces into ShareGPT multi-turn format
        ready for SFT with Unsloth / TRL.

        Each winning debate becomes one training example:
          - system: task description + success criteria
          - human/gpt turns: alternating Coder arguments and Critic rebuttals
          - final gpt turn: the winning code (positive label)

        Losing / deadlocked debates are exported as negative examples
        with the correction_hint as a supervision signal.

        Returns:
            Number of exported training examples.
        """
        self._flush()  # Ensure all buffered data is on disk

        output_path = Path(output_path)
        all_traces: list[dict[str, Any]] = []

        for jsonl_file in sorted(self._dir.glob("traces_*.jsonl")):
            with jsonl_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_traces.append(json.loads(line))

        if not all_traces:
            logger.warning("debate_collector.export_empty")
            return 0

        sft_examples: list[dict[str, Any]] = []
        for trace in all_traces:
            example = self._trace_to_sharegpt(trace)
            if example:
                sft_examples.append(example)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for ex in sft_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        logger.info(
            "debate_collector.exported",
            path=str(output_path),
            total_traces=len(all_traces),
            valid_examples=len(sft_examples),
        )
        return len(sft_examples)

    @staticmethod
    def _trace_to_sharegpt(trace: dict[str, Any]) -> dict[str, Any] | None:
        """Convert one raw debate trace into ShareGPT conversation format."""
        moves = trace.get("moves", [])
        if not moves:
            return None

        criteria_str = "\n".join(
            f"- {c}" for c in trace.get("success_criteria", [])
        )
        system = (
            f"Task: {trace['task']}\n\n"
            f"Success Criteria:\n{criteria_str}\n\n"
            f"You are an expert software engineer in a structured adversarial "
            f"debate. Produce correct, production-ready code that withstands "
            f"adversarial critique."
        )

        conversations: list[dict[str, str]] = []
        for move in moves:
            if move["actor"] == "coder":
                conversations.append({"from": "gpt", "value": move["content"]})
            elif move["actor"] == "critic":
                score = move.get("score", 0)
                failing = "\n".join(move.get("failing_tests", []))
                value = f"[CRITIC SCORE: {score}/10]\n{move['content']}"
                if failing:
                    value += f"\n\nFailing tests:\n{failing}"
                conversations.append({"from": "human", "value": value})
            elif move["actor"] == "judge":
                # Judge verdict — treat as final ground truth
                conversations.append({"from": "human", "value": f"[JUDGE]: {move['content']}"})

        if not conversations:
            return None

        return {
            "id": trace["id"],
            "source": "debate_traces",
            "outcome": trace["outcome"],
            "final_score": trace.get("final_score", 0),
            "system": system,
            "conversations": conversations,
        }

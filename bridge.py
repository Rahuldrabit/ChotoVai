#!/usr/bin/env python3
"""
bridge.py — NDJSON bridge between the ChotoVai VS Code extension and AgentFSM.

Usage:
    python bridge.py "<goal>"

Writes one JSON object per line to stdout.
All structlog / Rich tracing output goes to stderr so stdout stays clean.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports resolve
# regardless of the working directory the extension sets.
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Keep Rich / structlog off stdout
os.environ.setdefault("STRUCTLOG_PROCESSOR", "plain")


def emit(obj: dict) -> None:
    """Write one JSON line to stdout and flush immediately."""
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


async def main(goal: str) -> None:
    try:
        from src.orchestrator.fsm import AgentFSM  # type: ignore[import]
        from src.core.config import get_config      # type: ignore[import]
    except ImportError as exc:
        emit({"event": "error", "message": f"Import error: {exc}"})
        sys.exit(1)

    cfg = get_config()
    fsm = AgentFSM(data_dir=cfg.data_dir)

    try:
        async for event in fsm.run(goal):
            emit(event)
    except Exception as exc:
        emit({"event": "error", "message": str(exc)})
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        emit({"event": "error", "message": "Usage: bridge.py <goal>"})
        sys.exit(1)

    # Join all argv so the extension can pass the goal without shell quoting
    goal = " ".join(sys.argv[1:])
    asyncio.run(main(goal))

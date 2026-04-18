"""
SessionScratchpad — append-only external reasoning log per session.

Agents write thoughts, decisions, and summaries here so that long-running
tasks don't require the full history to fit in one context window. Each agent
reads only the tail (last N chars) when assembling context, and can read
a specific node's section if it needs deeper history.

File layout  (one per session):
  ./data/sessions/<session_id>/scratchpad.md

Entry format:
  ---
  [<ISO timestamp>] [<role>] [node:<node_id>]
  <free-text entry>
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

# ── Module-level singleton ────────────────────────────────────────────────────
_active: "SessionScratchpad | None" = None
_lock = threading.Lock()


def set_active_scratchpad(session_dir: Path) -> "SessionScratchpad":
    """Create (or reuse) the scratchpad for the current session."""
    global _active
    sp = SessionScratchpad(session_dir / "scratchpad.md")
    with _lock:
        _active = sp
    return sp


def get_active_scratchpad() -> "SessionScratchpad | None":
    return _active


# ── Class ─────────────────────────────────────────────────────────────────────

class SessionScratchpad:
    """
    Append-only markdown log for a single agent session.

    Thread-safe — multiple agents may append concurrently.
    All reads are chunk-based so the model never has to load the full file.
    """

    _SEPARATOR = "---"

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file_lock = threading.Lock()

        # Write a header if the file is new
        if not self._path.exists():
            self._path.write_text(
                "# Session Scratchpad\n"
                "_Append-only reasoning log. Do not edit manually._\n\n",
                encoding="utf-8",
            )

    # ── Write ─────────────────────────────────────────────────────────────────

    def append(self, entry: str, role: str = "", node_id: str = "") -> None:
        """Append one entry to the scratchpad (thread-safe)."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        parts = [f"[{ts}]"]
        if role:
            parts.append(f"[{role}]")
        if node_id:
            parts.append(f"[node:{node_id}]")
        header = " ".join(parts)

        block = f"\n{self._SEPARATOR}\n{header}\n{entry.strip()}\n"
        with self._file_lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(block)

        logger.debug("scratchpad.appended", node_id=node_id, role=role, chars=len(entry))

    # ── Read ──────────────────────────────────────────────────────────────────

    def read_tail(self, max_chars: int = 4000) -> str:
        """
        Read the last `max_chars` characters of the scratchpad.
        Used by context assembler to inject recent reasoning into every agent call.
        """
        if not self._path.exists():
            return ""
        try:
            with self._path.open("rb") as f:
                f.seek(0, 2)  # seek to end
                size = f.tell()
                start = max(0, size - max_chars)
                f.seek(start)
                raw = f.read()
            text = raw.decode("utf-8", errors="replace")
            # Don't start mid-entry — trim to the first separator
            if start > 0 and self._SEPARATOR in text:
                text = text[text.index(self._SEPARATOR):]
            return text.strip()
        except OSError as e:
            logger.warning("scratchpad.read_tail_failed", error=str(e))
            return ""

    def read_node(self, node_id: str) -> str:
        """
        Return all scratchpad entries tagged with `node_id`.
        Used by critics and re-planners to inspect a specific node's history.
        """
        if not self._path.exists():
            return ""
        try:
            content = self._path.read_text(encoding="utf-8")
        except OSError:
            return ""

        sections: list[str] = []
        current: list[str] = []
        in_target = False

        for line in content.splitlines():
            if line.strip() == self._SEPARATOR:
                if in_target and current:
                    sections.append("\n".join(current))
                current = []
                in_target = False
                continue
            # Header line detection
            if current == [] and f"[node:{node_id}]" in line:
                in_target = True
            if in_target:
                current.append(line)

        if in_target and current:
            sections.append("\n".join(current))

        return "\n---\n".join(sections)

    def read_by_role(self, role: str, max_chars: int = 4000) -> str:
        """
        Return scratchpad entries from a specific role, up to max_chars (most recent).
        Useful for agents querying what another role has previously recorded.
        """
        if not self._path.exists():
            return ""
        try:
            content = self._path.read_text(encoding="utf-8")
        except OSError:
            return ""

        tag = f"[{role}]"
        sections: list[str] = []
        current: list[str] = []
        in_target = False

        for line in content.splitlines():
            if line.strip() == self._SEPARATOR:
                if in_target and current:
                    sections.append("\n".join(current))
                current = []
                in_target = False
                continue
            # Header line: check for exact role tag before appending to current block
            if not current and tag in line:
                in_target = True
            if in_target:
                current.append(line)

        if in_target and current:
            sections.append("\n".join(current))

        if not sections:
            return ""

        # Join all matching sections, then trim to most recent max_chars
        joined = "\n---\n".join(sections)
        if len(joined) > max_chars:
            trimmed = joined[-max_chars:]
            if self._SEPARATOR in trimmed:
                trimmed = trimmed[trimmed.index(self._SEPARATOR):]
            return trimmed.strip()
        return joined.strip()

    @property
    def path(self) -> Path:
        return self._path

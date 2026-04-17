"""
MCP tool: read_file — read file contents with optional line range.
Returns file content as text in a structured result.
"""
from __future__ import annotations

import os
from pathlib import Path


def read_file(path: str, start_line: int | None = None, end_line: int | None = None) -> str:
    """
    Read file contents at the given path.

    Args:
        path: Absolute or relative file path.
        start_line: 1-indexed first line to return (optional).
        end_line: 1-indexed last line to return (optional).

    Returns:
        File contents as a string, optionally sliced to the requested range.
    """
    p = Path(path)
    if not p.exists():
        return f"ERROR: File not found: {path}"
    if not p.is_file():
        return f"ERROR: Path is not a file: {path}"

    # Safety: cap file size to avoid flooding context
    max_bytes = 500 * 1024  # 500 KB
    if p.stat().st_size > max_bytes:
        return f"ERROR: File too large ({p.stat().st_size} bytes). Use grep or sliced read."

    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except PermissionError:
        return f"ERROR: Permission denied: {path}"

    if start_line is None and end_line is None:
        return text

    lines = text.splitlines(keepends=True)
    s = (start_line - 1) if start_line else 0
    e = end_line if end_line else len(lines)
    return "".join(lines[s:e])


# MCP tool descriptor (JSON-RPC)
TOOL_DESCRIPTOR = {
    "name": "read_file",
    "description": "Read the contents of a file. Optionally specify a line range.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or relative path to the file"},
            "start_line": {"type": "integer", "description": "1-indexed first line (optional)"},
            "end_line": {"type": "integer", "description": "1-indexed last line (optional)"},
        },
        "required": ["path"],
    },
}

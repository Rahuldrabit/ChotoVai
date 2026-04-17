"""
MCP tool: write_file — write or overwrite a file's contents.
"""
from __future__ import annotations

from pathlib import Path


def write_file(path: str, content: str, create_dirs: bool = True) -> str:
    """
    Write content to a file.

    Args:
        path: Absolute or relative file path.
        content: Full text to write.
        create_dirs: If True, create parent directories as needed.

    Returns:
        Success message or error string.
    """
    p = Path(path)
    if create_dirs:
        p.parent.mkdir(parents=True, exist_ok=True)
    try:
        p.write_text(content, encoding="utf-8")
        return f"OK: wrote {len(content)} chars to {path}"
    except PermissionError:
        return f"ERROR: Permission denied: {path}"
    except OSError as e:
        return f"ERROR: {e}"


TOOL_DESCRIPTOR = {
    "name": "write_file",
    "description": "Write content to a file. Creates the file and any missing parent directories.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to write to"},
            "content": {"type": "string", "description": "Full content to write"},
            "create_dirs": {"type": "boolean", "description": "Create parent dirs if missing", "default": True},
        },
        "required": ["path", "content"],
    },
}

"""
MCP tool: shell — run a shell command in a subprocess with timeout and sandboxing.
"""
from __future__ import annotations

import subprocess


_BLOCKED_COMMANDS = frozenset(["rm", "rmdir", "del", "format", "shutdown", "reboot", "sudo"])


def shell(
    command: str,
    cwd: str | None = None,
    timeout_s: float = 30.0,
    env: dict[str, str] | None = None,
) -> dict[str, str | int]:
    """
    Execute a shell command.

    Returns a dict with stdout, stderr, and returncode.
    Blocks obviously destructive commands.
    """
    # Naive safety check — block obviously dangerous commands
    first_word = command.strip().split()[0].lower() if command.strip() else ""
    if first_word in _BLOCKED_COMMANDS:
        return {
            "stdout": "",
            "stderr": f"BLOCKED: command '{first_word}' is not permitted",
            "returncode": -1,
        }

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
        return {
            "stdout": result.stdout[-8000:] if len(result.stdout) > 8000 else result.stdout,
            "stderr": result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"TIMEOUT: command exceeded {timeout_s}s", "returncode": -1}
    except Exception as e:
        return {"stdout": "", "stderr": f"ERROR: {e}", "returncode": -1}


TOOL_DESCRIPTOR = {
    "name": "shell",
    "description": "Run a shell command and return stdout, stderr, and returncode.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Command to run"},
            "cwd": {"type": "string", "description": "Working directory (optional)"},
            "timeout_s": {"type": "number", "description": "Timeout in seconds", "default": 30.0},
        },
        "required": ["command"],
    },
}

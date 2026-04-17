"""
MCP tool: run_tests — run pytest (or other test runners) in a given directory.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def run_tests(
    path: str = ".",
    test_filter: str | None = None,
    timeout_s: float = 60.0,
    extra_args: list[str] | None = None,
) -> dict[str, str | int]:
    """
    Run the test suite via pytest.

    Args:
        path: Directory or file to run tests from.
        test_filter: Optional pytest -k filter expression.
        timeout_s: Timeout in seconds.
        extra_args: Additional pytest arguments.

    Returns:
        Dict with stdout, stderr, returncode.
    """
    cmd = ["python", "-m", "pytest", path, "-v", "--tb=short", "--no-header"]
    if test_filter:
        cmd += ["-k", test_filter]
    if extra_args:
        cmd.extend(extra_args)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=Path(path).parent if Path(path).is_file() else Path(path),
        )
        # Trim output to avoid flooding context
        stdout = result.stdout[-6000:] if len(result.stdout) > 6000 else result.stdout
        stderr = result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr
        return {"stdout": stdout, "stderr": stderr, "returncode": result.returncode}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"TIMEOUT: tests exceeded {timeout_s}s", "returncode": -1}
    except FileNotFoundError:
        return {"stdout": "", "stderr": "ERROR: pytest not found. Install with `pip install pytest`", "returncode": -1}


TOOL_DESCRIPTOR = {
    "name": "run_tests",
    "description": "Run the test suite via pytest. Returns stdout, stderr, and returncode.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to run tests from", "default": "."},
            "test_filter": {"type": "string", "description": "pytest -k filter expression"},
            "timeout_s": {"type": "number", "default": 60.0},
            "extra_args": {"type": "array", "items": {"type": "string"}},
        },
        "required": [],
    },
}

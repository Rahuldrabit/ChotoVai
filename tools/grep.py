"""
MCP tool: grep — search for a pattern in files.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path


def grep(
    pattern: str,
    path: str = ".",
    file_glob: str = "*.py",
    ignore_case: bool = False,
    max_results: int = 50,
) -> str:
    """
    Search for a regex pattern in files at path.

    Returns matching lines with file:line prefixes.
    Falls back to a pure-Python implementation if ripgrep is unavailable.
    """
    args = ["rg", "--line-number", "--no-heading", "--max-count", str(max_results)]
    if ignore_case:
        args.append("--ignore-case")
    args += ["--glob", file_glob, pattern, path]

    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            return result.stdout or "(no matches)"
        if result.returncode == 1:
            return "(no matches)"
        # rg not available — fall through to Python impl
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Pure-Python fallback
    flags = re.IGNORECASE if ignore_case else 0
    compiled = re.compile(pattern, flags)
    matches: list[str] = []
    search_root = Path(path)

    def _search(p: Path) -> None:
        if len(matches) >= max_results:
            return
        if p.is_dir():
            for child in sorted(p.iterdir()):
                _search(child)
        elif p.is_file() and p.match(file_glob):
            try:
                for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if compiled.search(line):
                        matches.append(f"{p}:{i}:{line}")
                        if len(matches) >= max_results:
                            break
            except OSError:
                pass

    _search(search_root)
    return "\n".join(matches) or "(no matches)"


TOOL_DESCRIPTOR = {
    "name": "grep",
    "description": "Search for a regex pattern in files. Uses ripgrep if available.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern to search for"},
            "path": {"type": "string", "description": "Directory or file to search in", "default": "."},
            "file_glob": {"type": "string", "description": "Glob filter for filenames", "default": "*.py"},
            "ignore_case": {"type": "boolean", "default": False},
            "max_results": {"type": "integer", "default": 50},
        },
        "required": ["pattern"],
    },
}

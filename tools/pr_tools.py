"""
MCP tools: pr_* / issue_* — GitHub + GitLab support via provider detection.

Notes:
- Uses `gh` for GitHub and `glab` for GitLab if available.
- These tools are intentionally bounded and schema-stable.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import Any, Literal


Provider = Literal["github", "gitlab", "unknown"]


@dataclass(frozen=True)
class _CmdResult:
    returncode: int
    stdout: str
    stderr: str


def _run(cmd: list[str], cwd: str | None, timeout_s: float) -> _CmdResult:
    try:
        r = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return _CmdResult(
            returncode=int(r.returncode),
            stdout=(r.stdout or "")[-12000:],
            stderr=(r.stderr or "")[-4000:],
        )
    except FileNotFoundError:
        return _CmdResult(returncode=-1, stdout="", stderr=f"ERROR: missing executable: {cmd[0]}")
    except subprocess.TimeoutExpired:
        return _CmdResult(returncode=-1, stdout="", stderr=f"ERROR: timeout after {timeout_s}s")


def _detect_provider(cwd: str | None) -> Provider:
    r = _run(["git", "remote", "-v"], cwd=cwd, timeout_s=5.0)
    if r.returncode != 0:
        return "unknown"
    txt = (r.stdout or "") + "\n" + (r.stderr or "")
    if "github.com" in txt:
        return "github"
    if "gitlab" in txt:
        return "gitlab"
    return "unknown"


def pr_fetch(number: int, cwd: str = ".", timeout_s: float = 15.0) -> dict[str, Any]:
    prov = _detect_provider(cwd)
    n = int(number)
    if n <= 0:
        return {"error": "invalid PR number"}

    if prov == "github":
        r = _run(["gh", "pr", "view", str(n), "--json", "title,body,author,state,url,baseRefName,headRefName"], cwd, timeout_s)
        return {"provider": "github", "returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}

    if prov == "gitlab":
        # `glab mr view` expects IID
        r = _run(["glab", "mr", "view", str(n), "--json"], cwd, timeout_s)
        return {"provider": "gitlab", "returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}

    return {"provider": "unknown", "error": "could not detect git provider (no remote?)"}


def issue_fetch(number: int, cwd: str = ".", timeout_s: float = 15.0) -> dict[str, Any]:
    prov = _detect_provider(cwd)
    n = int(number)
    if n <= 0:
        return {"error": "invalid issue number"}

    if prov == "github":
        r = _run(["gh", "issue", "view", str(n), "--json", "title,body,author,state,url,labels"], cwd, timeout_s)
        return {"provider": "github", "returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}

    if prov == "gitlab":
        # `glab issue view` expects IID
        r = _run(["glab", "issue", "view", str(n), "--json"], cwd, timeout_s)
        return {"provider": "gitlab", "returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}

    return {"provider": "unknown", "error": "could not detect git provider (no remote?)"}


def pr_create(
    title: str,
    body: str,
    cwd: str = ".",
    base: str | None = None,
    draft: bool = False,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """
    Create a PR/MR using the detected provider.

    Assumes the current branch is pushed (these CLIs typically require it).
    """
    prov = _detect_provider(cwd)
    t = (title or "").strip()
    b = (body or "").strip()
    if not t:
        return {"error": "title is required"}

    if prov == "github":
        cmd = ["gh", "pr", "create", "--title", t, "--body", b]
        if base:
            cmd += ["--base", base]
        if draft:
            cmd.append("--draft")
        r = _run(cmd, cwd, timeout_s)
        return {"provider": "github", "returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}

    if prov == "gitlab":
        cmd = ["glab", "mr", "create", "--title", t, "--description", b]
        if base:
            cmd += ["--target-branch", base]
        if draft:
            cmd.append("--draft")
        r = _run(cmd, cwd, timeout_s)
        return {"provider": "gitlab", "returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}

    return {"provider": "unknown", "error": "could not detect git provider (no remote?)"}


TOOL_DESCRIPTOR = {
    "name": "pr_tools",
    "description": "PR/issue tools (GitHub+GitLab) via provider detection.",
    "inputSchema": {"type": "object", "properties": {}, "required": []},
}


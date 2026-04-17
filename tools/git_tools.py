"""
MCP tools: git_* — bounded git operations.

Design:
- Prefer read-only introspection tools (`git_status`, `git_diff`, `git_log`)
- Side-effect tools (`git_commit`, `git_branch_create`, `git_push`) are safe-by-default
  and intended to be approval-gated by the MCP layer.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class _GitResult:
    returncode: int
    stdout: str
    stderr: str


def _run_git(args: list[str], cwd: str | None, timeout_s: float) -> _GitResult:
    try:
        r = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return _GitResult(
            returncode=int(r.returncode),
            stdout=(r.stdout or "")[-12000:],
            stderr=(r.stderr or "")[-4000:],
        )
    except FileNotFoundError:
        return _GitResult(returncode=-1, stdout="", stderr="ERROR: git not found on PATH")
    except subprocess.TimeoutExpired:
        return _GitResult(returncode=-1, stdout="", stderr=f"ERROR: timeout after {timeout_s}s")


def _is_git_repo(cwd: str | None) -> bool:
    r = _run_git(["rev-parse", "--is-inside-work-tree"], cwd=cwd, timeout_s=5.0)
    return r.returncode == 0 and "true" in (r.stdout or "").lower()


def git_status(cwd: str = ".", timeout_s: float = 10.0) -> dict[str, Any]:
    if not _is_git_repo(cwd):
        return {"returncode": 2, "stdout": "", "stderr": "ERROR: not a git repository"}
    r = _run_git(["status", "--porcelain=v1", "--branch"], cwd=cwd, timeout_s=timeout_s)
    return {"returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}


def git_diff(cwd: str = ".", staged: bool = False, pathspec: str | None = None, timeout_s: float = 10.0) -> dict[str, Any]:
    if not _is_git_repo(cwd):
        return {"returncode": 2, "stdout": "", "stderr": "ERROR: not a git repository"}
    args = ["diff"]
    if staged:
        args.append("--staged")
    if pathspec:
        args += ["--", pathspec]
    r = _run_git(args, cwd=cwd, timeout_s=timeout_s)
    return {"returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}


def git_log(cwd: str = ".", n: int = 10, timeout_s: float = 10.0) -> dict[str, Any]:
    if not _is_git_repo(cwd):
        return {"returncode": 2, "stdout": "", "stderr": "ERROR: not a git repository"}
    n = max(1, min(int(n), 50))
    r = _run_git(["log", f"-n{n}", "--pretty=format:%h %s"], cwd=cwd, timeout_s=timeout_s)
    return {"returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}


def git_branch_create(
    name: str,
    cwd: str = ".",
    checkout: bool = True,
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    if not _is_git_repo(cwd):
        return {"returncode": 2, "stdout": "", "stderr": "ERROR: not a git repository"}
    branch = (name or "").strip()
    if not branch or any(ch.isspace() for ch in branch):
        return {"returncode": 2, "stdout": "", "stderr": "ERROR: invalid branch name"}
    r = _run_git(["branch", branch], cwd=cwd, timeout_s=timeout_s)
    if r.returncode != 0:
        return {"returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
    if checkout:
        r2 = _run_git(["checkout", branch], cwd=cwd, timeout_s=timeout_s)
        return {"returncode": r2.returncode, "stdout": r2.stdout, "stderr": r2.stderr}
    return {"returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}


def git_commit(
    message: str,
    cwd: str = ".",
    add_all: bool = False,
    timeout_s: float = 20.0,
) -> dict[str, Any]:
    if not _is_git_repo(cwd):
        return {"returncode": 2, "stdout": "", "stderr": "ERROR: not a git repository"}
    msg = (message or "").strip()
    if not msg:
        return {"returncode": 2, "stdout": "", "stderr": "ERROR: empty commit message"}

    if add_all:
        add_r = _run_git(["add", "-A"], cwd=cwd, timeout_s=timeout_s)
        if add_r.returncode != 0:
            return {"returncode": add_r.returncode, "stdout": add_r.stdout, "stderr": add_r.stderr}

    r = _run_git(["commit", "-m", msg], cwd=cwd, timeout_s=timeout_s)
    return {"returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}


def git_push(
    cwd: str = ".",
    remote: str = "origin",
    ref: str = "HEAD",
    set_upstream: bool = True,
    force: bool = False,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    if not _is_git_repo(cwd):
        return {"returncode": 2, "stdout": "", "stderr": "ERROR: not a git repository"}
    if force:
        return {"returncode": 2, "stdout": "", "stderr": "ERROR: force push is blocked"}

    args = ["push"]
    if set_upstream:
        args.append("-u")
    args += [remote, ref]
    r = _run_git(args, cwd=cwd, timeout_s=timeout_s)
    return {"returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}


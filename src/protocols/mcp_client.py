"""
MCP client — dispatches tool calls to the registered tool implementations.
The agent calls tools through this, not directly.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable

import structlog

from src.core.schemas import ToolCall, ToolResult

logger = structlog.get_logger(__name__)

# Tool implementations imported here — keeps agents clean
from tools.read_file import read_file
from tools.write_file import write_file
from tools.patch_file import patch_file
from tools.grep import grep
from tools.shell import shell
from tools.run_tests import run_tests
from tools.code_graph import code_graph
from tools.web_search import web_search
from tools.web_fetch import web_fetch
from tools.git_tools import (
    git_status,
    git_diff,
    git_log,
    git_branch_create,
    git_commit,
    git_push,
)
from tools.pr_tools import pr_fetch, issue_fetch, pr_create
from tools.verify_repo import verify_repo
from tools.scratchpad_append import scratchpad_append
from tools.contracts_update import contracts_update
from tools.read_scratchpad import read_scratchpad

# Registry: tool name → callable
_TOOL_REGISTRY: dict[str, Callable[..., Any]] = {
    "read_file": read_file,
    "write_file": write_file,
    "patch_file": patch_file,
    "grep": grep,
    "shell": shell,
    "run_tests": run_tests,
    "code_graph": code_graph,
    "web_search": web_search,
    "web_fetch": web_fetch,
    "git_status": git_status,
    "git_diff": git_diff,
    "git_log": git_log,
    "git_branch_create": git_branch_create,
    "git_commit": git_commit,
    "git_push": git_push,
    "pr_fetch": pr_fetch,
    "issue_fetch": issue_fetch,
    "pr_create": pr_create,
    "verify_repo": verify_repo,
    "scratchpad_append": scratchpad_append,
    "contracts_update": contracts_update,
    "read_scratchpad": read_scratchpad,
}


def get_available_tools() -> list[str]:
    return list(_TOOL_REGISTRY.keys())


def _needs_approval(tool_name: str) -> bool:
    # Side-effect tools (write, commands, VCS/PR actions).
    return tool_name in {
        "write_file",
        "patch_file",
        "shell",
        "git_branch_create",
        "git_commit",
        "git_push",
        "pr_create",
    }


def _approval_mode() -> str:
    # "interactive" | "scoped_auto" | "auto" | "deny"
    return os.getenv("SLM_AGENT_APPROVAL_MODE", "scoped_auto").strip().lower()


def _is_high_blast_radius(tool_name: str, args: dict[str, Any]) -> bool:
    if tool_name in {"git_push", "pr_create"}:
        return True
    if tool_name == "git_commit":
        return bool(args.get("add_all"))
    if tool_name == "shell":
        cmd = str(args.get("command", "")).lower()
        # Treat install/build/deploy-ish commands as high risk.
        risky_tokens = ["pip install", "uv pip", "npm install", "pnpm", "yarn", "docker", "kubectl", "terraform"]
        return any(tok in cmd for tok in risky_tokens)
    if tool_name == "write_file":
        path = str(args.get("path", "")).replace("\\", "/").lower()
        high_paths = [
            ".env",
            "pyproject.toml",
            "requirements.txt",
            "package.json",
            "package-lock.json",
            "pnpm-lock.yaml",
            "yarn.lock",
            "dockerfile",
            "docker-compose",
        ]
        return any(path.endswith(p) for p in high_paths)
    return False


def _approve(tool_name: str, args: dict[str, Any]) -> tuple[bool, str]:
    mode = _approval_mode()
    if not _needs_approval(tool_name):
        return True, ""

    if mode == "auto":
        return True, ""
    if mode == "deny":
        return False, "DENIED: approval mode is 'deny'"
    if mode == "scoped_auto":
        # Auto-approve medium/low side effects; always prompt for high blast-radius.
        if not _is_high_blast_radius(tool_name, args):
            return True, ""

    # interactive
    try:
        preview = json.dumps(args, indent=2)[:2000]
    except Exception:
        preview = str(args)[:2000]
    ans = input(
        f"\nAPPROVE tool call?\n"
        f"- tool: {tool_name}\n"
        f"- args: {preview}\n"
        f"Approve? [y/N]: "
    ).strip().lower()
    if ans in {"y", "yes"}:
        return True, ""
    return False, "DENIED: user rejected tool call"


async def execute_tool(tool_call: ToolCall) -> ToolResult:
    """
    Execute a single MCP tool call and return a ToolResult.
    All exceptions are caught and surfaced as error results (never crash the agent loop).
    """
    import time

    name = tool_call.name
    args = tool_call.arguments

    if name not in _TOOL_REGISTRY:
        return ToolResult(
            tool_call_id=tool_call.id,
            name=name,
            content=f"ERROR: unknown tool '{name}'. Available: {', '.join(_TOOL_REGISTRY)}",
            is_error=True,
        )

    fn = _TOOL_REGISTRY[name]
    args_preview = {k: str(v)[:80] for k, v in args.items()}
    logger.info("mcp.dispatching", tool=name, args=args_preview)
    t0 = time.perf_counter()

    try:
        ok, reason = _approve(name, args)
        if not ok:
            duration_ms = (time.perf_counter() - t0) * 1000
            logger.warning("mcp.tool_denied", tool=name, reason=reason)
            return ToolResult(
                tool_call_id=tool_call.id,
                name=name,
                content=reason,
                is_error=True,
                duration_ms=duration_ms,
            )
        result = fn(**args)
        duration_ms = (time.perf_counter() - t0) * 1000

        # Normalize result to string
        if isinstance(result, dict):
            content = json.dumps(result, indent=2)
        elif isinstance(result, str):
            content = result
        else:
            content = str(result)

        logger.info(
            "mcp.tool_returned",
            tool=name,
            duration_ms=round(duration_ms, 1),
            ok=True,
            preview=content[:120],
        )
        return ToolResult(
            tool_call_id=tool_call.id,
            name=name,
            content=content,
            is_error=False,
            duration_ms=duration_ms,
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - t0) * 1000
        logger.warning("mcp.tool_error", tool=name, error=str(e), duration_ms=round(duration_ms, 1))
        return ToolResult(
            tool_call_id=tool_call.id,
            name=name,
            content=f"ERROR: {type(e).__name__}: {e}",
            is_error=True,
            duration_ms=duration_ms,
        )


def register_tool(name: str, fn: Callable) -> None:
    """Register an additional tool at runtime (e.g., graph_query for explorer)."""
    _TOOL_REGISTRY[name] = fn

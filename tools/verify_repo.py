"""
MCP tool: verify_repo — composite deterministic verification.

Wraps lint + typecheck + tests into a single structured response to keep tool menus small
and control flow deterministic.
"""

from __future__ import annotations

from typing import Any

from src.validators.deterministic import DeterministicValidator


def verify_repo(path: str = ".", run_tests: bool = True) -> dict[str, Any]:
    v = DeterministicValidator()
    results = v.validate(path, run_tests=run_tests)
    ok = all(r.outcome.value == "pass" for r in results if r.validator_name != "pyright") and all(
        r.outcome.value in {"pass", "uncertain"} for r in results
    )
    return {
        "ok": bool(ok),
        "results": [
            {
                "validator": r.validator_name,
                "outcome": r.outcome.value,
                "message": r.message,
                "hint": r.correction_hint,
                "duration_ms": r.duration_ms,
            }
            for r in results
        ],
    }


TOOL_DESCRIPTOR = {
    "name": "verify_repo",
    "description": "Run deterministic verification (ruff/pyright/pytest) and return structured results.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "default": "."},
            "run_tests": {"type": "boolean", "default": True},
        },
        "required": [],
    },
}


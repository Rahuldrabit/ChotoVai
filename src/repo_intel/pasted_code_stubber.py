"""User pasted-code stubber.

Goal: prevent raw, large code blobs from being injected into every worker call.
Instead, extract a *structural* sketch: defs/classes and key declarations.

This is intentionally deterministic (no model calls).
"""

from __future__ import annotations

import re


def stub_pasted_code(snippet: str, *, max_lines: int = 80, max_chars: int = 4000) -> str:
    if not snippet:
        return ""

    # Prefer signature-ish lines.
    lines = snippet.splitlines()
    keep: list[str] = []

    patterns = [
        re.compile(r"^\s*(from\s+\S+\s+import\s+.+|import\s+\S+.*)$"),
        re.compile(r"^\s*class\s+[A-Za-z_][A-Za-z0-9_]*\b.*:$"),
        re.compile(r"^\s*(async\s+)?def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(.*$"),
        re.compile(r"^\s*function\s+[A-Za-z_$][A-Za-z0-9_$]*\s*\(.*$"),
        re.compile(r"^\s*class\s+[A-Za-z_$][A-Za-z0-9_$]*\b.*$"),
    ]

    for line in lines:
        if any(p.match(line) for p in patterns):
            keep.append(line.rstrip())

        if len(keep) >= max_lines:
            break
        if sum(len(x) + 1 for x in keep) >= max_chars:
            break

    if not keep:
        # Fallback: show only the first few non-empty lines.
        for line in lines:
            if line.strip():
                keep.append(line.rstrip())
            if len(keep) >= min(12, max_lines):
                break

    out = "\n".join(keep).strip()
    if len(out) > max_chars:
        out = out[:max_chars]
    return out

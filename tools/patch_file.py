"""
MCP tool: patch_file — apply a unified diff patch to an existing file.

Preferred over write_file when editing a subset of lines, because:
- The model only needs to produce the changed hunks, not the whole file.
- Large files are edited safely without context-window truncation.
- The original content outside the hunks is never touched.

Diff format (standard unified diff, as produced by `diff -u` or `git diff`):
    --- a/path/to/file.py
    +++ b/path/to/file.py
    @@ -10,6 +10,8 @@
     unchanged line
    -removed line
    +added line
     unchanged line

The --- / +++ header lines are optional; the tool works without them.
Context lines (space-prefixed) must match the file exactly (modulo trailing whitespace).
"""
from __future__ import annotations

import re
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path


# ── Hunk parsing ──────────────────────────────────────────────────────────────

@dataclass
class Hunk:
    old_start: int          # 1-based line number in original file
    old_count: int
    new_start: int
    new_count: int
    lines: list[str] = field(default_factory=list)   # raw diff lines (+/-/ )


_HUNK_HEADER_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@"
)


def _parse_hunks(diff_text: str) -> list[Hunk]:
    """Parse unified diff text into a list of Hunk objects."""
    hunks: list[Hunk] = []
    current: Hunk | None = None

    for raw_line in diff_text.splitlines():
        m = _HUNK_HEADER_RE.match(raw_line)
        if m:
            if current is not None:
                hunks.append(current)
            old_start = int(m.group(1))
            old_count = int(m.group(2)) if m.group(2) is not None else 1
            new_start = int(m.group(3))
            new_count = int(m.group(4)) if m.group(4) is not None else 1
            current = Hunk(old_start, old_count, new_start, new_count)
            continue

        # Skip file header lines (--- +++ index)
        if raw_line.startswith(("--- ", "+++ ", "diff ", "index ")):
            continue

        if current is not None:
            current.lines.append(raw_line)

    if current is not None:
        hunks.append(current)

    return hunks


# ── Hunk application ──────────────────────────────────────────────────────────

def _apply_hunk(file_lines: list[str], hunk: Hunk) -> list[str]:
    """
    Apply a single hunk to file_lines (0-indexed list of lines, no trailing newlines).

    Raises ValueError if context lines don't match.
    """
    result: list[str] = []
    # Position in file_lines (0-based); hunk.old_start is 1-based
    file_pos = hunk.old_start - 1

    for diff_line in hunk.lines:
        if not diff_line:
            # Empty line in diff — treat as context (blank line)
            prefix, content = " ", ""
        else:
            prefix, content = diff_line[0], diff_line[1:]

        if prefix == " ":
            # Context line — must match file
            if file_pos >= len(file_lines):
                raise ValueError(
                    f"Context line past end of file at line {file_pos + 1}: {content!r}"
                )
            actual = file_lines[file_pos].rstrip()
            expected = content.rstrip()
            if actual != expected:
                raise ValueError(
                    f"Context mismatch at line {file_pos + 1}:\n"
                    f"  expected: {expected!r}\n"
                    f"  actual:   {actual!r}"
                )
            result.append(file_lines[file_pos])
            file_pos += 1

        elif prefix == "-":
            # Removal — consume file line, validate it matches
            if file_pos >= len(file_lines):
                raise ValueError(
                    f"Removal line past end of file at line {file_pos + 1}: {content!r}"
                )
            actual = file_lines[file_pos].rstrip()
            expected = content.rstrip()
            if actual != expected:
                raise ValueError(
                    f"Removal mismatch at line {file_pos + 1}:\n"
                    f"  expected: {expected!r}\n"
                    f"  actual:   {actual!r}"
                )
            file_pos += 1   # consume without appending to result

        elif prefix == "+":
            # Addition — insert new line
            result.append(content + "\n")

        # Ignore any other prefix (\ No newline at end of file, etc.)

    return result


def _apply_hunks(original: str, hunks: list[Hunk]) -> str:
    """
    Apply all hunks to the original file content and return the patched content.

    Hunks are applied in order; each hunk receives the incrementally modified line list.
    """
    # Preserve original line endings; work with \n-terminated lines
    lines = [l if l.endswith("\n") else l + "\n" for l in original.splitlines()]

    # Track offset caused by previous insertions/deletions
    offset = 0

    for hunk in hunks:
        # Adjust start position by accumulated offset
        adjusted_start = hunk.old_start + offset - 1   # 0-based index into current lines

        # Lines before this hunk
        before = lines[:adjusted_start]
        # Lines consumed by this hunk (old_count lines)
        after = lines[adjusted_start + hunk.old_count:]

        # Build replacement lines from hunk — pass adjusted 1-based start
        adjusted_hunk = dataclasses.replace(hunk, old_start=adjusted_start + 1)
        replacement = _apply_hunk(lines, adjusted_hunk)

        lines = before + replacement + after
        offset += (hunk.new_count - hunk.old_count)

    return "".join(lines)


# ── Public tool function ──────────────────────────────────────────────────────

def patch_file(path: str, diff: str, dry_run: bool = False) -> str:
    """
    Apply a unified diff to a file.

    Args:
        path:    Path to the file to patch.
        diff:    Unified diff text (output of `diff -u` or `git diff` style).
        dry_run: If True, validate the patch and return the result without writing.

    Returns:
        A status string: "OK: patched <path> (+N/-M lines)" on success,
        or an ERROR string describing what went wrong.
    """
    p = Path(path)
    if not p.exists():
        return f"ERROR: file not found: {path}"
    if not p.is_file():
        return f"ERROR: not a file: {path}"

    try:
        original = p.read_text(encoding="utf-8")
    except OSError as e:
        return f"ERROR: could not read {path}: {e}"

    hunks = _parse_hunks(diff)
    if not hunks:
        return "ERROR: no hunks found in diff — is this a valid unified diff?"

    try:
        patched = _apply_hunks(original, hunks)
    except ValueError as e:
        return f"ERROR: patch does not apply cleanly to {path}: {e}"

    added = sum(
        sum(1 for l in h.lines if l.startswith("+")) for h in hunks
    )
    removed = sum(
        sum(1 for l in h.lines if l.startswith("-")) for h in hunks
    )

    if dry_run:
        return (
            f"DRY RUN OK: patch applies cleanly to {path} "
            f"(+{added}/-{removed} lines, {len(hunks)} hunk(s))\n\n"
            f"--- patched result (first 60 lines) ---\n"
            + "\n".join(patched.splitlines()[:60])
        )

    try:
        p.write_text(patched, encoding="utf-8")
    except (PermissionError, OSError) as e:
        return f"ERROR: could not write {path}: {e}"

    return f"OK: patched {path} (+{added}/-{removed} lines, {len(hunks)} hunk(s))"


# ── Tool descriptor (MCP schema) ─────────────────────────────────────────────

TOOL_DESCRIPTOR = {
    "name": "patch_file",
    "description": (
        "Apply a unified diff (patch) to an existing file. "
        "Preferred over write_file for partial edits — the model only needs to produce "
        "the changed hunks, not rewrite the whole file. "
        "Use dry_run=true to validate before writing."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to patch",
            },
            "diff": {
                "type": "string",
                "description": (
                    "Unified diff text. Standard format:\n"
                    "@@ -old_start,old_count +new_start,new_count @@\n"
                    " context line\n"
                    "-removed line\n"
                    "+added line\n"
                    "The --- / +++ file headers are optional."
                ),
            },
            "dry_run": {
                "type": "boolean",
                "description": "If true, validate patch without writing to disk",
                "default": False,
            },
        },
        "required": ["path", "diff"],
    },
}

"""
read_scratchpad — retrieve entries from the session scratchpad.

Agents call this to recall past reasoning without re-reading full history.
Three query modes:
  - "recent"  : last max_chars of the full scratchpad (same as context injection)
  - "by_role" : entries written by a specific agent role (e.g. "coder", "critic")
  - "by_node" : entries tagged with a specific plan node ID
"""
from __future__ import annotations

from src.memory.scratchpad import get_active_scratchpad


def read_scratchpad(
    query_type: str = "recent",
    role: str = "",
    node_id: str = "",
    max_chars: int = 4000,
) -> str:
    """
    Read entries from the session reasoning scratchpad.

    Args:
        query_type: "recent" | "by_role" | "by_node"
        role:       Role to filter by when query_type="by_role" (e.g. "coder", "critic")
        node_id:    Plan node ID to filter by when query_type="by_node"
        max_chars:  Maximum characters to return (default 4000)

    Returns:
        Matching scratchpad entries as a string, or an informational message if empty.
    """
    sp = get_active_scratchpad()
    if sp is None:
        return "No active scratchpad for this session."

    if query_type == "by_node":
        if not node_id:
            return "ERROR: query_type='by_node' requires a non-empty node_id."
        result = sp.read_node(node_id)
        return result if result else f"No scratchpad entries found for node '{node_id}'."

    if query_type == "by_role":
        if not role:
            return "ERROR: query_type='by_role' requires a non-empty role."
        result = sp.read_by_role(role, max_chars=max_chars)
        return result if result else f"No scratchpad entries found for role '{role}'."

    # Default: "recent"
    result = sp.read_tail(max_chars=max_chars)
    return result if result else "Scratchpad is empty for this session."


TOOL_DESCRIPTOR = {
    "name": "read_scratchpad",
    "description": (
        "Read entries from the session reasoning scratchpad. "
        "Use 'recent' to get the latest reasoning tail (same view as context injection), "
        "'by_role' to filter entries written by a specific agent role (e.g. 'coder', 'critic'), "
        "or 'by_node' to retrieve all entries tied to a specific plan node ID. "
        "Call this before starting work to avoid re-exploring what other agents already found."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query_type": {
                "type": "string",
                "enum": ["recent", "by_role", "by_node"],
                "default": "recent",
                "description": "How to filter the scratchpad entries.",
            },
            "role": {
                "type": "string",
                "description": "Agent role to filter by (e.g. 'coder', 'critic', 'explorer'). Required when query_type='by_role'.",
            },
            "node_id": {
                "type": "string",
                "description": "Plan node ID to filter by (e.g. 'N3a'). Required when query_type='by_node'.",
            },
            "max_chars": {
                "type": "integer",
                "default": 4000,
                "description": "Maximum characters of output to return.",
            },
        },
        "required": [],
    },
}

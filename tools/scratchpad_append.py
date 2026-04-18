"""
scratchpad_append — MCP tool that lets agents write to the session scratchpad.

Agents call this during their TAOR loop to record:
- Decisions made ("I chose to use a factory pattern because...")
- Observations ("read_file revealed that UserRepo expects a session, not a connection")
- Summaries ("Completed __init__: injects db and secret, raises if either is None")

The orchestrator FSM writes node-level summaries automatically; this tool is for
finer-grained reasoning that the agent itself wants to preserve across context resets.
"""
from __future__ import annotations

from src.memory.scratchpad import get_active_scratchpad


def scratchpad_append(entry: str, role: str = "", node_id: str = "") -> str:
    """
    Append an entry to the current session's scratchpad.

    Args:
        entry:   The text to record (reasoning, decision, observation, summary).
        role:    Optional — the agent role writing this entry (e.g. "coder", "critic").
        node_id: Optional — the plan node this entry relates to (e.g. "N3a").

    Returns:
        "OK" on success, or an error string.
    """
    sp = get_active_scratchpad()
    if sp is None:
        return "ERROR: no active scratchpad session"
    if not entry or not entry.strip():
        return "ERROR: entry is empty"
    sp.append(entry.strip(), role=role, node_id=node_id)
    return f"OK: entry appended to scratchpad ({len(entry)} chars)"


TOOL_DESCRIPTOR = {
    "name": "scratchpad_append",
    "description": (
        "Append a reasoning entry, decision, or summary to the session scratchpad. "
        "Use this to record important observations, design decisions, or completed work "
        "so that future agents in this session can read them without needing the full "
        "conversation history."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "entry": {
                "type": "string",
                "description": "The text to record. Can be multiple lines.",
            },
            "role": {
                "type": "string",
                "description": "Your agent role (e.g. 'coder', 'critic', 'tester'). Optional.",
            },
            "node_id": {
                "type": "string",
                "description": "The plan node ID this entry relates to (e.g. 'N3a'). Optional.",
            },
        },
        "required": ["entry"],
    },
}

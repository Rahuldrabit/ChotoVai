"""
contracts_update — MCP tool that lets CoderAgents register code entity contracts.

After writing a class or function, the agent calls this to record the interface
(name, file, params, return type, dependencies) in the session's symbol table.

This allows subsequent agents to understand what has been built without reading
source files — critical for atomic codegen where each agent writes one piece.
"""
from __future__ import annotations

from src.memory.contracts import EntityContract, MethodSignature, get_active_contracts


def contracts_update(
    name: str,
    kind: str,
    file: str,
    methods: list[dict] | None = None,
    params: list[str] | None = None,
    returns: str = "None",
    depends_on: list[str] | None = None,
) -> str:
    """
    Register or update the contract for a code entity.

    Args:
        name:       The class or function name (e.g. "AuthService", "create_user").
        kind:       Entity kind: "class", "function", or "module".
        file:       Relative path from project root (e.g. "src/auth/service.py").
        methods:    For classes — list of method dicts:
                    [{"name": "...", "params": ["arg: Type"], "returns": "ReturnType"}]
        params:     For functions — list of parameter strings (e.g. ["email: str"]).
        returns:    Return type annotation string (e.g. "User", "None", "list[str]").
        depends_on: Names of other classes/functions this entity imports or calls.

    Returns:
        "OK" on success, or an error string.
    """
    cs = get_active_contracts()
    if cs is None:
        return "ERROR: no active contract store"

    if not name or not name.strip():
        return "ERROR: name is required"
    if kind not in ("class", "function", "module"):
        return "ERROR: kind must be 'class', 'function', or 'module'"
    if not file or not file.strip():
        return "ERROR: file is required"

    parsed_methods: list[MethodSignature] = []
    if methods:
        for m in methods:
            try:
                parsed_methods.append(MethodSignature.model_validate(m))
            except Exception as e:
                return f"ERROR: invalid method spec {m!r}: {e}"

    contract = EntityContract(
        name=name.strip(),
        kind=kind,
        file=file.strip(),
        methods=parsed_methods,
        params=params or [],
        returns=returns or "None",
        depends_on=depends_on or [],
    )

    cs.update(contract)
    return f"OK: contract for '{name}' ({kind}) saved to symbol table"


TOOL_DESCRIPTOR = {
    "name": "contracts_update",
    "description": (
        "Register or update the interface contract for a class or function you just wrote. "
        "Call this after writing any class or function so subsequent agents know the "
        "signature without reading source files. This builds the session's symbol table."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Class or function name."},
            "kind": {
                "type": "string",
                "enum": ["class", "function", "module"],
                "description": "Entity type.",
            },
            "file": {
                "type": "string",
                "description": "Relative file path from project root.",
            },
            "methods": {
                "type": "array",
                "description": "For classes: list of method signatures.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "params": {"type": "array", "items": {"type": "string"}},
                        "returns": {"type": "string"},
                    },
                    "required": ["name"],
                },
            },
            "params": {
                "type": "array",
                "items": {"type": "string"},
                "description": "For functions: parameter list (e.g. ['email: str', 'password: str']).",
            },
            "returns": {"type": "string", "description": "Return type (e.g. 'User', 'None')."},
            "depends_on": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Names of other entities this one imports or calls.",
            },
        },
        "required": ["name", "kind", "file"],
    },
}

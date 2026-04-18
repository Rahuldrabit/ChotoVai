"""
ContractStore — lightweight JSON symbol table tracking code entities produced by agents.

Every time a CoderAgent writes a class, function, or module, it calls the
`contracts_update` tool to register the entity's interface here. Subsequent
agents read this file (via `read_file` tool or context injection) to understand
the global architecture without reading actual source code.

Solves the "pieces don't fit together" problem: each agent knows the contract
(name, params, return type, dependencies) before writing its piece.

File layout  (one per session):
  ./data/sessions/<session_id>/contracts.json

Schema:
  {
    "ClassName": {
      "name": "ClassName",
      "kind": "class",
      "file": "src/auth/service.py",
      "methods": [
        {"name": "__init__", "params": ["db: Database", "secret: str"], "returns": "None"}
      ],
      "depends_on": ["Database", "User"]
    },
    "create_user": {
      "name": "create_user",
      "kind": "function",
      "file": "src/auth/service.py",
      "params": ["email: str", "password: str"],
      "returns": "User",
      "depends_on": ["UserRepository"]
    }
  }
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

# ── Module-level singleton ────────────────────────────────────────────────────
_active: "ContractStore | None" = None
_lock = threading.Lock()


def set_active_contracts(session_dir: Path) -> "ContractStore":
    """Create (or reuse) the contract store for the current session."""
    global _active
    cs = ContractStore(session_dir / "contracts.json")
    with _lock:
        _active = cs
    return cs


def get_active_contracts() -> "ContractStore | None":
    return _active


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class MethodSignature(BaseModel):
    name: str
    params: list[str] = Field(default_factory=list)
    returns: str = "None"


class EntityContract(BaseModel):
    """Interface contract for a single code entity (class, function, or module)."""
    name: str
    kind: str                                         # "class" | "function" | "module"
    file: str                                         # relative path from project root
    # Class fields
    methods: list[MethodSignature] = Field(default_factory=list)
    # Function fields
    params: list[str] = Field(default_factory=list)
    returns: str = "None"
    # Shared
    depends_on: list[str] = Field(default_factory=list)


# ── Store ─────────────────────────────────────────────────────────────────────

class ContractStore:
    """
    Thread-safe JSON store for EntityContracts.

    Agents write contracts via the `contracts_update` tool after generating code.
    The context assembler injects the compact form into every CODER/CRITIC/TESTER call.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}

        # Load existing contracts if file exists
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("contracts.load_failed", error=str(e))
                self._data = {}

    # ── Write ─────────────────────────────────────────────────────────────────

    def update(self, contract: EntityContract) -> None:
        """Add or overwrite the contract for an entity (thread-safe)."""
        with self._lock:
            self._data[contract.name] = contract.model_dump()
            self._flush()
        logger.debug("contracts.updated", name=contract.name, kind=contract.kind, file=contract.file)

    def _flush(self) -> None:
        """Write current state to disk. Must be called under self._lock."""
        try:
            self._path.write_text(
                json.dumps(self._data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning("contracts.flush_failed", error=str(e))

    # ── Read ──────────────────────────────────────────────────────────────────

    def get(self, name: str) -> EntityContract | None:
        """Look up a single entity by name."""
        with self._lock:
            raw = self._data.get(name)
        if raw is None:
            return None
        try:
            return EntityContract.model_validate(raw)
        except Exception:
            return None

    def get_all_compact(self) -> str:
        """
        Return a compact human-readable summary of all tracked entities.
        Injected into every CODER/CRITIC/TESTER agent call (~1 token per symbol).

        Example output:
          ## Code Contracts (symbol table)
          - class AuthService [src/auth/service.py]
              __init__(db: Database, secret: str) -> None
              create_user(email: str, password: str) -> User
          - function verify_token [src/auth/service.py]
              (token: str) -> Payload | None
        """
        with self._lock:
            data = dict(self._data)

        if not data:
            return ""

        lines = ["## Code Contracts (symbol table)"]
        for entry in data.values():
            try:
                c = EntityContract.model_validate(entry)
            except Exception:
                continue

            lines.append(f"- {c.kind} **{c.name}** [{c.file}]")
            if c.kind == "class":
                for m in c.methods:
                    params = ", ".join(m.params)
                    lines.append(f"    {m.name}({params}) -> {m.returns}")
            else:
                params = ", ".join(c.params)
                lines.append(f"    ({params}) -> {c.returns}")
            if c.depends_on:
                lines.append(f"    depends on: {', '.join(c.depends_on)}")

        return "\n".join(lines)

    @property
    def path(self) -> Path:
        return self._path

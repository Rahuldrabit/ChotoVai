"""
Model registry — maps agent role → VLLMClient instance.
Import `get_client(AgentRole.CODER)` anywhere to get the right client.
"""
from __future__ import annotations

from src.core.config import get_config
from src.core.schemas import AgentRole
from src.serving.vllm_client import VLLMClient

_registry: dict[AgentRole, VLLMClient] | None = None


def _build_registry() -> dict[AgentRole, VLLMClient]:
    cfg = get_config()
    m = cfg.models
    return {
        AgentRole.ORCHESTRATOR: VLLMClient(m.orchestrator, role="orchestrator"),
        AgentRole.CODER:        VLLMClient(m.coder, role="coder"),
        AgentRole.EXPLORER:     VLLMClient(m.explorer, role="explorer"),
        AgentRole.TESTER:       VLLMClient(m.tester, role="tester"),
        AgentRole.REFACTORER:   VLLMClient(m.coder, role="refactorer"),   # shares coder base
        AgentRole.CRITIC:       VLLMClient(m.critic, role="critic"),
        AgentRole.SUMMARIZER:   VLLMClient(m.summarizer, role="summarizer"),
        AgentRole.DOC_READER:   VLLMClient(m.coder, role="doc_reader"),
    }


def get_client(role: AgentRole) -> VLLMClient:
    """Return the inference client for the given agent role (singleton per role)."""
    global _registry
    if _registry is None:
        _registry = _build_registry()
    return _registry[role]


def get_escalation_client() -> VLLMClient:
    """Return the frontier LLM client for escalation."""
    cfg = get_config()
    return VLLMClient(cfg.models.escalation, role="escalation")


def reset_registry() -> None:
    """Reset registry — useful in tests."""
    global _registry
    _registry = None

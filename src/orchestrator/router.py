"""
Router — Maps agent roles to agent instances.
Replaces the static mapping in the FSM to support generic A2A messaging in the future.
"""
from __future__ import annotations

from src.core.schemas import AgentRole
from src.agents.base import BaseAgent
from src.agents.coder import CoderAgent
from src.agents.summarizer import SummarizerAgent
from src.agents.explorer import ExplorerAgent
from src.agents.tester import TesterAgent
from src.agents.refactorer import RefactorerAgent
from src.agents.critic import CriticAgent


def get_specialist(role: AgentRole) -> BaseAgent:
    """Return an instantiated agent for the given role."""
    
    mapping = {
        AgentRole.CODER: CoderAgent,
        AgentRole.SUMMARIZER: SummarizerAgent,
        AgentRole.EXPLORER: ExplorerAgent,
        AgentRole.TESTER: TesterAgent,
        AgentRole.REFACTORER: RefactorerAgent,
        AgentRole.CRITIC: CriticAgent,
        AgentRole.ORCHESTRATOR: CoderAgent,  # Orchestrator uses planner, not TAOR loop directly
    }

    agent_cls = mapping.get(role, CoderAgent)
    return agent_cls()

"""
CognitiveRouter — RLoT-inspired dynamic strategy selection.

Selects a CognitiveStrategy for each PlanNode at execution time, acting as
the 'Navigator' described in the RL-of-Thoughts (RLoT) methodology.

Current implementation: rule-based heuristics.
Interface designed for drop-in replacement with a RL-trained navigator
(<3K param policy net) without touching the FSM.

Strategy selection order:
  1. Explicit strategy set on PlanNode (planner hint) — respected as-is
  2. High retry count → ESCALATE (model is stuck)
  3. Code-producing roles (CODER, REFACTORER) and debate_enabled → DEBATE
  4. Complex multi-file / ambiguous tasks → REFINE
  5. Read-only roles (EXPLORER, TESTER, SUMMARIZER) → DIRECT
"""
from __future__ import annotations

import re

import structlog

from src.core.config import get_config
from src.core.schemas import AgentRole, AgentState, CognitiveStrategy, PlanNode

# Multi-concept conjunctions that signal a node covers too many concerns
_MULTI_CONCEPT_PATTERN = re.compile(
    r"\b(and|including|as well as|with support for|also|plus|alongside)\b",
    re.IGNORECASE,
)

logger = structlog.get_logger(__name__)

# Keyword patterns that signal high complexity → trigger DEBATE
_COMPLEX_PATTERNS = re.compile(
    r"\b(auth|security|concurren|race.condition|deadlock|transaction|"
    r"encrypt|permission|privilege|injection|overflow|async|thread|lock|"
    r"cache.invalidat|migration|schema.change)\b",
    re.IGNORECASE,
)

# Roles that produce code artifacts — candidates for adversarial debate
_CODE_PRODUCING_ROLES = {AgentRole.CODER, AgentRole.REFACTORER}

# Roles that are read-only and never need debate
_READ_ONLY_ROLES = {AgentRole.EXPLORER, AgentRole.SUMMARIZER, AgentRole.DOC_READER}


class CognitiveRouter:
    """
    Selects the optimal cognitive strategy for a given PlanNode at runtime.

    Usage:
        router = CognitiveRouter()
        strategy = router.select(node, state)
    """

    def __init__(self) -> None:
        self._cfg = get_config().agent

    def select(self, node: PlanNode, state: AgentState) -> CognitiveStrategy:
        """
        Return the CognitiveStrategy to use when executing this node.

        Args:
            node:  The PlanNode about to be executed.
            state: Current AgentState (for retry history and budget).

        Returns:
            CognitiveStrategy enum value.
        """
        # 1. Honour explicit planner hint (only if not DIRECT — DIRECT = unset)
        if node.cognitive_strategy != CognitiveStrategy.DIRECT:
            logger.debug(
                "cognitive_router.planner_hint",
                node_id=node.id,
                strategy=node.cognitive_strategy.value,
            )
            return node.cognitive_strategy

        # 2. Node is too broad AND is a top-level node (not already a decomposed child)
        if node.parent_node_id is None and self._is_too_broad(node):
            logger.info(
                "cognitive_router.decompose_broad_node",
                node_id=node.id,
                title=node.title,
            )
            return CognitiveStrategy.DECOMPOSE

        # 3. High retry count → model is stuck → escalate immediately
        if node.retry_count >= self._cfg.retry_limit - 1:
            logger.info(
                "cognitive_router.escalate_high_retries",
                node_id=node.id,
                retry_count=node.retry_count,
            )
            return CognitiveStrategy.ESCALATE

        # 4. Read-only roles → always DIRECT (no debate overhead)
        role = node.assigned_role or AgentRole.CODER
        if role in _READ_ONLY_ROLES:
            return CognitiveStrategy.DIRECT

        # 5. Code-producing roles + debate enabled → DEBATE or REFINE
        if role in _CODE_PRODUCING_ROLES and self._cfg.debate_enabled:
            if self._is_complex(node):
                logger.info(
                    "cognitive_router.debate_complex",
                    node_id=node.id,
                    title=node.title,
                )
                return CognitiveStrategy.DEBATE
            # Simple code task but still benefits from one critic pass
            return CognitiveStrategy.DEBATE

        # 6. TESTER role → VERIFY (run deterministic checks after generation)
        if role == AgentRole.TESTER:
            return CognitiveStrategy.VERIFY

        # 7. CRITIC role standalone → DIRECT (it IS the review)
        if role == AgentRole.CRITIC:
            return CognitiveStrategy.DIRECT

        # Fallback
        return CognitiveStrategy.DIRECT

    def _is_complex(self, node: PlanNode) -> bool:
        """Heuristic: is this task complex enough to warrant a full debate?"""
        # Keyword match in title + description
        combined = f"{node.title} {node.description}"
        if _COMPLEX_PATTERNS.search(combined):
            return True
        # Many success criteria → complex
        if len(node.success_criteria) >= 3:
            return True
        # Long description → likely complex
        if len(node.description.split()) > 60:
            return True
        return False

    def _is_too_broad(self, node: PlanNode) -> bool:
        """
        Heuristic: is this node too broad for a single agent pass?
        Triggers DECOMPOSE so the FSM splits it into atomic subtasks at runtime.
        """
        # Planner explicitly requested decomposition
        if node.cognitive_strategy == CognitiveStrategy.DECOMPOSE:
            return True
        # Read-only and test roles are inherently narrow — never decompose
        role = node.assigned_role or AgentRole.CODER
        if role in _READ_ONLY_ROLES or role == AgentRole.TESTER:
            return False
        # Very long description signals too many concerns packed into one node
        word_count = len(node.description.split())
        if word_count > 80:
            return True
        # Too many success criteria → the node is trying to do too much
        if len(node.success_criteria) > 3:
            return True
        # Title contains multi-concept conjunctions (e.g. "implement X and Y and Z")
        if len(_MULTI_CONCEPT_PATTERN.findall(node.title)) >= 2:
            return True
        return False

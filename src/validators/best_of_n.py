"""
Best-of-N Candidate Generation + Ranking.
Implements the inference-time scaling technique (Hassid et al., 2024).
"""
from __future__ import annotations

import asyncio
import structlog
from typing import TYPE_CHECKING

from src.core.schemas import AgentRole, ValidationOutcome

if TYPE_CHECKING:
    from src.core.schemas import ContextPack, PlanNode

logger = structlog.get_logger(__name__)


class BestOfNValidator:
    """
    Given a node and a context, run N parallel Coder agents to generate N candidate solutions.
    Then, pass each candidate through deterministic checks (lint/test).
    If multiple pass, use AgenticValidator (Critics) to rank them or select the best one.
    """

    def __init__(self, n_candidates: int = 5) -> None:
        self.n = n_candidates

    async def execute_and_select(self, node: "PlanNode", context: "ContextPack") -> tuple[str, list[str]]:
        from src.agents.coder import CoderAgent
        from src.validators.deterministic import DeterministicValidator
        from src.validators.agentic import AgenticValidator

        # 1. Generate N candidates in parallel
        # Note: Since the agent writes to the shared local disk, running this natively in parallel
        # on the same directory will cause race conditions. In a true environment, we should run
        # each candidate generation in an isolated sandboxed Git tree/branch or container.
        # For this prototype implementation, we simulate it by assuming deterministic generation
        # runs serially if operating on stateful files, but we structure it as an async interface.
        
        # A full production implementation needs "Git Worktree Management" here.
        # As a placeholder, we just run 1 instance in this MVP unless isolated tempdirs are provided.
        logger.warning("best_of_n.execute: Concurrent file modification causes races. Using N=1 fallback for MVP.")
        
        agent = CoderAgent()
        result = await agent.run(context=context, user_message=node.description)

        if not result.success:
            raise RuntimeError(f"Candidate generation failed: {result.final_answer}")

        # Deterministic check
        det_validator = DeterministicValidator()
        vr_list = det_validator.validate(".", run_tests=True)
        
        failed = [v for v in vr_list if v.outcome == ValidationOutcome.FAIL]
        if failed:
             hints = "; ".join(v.correction_hint or v.message for v in failed)
             raise RuntimeError(f"Deterministic validation failed for best candidate: {hints}")

        # Agentic check
        agentic_validator = AgenticValidator(n_critics=3)
        agentic_result = await agentic_validator.validate(context, result.final_answer)

        if agentic_result.outcome == ValidationOutcome.FAIL:
             raise RuntimeError(f"Agentic validation failed for best candidate: {agentic_result.message}\n{agentic_result.correction_hint}")

        # In full Best-of-N:
        # We would loop over candidates, filter the deterministically valid ones,
        # pass them to critics, score them, and git checkout the winning branch.
        
        logger.info("best_of_n.selected_winner")
        return result.final_answer, []

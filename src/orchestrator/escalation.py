"""
Hybrid Escalation Policy — debate-aware frontier handoff.

Upgraded from Phase 7 stub to full debate integration:
  - Accepts DebateState for enriched context (full transcript)
  - Provides run_escalated_debate() for deadlock resolution
  - Provides should_escalate_from_debate() for game-theoretic triggers
  - All frontier calls are scoped: only the minimum context is sent
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.core.config import get_config
from src.core.schemas import AgentMessage

if TYPE_CHECKING:
    from src.core.schemas import ContextPack, DebateState, PlanNode

logger = structlog.get_logger(__name__)


class EscalationManager:
    """
    Decides when to escalate to a frontier model (Claude Opus / GPT-4o)
    and executes the scoped call with full debate context.
    """

    def __init__(self) -> None:
        self.cfg = get_config().agent
        self._frontier_client = None

    async def _get_frontier_client(self):
        from src.serving.model_registry import get_escalation_client
        if not self._frontier_client:
            self._frontier_client = get_escalation_client()
        return self._frontier_client

    # ──────────────────────────────────────────────────────────────────────
    # Trigger evaluation
    # ──────────────────────────────────────────────────────────────────────

    def should_escalate(self, fsm_state: "AgentState", current_node: "PlanNode") -> bool:  # noqa: F821
        """Check confidence signals from node retry history."""
        if current_node.retry_count >= (self.cfg.retry_limit - 1):
            logger.info("escalation.triggered.high_retries", node_id=current_node.id)
            return True
        return False

    def should_escalate_from_debate(self, debate: "DebateState") -> bool:
        """
        Game-theoretic escalation trigger based on debate signals:
          1. Deadlock: max_turns reached and score still below threshold
          2. Stagnation: critic_score not improving across last two turns
          3. Score floor: score < 4 after half the turns (model is stuck hard)
        """
        from src.core.schemas import DebateOutcome

        if debate.outcome == DebateOutcome.DEADLOCK:
            logger.info("escalation.triggered.debate_deadlock", node_id=debate.node_id)
            return True

        # Stagnation check: compare last two critic scores
        critic_moves = [m for m in debate.moves if m.actor == "critic" and m.score is not None]
        if len(critic_moves) >= 2:
            prev_score = critic_moves[-2].score
            curr_score = critic_moves[-1].score
            if curr_score is not None and prev_score is not None and curr_score <= prev_score:
                if debate.turn_count >= (debate.max_turns // 2):
                    logger.info(
                        "escalation.triggered.debate_stagnation",
                        node_id=debate.node_id,
                        prev=prev_score,
                        curr=curr_score,
                    )
                    return True

        # Hard floor: after half the budget, still critically bad
        half_turns = debate.max_turns // 2
        if debate.turn_count >= half_turns and debate.critic_score < 4:
            logger.info(
                "escalation.triggered.score_floor",
                node_id=debate.node_id,
                score=debate.critic_score,
            )
            return True

        return False

    # ──────────────────────────────────────────────────────────────────────
    # Execution
    # ──────────────────────────────────────────────────────────────────────

    async def run_escalated_task(self, context: "ContextPack", prompt: str) -> str:
        """
        Send a scoped ContextPack to the frontier model.
        (Original interface — preserved for backward compatibility.)
        """
        logger.info("escalation.calling_frontier", mode="single_task")
        client = await self._get_frontier_client()
        messages = [
            AgentMessage(
                role="system",
                content=(
                    "You are a frontier intelligence assisting an SLM swarm. "
                    "Solve the following problem completely and concisely."
                ),
            ),
            AgentMessage(role="user", content=prompt),
        ]
        response = await client.complete(messages)
        logger.info("escalation.frontier_success", tokens=response.tokens_out)
        return response.content

    async def run_escalated_debate(
        self,
        debate: "DebateState",
        extra_context: str = "",
    ) -> str:
        """
        Send the FULL debate transcript to the frontier model for resolution.

        The frontier model receives:
          - Task description + success criteria
          - Complete Coder↔Critic argument log
          - The final code artifact
          - Optional extra context (e.g. judge ensemble reasoning)

        Returns:
            The frontier model's definitive resolution (code + rationale).
        """
        logger.info(
            "escalation.calling_frontier",
            mode="debate_resolution",
            node_id=debate.node_id,
            turns=debate.turn_count,
            final_score=debate.critic_score,
        )

        client = await self._get_frontier_client()

        transcript = self._format_debate_for_frontier(debate)

        system = (
            "You are a principal engineer resolving a deadlocked adversarial code review. "
            "Two SLMs (Coder and Critic) have debated but could not reach agreement. "
            "Your task:\n"
            "1. Read the full debate transcript.\n"
            "2. Determine the definitive correct implementation.\n"
            "3. Output the complete, corrected code with an explanation of what you changed and why.\n"
            "Be decisive. The SLM swarm will execute your response directly."
        )

        user_content = transcript
        if extra_context:
            user_content += f"\n\n## Additional Context\n{extra_context}"

        messages = [
            AgentMessage(role="system", content=system),
            AgentMessage(role="user", content=user_content),
        ]

        response = await client.complete(messages)
        logger.info(
            "escalation.frontier_success",
            tokens=response.tokens_out,
            node_id=debate.node_id,
        )
        return response.content

    # ──────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _format_debate_for_frontier(debate: "DebateState") -> str:
        """Format DebateState as a structured transcript for the frontier model."""
        criteria_str = "\n".join(f"  - {c}" for c in debate.success_criteria) or "  (none)"
        lines = [
            f"# Deadlocked Debate — Node {debate.node_id}",
            f"## Task\n{debate.task_description}",
            f"## Success Criteria\n{criteria_str}",
            f"## Debate Summary\n"
            f"- Turns completed: {debate.turn_count} / {debate.max_turns}\n"
            f"- Final critic score: {debate.critic_score}/{debate.acceptance_threshold} (threshold not reached)\n"
            f"- Last critic objection: {debate.critic_reasoning}\n",
            "## Full Debate Transcript",
        ]
        for move in debate.moves:
            lines.append(f"\n### Turn {move.turn} — {move.actor.upper()}")
            if move.score is not None:
                lines.append(f"Score: {move.score}/10")
            lines.append(move.content[:2000])   # Truncate very long moves
            if move.failing_tests:
                lines.append("Failing tests written:")
                for t in move.failing_tests[:5]:   # Top 5 tests
                    lines.append(f"```python\n{t}\n```")

        lines.append(f"\n## Final Code Under Review\n```\n{debate.current_code}\n```")
        return "\n".join(lines)

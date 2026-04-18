"""
Agentic validator — upgraded judge ensemble with confidence-based escalation.

Phase 4 upgrade over the original:
  - Score-based voting (0-10) instead of binary pass/fail
  - Inter-judge agreement metric (std_dev across scores)
  - Confidence signal: std_dev > threshold → judges disagree → escalate
  - Returns EnsembleVerdict (richer than ValidationResult)
  - Accepts optional DebateState for enriched context
"""
from __future__ import annotations

import asyncio
import json
import math
from typing import TYPE_CHECKING

import structlog

from src.core.config import get_config
from src.core.schemas import (
    EnsembleVerdict,
    ValidationOutcome,
    ValidationResult,
)

if TYPE_CHECKING:
    from src.core.schemas import ContextPack, DebateState

logger = structlog.get_logger(__name__)

_SCORED_CRITIC_SYSTEM = """\
You are a judge in a code-quality panel. Review the code and emit ONLY this JSON:
{
  "score": <int 0-10>,
  "reasoning": "<concise critique>",
  "correction_hint": "<specific fix instruction, or null if score >= 9>"
}

Scoring guide:
  0-4: Critical bugs or fundamentally wrong
  5-7: Works for happy path; notable gaps
  8:   Minor nits only
  9-10: Production-ready — no valid objections
"""


class AgenticValidator:
    """
    Ensemble of Critics. Runs N independent judges in parallel, aggregates
    scores, computes confidence, and fires escalation on disagreement.
    """

    def __init__(self, n_critics: int = 3) -> None:
        self.n = n_critics
        self._cfg = get_config().agent

    # ──────────────────────────────────────────────────────────────────────
    # Primary interface
    # ──────────────────────────────────────────────────────────────────────

    async def validate(
        self,
        context_pack: "ContextPack",
        changes_summary: str,
    ) -> ValidationResult:
        """
        Backward-compatible interface — returns a ValidationResult.
        Internally runs the full ensemble; escalation fires silently.
        """
        verdict = await self.validate_with_verdict(
            context_pack=context_pack,
            changes_summary=changes_summary,
        )
        return ValidationResult(
            validator_name=f"critic_ensemble_of_{self.n}",
            outcome=verdict.outcome,
            message=verdict.message,
            correction_hint=verdict.correction_hint,
        )

    async def validate_with_verdict(
        self,
        context_pack: "ContextPack",
        changes_summary: str,
        debate_state: "DebateState | None" = None,
    ) -> EnsembleVerdict:
        """
        Run the judge ensemble and return a full EnsembleVerdict.

        Args:
            context_pack:    Assembled context for the critic agents.
            changes_summary: The code / diff to review.
            debate_state:    Optional — passed to enrich the prompt with the
                             full debate transcript for better judge context.

        Returns:
            EnsembleVerdict with individual scores, agreement metrics,
            and escalation signal.
        """
        from src.agents.critic import CriticAgent

        critics = [CriticAgent() for _ in range(self.n)]

        # Build enriched prompt
        prompt = self._build_prompt(changes_summary, debate_state)

        # Fan-out: run all critics concurrently
        tasks = [
            self._score_single_critic(c, context_pack, prompt)
            for c in critics
        ]
        results: list[tuple[int, str, str | None]] = await asyncio.gather(*tasks)

        scores = [r[0] for r in results]
        reasonings = [r[1] for r in results]
        hints = [r[2] for r in results if r[2]]

        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std_dev = math.sqrt(variance)

        # Normalise std_dev to [0, 1] — max possible std_dev is 5.0 (range 0-10)
        confidence = max(0.0, 1.0 - (std_dev / 5.0))

        threshold = self._cfg.debate_acceptance_threshold
        outcome = (
            ValidationOutcome.PASS if mean >= threshold * 0.75
            else ValidationOutcome.FAIL
        )

        verdict = EnsembleVerdict(
            outcome=outcome,
            individual_scores=scores,
            mean_score=round(mean, 2),
            std_dev=round(std_dev, 2),
            confidence=round(confidence, 3),
            message=self._summarise(scores, reasonings, outcome),
            correction_hint="\n====\n".join(hints) if hints and outcome == ValidationOutcome.FAIL else None,
        )

        logger.info(
            "validator.ensemble",
            scores=scores,
            mean=verdict.mean_score,
            std_dev=verdict.std_dev,
            confidence=verdict.confidence,
            outcome=outcome.value,
        )

        # ── Write verdict to session scratchpad ───────────────────────
        try:
            from src.memory.scratchpad import get_active_scratchpad
            sp = get_active_scratchpad()
            if sp is not None:
                node_id = getattr(getattr(context_pack, "plan_node", None), "id", "unknown")
                issues_preview = "; ".join(hints[:2]) if hints else "none"
                sp.append(
                    f"Validation {outcome.value.upper()} — mean score {verdict.mean_score}/10 "
                    f"(scores: {scores}) — issues: {issues_preview}",
                    role="critic",
                    node_id=node_id,
                )
        except Exception:
            pass  # Never let scratchpad writes break validation flow

        # ── Confidence-drop escalation ─────────────────────────────────
        if confidence < self._cfg.debate_ensemble_confidence_threshold:
            logger.info(
                "validator.ensemble.escalating",
                confidence=confidence,
                threshold=self._cfg.debate_ensemble_confidence_threshold,
            )
            escalated = await self._escalate(verdict, changes_summary, debate_state)
            # Keep the original typed verdict unless escalation returns a real replacement.
            # This makes testing/mocking safer and preserves agreement metrics.
            if isinstance(escalated, EnsembleVerdict):
                verdict = escalated
            else:
                # Best-effort merge of common fields.
                try:
                    verdict.escalated = bool(getattr(escalated, "escalated", True))
                    verdict.escalation_response = getattr(escalated, "escalation_response", None)
                    verdict.outcome = getattr(escalated, "outcome", verdict.outcome)
                except Exception:
                    verdict.escalated = True

        return verdict

    # ──────────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────────

    async def _score_single_critic(
        self,
        critic: "CriticAgent",  # noqa: F821
        context_pack: "ContextPack",
        prompt: str,
    ) -> tuple[int, str, str | None]:
        """Run one critic, parse its score JSON. Returns (score, reasoning, hint)."""
        try:
            result = await critic.run(
                context=context_pack,
                user_message=prompt,
                extra_system=_SCORED_CRITIC_SYSTEM,
            )
            raw = result.final_answer.strip()
            if raw.startswith("```"):
                raw = raw.split("```", 2)[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            if raw.endswith("```"):
                raw = raw[:-3]
            data = json.loads(raw.strip())
            score = max(0, min(10, int(data.get("score", 5))))
            reasoning = data.get("reasoning", raw)
            hint = data.get("correction_hint")
            return score, reasoning, hint
        except Exception as exc:
            logger.warning("validator.ensemble.parse_error", error=str(exc))
            return 5, "Parse error — defaulting to uncertain score", None

    async def _escalate(
        self,
        verdict: EnsembleVerdict,
        changes_summary: str,
        debate_state: "DebateState | None",
    ) -> EnsembleVerdict:
        """Fire escalation to the frontier model to break judge disagreement."""
        try:
            from src.orchestrator.escalation import EscalationManager
            manager = EscalationManager()

            if debate_state is not None:
                response = await manager.run_escalated_debate(debate_state, changes_summary)
            else:
                from src.core.schemas import AgentMessage
                from src.serving.model_registry import get_escalation_client
                client = get_escalation_client()
                msgs = [
                    AgentMessage(role="system", content="You are a frontier code reviewer resolving a split judge decision."),
                    AgentMessage(role="user", content=f"Judge scores: {verdict.individual_scores}\n\nCode:\n{changes_summary}"),
                ]
                resp = await client.complete(msgs)
                response = resp.content

            verdict.escalated = True
            verdict.escalation_response = response
            # If the frontier agrees it's passing, upgrade the outcome
            if any(kw in response.lower() for kw in ("pass", "accept", "approved", "looks good")):
                verdict.outcome = ValidationOutcome.PASS
            logger.info("validator.ensemble.escalated", escalated=True)
        except Exception as exc:
            logger.error("validator.ensemble.escalation_failed", error=str(exc))

        return verdict

    @staticmethod
    def _build_prompt(
        changes_summary: str,
        debate_state: "DebateState | None",
    ) -> str:
        prompt = f"## Code to Review\n{changes_summary}"
        if debate_state:
            prompt += (
                f"\n\n## Debate Context\n"
                f"Task: {debate_state.task_description}\n"
                f"Turns debated: {debate_state.turn_count}\n"
                f"Last critic score: {debate_state.critic_score}/10\n"
                f"Last critic reasoning: {debate_state.critic_reasoning}"
            )
        return prompt

    @staticmethod
    def _summarise(
        scores: list[int],
        reasonings: list[str],
        outcome: ValidationOutcome,
    ) -> str:
        score_str = ", ".join(str(s) for s in scores)
        if outcome == ValidationOutcome.PASS:
            return f"Ensemble passed (scores: {score_str})"
        return f"Ensemble failed (scores: {score_str}). Key issues: {reasonings[0][:200]}"

"""
DebateGraph — the adversarial Coder↔Critic game loop.

Implements a graph-based state machine where:
  - Player A (Coder): reward = get critic_score >= acceptance_threshold
  - Player B (Critic): reward = find flaws; never fixes code, only attacks it
  - Referee (route_after_critic): reads score + turn_count, decides routing

Graph topology:
    coder_node → critic_node → [route_after_critic]
                                  ├─ score >= threshold  → ACCEPT (CODER_WINS)
                                  ├─ turns < max_turns   → CONTINUE (back to coder)
                                  └─ turns >= max_turns  → DEADLOCK (judge_node)

    judge_node → [deadlock resolution per config]
                   ├─ "judge_ensemble" → AgenticValidator ensemble vote
                   ├─ "escalate"       → EscalationManager frontier call
                   └─ "best_of_n"      → BestOfNValidator
"""
from __future__ import annotations

import time
import re
from datetime import datetime

import structlog

from src.core.config import get_config
from src.core.schemas import (
    AgentMessage,
    AgentRole,
    ContextPack,
    DebateMove,
    DebateOutcome,
    DebateState,
    PlanNode,
    ValidationOutcome,
)
from src.validators.deterministic import DeterministicValidator

logger = structlog.get_logger(__name__)

# Expose these names for test patching (tests patch `src.orchestrator.debate_graph.CoderAgent`, etc.).
# The runtime still instantiates inside methods, but having module-level bindings keeps the surface stable.
try:
    from src.agents.coder import CoderAgent as CoderAgent  # noqa: F401
    from src.agents.critic import CriticAgent as CriticAgent  # noqa: F401
    from src.validators.agentic import AgenticValidator as AgenticValidator  # noqa: F401
except Exception:
    # In minimal environments some deps may be missing; tests can still patch these attributes.
    CoderAgent = None  # type: ignore
    CriticAgent = None  # type: ignore
    AgenticValidator = None  # type: ignore

# ──────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ──────────────────────────────────────────────────────────────────────────────

_CODER_INITIAL_SYSTEM = """\
You are Player A in an adversarial code-generation game. Your reward is to produce code so
correct and complete that an adversarial Critic cannot find any valid flaw.

Rules:
1. Read relevant files BEFORE writing any code.
2. Write minimal, correct, production-ready code for EXACTLY the task described.
3. Follow existing patterns in the codebase (naming, style, structure).
4. Emit the complete implementation — no TODOs, no placeholders.
5. After each Critic attack, you MUST explicitly rebut each failing test or objection,
   then patch your code to address them. Do not ignore any critique.

Output:
- Use ```json {"tool": "<name>", "arguments": {...}} ``` to call tools.
- When complete, emit:
  FINAL_ANSWER: <brief summary of changes>
  ```python
  <full corrected code or unified diff>
  ```
"""

_CODER_REBUTTAL_TEMPLATE = """\
The Critic reviewed your code and scored it {score}/10 (threshold: {threshold}/10).

## Resolved Issues (Do NOT regress on these)
{compressed_history}

## Critic's Objections (you MUST address ALL of them)
{reasoning}

## Failing Tests Written by Critic
{failing_tests}

## Actual Pytest Output (if tests were runnable)
{test_run_output}

## Your Task
1. Rebut each objection in one sentence.
2. Fix every issue the Critic identified.
3. Re-emit your complete corrected implementation.

This is turn {turn} of {max_turns}. If you do not reach score >= {threshold},
the game goes to a Judge panel.
"""

_CRITIC_SYSTEM = """\
You are Player B in an adversarial code-review game. You are evaluating candidate implementations.
Your job is to select the best candidate and fiercely critique it.

Rules:
1. Review all Coder implementations against the task and success criteria.
2. Select the best candidate by number (best_candidate: 1, 2, ...).
3. Write concrete FAILING unit tests that expose bugs in the chosen candidate.
4. Identify edge cases, security issues, logic errors, and style violations.
5. Assign a score from 0 to 10 for the chosen candidate:
   - 0-4: Major bugs, non-functional, or fundamentally wrong approach
   - 5-7: Works for happy path but has significant gaps or edge cases
   - 8:   Minor nits only; nearly production-ready
   - 9-10: You cannot find a valid flaw — you CONCEDE

Your FINAL_ANSWER MUST be exactly this JSON (no other fields, no code blocks):
{
  "best_candidate": <1 or 2>,
  "score": <int 0-10>,
  "reasoning": "<your critique — be specific, cite line numbers>",
  "failing_tests": ["<pytest test case 1>", "<pytest test case 2>"]
}

Be ruthlessly adversarial. If you return score < 9, you MUST provide failing_tests or cite specific line numbers.
"""

_JUDGE_SYSTEM = """\
You are a neutral Judge resolving a deadlocked adversarial code debate.
You have received the full debate transcript. Your job is to:
1. Determine if the Coder's final code is acceptable (score >= 7) despite the deadlock.
2. Identify the single most important remaining issue.
3. Provide a definitive verdict.

Output exactly:
{
  "verdict": "accept" | "reject",
  "final_score": <int 0-10>,
  "rationale": "<one paragraph>",
  "correction_hint": "<if reject: exact instruction for the Coder>"
}
"""


def _repair_json(raw: str) -> dict | None:
    """Try to salvage JSON from messy SLM output (stray text before/after braces, single quotes)."""
    import json as _json
    start = raw.find('{')
    end = raw.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = raw[start:end + 1]
    try:
        return _json.loads(candidate)
    except _json.JSONDecodeError:
        pass
    try:
        return _json.loads(candidate.replace("'", '"'))
    except _json.JSONDecodeError:
        pass
    return None


# ──────────────────────────────────────────────────────────────────────────────
# DebateGraph
# ──────────────────────────────────────────────────────────────────────────────

class DebateGraph:
    """
    Runs the adversarial Coder↔Critic game for a single PlanNode.

    Usage:
        graph = DebateGraph()
        debate_state, result_summary = await graph.run(node, context_pack)
    """

    def __init__(self) -> None:
        self._cfg = get_config().agent

    async def run(
        self,
        node: PlanNode,
        context: ContextPack,
    ) -> tuple[DebateState, str]:
        """
        Execute the full debate game for one plan node.

        Returns:
            (DebateState, result_summary) — the completed game state and a
            human-readable summary of what the Coder produced.
        """
        cfg = self._cfg
        debate = DebateState(
            node_id=node.id,
            task_description=node.description,
            success_criteria=node.success_criteria,
            max_turns=cfg.debate_max_turns,
            acceptance_threshold=cfg.debate_acceptance_threshold,
            max_token_budget=cfg.debate_max_token_budget,
        )

        logger.info(
            "debate.start",
            node_id=node.id,
            max_turns=debate.max_turns,
            threshold=debate.acceptance_threshold,
        )

        # ── Round loop ────────────────────────────────────────────────────
        while debate.outcome == DebateOutcome.IN_PROGRESS:
            debate.turn_count += 1

            # ── CODER NODE (Parallel) ─────────────────────────────────────
            candidates = await self._coder_node(debate, context)
            logger.info("debate.coder_moves", node_id=node.id, turn=debate.turn_count, candidates=len(candidates))

            # ── GROUND TRUTH SANDBOX ──────────────────────────────────────
            sandbox_failures = []
            validator = DeterministicValidator()
            for cand in candidates:
                results = validator.validate_content(".", cand, run_tests=False)
                failed = [v for v in results if v.outcome == ValidationOutcome.FAIL]
                if failed:
                    sandbox_failures.append("; ".join(v.message or v.details or "" for v in failed))
                else:
                    sandbox_failures.append("")

            viable_candidates = [c for i, c in enumerate(candidates) if not sandbox_failures[i]]
            
            if not viable_candidates:
                err_msg = f"All {len(candidates)} parallel branches failed sandbox syntax checks:\n{sandbox_failures}"
                logger.warning("debate.sandbox_failed_all", node_id=node.id)
                debate.critic_score = 0
                debate.critic_reasoning = err_msg
                debate.critic_failing_tests = []
                debate.current_code = candidates[0]
                debate.moves.append(DebateMove(turn=debate.turn_count, actor="sandbox", content=err_msg, score=0, tokens_used=0))
                continue

            # ── CRITIC NODE ───────────────────────────────────────────────
            debate = await self._critic_node(debate, viable_candidates, context)
            logger.info("debate.critic_move", node_id=node.id, turn=debate.turn_count, score=debate.critic_score)

            # ── EXECUTE FAILING TESTS ─────────────────────────────────────
            # Run the critic's failing tests via subprocess to get real pytest output
            if debate.critic_failing_tests:
                from pathlib import Path
                import tempfile
                tmp_file = None
                try:
                    # Write tests to a temporary file in tests directory
                    tests_dir = Path(".") / "tests"
                    tests_dir.mkdir(parents=True, exist_ok=True)
                    tmp_file = tests_dir / f"_debate_tmp_{debate.node_id.replace('/', '_')}_t{debate.turn_count}.py"
                    tmp_file.write_text("\n\n".join(debate.critic_failing_tests))

                    # Run pytest on the temp file
                    validator = DeterministicValidator()
                    pytest_result = validator._run_pytest(str(tmp_file))
                    debate.last_test_run_output = pytest_result.message or ""
                    logger.info("debate.test_execution", node_id=node.id, tests_count=len(debate.critic_failing_tests))
                except Exception as e:
                    logger.warning("debate.test_execution_error", error=str(e))
                    debate.last_test_run_output = f"ERROR running tests: {e}"
                finally:
                    if tmp_file and tmp_file.exists():
                        tmp_file.unlink()

            # ── REFEREE EDGE ──────────────────────────────────────────────
            routing = self._route_after_critic(debate)

            if routing == "accept":
                debate.outcome = DebateOutcome.CODER_WINS
                debate.completed_at = datetime.utcnow()
                logger.info("debate.resolved", node_id=node.id, outcome="coder_wins", turns=debate.turn_count, final_score=debate.critic_score)
                break

            if routing == "deadlock":
                logger.warning("debate.deadlock", node_id=node.id, turns=debate.turn_count, final_score=debate.critic_score)
                debate = await self._judge_node(debate, context)
                debate.completed_at = datetime.utcnow()
                break

        return debate, debate.current_code

    # ──────────────────────────────────────────────────────────────────────
    # Graph Nodes
    # ──────────────────────────────────────────────────────────────────────

    async def _coder_node(self, debate: DebateState, context: ContextPack) -> list[str]:
        """Player A (Parallel) — generate or patch the implementation in N branches."""
        import asyncio
        global CoderAgent
        if CoderAgent is None:  # pragma: no cover
            from src.agents.coder import CoderAgent as _CoderAgent
            CoderAgent = _CoderAgent  # type: ignore

        t0 = time.perf_counter()

        extra = None
        if debate.turn_count > 1 and debate.critic_reasoning:
            failing_str = "\n".join(f"  - {t}" for t in debate.critic_failing_tests) or "  (none provided)"
            test_output_str = debate.last_test_run_output or "  (tests could not be executed)"
            extra = _CODER_REBUTTAL_TEMPLATE.format(
                score=debate.critic_score,
                threshold=debate.acceptance_threshold,
                compressed_history=debate.compressed_history or "(None yet)",
                reasoning=debate.critic_reasoning,
                failing_tests=failing_str,
                test_run_output=test_output_str,
                turn=debate.turn_count,
                max_turns=debate.max_turns,
            )

        agent1 = CoderAgent()  # type: ignore

        if debate.turn_count == 1:
            # Turn 1: spawn 2 GoT branches — diverse initial approaches give the critic real signal
            agent2 = CoderAgent()  # type: ignore
            sys1 = _CODER_INITIAL_SYSTEM + "\nFocus heavily on robustness and error handling."
            sys2 = _CODER_INITIAL_SYSTEM + "\nFocus heavily on performance and clean architecture."
            results = await asyncio.gather(
                agent1.run(context=context, user_message=debate.task_description, extra_system=sys1),
                agent2.run(context=context, user_message=debate.task_description, extra_system=sys2),
            )
        else:
            # Rebuttal turns: single focused coder — both branches would get identical critic feedback
            results = [await agent1.run(context=context, user_message=debate.task_description, extra_system=extra)]

        tokens = sum(r.tokens_used for r in results)
        debate.tokens_used += tokens
        for r in results:
            debate.tool_trace.extend(r.tool_trace)

        candidates = [r.final_answer for r in results if r.success] or [results[0].final_answer]

        # Keep the debate transcript stable: one coder move per turn.
        debate.moves.append(
            DebateMove(
                turn=debate.turn_count,
                actor="coder",
                content=candidates[0],
                tokens_used=tokens,
            )
        )

        return candidates

    async def _critic_node(self, debate: DebateState, candidates: list[str], context: ContextPack) -> DebateState:
        """Player B — Evaluate parallel candidates, merge, and critique."""
        import json
        global CriticAgent
        if CriticAgent is None:  # pragma: no cover
            from src.agents.critic import CriticAgent as _CriticAgent
            CriticAgent = _CriticAgent  # type: ignore
        agent = CriticAgent()  # type: ignore

        criteria_str = "\n".join(f"  - {c}" for c in debate.success_criteria) or "  (none specified)"

        if debate.turn_count == 1:
            focus = "Focus primarily on Security, Architecure & Core Logic."
        elif debate.turn_count == 2:
            focus = "Focus primarily on Edge Cases & Unhandled Exceptions."
        else:
            focus = "Focus primarily on Code Quality, Performance & Best Practices."

        cands_str = "\n\n".join(f"## Candidate {i+1}\n```python\n{c}\n```" for i, c in enumerate(candidates))

        prompt = f"""\
## Task
{debate.task_description}

## Success Criteria
{criteria_str}

## Current Focus
{focus}

{cands_str}

Select the best candidate, write failing tests against it, and assign a score.
"""

        result = await agent.run(
            context=context,
            user_message=prompt,
            extra_system=_CRITIC_SYSTEM,
            accept_plain_text_final=True,
        )
        tokens = result.tokens_used
        debate.tokens_used += tokens

        score = 0
        reasoning = result.final_answer
        failing_tests: list[str] = []
        chosen_code = candidates[0]

        try:
            raw = result.final_answer.strip()
            if raw.startswith("```"):
                raw = raw.split("```", 2)[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            if raw.endswith("```"):
                raw = raw[:-3]
            data = json.loads(raw.strip())

            best_idx = max(0, min(len(candidates) - 1, int(data.get("best_candidate", 1)) - 1))
            chosen_code = candidates[best_idx]
            score = max(0, min(10, int(data.get("score", 0))))
            reasoning = data.get("reasoning", result.final_answer)
            failing_tests = data.get("failing_tests", [])

            if score < debate.acceptance_threshold and not failing_tests and "line" not in reasoning.lower():
                logger.warning("debate.critic_sycophancy_clamp", node_id=debate.node_id)
                score = debate.acceptance_threshold
            debate.json_parse_failures = 0  # clean parse — reset streak
        except Exception as exc:
            logger.warning("debate.critic_parse_error", error=str(exc))
            raw_fallback = result.final_answer
            # Try JSON repair before falling back to regex
            repaired = _repair_json(raw_fallback)
            if repaired is not None:
                try:
                    best_idx = max(0, min(len(candidates) - 1, int(repaired.get("best_candidate", 1)) - 1))
                    chosen_code = candidates[best_idx]
                    score = max(0, min(10, int(repaired.get("score", 0))))
                    reasoning = repaired.get("reasoning", raw_fallback)
                    failing_tests = repaired.get("failing_tests", [])
                    logger.info("debate.critic_json_repaired", node_id=debate.node_id, score=score)
                    debate.json_parse_failures = 0  # repaired — reset streak
                except Exception:
                    repaired = None
            if repaired is None:
                debate.json_parse_failures += 1
                score_match = re.search(r'"score"\s*:\s*(\d+)', raw_fallback)
                if score_match:
                    score = max(0, min(10, int(score_match.group(1))))
                    reasoning = (
                        f"Critic JSON parse error, recovered score={score}. "
                        f"Raw output: {raw_fallback}"
                    )
                else:
                    score = 3

        debate.current_code = chosen_code
        debate.critic_score = score
        debate.critic_reasoning = reasoning
        debate.critic_failing_tests = failing_tests

        debate.moves.append(DebateMove(
            turn=debate.turn_count,
            actor="critic",
            content=reasoning,
            score=score,
            failing_tests=failing_tests,
            tokens_used=tokens,
        ))

        return debate

    async def _judge_node(self, debate: DebateState, context: ContextPack) -> DebateState:
        """Deadlock resolution — judge ensemble or frontier escalation."""
        cfg = self._cfg
        strategy = cfg.debate_deadlock_strategy

        logger.info("debate.judge_node", strategy=strategy, node_id=debate.node_id)

        if strategy == "escalate":
            debate = await self._resolve_via_escalation(debate)
        elif strategy == "best_of_n":
            # best_of_n is a separate concerns — fall through to ensemble
            debate = await self._resolve_via_ensemble(debate, context)
        else:  # "judge_ensemble" (default)
            debate = await self._resolve_via_ensemble(debate, context)

        return debate

    async def _resolve_via_ensemble(
        self, debate: DebateState, context: ContextPack
    ) -> DebateState:
        """Run AgenticValidator ensemble as the judge panel."""
        global AgenticValidator
        if AgenticValidator is None:  # pragma: no cover
            from src.validators.agentic import AgenticValidator as _AgenticValidator
            AgenticValidator = _AgenticValidator  # type: ignore

        validator = AgenticValidator(n_critics=3)
        verdict = await validator.validate_with_verdict(
            context_pack=context,
            changes_summary=debate.current_code,
            debate_state=debate,
        )

        if verdict.outcome == ValidationOutcome.PASS:
            debate.outcome = DebateOutcome.CODER_WINS
        elif verdict.escalated:
            debate.outcome = DebateOutcome.ESCALATED
            debate.critic_reasoning = verdict.escalation_response or verdict.correction_hint or ""
        else:
            debate.outcome = DebateOutcome.DEADLOCK
            debate.critic_reasoning = verdict.correction_hint or verdict.message

        move = DebateMove(
            turn=debate.turn_count + 1,
            actor="judge",
            content=verdict.message,
            score=int(verdict.mean_score),
        )
        debate.moves.append(move)
        return debate

    async def _resolve_via_escalation(self, debate: DebateState) -> DebateState:
        """Send full debate transcript to the frontier model for resolution."""
        from src.orchestrator.escalation import EscalationManager

        manager = EscalationManager()
        transcript = self._format_transcript(debate)
        response = await manager.run_escalated_debate(debate, transcript)

        debate.outcome = DebateOutcome.ESCALATED
        debate.current_code = response
        move = DebateMove(
            turn=debate.turn_count + 1,
            actor="judge",
            content=response,
            score=10,  # Frontier verdict is authoritative
        )
        debate.moves.append(move)
        return debate

    # ──────────────────────────────────────────────────────────────────────
    # Referee — conditional routing edge
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _route_after_critic(debate: DebateState) -> str:
        """
        The rules of engagement — determines the next state transition.

        Returns:
            "accept"          → Critic concedes; Coder wins (Nash equilibrium)
            "continue_debate" → Route back to Coder with critique attached
            "deadlock"        → max_turns reached; send to Judge
        """
        if debate.critic_score >= debate.acceptance_threshold:
            return "accept"
        if debate.json_parse_failures >= 2:
            logger.warning(
                "debate.auto_accept_parse_failures",
                node_id=debate.node_id,
                failures=debate.json_parse_failures,
            )
            return "accept"
        if debate.turn_count >= debate.max_turns:
            return "deadlock"
        if debate.tokens_used >= debate.max_token_budget:
            logger.warning("debate.ttl_budget_exhausted", node_id=debate.node_id)
            return "deadlock"
        return "continue_debate"

    # ──────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _format_transcript(debate: DebateState) -> str:
        """Format the full debate history as a human-readable transcript."""
        lines = [
            f"# Debate Transcript — Node {debate.node_id}",
            f"Task: {debate.task_description}",
            "",
        ]
        for move in debate.moves:
            lines.append(f"## Turn {move.turn} — {move.actor.upper()}")
            if move.score is not None:
                lines.append(f"Score: {move.score}/10")
            lines.append(move.content)
            if move.failing_tests:
                lines.append("### Failing Tests")
                for t in move.failing_tests:
                    lines.append(f"```python\n{t}\n```")
            lines.append("")
        lines.append(f"**Final code under review:**\n```\n{debate.current_code}\n```")
        return "\n".join(lines)

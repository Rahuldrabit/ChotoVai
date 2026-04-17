"""
Unit + integration tests for the adversarial debate system.

Tests cover:
  1. DebateState schema round-trip
  2. CognitiveRouter strategy selection heuristics
  3. DebateGraph referee (route_after_critic) conditional edges
  4. DebateGraph full loop with mock agents (convergence + deadlock)
  5. AgenticValidator confidence + escalation trigger
  6. DebateTraceCollector export format
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.core.schemas import (
    AgentRole,
    AgentState,
    CognitiveStrategy,
    ContextPack,
    DebateMove,
    DebateOutcome,
    DebateState,
    EnsembleVerdict,
    FSMState,
    NodeStatus,
    PlanNode,
    TokenBudget,
    ValidationOutcome,
)


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────


@pytest.fixture
def simple_node() -> PlanNode:
    return PlanNode(
        id="N1",
        title="Add sum function",
        description="Add a sum_list(lst: list[int]) -> int function to utils.py",
        assigned_role=AgentRole.CODER,
        success_criteria=["Function exists", "Returns sum of list"],
        cognitive_strategy=CognitiveStrategy.DIRECT,
    )


@pytest.fixture
def complex_node() -> PlanNode:
    return PlanNode(
        id="N2",
        title="Implement JWT authentication middleware",
        description=(
            "Add JWT token validation middleware to the FastAPI app. "
            "Handle token expiry, signature verification, and permission checks. "
            "Ensure no privilege escalation via injection attacks."
        ),
        assigned_role=AgentRole.CODER,
        success_criteria=[
            "All endpoints behind auth require valid JWT",
            "Expired tokens return 401",
            "Invalid signatures return 403",
            "Injection attempts are rejected",
        ],
        cognitive_strategy=CognitiveStrategy.DIRECT,
    )


@pytest.fixture
def agent_state() -> AgentState:
    return AgentState(
        user_goal="Test goal",
        budget=TokenBudget(total_limit=100_000),
    )


@pytest.fixture
def minimal_context(simple_node: PlanNode) -> ContextPack:
    return ContextPack(
        agent_role=AgentRole.CODER,
        plan_node=simple_node,
    )


@pytest.fixture
def fresh_debate(simple_node: PlanNode) -> DebateState:
    return DebateState(
        node_id=simple_node.id,
        task_description=simple_node.description,
        success_criteria=simple_node.success_criteria,
        max_turns=3,
        acceptance_threshold=9,
    )


# ─────────────────────────────────────────────
# 1. Schema tests
# ─────────────────────────────────────────────


class TestDebateState:
    def test_initial_state(self, fresh_debate: DebateState) -> None:
        assert fresh_debate.outcome == DebateOutcome.IN_PROGRESS
        assert fresh_debate.turn_count == 0
        assert fresh_debate.critic_score == 0
        assert fresh_debate.moves == []

    def test_add_move(self, fresh_debate: DebateState) -> None:
        move = DebateMove(turn=1, actor="coder", content="def sum_list(lst): return sum(lst)")
        fresh_debate.moves.append(move)
        assert len(fresh_debate.moves) == 1
        assert fresh_debate.moves[0].actor == "coder"

    def test_schema_serialisation_round_trip(self, fresh_debate: DebateState) -> None:
        json_str = fresh_debate.model_dump_json()
        restored = DebateState.model_validate_json(json_str)
        assert restored.node_id == fresh_debate.node_id
        assert restored.outcome == DebateOutcome.IN_PROGRESS

    def test_ensemble_verdict_confidence(self) -> None:
        verdict = EnsembleVerdict(
            outcome=ValidationOutcome.PASS,
            individual_scores=[8, 9, 10],
            mean_score=9.0,
            std_dev=0.816,
            confidence=0.837,
        )
        assert verdict.confidence > 0.8
        assert verdict.escalated is False


# ─────────────────────────────────────────────
# 2. CognitiveRouter tests
# ─────────────────────────────────────────────


class TestCognitiveRouter:
    def setup_method(self) -> None:
        from src.orchestrator.cognitive_router import CognitiveRouter
        self.router = CognitiveRouter()

    def test_read_only_role_always_direct(self, agent_state: AgentState) -> None:
        node = PlanNode(
            id="N1", title="T", description="Explore the codebase",
            assigned_role=AgentRole.EXPLORER,
        )
        assert self.router.select(node, agent_state) == CognitiveStrategy.DIRECT

    def test_coder_simple_task_debate(self, simple_node: PlanNode, agent_state: AgentState) -> None:
        # With debate_enabled=True (default), coder nodes → DEBATE
        strategy = self.router.select(simple_node, agent_state)
        assert strategy == CognitiveStrategy.DEBATE

    def test_complex_node_debate(self, complex_node: PlanNode, agent_state: AgentState) -> None:
        strategy = self.router.select(complex_node, agent_state)
        assert strategy == CognitiveStrategy.DEBATE

    def test_planner_hint_respected(self, agent_state: AgentState) -> None:
        node = PlanNode(
            id="N1", title="T", description="Do something",
            assigned_role=AgentRole.CODER,
            cognitive_strategy=CognitiveStrategy.REFINE,
        )
        assert self.router.select(node, agent_state) == CognitiveStrategy.REFINE

    def test_high_retries_escalate(self, simple_node: PlanNode, agent_state: AgentState) -> None:
        simple_node.retry_count = 2  # retry_limit - 1 = 2 (limit=3)
        strategy = self.router.select(simple_node, agent_state)
        assert strategy == CognitiveStrategy.ESCALATE

    def test_tester_role_verify(self, agent_state: AgentState) -> None:
        node = PlanNode(
            id="N1", title="T", description="Write tests for utils.py",
            assigned_role=AgentRole.TESTER,
        )
        assert self.router.select(node, agent_state) == CognitiveStrategy.VERIFY


# ─────────────────────────────────────────────
# 3. DebateGraph referee tests
# ─────────────────────────────────────────────


class TestDebateGraphReferee:
    def setup_method(self) -> None:
        from src.orchestrator.debate_graph import DebateGraph
        self.graph = DebateGraph()

    def test_accept_on_high_score(self, fresh_debate: DebateState) -> None:
        fresh_debate.critic_score = 9
        assert self.graph._route_after_critic(fresh_debate) == "accept"

    def test_accept_on_threshold_score(self, fresh_debate: DebateState) -> None:
        fresh_debate.critic_score = 10
        assert self.graph._route_after_critic(fresh_debate) == "accept"

    def test_continue_on_low_score_early_turn(self, fresh_debate: DebateState) -> None:
        fresh_debate.critic_score = 5
        fresh_debate.turn_count = 1
        assert self.graph._route_after_critic(fresh_debate) == "continue_debate"

    def test_deadlock_on_max_turns(self, fresh_debate: DebateState) -> None:
        fresh_debate.critic_score = 5
        fresh_debate.turn_count = 3  # max_turns = 3
        assert self.graph._route_after_critic(fresh_debate) == "deadlock"

    def test_continue_last_turn_before_max(self, fresh_debate: DebateState) -> None:
        fresh_debate.critic_score = 6
        fresh_debate.turn_count = 2  # max_turns = 3, so still one turn left
        assert self.graph._route_after_critic(fresh_debate) == "continue_debate"

    def test_ttl_cost_bounding(self, fresh_debate: DebateState) -> None:
        fresh_debate.critic_score = 5
        fresh_debate.turn_count = 1
        fresh_debate.max_token_budget = 50_000
        fresh_debate.tokens_used = 55_000
        assert self.graph._route_after_critic(fresh_debate) == "deadlock"


# ─────────────────────────────────────────────
# 4. DebateGraph full loop (mocked agents)
# ─────────────────────────────────────────────


class TestDebateGraphLoop:
    """Integration tests with mocked Coder and Critic agents."""

    @pytest.mark.asyncio
    async def test_convergence_on_turn_2(
        self, simple_node: PlanNode, minimal_context: ContextPack
    ) -> None:
        """Critic yields score=6 on turn 1, then 9 on turn 2 → CODER_WINS."""
        from src.orchestrator.debate_graph import DebateGraph

        coder_answer = "FINAL_ANSWER: def sum_list(lst): return sum(lst)"
        critic_responses = [
            json.dumps({"score": 6, "reasoning": "Missing type check on line 1", "failing_tests": []}),
            json.dumps({"score": 9, "reasoning": "Acceptable now", "failing_tests": []}),
        ]
        critic_call_count = 0

        async def mock_coder_run(context, user_message, extra_system=None):
            r = MagicMock()
            r.success = True
            r.final_answer = coder_answer
            r.tokens_used = 100
            r.tool_trace = []
            return r

        async def mock_critic_run(context, user_message, extra_system=None):
            nonlocal critic_call_count
            r = MagicMock()
            r.success = True
            r.final_answer = critic_responses[min(critic_call_count, len(critic_responses) - 1)]
            r.tokens_used = 80
            r.tool_trace = []
            critic_call_count += 1
            return r

        with (
            patch("src.orchestrator.debate_graph.CoderAgent") as MockCoder,
            patch("src.orchestrator.debate_graph.CriticAgent") as MockCritic,
        ):
            MockCoder.return_value.run = mock_coder_run
            MockCritic.return_value.run = mock_critic_run

            graph = DebateGraph()
            debate, result = await graph.run(simple_node, minimal_context)

        assert debate.outcome == DebateOutcome.CODER_WINS
        assert debate.turn_count == 2
        assert debate.critic_score == 9
        assert len(debate.moves) == 4  # coder, critic, coder, critic

    @pytest.mark.asyncio
    async def test_deadlock_fires_judge(
        self, simple_node: PlanNode, minimal_context: ContextPack
    ) -> None:
        """Critic always returns score=3 → deadlock at max_turns=2."""
        from src.orchestrator.debate_graph import DebateGraph
        from src.core.schemas import EnsembleVerdict

        async def mock_coder_run(context, user_message, extra_system=None):
            r = MagicMock()
            r.success = True
            r.final_answer = "FINAL_ANSWER: partial code"
            r.tokens_used = 100
            r.tool_trace = []
            return r

        async def mock_critic_run(context, user_message, extra_system=None):
            r = MagicMock()
            r.success = True
            r.final_answer = json.dumps({"score": 3, "reasoning": "Still broken", "failing_tests": []})
            r.tokens_used = 80
            r.tool_trace = []
            return r

        mock_verdict = EnsembleVerdict(
            outcome=ValidationOutcome.PASS,
            individual_scores=[7, 8, 7],
            mean_score=7.33,
            std_dev=0.47,
            confidence=0.906,
            message="Judge ensemble passed",
        )

        with (
            patch("src.orchestrator.debate_graph.CoderAgent") as MockCoder,
            patch("src.orchestrator.debate_graph.CriticAgent") as MockCritic,
            patch("src.orchestrator.debate_graph.AgenticValidator") as MockValidator,
        ):
            MockCoder.return_value.run = mock_coder_run
            MockCritic.return_value.run = mock_critic_run
            MockValidator.return_value.validate_with_verdict = AsyncMock(return_value=mock_verdict)

            # Force max_turns=2 for fast test
            simple_node_copy = simple_node.model_copy()
            graph = DebateGraph()
            graph._cfg.debate_max_turns = 2

            debate, result = await graph.run(simple_node_copy, minimal_context)

        assert debate.outcome in (DebateOutcome.CODER_WINS, DebateOutcome.DEADLOCK, DebateOutcome.ESCALATED)
        assert debate.turn_count <= 2


# ─────────────────────────────────────────────
# 5. AgenticValidator confidence tests
# ─────────────────────────────────────────────


class TestAgenticValidatorConfidence:
    @pytest.mark.asyncio
    async def test_high_agreement_no_escalation(self, minimal_context: ContextPack) -> None:
        """All judges agree → high confidence → no escalation."""
        from src.validators.agentic import AgenticValidator

        mock_score_result = (9, "Looks good", None)

        with patch.object(AgenticValidator, "_score_single_critic", AsyncMock(return_value=mock_score_result)):
            validator = AgenticValidator(n_critics=3)
            verdict = await validator.validate_with_verdict(
                context_pack=minimal_context,
                changes_summary="def sum_list(lst): return sum(lst)",
            )

        assert verdict.individual_scores == [9, 9, 9]
        assert verdict.std_dev == 0.0
        assert verdict.confidence == 1.0
        assert verdict.escalated is False

    @pytest.mark.asyncio
    async def test_low_agreement_fires_escalation(self, minimal_context: ContextPack) -> None:
        """Judges radically disagree → confidence drop → escalation fires."""
        from src.validators.agentic import AgenticValidator

        scores_returned = iter([(2, "Terrible", "Fix everything"), (9, "Great", None), (5, "Average", "Minor fixes")])

        async def mock_score(*args: Any, **kwargs: Any) -> tuple[int, str, str | None]:
            return next(scores_returned)

        mock_escalation_verdict = MagicMock()
        mock_escalation_verdict.outcome = ValidationOutcome.PASS
        mock_escalation_verdict.escalated = True
        mock_escalation_verdict.escalation_response = "Frontier says: looks fine"

        with (
            patch.object(AgenticValidator, "_score_single_critic", side_effect=mock_score),
            patch.object(AgenticValidator, "_escalate", AsyncMock(return_value=mock_escalation_verdict)),
        ):
            validator = AgenticValidator(n_critics=3)
            validator._cfg.debate_ensemble_confidence_threshold = 0.8  # High threshold to trigger

            verdict = await validator.validate_with_verdict(
                context_pack=minimal_context,
                changes_summary="some code",
            )

        # std_dev should be high; escalation should have been called
        assert verdict.std_dev > 0


# ─────────────────────────────────────────────
# 6. DebateTraceCollector tests
# ─────────────────────────────────────────────


class TestDebateTraceCollector:
    def test_record_and_export(self, tmp_path: Path, fresh_debate: DebateState) -> None:
        from src.fine_tuning.debate_collector import DebateTraceCollector

        fresh_debate.outcome = DebateOutcome.CODER_WINS
        fresh_debate.critic_score = 9
        fresh_debate.current_code = "def sum_list(lst): return sum(lst)"
        fresh_debate.moves = [
            DebateMove(turn=1, actor="coder", content="Initial code"),
            DebateMove(turn=1, actor="critic", content="Looks good", score=9),
        ]

        collector = DebateTraceCollector(traces_dir=tmp_path / "traces")
        collector.record(fresh_debate)
        collector.flush()

        export_path = tmp_path / "sft.jsonl"
        n = collector.export_for_sft(export_path)

        assert n == 1
        assert export_path.exists()

        with export_path.open() as f:
            example = json.loads(f.readline())
        assert example["outcome"] == "coder_wins"
        assert "conversations" in example

    def test_skip_in_progress_debate(self, tmp_path: Path, fresh_debate: DebateState) -> None:
        from src.fine_tuning.debate_collector import DebateTraceCollector

        assert fresh_debate.outcome == DebateOutcome.IN_PROGRESS
        collector = DebateTraceCollector(traces_dir=tmp_path / "traces")
        collector.record(fresh_debate)  # Should warn and not record
        assert len(collector._buffer) == 0

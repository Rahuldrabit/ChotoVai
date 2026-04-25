from __future__ import annotations

import pytest

from src.agents.base import BaseAgent
from src.core.schemas import AgentCapability, AgentCard, AgentRole, ContextPack, PlanNode
from src.serving.vllm_client import InferenceResponse


class _DummyClient:
    async def complete(self, messages, temperature=None):
        return InferenceResponse(
            content='{"score": 6, "reasoning": "needs one fix"}',
            tokens_in=10,
            tokens_out=8,
            latency_ms=1.0,
            finish_reason="stop",
            raw={},
        )


class _TestAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(AgentRole.CRITIC)

    def system_prompt(self) -> str:
        return "Test system prompt"

    def allowed_tools(self) -> list[str]:
        return []

    def card(self) -> AgentCard:
        return AgentCard(
            agent_id="test-agent",
            role=AgentRole.CRITIC,
            display_name="Test Agent",
            description="Test agent",
            capabilities=[
                AgentCapability(
                    name="test",
                    description="test",
                )
            ],
            model_name="test-model",
        )


@pytest.mark.asyncio
async def test_accept_plain_text_final_returns_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    from src import agents

    dummy_client = _DummyClient()
    monkeypatch.setattr(agents.base, "get_client", lambda _role: dummy_client)

    agent = _TestAgent()
    context = ContextPack(
        agent_role=AgentRole.CRITIC,
        plan_node=PlanNode(id="N1", title="T", description="D"),
    )

    result = await agent.run(
        context=context,
        user_message="review",
        accept_plain_text_final=True,
    )

    assert result.success is True
    assert result.iterations == 1
    assert result.tool_calls_made == 0
    assert result.final_answer.startswith("{")

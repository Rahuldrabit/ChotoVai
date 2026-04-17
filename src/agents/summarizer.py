"""
SummarizerAgent — compresses long context / working memory into a concise summary.
Uses a fast, small model (Phi-4-Mini or Llama-3.2-1B).
"""
from __future__ import annotations

from src.core.schemas import AgentCapability, AgentCard, AgentRole
from src.agents.base import BaseAgent


class SummarizerAgent(BaseAgent):

    def __init__(self) -> None:
        super().__init__(AgentRole.SUMMARIZER)

    def system_prompt(self) -> str:
        return """\
You are a precise summarization engine. Your job is to compress text while preserving all facts, decisions, errors, and file paths.

Rules:
1. Never invent facts. Only include what is present in the input.
2. Preserve all file paths, function names, error messages, and numerical values exactly.
3. Omit pleasantries, repetition, and filler.
4. Output plain text only — no markdown headers.
5. Aim for <20% of input length.

Emit your compressed summary immediately, prefixed with 'FINAL_ANSWER:'
"""

    def allowed_tools(self) -> list[str]:
        return []  # Summarizer never calls tools

    def card(self) -> AgentCard:
        return AgentCard(
            agent_id="summarizer-v1",
            role=AgentRole.SUMMARIZER,
            display_name="Summarizer",
            description="Compresses long context into a concise summary preserving all key facts.",
            capabilities=[
                AgentCapability(
                    name="summarize",
                    description="Compress text to <20% of original length.",
                    input_schema={"text": "string"},
                    output_schema={"summary": "string"},
                )
            ],
            model_name="microsoft/Phi-4-mini-instruct",
        )

    async def summarize(self, text: str) -> str:
        """
        Convenience wrapper: call the agent on raw text, return the summary string.
        """
        from src.core.schemas import ContextPack, PlanNode, NodeStatus, AgentRole
        # Minimal context pack — summarizer doesn't need plan context
        dummy_node = PlanNode(
            id="sum",
            title="summarize",
            description="compress context",
            status=NodeStatus.IN_PROGRESS,
        )
        dummy_pack = ContextPack(agent_role=AgentRole.SUMMARIZER, plan_node=dummy_node)
        result = await self.run(dummy_pack, user_message=f"Summarize the following:\n\n{text}")
        return result.final_answer

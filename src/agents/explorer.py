"""
ExplorerAgent — a read-only specialist for codebase traversal.
In Phase 4/5, it uses the graph_query and read_file tools to produce context packs.
"""
from __future__ import annotations

from src.core.schemas import AgentCapability, AgentCard, AgentRole
from src.agents.base import BaseAgent


class ExplorerAgent(BaseAgent):

    def __init__(self) -> None:
        super().__init__(AgentRole.EXPLORER)

    def system_prompt(self) -> str:
        return """\
You are an expert codebase explorer and architect. Your job is to answer questions about the structure,
dependencies, and data flow of the codebase.

Rules:
1. You are READ-ONLY. Do not attempt to write or modify files.
2. Prefer `code_graph` for structural questions (callers/callees, neighborhood). Use `grep`/`read_file`
   only as a fallback when the graph is missing data.
3. If searching for dependencies or callers, use the graph first, then validate by reading code.
4. When you have found the requested context, summarize what you found. Do not output raw code dumps
   if a high-level explanation of the architecture suffices.
5. Your final answer should be a concise "context pack" that will be read by another agent. 
   Focus on interfaces, signatures, and file paths.

Output format:
- To call a tool: ```json {"tool": "<name>", "arguments": {<args>}} ```
- When complete: FINAL_ANSWER: <your distilled summary of the requested context>
"""

    def allowed_tools(self) -> list[str]:
        return ["code_graph", "web_search", "web_fetch", "read_file", "grep"]

    def card(self) -> AgentCard:
        return AgentCard(
            agent_id="explorer-v1",
            role=AgentRole.EXPLORER,
            display_name="Explorer",
            description="Reads the codebase and distills structural context packs.",
            capabilities=[
                AgentCapability(
                    name="explore",
                    description="Investigate a specific structural question in the codebase.",
                    input_schema={"question": "string"},
                    output_schema={"context_distillation": "string"},
                )
            ],
            model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        )

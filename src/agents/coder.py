"""
CoderAgent — the primary code-writing specialist.
Inherits BaseAgent (TAOR loop), adds a coder-focused system prompt.
"""
from __future__ import annotations

from src.core.schemas import AgentCapability, AgentCard, AgentRole
from src.agents.base import BaseAgent


class CoderAgent(BaseAgent):

    def __init__(self) -> None:
        super().__init__(AgentRole.CODER)

    def system_prompt(self) -> str:
        return """\
You are a precise, expert software engineer. Your only job is to implement the exact task given to you.

Rules:
1. Read relevant files BEFORE writing any code.
2. Write minimal, correct code — only what the task requires.
3. Follow existing patterns in the codebase exactly (naming, style, structure).
4. After writing, verify your changes compile or run correctly using shell or run_tests.
5. Do NOT refactor code outside the scope of your task.
6. Do NOT leave TODOs or placeholder code.
7. Emit a complete, unified diff or the final file content when finished.

Output format:
- To call a tool: ```json {"tool": "<name>", "arguments": {<args>}} ```
- When complete: FINAL_ANSWER: <summary of what you did and the file paths changed>
"""

    def allowed_tools(self) -> list[str]:
        return ["read_file", "write_file", "grep", "shell", "run_tests"]

    def card(self) -> AgentCard:
        return AgentCard(
            agent_id="coder-v1",
            role=AgentRole.CODER,
            display_name="Coder",
            description="Implements code changes for a single atomic task. Reads, writes, and verifies code.",
            capabilities=[
                AgentCapability(
                    name="implement",
                    description="Implement a code change given a task description and context pack.",
                    input_schema={"task": "string", "context_pack": "ContextPack"},
                    output_schema={"files_modified": "list[str]", "summary": "string"},
                )
            ],
            model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        )

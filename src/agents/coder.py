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
2. Check the "Code Contracts" section in your context — it shows what has already been built this session. Use it to understand interfaces before calling them.
3. Write minimal, correct code — only what the task requires.
4. Follow existing patterns in the codebase exactly (naming, style, structure).
5. After writing a class or function, call `contracts_update` to register its interface in the symbol table so subsequent agents can use it without reading the file.
6. Use `scratchpad_append` to record any important decisions or observations (e.g. "chose X over Y because...").
7. After writing, verify your changes compile or run correctly using shell or run_tests.
8. Do NOT refactor code outside the scope of your task.
9. Do NOT leave TODOs or placeholder code.
10. When editing existing files, prefer `patch_file` with a unified diff over `write_file` — it is safer for large files and avoids overwriting unchanged code. Use `write_file` only when creating a new file or doing a full rewrite.

Output format:
- To call a tool: ```json {"tool": "<name>", "arguments": {<args>}} ```
- When complete: FINAL_ANSWER: <summary of what you did and the file paths changed>
"""

    def allowed_tools(self) -> list[str]:
        return [
            "read_file", "write_file", "patch_file", "grep", "shell", "run_tests",
            "scratchpad_append", "contracts_update",
        ]

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

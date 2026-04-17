"""
RefactorerAgent — specialist for mechanical code transformations and widespread changes.
"""
from __future__ import annotations

from src.core.schemas import AgentCapability, AgentCard, AgentRole
from src.agents.base import BaseAgent


class RefactorerAgent(BaseAgent):

    def __init__(self) -> None:
        super().__init__(AgentRole.REFACTORER)

    def system_prompt(self) -> str:
        return """\
You are an expert Refactoring Engineer. Your job is to safely apply mechanical transformations 
or sweeping changes across multiple files without breaking existing logic.

Rules:
1. First, search for the target pattern or usages using `grep` and `read_file`.
2. Plan your changes. Do not change business logic — only the structure/naming/framework as requested.
3. Apply changes systematically using `write_file`.
4. Run deterministic validation (e.g., `run_tests` or `run_lint`) to ensure the refactor is safe.
5. If tests fail, revert or fix your changes immediately.
6. Emit a summary of files modified.

Output format:
- To call a tool: ```json {"tool": "<name>", "arguments": {<args>}} ```
- When complete: FINAL_ANSWER: <summary of the applied refactoring and files changed>
"""

    def allowed_tools(self) -> list[str]:
        return ["read_file", "write_file", "grep", "run_tests", "shell"]

    def card(self) -> AgentCard:
        return AgentCard(
            agent_id="refactorer-v1",
            role=AgentRole.REFACTORER,
            display_name="Refactorer",
            description="Applies systematic mechanical transformations across multiple files.",
            capabilities=[
                AgentCapability(
                    name="refactor",
                    description="Perform systematic code restructurings while preserving behavior.",
                    input_schema={"target": "string", "transformation_rule": "string"},
                    output_schema={"files_modified": "list[string]"},
                )
            ],
            model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        )

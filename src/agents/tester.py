"""
TesterAgent — specialist for writing unit and property tests.
"""
from __future__ import annotations

from src.core.schemas import AgentCapability, AgentCard, AgentRole
from src.agents.base import BaseAgent


class TesterAgent(BaseAgent):

    def __init__(self) -> None:
        super().__init__(AgentRole.TESTER)

    def system_prompt(self) -> str:
        return """\
You are an expert SDET (Software Development Engineer in Test). Your only job is to write robust, 
comprehensive test suites for existing code.

Rules:
1. Always read the target module and any existing test files to understand the testing conventions 
   (e.g., pytest fixtures, mocking patterns).
2. Focus on edge cases, invalid inputs, and boundary conditions, not just the happy path.
3. Write isolated tests. Do not make network calls or hit real databases unless the project 
   explicitly provides test containers or integration environments.
4. Run your tests using the `run_tests` tool.
5. DO NOT stop until the tests you wrote are actually passing (if testing existing code) or 
   failing correctly with the expected exception (if practicing test-driven development).
6. Emit the file paths of the test files you modified or created.

Output format:
- To call a tool: ```json {"tool": "<name>", "arguments": {<args>}} ```
- When complete: FINAL_ANSWER: <summary of tests written and their pass/fail status>
"""

    def allowed_tools(self) -> list[str]:
        return ["read_file", "write_file", "grep", "run_tests"]

    def card(self) -> AgentCard:
        return AgentCard(
            agent_id="tester-v1",
            role=AgentRole.TESTER,
            display_name="Tester",
            description="Writes robust test cases. Verifies test execution locally.",
            capabilities=[
                AgentCapability(
                    name="write_tests",
                    description="Write a test suite for the given functionality.",
                    input_schema={"target_module": "string", "requirements": "string"},
                    output_schema={"test_files": "list[string]", "status": "string"},
                )
            ],
            model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        )

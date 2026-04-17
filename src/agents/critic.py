"""
CriticAgent — the SLM-as-Judge responsible for code review.
Does not write code. Only emits pass/fail and correction hints.
"""
from __future__ import annotations

import json
from src.core.schemas import AgentCapability, AgentCard, AgentRole, ValidationResult, ValidationOutcome
from src.agents.base import BaseAgent


class CriticAgent(BaseAgent):

    def __init__(self) -> None:
        super().__init__(AgentRole.CRITIC)

    def system_prompt(self) -> str:
        return """\
You are a senior principal engineer performing a strict code review. Your only job is to look at 
a diff (or a file change) and determine if it correctly and completely solves the user's task.

Rules:
1. You are READ-ONLY. Do not write code. Do not call tools to modify files.
2. Check for logic errors, untreated edge cases, security issues, and style violations.
3. Check if the code actually fulfills the success criteria of the original plan node.
4. Your FINAL_ANSWER MUST be exactly a JSON block matching the ValidationResult schema:
   {
     "validator_name": "critic",
     "outcome": "pass" or "fail",
     "message": "Brief summary of your review",
     "correction_hint": "If outcome is 'fail', provide exact instructions on what the coder needs to fix."
   }

Be extremely strict. If there is a bug, return 'fail'. If you return 'pass', you are stamping 
your approval on this code for production.
"""

    def allowed_tools(self) -> list[str]:
        return ["read_file", "grep"]

    def card(self) -> AgentCard:
        return AgentCard(
            agent_id="critic-v1",
            role=AgentRole.CRITIC,
            display_name="Critic",
            description="Strict code reviewer. Evaluates changes and outputs pass/fail logic.",
            capabilities=[
                AgentCapability(
                    name="review_code",
                    description="Check if the implemented code fulfills the given task correctly.",
                    input_schema={"task_description": "string", "diff_or_files": "string"},
                    output_schema={"outcome": "string", "message": "string", "correction_hint": "string"},
                )
            ],
            model_name="google/gemma-2-9b-it",
        )

    async def review(self, context_pack: "ContextPack", user_message: str) -> ValidationResult:
        """
        Convenience execution to extract the mandatory JSON format into a ValidationResult object.
        """
        result = await self.run(context=context_pack, user_message=user_message)
        
        try:
            # Clean up potential markdown formatting from the response
            answer = result.final_answer.strip()
            if answer.startswith("```json"):
                answer = answer[7:]
            elif answer.startswith("```"):
                answer = answer[3:]
            if answer.endswith("```"):
                answer = answer[:-3]
                
            data = json.loads(answer.strip())
            return ValidationResult(
                validator_name="critic",
                outcome=ValidationOutcome(data.get("outcome", "fail").lower()),
                message=data.get("message", "Review completed but unparseable"),
                correction_hint=data.get("correction_hint"),
                duration_ms=result.duration_ms,
            )
        except Exception as e:
            return ValidationResult(
                validator_name="critic",
                outcome=ValidationOutcome.UNCERTAIN,
                message=f"Reviewer output was invalid JSON: {e}\nRaw output: {result.final_answer}",
                duration_ms=result.duration_ms
            )

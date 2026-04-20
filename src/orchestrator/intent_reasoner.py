"""
Intent Reasoner — intercepts user queries before planning.
Deconstructs intent, extracts code blocks, and reconstructs an explicit instruction set.
"""
from __future__ import annotations

import structlog
from pydantic import BaseModel, Field

from src.core.schemas import AgentMessage, AgentRole, IntentAnalysis
from src.serving.model_registry import get_client

logger = structlog.get_logger(__name__)

_INTENT_SYSTEM = """\
You are the Cognitive Intent Reasoner. Your job is to analyze a user's raw prompt, \
and generate exactly 3 DISTINCT interpretations of their true intent (e.g., literal vs defensive vs architectural).

For each interpretation:
1. Extract the `primary_intent` (1-2 sentences max).
2. List `explicit_constraints` the user asked for (e.g. "no downtime", "use fastAPI").
3. List `implicit_assumptions` the user likely implies but didn't state.
4. Extract any pasted raw code or function definitions entirely into `extracted_code_snippets`.
5. Provide a `rewritten_query` that is a highly explicit, structured set of instructions.

Output valid JSON matching the schema containing an array 'interpretations' of exactly 3 variants.
"""

_EVALUATOR_SYSTEM = """\
You are a senior technical judge. Review the user's raw prompt, and evaluate the 3 proposed interpretations of their intent.
Select the single best interpretation that is the most logical, safest, and most actionable for an AI Planner.
Your output must be JSON specifying the 0-indexed integer of the best interpretation, and a brief reasoning.
"""

_LITE_REWRITE_SYSTEM = """\
You are a technical intent clarifier. Given a user's coding task, produce a single \
highly explicit, structured instruction set that an AI planner can execute unambiguously.
Output valid JSON with exactly these fields:
- primary_intent (string): one-sentence summary of what the user wants
- explicit_constraints (list[str]): requirements the user stated directly
- implicit_assumptions (list[str]): things the user likely implies but didn't state
- extracted_code_snippets (list[str]): any pasted code blocks from the query
- rewritten_query (string): a detailed, explicit version of the task for the planner
"""

class _IntentToTResponse(BaseModel):
    interpretations: list[IntentAnalysis] = Field(..., min_length=3, max_length=3)

class _IntentEvaluation(BaseModel):
    best_index: int = Field(description="The 0-based index of the best interpretation (0, 1, or 2)")
    reasoning: str

class IntentReasoner:
    """
    Analyzes raw user queries to extract intent, constraints, and pasted code.
    Uses a Tree of Thoughts (ToT) approach by generating multiple interpretations and scoring them.
    """

    def __init__(self) -> None:
        self._client = get_client(AgentRole.ORCHESTRATOR)

    async def analyze_and_rewrite(self, raw_query: str, max_retries: int = 2) -> IntentAnalysis:
        """Analyze the query using ToT and return the best structured IntentAnalysis."""
        # Phase 1: Generate 3 interpretations
        messages = [
            AgentMessage(role="system", content=_INTENT_SYSTEM),
            AgentMessage(role="user", content=raw_query),
        ]

        tot_response: _IntentToTResponse | None = None
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                response_obj = await self._client.complete_json(
                    messages=messages,
                    schema=_IntentToTResponse,
                )
                assert isinstance(response_obj, _IntentToTResponse)
                tot_response = response_obj
                logger.info(
                    "intent_reasoner.tot_generated",
                    variants=len(tot_response.interpretations),
                    attempt=attempt,
                )
                break
            except Exception as e:
                last_error = e
                logger.warning("intent_reasoner.tot_failed", attempt=attempt, error=str(e))
                messages.append(AgentMessage(role="assistant", content=f"(failed: {e})"))
                messages.append(AgentMessage(role="user", content="Fix JSON output. Must be exactly 3 interpretations."))

        if not tot_response:
            raise RuntimeError(f"IntentReasoner ToT phase failed: {last_error}")

        # Phase 2: Evaluate and select the best interpretation
        eval_messages = [
            AgentMessage(role="system", content=_EVALUATOR_SYSTEM),
            AgentMessage(role="user", content=f"Raw Query:\n{raw_query}\n\nProposed Interpretations:\n"),
        ]
        
        for idx, interpretation in enumerate(tot_response.interpretations):
            eval_messages[-1].content += f"[{idx}] Intent: {interpretation.primary_intent}\n"
            eval_messages[-1].content += f"    Rewritten: {interpretation.rewritten_query}\n"

        for attempt in range(1, max_retries + 1):
            try:
                eval_obj = await self._client.complete_json(
                    messages=eval_messages,
                    schema=_IntentEvaluation,
                )
                assert isinstance(eval_obj, _IntentEvaluation)
                best_idx = max(0, min(2, eval_obj.best_index))
                
                logger.info(
                    "intent_reasoner.evaluation_complete",
                    best_index=best_idx,
                    reasoning=eval_obj.reasoning,
                )
                return tot_response.interpretations[best_idx]
            except Exception as e:
                logger.warning("intent_reasoner.eval_failed", attempt=attempt, error=str(e))
                eval_messages.append(AgentMessage(role="assistant", content=f"(failed: {e})"))
                eval_messages.append(AgentMessage(role="user", content="Output valid JSON with best_index and reasoning."))
        
        # Fallback to the first interpretation if evaluation fails
        logger.warning("intent_reasoner.eval_fallback")
        return tot_response.interpretations[0]

    async def rewrite_lite(self, raw_query: str, max_retries: int = 2) -> IntentAnalysis:
        """Single-call intent rewrite — no ToT selection phase. Used for MODERATE tasks."""
        messages = [
            AgentMessage(role="system", content=_LITE_REWRITE_SYSTEM),
            AgentMessage(role="user", content=raw_query),
        ]
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                result = await self._client.complete_json(messages=messages, schema=IntentAnalysis)
                assert isinstance(result, IntentAnalysis)
                logger.info("intent_reasoner.lite_rewrite_complete", attempt=attempt)
                return result
            except Exception as e:
                last_error = e
                logger.warning("intent_reasoner.lite_rewrite_failed", attempt=attempt, error=str(e))
                messages.append(AgentMessage(role="assistant", content=f"(failed: {e})"))
                messages.append(AgentMessage(role="user", content="Fix JSON and retry."))
        # Fallback: return minimal analysis preserving the raw query
        logger.warning("intent_reasoner.lite_rewrite_fallback", error=str(last_error))
        return IntentAnalysis(
            primary_intent=raw_query,
            rewritten_query=raw_query,
        )

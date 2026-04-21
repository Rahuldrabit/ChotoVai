"""
Planner — the Orchestrator's main module.
Parses the user goal and produces a TaskDAG (structured plan) using the orchestrator LLM.
Schema enforcement: output MUST parse as TaskDAG_Response or it is rejected.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

import structlog

from src.core.schemas import AgentMessage, AgentRole, NodeStatus, PlanNode, TaskDAG
from src.serving.model_registry import get_client

logger = structlog.get_logger(__name__)


class _PlanNodeSpec(BaseModel):
    """Planner output schema for a single plan node."""
    id: str
    title: str
    description: str
    assigned_role: str  # AgentRole value
    depends_on: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    cognitive_strategy: str = "direct"  # CognitiveStrategy value


class _TaskDAGResponse(BaseModel):
    """The JSON schema the orchestrator LLM must emit."""
    title: str
    description: str
    nodes: list[_PlanNodeSpec]


_DRAFT_SYSTEM = """\
You are a senior software architect. Analyze the user's coding goal and propose exactly 3 DISTINCT high-level architectural approaches to solve it.
Each approach should outline the core logic, which files might be touched, and potential risks or tradeoffs.
Output valid JSON containing an array 'approaches' of exactly 3 strings.
"""

_EVALUATOR_SYSTEM = """\
You are a senior technical lead. Review the 3 proposed approaches to solve the user's coding goal.
Select the single best approach based on robustness, simplicity, and alignment with modern software patterns.
Your output must be JSON specifying the 0-indexed integer of the best approach, and a brief reasoning.
"""

_PLANNER_SYSTEM = """\
You are a senior software architect and task planner. Your ONLY job is to analyse a coding goal and a selected architectural approach,
and produce a structured execution plan as a JSON task graph.

Rules:
1. Break the goal into ATOMIC, independently-delegatable sub-tasks based strictly on the selected approach.
2. Identify dependencies explicitly in each node's `depends_on` field (list of node IDs).
3. Assign the correct `assigned_role` for each node:
   - "coder"      — write or modify code
   - "tester"     — write tests
   - "explorer"   — investigate/read the codebase
   - "refactorer" — mechanical code transformations
   - "critic"     — code review
   - "doc_reader" — read documentation
4. Provide 1–3 concrete `success_criteria` per node (how we know the node is done).
5. Set `cognitive_strategy` for each node:
   - "debate"    — complex or security-critical code (uses adversarial Coder↔Critic loop)
   - "refine"    — medium complexity (one critic improvement pass)
   - "direct"    — simple, read-only, or mechanical tasks (single-shot)
   - "verify"    — test-focused nodes that need deterministic validation
   - "escalate"  — known hard problems that require frontier model reasoning
   - "decompose" — node covers too many concerns; the executor will split it into subtasks at runtime
6. Output ONLY valid JSON matching the schema — no explanations, no markdown.

Node IDs must be short strings: "N1", "N2", etc.
"""

class _DraftResponse(BaseModel):
    approaches: list[str] = Field(..., min_length=3, max_length=3)

class _ApproachEvaluation(BaseModel):
    best_index: int = Field(description="0-based index of the best approach")
    reasoning: str

class Planner:
    """
    Converts a user goal string into a TaskDAG using the orchestrator LLM with
    guided JSON decoding, enhanced with a ToT approach-drafting phase.
    """

    def __init__(self) -> None:
        self._client = get_client(AgentRole.ORCHESTRATOR)

    async def _draft_and_select_approach(self, user_content: str, max_retries: int) -> str:
        """Phase 1 & 2: Generate candidate approaches and select the best one."""
        draft_msgs = [
            AgentMessage(role="system", content=_DRAFT_SYSTEM),
            AgentMessage(role="user", content=user_content),
        ]
        
        draft_resp: _DraftResponse | None = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = await self._client.complete_json(messages=draft_msgs, schema=_DraftResponse)
                assert isinstance(resp, _DraftResponse)
                draft_resp = resp
                break
            except Exception as e:
                logger.warning("planner.draft_failed", attempt=attempt, error=str(e))
                draft_msgs.append(AgentMessage(role="assistant", content=str(e)))
                draft_msgs.append(AgentMessage(role="user", content="Fix JSON output. Must have 3 approaches."))

        if not draft_resp:
            logger.warning("planner.draft_fallback")
            return "Proceed with a direct, standard architectural approach."

        # Evaluate
        eval_content = f"Goal:\n{user_content}\n\nApproaches:\n"
        for i, app in enumerate(draft_resp.approaches):
            eval_content += f"[{i}] {app}\n\n"
            
        eval_msgs = [
            AgentMessage(role="system", content=_EVALUATOR_SYSTEM),
            AgentMessage(role="user", content=eval_content),
        ]
        
        for attempt in range(1, max_retries + 1):
            try:
                eval_obj = await self._client.complete_json(messages=eval_msgs, schema=_ApproachEvaluation)
                assert isinstance(eval_obj, _ApproachEvaluation)
                best_idx = max(0, min(2, eval_obj.best_index))
                logger.info("planner.approach_selected", best_index=best_idx, reasoning=eval_obj.reasoning)
                return draft_resp.approaches[best_idx]
            except Exception as e:
                logger.warning("planner.eval_failed", attempt=attempt, error=str(e))
                eval_msgs.append(AgentMessage(role="assistant", content=str(e)))
                eval_msgs.append(AgentMessage(role="user", content="Fix JSON output."))

        return draft_resp.approaches[0]

    async def plan(
        self,
        goal: str,
        memory_summary: str | None = None,
        repo_summary: str | None = None,
        max_retries: int = 3,
    ) -> TaskDAG:
        """
        Generate a TaskDAG for the given goal using a ToT strategy.
        """
        base_content = f"Goal: {goal}"
        if repo_summary:
            base_content += f"\n\nCodebase context:\n{repo_summary}"
        if memory_summary:
            base_content += f"\n\nRelevant past experience:\n{memory_summary}"

        # Phase 1 & 2: ToT Drafting
        selected_approach = await self._draft_and_select_approach(base_content, max_retries)

        # Phase 3: DAG Generation
        dag_content = f"{base_content}\n\nSelected Architectural Approach:\n{selected_approach}\n\nNow, generate the exact TaskDAG to execute this approach."
        messages = [
            AgentMessage(role="system", content=_PLANNER_SYSTEM),
            AgentMessage(role="user", content=dag_content),
        ]

        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                response_obj = await self._client.complete_json(
                    messages=messages,
                    schema=_TaskDAGResponse,
                )
                assert isinstance(response_obj, _TaskDAGResponse)
                dag = self._convert(response_obj, goal)
                logger.info("planner.success", nodes=len(dag.nodes), attempt=attempt)
                return dag
            except Exception as e:
                last_error = e
                logger.warning("planner.failed", attempt=attempt, error=str(e))
                messages.append(AgentMessage(role="assistant", content=f"(previous attempt failed: {e})"))
                messages.append(AgentMessage(role="user", content="Your JSON was invalid. Please correct it and output ONLY valid JSON."))

        raise RuntimeError(f"Planner failed after {max_retries} attempts: {last_error}")

    async def plan_lite(
        self,
        goal: str,
        memory_summary: str | None = None,
        repo_summary: str | None = None,
        max_retries: int = 2,
    ) -> TaskDAG:
        """
        Lightweight single-call planner for MODERATE complexity tasks.
        Skips the ToT approach-drafting phase — calls the model once directly.
        """
        base_content = f"Goal: {goal}"
        if repo_summary:
            base_content += f"\n\nCodebase context:\n{repo_summary}"
        if memory_summary:
            base_content += f"\n\nRelevant past experience:\n{memory_summary}"

        messages = [
            AgentMessage(role="system", content=_PLANNER_SYSTEM),
            AgentMessage(role="user", content=f"{base_content}\n\nGenerate a TaskDAG to execute this goal."),
        ]
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                response_obj = await self._client.complete_json(
                    messages=messages,
                    schema=_TaskDAGResponse,
                )
                assert isinstance(response_obj, _TaskDAGResponse)
                dag = self._convert(response_obj, goal)
                logger.info("planner.lite_success", nodes=len(dag.nodes), attempt=attempt)
                return dag
            except Exception as e:
                last_error = e
                logger.warning("planner.lite_failed", attempt=attempt, error=str(e))
                messages.append(AgentMessage(role="assistant", content=f"(failed: {e})"))
                messages.append(AgentMessage(role="user", content="Fix JSON and output ONLY valid JSON."))
        raise RuntimeError(f"plan_lite failed after {max_retries} attempts: {last_error}")

    @staticmethod
    def _convert(response: _TaskDAGResponse, goal: str) -> TaskDAG:
        """Convert the LLM response schema into a TaskDAG."""
        from src.core.schemas import CognitiveStrategy
        nodes = []
        for spec in response.nodes:
            # Validate role
            try:
                role = AgentRole(spec.assigned_role)
            except ValueError:
                role = AgentRole.CODER  # default fallback

            # Validate cognitive strategy
            try:
                strategy = CognitiveStrategy(spec.cognitive_strategy)
            except ValueError:
                strategy = CognitiveStrategy.DIRECT  # default fallback

            nodes.append(PlanNode(
                id=spec.id,
                title=spec.title,
                description=spec.description,
                assigned_role=role,
                depends_on=spec.depends_on,
                success_criteria=spec.success_criteria,
                cognitive_strategy=strategy,
                status=NodeStatus.PENDING,
            ))

        return TaskDAG(
            title=response.title,
            description=response.description or goal,
            nodes=nodes,
        )

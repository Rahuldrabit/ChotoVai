"""
NodeDecomposer — breaks a broad PlanNode into 2–5 atomic child nodes.

Called by the FSM when CognitiveStrategy.DECOMPOSE is selected.
Uses the same ORCHESTRATOR LLM client as the Planner.
"""
from __future__ import annotations

from pydantic import BaseModel, Field
import structlog

from src.core.schemas import AgentMessage, AgentRole, CognitiveStrategy, NodeStatus, PlanNode
from src.serving.model_registry import get_client

logger = structlog.get_logger(__name__)


_DECOMPOSE_SYSTEM = """\
You are given a single task node from a coding agent's plan that is too broad or complex
for one agent to complete in a single pass.

Your job: decompose it into 2–5 ATOMIC, independently-executable subtasks.

Rules:
1. Each subtask must be completable by a single specialist agent in one pass.
2. Assign the correct `assigned_role`:
   - "coder"      — write or modify code
   - "tester"     — write tests
   - "explorer"   — investigate/read the codebase
   - "refactorer" — mechanical code transformations
   - "critic"     — code review
3. Assign `cognitive_strategy`:
   - "debate"   — complex or security-critical code
   - "refine"   — medium complexity
   - "direct"   — simple, read-only, or mechanical
   - "verify"   — test-focused
4. Node IDs: prefix with the parent node ID.
   Parent "N3" → children "N3a", "N3b", "N3c", etc.
5. depends_on: first child has NO deps; each later child depends on the previous one
   (sequential). Only omit sequential deps if tasks are truly independent.
6. Provide 1–2 concrete `success_criteria` per subtask.
7. Output ONLY valid JSON: {"nodes": [<list of node specs>]}. No markdown, no explanation.
"""


class _ChildNodeSpec(BaseModel):
    id: str
    title: str
    description: str
    assigned_role: str = "coder"
    depends_on: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    cognitive_strategy: str = "direct"


class _DecomposeResponse(BaseModel):
    nodes: list[_ChildNodeSpec] = Field(min_length=2, max_length=5)


class NodeDecomposer:
    """
    Decomposes a single broad PlanNode into 2–5 atomic child PlanNodes
    by asking the ORCHESTRATOR LLM to split it.
    """

    def __init__(self) -> None:
        self._client = get_client(AgentRole.ORCHESTRATOR)

    async def decompose(
        self,
        node: PlanNode,
        context_summary: str = "",
        max_retries: int = 3,
    ) -> list[PlanNode]:
        """
        Ask the LLM to break `node` into child nodes.

        Returns a list of 2–5 PlanNode objects ready to be injected into the DAG.
        Falls back to a single "direct" child mirroring the parent if the LLM fails.
        """
        user_content = (
            f"Parent Node ID: {node.id}\n"
            f"Title: {node.title}\n"
            f"Description: {node.description}\n"
            f"Success Criteria:\n"
            + "\n".join(f"  - {c}" for c in node.success_criteria)
        )
        if context_summary:
            user_content += f"\n\nCodebase Context (brief):\n{context_summary[:600]}"

        messages = [
            AgentMessage(role="system", content=_DECOMPOSE_SYSTEM),
            AgentMessage(role="user", content=user_content),
        ]

        for attempt in range(1, max_retries + 1):
            try:
                resp = await self._client.complete_json(
                    messages=messages, schema=_DecomposeResponse
                )
                assert isinstance(resp, _DecomposeResponse)
                children = self._convert(resp, node)
                logger.info(
                    "node_decomposer.success",
                    parent_id=node.id,
                    children=[c.id for c in children],
                    attempt=attempt,
                )
                return children
            except Exception as exc:
                logger.warning(
                    "node_decomposer.failed",
                    attempt=attempt,
                    parent_id=node.id,
                    error=str(exc),
                )
                messages.append(AgentMessage(role="assistant", content=str(exc)))
                messages.append(
                    AgentMessage(
                        role="user",
                        content="Your JSON was invalid. Correct it and output ONLY valid JSON.",
                    )
                )

        # Fallback: treat the node as a single direct child (no-op decomposition)
        logger.warning("node_decomposer.fallback", parent_id=node.id)
        fallback = PlanNode(
            id=f"{node.id}a",
            title=node.title,
            description=node.description,
            assigned_role=node.assigned_role,
            depends_on=[],
            success_criteria=node.success_criteria,
            cognitive_strategy=CognitiveStrategy.DIRECT,
            status=NodeStatus.PENDING,
            parent_node_id=node.id,
        )
        return [fallback]

    @staticmethod
    def _convert(resp: _DecomposeResponse, parent: PlanNode) -> list[PlanNode]:
        children: list[PlanNode] = []
        for spec in resp.nodes:
            try:
                role = AgentRole(spec.assigned_role)
            except ValueError:
                role = parent.assigned_role or AgentRole.CODER

            try:
                strategy = CognitiveStrategy(spec.cognitive_strategy)
            except ValueError:
                strategy = CognitiveStrategy.DIRECT

            children.append(
                PlanNode(
                    id=spec.id,
                    title=spec.title,
                    description=spec.description,
                    assigned_role=role,
                    depends_on=spec.depends_on,
                    success_criteria=spec.success_criteria,
                    cognitive_strategy=strategy,
                    status=NodeStatus.PENDING,
                )
            )
        return children

"""
Plan state — the orchestrator's task DAG serialized to disk + in-memory.
Single source of truth for where the system is in the task.
Only the orchestrator writes this; all other agents read it.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import UUID

import structlog

from src.core.schemas import NodeStatus, PlanNode, TaskDAG

logger = structlog.get_logger(__name__)


class PlanState:
    """
    Wraps TaskDAG with persistence to a JSON file (checkpointing).
    Also renders a Markdown view for human inspection.
    """

    def __init__(self, data_dir: str | Path = "./data") -> None:
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._dag: TaskDAG | None = None

    @property
    def dag(self) -> TaskDAG | None:
        return self._dag

    def set_dag(self, dag: TaskDAG) -> None:
        self._dag = dag
        self._persist()

    def update_node(self, node_id: str, **kwargs) -> None:
        """Update fields on a plan node and persist."""
        if self._dag is None:
            raise RuntimeError("No DAG loaded")
        node = self._dag.get_node(node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found in DAG")
        for k, v in kwargs.items():
            setattr(node, k, v)
        self._dag.updated_at = datetime.utcnow()
        self._persist()
        logger.debug("plan_state.node_updated", node_id=node_id, updates=kwargs)

    def mark_node_complete(self, node_id: str, result_summary: str, files_modified: list[str] | None = None) -> None:
        self.update_node(
            node_id,
            status=NodeStatus.COMPLETE,
            result_summary=result_summary,
            files_modified=files_modified or [],
            completed_at=datetime.utcnow(),
        )

    def mark_node_failed(self, node_id: str, reason: str) -> None:
        self.update_node(node_id, status=NodeStatus.FAILED, result_summary=reason)

    def load(self, task_id: str | UUID) -> TaskDAG | None:
        """Load a previously persisted DAG from disk."""
        p = self._dir / f"plan_{task_id}.json"
        if not p.exists():
            return None
        try:
            self._dag = TaskDAG.model_validate_json(p.read_text())
            logger.info("plan_state.loaded", task_id=str(task_id))
            return self._dag
        except Exception as e:
            logger.warning("plan_state.load_failed", error=str(e))
            return None

    def to_markdown(self) -> str:
        """Render the current DAG as a checklist Markdown string."""
        if self._dag is None:
            return "(no plan loaded)"
        lines = [f"# {self._dag.title}\n", f"_{self._dag.description}_\n"]
        for node in self._dag.nodes:
            icon = {
                NodeStatus.PENDING: "- [ ]",
                NodeStatus.IN_PROGRESS: "- [/]",
                NodeStatus.COMPLETE: "- [x]",
                NodeStatus.FAILED: "- [!]",
                NodeStatus.SKIPPED: "- [-]",
            }.get(node.status, "- [ ]")
            deps = f" (deps: {', '.join(node.depends_on)})" if node.depends_on else ""
            lines.append(f"{icon} **{node.id}**: {node.title}{deps}")
            if node.result_summary:
                lines.append(f"    > {node.result_summary}")
        return "\n".join(lines)

    def _persist(self) -> None:
        if self._dag is None:
            return
        p = self._dir / f"plan_{self._dag.task_id}.json"
        p.write_text(self._dag.model_dump_json(indent=2))
        md_p = self._dir / f"plan_{self._dag.task_id}.md"
        md_p.write_text(self.to_markdown())

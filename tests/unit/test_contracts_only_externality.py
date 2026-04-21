from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import reset_config
from src.core.schemas import AgentRole, PlanNode
from src.memory.context_assembler import ContextAssembler
from src.memory.scratchpad import SessionScratchpad


@pytest.mark.asyncio
async def test_contracts_only_externality_blocks_scratchpad_for_workers(tmp_path: Path) -> None:
    reset_config()  # ensure default config (contracts_only_externality=True)

    sp = SessionScratchpad(tmp_path / "scratchpad.md")
    sp.append("hello from scratchpad", role="orchestrator", node_id="session")

    assembler = ContextAssembler(episodic_store=None, scratchpad=sp, contracts=None)
    node = PlanNode(id="N1", title="Do thing", description="Work")

    ctx_worker = await assembler.assemble(role=AgentRole.CODER, plan_node=node)
    assert ctx_worker.scratchpad_tail is None

    ctx_orch = await assembler.assemble(role=AgentRole.ORCHESTRATOR, plan_node=node)
    assert ctx_orch.scratchpad_tail is not None

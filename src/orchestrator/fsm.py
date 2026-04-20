"""
Deterministic FSM (Finite State Machine) controlling the orchestration loop.
States: PLANNING → EXECUTING → (DEBATING) → VALIDATING → (ESCALATING) → COMPLETE | FAILED

Phase 4 upgrade: debate-aware _execute_node routes code-producing nodes through
the adversarial DebateGraph instead of single-shot TAOR execution. Strategy
selection is delegated to CognitiveRouter (RLoT-style).
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import AsyncIterator

import structlog

from src.core.config import get_config
from src.core.schemas import (
    AgentRole,
    AgentState,
    CognitiveStrategy,
    ContextPack,
    DebateOutcome,
    FSMState,
    NodeStatus,
    PlanNode,
    TaskDAG,
    TokenBudget,
    ValidationOutcome,
    ValidationResult,
)
from src.memory.context_assembler import ContextAssembler
from src.memory.contracts import ContractStore, set_active_contracts
from src.memory.episodic import EpisodicStore
from src.memory.plan_state import PlanState
from src.memory.scratchpad import SessionScratchpad, set_active_scratchpad
from src.orchestrator.cognitive_router import CognitiveRouter
from src.orchestrator.planner import Planner
from src.orchestrator.task_router import TaskComplexity, TaskRouter
from src.validators.deterministic import DeterministicValidator
from tools.verify_repo import verify_repo

logger = structlog.get_logger(__name__)


class AgentFSM:
    """
    Deterministic orchestration loop with adversarial debate integration.

    Usage:
        fsm = AgentFSM()
        async for event in fsm.run("Add rate limiting to the login endpoint."):
            print(event)
    """

    def __init__(self, data_dir: str | Path | None = None) -> None:
        cfg = get_config()
        self._cfg = cfg.agent
        self._data_dir = Path(data_dir or cfg.data_dir)
        self._planner = Planner()
        self._validator = DeterministicValidator()
        self._episodic = EpisodicStore()
        self._plan_state = PlanState(data_dir=self._data_dir)
        self._cognitive_router = CognitiveRouter()
        from src.serving.model_registry import get_client
        self._task_router = TaskRouter(client=get_client(AgentRole.ORCHESTRATOR))
        # Scratchpad and contracts are initialised per-session in run()
        self._scratchpad: SessionScratchpad | None = None
        self._contracts: ContractStore | None = None
        self._context_assembler = ContextAssembler(episodic_store=self._episodic)
        # Per-session safety valve tracking
        self._trivial_promoted = False
        self._current_complexity: TaskComplexity | None = None

    async def run(self, goal: str) -> AsyncIterator[dict]:
        """
        Execute the full orchestration loop for a user goal.
        Yields progress events as dicts for the CLI to display.
        """
        t0 = time.perf_counter()

        # Initialize state
        budget = TokenBudget(total_limit=self._cfg.max_total_tokens)
        state = AgentState(user_goal=goal, budget=budget)
        self._checkpoint(state)

        # ── Per-session external memory (scratchpad + contracts) ──────────
        session_dir = self._data_dir / "sessions" / str(state.session_id)
        self._scratchpad = set_active_scratchpad(session_dir)
        self._contracts = set_active_contracts(session_dir)
        self._context_assembler = ContextAssembler(
            episodic_store=self._episodic,
            scratchpad=self._scratchpad,
            contracts=self._contracts,
        )
        self._scratchpad.append(
            f"Session started.\nGoal: {goal}",
            role="orchestrator",
            node_id="session",
        )

        yield {"event": "start", "goal": goal, "session_id": str(state.session_id)}

        # ── COMPLEXITY ROUTING ────────────────────────────────────────────
        # Classify the goal before spending model calls on planning.
        # TRIVIAL  → build 1-node DAG directly (0 extra model calls)
        # MODERATE → lite planner, single model call, skip ToT drafting
        # COMPLEX  → full intent reasoning + ToT planning (existing path)
        classification = await self._task_router.classify(goal)
        complexity = classification.complexity
        self._current_complexity = complexity
        self._trivial_promoted = False  # Reset per-session
        yield {"event": "routing", "complexity": complexity.value}
        logger.info("fsm.routing", complexity=complexity.value, goal=goal[:80])

        planning_goal = goal
        extracted_code = classification.code_snippets  # Preserve extracted code for intent reasoning
        dag: TaskDAG

        if complexity == TaskComplexity.TRIVIAL:
            dag = self._task_router.build_trivial_dag(goal)

        elif complexity == TaskComplexity.MODERATE:
            state.fsm_state = FSMState.PLANNING
            self._checkpoint(state)

            # Lite intent rewrite — 1 model call, no ToT selection
            planning_goal = goal
            yield {"event": "reasoning", "message": "Clarifying intent (lite)..."}
            try:
                from src.orchestrator.intent_reasoner import IntentReasoner
                reasoner = IntentReasoner()
                intent = await reasoner.rewrite_lite(raw_query=goal, code_snippets=extracted_code)
                state.intent_analysis = intent
                self._checkpoint(state)
                planning_goal = intent.rewritten_query
            except Exception as e:
                yield {"event": "warn", "message": f"Lite intent rewrite failed: {e}. Using raw goal."}

            yield {"event": "planning", "message": "Generating task plan (lite)..."}
            try:
                dag = await self._planner.plan_lite(goal=planning_goal)
            except Exception as e:
                state.fsm_state = FSMState.FAILED
                state.error_history.append(f"Planning failed: {e}")
                self._checkpoint(state)
                yield {"event": "failed", "reason": str(e)}
                return

        else:  # COMPLEX — full intent reasoning + ToT planning
            # ── REASONING ─────────────────────────────────────────────────
            yield {"event": "reasoning", "message": "Analyzing user intent..."}
            try:
                from src.orchestrator.intent_reasoner import IntentReasoner
                reasoner = IntentReasoner()
                intent = await reasoner.analyze_and_rewrite(raw_query=goal, code_snippets=extracted_code)
                state.intent_analysis = intent
                self._checkpoint(state)
                planning_goal = intent.rewritten_query
            except Exception as e:
                yield {"event": "warn", "message": f"Intent reasoning failed: {e}. Falling back to raw goal."}

            # ── PLANNING ──────────────────────────────────────────────────
            state.fsm_state = FSMState.PLANNING
            self._checkpoint(state)
            yield {"event": "planning", "message": "Generating task plan..."}
            try:
                dag = await self._planner.plan(goal=planning_goal)
            except Exception as e:
                state.fsm_state = FSMState.FAILED
                state.error_history.append(f"Planning failed: {e}")
                self._checkpoint(state)
                yield {"event": "failed", "reason": str(e)}
                return

        # ── Persist plan (all tiers) ──────────────────────────────────────
        state.dag = dag
        self._plan_state.set_dag(dag)
        plan_md = self._plan_state.to_markdown()
        yield {
            "event": "plan_ready",
            "plan_markdown": plan_md,
            "node_count": len(dag.nodes),
        }
        if self._scratchpad:
            self._scratchpad.append(
                f"Plan generated ({len(dag.nodes)} nodes, complexity={complexity.value}):\n{plan_md}",
                role="planner",
                node_id="plan",
            )

        # ── EXECUTION LOOP ──────────────────────────────────────────────
        state.fsm_state = FSMState.EXECUTING
        self._checkpoint(state)

        while not dag.is_complete() and not dag.has_failed():
            ready = dag.ready_nodes()
            if not ready:
                state.fsm_state = FSMState.FAILED
                state.error_history.append("No ready nodes — possible DAG cycle")
                break

            for node in ready:
                if state.budget.remaining <= 0:
                    yield {"event": "budget_exceeded"}
                    state.fsm_state = FSMState.FAILED
                    break

                self._plan_state.update_node(node.id, status=NodeStatus.IN_PROGRESS)
                yield {"event": "node_start", "node_id": node.id, "title": node.title}

                # ── Cognitive routing ──────────────────────────────────
                strategy = self._cognitive_router.select(node, state)
                yield {
                    "event": "cognitive_strategy",
                    "node_id": node.id,
                    "strategy": strategy.value,
                }

                try:
                    result_summary, files_modified = await self._execute_node(
                        node, state, strategy
                    )
                    self._plan_state.mark_node_complete(node.id, result_summary, files_modified)
                    if self._scratchpad:
                        self._scratchpad.append(
                            f"Node {node.id} complete: {node.title}\n{result_summary}"
                            + (f"\nFiles: {', '.join(files_modified)}" if files_modified else ""),
                            role="executor",
                            node_id=node.id,
                        )
                    yield {
                        "event": "node_complete",
                        "node_id": node.id,
                        "summary": result_summary,
                        "files": files_modified,
                        "tools": state.last_node_tools,
                        "verify": state.last_node_validations,
                    }
                except Exception as e:
                    node.retry_count += 1
                    if node.retry_count >= self._cfg.retry_limit:
                        self._plan_state.mark_node_failed(node.id, str(e))
                        # Safety valve: if TRIVIAL node exhausted retries, promote to MODERATE
                        if self._current_complexity == TaskComplexity.TRIVIAL and not self._trivial_promoted:
                            self._trivial_promoted = True
                            yield {"event": "tier_promoted", "from": "trivial", "to": "moderate"}
                            logger.info("fsm.tier_promoted", reason="trivial_node_failed", node_id=node.id)
                            try:
                                new_dag = await self._planner.plan_lite(goal)
                                self._plan_state.set_dag(new_dag)
                                dag = new_dag  # Update local reference
                                self._current_complexity = TaskComplexity.MODERATE
                                # Reset this node's status for retry
                                self._plan_state.update_node(node.id, status=NodeStatus.PENDING)
                                node.retry_count = 0  # Reset retry counter for new plan
                                yield {
                                    "event": "node_retry",
                                    "node_id": node.id,
                                    "attempt": node.retry_count,
                                }
                                continue  # Restart execution loop with new plan
                            except Exception as plan_error:
                                logger.error("fsm.tier_promotion_failed", error=str(plan_error))
                                state.error_history.append(f"Tier promotion failed: {plan_error}")
                                # Fall through to original failure handling
                        state.error_history.append(f"Node {node.id} failed: {e}")
                        yield {"event": "node_failed", "node_id": node.id, "error": str(e)}
                    else:
                        self._plan_state.update_node(node.id, status=NodeStatus.PENDING)
                        yield {
                            "event": "node_retry",
                            "node_id": node.id,
                            "attempt": node.retry_count,
                        }

            state.iteration_count += 1
            if state.iteration_count > self._cfg.max_iterations * len(dag.nodes):
                state.fsm_state = FSMState.FAILED
                state.error_history.append("Exceeded global iteration limit")
                break

        # ── FINAL STATUS & VALIDATION ────────────────────────────────────
        duration_s = (time.perf_counter() - t0)
        if dag.is_complete():
            state.fsm_state = FSMState.VALIDATING
            self._checkpoint(state)
            yield {"event": "validating", "message": "Checking final system state against original intent..."}

            try:
                from src.validators.agentic import AgenticValidator
                ctx = await self._context_assembler.assemble(
                    role=AgentRole.CRITIC,
                    plan_node=PlanNode(id="final", title="Final Validation", description=goal)
                )

                # Verify the final state explicitly against extracted explicit intent constraints
                changes_str = "Completed all task DAG nodes."
                if state.intent_analysis:
                    constraints = ", ".join(state.intent_analysis.explicit_constraints)
                    changes_str = f"{changes_str} Verify these constraints were met: {constraints}. Primary Goal: {state.intent_analysis.primary_intent}"

                validator = AgenticValidator(n_critics=1)
                verdict = await validator.validate_with_verdict(context_pack=ctx, changes_summary=changes_str)
                if verdict.outcome == ValidationOutcome.FAIL:
                    state.error_history.append(f"Intent drift detected: {verdict.correction_hint}")
                    state.fsm_state = FSMState.FAILED
                    self._checkpoint(state)
                    yield {"event": "failed", "reason": f"Final intent check failed: {verdict.correction_hint}", "errors": state.error_history}
                    return
            except Exception as e:
                logger.warning("fsm.final_validation_error", error=str(e))
                state.error_history.append(f"Validation error: {e}")
                state.fsm_state = FSMState.FAILED
                self._checkpoint(state)
                yield {"event": "failed", "reason": f"Validation step failed: {e}", "errors": state.error_history}
                return

            state.fsm_state = FSMState.COMPLETE
            self._checkpoint(state)
            yield {
                "event": "complete",
                "duration_s": round(duration_s, 1),
                "tokens_used": state.budget.used,
                "nodes_completed": sum(1 for n in dag.nodes if n.status == NodeStatus.COMPLETE),
                "debates_run": len(state.debate_history),
            }
        else:
            state.fsm_state = FSMState.FAILED
            self._checkpoint(state)
            yield {
                "event": "failed",
                "duration_s": round(duration_s, 1),
                "errors": state.error_history,
            }

    async def _execute_node(
        self,
        node: PlanNode,
        state: AgentState,
        strategy: CognitiveStrategy,
    ) -> tuple[str, list[str]]:
        """
        Execute a single plan node using the CognitiveStrategy selected by the router.

        Strategies:
          DEBATE    → adversarial DebateGraph (Coder↔Critic game)
          ESCALATE  → direct frontier model call
          VERIFY    → single-shot agent + full deterministic validation
          DIRECT    → single-shot TAOR (original behaviour)
          REFINE    → single-shot + agentic critic (single improvement pass)
        """
        role = node.assigned_role or AgentRole.CODER

        # Assemble context (shared across all strategies)
        context = await self._context_assembler.assemble(role=role, plan_node=node)

        # Inject injected code snippets from user intent reasoning directly into context pack
        if state.intent_analysis and state.intent_analysis.extracted_code_snippets:
            from src.core.schemas import CodeSnippet
            for i, snippet in enumerate(state.intent_analysis.extracted_code_snippets):
                context.code_snippets.append(CodeSnippet(
                    file_path=f"<user_pasted_snippet_{i+1}>",
                    start_line=1,
                    end_line=len(snippet.splitlines()),
                    content=snippet,
                    summary="Raw code snippet provided by the user in the initial prompt."
                ))

        if strategy == CognitiveStrategy.DECOMPOSE:
            return await self._execute_decompose(node, state, context)
        elif strategy == CognitiveStrategy.DEBATE:
            return await self._execute_debate(node, state, context)
        elif strategy == CognitiveStrategy.ESCALATE:
            return await self._execute_escalated(node, state, context)
        elif strategy == CognitiveStrategy.REFINE:
            return await self._execute_refine(node, state, context)
        else:
            # DIRECT, VERIFY — single-shot
            return await self._execute_single_shot(node, state, context, run_tests=(strategy == CognitiveStrategy.VERIFY))

    async def _execute_decompose(
        self,
        node: PlanNode,
        state: AgentState,
        context: ContextPack,
    ) -> tuple[str, list[str]]:
        """
        Break a broad node into 2–5 atomic child nodes and inject them into the live DAG.

        The parent node is immediately marked COMPLETE (its "work" is the decomposition).
        Child nodes are inserted into the DAG with correct dependency wiring, so the main
        execution loop picks them up on the next ready_nodes() call without any special handling.
        """
        from src.orchestrator.node_decomposer import NodeDecomposer

        decomposer = NodeDecomposer()
        repo_summary = getattr(context, "repo_summary", None) or ""
        children = await decomposer.decompose(node, context_summary=repo_summary)

        # Inject children into the live DAG (rewires downstream deps automatically)
        assert state.dag is not None
        state.dag.decompose_node(node.id, children)

        # Persist the updated DAG so the plan state store knows about the new nodes
        self._plan_state.set_dag(state.dag)

        logger.info(
            "fsm.node_decomposed",
            parent_id=node.id,
            children=[c.id for c in children],
        )

        child_titles = ", ".join(c.title for c in children)
        return f"Decomposed into {len(children)} subtasks: {child_titles}", []

    async def _execute_debate(
        self,
        node: PlanNode,
        state: AgentState,
        context: ContextPack,
    ) -> tuple[str, list[str]]:
        """Route through the adversarial DebateGraph."""
        from src.orchestrator.debate_graph import DebateGraph
        # from src.fine_tuning.debate_collector import DebateTraceCollector

        state.fsm_state = FSMState.DEBATING
        graph = DebateGraph()

        debate, result_summary = await graph.run(node, context)

        state.budget.used += debate.tokens_used
        state.last_node_tools = debate.tool_trace
        state.active_debate = debate
        state.debate_history.append(debate)
        state.active_debate = None

        # # Async trace collection for IMAGINE (fire-and-forget, not on hot path)
        # try:
        #     collector = DebateTraceCollector()
        #     collector.record(debate)
        # except Exception as exc:
        #     logger.warning("fsm.debate_trace_error", error=str(exc))

        # Post-debate deterministic validation (composite)
        # Scope pytest to modified files only to avoid running entire test suite
        test_path = debate.files_modified[0] if debate.files_modified else "."
        validation_results = self._verify(path=test_path, run_tests=True)
        state.validation_results.extend(validation_results)
        state.last_node_validations = [
            {"validator": r.validator_name, "outcome": r.outcome.value, "message": r.message}
            for r in validation_results
        ]
        failed = [v for v in validation_results if v.outcome == ValidationOutcome.FAIL]

        if debate.outcome == DebateOutcome.ESCALATED and not result_summary:
            raise RuntimeError("Debate escalated but no result produced")

        if failed:
            hints = "; ".join(v.correction_hint or v.message for v in failed)
            raise RuntimeError(f"Post-debate validation failed: {hints}")

        state.fsm_state = FSMState.EXECUTING
        return result_summary, debate.files_modified

    async def _execute_single_shot(
        self,
        node: PlanNode,
        state: AgentState,
        context: ContextPack,
        run_tests: bool = False,
    ) -> tuple[str, list[str]]:
        """Original single-shot TAOR execution."""
        role = node.assigned_role or AgentRole.CODER
        agent = self._get_agent(role)

        result = await agent.run(context=context, user_message=node.description)
        state.budget.used += result.tokens_used
        state.last_node_tools = result.tool_trace

        if not result.success:
            raise RuntimeError(result.final_answer)

        validation_results = self._verify(path=".", run_tests=run_tests)
        state.validation_results.extend(validation_results)
        state.last_node_validations = [
            {"validator": r.validator_name, "outcome": r.outcome.value, "message": r.message}
            for r in validation_results
        ]
        failed = [v for v in validation_results if v.outcome == ValidationOutcome.FAIL]
        if failed:
            hints = "; ".join(v.correction_hint or v.message for v in failed)
            raise RuntimeError(f"Validation failed: {hints}")

        return result.final_answer, []

    def _verify(self, path: str, run_tests: bool) -> list[ValidationResult]:
        """Run composite deterministic verification and normalize into ValidationResult objects."""
        payload = verify_repo(path=path, run_tests=run_tests)
        results: list[ValidationResult] = []
        for item in payload.get("results", []):
            outcome_raw = str(item.get("outcome", "uncertain"))
            try:
                outcome = ValidationOutcome(outcome_raw)
            except ValueError:
                outcome = ValidationOutcome.UNCERTAIN
            results.append(
                ValidationResult(
                    validator_name=str(item.get("validator", "")),
                    outcome=outcome,
                    message=str(item.get("message", ""))[:500],
                    correction_hint=item.get("hint"),
                    duration_ms=float(item.get("duration_ms", 0.0) or 0.0),
                )
            )
        return results

    async def _execute_refine(
        self,
        node: PlanNode,
        state: AgentState,
        context: ContextPack,
    ) -> tuple[str, list[str]]:
        """Single-shot generation followed by one critic improvement pass."""
        from src.validators.agentic import AgenticValidator

        # First pass
        result_summary, files = await self._execute_single_shot(node, state, context)

        # Single critic review
        validator = AgenticValidator(n_critics=1)
        verdict = await validator.validate_with_verdict(
            context_pack=context,
            changes_summary=result_summary,
        )

        if verdict.outcome == ValidationOutcome.FAIL and verdict.correction_hint:
            # Re-run with hint injected — one refinement pass only
            agent = self._get_agent(node.assigned_role or AgentRole.CODER)
            hint_msg = f"Please revise your implementation.\n\nCritic feedback:\n{verdict.correction_hint}"
            result2 = await agent.run(context=context, user_message=hint_msg)
            state.budget.used += result2.tokens_used
            if result2.success:
                result_summary = result2.final_answer

        return result_summary, files

    async def _execute_escalated(
        self,
        node: PlanNode,
        state: AgentState,
        context: ContextPack,
    ) -> tuple[str, list[str]]:
        """Direct escalation to frontier model (ESCALATE strategy)."""
        from src.orchestrator.escalation import EscalationManager

        state.fsm_state = FSMState.ESCALATING
        manager = EscalationManager()
        state.escalation_count += 1

        result = await manager.run_escalated_task(context, node.description)
        state.fsm_state = FSMState.EXECUTING
        return result, []

    @staticmethod
    def _get_agent(role: AgentRole):
        """Lazy-import and return the correct agent instance."""
        from src.orchestrator.router import get_specialist
        return get_specialist(role)

    def _checkpoint(self, state: AgentState) -> None:
        """Persist FSM state to disk."""
        p = self._data_dir / f"state_{state.session_id}.json"
        try:
            p.write_text(state.model_dump_json(indent=2))
        except Exception as e:
            logger.warning("fsm.checkpoint_failed", error=str(e))

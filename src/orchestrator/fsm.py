"""
Deterministic FSM (Finite State Machine) controlling the orchestration loop.
States: PLANNING → EXECUTING → (DEBATING) → VALIDATING → (ESCALATING) → COMPLETE | FAILED

Phase 4 upgrade: debate-aware _execute_node routes code-producing nodes through
the adversarial DebateGraph instead of single-shot TAOR execution. Strategy
selection is delegated to CognitiveRouter (RLoT-style).
"""
from __future__ import annotations

import asyncio
import re
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
from src.memory.recursive_summarizer import RecursiveSummarizer, RecursiveSummarizerConfig
from src.memory.scratchpad_janitor import JanitorConfig, ScratchpadJanitor
from src.memory.plan_state import PlanState
from src.memory.scratchpad import SessionScratchpad, set_active_scratchpad
from src.orchestrator.cognitive_router import CognitiveRouter
from src.orchestrator.planner import Planner
from src.orchestrator.task_router import TaskComplexity, TaskRouter
from src.repo_intel.pasted_code_stubber import stub_pasted_code
from src.repo_intel.stub_extractor import extract_stubs
from src.repo_intel.symbol_slicer import SliceConfig, slice_symbols
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
        self._mem_cfg = cfg.memory
        self._data_dir = Path(data_dir or cfg.data_dir)
        self._planner = Planner()
        self._validator = DeterministicValidator()
        self._episodic = EpisodicStore()
        self._plan_state = PlanState(data_dir=self._data_dir)
        self._cognitive_router = CognitiveRouter()
        from src.serving.model_registry import get_client
        self._repo_root = Path(__file__).resolve().parents[2]
        self._task_router = TaskRouter(client=get_client(AgentRole.ORCHESTRATOR), repo_root=self._repo_root)

        # Tracker (recursive summarizer) + Janitor (scratchpad compaction)
        from src.agents.summarizer import SummarizerAgent
        summarizer_agent = SummarizerAgent()
        tracker_cfg = RecursiveSummarizerConfig(
            trigger_chars=self._mem_cfg.tracker_trigger_chars,
            chunk_chars=self._mem_cfg.tracker_chunk_chars,
            target_chars=self._mem_cfg.tracker_target_chars,
        )
        self._tracker = RecursiveSummarizer(summarize_fn=summarizer_agent.summarize, cfg=tracker_cfg)
        janitor_cfg = JanitorConfig(
            max_chars_before_compact=self._mem_cfg.scratchpad_max_chars_before_compact,
            keep_recent_chars=self._mem_cfg.scratchpad_keep_recent_chars,
            summary_target_chars=self._mem_cfg.scratchpad_summary_target_chars,
        )
        self._janitor = ScratchpadJanitor(self._tracker, cfg=janitor_cfg)
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

        planning_goal = classification.nl_intent or goal
        extracted_code = classification.code_snippets  # Preserve extracted code for intent reasoning
        repo_summary = classification.prefetched_stub_map
        state.mentioned_files = list(classification.mentioned_files or [])
        state.prefetched_repo_summary = repo_summary or None
        dag: TaskDAG

        if classification.mentioned_files:
            yield {
                "event": "prefetch",
                "mentioned_files": classification.mentioned_files,
            }
            if self._scratchpad:
                self._scratchpad.append(
                    "Prefetched stubs for mentioned files:\n" + "\n".join(classification.mentioned_files),
                    role="prefetcher",
                    node_id="prefetch",
                )

        if complexity == TaskComplexity.TRIVIAL:
            dag = self._task_router.build_trivial_dag(goal)

        elif complexity == TaskComplexity.MODERATE:
            state.fsm_state = FSMState.PLANNING
            self._checkpoint(state)

            # Lite intent rewrite — 1 model call, no ToT selection
            planning_goal = classification.nl_intent or goal
            yield {"event": "reasoning", "message": "Clarifying intent (lite)..."}
            try:
                from src.orchestrator.intent_reasoner import IntentReasoner
                reasoner = IntentReasoner()

                tracker_input = (goal or "") + ("\n\n" + "\n\n".join(extracted_code) if extracted_code else "")
                if self._tracker.should_summarize(tracker_input):
                    summarized = await self._tracker.summarize(
                        tracker_input,
                        hint=(
                            "Summarize the user's goal and any pasted code into a compact description. "
                            "Do not include verbatim code; preserve file paths and symbol names." 
                        ),
                    )
                    intent = await reasoner.rewrite_lite(raw_query=summarized, code_snippets=[])
                else:
                    intent = await reasoner.rewrite_lite(raw_query=planning_goal, code_snippets=extracted_code)
                state.intent_analysis = intent
                self._checkpoint(state)
                planning_goal = self._task_router._strip_code_blocks(intent.rewritten_query)
            except Exception as e:
                yield {"event": "warn", "message": f"Lite intent rewrite failed: {e}. Using raw goal."}

            yield {"event": "planning", "message": "Generating task plan (lite)..."}
            try:
                dag = await self._planner.plan_lite(goal=planning_goal, repo_summary=repo_summary)
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

                tracker_input = (goal or "") + ("\n\n" + "\n\n".join(extracted_code) if extracted_code else "")
                if self._tracker.should_summarize(tracker_input):
                    summarized = await self._tracker.summarize(
                        tracker_input,
                        hint=(
                            "Summarize the user's goal and any pasted code into a compact description. "
                            "Do not include verbatim code; preserve file paths and symbol names." 
                        ),
                    )
                    intent = await reasoner.analyze_and_rewrite(raw_query=summarized, code_snippets=[])
                else:
                    intent = await reasoner.analyze_and_rewrite(raw_query=planning_goal, code_snippets=extracted_code)
                state.intent_analysis = intent
                self._checkpoint(state)
                planning_goal = self._task_router._strip_code_blocks(intent.rewritten_query)
            except Exception as e:
                yield {"event": "warn", "message": f"Intent reasoning failed: {e}. Falling back to raw goal."}

            # ── PLANNING ──────────────────────────────────────────────────
            state.fsm_state = FSMState.PLANNING
            self._checkpoint(state)
            yield {"event": "planning", "message": "Generating task plan..."}
            try:
                dag = await self._planner.plan(goal=planning_goal, repo_summary=repo_summary)
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
                                new_dag = await self._planner.plan_lite(goal, repo_summary=repo_summary)
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

        # Janitor: compact scratchpad if it has grown too large
        if self._scratchpad:
            try:
                await self._janitor.maybe_compact(self._scratchpad)
            except Exception as e:  # pragma: no cover
                logger.debug("fsm.janitor_failed", error=str(e))

        # Markovian working summary: direct deps + optional 1 more hop
        working_summary = self._build_markov_summary(node, state, max_chars=2000, max_hops=2)

        from src.core.schemas import CodeSnippet

        code_snippets: list[CodeSnippet] = []
        code_snippets.extend(self._slice_repo_context_for_node(node, state))

        # Replace raw pasted code injection with deterministic stubs (no verbatim bodies)
        if state.intent_analysis and state.intent_analysis.extracted_code_snippets:
            for i, raw in enumerate(state.intent_analysis.extracted_code_snippets, start=1):
                stub = stub_pasted_code(raw)
                if not stub:
                    continue
                code_snippets.append(
                    CodeSnippet(
                        file_path=f"<user_pasted_stub_{i}>",
                        start_line=1,
                        end_line=stub.count("\n") + 1,
                        content=stub + "\n",
                        language="text",
                        summary="Deterministic stub of user-pasted code (signatures/imports only).",
                    )
                )

        # Assemble context (shared across all strategies)
        context = await self._context_assembler.assemble(
            role=role,
            plan_node=node,
            working_summary=working_summary or None,
            code_snippets=code_snippets or None,
        )

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

    def _slice_repo_context_for_node(self, node: PlanNode, state: AgentState) -> list["CodeSnippet"]:
        """Return small, symbol-level snippets relevant to this node.

        Uses only repo-local paths (captured during routing) and slices by symbol
        names referenced in the node title/description.
        """
        node_text = (node.title + "\n" + node.description).strip()
        if not node_text:
            return []

        mentioned: list[str] = [p for p in (state.mentioned_files or []) if p]
        mentioned.extend(p for p in self._extract_file_mentions_from_text(node_text) if p not in mentioned)
        if not mentioned:
            return []

        max_files = 4
        max_symbols_per_file = 3
        slice_cfg = SliceConfig(max_snippets_total=8, max_snippet_chars=6000, max_total_chars=18_000)

        results: list[CodeSnippet] = []
        for rel in mentioned[:max_files]:
            path = self._resolve_repo_path(rel)
            if path is None:
                continue
            try:
                path.relative_to(self._repo_root.resolve())
            except Exception:
                continue
            if not path.exists():
                continue

            # Directory mention: expand into bounded file list
            paths: list[Path]
            if path.is_dir():
                paths = self._expand_repo_dir(path, max_files=max_files)
            else:
                if not path.is_file():
                    continue
                paths = [path]

            for file_path in paths:
                if len(results) >= slice_cfg.max_snippets_total:
                    return results
                stubs = extract_stubs(file_path, max_symbols=200)
                matched: list[str] = []
                for s in stubs:
                    name = getattr(s, "name", "")
                    if not name:
                        continue
                    if re.search(rf"\b{re.escape(name)}\b", node_text):
                        matched.append(name)
                    if len(matched) >= max_symbols_per_file:
                        break
                if not matched:
                    continue

                for snip in slice_symbols(file_path, matched, cfg=slice_cfg):
                    results.append(snip)
                    if len(results) >= slice_cfg.max_snippets_total:
                        return results

        return results

    def _extract_file_mentions_from_text(self, text: str) -> list[str]:
        exts = r"py|js|jsx|ts|tsx|go|rs|java"
        pat = re.compile(rf"(?P<p>(?:[A-Za-z]:)?[\\/][^\s`\"']+?\.(?:{exts})|[^\s`\"']+?\.(?:{exts}))")
        found: list[str] = []
        for m in pat.finditer(text):
            p = m.group("p").strip().strip("'\"`")
            p = p.rstrip(").,;:")
            if p and p not in found and ".." not in p:
                found.append(p.replace("\\", "/"))
        # Directory mentions (common roots)
        dir_pat = re.compile(r"(?P<p>(?:src|tests|tools|infra|fine_tuning)(?:[\\/][^\s`\"']+)+[\\/]?)")
        for m in dir_pat.finditer(text):
            p = m.group("p").strip().strip("'\"`")
            p = p.rstrip(").,;:")
            if "." in Path(p).name:
                continue
            if p and p not in found and ".." not in p:
                found.append(p.replace("\\", "/"))
        # Module mentions like src.orchestrator.fsm
        mod_pat = re.compile(r"\bsrc\.(?:[A-Za-z_][A-Za-z0-9_]*\.?)+\b")
        for m in mod_pat.finditer(text):
            mod = m.group(0)
            if mod:
                p = mod.replace(".", "/") + ".py"
                if p not in found:
                    found.append(p)
        return found

    def _resolve_repo_path(self, mention: str) -> Path | None:
        m = (mention or "").strip().replace("\\", "/")
        if not m or ".." in m or m.startswith("~"):
            return None
        p = Path(m)
        if p.is_absolute() or (len(m) > 2 and m[1] == ":"):
            try:
                rp = p.resolve()
                rp.relative_to(self._repo_root.resolve())
                return rp
            except Exception:
                return None

        candidate = (self._repo_root / p).resolve()
        try:
            candidate.relative_to(self._repo_root.resolve())
            return candidate
        except Exception:
            return None

    def _expand_repo_dir(self, directory: Path, *, max_files: int) -> list[Path]:
        if max_files <= 0:
            return []
        exts = {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java"}
        files: list[Path] = []
        try:
            for f in directory.rglob("*"):
                if len(files) >= max_files:
                    break
                if f.is_file() and f.suffix.lower() in exts:
                    files.append(f)
        except OSError:
            return []
        return sorted(files, key=lambda p: str(p).lower())[:max_files]

    def _build_markov_summary(
        self,
        node: PlanNode,
        state: AgentState,
        *,
        max_chars: int,
        max_hops: int,
    ) -> str:
        dag = state.dag
        if dag is None or not node.depends_on:
            return ""

        visited: set[str] = set()
        lines: list[str] = ["Predecessor node summaries (Markov context):"]

        def add(nid: str, hops_left: int) -> None:
            if nid in visited:
                return
            visited.add(nid)
            dep = dag.get_node(nid)
            if dep is None:
                return
            if dep.result_summary:
                lines.append(f"- {dep.id} {dep.title}: {dep.result_summary}")
                if getattr(dep, "files_modified", None):
                    fm = [f for f in (dep.files_modified or []) if f]
                    if fm:
                        lines.append(f"  files: {', '.join(fm[:12])}")

            if hops_left <= 0:
                return
            for up in dep.depends_on or []:
                add(up, hops_left - 1)

        for d in node.depends_on:
            add(d, max_hops - 1)
            if sum(len(x) + 1 for x in lines) >= max_chars:
                break

        text = "\n".join(lines).strip()
        return text[:max_chars]

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

"""
Pydantic schemas — the typed contracts between every component.
Import these everywhere instead of passing raw dicts.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────


class AgentRole(str, Enum):
    ORCHESTRATOR = "orchestrator"
    CODER = "coder"
    EXPLORER = "explorer"
    TESTER = "tester"
    REFACTORER = "refactorer"
    CRITIC = "critic"
    SUMMARIZER = "summarizer"
    DOC_READER = "doc_reader"


class NodeStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


class FSMState(str, Enum):
    PLANNING = "planning"
    EXECUTING = "executing"
    DEBATING = "debating"
    VALIDATING = "validating"
    ESCALATING = "escalating"
    COMPLETE = "complete"
    FAILED = "failed"


class ValidationOutcome(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNCERTAIN = "uncertain"


class DebateOutcome(str, Enum):
    """Resolution state of an adversarial Coder↔Critic debate game."""
    IN_PROGRESS = "in_progress"
    CODER_WINS = "coder_wins"     # Critic score >= acceptance_threshold
    DEADLOCK = "deadlock"          # max_turns reached without concession
    ESCALATED = "escalated"        # Handed off to frontier model


class CognitiveStrategy(str, Enum):
    """RLoT-style cognitive logic blocks for dynamic routing.

    The CognitiveRouter (or a RL navigator) selects one of these per
    PlanNode to decide how execution should proceed.
    """
    DECOMPOSE = "decompose"   # Break into sub-tasks
    DEBATE = "debate"          # Force adversarial Coder↔Critic loop
    REFINE = "refine"          # Iterative self-improvement
    VERIFY = "verify"          # Run validators only
    ESCALATE = "escalate"      # Hand off to frontier model immediately
    DIRECT = "direct"          # Single-shot — no debate (default)


class MemoryType(str, Enum):
    WORKING = "working"
    PLAN = "plan"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


# ─────────────────────────────────────────────
# Tool Call Schemas (MCP / ReAct)
# ─────────────────────────────────────────────


class ToolCall(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool_call_id: UUID
    name: str
    content: str
    is_error: bool = False
    duration_ms: float = 0.0


# ─────────────────────────────────────────────
# Agent Message / Conversation
# ─────────────────────────────────────────────


class AgentMessage(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    name: str | None = None            # Agent role name
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    token_count: int | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────
# Intent Reasoning
# ─────────────────────────────────────────────


class IntentAnalysis(BaseModel):
    """Structured output from analyzing the raw user goal before planning."""
    primary_intent: str
    explicit_constraints: list[str] = Field(default_factory=list)
    implicit_assumptions: list[str] = Field(default_factory=list)
    extracted_code_snippets: list[str] = Field(default_factory=list)
    rewritten_query: str


# ─────────────────────────────────────────────
# Task DAG — Plan State
# ─────────────────────────────────────────────


class PlanNode(BaseModel):
    id: str                            # Short stable ID e.g. "N1", "N2"
    title: str
    description: str
    assigned_role: AgentRole | None = None
    depends_on: list[str] = Field(default_factory=list)   # Node IDs
    success_criteria: list[str] = Field(default_factory=list)
    cognitive_strategy: CognitiveStrategy = CognitiveStrategy.DIRECT
    status: NodeStatus = NodeStatus.PENDING
    result_summary: str | None = None
    files_modified: list[str] = Field(default_factory=list)
    retry_count: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    parent_node_id: str | None = None  # Set when created by NodeDecomposer


class TaskDAG(BaseModel):
    task_id: UUID = Field(default_factory=uuid4)
    title: str
    description: str
    nodes: list[PlanNode]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def get_node(self, node_id: str) -> PlanNode | None:
        return next((n for n in self.nodes if n.id == node_id), None)

    def ready_nodes(self) -> list[PlanNode]:
        """Return nodes whose dependencies are all complete."""
        complete_ids = {n.id for n in self.nodes if n.status == NodeStatus.COMPLETE}
        return [
            n for n in self.nodes
            if n.status == NodeStatus.PENDING
            and all(dep in complete_ids for dep in n.depends_on)
        ]

    def decompose_node(self, parent_id: str, children: list["PlanNode"]) -> None:
        """
        Inject child nodes into the DAG in place of a broad parent node.

        - Children inherit the parent's upstream depends_on
        - Each subsequent child depends on the previous one (sequential by default)
        - Every downstream node that depended on parent_id is rewired to depend on
          the last child instead, ensuring correct execution order
        - The parent node is left in-place; the caller is responsible for marking
          it COMPLETE immediately after calling this method
        """
        parent = next(n for n in self.nodes if n.id == parent_id)
        last_child_id = children[-1].id

        for i, child in enumerate(children):
            child.parent_node_id = parent_id
            if i == 0:
                # First child inherits the parent's upstream deps
                child.depends_on = list(parent.depends_on)
            else:
                # Each subsequent child depends on the previous child
                child.depends_on = [children[i - 1].id]

        # Rewire downstream nodes: swap parent_id → last child id
        for node in self.nodes:
            if parent_id in node.depends_on:
                node.depends_on = [
                    last_child_id if d == parent_id else d
                    for d in node.depends_on
                ]

        # Insert children immediately after the parent in the node list
        idx = next(i for i, n in enumerate(self.nodes) if n.id == parent_id)
        self.nodes[idx + 1:idx + 1] = children

    def is_complete(self) -> bool:
        return all(n.status in (NodeStatus.COMPLETE, NodeStatus.SKIPPED) for n in self.nodes)

    def has_failed(self) -> bool:
        return any(n.status == NodeStatus.FAILED for n in self.nodes)


# ─────────────────────────────────────────────
# Context Pack — assembled per agent call
# ─────────────────────────────────────────────


class CodeSnippet(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    content: str
    language: str = "python"
    summary: str | None = None


class ContextPack(BaseModel):
    """The precise context assembled for a single specialist agent call."""
    agent_role: AgentRole
    plan_node: PlanNode
    code_snippets: list[CodeSnippet] = Field(default_factory=list)
    episodic_summaries: list[str] = Field(default_factory=list)
    semantic_rules: list[str] = Field(default_factory=list)
    working_summary: str | None = None      # Compressed working memory
    scratchpad_tail: str | None = None      # Last ~4K chars of session scratchpad
    contracts_context: str | None = None    # Compact symbol table from ContractStore
    total_tokens: int = 0


# ─────────────────────────────────────────────
# Validation / Critic
# ─────────────────────────────────────────────


class ValidationResult(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    validator_name: str              # "ruff", "pyright", "pytest", "critic_ensemble"
    outcome: ValidationOutcome
    message: str
    details: str | None = None
    correction_hint: str | None = None  # Structured hint fed back to orchestrator
    duration_ms: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────
# Memory Schemas
# ─────────────────────────────────────────────


class EpisodicEntry(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    task_id: UUID
    task_description: str
    agent_role: AgentRole
    action_summary: str
    files_touched: list[str] = Field(default_factory=list)
    outcome: ValidationOutcome
    model_name: str
    lora_version: str | None = None
    duration_ms: float = 0.0
    token_count: int = 0
    embedding: list[float] | None = None   # Set by episodic store on insert
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SemanticRule(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    rule_text: str
    tags: list[str] = Field(default_factory=list)    # e.g. ["auth", "middleware", "testing"]
    source_task_ids: list[UUID] = Field(default_factory=list)
    human_approved: bool = False
    confidence: float = 1.0
    use_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used_at: datetime | None = None


# ─────────────────────────────────────────────
# Global Agent State (FSM)
# ─────────────────────────────────────────────


class TokenBudget(BaseModel):
    total_limit: int
    used: int = 0
    escalation_used: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.total_limit - self.used)

    @property
    def pct_used(self) -> float:
        return self.used / self.total_limit if self.total_limit else 0.0


class AgentState(BaseModel):
    """Typed state object passed through every FSM node."""
    session_id: UUID = Field(default_factory=uuid4)
    fsm_state: FSMState = FSMState.PLANNING
    user_goal: str
    mentioned_files: list[str] = Field(default_factory=list)
    prefetched_repo_summary: str | None = None
    intent_analysis: IntentAnalysis | None = None
    dag: TaskDAG | None = None
    active_node_id: str | None = None
    error_history: list[str] = Field(default_factory=list)
    validation_results: list[ValidationResult] = Field(default_factory=list)
    budget: TokenBudget = Field(default_factory=lambda: TokenBudget(total_limit=200_000))
    escalation_count: int = 0
    iteration_count: int = 0
    active_debate: DebateState | None = None        # Set while a debate is running
    debate_history: list[DebateState] = Field(default_factory=list)  # Completed debates
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    # Transient: last node tool trace (not persisted/relied upon). Used for CLI transcript only.
    last_node_tools: list[dict[str, Any]] = Field(default_factory=list, exclude=True)
    # Transient: last node deterministic verification results (CLI transcript only).
    last_node_validations: list[dict[str, Any]] = Field(default_factory=list, exclude=True)


# ─────────────────────────────────────────────
# A2A Agent Card
# ─────────────────────────────────────────────


class AgentCapability(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)


class AgentCard(BaseModel):
    agent_id: str
    role: AgentRole
    display_name: str
    description: str
    capabilities: list[AgentCapability]
    endpoint: str | None = None   # For remote agents; None = in-process
    model_name: str
    max_context_tokens: int = 8192


# ─────────────────────────────────────────────
# Adversarial Debate Schemas
# ─────────────────────────────────────────────


class DebateMove(BaseModel):
    """A single move (argument) in the adversarial debate game."""
    turn: int
    actor: str                          # "coder" | "critic" | "judge"
    content: str                        # The argument / code / critique
    score: int | None = None            # Critic's score (0-10); None for coder moves
    failing_tests: list[str] = Field(default_factory=list)
    tokens_used: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DebateState(BaseModel):
    """Shared state for the adversarial Coder↔Critic debate game.

    This is the 'game board' — both players read it and write their moves.
    The conditional edge (referee) reads critic_score and turn_count to
    decide whether to loop back to Coder, declare CODER_WINS, or DEADLOCK.
    """
    id: UUID = Field(default_factory=uuid4)
    node_id: str                         # PlanNode being debated
    task_description: str                # What the Coder must implement
    success_criteria: list[str] = Field(default_factory=list)

    # Artifact under debate
    current_code: str = ""               # Coder's latest output / diff
    files_modified: list[str] = Field(default_factory=list)

    # Critic's latest judgment
    critic_score: int = 0                # 0-10 assigned by Critic each round
    critic_reasoning: str = ""           # Critic's argument for its score
    critic_failing_tests: list[str] = Field(default_factory=list)
    last_test_run_output: str = ""       # Actual pytest output for failing tests (for coder rebuttal)

    # Full adversarial transcript
    moves: list[DebateMove] = Field(default_factory=list)

    # Game control
    turn_count: int = 0
    max_turns: int = 5                   # Nash equilibrium ceiling
    max_token_budget: int = 50_000       # TTL cost bound
    compressed_history: str = ""         # "Lost in the middle" mitigation summary
    acceptance_threshold: int = 9        # score >= this → Coder wins (Critic concedes)
    outcome: DebateOutcome = DebateOutcome.IN_PROGRESS

    # Parse failure tracking — auto-accept if critic can't produce valid JSON repeatedly
    json_parse_failures: int = 0

    # Accounting
    tokens_used: int = 0
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)

    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None


class EnsembleVerdict(BaseModel):
    """Result from a judge ensemble with inter-judge agreement metrics.

    std_dev across individual_scores is the confidence signal: high std_dev
    means judges disagree → confidence drops → escalation fires.
    """
    outcome: ValidationOutcome
    individual_scores: list[int] = Field(default_factory=list)
    mean_score: float = 0.0
    std_dev: float = 0.0
    confidence: float = 1.0              # 1.0 − normalized_std_dev
    escalated: bool = False
    escalation_response: str | None = None
    message: str = ""
    correction_hint: str | None = None


# Forward-reference resolution
AgentState.model_rebuild()

"""
TaskRouter — dynamic complexity classifier for the FSM.

Classifies each incoming goal across 4 independent dimensions:
  1. Verb type (creation / modification / reasoning)
  2. Scope width (single item → system-wide)
  3. Requirement count (number of logical connectors)
  4. Structural complexity flags (conditionals, concurrency, etc.)

Clear cases are decided in <1 ms with no model calls.
Ambiguous cases (score 4–5) trigger a single minimal model call (max_tokens=5).
"""
from __future__ import annotations

import re
from enum import Enum

import structlog

from src.core.schemas import AgentRole, CognitiveStrategy, NodeStatus, PlanNode, TaskDAG

logger = structlog.get_logger(__name__)


class TaskComplexity(str, Enum):
    TRIVIAL  = "trivial"   # skip intent + planning → 1 model call
    MODERATE = "moderate"  # skip intent + ToT draft → 2 model calls
    COMPLEX  = "complex"   # full pipeline (existing behaviour)


# ── Dimension 1 — Verb scoring ──────────────────────────────────────────────
# Patterns tested in order; first match wins.
# Score 4 = clearly complex reasoning verb
# Score 1 = modification or contextual creation ("add X to Y")
# Score 0 = pure creation ("write X", "create X")
_VERB_SCORES: list[tuple[re.Pattern, int]] = [
    # Reasoning / architectural verbs → 4
    (re.compile(
        r"^(refactor|migrate|redesign|architect|optimise|optimize|"
        r"analyse|analyze|debug|integrate|implement|"
        r"secure|audit|benchmark|profile|decompose)\b",
        re.IGNORECASE,
    ), 4),
    # Modification verbs → 1
    (re.compile(
        r"^(update|fix|change|rename|move|remove|delete|edit|patch|bump|extend|"
        r"improve|enhance|convert|transform|add|make|build|scaffold|"
        r"init|initialise|initialize|stub)\b",
        re.IGNORECASE,
    ), 1),
    # Pure creation / output verbs → 0
    (re.compile(
        r"^(write|create|generate|print|output|produce|draft|emit)\b",
        re.IGNORECASE,
    ), 0),
    # Discovery / read verbs → 1
    (re.compile(
        r"^(explain|show|find|list|describe|summarise|summarize|review|check|"
        r"inspect|search|look)\b",
        re.IGNORECASE,
    ), 1),
]

# ── Dimension 2 — Scope width ────────────────────────────────────────────────
_SCOPE_SYSTEM = re.compile(
    r"\b(entire|throughout|system|codebase|architecture|everywhere|every file|"
    r"all files|globally|across the)\b",
    re.IGNORECASE,
)
_SCOPE_MODULE = re.compile(
    r"\b(class|module|package|component|service|layer|subsystem)\b",
    re.IGNORECASE,
)
_SCOPE_SINGLE = re.compile(
    r"\b(function|method|variable|constant|parameter|line|field|property)\b",
    re.IGNORECASE,
)

# ── Dimension 3 — Requirement connectors ─────────────────────────────────────
_CONNECTORS = re.compile(
    r"(;\s+| and also | also | then | after that | while |, and |\band\b.*\balso\b)",
    re.IGNORECASE,
)

# ── Dimension 4 — Structural complexity flags ────────────────────────────────
_COMPLEXITY_FLAGS = re.compile(
    r"\b(if |when |depending|based on|should also|might|consider|ensure|"
    r"without breaking|backward.?compat|thread|concurren|async|race.?condition|"
    r"lock|deadlock|atomic|transaction|rollback)\b",
    re.IGNORECASE,
)

# ── Dimension 1 — Modification verbs (for boost logic) ──────────────────────
_MODIFICATION_VERBS = re.compile(
    r"^(update|fix|change|rename|move|remove|delete|edit|patch|bump|extend|"
    r"improve|enhance|convert|transform|add|make|build|scaffold|"
    r"init|initialise|initialize|stub)\b",
    re.IGNORECASE,
)


class TaskRouter:
    """
    Multi-signal complexity classifier.  No model calls for clear cases.
    Pass a VLLMClient instance to enable the fast ambiguous-case fallback.
    """

    def __init__(self, client=None) -> None:
        self._client = client  # VLLMClient | None

    @staticmethod
    def _strip_code_blocks(text: str) -> str:
        """Remove markdown fenced code blocks before heuristic scoring.

        Prevents code snippets (with semicolons, if statements, etc.) from
        polluting Dimension 3 and 4 scores.
        """
        return re.sub(r'```[\s\S]*?```', '', text, flags=re.DOTALL).strip()

    # ── Public API ───────────────────────────────────────────────────────────

    async def classify(self, goal: str) -> TaskComplexity:
        """Classify a goal. LLM-first when a client is available; heuristic fallback otherwise."""
        # ── Primary path: ask the model (fast max_tokens=5 call) ────────────
        if self._client is not None:
            result = await self._model_classify(goal)
            if result is not None:
                logger.info("task_router.classified_by_model", complexity=result.value, goal=goal[:60])
                return result
            logger.warning("task_router.model_classify_failed_fallback", goal=goal[:60])

        # ── Fallback: heuristic scoring (model unavailable or errored) ───────
        score = self._score(goal)
        logger.debug("task_router.classified_by_heuristic", score=score, goal=goal[:60])
        if score <= 1:
            return TaskComplexity.TRIVIAL
        if score >= 5:
            return TaskComplexity.COMPLEX
        return TaskComplexity.MODERATE

    def build_trivial_dag(self, goal: str) -> TaskDAG:
        """Build a 1-node DAG for trivial tasks — skips the Planner entirely."""
        node = PlanNode(
            id="N1",
            title="Execute Task",
            description=goal,
            assigned_role=AgentRole.CODER,
            depends_on=[],
            success_criteria=["Task completed as specified by the user"],
            cognitive_strategy=CognitiveStrategy.DIRECT,
            status=NodeStatus.PENDING,
        )
        return TaskDAG(title="Direct Execution", description=goal, nodes=[node])

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _score(self, goal: str) -> int:
        lower = goal.lower().strip()
        score = 0

        # Dimension 1: verb type
        verb_score = 1  # default for unknown verbs
        is_modification = False
        for pattern, s in _VERB_SCORES:
            if pattern.match(lower):
                verb_score = s
                is_modification = _MODIFICATION_VERBS.match(lower) is not None
                break
        score += verb_score

        # Dimension 2: scope width
        if _SCOPE_SYSTEM.search(lower):
            score += 3
        elif _SCOPE_MODULE.search(lower):
            score += 2
        elif _SCOPE_SINGLE.search(lower):
            score += 0
        else:
            score += 1  # file-level default

        # Dimension 3: requirement count (score on NL only, stripping code)
        nl_only = self._strip_code_blocks(goal)
        n_connectors = len(_CONNECTORS.findall(nl_only))
        score += 0 if n_connectors <= 1 else (1 if n_connectors <= 3 else 3)

        # Dimension 4: structural complexity flags (score on NL only, stripping code)
        nl_lower = nl_only.lower()
        n_flags = len(_COMPLEXITY_FLAGS.findall(nl_lower))
        d4_score = 0 if n_flags == 0 else (1 if n_flags == 1 else 3)
        score += d4_score

        # Boost: if verb is "fix"/"update"/"patch" AND structural flags present
        if is_modification and d4_score >= 1:
            score += 2

        return score

    async def _model_classify(self, goal: str) -> TaskComplexity | None:
        """Single max_tokens=5 call — returns None on any failure."""
        try:
            from src.core.schemas import AgentMessage
            resp = await self._client.complete(
                messages=[
                    AgentMessage(
                        role="system",
                        content=(
                            "Classify this coding task in one word: trivial, moderate, or complex. "
                            "No explanation — output exactly one word."
                        ),
                    ),
                    AgentMessage(role="user", content=goal),
                ],
                max_tokens=5,
                temperature=0.0,
            )
            word = resp.content.strip().lower().split()[0]
            logger.debug("task_router.model_classified", word=word)
            if word == "trivial":
                return TaskComplexity.TRIVIAL
            if word == "complex":
                return TaskComplexity.COMPLEX
            if word == "moderate":
                return TaskComplexity.MODERATE
        except Exception as e:
            logger.warning("task_router.model_classify_failed", error=str(e))
        return None

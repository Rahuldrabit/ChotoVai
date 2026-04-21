"""
TaskRouter — dynamic complexity classifier for the FSM.

Classifies each incoming goal across 4 independent dimensions:
  1. Verb type (creation / modification / reasoning)
  2. Scope width (single item → system-wide)
  3. Requirement count (number of logical connectors)
  4. Structural complexity flags (conditionals, concurrency, etc.)

Clear cases are decided in <1 ms with no model calls.
Ambiguous cases (score 4–5) trigger a single minimal model call (max_tokens=5).

Code extraction happens at the start — code blocks are separated from NL intent
so that classification and intent rewriting see clean intent, and code snippets
are preserved for downstream processing.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import structlog

from src.core.config import get_config
from src.core.schemas import AgentRole, CognitiveStrategy, NodeStatus, PlanNode, TaskDAG
from src.repo_intel.stub_extractor import extract_stubs, render_stub_map

logger = structlog.get_logger(__name__)


class TaskComplexity(str, Enum):
    TRIVIAL  = "trivial"   # skip intent + planning → 1 model call
    MODERATE = "moderate"  # skip intent + ToT draft → 2 model calls
    COMPLEX  = "complex"   # full pipeline (existing behaviour)


@dataclass
class ClassificationResult:
    """Result of complexity classification with extracted code and cleaned intent."""
    complexity: TaskComplexity
    nl_intent: str                      # Natural language intent (code stripped)
    code_snippets: list[str]            # Extracted code blocks (markdown fenced)
    raw_goal: str                       # Original unmodified goal
    mentioned_files: list[str]          # Resolved file paths mentioned by user (best-effort)
    prefetched_stub_map: str            # Rendered stub-only map for Planner injection


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

    def __init__(self, client=None, repo_root: Path | None = None) -> None:
        self._client = client  # VLLMClient | None
        self._repo_root = repo_root
        self._cfg = get_config()

    @staticmethod
    def _strip_code_blocks(text: str) -> str:
        """Remove markdown fenced code blocks before heuristic scoring.

        Prevents code snippets (with semicolons, if statements, etc.) from
        polluting Dimension 3 and 4 scores.
        """
        return re.sub(r'```[\s\S]*?```', '', text, flags=re.DOTALL).strip()

    @staticmethod
    def _extract_code(text: str) -> tuple[str, list[str]]:
        """Extract code blocks from text.

        Returns:
            (nl_intent, code_snippets) where:
            - nl_intent: text with code blocks removed
            - code_snippets: list of extracted code blocks (markdown fenced format)
        """
        code_blocks = re.findall(r'```[\s\S]*?```', text, flags=re.DOTALL)
        nl_intent = re.sub(r'```[\s\S]*?```', '', text, flags=re.DOTALL).strip()
        return nl_intent, code_blocks

    # ── Public API ───────────────────────────────────────────────────────────

    async def classify(self, goal: str) -> ClassificationResult:
        """Classify a goal and extract code.

        Returns ClassificationResult with:
        - complexity: TaskComplexity (TRIVIAL/MODERATE/COMPLEX)
        - nl_intent: Natural language intent (code blocks stripped)
        - code_snippets: List of extracted code blocks
        - raw_goal: Original unmodified goal

        LLM-first classification when client available; heuristic fallback otherwise.
        """
        # ── Step 1: Extract code from goal ────────────────────────────────────
        nl_intent, code_snippets = self._extract_code(goal)
        logger.debug("task_router.code_extracted", num_snippets=len(code_snippets), goal=goal[:60])

        # ── Step 2: Classify based on clean NL intent only ────────────────────
        # Primary path: ask the model (fast max_tokens=5 call)
        complexity = None
        if self._client is not None:
            complexity = await self._model_classify(nl_intent)
            if complexity is not None:
                logger.info("task_router.classified_by_model", complexity=complexity.value, goal=goal[:60])
            else:
                logger.warning("task_router.model_classify_failed_fallback", goal=goal[:60])

        # Fallback: heuristic scoring (model unavailable or errored)
        if complexity is None:
            score = self._score(nl_intent)
            logger.debug("task_router.classified_by_heuristic", score=score, goal=goal[:60])
            if score <= 1:
                complexity = TaskComplexity.TRIVIAL
            elif score >= 5:
                complexity = TaskComplexity.COMPLEX
            else:
                complexity = TaskComplexity.MODERATE

        mentioned_files, stub_map = self._prefetch_stubs(nl_intent)

        return ClassificationResult(
            complexity=complexity,
            nl_intent=nl_intent,
            code_snippets=code_snippets,
            raw_goal=goal,
            mentioned_files=mentioned_files,
            prefetched_stub_map=stub_map,
        )

    # ── Prefetcher: mentioned-file stub extraction ─────────────────────────

    def _prefetch_stubs(self, nl_intent: str) -> tuple[list[str], str]:
        """Extract and render stubs for user-mentioned files.

        Deterministic and bounded. Never returns full file contents.
        """
        repo_root = self._repo_root or Path(__file__).resolve().parents[2]
        max_files = self._cfg.repo_intel.prefetch_max_files
        max_symbols = self._cfg.repo_intel.prefetch_max_symbols_per_file
        max_render = self._cfg.repo_intel.prefetch_max_render_chars

        mentions = self._extract_file_mentions(nl_intent)
        mentions.extend(m for m in self._extract_dir_mentions(nl_intent) if m not in mentions)
        mentions.extend(m for m in self._extract_module_mentions(nl_intent) if m not in mentions)
        if not mentions:
            return [], ""

        resolved: list[Path] = []
        for m in mentions:
            p = self._resolve_mention(repo_root, m)
            if p is None:
                continue
            if p in resolved:
                continue
            if p.is_dir():
                for f in self._expand_directory(p, repo_root=repo_root, max_files=max_files - len(resolved)):
                    if f not in resolved:
                        resolved.append(f)
                    if len(resolved) >= max_files:
                        break
            else:
                resolved.append(p)
            if len(resolved) >= max_files:
                break

        if not resolved:
            return [], ""

        # Key by repo-relative display paths to avoid leaking absolute paths into prompts.
        stubs_by_file: dict[Path, list] = {}
        resolved_strs: list[str] = []
        for p in resolved:
            try:
                if self._is_ignored_path(p, repo_root):
                    continue
                if not p.is_file():
                    continue
                if p.stat().st_size > self._cfg.repo_intel.max_file_size_kb * 1024:
                    continue
            except OSError:
                continue

            stubs = extract_stubs(p, max_symbols=max_symbols)
            if not stubs:
                continue
            try:
                display = p.relative_to(repo_root)
            except Exception:
                continue
            stubs_by_file[display] = stubs
            resolved_strs.append(str(display).replace("\\", "/"))

        stub_map = render_stub_map(stubs_by_file, max_chars=max_render)
        return resolved_strs, stub_map

    @staticmethod
    def _extract_file_mentions(text: str) -> list[str]:
        # Capture path-like tokens with known code extensions.
        # Examples: src/foo.py, `src/foo.py`, (src/foo.py:12)
        exts = r"py|js|jsx|ts|tsx|go|rs|java"
        pat = re.compile(rf"(?P<p>(?:[A-Za-z]:)?[\\/][^\s`\"']+?\.(?:{exts})|[^\s`\"']+?\.(?:{exts}))")
        found: list[str] = []
        for m in pat.finditer(text):
            p = m.group("p").strip().strip("'\"`",)
            p = p.rstrip(").,;:")
            if p and p not in found:
                found.append(p)
        return found

    @staticmethod
    def _extract_dir_mentions(text: str) -> list[str]:
        # Directory-like mentions (no extension), restricted to common repo roots.
        # Examples: src/orchestrator/, tests/unit
        pat = re.compile(
            r"(?P<p>(?:src|tests|tools|infra|fine_tuning)(?:[\\/][^\s`\"']+)+[\\/]?)",
            flags=re.IGNORECASE,
        )
        found: list[str] = []
        for m in pat.finditer(text):
            p = m.group("p").strip().strip("'\"`")
            p = p.rstrip(").,;:")
            # Skip things that look like files already
            if "." in Path(p).name:
                continue
            if p and p not in found:
                found.append(p)
        return found

    @staticmethod
    def _extract_module_mentions(text: str) -> list[str]:
        # Python-ish module mentions like src.orchestrator.fsm
        # Only accept those starting with "src." to avoid matching versions like v1.2.3.
        pat = re.compile(r"\bsrc\.(?:[A-Za-z_][A-Za-z0-9_]*\.?)+\b")
        found: list[str] = []
        for m in pat.finditer(text):
            mod = m.group(0)
            if mod and mod not in found:
                # Convert to path-like mention
                found.append(mod.replace(".", "/") + ".py")
        return found

    def _expand_directory(self, directory: Path, *, repo_root: Path, max_files: int) -> list[Path]:
        if max_files <= 0:
            return []

        # Prefer non-recursive first; then recurse if still room.
        exts = {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java"}
        candidates: list[Path] = []

        try:
            for child in directory.iterdir():
                if len(candidates) >= max_files:
                    break
                if child.is_file() and child.suffix.lower() in exts and not self._is_ignored_path(child, repo_root):
                    candidates.append(child)
        except OSError:
            return []

        if len(candidates) >= max_files:
            return sorted(candidates, key=lambda p: str(p).lower())[:max_files]

        # Fill remaining slots with a shallow recursive scan.
        try:
            for f in directory.rglob("*"):
                if len(candidates) >= max_files:
                    break
                if not f.is_file():
                    continue
                if f.suffix.lower() not in exts:
                    continue
                if self._is_ignored_path(f, repo_root):
                    continue
                if f in candidates:
                    continue
                candidates.append(f)
        except OSError:
            pass

        return sorted(candidates, key=lambda p: str(p).lower())[:max_files]

    @staticmethod
    def _is_ignored_path(path: Path, repo_root: Path) -> bool:
        try:
            rel = path.resolve().relative_to(repo_root.resolve())
        except Exception:
            return True
        ignored_parts = {".git", "node_modules", ".venv", "venv", "__pycache__", "data"}
        return any(part in ignored_parts for part in rel.parts)

    def _iter_repo_filename_matches(self, repo_root: Path, filename: str, *, max_matches: int = 50) -> list[Path]:
        """Deterministically find matching filenames under repo_root.

        Uses a sorted directory walk so the returned list order is stable across OS/filesystems.
        """
        if not filename:
            return []

        ignored_dirs = {".git", "node_modules", ".venv", "venv", "__pycache__", "data"}
        matches: list[Path] = []

        for root, dirs, files in os.walk(repo_root):
            # Deterministic traversal
            dirs.sort(key=lambda d: d.lower())
            files.sort(key=lambda f: f.lower())

            # Prune ignored dirs (case-insensitive)
            dirs[:] = [d for d in dirs if d.lower() not in {x.lower() for x in ignored_dirs}]

            if filename in files:
                p = (Path(root) / filename)
                if self._is_ignored_path(p, repo_root):
                    continue
                matches.append(p)
                if len(matches) >= max_matches:
                    break

        return matches

    def _resolve_mention(self, repo_root: Path, mention: str) -> Path | None:
        """Resolve a mention to a repo-local file path (best-effort)."""
        m = mention.strip().replace("\\", "/")
        if not m or ".." in m or m.startswith("~"):
            return None

        p = Path(m)
        if p.is_absolute() or (len(m) > 2 and m[1] == ":"):
            # Absolute path mention — only accept if inside repo root
            try:
                rp = p.resolve()
                rp.relative_to(repo_root.resolve())
                return rp
            except Exception:
                return None

        candidate = (repo_root / p).resolve()
        try:
            candidate.relative_to(repo_root.resolve())
        except Exception:
            return None

        if candidate.exists():
            return candidate

        # If mention was just a filename, search a small subset
        if "/" not in m:
            matches = self._iter_repo_filename_matches(repo_root, m, max_matches=50)
            if matches:
                # Choose first deterministic match (already in stable order)
                try:
                    return matches[0].resolve()
                except Exception:
                    return None
        return None

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

    async def _model_classify(self, nl_intent: str) -> TaskComplexity | None:
        """Single max_tokens=5 call — returns None on any failure.

        Takes already-clean NL intent (code blocks pre-extracted by caller).
        """
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
                    AgentMessage(role="user", content=nl_intent),
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

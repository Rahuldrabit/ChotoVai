"""
Context assembler — builds the precise ContextPack for each specialist agent call.

Assembly recipe (from the report):
  - Current plan node
  - Top-3 most relevant episodic trajectories (vector search)
  - Semantic rules tagged with the files being modified
  - Code-graph 2-hop neighborhood of the target function
  - Working memory summary (if compacted)
Total: ~2–3K tokens, assembled fresh per call.
"""
from __future__ import annotations

import structlog

from src.core.config import get_config
from src.core.schemas import (
    AgentRole,
    CodeSnippet,
    ContextPack,
    EpisodicEntry,
    PlanNode,
    SemanticRule,
    ValidationOutcome,
)
from src.memory.episodic import EpisodicStore

logger = structlog.get_logger(__name__)


class ContextAssembler:
    """
    Assembles a ContextPack for a single specialist call.
    Pass in optional components; missing ones are silently omitted.
    """

    def __init__(self, episodic_store: EpisodicStore | None = None) -> None:
        self._episodic = episodic_store
        self._cfg = get_config().memory

    async def assemble(
        self,
        role: AgentRole,
        plan_node: PlanNode,
        working_summary: str | None = None,
        code_snippets: list[CodeSnippet] | None = None,
        semantic_rules: list[SemanticRule] | None = None,
        token_budget: int = 3000,
    ) -> ContextPack:
        """
        Build a ContextPack for the given role and plan node.

        Args:
            role:            The specialist receiving this context.
            plan_node:       The current task node from the DAG.
            working_summary: Compressed working memory (optional).
            code_snippets:   Code snippets from graph retrieval (optional).
            semantic_rules:  Project rules applicable to this task (optional).
            token_budget:    Max tokens for the pack (enforced by truncation).
        """
        # Retrieve relevant episodic memories
        episodic_summaries: list[str] = []
        if self._episodic:
            query = f"{plan_node.title}: {plan_node.description}"
            try:
                entries: list[EpisodicEntry] = await self._episodic.retrieve(
                    query=query,
                    top_k=self._cfg.episodic_top_k,
                    outcome_filter=ValidationOutcome.PASS,
                )
                episodic_summaries = [e.action_summary for e in entries]
            except Exception as e:
                logger.warning("context_assembler.episodic_failed", error=str(e))

        # Apply semantic rule filtering — keep rules relevant to plan node description
        filtered_rules: list[str] = []
        if semantic_rules:
            node_text = (plan_node.title + " " + plan_node.description).lower()
            for rule in semantic_rules:
                # Include rule if any of its tags appear in the node description
                if not rule.tags or any(tag.lower() in node_text for tag in rule.tags):
                    filtered_rules.append(rule.rule_text)

        # Trim code snippets to fit budget (rough estimate: 4 chars per token)
        trimmed_snippets: list[CodeSnippet] = []
        used_chars = len(plan_node.description) + sum(len(s) for s in episodic_summaries + filtered_rules)
        budget_chars = token_budget * 4

        if code_snippets:
            for snippet in code_snippets:
                cost = len(snippet.content)
                if used_chars + cost <= budget_chars:
                    trimmed_snippets.append(snippet)
                    used_chars += cost
                else:
                    # Include partial: just the summary if available
                    if snippet.summary:
                        trimmed_snippets.append(CodeSnippet(
                            file_path=snippet.file_path,
                            start_line=snippet.start_line,
                            end_line=snippet.end_line,
                            content=f"[Truncated — summary]: {snippet.summary}",
                            language=snippet.language,
                        ))

        pack = ContextPack(
            agent_role=role,
            plan_node=plan_node,
            code_snippets=trimmed_snippets,
            episodic_summaries=episodic_summaries,
            semantic_rules=filtered_rules,
            working_summary=working_summary,
            total_tokens=used_chars // 4,
        )

        logger.debug(
            "context_assembled",
            role=role.value,
            snippets=len(trimmed_snippets),
            episodic=len(episodic_summaries),
            rules=len(filtered_rules),
            est_tokens=pack.total_tokens,
        )
        return pack

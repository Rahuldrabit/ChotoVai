"""
BaseAgent — the TAOR (Think → Act → Observe → Repeat) loop.

ALL agent specialists inherit from this. The loop is deterministic at the
runtime level — all intelligence lives in the model and its prompts.

Design contracts:
  - The orchestrator, not the agent, decides when to stop.
    Agents always loop until they emit a FINAL_ANSWER or hit limits.
  - Tool results are fed back as user messages (standard ReAct format).
  - The agent never accumulates state across invocations; context is assembled fresh.
"""
from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any
from uuid import uuid4

import structlog

from src.core.config import get_config
from src.core.schemas import (
    AgentCard,
    AgentCapability,
    AgentMessage,
    AgentRole,
    ContextPack,
    ToolCall,
)
from src.protocols.mcp_client import execute_tool
from src.serving.model_registry import get_client
from src.serving.vllm_client import InferenceResponse

logger = structlog.get_logger(__name__)


class AgentResult:
    """Structured result returned from a single agent invocation."""

    __slots__ = (
        "role",
        "final_answer",
        "messages",
        "tool_calls_made",
        "tool_trace",
        "tokens_used",
        "iterations",
        "duration_ms",
        "success",
    )

    def __init__(
        self,
        role: AgentRole,
        final_answer: str,
        messages: list[AgentMessage],
        tool_calls_made: int,
        tool_trace: list[dict[str, Any]],
        tokens_used: int,
        iterations: int,
        duration_ms: float,
        success: bool,
    ) -> None:
        self.role = role
        self.final_answer = final_answer
        self.messages = messages
        self.tool_calls_made = tool_calls_made
        self.tool_trace = tool_trace
        self.tokens_used = tokens_used
        self.iterations = iterations
        self.duration_ms = duration_ms
        self.success = success


class BaseAgent(ABC):
    """
    Abstract base agent implementing the TAOR loop.

    Subclasses must implement:
        - system_prompt() → str
        - allowed_tools() → list[str]
        - card() → AgentCard

    The loop parses model output for tool calls in JSON code blocks:
        ```json
        {"tool": "read_file", "arguments": {"path": "..."}}
        ```
    When the model emits text NOT in a JSON block, it is treated as the final answer.
    """

    # Regex to extract a JSON tool call from model output
    _TOOL_CALL_RE = re.compile(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        re.DOTALL | re.IGNORECASE,
    )
    _FINAL_MARKER = "FINAL_ANSWER:"

    def __init__(self, role: AgentRole) -> None:
        self.role = role
        self._client = get_client(role)
        self._cfg = get_config().agent

    @abstractmethod
    def system_prompt(self) -> str:
        """Return the agent's system prompt."""

    @abstractmethod
    def allowed_tools(self) -> list[str]:
        """Return tool names this agent is permitted to call (≤10 per report)."""

    @abstractmethod
    def card(self) -> AgentCard:
        """Return the A2A Agent Card describing this agent's capabilities."""

    # ──────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────

    async def run(
        self,
        context: ContextPack,
        user_message: str,
        extra_system: str | None = None,
        temperature: float | None = None,
        accept_plain_text_final: bool = False,
    ) -> AgentResult:
        """
        Execute the TAOR loop for one invocation.

        Args:
            context:      Assembled context pack (plan node + code + memory).
            user_message: The concrete task instruction for this invocation.
            extra_system: Optional additional system instructions (injected after base prompt).

        Returns:
            AgentResult with the final answer and execution statistics.
        """
        t0 = time.perf_counter()
        messages = self._build_initial_messages(context, user_message, extra_system)
        total_tokens = 0
        tool_calls_made = 0
        tool_trace: list[dict[str, Any]] = []
        iterations = 0

        node_id = context.plan_node.id if context.plan_node else "?"
        logger.info(
            "agent.start",
            role=self.role.value,
            node=node_id,
            task=user_message[:100],
            tools=self.allowed_tools(),
        )

        last_iter_sig = ""
        stuck_count = 0

        while iterations < self._cfg.max_iterations:
            iterations += 1
            logger.info("agent.think", role=self.role.value, node=node_id, iteration=iterations)

            # ── THINK ──
            response: InferenceResponse = await self._client.complete(
                messages,
                temperature=temperature,
            )
            total_tokens += response.tokens_in + response.tokens_out
            content = response.content.strip()

            # Append assistant turn to conversation
            messages.append(AgentMessage(role="assistant", content=content))

            # ── CHECK FOR FINAL ANSWER ──
            if self._FINAL_MARKER in content:
                answer = content.split(self._FINAL_MARKER, 1)[-1].strip()
                duration_ms = (time.perf_counter() - t0) * 1000
                logger.info(
                    "agent.done",
                    role=self.role.value,
                    node=node_id,
                    iterations=iterations,
                    tool_calls=tool_calls_made,
                    tokens=total_tokens,
                    duration_ms=round(duration_ms),
                )
                return AgentResult(
                    role=self.role,
                    final_answer=answer,
                    messages=messages,
                    tool_calls_made=tool_calls_made,
                    tool_trace=tool_trace,
                    tokens_used=total_tokens,
                    iterations=iterations,
                    duration_ms=duration_ms,
                    success=True,
                )

            # ── ACT — parse & execute tool calls ──
            tool_call_matches = self._TOOL_CALL_RE.findall(content)

            # Some callers (for example the debate critic) intentionally return
            # raw JSON/text without FINAL_ANSWER markers.
            if accept_plain_text_final and not tool_call_matches:
                duration_ms = (time.perf_counter() - t0) * 1000
                logger.info(
                    "agent.done_plain_text",
                    role=self.role.value,
                    node=node_id,
                    iterations=iterations,
                    tool_calls=tool_calls_made,
                    tokens=total_tokens,
                    duration_ms=round(duration_ms),
                )
                return AgentResult(
                    role=self.role,
                    final_answer=content,
                    messages=messages,
                    tool_calls_made=tool_calls_made,
                    tool_trace=tool_trace,
                    tokens_used=total_tokens,
                    iterations=iterations,
                    duration_ms=duration_ms,
                    success=True,
                )

            if not tool_call_matches:
                # No tool call and no FINAL_ANSWER → prompt the model to be explicit
                logger.debug("agent.no_tool_call", role=self.role.value, node=node_id, iteration=iterations)
                messages.append(AgentMessage(
                    role="user",
                    content=(
                        "Your response did not contain a tool call or FINAL_ANSWER. "
                        "Either call a tool using a ```json {\"tool\": ..., \"arguments\": ...}``` "
                        "block, or emit your final answer prefixed with 'FINAL_ANSWER:'"
                    ),
                ))
                continue

            # ── OBSERVE — execute all tool calls in this turn ──
            observation_parts: list[str] = []
            for match in tool_call_matches:
                try:
                    parsed = json.loads(match)
                    tool_name = parsed.get("tool") or parsed.get("name", "")
                    arguments = parsed.get("arguments") or parsed.get("args") or {}
                except (json.JSONDecodeError, AttributeError) as e:
                    observation_parts.append(f"ERROR: could not parse tool call JSON: {e}\nRaw: {match}")
                    continue

                if tool_name not in self.allowed_tools():
                    logger.warning(
                        "agent.tool_not_allowed",
                        role=self.role.value,
                        tool=tool_name,
                        allowed=self.allowed_tools(),
                    )
                    observation_parts.append(
                        f"ERROR: tool '{tool_name}' is not in your allowed set: {self.allowed_tools()}"
                    )
                    continue

                # Log the outgoing tool call
                args_preview = {k: str(v)[:80] for k, v in arguments.items()}
                logger.info(
                    "agent.tool_call",
                    role=self.role.value,
                    node=node_id,
                    tool=tool_name,
                    args=args_preview,
                )

                tool_call = ToolCall(name=tool_name, arguments=arguments)
                result = await execute_tool(tool_call)
                tool_calls_made += 1

                # Log the returned result
                result_preview = (
                    result.content[:120]
                    if not result.is_error
                    else f"ERROR — {result.content[:100]}"
                )
                logger.info(
                    "agent.tool_result",
                    role=self.role.value,
                    node=node_id,
                    tool=tool_name,
                    ok=not result.is_error,
                    duration_ms=round(result.duration_ms),
                    preview=result_preview,
                )

                tool_trace.append(
                    {
                        "tool": tool_name,
                        "arguments": arguments,
                        "is_error": bool(result.is_error),
                        "duration_ms": result.duration_ms,
                    }
                )
                prefix = "Result" if not result.is_error else "Error"
                observation_parts.append(f"[{prefix} from {tool_name}]\n{result.content}")

            # ── STUCK DETECTION — break if this iteration repeats the same tool calls ──
            iter_sig = "|".join(
                f"{tc.get('tool')}:{sorted((tc.get('arguments') or {}).items())}"
                for tc in tool_trace[-len(tool_call_matches):]
            ) if tool_call_matches else ""
            if iter_sig and iter_sig == last_iter_sig:
                stuck_count += 1
                if stuck_count >= 2:
                    logger.warning(
                        "agent.stuck_detected",
                        role=self.role.value,
                        node=node_id,
                        iteration=iterations,
                        sig=iter_sig[:120],
                    )
                    messages.append(AgentMessage(role="user", content="\n\n".join(observation_parts)))
                    break
            else:
                stuck_count = 0
                last_iter_sig = iter_sig

            # Feed observations back as a user turn
            messages.append(AgentMessage(
                role="user",
                content="\n\n".join(observation_parts),
            ))

        # ── LIMIT REACHED ──
        duration_ms = (time.perf_counter() - t0) * 1000
        logger.warning(
            "agent.limit_reached",
            role=self.role.value,
            node=node_id,
            iterations=iterations,
            tool_calls=tool_calls_made,
            tokens=total_tokens,
            duration_ms=round(duration_ms),
        )
        return AgentResult(
            role=self.role,
            final_answer="ERROR: agent reached maximum iterations without completing the task.",
            messages=messages,
            tool_calls_made=tool_calls_made,
            tool_trace=tool_trace,
            tokens_used=total_tokens,
            iterations=iterations,
            duration_ms=duration_ms,
            success=False,
        )

    # ──────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────

    def _build_initial_messages(
        self,
        context: ContextPack,
        user_message: str,
        extra_system: str | None,
    ) -> list[AgentMessage]:
        """Assemble the starting message list for this invocation."""
        system_content = self.system_prompt()

        if context.working_summary:
            system_content += f"\n\n## Working Memory Summary\n{context.working_summary}"

        if context.scratchpad_tail:
            system_content += f"\n\n## Session Scratchpad (recent reasoning log)\n{context.scratchpad_tail}"

        if context.contracts_context:
            system_content += f"\n\n{context.contracts_context}"

        if context.semantic_rules:
            rules_block = "\n".join(f"- {r}" for r in context.semantic_rules)
            system_content += f"\n\n## Project Rules\n{rules_block}"

        if context.episodic_summaries:
            ep_block = "\n".join(f"- {s}" for s in context.episodic_summaries)
            system_content += f"\n\n## Relevant Past Experiences\n{ep_block}"

        if context.code_snippets:
            snippets = []
            for s in context.code_snippets:
                snippets.append(f"### {s.file_path} (lines {s.start_line}–{s.end_line})\n```{s.language}\n{s.content}\n```")
            system_content += "\n\n## Relevant Code\n" + "\n\n".join(snippets)

        if extra_system:
            system_content += f"\n\n{extra_system}"

        tool_list = "\n".join(f"- `{t}`" for t in self.allowed_tools())
        system_content += f"\n\n## Available Tools\n{tool_list}\nCall tools with a ```json {{\"tool\": \"name\", \"arguments\": {{...}}}}``` block.\nWhen finished, emit your answer prefixed with 'FINAL_ANSWER:'"

        return [
            AgentMessage(role="system", content=system_content),
            AgentMessage(role="user", content=user_message),
        ]

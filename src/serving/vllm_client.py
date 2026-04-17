"""
Async inference client wrapping any OpenAI-compatible API endpoint.
Works with: vLLM, Ollama (--openai-compatible), OpenRouter, Anthropic (via adapter).
"""
from __future__ import annotations

import time
from typing import Any, AsyncIterator

import structlog
import tiktoken
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import ModelEndpoint
from src.core.schemas import AgentMessage
from src.core.tracing import trace_model_call

logger = structlog.get_logger(__name__)

# Fallback tokenizer for token counting (cl100k_base is close enough for most models)
_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


class InferenceResponse(BaseModel):
    content: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    finish_reason: str
    raw: dict[str, Any] = {}


class VLLMClient:
    """
    Async inference client for a single model endpoint.
    Each agent role gets its own VLLMClient instance.
    """

    def __init__(
        self,
        endpoint: ModelEndpoint,
        role: str = "unknown",
        lora_version: str | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.role = role
        self.lora_version = lora_version
        self._client = AsyncOpenAI(
            base_url=endpoint.base_url,
            api_key=endpoint.api_key,
            timeout=endpoint.timeout_s,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def complete(
        self,
        messages: list[AgentMessage],
        response_schema: type[BaseModel] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> InferenceResponse:
        """
        Send a chat completion request.

        If response_schema is provided, uses JSON mode with guided decoding
        to guarantee the response parses as the given Pydantic model.
        """
        t0 = time.perf_counter()

        openai_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
        ]

        kwargs: dict[str, Any] = {
            "model": self.endpoint.model_name,
            "messages": openai_messages,
            "temperature": temperature if temperature is not None else self.endpoint.temperature,
            "max_tokens": max_tokens or self.endpoint.max_tokens,
        }

        if response_schema is not None:
            # Use guided JSON decoding — vLLM supports this via response_format
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_schema.__name__,
                    "schema": response_schema.model_json_schema(),
                    "strict": True,
                },
            }

        if extra_body:
            kwargs["extra_body"] = extra_body

        response: ChatCompletion = await self._client.chat.completions.create(**kwargs)

        latency_ms = (time.perf_counter() - t0) * 1000
        tokens_in = response.usage.prompt_tokens if response.usage else 0
        tokens_out = response.usage.completion_tokens if response.usage else 0
        content = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason or "stop"

        trace_model_call(
            role=self.role,
            model_name=self.endpoint.model_name,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            lora_version=self.lora_version,
        )

        logger.debug(
            "model_call",
            role=self.role,
            model=self.endpoint.model_name,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=round(latency_ms, 1),
            finish_reason=finish_reason,
        )

        return InferenceResponse(
            content=content,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            raw=response.model_dump(),
        )

    async def complete_json(
        self,
        messages: list[AgentMessage],
        schema: type[BaseModel],
        **kwargs: Any,
    ) -> BaseModel:
        """
        Convenience wrapper: call complete() + parse the JSON response
        into the given Pydantic model. Raises on parse failure.
        """
        resp = await self.complete(messages, response_schema=schema, **kwargs)
        return schema.model_validate_json(resp.content)

    async def stream(
        self,
        messages: list[AgentMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Streaming completion — yields content deltas as they arrive."""
        openai_messages = [{"role": m.role, "content": m.content} for m in messages]
        stream = await self._client.chat.completions.create(
            model=self.endpoint.model_name,
            messages=openai_messages,
            temperature=temperature if temperature is not None else self.endpoint.temperature,
            max_tokens=max_tokens or self.endpoint.max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

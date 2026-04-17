"""
OpenTelemetry tracing setup + Langfuse exporter.
Every model call, tool call, and agent step is traced.
"""
from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any

import structlog
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Status, StatusCode

logger = structlog.get_logger(__name__)

_tracer: trace.Tracer | None = None


def setup_tracing(
    service_name: str = "slm-agent",
    enable_langfuse: bool = False,
    langfuse_public_key: str = "",
    langfuse_secret_key: str = "",
    langfuse_host: str = "https://cloud.langfuse.com",
    enable_console: bool = False,
) -> trace.Tracer:
    """Initialize OpenTelemetry tracer. Call once at startup."""
    global _tracer

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if enable_console:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    if enable_langfuse and langfuse_public_key:
        try:
            from langfuse.opentelemetry import LangfuseExporter  # type: ignore

            exporter = LangfuseExporter(
                public_key=langfuse_public_key,
                secret_key=langfuse_secret_key,
                host=langfuse_host,
            )
            provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info("Langfuse tracing enabled", host=langfuse_host)
        except ImportError:
            logger.warning("langfuse not installed; tracing to console only")

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name)
    return _tracer


def get_tracer() -> trace.Tracer:
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer("slm-agent")
    return _tracer


def trace_model_call(
    role: str,
    model_name: str,
    tokens_in: int,
    tokens_out: int,
    latency_ms: float,
    lora_version: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Record a single model inference call as a span."""
    tracer = get_tracer()
    with tracer.start_as_current_span(f"model.call.{role}") as span:
        span.set_attribute("agent.role", role)
        span.set_attribute("model.name", model_name)
        span.set_attribute("model.tokens_in", tokens_in)
        span.set_attribute("model.tokens_out", tokens_out)
        span.set_attribute("model.latency_ms", latency_ms)
        if lora_version:
            span.set_attribute("model.lora_version", lora_version)
        if extra:
            for k, v in extra.items():
                span.set_attribute(f"extra.{k}", str(v))


def traced(span_name: str | None = None) -> Callable:
    """Decorator that wraps async or sync functions in an OTel span."""
    def decorator(fn: Callable) -> Callable:
        name = span_name or f"{fn.__module__}.{fn.__qualname__}"

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            with tracer.start_as_current_span(name) as span:
                t0 = time.perf_counter()
                try:
                    result = await fn(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as exc:
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    span.record_exception(exc)
                    raise
                finally:
                    elapsed = (time.perf_counter() - t0) * 1000
                    span.set_attribute("duration_ms", elapsed)

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            with tracer.start_as_current_span(name) as span:
                t0 = time.perf_counter()
                try:
                    result = fn(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as exc:
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    span.record_exception(exc)
                    raise
                finally:
                    elapsed = (time.perf_counter() - t0) * 1000
                    span.set_attribute("duration_ms", elapsed)

        import asyncio
        if asyncio.iscoroutinefunction(fn):
            return async_wrapper
        return sync_wrapper

    return decorator

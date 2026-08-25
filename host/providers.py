"""Provider client factories and completion dispatch.

Transport-neutral module extracted from ``hosts/little_lisa/main.py``.
All state access goes through :class:`HostRuntime`; all UI side-effects
go through :class:`TransportCallbacks`. This module never imports Rich,
never calls ``console.print``, never calls ``_append_log_event``, and
never calls ``_publish_token_usage``. Logging and usage publishing are
wired through the runtime in a later step; for now they are no-op stubs.

Public surface::

    from host.providers import complete_live, azure_client_instance

    result = await complete_live(
        runtime=runtime,
        transport=transport,
        messages=msg_list,
        model_def=model_def,
        max_tokens=4096,
        temperature=0.2,
    )
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import time
from typing import Any

import httpx
from openai import AsyncAzureOpenAI, AsyncOpenAI
from pydantic import BaseModel

from host.callbacks import NullTransport, TransportCallbacks
from host.model_runtime import StubCompletion, StubModelDefinition
from host.provider_completions import (
    _extract_anthropic_error_detail,
    _extract_anthropic_text,
    _extract_anthropic_tool_calls,
    _extract_completion_tool_calls,
    _extract_openai_error_detail,
    _first_message_content,
    _normalize_ollama_host_to_base_url,
    _normalize_structured_content,
    _to_anthropic_payload_messages,
    _to_anthropic_tools,
)
from host.provider_completions import (
    _extract_usage_counts as _provider_extract_usage_counts,
)
from host.provider_completions import (
    _normalize_google_base_url as _provider_normalize_google_base_url,
)
from host.runtime import HostRuntime
from host.token_accounting import coerce_non_negative_int

logger = logging.getLogger(__name__)

# ── Environment-variable name constants (LISA_*) ─────────────────────
# Provider credential/env configuration. Renamed from the legacy
# LITTLE_LISA_* prefixes — no compat aliases.

LISA_AZURE_ENDPOINT_ENV = "LISA_AZURE_OPENAI_ENDPOINT"
LISA_AZURE_API_KEY_ENV = "LISA_AZURE_OPENAI_API_KEY"
LISA_AZURE_API_VERSION_ENV = "LISA_AZURE_OPENAI_API_VERSION"
# The Responses API (``/openai/responses``) is only available at preview
# api-versions; the GA ``2024-10-21`` only covers ``/chat/completions``
# and returns 404 for the Responses endpoint. Codex-family models on
# Azure require the Responses API, so they need a client built with a
# Responses-capable api-version. No code default — set
# ``LISA_AZURE_OPENAI_RESPONSES_API_VERSION`` in the environment (e.g.
# ``.env``); the factory raises when it is missing.
LISA_AZURE_RESPONSES_API_VERSION_ENV = "LISA_AZURE_OPENAI_RESPONSES_API_VERSION"
LISA_OPENAI_API_KEY_ENV = "LISA_OPENAI_API_KEY"
LISA_GOOGLE_API_KEY_ENV = "LISA_GOOGLE_API_KEY"
LISA_GOOGLE_API_BASE_URL_ENV = "LISA_GOOGLE_API_BASE_URL"
LISA_OLLAMA_HOST_ENV = "LISA_OLLAMA_HOST"
LISA_OLLAMA_API_KEY_ENV = "LISA_OLLAMA_API_KEY"
LISA_ANTHROPIC_BASE_URL_ENV = "LISA_ANTHROPIC_BASE_URL"
LISA_ANTHROPIC_API_KEY_ENV = "LISA_ANTHROPIC_API_KEY"
LISA_ANTHROPIC_VERSION_ENV = "LISA_ANTHROPIC_VERSION"

_DEFAULT_GOOGLE_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


# ── Pure helpers ──────────────────────────────────────────────────────


def _is_pydantic_model_class(obj: Any) -> bool:
    """Return True when obj is a Pydantic BaseModel class."""
    return isinstance(obj, type) and issubclass(obj, BaseModel)


def _build_json_schema_response_format(
    response_format: type[Any],
) -> dict[str, Any]:
    """Build an OpenAI-compatible json_schema response_format dict from a Pydantic model.

    Emits the full schema so the provider can constrain the model to the declared
    shape, not merely ``json_object``. ``strict`` is left unset (defaults to
    false on the provider side) because structured output models have optional
    fields with defaults that would otherwise violate strict-schema constraints.
    """
    schema = response_format.model_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": getattr(response_format, "__name__", "response") or "response",
            "schema": schema,
        },
    }


def _normalize_response_format(response_format: type[Any] | None) -> dict[str, Any] | None:
    """Return the OpenAI-compatible response_format dict, or None.

    When ``response_format`` is a Pydantic ``BaseModel`` subclass, emit a
    ``json_schema`` response_format so the provider receives the full schema and
    can constrain the model to the declared shape. Non-Pydantic dict values are
    passed through unchanged so callers that pre-build a response_format dict
    are not disrupted.
    """
    if response_format is None:
        return None
    if _is_pydantic_model_class(response_format):
        return _build_json_schema_response_format(response_format)
    if isinstance(response_format, dict):
        return dict(response_format)
    return {"type": "json_object"}


def normalize_google_base_url(base_url: str) -> str:
    """Return a supported Google OpenAI-compatible base URL."""
    return _provider_normalize_google_base_url(base_url, _DEFAULT_GOOGLE_OPENAI_BASE_URL)


def _requires_thought_signature(model_def: StubModelDefinition) -> bool:
    """Return True for Gemini thinking models on the OpenAI-compatible endpoint.

    Gemini 2.5 Pro / 3.x models think by default and emit a
    ``thoughtSignature`` on every ``functionCall`` part. The signature must
    be echoed verbatim on the next turn or Gemini rejects the request with
    ``400 INVALID_ARGUMENT: Function call is missing a thought_signature``.
    The kernel synthesizes ``transition_stage`` tool calls (planned handoff,
    forced fork, forced plan) with no signature; those calls must be pruned
    before the request reaches Gemini. Only Google-provider thinking
    models are affected — OpenAI / Azure / Ollama do not validate
    ``thought_signature``.
    """

    if model_def.provider != "google":
        return False
    return bool(getattr(model_def, "thinking_model", False))


def completion_response_format_name(response_format: type[Any] | None) -> str:
    """Return the __name__ of the response_format type, or empty string."""
    if response_format is None:
        return ""
    return str(getattr(response_format, "__name__", "") or "")


def build_completion_kwargs(
    *,
    model_def: StubModelDefinition,
    max_tokens: int,
    temperature: float,
    tools: list[dict[str, Any]] | None,
    response_format: type[Any] | None,
) -> dict[str, Any]:
    """Build the kwargs dict for an OpenAI-compatible chat completion call."""
    kwargs: dict[str, Any] = {
        "model": model_def.model_id,
        "messages": [],  # placeholder — caller sets messages via kwargs.update
    }
    if model_def.provider == "azure_openai" or model_def.reasoning_model:
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = temperature
    if tools:
        kwargs["tools"] = tools
    normalized_format = _normalize_response_format(response_format)
    if normalized_format is not None:
        kwargs["response_format"] = normalized_format
    return kwargs


def extract_usage_counts(usage_payload: Any) -> tuple[int, int, int, int, int]:
    """Extract token usage from provider response.

    Returns: (prompt_tokens, completion_tokens, total_tokens, cached_tokens, reasoning_tokens)
    """
    return _provider_extract_usage_counts(usage_payload, coerce_non_negative_int)


def derive_openai_compatible_truncated(
    *,
    finish_reason: str | None,
    completion_tokens: int,
    requested_tokens: int,
) -> bool | None:
    """Derive the authoritative ``truncated`` signal for an OpenAI-compatible response.

    ``"length"`` is the authoritative hit-max-tokens case → ``True``.
    A missing signal (``None``) on a near-budget completion (≥90% of the
    requested token budget) is treated as truncation — this is the
    Ollama-defensive case where local backends omit ``finish_reason`` on
    long docs. ``"stop"``/``"tool_calls"`` → ``False``. Anything else →
    ``None`` (adapter genuinely cannot tell).
    """
    if finish_reason == "length":
        return True
    if finish_reason in {"stop", "tool_calls"}:
        return False
    if (
        finish_reason is None
        and requested_tokens > 0
        and completion_tokens >= int(requested_tokens * 0.9)
    ):
        return True
    return None


def derive_responses_api_truncated(*, status: str | None) -> bool | None:
    """Derive ``truncated`` for the Azure Responses API.

    Status vocabulary: ``"completed"`` / ``"incomplete"`` / ``"failed"``.
    ``"incomplete"`` is the authoritative truncation signal.
    """
    if status == "incomplete":
        return True
    if status == "completed":
        return False
    return None


def derive_anthropic_truncated(*, stop_reason: str | None) -> bool | None:
    """Derive ``truncated`` for an Anthropic completion from its native ``stop_reason``.

    ``max_tokens`` / ``max_sequence`` → ``True`` (hit a limit).
    ``end_turn`` / ``stop_sequence`` / ``tool_use`` → ``False`` (clean stop).
    Anything else (including ``None``) → ``None`` (adapter cannot tell).
    """
    if stop_reason in {"max_tokens", "max_sequence"}:
        return True
    if stop_reason in {"end_turn", "stop_sequence", "tool_use"}:
        return False
    return None


def with_duration(result: StubCompletion, elapsed_ms: float) -> StubCompletion:
    """Rebuild a frozen StubCompletion with the host-measured duration."""
    return StubCompletion(
        content=result.content,
        tool_calls=result.tool_calls,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.total_tokens,
        cached_tokens=result.cached_tokens,
        reasoning_tokens=result.reasoning_tokens,
        duration_ms=elapsed_ms,
    )


def stage_status_context(stream_stage: bool, stage_label: str) -> Any:
    """Return a no-op context manager for the provider call.

    Transport stage notification is handled via :meth:`TransportCallbacks.on_stage`
    in the completion dispatch functions. This context manager exists only for
    structural compatibility — it does nothing.
    """
    return contextlib.nullcontext()


def _extract_markdown_section(content: str, heading: str) -> str:
    """Extract the body text under a ``## heading`` markdown section."""
    marker = re.escape(f"## {heading}")
    pattern = re.compile(rf"(?ms)^{marker}\n\n(.*?)(?=\n\n## |\Z)")
    match = pattern.search(content)
    if not match:
        return ""
    return match.group(1).strip()


def summarize_turn_messages(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a structured summary of the turn's message list for logging/debug."""
    system_prompt = ""
    user_payload = ""
    assistant_tool_call_count = 0
    tool_result_count = 0

    for message in messages:
        role = str(message.get("role") or "").strip().lower()
        content = message.get("content")
        if role == "system" and isinstance(content, str) and not system_prompt:
            system_prompt = content
        if role == "user" and isinstance(content, str) and not user_payload:
            user_payload = content
        if role == "assistant":
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                assistant_tool_call_count += len(tool_calls)
        if role == "tool":
            tool_result_count += 1

    max_preview_chars = 240
    system_prompt_preview = system_prompt[:max_preview_chars]
    return {
        "system_prompt": {
            "chars": len(system_prompt),
            "preview": system_prompt_preview,
            "omitted_for_brevity": len(system_prompt) > len(system_prompt_preview),
        },
        "assembled_user_payload": {
            "pre_turn_briefing": _extract_markdown_section(user_payload, "Pre-turn Briefing"),
            "memory_recall": _extract_markdown_section(user_payload, "Memory Recall"),
            "dynamic_user_context": _extract_markdown_section(user_payload, "Dynamic User Context"),
            "selected_user_context": _extract_markdown_section(
                user_payload, "Selected User Context"
            ),
            "user_query": _extract_markdown_section(user_payload, "User Query"),
            "chars": len(user_payload),
        },
        "message_count": len(messages),
        "assistant_tool_call_count": assistant_tool_call_count,
        "tool_result_count": tool_result_count,
    }


# ── Logging and usage stubs (no-ops — wired through runtime later) ────


def log_completion_request(
    *,
    messages: list[dict[str, Any]],
    model_def: StubModelDefinition,
    max_tokens: int,
    temperature: float,
    response_format_name: str,
    tools: list[dict[str, Any]] | None,
) -> None:
    """Log the completion.request debug event.

    Currently a no-op — logging will be wired through the runtime in a later step.
    """
    _ = (messages, model_def, max_tokens, temperature, response_format_name, tools)


def log_completion_error(
    *,
    exc: Exception,
    model_def: StubModelDefinition,
    response_format_name: str,
    elapsed_ms: float,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> None:
    """Log the completion.error event for a failed provider call.

    Emits a structured ``logger.error`` record so provider failures
    (400s, timeouts, auth errors) are visible in the JSONL turn log with
    enough context to diagnose — ``exc_type``, a concise detail string,
    provider/model, API version, finish_reason, and a token/prompt-size
    estimate. Never raises: a logging failure must not break the re-raise
    that the caller performs after this call.
    """
    try:
        detail = _extract_openai_error_detail(exc)
        # Extract finish_reason from the HTTP response when available
        # (some providers include it in the error body for length/timeout
        # truncations surfaced as errors).
        finish_reason = ""
        response = getattr(exc, "response", None)
        if response is not None:
            try:
                body = response.json()
                if isinstance(body, dict):
                    finish_reason = str(body.get("finish_reason") or body.get("error", {}).get("code") or "")
            except (ValueError, AttributeError):
                pass
        prompt_token_estimate = sum(len(m.get("content", "")) for m in messages) // 4
        logger.error(
            "completion.error: exc_type=%s detail=%s provider=%s model_id=%s "
            "api_version=%s response_format=%s elapsed_ms=%.1f finish_reason=%s "
            "message_count=%d tool_count=%d prompt_token_estimate=%d",
            type(exc).__name__,
            detail[:500],
            model_def.provider,
            model_def.model_id,
            getattr(model_def, "api_version", ""),
            response_format_name,
            elapsed_ms,
            finish_reason,
            len(messages),
            len(tools) if tools else 0,
            prompt_token_estimate,
        )
    except Exception:  # pylint: disable=broad-except
        logger.debug("log_completion_error failed", exc_info=True)


def log_completion_response(
    *,
    model_def: StubModelDefinition,
    response_format_name: str,
    elapsed_ms: float,
    result: StubCompletion,
) -> None:
    """Log the completion.response debug event for a successful provider call.

    Emits a structured ``logger.debug`` record so successful completions
    are visible in the JSONL turn log under debug-level logging: provider,
    model, elapsed time, finish_reason, and the token counters from the
    provider's usage object. Kept at debug level so the default log stays
    lean; gated implicitly by the logger's effective level.
    """
    try:
        logger.debug(
            "completion.response: provider=%s model_id=%s response_format=%s "
            "elapsed_ms=%.1f finish_reason=%s prompt_tokens=%d completion_tokens=%d "
            "cached_tokens=%d reasoning_tokens=%d",
            model_def.provider,
            model_def.model_id,
            response_format_name,
            elapsed_ms,
            getattr(result, "finish_reason", ""),
            getattr(result, "prompt_tokens", 0),
            getattr(result, "completion_tokens", 0),
            getattr(result, "cached_tokens", 0),
            getattr(result, "reasoning_tokens", 0),
        )
    except Exception:  # pylint: disable=broad-except
        logger.debug("log_completion_response failed", exc_info=True)


async def publish_completion_usage(
    *,
    runtime: HostRuntime,
    model_def: StubModelDefinition,
    result: StubCompletion,
    elapsed_ms: float,
    call_site: str,
) -> None:
    """Publish one ``TokenUsageEvent`` for a completed provider call.

    Builds a durable token-usage record from the completion's token counters
    and the host-measured wall-clock duration, then publishes it to the
    runtime's ``TokenUsageTelemetryHandler``. The handler appends it as one
    JSONL line to ``token-usage.jsonl``; ``introspect`` reads that file to
    build the 24h token summary. Failures are swallowed — telemetry must
    never break the turn loop.
    """
    handler = runtime.token_usage_handler
    if handler is None:
        return
    try:
        from kernel.token_usage import build_token_usage_event

        event = build_token_usage_event(
            provider=model_def.provider,
            model_id=model_def.model_id,
            kind="chat_completion",
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            cached_tokens=result.cached_tokens,
            reasoning_tokens=result.reasoning_tokens,
            duration_ms=elapsed_ms,
            latency_ms=int(elapsed_ms),
            call_site=call_site,
        )
        await handler.publish(event)
    except Exception:  # pylint: disable=broad-except
        logger.warning("token-usage publish failed for %s", call_site, exc_info=True)


# ── Provider client factories ────────────────────────────────────────


def azure_client_instance(runtime: HostRuntime) -> AsyncAzureOpenAI:
    """Return the cached Azure OpenAI client, creating it on first use."""
    existing_client = runtime.azure_client
    if isinstance(existing_client, AsyncAzureOpenAI):
        return existing_client

    endpoint = (
        os.getenv(LISA_AZURE_ENDPOINT_ENV, "").strip()
        or os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    )
    api_key = (
        os.getenv(LISA_AZURE_API_KEY_ENV, "").strip()
        or os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    )
    if not endpoint or not api_key:
        raise ValueError(
            "Azure OpenAI is not configured. Set "
            f"{LISA_AZURE_ENDPOINT_ENV} and {LISA_AZURE_API_KEY_ENV} "
            "(or AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY)."
        )
    api_version = os.getenv(LISA_AZURE_API_VERSION_ENV, "").strip() or "2024-10-21"
    client = AsyncAzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key,
    )
    runtime.azure_client = client
    return client


def azure_responses_client_instance(runtime: HostRuntime) -> AsyncAzureOpenAI:
    """Return the cached Azure OpenAI client for the Responses API.

    The Responses endpoint (``/openai/responses``) requires a preview
    api-version; the GA ``2024-10-21`` returns 404 for it. This factory
    builds a separate client cached on ``runtime.azure_responses_client``
    so the chat-completions client keeps its own api-version.
    ``LISA_AZURE_OPENAI_RESPONSES_API_VERSION`` is required (no code
    default); set it in the environment (e.g. ``.env``) to the preview
    version the Azure deployment exposes.
    """
    existing_client = runtime.azure_responses_client
    if isinstance(existing_client, AsyncAzureOpenAI):
        return existing_client

    endpoint = (
        os.getenv(LISA_AZURE_ENDPOINT_ENV, "").strip()
        or os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    )
    api_key = (
        os.getenv(LISA_AZURE_API_KEY_ENV, "").strip()
        or os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    )
    if not endpoint or not api_key:
        raise ValueError(
            "Azure OpenAI is not configured. Set "
            f"{LISA_AZURE_ENDPOINT_ENV} and {LISA_AZURE_API_KEY_ENV} "
            "(or AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY)."
        )
    api_version = os.getenv(LISA_AZURE_RESPONSES_API_VERSION_ENV, "").strip()
    if not api_version:
        raise ValueError(
            "The Azure Responses API requires a preview api-version that "
            "supports ``/openai/responses``. Set "
            f"{LISA_AZURE_RESPONSES_API_VERSION_ENV} in the environment "
            "(e.g. .env) to the preview version the deployment exposes."
        )
    client = AsyncAzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key,
    )
    runtime.azure_responses_client = client
    return client


def openai_client_instance(runtime: HostRuntime) -> AsyncOpenAI:
    """Return the cached OpenAI client, creating it on first use."""
    existing_client = runtime.openai_client
    if isinstance(existing_client, AsyncOpenAI):
        return existing_client

    api_key = (
        os.getenv(LISA_OPENAI_API_KEY_ENV, "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    )
    if not api_key:
        raise ValueError(
            f"OpenAI is not configured. Set {LISA_OPENAI_API_KEY_ENV} (or OPENAI_API_KEY)."
        )
    client = AsyncOpenAI(api_key=api_key)
    runtime.openai_client = client
    return client


def google_client_instance(runtime: HostRuntime) -> AsyncOpenAI:
    """Return the cached Google Gemini (OpenAI-compatible) client, creating it on first use."""
    existing_client = runtime.google_client
    if isinstance(existing_client, AsyncOpenAI):
        return existing_client

    api_key = (
        os.getenv(LISA_GOOGLE_API_KEY_ENV, "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    )
    endpoint = normalize_google_base_url(
        os.getenv(LISA_GOOGLE_API_BASE_URL_ENV, "").strip()
        or os.getenv("GOOGLE_API_BASE_URL", "").strip()
    )
    if not api_key:
        raise ValueError(
            f"Google Gemini is not configured. Set {LISA_GOOGLE_API_KEY_ENV} (or GOOGLE_API_KEY)."
        )
    client = AsyncOpenAI(api_key=api_key, base_url=endpoint)
    runtime.google_client = client
    return client


def ollama_client_instance(runtime: HostRuntime) -> AsyncOpenAI:
    """Return the cached Ollama (OpenAI-compatible) client, creating it on first use."""
    existing_client = runtime.ollama_client
    if isinstance(existing_client, AsyncOpenAI):
        return existing_client

    configured_host = (
        os.getenv(LISA_OLLAMA_HOST_ENV, "").strip() or os.getenv("OLLAMA_HOST", "").strip()
    )
    base_url = _normalize_ollama_host_to_base_url(configured_host) or "http://localhost:11434/v1/"

    api_key = (
        os.getenv(LISA_OLLAMA_API_KEY_ENV, "").strip() or os.getenv("OLLAMA_API_KEY", "").strip()
    )
    client = AsyncOpenAI(api_key=api_key or "ollama", base_url=base_url)
    runtime.ollama_client = client
    return client


def anthropic_http_client_instance(runtime: HostRuntime) -> httpx.AsyncClient:
    """Return the cached Anthropic HTTP client, creating it on first use."""
    existing_client = runtime.anthropic_http_client
    if isinstance(existing_client, httpx.AsyncClient):
        return existing_client

    configured_base_url = os.getenv(LISA_ANTHROPIC_BASE_URL_ENV, "").strip()
    base_url = (
        configured_base_url.rstrip("/") if configured_base_url else "https://api.anthropic.com"
    )
    client = httpx.AsyncClient(base_url=base_url, timeout=60.0)
    runtime.anthropic_http_client = client
    return client



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

    Currently a no-op — logging will be wired through the runtime in a later step.
    """
    _ = (exc, model_def, response_format_name, elapsed_ms, messages, tools)


def log_completion_response(
    *,
    model_def: StubModelDefinition,
    response_format_name: str,
    elapsed_ms: float,
    result: StubCompletion,
) -> None:
    """Log the completion.response debug event for a successful provider call.

    Currently a no-op — logging will be wired through the runtime in a later step.
    """
    _ = (model_def, response_format_name, elapsed_ms, result)


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


# ── Completion dispatch ──────────────────────────────────────────────


async def complete_anthropic(
    *,
    runtime: HostRuntime,
    messages: list[dict[str, Any]],
    model_def: StubModelDefinition,
    max_tokens: int,
    temperature: float,
    tools: list[dict[str, Any]] | None = None,
    response_format: type[Any] | None = None,
) -> StubCompletion:
    """Execute an Anthropic provider completion via the HTTP API."""
    api_key = (
        os.getenv(LISA_ANTHROPIC_API_KEY_ENV, "").strip()
        or os.getenv("ANTHROPIC_API_KEY", "").strip()
    )
    if not api_key:
        raise ValueError(
            f"Anthropic is not configured. Set {LISA_ANTHROPIC_API_KEY_ENV} (or ANTHROPIC_API_KEY)."
        )

    system_prompt, anthropic_messages = _to_anthropic_payload_messages(messages)
    payload: dict[str, Any] = {
        "model": model_def.model_id,
        "max_tokens": max_tokens,
        "messages": anthropic_messages,
    }
    if temperature >= 0:
        payload["temperature"] = temperature
    if system_prompt:
        payload["system"] = system_prompt

    anthropic_tools, name_map = _to_anthropic_tools(tools)
    if anthropic_tools:
        payload["tools"] = anthropic_tools

    if response_format is not None:
        format_prompt = "Return only a valid JSON object with no Markdown wrapper."
        payload["system"] = f"{payload.get('system', '').strip()}\n\n{format_prompt}".strip()

    headers = {
        "x-api-key": api_key,
        "anthropic-version": os.getenv(LISA_ANTHROPIC_VERSION_ENV, "").strip() or "2023-06-01",
    }
    response = await anthropic_http_client_instance(runtime).post(
        "/v1/messages",
        headers=headers,
        json=payload,
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = _extract_anthropic_error_detail(exc.response)
        message = (
            f"Anthropic request failed ({exc.response.status_code}) for model "
            f"'{model_def.model_id}': {detail}"
        )
        if exc.response.status_code == httpx.codes.NOT_FOUND:
            message += (
                " Check the model_id in models.yaml and confirm the model is available "
                "for your Anthropic account."
            )
        raise httpx.HTTPStatusError(
            message,
            request=exc.request,
            response=exc.response,
        ) from exc

    response_json = response.json()
    content_blocks = response_json.get("content")
    content = _extract_anthropic_text(content_blocks)
    tool_calls = _extract_anthropic_tool_calls(content_blocks, name_map)
    (
        prompt_tokens,
        completion_tokens,
        total_tokens,
        cached_tokens,
        reasoning_tokens,
    ) = extract_usage_counts(response_json.get("usage"))

    content = _normalize_structured_content(content, response_format)

    # Anthropic's native stop_reason is the authoritative termination
    # signal. Map it to finish_reason (fixing a latent bug where it was
    # discarded → operations always saw None) and derive truncated from
    # the max_tokens/max_sequence cases.
    stop_reason = str(response_json.get("stop_reason") or "").strip().lower() or None
    truncated = derive_anthropic_truncated(stop_reason=stop_reason)

    return StubCompletion(
        content=content,
        tool_calls=tool_calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
        finish_reason=stop_reason,
        truncated=truncated,
    )


async def dispatch_openai_compatible_provider(
    *,
    runtime: HostRuntime,
    model_def: StubModelDefinition,
    kwargs: dict[str, Any],
) -> Any:
    """Dispatch to the correct OpenAI-compatible client based on provider name."""
    provider = model_def.provider
    if provider == "azure_openai":
        return await azure_client_instance(runtime).chat.completions.create(**kwargs)
    if provider == "openai":
        return await openai_client_instance(runtime).chat.completions.create(**kwargs)
    if provider == "ollama":
        return await ollama_client_instance(runtime).chat.completions.create(**kwargs)
    if provider == "google":
        return await google_client_instance(runtime).chat.completions.create(**kwargs)
    raise ValueError(f"Unsupported provider '{provider}' in host manifest.")


async def complete_live(
    *,
    runtime: HostRuntime,
    transport: TransportCallbacks | None = None,
    messages: list[dict[str, Any]],
    model_def: StubModelDefinition,
    max_tokens: int,
    temperature: float,
    tools: list[dict[str, Any]] | None = None,
    response_format: type[Any] | None = None,
) -> StubCompletion:
    """Execute a live provider completion against configured provider APIs."""
    tport = transport or NullTransport()

    stream_stage = runtime.config.streaming.enabled and runtime.config.streaming.mode in {
        "stage",
        "hybrid",
    }
    stage_label = f"{model_def.provider}:{model_def.model_id}"
    started_at = time.perf_counter()
    response_format_name = completion_response_format_name(response_format)
    kwargs = build_completion_kwargs(
        model_def=model_def,
        max_tokens=max_tokens,
        temperature=temperature,
        tools=tools,
        response_format=response_format,
    )

    log_completion_request(
        messages=messages,
        model_def=model_def,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format_name=response_format_name,
        tools=tools,
    )

    if model_def.provider == "anthropic":
        return await complete_live_anthropic(
            runtime=runtime,
            transport=tport,
            messages=messages,
            model_def=model_def,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            response_format=response_format,
            stream_stage=stream_stage,
            stage_label=stage_label,
            started_at=started_at,
            response_format_name=response_format_name,
        )

    if _uses_openai_responses_api(model_def):
        return await complete_live_responses(
            runtime=runtime,
            transport=tport,
            messages=messages,
            model_def=model_def,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            response_format=response_format,
            stream_stage=stream_stage,
            stage_label=stage_label,
            started_at=started_at,
            response_format_name=response_format_name,
        )

    return await complete_live_openai_compatible(
        runtime=runtime,
        transport=tport,
        messages=messages,
        model_def=model_def,
        response_format=response_format,
        stream_stage=stream_stage,
        stage_label=stage_label,
        started_at=started_at,
        response_format_name=response_format_name,
        kwargs=kwargs,
        tools=tools,
    )


async def complete_live_anthropic(
    *,
    runtime: HostRuntime,
    transport: TransportCallbacks | None = None,
    messages: list[dict[str, Any]],
    model_def: StubModelDefinition,
    max_tokens: int,
    temperature: float,
    tools: list[dict[str, Any]] | None,
    response_format: type[Any] | None,
    stream_stage: bool,
    stage_label: str,
    started_at: float,
    response_format_name: str,
) -> StubCompletion:
    """Execute an Anthropic provider completion with stage callbacks."""
    tport = transport or NullTransport()
    try:
        if stream_stage:
            await tport.on_stage("model_call", {"label": stage_label})
        result = await complete_anthropic(
            runtime=runtime,
            messages=messages,
            model_def=model_def,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            response_format=response_format,
        )
        if stream_stage:
            await tport.on_stage("model_call_complete", {"label": stage_label})
    except Exception as exc:  # pylint: disable=broad-except
        if stream_stage:
            await tport.on_stage("model_call_failed", {"label": stage_label})
        elapsed_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
        log_completion_error(
            exc=exc,
            model_def=model_def,
            response_format_name=response_format_name,
            elapsed_ms=elapsed_ms,
            messages=messages,
            tools=tools,
        )
        raise

    elapsed_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
    log_completion_response(
        model_def=model_def,
        response_format_name=response_format_name,
        elapsed_ms=elapsed_ms,
        result=result,
    )
    final_result = with_duration(result, elapsed_ms)
    await publish_completion_usage(
        runtime=runtime,
        model_def=model_def,
        result=final_result,
        elapsed_ms=elapsed_ms,
        call_site="complete_live.anthropic",
    )
    return final_result


def _uses_openai_responses_api(model_def: StubModelDefinition) -> bool:
    """Return True when this model must be called via the Responses API.

    Codex-family models on Azure OpenAI only expose the Responses API; the
    chat-completions endpoint rejects them with ``The requested operation is
    unsupported.`` The field is forwarded from the model manifest's
    ``completion_api`` (default ``chat_completions``), so only Responses-only
    models opt in.
    """
    completion_api = str(getattr(model_def, "completion_api", "chat_completions") or "")
    return completion_api.strip().lower() == "responses"


def _build_responses_payload(
    *,
    model_def: StubModelDefinition,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    tools: list[dict[str, Any]] | None,
    response_format: type[Any] | None,
) -> dict[str, Any]:
    """Build the kwargs dict for one OpenAI Responses API call.

    Translates chat-completions history into Responses ``input`` items and
    tool definitions into Responses function tools. Structured-output hints
    become ``text.format``. Codex models use ``max_output_tokens``.
    """
    from kernel.provider_core import (
        _normalize_openai_response_format,
        _normalize_openai_response_tool,
        build_openai_responses_input,
    )

    payload: dict[str, Any] = {
        "model": model_def.model_id,
        "input": build_openai_responses_input(list(messages)),
        "max_output_tokens": max_tokens,
    }
    if not model_def.reasoning_model and not getattr(model_def, "thinking_model", False):
        payload["temperature"] = temperature
    if tools:
        payload["tools"] = [_normalize_openai_response_tool(tool) for tool in tools]
    normalized_format = _normalize_openai_response_format(response_format)
    if normalized_format is not None:
        payload["text"] = {"format": normalized_format}
    return payload


async def complete_live_responses(
    *,
    runtime: HostRuntime,
    transport: TransportCallbacks | None = None,
    messages: list[dict[str, Any]],
    model_def: StubModelDefinition,
    max_tokens: int,
    temperature: float,
    tools: list[dict[str, Any]] | None = None,
    response_format: type[Any] | None = None,
    stream_stage: bool,
    stage_label: str,
    started_at: float,
    response_format_name: str,
) -> StubCompletion:
    """Execute a Responses-API provider completion (e.g. Codex on Azure)."""
    from kernel.provider_core import (
        extract_responses_content,
        extract_responses_tool_calls,
        extract_responses_usage,
    )

    tport = transport or NullTransport()
    payload = _build_responses_payload(
        model_def=model_def,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        tools=tools,
        response_format=response_format,
    )
    try:
        if stream_stage:
            await tport.on_stage("model_call", {"label": stage_label})
        client = azure_client_instance(runtime)
        response = await client.responses.create(**payload)
        if stream_stage:
            await tport.on_stage("model_call_complete", {"label": stage_label})
    except Exception as exc:  # pylint: disable=broad-except
        if stream_stage:
            await tport.on_stage("model_call_failed", {"label": stage_label})
        elapsed_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
        log_completion_error(
            exc=exc,
            model_def=model_def,
            response_format_name=response_format_name,
            elapsed_ms=elapsed_ms,
            messages=messages,
            tools=tools,
        )
        raise

    content = extract_responses_content(response)
    tool_calls = extract_responses_tool_calls(response)
    usage = extract_responses_usage(response)
    finish_reason = getattr(response, "status", None)
    truncated = derive_responses_api_truncated(status=finish_reason)
    elapsed_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)

    result = StubCompletion(
        content=content,
        tool_calls=tool_calls,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
        total_tokens=getattr(usage, "total_tokens", 0) if usage else 0,
        cached_tokens=getattr(usage, "cached_tokens", 0) if usage else 0,
        reasoning_tokens=getattr(usage, "reasoning_tokens", 0) if usage else 0,
        duration_ms=elapsed_ms,
        finish_reason=finish_reason,
        truncated=truncated,
        reasoning_content="",
    )
    log_completion_response(
        model_def=model_def,
        response_format_name=response_format_name,
        elapsed_ms=elapsed_ms,
        result=result,
    )
    await publish_completion_usage(
        runtime=runtime,
        model_def=model_def,
        result=result,
        elapsed_ms=elapsed_ms,
        call_site="complete_live.responses",
    )
    return result


async def complete_live_openai_compatible(
    *,
    runtime: HostRuntime,
    transport: TransportCallbacks | None = None,
    messages: list[dict[str, Any]],
    model_def: StubModelDefinition,
    response_format: type[Any] | None,
    stream_stage: bool,
    stage_label: str,
    started_at: float,
    response_format_name: str,
    kwargs: dict[str, Any],
    tools: list[dict[str, Any]] | None,
) -> StubCompletion:
    """Execute an OpenAI-compatible provider completion with stage callbacks."""
    tport = transport or NullTransport()
    kwargs["messages"] = messages
    try:
        if stream_stage:
            await tport.on_stage("model_call", {"label": stage_label})
        response = await dispatch_openai_compatible_provider(
            runtime=runtime,
            model_def=model_def,
            kwargs=kwargs,
        )
        if stream_stage:
            await tport.on_stage("model_call_complete", {"label": stage_label})
    except Exception as exc:  # pylint: disable=broad-except
        if stream_stage:
            await tport.on_stage("model_call_failed", {"label": stage_label})
        elapsed_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
        log_completion_error(
            exc=exc,
            model_def=model_def,
            response_format_name=response_format_name,
            elapsed_ms=elapsed_ms,
            messages=messages,
            tools=tools,
        )
        raise

    choice = response.choices[0]
    message = choice.message
    raw_reasoning = getattr(message, "reasoning_content", None)
    reasoning_content = raw_reasoning if isinstance(raw_reasoning, str) else ""
    content = _first_message_content(message)
    tool_calls = _extract_completion_tool_calls(getattr(message, "tool_calls", None))
    finish_reason = getattr(choice, "finish_reason", None)
    (
        prompt_tokens,
        completion_tokens,
        total_tokens,
        cached_tokens,
        reasoning_tokens,
    ) = extract_usage_counts(getattr(response, "usage", None))

    content = _normalize_structured_content(content, response_format)

    requested_tokens = int(kwargs.get("max_completion_tokens") or kwargs.get("max_tokens") or 0)
    truncated = derive_openai_compatible_truncated(
        finish_reason=finish_reason,
        completion_tokens=completion_tokens,
        requested_tokens=requested_tokens,
    )

    elapsed_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
    result = StubCompletion(
        content=content,
        tool_calls=tool_calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
        duration_ms=elapsed_ms,
        finish_reason=finish_reason,
        truncated=truncated,
        reasoning_content=reasoning_content,
    )
    log_completion_response(
        model_def=model_def,
        response_format_name=response_format_name,
        elapsed_ms=elapsed_ms,
        result=result,
    )
    await publish_completion_usage(
        runtime=runtime,
        model_def=model_def,
        result=result,
        elapsed_ms=elapsed_ms,
        call_site="complete_live.openai_compatible",
    )
    return result


# ── Streaming completion dispatch ────────────────────────────────────


async def complete_live_stream(
    *,
    runtime: HostRuntime,
    transport: TransportCallbacks | None = None,
    messages: list[dict[str, Any]],
    model_def: StubModelDefinition,
    max_tokens: int,
    temperature: float,
    tools: list[dict[str, Any]] | None = None,
    response_format: type[Any] | None = None,
) -> Any:
    """Execute a streaming provider completion, emitting text chunks to the transport.

    This is the streaming counterpart of :func:`complete_live`. It is wired
    as ``TurnHooks.complete_stream`` when ``config.streaming.mode`` is
    ``"text"`` or ``"hybrid"``. The turn runner iterates the async generator
    this function returns, yielding :class:`kernel.provider_core.StreamedChunk`
    objects; this function calls ``transport.on_text_chunk`` for each text
    delta so the host surface renders tokens as they arrive.

    Stage callbacks (``on_stage``) fire only when ``mode`` is ``"hybrid"``
    (or ``"stage"``, though ``stage`` alone does not wire streaming). When
    the stream's ``finish_reason`` is ``"tool_calls"``, the caller (turn
    runner) falls back to the non-streaming ``complete()`` for that
    iteration — this function does not parse tool-call deltas (v1).

    ``char_delay_ms`` (from ``config.streaming.char_delay_ms``) introduces
    an optional per-chunk asyncio sleep so terminal streaming can be paced
    for readability. ``0`` means no delay.
    """

    tport = transport or NullTransport()
    streaming_cfg = runtime.config.streaming
    stream_stage = streaming_cfg.enabled and streaming_cfg.mode in {"stage", "hybrid"}
    char_delay_ms = streaming_cfg.char_delay_ms
    stage_label = f"{model_def.provider}:{model_def.model_id}"
    started_at = time.perf_counter()
    response_format_name = completion_response_format_name(response_format)

    log_completion_request(
        messages=messages,
        model_def=model_def,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format_name=response_format_name,
        tools=tools,
    )

    if stream_stage:
        await tport.on_stage("model_call", {"label": stage_label})

    final_usage: Any = None
    try:
        if model_def.provider == "anthropic":
            async for chunk in _stream_anthropic_chunks(
                runtime=runtime,
                messages=messages,
                model_def=model_def,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
            ):
                if chunk.usage is not None:
                    final_usage = chunk.usage
                if chunk.content or chunk.reasoning_content:
                    await tport.on_text_chunk(chunk.content, reasoning=chunk.reasoning_content)
                    if char_delay_ms > 0:
                        await asyncio.sleep(char_delay_ms / 1000.0)
                yield chunk
        else:
            async for chunk in _stream_openai_compatible_chunks(
                runtime=runtime,
                messages=messages,
                model_def=model_def,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
                response_format=response_format,
            ):
                if chunk.usage is not None:
                    final_usage = chunk.usage
                if chunk.content or chunk.reasoning_content:
                    await tport.on_text_chunk(chunk.content, reasoning=chunk.reasoning_content)
                    if char_delay_ms > 0:
                        await asyncio.sleep(char_delay_ms / 1000.0)
                yield chunk
    except Exception as exc:  # pylint: disable=broad-except
        if stream_stage:
            await tport.on_stage("model_call_failed", {"label": stage_label})
        elapsed_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
        log_completion_error(
            exc=exc,
            model_def=model_def,
            response_format_name=response_format_name,
            elapsed_ms=elapsed_ms,
            messages=messages,
            tools=tools,
        )
        raise

    if stream_stage:
        await tport.on_stage("model_call_complete", {"label": stage_label})

    # Publish the streamed call's token usage to the durable telemetry
    # handler. The final ``StreamedChunk`` carries the provider's usage
    # object (a ``CompletionUsage``); without this the streaming path's
    # tokens never reach ``token-usage.jsonl`` and the session count stays
    # zero even though the model was called.
    elapsed_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
    if final_usage is not None:
        streamed_result = StubCompletion(
            content="",
            tool_calls=[],
            prompt_tokens=getattr(final_usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(final_usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(final_usage, "total_tokens", 0) or 0,
            cached_tokens=getattr(final_usage, "cached_tokens", 0) or 0,
            reasoning_tokens=getattr(final_usage, "reasoning_tokens", 0) or 0,
            duration_ms=elapsed_ms,
        )
        await publish_completion_usage(
            runtime=runtime,
            model_def=model_def,
            result=streamed_result,
            elapsed_ms=elapsed_ms,
            call_site="complete_live.stream",
        )


async def _stream_openai_compatible_chunks(
    *,
    runtime: HostRuntime,
    messages: list[dict[str, Any]],
    model_def: StubModelDefinition,
    max_tokens: int,
    temperature: float,
    tools: list[dict[str, Any]] | None,
    response_format: type[Any] | None,
) -> Any:
    """Yield ``StreamedChunk`` objects from an OpenAI-compatible streaming endpoint."""
    from kernel.provider_core import StreamedChunk, extract_openai_completion_usage

    kwargs = build_completion_kwargs(
        model_def=model_def,
        max_tokens=max_tokens,
        temperature=temperature,
        tools=tools,
        response_format=response_format,
    )
    kwargs["messages"] = messages
    kwargs["stream"] = True
    kwargs["stream_options"] = {"include_usage": True}

    stream = await dispatch_openai_compatible_provider(
        runtime=runtime,
        model_def=model_def,
        kwargs=kwargs,
    )
    async for event in stream:
        delta_content = ""
        delta_reasoning = ""
        finish_reason = None
        choices = getattr(event, "choices", None) or []
        if choices:
            choice = choices[0]
            delta = getattr(choice, "delta", None)
            if delta is not None:
                delta_content = getattr(delta, "content", None) or ""
                # gpt-5.x reasoning models emit "thinking" as
                # ``delta.reasoning_content`` (separate from
                # ``delta.content``). Capture it so the aggregated
                # completion can surface it as a fallback when the
                # visible answer is empty.
                delta_reasoning = getattr(delta, "reasoning_content", None) or ""
            finish_reason = getattr(choice, "finish_reason", None)
        usage = extract_openai_completion_usage(event)
        yield StreamedChunk(
            content=delta_content,
            reasoning_content=delta_reasoning,
            finish_reason=finish_reason,
            usage=usage,
        )


async def _stream_anthropic_chunks(
    *,
    runtime: HostRuntime,
    messages: list[dict[str, Any]],
    model_def: StubModelDefinition,
    max_tokens: int,
    temperature: float,
    tools: list[dict[str, Any]] | None,
) -> Any:
    """Yield ``StreamedChunk`` objects from the Anthropic Messages SSE stream."""
    from kernel.provider_core import StreamedChunk

    api_key = (
        os.getenv(LISA_ANTHROPIC_API_KEY_ENV, "").strip()
        or os.getenv("ANTHROPIC_API_KEY", "").strip()
    )
    if not api_key:
        raise ValueError(
            f"Anthropic is not configured. Set {LISA_ANTHROPIC_API_KEY_ENV} (or ANTHROPIC_API_KEY)."
        )

    system_prompt, anthropic_messages = _to_anthropic_payload_messages(messages)
    payload: dict[str, Any] = {
        "model": model_def.model_id,
        "max_tokens": max_tokens,
        "messages": anthropic_messages,
        "stream": True,
    }
    if temperature >= 0:
        payload["temperature"] = temperature
    if system_prompt:
        payload["system"] = system_prompt
    anthropic_tools, _name_map = _to_anthropic_tools(tools)
    if anthropic_tools:
        payload["tools"] = anthropic_tools

    headers = {
        "x-api-key": api_key,
        "anthropic-version": os.getenv(LISA_ANTHROPIC_VERSION_ENV, "").strip() or "2023-06-01",
        "accept": "text/event-stream",
    }

    client = anthropic_http_client_instance(runtime)
    async with client.stream("POST", "/v1/messages", headers=headers, json=payload) as response:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = _extract_anthropic_error_detail(exc.response)
            raise httpx.HTTPStatusError(
                f"Anthropic stream failed ({exc.response.status_code}) for model "
                f"'{model_def.model_id}': {detail}",
                request=exc.request,
                response=exc.response,
            ) from exc

        current_event: str = ""
        async for line in response.aiter_lines():
            if not line:
                continue
            if line.startswith("event:"):
                current_event = line[len("event:") :].strip()
                continue
            if not line.startswith("data:"):
                continue
            data_str = line[len("data:") :].strip()
            if not data_str:
                continue
            try:
                import json as _json

                data = _json.loads(data_str)
            except ValueError:
                continue
            event_type = data.get("type") or current_event
            if event_type == "content_block_delta":
                delta = data.get("delta") or {}
                delta_type = delta.get("type")
                if delta_type == "text_delta":
                    text = delta.get("text") or ""
                    if text:
                        yield StreamedChunk(content=text)
                elif delta_type == "thinking_delta":
                    # Extended-thinking models stream their reasoning as
                    # ``thinking_delta`` blocks (separate from ``text_delta``).
                    # Surface them as ``reasoning_content`` so the host transport
                    # renders the thinking live — without this, streamed thinking
                    # is silently dropped and the user never sees it.
                    thinking = delta.get("thinking") or ""
                    if thinking:
                        yield StreamedChunk(reasoning_content=thinking)
            elif event_type == "message_delta":
                delta = data.get("delta") or {}
                stop_reason = delta.get("stop_reason")
                if stop_reason:
                    finish = "tool_calls" if stop_reason == "tool_use" else "stop"
                    usage_data = data.get("usage")
                    usage = None
                    if usage_data:
                        usage = _make_completion_usage(usage_data)
                    yield StreamedChunk(finish_reason=finish, usage=usage)
            elif event_type == "message_stop":
                yield StreamedChunk(finish_reason="stop")


def _make_completion_usage(usage_data: dict[str, Any]) -> Any:
    """Build a ``CompletionUsage`` from an Anthropic usage dict (streaming)."""
    from kernel.provider_core import CompletionUsage

    prompt_tokens = int(usage_data.get("input_tokens", 0) or 0)
    completion_tokens = int(usage_data.get("output_tokens", 0) or 0)
    total = prompt_tokens + completion_tokens
    return CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total,
    )

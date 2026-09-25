"""Anthropic provider gateway.

Implements the kernel's ``ProviderGateway`` protocol for Anthropic Messages
API completions. Uses ``httpx`` for the REST API; no ``anthropic`` SDK
dependency.
"""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import AsyncIterator, Callable
from typing import Any

from kernel.provider_contracts import (
    CompletionRequest,
    CompletionResult,
    CompletionUsage,
    ProviderEventEmitter,
    ProviderStreamEvent,
    StreamedChunk,
    extract_request_id,
    provider_gateway_error_from_exception,
)
from kernel.token_usage import coerce_non_negative_int
from sdk.providers.openai_gateway import (
    _normalize_openai_tool_arguments,
    normalize_structured_content,
)


def _derive_anthropic_truncated(stop_reason: str | None) -> bool | None:
    """Derive the authoritative ``truncated`` flag from Anthropic's stop_reason.

    ``max_tokens`` / ``max_sequence`` → ``True`` (hit a limit).
    ``end_turn`` / ``stop_sequence`` / ``tool_use`` → ``False`` (clean stop).
    Anything else (including ``None``) → ``None`` (cannot tell).
    """

    if stop_reason in {"max_tokens", "max_sequence"}:
        return True
    if stop_reason in {"end_turn", "stop_sequence", "tool_use"}:
        return False
    return None


def extract_anthropic_usage(response: Any) -> CompletionUsage | None:
    """Extract normalized usage from one Anthropic Messages response.

    Anthropic uses ``input_tokens`` / ``output_tokens`` instead of
    ``prompt_tokens`` / ``completion_tokens``. Cache hits surface under
    ``cache_creation_input_tokens`` and ``cache_read_input_tokens``.
    """

    raw_usage = getattr(response, "usage", None)
    if raw_usage is None and isinstance(response, dict):
        raw_usage = response.get("usage")
    if raw_usage is None:
        return None
    prompt_tokens = coerce_non_negative_int(
        getattr(raw_usage, "input_tokens", None)
        if not isinstance(raw_usage, dict)
        else raw_usage.get("input_tokens")
    )
    completion_tokens = coerce_non_negative_int(
        getattr(raw_usage, "output_tokens", None)
        if not isinstance(raw_usage, dict)
        else raw_usage.get("output_tokens")
    )
    cache_creation = coerce_non_negative_int(
        getattr(raw_usage, "cache_creation_input_tokens", None)
        if not isinstance(raw_usage, dict)
        else raw_usage.get("cache_creation_input_tokens")
    )
    cache_read = coerce_non_negative_int(
        getattr(raw_usage, "cache_read_input_tokens", None)
        if not isinstance(raw_usage, dict)
        else raw_usage.get("cache_read_input_tokens")
    )
    all_zero = (
        prompt_tokens == 0 and completion_tokens == 0 and cache_creation == 0 and cache_read == 0
    )
    usage_keys_present = isinstance(raw_usage, dict) and any(
        raw_usage.get(key) is not None
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        )
    )
    # All counters zero/empty with no usage keys on the wire (some partial
    # events report nothing) — return None so an empty ``usage`` does not
    # silently fabricate a 0/0 completion record.
    if all_zero and not usage_keys_present:
        return None
    return CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cached_tokens=cache_read,
        # ``cache_creation_input_tokens`` reports tokens *written into* the
        # prompt cache at the write rate. It is NOT reasoning output, so it
        # must not be folded into ``reasoning_tokens`` (the pre-split
        # conflation); it rides the dedicated counter instead.
        cache_creation_tokens=cache_creation,
    )


def _build_anthropic_system_blocks(
    system_text: str,
    segments: tuple[Any, ...],
    *,
    cache_enabled: bool,
) -> str | list[dict[str, Any]]:
    """Build the ``system`` parameter for one Anthropic Messages request.

    When ``segments`` carries static/dynamic typing (the kernel's typed
    prompt-segment contract) and ``cache_enabled`` is on, the stable prefix
    is emitted as content blocks with an ephemeral ``cache_control`` marker
    on the final stable block so Anthropic caches up to the dynamic tail.
    Dynamic segments (routing hint, per-iteration lifecycle briefing) are
    appended after the breakpoint without cache control.

    When caching is disabled or no segments were supplied, fall back to the
    legacy single-string ``system`` parameter — byte-identical behavior to
    the pre-segmentation gateway.
    """
    if not segments:
        return system_text
    # The segment-flattened string is the canonical text whenever segments
    # are present; it equals ``system_text`` when the caller also supplied
    # the same content as system-role messages but is self-sufficient when
    # they did not (the segment carrier IS the prompt source of truth).
    flattened = "\n\n".join(
        str(getattr(segment, "text", "") or "")
        for segment in segments
        if str(getattr(segment, "text", "") or "").strip()
    )
    blocks: list[dict[str, Any]] = []
    last_stable_index = -1
    for segment in segments:
        text = str(getattr(segment, "text", "") or "")
        if not text.strip():
            continue
        if getattr(segment, "is_stable", True):
            last_stable_index = len(blocks)
        blocks.append({"type": "text", "text": text})
    if not blocks:
        return system_text
    if cache_enabled and last_stable_index >= 0:
        blocks[last_stable_index]["cache_control"] = {"type": "ephemeral"}
        return blocks
    # Cache disabled, or no stable anchor: the wire shape falls back to the
    # single ``system`` string — byte-identical to the pre-segmentation
    # gateway — so billing/cache behavior is exactly preserved on rollback.
    return flattened


def _openai_tool_to_anthropic_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Translate one OpenAI-format tool definition to Anthropic tool format.

    The turn runner produces tools via ``mcp_tool_to_openai_function``::

        {"type": "function", "function": {"name": ..., "description": ...,
        "parameters": <JSON-Schema>}}

    Anthropic expects::

        {"name": ..., "description": ..., "input_schema": <JSON-Schema>}

    Anthropic tool names must match ``^[a-zA-Z0-9_-]{1,128}$``. The kernel's
    namespaced tool ids use a ``.`` separator (``server_id.tool_name``) which
    violates this pattern, so the name is sanitized via
    :func:`_sanitize_anthropic_tool_name` before being sent. Callers that
    need to recover the original name from a response ``tool_use`` block
    must use the reverse map built by
    :func:`_build_anthropic_tools_with_name_map`.
    """

    function_payload: dict[str, Any] = (
        tool.get("function") if isinstance(tool.get("function"), dict) else tool
    )
    raw_name = str(function_payload.get("name", ""))
    return {
        "name": _sanitize_anthropic_tool_name(raw_name),
        "description": str(function_payload.get("description", "")),
        "input_schema": function_payload.get("parameters") or {"type": "object", "properties": {}},
    }


# Anthropic rejects tool names that don't match this pattern. The kernel's
# namespaced tool ids use a ``.`` separator (and pack-contributed server ids
# can contain dots too), so any character outside the allowed set is replaced
# with ``_``. The transformation is made reversible by pairing it with a
# per-request name map (see ``_build_anthropic_tools_with_name_map``).
_ANTHROPIC_TOOL_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")

#: Anthropic tool-name length bound (longer names are truncated).
_ANTHROPIC_TOOL_NAME_MAX_CHARS = 128


def _sanitize_anthropic_tool_name(name: str) -> str:
    """Return a tool name Anthropic will accept (``^[a-zA-Z0-9_-]{1,128}$``).

    Replaces every character outside ``[a-zA-Z0-9_-]`` (notably the ``.``
    used by the kernel's namespaced tool ids) with ``_``. Already-valid names
    pass through unchanged, so bare synthetic tool names like ``memory.add``
    become ``memory_add`` while plain names like ``get_weather`` are
    unaffected. Truncates to 128 characters to satisfy the length bound.
    """
    sanitized = _ANTHROPIC_TOOL_NAME_RE.sub("_", name)
    if len(sanitized) > _ANTHROPIC_TOOL_NAME_MAX_CHARS:
        sanitized = sanitized[:_ANTHROPIC_TOOL_NAME_MAX_CHARS]
    return sanitized or "_"


def _build_anthropic_tools_with_name_map(
    tools: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Translate OpenAI tools to Anthropic format and capture a name map.

    Returns a tuple of ``(anthropic_tools, sanitized_to_original)`` where the
    map lets response parsing restore the original (namespaced) tool name
    from the sanitized name Anthropic echoes back in ``tool_use`` blocks.
    """
    anthropic_tools: list[dict[str, Any]] = []
    sanitized_to_original: dict[str, str] = {}
    for tool in tools:
        anthropic_tool = _openai_tool_to_anthropic_tool(tool)
        sanitized_name = anthropic_tool["name"]
        function_payload = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        original_name = str(function_payload.get("name", ""))
        # First-seen-wins so a duplicate sanitized name maps back to the
        # first original; the kernel's catalog is first-registered-wins too.
        sanitized_to_original.setdefault(sanitized_name, original_name)
        anthropic_tools.append(anthropic_tool)
    return anthropic_tools, sanitized_to_original


def _build_anthropic_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate OpenAI-style chat messages to Anthropic messages format.

    Handles three conversions:
    - ``tool``-role messages → Anthropic ``user`` messages with ``tool_result``
      content blocks.
    - ``assistant`` messages with ``tool_calls`` → Anthropic ``assistant``
      messages with ``text`` + ``tool_use`` content blocks.
    - All other messages pass through with ``role`` and ``content`` preserved.
    """

    anthropic_messages: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role", "user")
        if role == "tool":
            tool_result_content: list[dict[str, Any]] = [
                {
                    "type": "tool_result",
                    "tool_use_id": str(message.get("tool_call_id", "")),
                    "content": str(message.get("content", "")),
                }
            ]
            anthropic_messages.append({"role": "user", "content": tool_result_content})
            continue
        if role == "assistant" and message.get("tool_calls"):
            content_blocks: list[dict[str, Any]] = []
            text = message.get("content")
            if text:
                content_blocks.append({"type": "text", "text": str(text)})
            for tool_call in message["tool_calls"]:
                function_payload = (
                    tool_call.get("function")
                    if isinstance(tool_call.get("function"), dict)
                    else tool_call
                )
                raw_input = function_payload.get("arguments", "{}")
                if isinstance(raw_input, str):
                    try:
                        parsed_input = json.loads(raw_input)
                    except json.JSONDecodeError:
                        parsed_input = {}
                elif isinstance(raw_input, dict):
                    parsed_input = raw_input
                else:
                    parsed_input = {}
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": str(tool_call.get("id", "")),
                        "name": str(function_payload.get("name", "")),
                        "input": parsed_input,
                    }
                )
            anthropic_messages.append({"role": "assistant", "content": content_blocks})
            continue
        anthropic_messages.append({"role": role, "content": message.get("content", "")})
    return anthropic_messages


class AnthropicGateway:
    """Gateway for Anthropic message completions."""

    def __init__(
        self,
        client_factory: Callable[[], Any],
        *,
        api_key: str = "",
        api_version: str = "2023-06-01",
        event_emitter: ProviderEventEmitter | None = None,
        prompt_cache_enabled: bool = True,
    ) -> None:
        """Construct the gateway.

        ``client_factory`` returns a raw ``httpx.AsyncClient`` pointed at the
        Anthropic base URL (the host's runtime-cached client). Auth is
        injected per-request by this gateway — ``api_key`` (``x-api-key``) and
        ``api_version`` (``anthropic-version``) are NOT expected to be baked
        into the client's default headers.

        ``prompt_cache_enabled`` controls whether typed system-prompt segments
        are serialized with an ephemeral ``cache_control`` breakpoint on the
        final stable block. When False, requests fall back to the legacy
        single-string ``system`` parameter — byte-identical to the
        pre-segmentation gateway; the knob exists as the immediate rollback
        for the host's ``LISA_PROMPT_CACHE_ENABLED`` opt-out.
        """
        self._client_factory = client_factory
        self._api_key = api_key.strip()
        self._api_version = api_version.strip() or "2023-06-01"
        self._event_emitter = event_emitter
        self._prompt_cache_enabled = prompt_cache_enabled

    def _headers(self) -> dict[str, str]:
        """Build the per-request auth/version headers."""

        headers: dict[str, str] = {"anthropic-version": self._api_version}
        if self._api_key:
            headers["x-api-key"] = self._api_key
        return headers

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        client = self._client_factory()
        # Legacy merged ALL system messages with blank lines; taking only the
        # first silently dropped the rest of the assembled prompt.
        system_text = "\n\n".join(
            str(message.get("content") or "")
            for message in request.messages
            if message.get("role") == "system" and message.get("content")
        ).strip()
        # Structured-output requests get an explicit JSON-only instruction —
        # Anthropic has no response_format constraint on the wire.
        if request.response_format is not None:
            format_prompt = "Return only a valid JSON object with no Markdown wrapper."
            system_text = f"{system_text}\n\n{format_prompt}".strip()
        anthropic_messages = _build_anthropic_messages(
            [message for message in request.messages if message["role"] != "system"]
        )
        payload: dict[str, Any] = {
            "model": request.model_def.model_id,
            "max_tokens": request.max_tokens,
            "messages": anthropic_messages,
        }
        _system = _build_anthropic_system_blocks(
            system_text,
            getattr(request, "system_prompt_segments", ()) or (),
            cache_enabled=self._prompt_cache_enabled,
        )
        if isinstance(_system, str):
            if _system:
                payload["system"] = _system
        else:
            payload["system"] = _system
        if request.temperature:
            payload["temperature"] = request.temperature
        if request.tools:
            anthropic_tools, name_map = _build_anthropic_tools_with_name_map(request.tools)
            payload["tools"] = anthropic_tools
        else:
            name_map = {}
        if getattr(request.model_def, "thinking_model", False):
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": request.max_tokens,
            }
        await self._emit(
            {
                "type": "provider_call_started",
                "provider": request.model_def.provider,
                "model_id": request.model_def.model_id,
                "display_name": request.model_def.display_name,
            }
        )
        try:
            response = await client.post("/v1/messages", json=payload, headers=self._headers())
            response.raise_for_status()
            response_json = response.json()
        except Exception as exc:  # pylint: disable=broad-except
            await self._emit(
                {
                    "type": "provider_call_failed",
                    "provider": request.model_def.provider,
                    "model_id": request.model_def.model_id,
                    "display_name": request.model_def.display_name,
                    "request_id": extract_request_id(exc),
                    "error": str(exc) or type(exc).__name__,
                }
            )
            raise provider_gateway_error_from_exception(request, exc) from exc

        content_blocks = response_json.get("content") or []
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type") or "")
            if block_type == "text":
                text_parts.append(str(block.get("text") or ""))
            elif block_type == "tool_use":
                sanitized_name = str(block.get("name") or "")
                # Restore the original (namespaced) tool name from the
                # sanitized name Anthropic echoes back. Falls back to the
                # sanitized name if no mapping exists (e.g. the model
                # hallucinated a name we never advertised).
                restored_name = name_map.get(sanitized_name, sanitized_name)
                tool_calls.append(
                    {
                        "id": str(block.get("id") or ""),
                        "type": "function",
                        "name": restored_name,
                        "arguments": _normalize_openai_tool_arguments(block.get("input") or {}),
                    }
                )
        request_id = extract_request_id(response_json) or str(
            response.headers.get("request-id", "") or ""
        )
        await self._emit(
            {
                "type": "provider_call_completed",
                "provider": request.model_def.provider,
                "model_id": request.model_def.model_id,
                "display_name": request.model_def.display_name,
                "request_id": request_id,
            }
        )
        # Anthropic's native stop_reason is the authoritative termination
        # signal; map it to finish_reason and derive the truncation flag.
        stop_reason = str(response_json.get("stop_reason") or "").strip().lower() or None
        return CompletionResult(
            content=normalize_structured_content("\n".join(text_parts), request.response_format),
            provider=request.model_def.provider,
            model_id=request.model_def.model_id,
            display_name=request.model_def.display_name,
            request_id=request_id,
            tool_calls=tool_calls,
            usage=extract_anthropic_usage(response_json),
            finish_reason=stop_reason,
            truncated=_derive_anthropic_truncated(stop_reason),
        )

    async def complete_stream(self, request: CompletionRequest) -> AsyncIterator[StreamedChunk]:
        """Stream completion chunks from the Anthropic Messages API.

        Uses raw ``httpx`` SSE streaming against ``/v1/messages`` (matching
        the host's runtime-cached client). Aggregates:

        - ``content_block_delta`` / ``text_delta`` → visible content chunks.
        - ``content_block_delta`` / ``thinking_delta`` → reasoning chunks
          (thinking models put their "thinking" here, separate from the
          answer).
        - ``message_delta`` → authoritative ``stop_reason``, mapped to
          ``finish_reason`` (``tool_use`` → ``tool_calls`` so the turn
          runner's tool-call fallback fires).

        Usage is read from the final ``message_stop`` payload when present.
        """

        client = self._client_factory()
        system_text = "\n\n".join(
            str(message.get("content") or "")
            for message in request.messages
            if message.get("role") == "system" and message.get("content")
        ).strip()
        if request.response_format is not None:
            format_prompt = "Return only a valid JSON object with no Markdown wrapper."
            system_text = f"{system_text}\n\n{format_prompt}".strip()
        anthropic_messages = _build_anthropic_messages(
            [message for message in request.messages if message["role"] != "system"]
        )
        payload: dict[str, Any] = {
            "model": request.model_def.model_id,
            "max_tokens": request.max_tokens,
            "messages": anthropic_messages,
            "stream": True,
        }
        _system = _build_anthropic_system_blocks(
            system_text,
            getattr(request, "system_prompt_segments", ()) or (),
            cache_enabled=self._prompt_cache_enabled,
        )
        if isinstance(_system, str):
            if _system:
                payload["system"] = _system
        else:
            payload["system"] = _system
        if request.temperature:
            payload["temperature"] = request.temperature
        if request.tools:
            payload["tools"] = [_openai_tool_to_anthropic_tool(tool) for tool in request.tools]
        if getattr(request.model_def, "thinking_model", False):
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": request.max_tokens,
            }
        await self._emit(
            {
                "type": "provider_call_started",
                "provider": request.model_def.provider,
                "model_id": request.model_def.model_id,
                "display_name": request.model_def.display_name,
            }
        )
        try:
            finish_reason: str | None = None
            stream_usage: CompletionUsage | None = None
            async with client.stream(
                "POST", "/v1/messages", json=payload, headers=self._headers()
            ) as response:
                response.raise_for_status()
                async for raw_line in response.aiter_lines():
                    line = raw_line.strip()
                    if not line.startswith("data:"):
                        continue
                    data_text = line[len("data:") :].strip()
                    if not data_text or data_text == "[DONE]":
                        continue
                    try:
                        event = json.loads(data_text)
                    except ValueError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    event_type = str(event.get("type") or "")
                    delta = event.get("delta") if isinstance(event.get("delta"), dict) else {}
                    if event_type == "content_block_delta":
                        delta_type = str(delta.get("type") or "")
                        if delta_type == "text_delta" and delta.get("text"):
                            yield StreamedChunk(content=str(delta["text"]))
                        elif delta_type == "thinking_delta" and delta.get("thinking"):
                            yield StreamedChunk(reasoning_content=str(delta["thinking"]))
                    elif event_type == "message_delta":
                        # Authoritative stop_reason AND cumulative usage both
                        # arrive here (Anthropic puts ``usage`` in this
                        # event's delta), before message_stop. Map Anthropic's
                        # stop vocabulary to the OpenAI-style one the turn
                        # runner branches on.
                        stop_reason = str(delta.get("stop_reason") or "").strip().lower()
                        if stop_reason:
                            finish_reason = (
                                "tool_calls" if stop_reason == "tool_use" else stop_reason
                            )
                        stream_usage = extract_anthropic_usage({"usage": delta.get("usage")})
                    elif event_type == "message_stop":
                        # Usage may ride message_delta (real Anthropic) or
                        # message_stop (defensive); prefer whichever is set.
                        final_usage = stream_usage or extract_anthropic_usage(event)
                        yield StreamedChunk(
                            finish_reason=finish_reason or "stop",
                            usage=final_usage,
                        )
        except Exception as exc:  # pylint: disable=broad-except
            await self._emit(
                {
                    "type": "provider_call_failed",
                    "provider": request.model_def.provider,
                    "model_id": request.model_def.model_id,
                    "display_name": request.model_def.display_name,
                    "request_id": extract_request_id(exc),
                    "error": str(exc) or type(exc).__name__,
                }
            )
            raise provider_gateway_error_from_exception(request, exc) from exc
        await self._emit(
            {
                "type": "provider_call_completed",
                "provider": request.model_def.provider,
                "model_id": request.model_def.model_id,
                "display_name": request.model_def.display_name,
                "request_id": "",
            }
        )

    async def _emit(self, event: ProviderStreamEvent) -> None:
        if self._event_emitter is None:
            return
        maybe_awaitable = self._event_emitter(event)
        if inspect.isawaitable(maybe_awaitable):
            await maybe_awaitable

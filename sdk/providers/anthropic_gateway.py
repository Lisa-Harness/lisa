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

from sdk.providers.openai_gateway import _normalize_openai_tool_arguments


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
    if prompt_tokens == 0 and completion_tokens == 0:
        return None
    return CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cached_tokens=cache_read,
        reasoning_tokens=cache_creation,
    )


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


def _sanitize_anthropic_tool_name(name: str) -> str:
    """Return a tool name Anthropic will accept (``^[a-zA-Z0-9_-]{1,128}$``).

    Replaces every character outside ``[a-zA-Z0-9_-]`` (notably the ``.``
    used by the kernel's namespaced tool ids) with ``_``. Already-valid names
    pass through unchanged, so bare synthetic tool names like ``memory.add``
    become ``memory_add`` while plain names like ``get_weather`` are
    unaffected. Truncates to 128 characters to satisfy the length bound.
    """
    sanitized = _ANTHROPIC_TOOL_NAME_RE.sub("_", name)
    if len(sanitized) > 128:
        sanitized = sanitized[:128]
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
        event_emitter: ProviderEventEmitter | None = None,
    ) -> None:
        self._client_factory = client_factory
        self._event_emitter = event_emitter

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        client = self._client_factory()
        system_text = next(
            (message["content"] for message in request.messages if message["role"] == "system"),
            "",
        )
        anthropic_messages = _build_anthropic_messages(
            [message for message in request.messages if message["role"] != "system"]
        )
        create_kwargs: dict[str, Any] = {
            "model": request.model_def.model_id,
            "max_tokens": request.max_tokens,
            "messages": anthropic_messages,
        }
        if system_text:
            create_kwargs["system"] = system_text
        if request.temperature:
            create_kwargs["temperature"] = request.temperature
        if request.tools:
            anthropic_tools, name_map = _build_anthropic_tools_with_name_map(request.tools)
            create_kwargs["tools"] = anthropic_tools
        else:
            name_map = {}
        if getattr(request.model_def, "thinking_model", False):
            create_kwargs["thinking"] = {
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
            response = await client.messages.create(**create_kwargs)
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

        content_blocks = getattr(response, "content", [])
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in content_blocks:
            block_type = getattr(block, "type", "")
            if block_type == "text":
                text_parts.append(getattr(block, "text", "") or "")
            elif block_type == "tool_use":
                sanitized_name = getattr(block, "name", "")
                # Restore the original (namespaced) tool name from the
                # sanitized name Anthropic echoes back. Falls back to the
                # sanitized name if no mapping exists (e.g. the model
                # hallucinated a name we never advertised).
                restored_name = name_map.get(sanitized_name, sanitized_name)
                tool_calls.append(
                    {
                        "id": getattr(block, "id", ""),
                        "type": "function",
                        "name": restored_name,
                        "arguments": _normalize_openai_tool_arguments(getattr(block, "input", {})),
                    }
                )
        request_id = extract_request_id(response)
        await self._emit(
            {
                "type": "provider_call_completed",
                "provider": request.model_def.provider,
                "model_id": request.model_def.model_id,
                "display_name": request.model_def.display_name,
                "request_id": request_id,
            }
        )
        return CompletionResult(
            content="\n".join(text_parts),
            provider=request.model_def.provider,
            model_id=request.model_def.model_id,
            display_name=request.model_def.display_name,
            request_id=request_id,
            tool_calls=tool_calls,
            usage=extract_anthropic_usage(response),
        )

    async def complete_stream(self, request: CompletionRequest) -> AsyncIterator[StreamedChunk]:
        """Stream completion chunks from the Anthropic Messages API.

        v1 streams tokens only. Uses ``client.messages.stream()`` which yields
        events; we extract ``text_delta`` events for content deltas and the
        ``message_stop`` event for ``finish_reason``. Usage is read from the
        final message object via ``extract_anthropic_usage``.
        """

        client = self._client_factory()
        system_text = next(
            (message["content"] for message in request.messages if message["role"] == "system"),
            "",
        )
        anthropic_messages = _build_anthropic_messages(
            [message for message in request.messages if message["role"] != "system"]
        )
        create_kwargs: dict[str, Any] = {
            "model": request.model_def.model_id,
            "max_tokens": request.max_tokens,
            "messages": anthropic_messages,
        }
        if system_text:
            create_kwargs["system"] = system_text
        if request.temperature:
            create_kwargs["temperature"] = request.temperature
        if request.tools:
            create_kwargs["tools"] = [
                _openai_tool_to_anthropic_tool(tool) for tool in request.tools
            ]
        if getattr(request.model_def, "thinking_model", False):
            create_kwargs["thinking"] = {
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
            async with client.messages.stream(**create_kwargs) as stream:
                async for event in stream:
                    event_type = getattr(event, "type", "")
                    if event_type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        delta_text = getattr(delta, "text", None) if delta is not None else None
                        if delta_text:
                            yield StreamedChunk(content=delta_text)
                    elif event_type == "message_stop":
                        yield StreamedChunk(finish_reason="stop")
                final_message = await stream.get_final_message()
                usage = extract_anthropic_usage(final_message)
                if usage is not None:
                    yield StreamedChunk(usage=usage)
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
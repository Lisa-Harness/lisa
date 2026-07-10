"""Provider completion helpers for Little Lisa.

Pure, stateless functions for translating messages, extracting tool calls,
parsing usage counts, and normalizing structured content across provider
payload shapes. No runtime-state access, no side effects, no I/O.

The client-instance factories and the live completion dispatch stay in
``main.py`` because they depend on ``_RUNTIME_STATE`` and the console
logger. Those functions import the helpers from this module.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from pydantic import ValidationError


def _extract_anthropic_error_detail(response: httpx.Response) -> str:
    """Extract a concise detail string from an Anthropic error response."""

    try:
        payload = response.json()
    except ValueError:
        response_text = response.text.strip()
        return response_text or "no error payload provided by Anthropic"

    if not isinstance(payload, dict):
        return "unexpected non-object error payload from Anthropic"

    error_payload = payload.get("error")
    if isinstance(error_payload, dict):
        error_type = str(error_payload.get("type") or "").strip()
        error_message = str(error_payload.get("message") or "").strip()
        if error_type and error_message:
            return f"{error_type}: {error_message}"
        if error_message:
            return error_message
        if error_type:
            return error_type

    top_level_message = str(payload.get("message") or "").strip()
    return top_level_message or "no error payload provided by Anthropic"


def _normalize_response_format(response_format: type[Any] | None) -> dict[str, str] | None:
    """Return the OpenAI-compatible response_format dict, or None."""
    if response_format is None:
        return None
    return {"type": "json_object"}


def _extract_completion_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    """Normalize OpenAI-style tool call objects into plain dicts."""
    if not isinstance(tool_calls, list):
        return []
    normalized: list[dict[str, Any]] = []
    for call in tool_calls:
        call_id = str(getattr(call, "id", "") or "")
        call_type = str(getattr(call, "type", "function") or "function")
        function = getattr(call, "function", None)
        function_name = str(getattr(function, "name", "") or "")
        function_arguments = str(getattr(function, "arguments", "") or "")
        entry: dict[str, Any] = {
            "id": call_id,
            "type": call_type,
            "function": {
                "name": function_name,
                "arguments": function_arguments,
            },
        }
        # Gemini thinking models attach thought_signature to each function call;
        # it must be echoed verbatim in the corresponding tool-result message.
        thought_sig = getattr(call, "thought_signature", None) or getattr(
            function, "thought_signature", None
        )
        if thought_sig:
            entry["function"]["thought_signature"] = thought_sig
        normalized.append(entry)
    return normalized


def _extract_usage_counts(
    usage_payload: Any,
    coerce_int: Any,
) -> tuple[int, int, int, int, int]:
    """Extract token and sub-counter usage across provider payload shapes.

    Args:
        usage_payload: The raw usage object from a provider response (dict or
            attribute-based object).
        coerce_int: A callable that coerces values to non-negative ints
            (typically ``token_accounting.coerce_non_negative_int``).

    Returns:
        A 5-tuple ``(prompt_tokens, completion_tokens, total_tokens,
        cached_tokens, reasoning_tokens)``. Sub-counters default to 0 when
        the provider did not surface them.
    """

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    cached_tokens = 0
    reasoning_tokens = 0
    if isinstance(usage_payload, dict):
        prompt_tokens = coerce_int(
            usage_payload.get("prompt_tokens") or usage_payload.get("input_tokens")
        )
        completion_tokens = coerce_int(
            usage_payload.get("completion_tokens") or usage_payload.get("output_tokens")
        )
        total_tokens = coerce_int(usage_payload.get("total_tokens"))
        cached_tokens = coerce_int(
            usage_payload.get("cached_tokens") or usage_payload.get("cache_read_input_tokens")
        )
        reasoning_tokens = coerce_int(
            usage_payload.get("reasoning_tokens")
            or usage_payload.get("cache_creation_input_tokens")
        )
    elif usage_payload is not None:
        prompt_tokens = coerce_int(
            getattr(usage_payload, "prompt_tokens", None)
            or getattr(usage_payload, "input_tokens", None)
        )
        completion_tokens = coerce_int(
            getattr(usage_payload, "completion_tokens", None)
            or getattr(usage_payload, "output_tokens", None)
        )
        total_tokens = coerce_int(getattr(usage_payload, "total_tokens", None))
        cached_tokens = coerce_int(
            getattr(usage_payload, "cached_tokens", None)
            or getattr(usage_payload, "cache_read_input_tokens", None)
        )
        reasoning_tokens = coerce_int(
            getattr(usage_payload, "reasoning_tokens", None)
            or getattr(usage_payload, "cache_creation_input_tokens", None)
        )

    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    return prompt_tokens, completion_tokens, total_tokens, cached_tokens, reasoning_tokens


def _parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    """Parse tool call arguments from string or dict into a dict."""
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        return {}
    text = arguments.strip()
    if not text:
        return {}
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}
    if isinstance(decoded, dict):
        return decoded
    return {"value": decoded}


def _to_anthropic_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Translate OpenAI-style tool schemas to Anthropic tool format."""
    if not tools:
        return []
    translated: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function_payload = tool.get("function")
        if not isinstance(function_payload, dict):
            continue
        name = str(function_payload.get("name") or "").strip()
        if not name:
            continue
        description = str(function_payload.get("description") or "").strip()
        parameters = function_payload.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {"type": "object", "properties": {}}
        translated.append(
            {
                "name": name,
                "description": description,
                "input_schema": parameters,
            }
        )
    return translated


def _translate_tool_result_message(message: dict[str, Any]) -> dict[str, Any]:
    """Translate an OpenAI tool-result message to Anthropic format."""
    tool_call_id = str(message.get("tool_call_id") or "").strip() or "tool_call"
    tool_result_content = str(message.get("content") or "")
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": tool_result_content,
            }
        ],
    }


def _translate_assistant_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate assistant tool_calls from OpenAI format to Anthropic blocks."""
    translated_calls: list[dict[str, Any]] = []
    raw_tool_calls = message.get("tool_calls")
    if not isinstance(raw_tool_calls, list):
        return translated_calls
    for index, raw_tool_call in enumerate(raw_tool_calls, start=1):
        if not isinstance(raw_tool_call, dict):
            continue
        function_payload = raw_tool_call.get("function")
        if isinstance(function_payload, dict):
            tool_name = str(function_payload.get("name") or "").strip()
            tool_arguments = function_payload.get("arguments")
        else:
            tool_name = str(raw_tool_call.get("name") or "").strip()
            tool_arguments = raw_tool_call.get("arguments")
        if not tool_name:
            continue
        tool_id = str(raw_tool_call.get("id") or "").strip() or f"tool_call_{index}"
        translated_calls.append(
            {
                "type": "tool_use",
                "id": tool_id,
                "name": tool_name,
                "input": _parse_tool_arguments(tool_arguments),
            }
        )
    return translated_calls


def _translate_chat_message(message: dict[str, Any], role: str) -> dict[str, Any]:
    """Translate a user/assistant chat message to Anthropic content-block format."""
    blocks: list[dict[str, Any]] = []
    text_content = str(message.get("content") or "")
    if text_content:
        blocks.append({"type": "text", "text": text_content})
    if role == "assistant":
        blocks.extend(_translate_assistant_tool_calls(message))
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    return {"role": role, "content": blocks}


def _to_anthropic_payload_messages(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert OpenAI-style messages to Anthropic (system, messages) format.

    Flattened from nested closures into module-level helpers for testability.
    """
    system_parts: list[str] = []
    translated_messages: list[dict[str, Any]] = []

    for message in messages:
        role = str(message.get("role") or "").strip().lower()
        if role == "system":
            text = str(message.get("content") or "").strip()
            if text:
                system_parts.append(text)
            continue

        if role == "tool":
            translated_messages.append(_translate_tool_result_message(message))
            continue

        if role not in {"user", "assistant"}:
            continue

        translated_messages.append(_translate_chat_message(message, role))

    if not translated_messages:
        translated_messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": ""}],
            }
        )

    system_prompt = "\n\n".join(system_parts).strip()
    return (system_prompt or None, translated_messages)


def _extract_anthropic_text(content_blocks: Any) -> str:
    """Extract concatenated text from Anthropic content blocks."""
    if not isinstance(content_blocks, list):
        return ""
    text_parts: list[str] = []
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        if str(block.get("type") or "") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str) and text:
            text_parts.append(text)
    return "\n".join(text_parts).strip()


def _extract_anthropic_tool_calls(content_blocks: Any) -> list[dict[str, Any]]:
    """Extract tool_use blocks from Anthropic content as OpenAI-style dicts."""
    if not isinstance(content_blocks, list):
        return []
    calls: list[dict[str, Any]] = []
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        if str(block.get("type") or "") != "tool_use":
            continue
        calls.append(
            {
                "id": str(block.get("id") or ""),
                "type": "function",
                "function": {
                    "name": str(block.get("name") or ""),
                    "arguments": json.dumps(block.get("input") or {}),
                },
            }
        )
    return calls


def _first_message_content(message: Any) -> str:
    """Extract text content from an OpenAI chat completion message."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            item_text = getattr(item, "text", "")
            if item_text:
                text_parts.append(str(item_text))
        return "\n".join(text_parts).strip()
    return str(content or "")


def _extract_json_object(text: str) -> str:
    """Extract the first likely JSON object from a model response."""

    normalized = str(text or "").strip()
    if normalized.startswith("```json"):
        normalized = normalized[7:]
    elif normalized.startswith("```"):
        normalized = normalized[3:]
    if normalized.endswith("```"):
        normalized = normalized[:-3]
    normalized = normalized.strip()

    if normalized.startswith("{") and normalized.endswith("}"):
        return normalized

    start = normalized.find("{")
    end = normalized.rfind("}")
    if start >= 0 and end > start:
        return normalized[start : end + 1]
    return normalized


def _normalize_structured_content(
    content: str,
    response_format: type[Any] | None,
) -> str:
    """Return normalized structured JSON when possible, otherwise original content."""

    if (
        response_format is None
        or not content
        or not hasattr(response_format, "model_validate_json")
    ):
        return content
    try:
        parsed = response_format.model_validate_json(_extract_json_object(content))
    except ValidationError:
        return content
    normalized = parsed.model_dump_json()
    if not isinstance(normalized, str):
        return content
    return normalized


def _normalize_google_base_url(base_url: str, default_url: str) -> str:
    """Return a supported Google OpenAI-compatible base URL.

    Args:
        base_url: The configured base URL (may be empty).
        default_url: The default Google OpenAI-compatible endpoint.
    """
    normalized = base_url.strip()
    if not normalized:
        return default_url

    lowered = normalized.lower()
    if "generativelanguage.googleapis.com" in lowered and "/openai" not in lowered:
        return default_url

    if not normalized.endswith("/"):
        normalized = f"{normalized}/"
    return normalized


def _normalize_ollama_host_to_base_url(host: str) -> str:
    """Normalize an Ollama host string to an OpenAI-compatible base URL."""
    normalized = host.strip().rstrip("/")
    if not normalized:
        return ""
    if not normalized.startswith(("http://", "https://")):
        normalized = f"http://{normalized}"
    if normalized.endswith("/v1"):
        return f"{normalized}/"
    return f"{normalized}/v1/"

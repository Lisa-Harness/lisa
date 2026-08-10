"""OpenAI and Azure OpenAI provider gateways.

Implements the kernel's ``ProviderGateway`` and ``EmbeddingGateway`` protocols
for OpenAI-compatible chat completions, the OpenAI Responses API, and
embeddings. This module depends on the ``openai`` SDK.
"""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import AsyncIterator, Callable
from typing import Any, cast

from openai import AsyncAzureOpenAI, AsyncOpenAI
from pydantic import BaseModel as PydanticBaseModel

from kernel.provider_contracts import (
    ChatMessages,
    CompletionRequest,
    CompletionResult,
    CompletionUsage,
    EmbeddingRequest,
    EmbeddingResult,
    ModelDescriptor,
    ProviderEventEmitter,
    ProviderGatewayError,
    ProviderStreamEvent,
    StreamedChunk,
    embedding_gateway_error_from_exception,
    extract_request_id,
    is_pydantic_model_class,
    provider_gateway_error_from_exception,
)
from kernel.token_usage import coerce_non_negative_int


# ── Message normalization ────────────────────────────────────────────────


def _contiguous_tool_result_ids(messages: ChatMessages) -> set[str]:
    """Return ids of ``role: tool`` results in contiguous post-assistant blocks.

    For each assistant message with ``tool_calls``, collects the
    ``tool_call_id``s of the contiguous ``role: tool`` block immediately
    following it (stopping at the first non-tool message). A result that
    appears after an intervening non-tool message is NOT included — strict
    OpenAI backends reject such non-contiguous pairings.
    """
    contiguous_ids: set[str] = set()
    for index, message in enumerate(messages):
        if str(message.get("role", "") or "") != "assistant":
            continue
        raw_tool_calls = message.get("tool_calls")
        if not isinstance(raw_tool_calls, list) or not raw_tool_calls:
            continue
        for follower in messages[index + 1 :]:
            if str(follower.get("role", "") or "") != "tool":
                break
            contiguous_ids.add(str(follower.get("tool_call_id", "") or ""))
    return contiguous_ids


def _retained_assistant_tool_call_ids(messages: ChatMessages) -> set[str]:
    """Return every ``id`` referenced by a retained assistant ``tool_calls`` list."""
    retained_ids: set[str] = set()
    for message in messages:
        if str(message.get("role", "") or "") != "assistant":
            continue
        raw_tool_calls = message.get("tool_calls")
        if not isinstance(raw_tool_calls, list):
            continue
        for tool_call in raw_tool_calls:
            if isinstance(tool_call, dict):
                retained_ids.add(str(tool_call.get("id", "") or ""))
    return retained_ids


def _enforce_tool_result_contiguity(messages: ChatMessages) -> ChatMessages:
    """Prune assistant ``tool_calls`` whose results are not contiguous.

    Strict OpenAI-compatible backends (Azure OpenAI, OpenAI) require every
    ``role: tool`` result for an assistant's ``tool_calls`` to immediately
    follow that assistant message. A non-tool message (e.g. a ``user``
    nudge the turn loop appends mid-batch after a ``transition_stage``
    handoff) between the assistant ``tool_calls`` and a later tool result
    breaks the pairing and is rejected with ``An assistant message with
    'tool_calls' must be followed by tool messages responding to each
    'tool_call_id'`` (a 400 that kills the whole turn, the 2026-08-06
    ``plan → okf`` incident).

    The ``answered_ids`` prune in :func:`normalize_openai_messages` only
    checks a result exists ANYWHERE; it does not enforce ordering. This
    pass computes the contiguous ``role: tool`` block immediately after
    each assistant message and prunes any retained ``tool_call`` whose id
    is NOT in that block (converting the assistant turn to plain text),
    then drops the now-orphaned trailing tool results so every backend
    accepts the payload regardless of which mid-batch break path left the
    history interleaved.
    """
    contiguous_answered_ids = _contiguous_tool_result_ids(messages)
    if not contiguous_answered_ids:
        return messages
    for message in messages:
        if str(message.get("role", "") or "") != "assistant":
            continue
        raw_tool_calls = message.get("tool_calls")
        if not isinstance(raw_tool_calls, list) or not raw_tool_calls:
            continue
        contiguous_kept = [
            tool_call
            for tool_call in raw_tool_calls
            if not isinstance(tool_call, dict)
            or str(tool_call.get("id", "") or "") in contiguous_answered_ids
        ]
        if contiguous_kept:
            message["tool_calls"] = contiguous_kept
        else:
            message.pop("tool_calls", None)
    # Drop tool results whose id is no longer referenced by any retained
    # assistant tool_call (the calls pruned above). Strict backends reject
    # orphaned tool results too.
    retained_assistant_ids = _retained_assistant_tool_call_ids(messages)
    return [
        message
        for message in messages
        if str(message.get("role", "") or "") != "tool"
        or str(message.get("tool_call_id", "") or "") in retained_assistant_ids
    ]


def normalize_openai_messages(
    messages: ChatMessages,
    *,
    require_thought_signature: bool = False,
) -> ChatMessages:
    """Clone one chat history with OpenAI-compatible tool-call payloads.

    Also guarantees every ``assistant`` message carries a string ``content``.
    The turn runner appends assistant messages with ``tool_calls`` and no
    ``content`` key when the model emits tool calls without accompanying
    text. The OpenAI spec treats ``content`` as optional for those messages,
    but strict OpenAI-compatible backends (notably Ollama) reject
    ``content: <nil>`` with ``invalid message content type: <nil>``. Defaulting
    to ``""`` keeps the payload acceptable to both lenient and strict backends.

    Prunes orphaned ``tool_calls``: an assistant message that declares a
    tool call with no later ``role: tool`` message answering that
    ``tool_call_id`` is malformed. This arises when the turn loop breaks
    mid-tool-loop (forced plan/fork handoff, ``entry_failed``,
    ``plan_failed``, iteration-cap yield) after appending the assistant
    message but before attaching its tool result. Strict OpenAI backends
    reject such history with ``An assistant message with 'tool_calls' must
    be followed by tool messages responding to each 'tool_call_id'`` (a
    400 that previously killed the whole turn). Pruning the orphaned calls
    here, at the single chokepoint every OpenAI-compatible request passes
    through, keeps the payload well-formed regardless of which break path
    left the history dangling.

    When ``require_thought_signature`` is True (Gemini thinking models via
    the OpenAI-compatible endpoint), assistant ``tool_calls`` entries that
    lack a ``thought_signature`` are also pruned. Gemini 2.5 Pro / 3.x
    models think by default and emit a ``thoughtSignature`` on every
    ``functionCall`` part; the signature must be echoed verbatim on the
    next turn. The kernel synthesizes ``transition_stage`` tool calls
    (planned handoff, forced fork, forced plan) with no signature — those
    calls have no model reasoning to sign. Sending them to Gemini yields
    ``400 INVALID_ARGUMENT: Function call is missing a thought_signature in
    functionCall parts``. Stripping the signature-less call (and its
    matching tool result) converts the assistant turn into a plain text
    message, which Gemini accepts. The synthesized handoff's effect is
    already reflected in the system-prompt swap and tool-surface rebuild
    that happened in-loop; the message-history tool call is cosmetic.

    Google/Gemini additionally rejects requests whose final message is an
    ``assistant`` turn: ``400 Requests ending with a model turn are not
    supported``. When the signature-less pruning leaves the history ending
    with an assistant message, a synthetic ``user`` nudge is appended so
    the request ends with a ``user``/``tool`` role as Google requires.
    This nudge is scoped to the Gemini path only; OpenAI/Azure accept a
    trailing assistant message.
    """

    # Collect every tool_call_id that has a matching tool-result message.
    answered_ids = {
        str(message.get("tool_call_id", "") or "")
        for message in messages
        if str(message.get("role", "") or "") == "tool"
    }

    # When Gemini thinking-model round-tripping is required, collect the
    # ids of tool_calls that carry a ``thought_signature`` (top-level or
    # nested under ``function``). Only those survive pruning; calls the
    # kernel synthesized (planned handoff / forced fork / forced plan) have
    # no signature and must be stripped to avoid the 400.
    signed_ids: set[str] = set()
    if require_thought_signature:
        for message in messages:
            if str(message.get("role", "") or "") != "assistant":
                continue
            raw_tool_calls = message.get("tool_calls")
            if not isinstance(raw_tool_calls, list):
                continue
            for tool_call in raw_tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                call_id = str(tool_call.get("id", "") or "")
                if not call_id:
                    continue
                thought_sig = tool_call.get("thought_signature")
                if not thought_sig:
                    fn_payload = tool_call.get("function")
                    if isinstance(fn_payload, dict):
                        thought_sig = fn_payload.get("thought_signature")
                if thought_sig:
                    signed_ids.add(call_id)

    normalized_messages: ChatMessages = []
    for message in messages:
        normalized_message = dict(message)
        if str(normalized_message.get("role", "") or "") == "assistant":
            if normalized_message.get("content") is None:
                normalized_message["content"] = ""
        raw_tool_calls = normalized_message.get("tool_calls")
        if isinstance(raw_tool_calls, list):
            pruned = [
                tool_call
                for tool_call in raw_tool_calls
                if not isinstance(tool_call, dict)
                or str(tool_call.get("id", "") or "") in answered_ids
            ]
            # Gemini thinking models: further prune any tool_call that lacks
            # a ``thought_signature``. The kernel's synthesized
            # ``transition_stage`` calls (planned handoff / forced fork /
            # forced plan) have no model reasoning to sign and would trigger
            # ``400 INVALID_ARGUMENT: Function call is missing a
            # thought_signature``. Drop them and their matching tool
            # results so the history stays well-formed for Gemini.
            if require_thought_signature:
                pruned = [
                    tool_call
                    for tool_call in pruned
                    if not isinstance(tool_call, dict)
                    or str(tool_call.get("id", "") or "") in signed_ids
                ]
            # Only normalize the tool-call payloads we keep; drop the key
            # entirely when nothing remains so the message is a plain text
            # assistant turn (valid) instead of an empty ``tool_calls: []``.
            if pruned:
                normalized_message["tool_calls"] = [
                    _normalize_openai_tool_call(tool_call)
                    if isinstance(tool_call, dict)
                    else tool_call
                    for tool_call in pruned
                ]
            else:
                normalized_message.pop("tool_calls", None)
        normalized_messages.append(normalized_message)

    normalized_messages = _enforce_tool_result_contiguity(normalized_messages)

    # When signature-less tool_calls are pruned, the matching ``role: tool``
    # result messages become orphaned (no assistant tool_call references
    # them). Gemini does not reject orphaned tool results outright, but
    # strict OpenAI backends do, so drop them for parity with the existing
    # orphaned-tool-call pruning. This keeps the payload well-formed for
    # every backend, not just Gemini.
    if require_thought_signature:
        retained_assistant_ids = set()
        for message in normalized_messages:
            if str(message.get("role", "") or "") != "assistant":
                continue
            raw_tool_calls = message.get("tool_calls")
            if not isinstance(raw_tool_calls, list):
                continue
            for tool_call in raw_tool_calls:
                if isinstance(tool_call, dict):
                    retained_assistant_ids.add(str(tool_call.get("id", "") or ""))
        normalized_messages = [
            message
            for message in normalized_messages
            if str(message.get("role", "") or "") != "tool"
            or str(message.get("tool_call_id", "") or "") in retained_assistant_ids
        ]
        # Google/Gemini rejects a request whose last message is an
        # ``assistant`` turn: ``400 Requests ending with a model turn are
        # not supported``. The signature-less synthesized
        # ``transition_stage`` pruning above can convert a trailing
        # assistant ``tool_calls`` message into plain text (dropping its
        # orphaned tool result), leaving the history ending with that
        # assistant message. When that happens, append a synthetic
        # ``user`` nudge so the request ends with a ``user``/``tool`` role
        # as Google requires. This is scoped to the Gemini
        # (``require_thought_signature``) path; OpenAI/Azure accept a
        # trailing assistant message.
        if normalized_messages and str(
            normalized_messages[-1].get("role", "") or ""
        ) == "assistant":
            normalized_messages.append(
                {"role": "user", "content": "Continue."}
            )
    return normalized_messages


# ── Request builders ─────────────────────────────────────────────────────


def build_openai_completion_kwargs(request: CompletionRequest) -> dict[str, Any]:
    """Build provider kwargs for one OpenAI-compatible completion request."""

    kwargs: dict[str, Any] = {
        "model": request.model_def.model_id,
        "messages": cast(Any, normalize_openai_messages(request.messages)),
    }
    if request.model_def.reasoning_model:
        kwargs["max_completion_tokens"] = request.max_tokens
    elif request.model_def.thinking_model:
        kwargs["max_tokens"] = request.max_tokens
    else:
        kwargs["max_tokens"] = request.max_tokens
        kwargs["temperature"] = request.temperature
    if request.tools:
        kwargs["tools"] = request.tools
    if request.response_format:
        kwargs["response_format"] = request.response_format
    return kwargs


def build_openai_responses_kwargs(request: CompletionRequest) -> dict[str, Any]:
    """Build provider kwargs for one OpenAI Responses API request."""

    kwargs: dict[str, Any] = {
        "model": request.model_def.model_id,
        "input": build_openai_responses_input(request.messages),
        "max_output_tokens": request.max_tokens,
    }
    if not request.model_def.reasoning_model and not request.model_def.thinking_model:
        kwargs["temperature"] = request.temperature
    if request.tools:
        kwargs["tools"] = [_normalize_openai_response_tool(tool) for tool in request.tools]
    if request.response_format:
        kwargs["text"] = {"format": _normalize_openai_response_format(request.response_format)}
    return kwargs


# The Responses API rejects tool/function names that don't match
# ``^[a-zA-Z0-9_-]+$``. Lisa's tool names use dots as namespace separators
# (e.g. ``lisa-storage.read_file``, ``coding.read_file``); the chat-completions
# endpoint accepts them, but the Responses endpoint returns 400
# ``Invalid 'input[N].name': string does not match pattern``. Sanitize by
# replacing every disallowed character with ``_`` so the name is always valid
# while staying deterministic and reversible for the tool dispatcher (which
# matches on the sanitized name the model emits).
_RESPONSES_NAME_INVALID = re.compile(r"[^a-zA-Z0-9_-]")


def _sanitize_responses_tool_name(name: str) -> str:
    """Replace characters the Responses API rejects in function names."""

    return _RESPONSES_NAME_INVALID.sub("_", name)


def build_responses_tool_name_map(
    tools: list[dict[str, Any]] | None,
) -> dict[str, str]:
    """Build a ``sanitized_name -> original_name`` reverse map.

    The Responses API requires function names matching ``^[a-zA-Z0-9_-]+$``;
    Lisa's dotted names (e.g. ``lisa-storage.read_file``) are sanitized to
    ``lisa-storage_read_file`` before being sent. The model returns the
    sanitized form; callers use this map to restore the original name so
    the kernel's tool dispatcher can match it against its dotted-name
    catalogs.
    """

    name_map: dict[str, str] = {}
    if not tools:
        return name_map
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        function_payload = tool.get("function")
        if not isinstance(function_payload, dict):
            continue
        original = str(function_payload.get("name", "") or "")
        if not original:
            continue
        sanitized = _sanitize_responses_tool_name(original)
        if sanitized != original:
            name_map[sanitized] = original
    return name_map


def restore_responses_tool_names(
    tool_calls: list[dict[str, Any]],
    name_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Restore original tool names in extracted Responses API tool calls."""

    if not name_map:
        return tool_calls
    for call in tool_calls:
        sanitized = str(call.get("name", "") or "")
        if sanitized in name_map:
            call["name"] = name_map[sanitized]
    return tool_calls


def build_openai_responses_input(messages: ChatMessages) -> list[dict[str, Any]]:
    """Translate chat-completions history into Responses API input items."""

    normalized_messages = normalize_openai_messages(messages)
    input_items: list[dict[str, Any]] = []
    for message in normalized_messages:
        role = str(message.get("role", "") or "")
        content = message.get("content", "")
        if role in {"system", "developer", "user"}:
            input_items.append(
                {
                    "type": "message",
                    "role": role,
                    "content": "" if content is None else content,
                }
            )
            continue
        if role == "assistant":
            if content not in (None, "", []):
                input_items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": content,
                    }
                )
            raw_tool_calls = message.get("tool_calls")
            if isinstance(raw_tool_calls, list):
                for raw_tool_call in raw_tool_calls:
                    if not isinstance(raw_tool_call, dict):
                        continue
                    function_payload = raw_tool_call.get("function")
                    if not isinstance(function_payload, dict):
                        continue
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": str(raw_tool_call.get("id", "") or ""),
                            "name": _sanitize_responses_tool_name(
                                str(function_payload.get("name", "") or "")
                            ),
                            "arguments": _normalize_openai_tool_arguments(
                                function_payload.get("arguments", "")
                            ),
                            "status": "completed",
                        }
                    )
            continue
        if role == "tool":
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": str(message.get("tool_call_id", "") or ""),
                    "output": "" if content is None else content,
                    "status": "completed",
                }
            )
    return input_items


def build_openai_embedding_kwargs(request: EmbeddingRequest) -> dict[str, Any]:
    """Build provider kwargs for one OpenAI-style embeddings request."""

    kwargs: dict[str, Any] = {
        "model": request.model_id,
        "input": request.input_texts,
    }
    if request.dimensions is not None:
        kwargs["dimensions"] = request.dimensions
    return kwargs


# ── Response extractors ──────────────────────────────────────────────────


def extract_openai_content(response: Any) -> str:
    """Extract text or parsed JSON content from one OpenAI-compatible response.

    For gpt-5.x reasoning models the model's substantive output can land in
    ``message.reasoning_content`` while ``message.content`` is empty or
    terse. Fall back to ``reasoning_content`` so the caller sees the model's
    actual answer instead of an empty string (which the turn runner would
    otherwise turn into "The model did not produce a final response.").
    """

    choices = getattr(response, "choices", [])
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", "")
    if content:
        return content
    parsed = getattr(message, "parsed", None)
    if parsed is not None:
        if isinstance(parsed, PydanticBaseModel):
            return parsed.model_dump_json()
        try:
            return json.dumps(parsed)
        except Exception:  # pylint: disable=broad-except
            return str(parsed)
    # Visible answer empty — fall back to reasoning content for reasoning
    # models. Empty string if neither is present.
    reasoning = getattr(message, "reasoning_content", None)
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    return ""


def extract_responses_content(response: Any) -> str:
    """Extract text content from one OpenAI Responses API response."""

    output_text = getattr(response, "output_text", "")
    if isinstance(output_text, str) and output_text:
        return output_text

    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", "") != "message":
            continue
        for part in getattr(item, "content", []) or []:
            text = getattr(part, "text", "")
            if isinstance(text, str) and text:
                chunks.append(text)
    return "\n".join(chunks)


def extract_openai_tool_calls(response: Any) -> list[dict[str, Any]]:
    """Extract OpenAI-style function tool calls from the first response choice."""

    choices = getattr(response, "choices", [])
    if not choices:
        return []
    message = getattr(choices[0], "message", None)
    raw_calls = getattr(message, "tool_calls", None) or []
    extracted: list[dict[str, Any]] = []
    for call in raw_calls:
        function_obj = getattr(call, "function", None)
        entry: dict[str, Any] = {
            "id": getattr(call, "id", ""),
            "type": getattr(call, "type", ""),
            "name": getattr(function_obj, "name", "") if function_obj else "",
            "arguments": getattr(function_obj, "arguments", "") if function_obj else "",
        }
        thought_sig = getattr(call, "thought_signature", None) or getattr(
            function_obj, "thought_signature", None
        )
        if thought_sig:
            entry["thought_signature"] = thought_sig
        extracted.append(entry)
    return extracted


def extract_responses_tool_calls(response: Any) -> list[dict[str, Any]]:
    """Extract Responses API function tool calls into Lisa's normalized shape."""

    extracted: list[dict[str, Any]] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", "") != "function_call":
            continue
        extracted.append(
            {
                "id": str(getattr(item, "id", "") or getattr(item, "call_id", "") or ""),
                "type": "function",
                "name": str(getattr(item, "name", "") or ""),
                "arguments": _normalize_openai_tool_arguments(getattr(item, "arguments", "")),
            }
        )
    return extracted


def extract_openai_embeddings(response: Any) -> list[list[float]]:
    """Extract embedding vectors from one OpenAI-style embeddings response."""

    vectors: list[list[float]] = []
    for item in getattr(response, "data", []) or []:
        raw_embedding = getattr(item, "embedding", None)
        if raw_embedding is None:
            continue
        vectors.append([float(value) for value in raw_embedding])
    return vectors


def extract_openai_completion_usage(response: Any) -> CompletionUsage | None:
    """Extract normalized per-call usage from one OpenAI-style chat response.

    Returns None when the response has no usage object (e.g. some preview
    endpoints or explicitly disabled usage). All four known sub-counters
    are read defensively so a missing field never raises.
    """

    raw_usage = getattr(response, "usage", None)
    if raw_usage is None and isinstance(response, dict):
        raw_usage = response.get("usage")
    if raw_usage is None:
        return None
    prompt_tokens = coerce_non_negative_int(
        getattr(raw_usage, "prompt_tokens", None)
        if not isinstance(raw_usage, dict)
        else raw_usage.get("prompt_tokens")
    )
    completion_tokens = coerce_non_negative_int(
        getattr(raw_usage, "completion_tokens", None)
        if not isinstance(raw_usage, dict)
        else raw_usage.get("completion_tokens")
    )
    total_tokens = coerce_non_negative_int(
        getattr(raw_usage, "total_tokens", None)
        if not isinstance(raw_usage, dict)
        else raw_usage.get("total_tokens")
    )
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens

    prompt_details = (
        getattr(raw_usage, "prompt_tokens_details", None)
        if not isinstance(raw_usage, dict)
        else raw_usage.get("prompt_tokens_details")
    )
    cached_tokens = 0
    if prompt_details is not None:
        cached_tokens = coerce_non_negative_int(
            getattr(prompt_details, "cached_tokens", None)
            if not isinstance(prompt_details, dict)
            else prompt_details.get("cached_tokens")
        )

    completion_details = (
        getattr(raw_usage, "completion_tokens_details", None)
        if not isinstance(raw_usage, dict)
        else raw_usage.get("completion_tokens_details")
    )
    reasoning_tokens = 0
    if completion_details is not None:
        reasoning_tokens = coerce_non_negative_int(
            getattr(completion_details, "reasoning_tokens", None)
            if not isinstance(completion_details, dict)
            else completion_details.get("reasoning_tokens")
        )

    if prompt_tokens == 0 and completion_tokens == 0 and total_tokens == 0:
        return None
    return CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def extract_responses_usage(response: Any) -> CompletionUsage | None:
    """Extract normalized usage from one OpenAI Responses API response."""

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
    total_tokens = coerce_non_negative_int(
        getattr(raw_usage, "total_tokens", None)
        if not isinstance(raw_usage, dict)
        else raw_usage.get("total_tokens")
    )
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens

    input_details = (
        getattr(raw_usage, "input_tokens_details", None)
        if not isinstance(raw_usage, dict)
        else raw_usage.get("input_tokens_details")
    )
    cached_tokens = 0
    if input_details is not None:
        cached_tokens = coerce_non_negative_int(
            getattr(input_details, "cached_tokens", None)
            if not isinstance(input_details, dict)
            else input_details.get("cached_tokens")
        )

    output_details = (
        getattr(raw_usage, "output_tokens_details", None)
        if not isinstance(raw_usage, dict)
        else raw_usage.get("output_tokens_details")
    )
    reasoning_tokens = 0
    if output_details is not None:
        reasoning_tokens = coerce_non_negative_int(
            getattr(output_details, "reasoning_tokens", None)
            if not isinstance(output_details, dict)
            else output_details.get("reasoning_tokens")
        )

    if prompt_tokens == 0 and completion_tokens == 0 and total_tokens == 0:
        return None
    return CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def extract_openai_embedding_usage(response: Any) -> CompletionUsage | None:
    """Extract normalized usage from one OpenAI-style embeddings response."""

    raw_usage = getattr(response, "usage", None)
    if raw_usage is None and isinstance(response, dict):
        raw_usage = response.get("usage")
    if raw_usage is None:
        return None
    prompt_tokens = coerce_non_negative_int(
        getattr(raw_usage, "prompt_tokens", None)
        if not isinstance(raw_usage, dict)
        else raw_usage.get("prompt_tokens")
    )
    total_tokens = coerce_non_negative_int(
        getattr(raw_usage, "total_tokens", None)
        if not isinstance(raw_usage, dict)
        else raw_usage.get("total_tokens")
    )
    if prompt_tokens == 0 and total_tokens == 0:
        return None
    return CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=0,
        total_tokens=total_tokens or prompt_tokens,
        cached_tokens=0,
        reasoning_tokens=0,
    )


# ── Normalization helpers ────────────────────────────────────────────────


def _normalize_openai_tool_arguments(raw_arguments: Any) -> str:
    if isinstance(raw_arguments, str):
        return raw_arguments
    if raw_arguments is None:
        return ""
    try:
        return json.dumps(raw_arguments)
    except TypeError:
        return str(raw_arguments)


def _normalize_openai_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    function_payload = tool_call.get("function")
    if isinstance(function_payload, dict):
        raw_name = function_payload.get("name", "")
        raw_arguments = function_payload.get("arguments", "")
    else:
        raw_name = tool_call.get("name", "")
        raw_arguments = tool_call.get("arguments", "")
    normalized: dict[str, Any] = {
        "id": tool_call.get("id", ""),
        "type": tool_call.get("type", "function") or "function",
        "function": {
            "name": str(raw_name or ""),
            "arguments": _normalize_openai_tool_arguments(raw_arguments),
        },
    }
    thought_sig = tool_call.get("thought_signature")
    if not thought_sig:
        fn = tool_call.get("function")
        if isinstance(fn, dict):
            thought_sig = fn.get("thought_signature")
    if thought_sig:
        normalized["thought_signature"] = thought_sig
    return normalized


def _normalize_openai_response_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Translate one chat-completions tool definition into Responses API shape."""

    if tool.get("type") != "function":
        return dict(tool)

    function_payload = tool.get("function")
    if not isinstance(function_payload, dict):
        return dict(tool)

    normalized: dict[str, Any] = {
        "type": "function",
        "name": _sanitize_responses_tool_name(
            str(function_payload.get("name", "") or "")
        ),
        "parameters": function_payload.get("parameters", {}),
    }
    description = function_payload.get("description")
    if description:
        normalized["description"] = description
    strict = function_payload.get("strict")
    if strict is not None:
        normalized["strict"] = strict
    defer_loading = function_payload.get("defer_loading")
    if defer_loading is not None:
        normalized["defer_loading"] = defer_loading
    return normalized


def _normalize_openai_response_format(response_format: Any) -> Any:
    """Translate one structured-output hint into Responses API text.format."""

    if is_pydantic_model_class(response_format):
        return {
            "type": "json_schema",
            "name": response_format.__name__,
            "schema": response_format.model_json_schema(),
            "strict": True,
        }
    if isinstance(response_format, dict):
        return dict(response_format)
    return response_format


def _uses_openai_responses_api(model_def: ModelDescriptor) -> bool:
    completion_api = str(getattr(model_def, "completion_api", "chat_completions") or "")
    return completion_api.strip().lower() == "responses"


# ── Gateways ─────────────────────────────────────────────────────────────


class AzureOpenAIGateway:
    """Gateway for Azure OpenAI chat completions, responses, and embeddings."""

    def __init__(
        self,
        *,
        complete_client_factory: Callable[[], AsyncAzureOpenAI],
        event_emitter: ProviderEventEmitter | None = None,
    ) -> None:
        self._complete_client_factory = complete_client_factory
        self._event_emitter = event_emitter

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        client = self._complete_client_factory()
        await self._emit(
            {
                "type": "provider_call_started",
                "provider": request.model_def.provider,
                "model_id": request.model_def.model_id,
                "display_name": request.model_def.display_name,
            }
        )
        try:
            if _uses_openai_responses_api(request.model_def):
                response = await client.responses.create(**build_openai_responses_kwargs(request))
                content = extract_responses_content(response)
                tool_calls = extract_responses_tool_calls(response)
                usage = extract_responses_usage(response)
            else:
                kwargs = build_openai_completion_kwargs(request)
                if is_pydantic_model_class(request.response_format):
                    response = await client.beta.chat.completions.parse(**kwargs)
                else:
                    response = await client.chat.completions.create(**kwargs)
                content = extract_openai_content(response)
                tool_calls = extract_openai_tool_calls(response)
                usage = extract_openai_completion_usage(response)
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
            content=content,
            provider=request.model_def.provider,
            model_id=request.model_def.model_id,
            display_name=request.model_def.display_name,
            request_id=request_id,
            tool_calls=tool_calls,
            usage=usage,
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Execute one Azure OpenAI embeddings request."""

        client = self._complete_client_factory()
        kwargs = build_openai_embedding_kwargs(request)
        try:
            response = await client.embeddings.create(**kwargs)
        except Exception as exc:  # pylint: disable=broad-except
            raise embedding_gateway_error_from_exception(request, exc) from exc
        return EmbeddingResult(
            embeddings=extract_openai_embeddings(response),
            provider=request.provider,
            model_id=request.model_id,
            request_id=extract_request_id(response),
            usage=extract_openai_embedding_usage(response),
        )

    async def _emit(self, event: ProviderStreamEvent) -> None:
        if self._event_emitter is None:
            return
        maybe_awaitable = self._event_emitter(event)
        if inspect.isawaitable(maybe_awaitable):
            await maybe_awaitable


class OpenAICompatibleGateway:
    """Gateway for OpenAI-compatible chat completions and embeddings."""

    def __init__(
        self,
        client_factory: Callable[[], AsyncOpenAI],
        *,
        event_emitter: ProviderEventEmitter | None = None,
    ) -> None:
        self._client_factory = client_factory
        self._event_emitter = event_emitter

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        client = self._client_factory()
        kwargs = build_openai_completion_kwargs(request)
        await self._emit(
            {
                "type": "provider_call_started",
                "provider": request.model_def.provider,
                "model_id": request.model_def.model_id,
                "display_name": request.model_def.display_name,
            }
        )
        try:
            if is_pydantic_model_class(request.response_format):
                response = await client.beta.chat.completions.parse(**kwargs)
            else:
                response = await client.chat.completions.create(**kwargs)
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
            content=extract_openai_content(response),
            provider=request.model_def.provider,
            model_id=request.model_def.model_id,
            display_name=request.model_def.display_name,
            request_id=request_id,
            tool_calls=extract_openai_tool_calls(response),
            usage=extract_openai_completion_usage(response),
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Execute one OpenAI-compatible embeddings request."""

        client = self._client_factory()
        kwargs = build_openai_embedding_kwargs(request)
        try:
            response = await client.embeddings.create(**kwargs)
        except Exception as exc:  # pylint: disable=broad-except
            raise embedding_gateway_error_from_exception(request, exc) from exc
        return EmbeddingResult(
            embeddings=extract_openai_embeddings(response),
            provider=request.provider,
            model_id=request.model_id,
            request_id=extract_request_id(response),
            usage=extract_openai_embedding_usage(response),
        )

    async def complete_stream(self, request: CompletionRequest) -> AsyncIterator[StreamedChunk]:
        """Stream completion chunks from an OpenAI-compatible endpoint.

        v1 streams tokens only. Tool-call deltas are not parsed; the host
        gets text deltas and a final ``finish_reason``. When the model wants
        to call tools, the stream's ``finish_reason`` will be ``'tool_calls'``
        and the host should fall back to ``complete()`` for that iteration.
        """

        client = self._client_factory()
        kwargs = build_openai_completion_kwargs(request)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        await self._emit(
            {
                "type": "provider_call_started",
                "provider": request.model_def.provider,
                "model_id": request.model_def.model_id,
                "display_name": request.model_def.display_name,
            }
        )
        try:
            stream = await client.chat.completions.create(**kwargs)
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
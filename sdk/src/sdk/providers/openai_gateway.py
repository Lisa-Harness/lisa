"""OpenAI and Azure OpenAI provider gateways.

Implements the kernel's ``ProviderGateway`` and ``EmbeddingGateway`` protocols
for OpenAI-compatible chat completions, the OpenAI Responses API, and
embeddings. This module depends on the ``openai`` SDK.
"""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, cast

from kernel.operations.structured_output import parse_structured_payload
from kernel.provider_contracts import (
    ChatMessages,
    CompletionRequest,
    CompletionResult,
    CompletionUsage,
    EmbeddingRequest,
    EmbeddingResult,
    ModelDescriptor,
    ProviderContinuation,
    ProviderEventEmitter,
    ProviderStreamEvent,
    StreamedChunk,
    embedding_gateway_error_from_exception,
    extract_request_id,
    is_pydantic_model_class,
    provider_gateway_error_from_exception,
)
from kernel.token_usage import coerce_non_negative_int
from openai import AsyncOpenAI
from openai.lib._parsing import type_to_response_format_param
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ValidationError

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


def _collect_answered_tool_ids(messages: ChatMessages) -> set[str]:
    """Collect every tool_call_id that has a matching tool-result message."""
    return {
        str(message.get("tool_call_id", "") or "")
        for message in messages
        if str(message.get("role", "") or "") == "tool"
    }


def _collect_signed_tool_ids(messages: ChatMessages) -> set[str]:
    """Collect ids of assistant tool_calls that carry a ``thought_signature``.

    Gemini thinking models emit a ``thoughtSignature`` on every
    ``functionCall`` part; the signature must be echoed verbatim on the
    next turn. The signature is read from the top level or nested under
    ``function``. Calls without a signature (kernel-synthesized handoff
    calls) must be stripped to avoid the 400.
    """
    signed_ids: set[str] = set()
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
    return signed_ids


def _normalize_message_tool_calls(
    message: dict[str, Any],
    answered_ids: set[str],
    signed_ids: set[str],
    *,
    require_thought_signature: bool,
) -> dict[str, Any]:
    """Default the assistant ``content`` and prune orphaned tool_calls on one message.

    Guarantees every ``assistant`` message carries a string ``content``
    (strict OpenAI-compatible backends reject ``content: <nil>``), prunes
    ``tool_calls`` entries with no matching tool result (or, for Gemini
    thinking models, without a ``thought_signature``), and drops the
    ``tool_calls`` key entirely when nothing remains.
    """
    normalized_message = dict(message)
    if (
        str(normalized_message.get("role", "") or "") == "assistant"
        and normalized_message.get("content") is None
    ):
        normalized_message["content"] = ""
    raw_tool_calls = normalized_message.get("tool_calls")
    if isinstance(raw_tool_calls, list):
        pruned = [
            tool_call
            for tool_call in raw_tool_calls
            if not isinstance(tool_call, dict) or str(tool_call.get("id", "") or "") in answered_ids
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
                _normalize_openai_tool_call(tool_call) if isinstance(tool_call, dict) else tool_call
                for tool_call in pruned
            ]
        else:
            normalized_message.pop("tool_calls", None)
    return normalized_message


def _prune_orphaned_tool_results_gemini(
    normalized_messages: ChatMessages,
) -> ChatMessages:
    """Drop tool results orphaned by signature-less pruning and fix the trailing role.

    When signature-less tool_calls are pruned, the matching ``role: tool``
    result messages become orphaned (no assistant tool_call references
    them). Gemini does not reject orphaned tool results outright, but
    strict OpenAI backends do, so drop them for parity with the existing
    orphaned-tool-call pruning.

    Google/Gemini rejects a request whose last message is an ``assistant``
    turn (``400 Requests ending with a model turn are not supported``).
    The signature-less synthesized ``transition_stage`` pruning can convert
    a trailing assistant ``tool_calls`` message into plain text, leaving
    the history ending with that assistant message. When that happens,
    append a synthetic ``user`` nudge so the request ends with a
    ``user``/``tool`` role as Google requires. Scoped to the Gemini
    (``require_thought_signature``) path; OpenAI/Azure accept a trailing
    assistant message.
    """
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
    pruned_messages = [
        message
        for message in normalized_messages
        if str(message.get("role", "") or "") != "tool"
        or str(message.get("tool_call_id", "") or "") in retained_assistant_ids
    ]
    if pruned_messages and str(pruned_messages[-1].get("role", "") or "") == "assistant":
        pruned_messages.append({"role": "user", "content": "Continue."})
    return pruned_messages


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
    answered_ids = _collect_answered_tool_ids(messages)

    # When Gemini thinking-model round-tripping is required, collect the
    # ids of tool_calls that carry a ``thought_signature`` (top-level or
    # nested under ``function``). Only those survive pruning; calls the
    # kernel synthesized (planned handoff / forced fork / forced plan) have
    # no signature and must be stripped to avoid the 400.
    signed_ids: set[str] = set()
    if require_thought_signature:
        signed_ids = _collect_signed_tool_ids(messages)

    normalized_messages: ChatMessages = [
        _normalize_message_tool_calls(
            message,
            answered_ids,
            signed_ids,
            require_thought_signature=require_thought_signature,
        )
        for message in messages
    ]

    normalized_messages = _enforce_tool_result_contiguity(normalized_messages)

    if require_thought_signature:
        normalized_messages = _prune_orphaned_tool_results_gemini(normalized_messages)
    return normalized_messages


# ── Request builders ─────────────────────────────────────────────────────


def sanitize_chat_tool_names(
    tools: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Sanitize function tool names for strict chat-completions backends.

    Some OpenAI-compatible backends (OpenRouter upstreams such as Decart,
    and others) enforce ``^[a-zA-Z0-9_-]+$`` on function names and reject
    Lisa's dotted namespace names (e.g. ``lisa-storage.read_file``) with
    ``400 backend_invalid_argument``. Sanitize every disallowed character
    to ``_`` on the wire and return the reverse map so extracted tool calls
    can be restored to their original names for the kernel dispatcher.

    Returns ``(sanitized_tools, name_map)`` where ``name_map`` maps
    ``sanitized_name -> original_name`` (empty when nothing was renamed).
    """

    if not tools:
        return tools or [], {}
    sanitized_tools: list[dict[str, Any]] = []
    name_map: dict[str, str] = {}
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            sanitized_tools.append(tool)
            continue
        function_payload = tool.get("function")
        if not isinstance(function_payload, dict):
            sanitized_tools.append(tool)
            continue
        original = str(function_payload.get("name", "") or "")
        sanitized = _sanitize_responses_tool_name(original)
        if original and sanitized != original:
            name_map[sanitized] = original
            patched = dict(tool)
            patched["function"] = {**function_payload, "name": sanitized}
            sanitized_tools.append(patched)
        else:
            sanitized_tools.append(tool)
    return sanitized_tools, name_map


def restore_chat_tool_names(
    tool_calls: list[dict[str, Any]],
    name_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Restore original dotted tool names in extracted chat-completions calls."""

    if not name_map:
        return tool_calls
    for call in tool_calls:
        sanitized = str(call.get("name", "") or "")
        if sanitized in name_map:
            call["name"] = name_map[sanitized]
    return tool_calls


def sanitize_chat_history_tool_names(
    messages: ChatMessages,
    name_map: dict[str, str],
) -> ChatMessages:
    """Sanitize tool-call names inside assistant message history.

    Past turns record the original dotted names (the kernel dispatcher
    matches on them), but strict backends reject those names anywhere in
    the payload — including historical ``tool_calls``. Rewrite every
    assistant ``tool_calls`` entry through the same sanitizer, extending
    ``name_map`` so responses referencing those calls restore correctly.
    """

    if not messages:
        return messages
    sanitized_messages: ChatMessages = []
    changed = False
    for message in messages:
        raw_tool_calls = message.get("tool_calls")
        if not isinstance(raw_tool_calls, list) or not raw_tool_calls:
            sanitized_messages.append(message)
            continue
        patched_calls: list[Any] = []
        message_changed = False
        for tool_call in raw_tool_calls:
            if not isinstance(tool_call, dict):
                patched_calls.append(tool_call)
                continue
            function_payload = tool_call.get("function")
            if not isinstance(function_payload, dict):
                patched_calls.append(tool_call)
                continue
            original = str(function_payload.get("name", "") or "")
            sanitized = _sanitize_responses_tool_name(original)
            if original and sanitized != original:
                name_map.setdefault(sanitized, original)
                patched_calls.append(
                    {
                        **tool_call,
                        "function": {**function_payload, "name": sanitized},
                    }
                )
                message_changed = True
            else:
                patched_calls.append(tool_call)
        if message_changed:
            changed = True
            sanitized_messages.append({**message, "tool_calls": patched_calls})
        else:
            sanitized_messages.append(message)
    return sanitized_messages if changed else messages


# Providers whose OpenAI-shaped endpoints honour ``prompt_cache_key``. Other
# OpenAI-compatible backends (Ollama, OpenRouter upstreams) may reject
# unknown parameters, so they never receive it.
_PROMPT_CACHE_KEY_PROVIDERS = frozenset({"openai", "azure_openai"})
# (provider, model_id) pairs whose deployment rejected the key; they stop
# receiving it for the life of the process instead of failing every call.
_PROMPT_CACHE_KEY_REJECTED: set[tuple[str, str]] = set()


def _cache_key_target(request: CompletionRequest) -> tuple[str, str]:
    provider = str(getattr(request.model_def, "provider", "") or "").strip().lower()
    return provider, str(request.model_def.model_id)


def _add_prompt_cache_key(kwargs: dict[str, Any], request: CompletionRequest) -> None:
    target = _cache_key_target(request)
    if (
        request.prompt_cache_key
        and target[0] in _PROMPT_CACHE_KEY_PROVIDERS
        and target not in _PROMPT_CACHE_KEY_REJECTED
    ):
        kwargs["prompt_cache_key"] = request.prompt_cache_key


_BREAKPOINT_FIELD = "prompt_cache_breakpoint"
_EXPLICIT_BREAKPOINT = {"mode": "explicit"}
# Azure reads at most the latest 50 breakpoints per request; one is reserved
# for the stable system prefix so long tool loops never push it out.
_MAX_BOUNDARY_BREAKPOINTS = 49
# (provider, model_id) pairs whose deployment rejected explicit breakpoints.
_PROMPT_CACHE_BREAKPOINTS_REJECTED: set[tuple[str, str]] = set()


def _mark_breakpoint(message: dict[str, Any]) -> dict[str, Any]:
    """Return ``message`` with an explicit cache breakpoint on its last text block."""

    content = message.get("content")
    if isinstance(content, str):
        if not content:
            return message
        block = {"type": "text", "text": content, _BREAKPOINT_FIELD: dict(_EXPLICIT_BREAKPOINT)}
        return {**message, "content": [block]}
    if isinstance(content, list):
        for index in range(len(content) - 1, -1, -1):
            part = content[index]
            if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                marked = list(content)
                marked[index] = {**part, _BREAKPOINT_FIELD: dict(_EXPLICIT_BREAKPOINT)}
                return {**message, "content": marked}
    return message


def _split_system_at_stable_prefix(
    message: dict[str, Any], segments: tuple[Any, ...]
) -> dict[str, Any] | None:
    """Split one system message into segment blocks, breakpoint on the last stable one.

    Returns ``None`` when the segments do not reproduce the message text, so
    the caller marks the whole message instead.
    """

    texts = [
        (str(getattr(segment, "text", "") or ""), bool(getattr(segment, "is_stable", True)))
        for segment in segments
    ]
    texts = [(text, stable) for text, stable in texts if text.strip()]
    if not texts or message.get("content") != "\n\n".join(text for text, _ in texts):
        return None
    last_stable = max((i for i, (_, stable) in enumerate(texts) if stable), default=-1)
    if last_stable < 0:
        return None
    blocks: list[dict[str, Any]] = []
    for index, (text, _) in enumerate(texts):
        separator = "\n\n" if index < len(texts) - 1 else ""
        block: dict[str, Any] = {"type": "text", "text": text + separator}
        if index == last_stable:
            block[_BREAKPOINT_FIELD] = dict(_EXPLICIT_BREAKPOINT)
        blocks.append(block)
    return {**message, "content": blocks}


def _uses_prompt_cache_breakpoints(request: CompletionRequest) -> bool:
    target = _cache_key_target(request)
    return (
        bool(getattr(request.model_def, "prompt_cache_breakpoints", False))
        and target[0] in _PROMPT_CACHE_KEY_PROVIDERS
        and target not in _PROMPT_CACHE_BREAKPOINTS_REJECTED
    )


def add_prompt_cache_breakpoints(
    messages: ChatMessages, segments: tuple[Any, ...] = ()
) -> ChatMessages:
    """Place explicit prompt-cache breakpoints where later requests can reuse a prefix.

    Breakpoint-model caches (Azure GPT-5.6+) only read prefixes that end at a
    breakpoint. Marks:

    * the end of the stable system prefix, so a new message, a trimmed tool
      result, or a changed routing hint still reuses the system + tools prefix;
    * every message that ended an earlier request (the message right before
      each assistant reply) plus the final message, so each tool-loop
      iteration and any reduction rewrite reuses the longest unchanged prefix.

    Positions are a pure function of the history, so a boundary marked in one
    request is marked identically in every later request.
    """

    marked = list(messages)
    leading_system = 0
    while leading_system < len(marked) and marked[leading_system].get("role") in {
        "system",
        "developer",
    }:
        leading_system += 1
    if leading_system:
        last_system = leading_system - 1
        split = (
            _split_system_at_stable_prefix(marked[0], segments)
            if leading_system == 1 and segments
            else None
        )
        marked[last_system] = split or _mark_breakpoint(marked[last_system])
    boundaries = [
        index
        for index in range(leading_system, len(marked))
        if index == len(marked) - 1 or marked[index + 1].get("role") == "assistant"
    ]
    for index in boundaries[-_MAX_BOUNDARY_BREAKPOINTS:]:
        if marked[index].get("role") != "assistant":
            marked[index] = _mark_breakpoint(marked[index])
    return marked


def _strip_prompt_cache_breakpoints(messages: Any) -> Any:
    if not isinstance(messages, list):
        return messages
    stripped: list[Any] = []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list) and any(
            isinstance(part, dict) and _BREAKPOINT_FIELD in part for part in content
        ):
            parts = [
                {k: v for k, v in part.items() if k != _BREAKPOINT_FIELD}
                if isinstance(part, dict)
                else part
                for part in content
            ]
            stripped.append({**message, "content": parts})
            continue
        stripped.append(message)
    return stripped


async def _create_with_cache_fallback(
    create: Callable[..., Awaitable[Any]], kwargs: dict[str, Any], request: CompletionRequest
) -> Any:
    """Retry once without prompt-cache parameters a deployment rejects."""
    try:
        return await create(**kwargs)
    except Exception as exc:  # pylint: disable=broad-except
        error = str(exc)
        retry = dict(kwargs)
        if "prompt_cache_key" in kwargs and "prompt_cache_key" in error:
            _PROMPT_CACHE_KEY_REJECTED.add(_cache_key_target(request))
            retry.pop("prompt_cache_key")
        elif _BREAKPOINT_FIELD in error and _uses_prompt_cache_breakpoints(request):
            _PROMPT_CACHE_BREAKPOINTS_REJECTED.add(_cache_key_target(request))
            retry["messages"] = _strip_prompt_cache_breakpoints(kwargs.get("messages"))
        else:
            raise
    return await create(**retry)


def build_openai_completion_kwargs(request: CompletionRequest) -> dict[str, Any]:
    """Build provider kwargs for one OpenAI-compatible completion request."""

    kwargs: dict[str, Any] = {
        "model": request.model_def.model_id,
        "messages": cast(
            Any,
            [
                {key: value for key, value in message.items() if key != "provider_continuation"}
                for message in normalize_openai_messages(request.messages)
            ],
        ),
    }
    # Azure deployments take ``max_completion_tokens`` regardless of the
    # reasoning flag (the legacy host dispatch special-cased the provider for
    # exactly this); public OpenAI only accepts it on reasoning models.
    is_azure = (
        str(getattr(request.model_def, "provider", "") or "").strip().lower() == "azure_openai"
    )
    if request.model_def.reasoning_model or is_azure:
        kwargs["max_completion_tokens"] = request.max_tokens
    elif request.model_def.thinking_model:
        kwargs["max_tokens"] = request.max_tokens
    else:
        kwargs["max_tokens"] = request.max_tokens
        kwargs["temperature"] = request.temperature
    if request.tools:
        kwargs["tools"] = request.tools
    if request.response_format:
        # Pydantic models are converted to the strict json_schema wire
        # param exactly as ``client.beta.chat.completions.parse`` does
        # (same ``type_to_response_format_param`` helper), but the call
        # itself goes through plain ``create``. The ``parse`` wrapper
        # validates the returned content inside the SDK call and raises
        # pydantic ``ValidationError`` on fenced JSON (```` ```json ... ````
        # — glm/kimi via Ollama and OpenRouter do this routinely), escaping
        # as a gateway error before the fence-tolerant
        # ``normalize_structured_content`` can clean it. Plain ``create``
        # leaves client-side validation to ``normalize_structured_content``.
        kwargs["response_format"] = type_to_response_format_param(request.response_format)
    _add_prompt_cache_key(kwargs, request)
    if _uses_prompt_cache_breakpoints(request):
        kwargs["messages"] = add_prompt_cache_breakpoints(
            kwargs["messages"], tuple(getattr(request, "system_prompt_segments", ()) or ())
        )
    return kwargs


def build_openai_responses_kwargs(request: CompletionRequest) -> dict[str, Any]:
    """Build provider kwargs for one OpenAI Responses API request."""

    kwargs: dict[str, Any] = {
        "model": request.model_def.model_id,
        "input": build_openai_responses_input(
            request.messages,
            provider=request.model_def.provider,
            model_id=request.model_def.model_id,
        ),
        "max_output_tokens": request.max_tokens,
        "store": False,
        "include": ["reasoning.encrypted_content"],
    }
    if not request.model_def.reasoning_model and not request.model_def.thinking_model:
        kwargs["temperature"] = request.temperature
    if request.tools:
        kwargs["tools"] = [_normalize_openai_response_tool(tool) for tool in request.tools]
    if request.response_format:
        kwargs["text"] = {"format": _normalize_openai_response_format(request.response_format)}
    _add_prompt_cache_key(kwargs, request)
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


def _responses_reasoning_items(
    message: dict[str, Any], *, provider: str, model_id: str
) -> list[dict[str, Any]]:
    continuation = message.get("provider_continuation")
    if not isinstance(continuation, dict):
        return []
    if continuation.get("provider") != provider or continuation.get("model_id") != model_id:
        return []
    return [
        dict(item)
        for item in continuation.get("items", [])
        if isinstance(item, dict)
        and item.get("type") == "reasoning"
        and item.get("encrypted_content")
    ]


def build_openai_responses_input(
    messages: ChatMessages, *, provider: str = "", model_id: str = ""
) -> list[dict[str, Any]]:
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
            input_items.extend(
                _responses_reasoning_items(message, provider=provider, model_id=model_id)
            )
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


def extract_responses_continuation(
    response: Any, model_def: ModelDescriptor
) -> ProviderContinuation | None:
    """Keep encrypted reasoning on the owning assistant message, never as prose."""
    items: list[dict[str, Any]] = []
    for item in getattr(response, "output", None) or []:
        if isinstance(item, dict):
            payload = dict(item)
        elif isinstance(item, PydanticBaseModel):
            payload = item.model_dump(mode="json", exclude_none=True)
        else:
            payload = vars(item)
        if payload.get("type") == "reasoning" and payload.get("encrypted_content"):
            items.append(dict(payload))
    if not items:
        return None
    return ProviderContinuation(
        provider=model_def.provider, model_id=model_def.model_id, items=items
    )


def _stringify_parsed_content(parsed: Any) -> str:
    """Serialize one parsed (structured-output) payload to a string.

    Pydantic models dump to JSON; other payloads fall back to ``json.dumps``
    and, when that fails (non-serializable values), to ``str()``.
    """
    if isinstance(parsed, PydanticBaseModel):
        return parsed.model_dump_json()
    try:
        return json.dumps(parsed)
    except Exception:  # pylint: disable=broad-except
        return str(parsed)


def extract_openai_content(response: Any) -> str:
    """Extract VISIBLE text or parsed JSON content from one OpenAI-compatible response.

    Reasoning text (``reasoning_content`` / Ollama's ``reasoning``) is
    deliberately NOT folded in here. Keeping the visible answer separate
    lets the kernel's user-facing display sites tell a real answer apart
    from leaked chain-of-thought so they can bound it (the 2026-10-04
    incident: a non-streaming reasoning model burned its whole budget on
    thinking; folding it into ``content`` defeated every "is this a real
    answer?" guard and surfaced a 119KB raw monologue). Callers that want
    the folded view (structured-output parsers) use ``completion_content``
    / ``extract_openai_reasoning``.

    Returns the visible answer, or the parsed structured payload when
    present, or ``""`` when the model produced no visible text (even if
    it produced reasoning — read that via ``extract_openai_reasoning``).
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
        return _stringify_parsed_content(parsed)
    return ""


def extract_openai_reasoning(response: Any) -> str:
    """Extract reasoning text from one OpenAI-compatible response, safely.

    ``reasoning_content`` is the gpt-5.x wire name; Ollama's
    OpenAI-compatible endpoint names the same field ``reasoning``. The openai
    SDK preserves unknown fields (``model_config extra="allow"``), so both
    are readable via ``getattr``. Returns ``""`` when neither is present.

    The kernel surfaces this under the ``reasoning_tail`` marker (a bounded,
    labeled quote) instead of presenting raw thought as the answer — see
    ``kernel.operations.structured_output.reasoning_tail_excerpt``.
    """

    choices = getattr(response, "choices", [])
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    for attr in ("reasoning_content", "reasoning"):
        reasoning = getattr(message, attr, None)
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning
    return ""


def _derive_openai_truncated(finish_reason: str | None) -> bool | None:
    """Derive the normalized truncation flag from an OpenAI finish reason.

    ``length`` means the generation hit the token limit mid-payload — the
    authoritative truncation signal. ``stop``/``tool_calls`` are clean
    terminations. Anything else (provider-specific reasons, missing) is
    unknown: ``None``, not ``False`` — callers treat ``None`` as "no
    information" (a single safety-net continuation, not a clean stop).
    Mirrors ``_derive_anthropic_truncated`` / Bedrock's
    ``_derive_bedrock_truncated`` so every gateway honors the same contract.
    """
    if finish_reason == "length":
        return True
    if finish_reason in {"stop", "tool_calls"}:
        return False
    return None


def extract_openai_finish_reason(response: Any) -> tuple[str | None, bool | None]:
    """Extract ``(finish_reason, truncated)`` from an OpenAI-compatible response.

    Reads ``choices[0].finish_reason`` via ``getattr`` so stubbed and real
    responses both work. The 2026-09-19 narration-loop incident: the
    OpenAI-compatible ``complete()`` never surfaced ``finish_reason`` or
    ``truncated``, so a ``finish_reason='length'`` truncation cut a
    response mid-tool-call and the kernel saw only the narration half —
    no guard could fire. Bedrock/Google/Anthropic gateways already
    populated both fields; this brings the gateway serving
    openai/ollama/openrouter (the primary deployment path) onto the same
    contract.
    """
    choices = getattr(response, "choices", [])
    if not choices:
        return (None, None)
    raw = getattr(choices[0], "finish_reason", None)
    finish_reason = str(raw).strip().lower() if raw else None
    return (finish_reason, _derive_openai_truncated(finish_reason))


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
    for item in sorted(
        getattr(response, "data", []) or [], key=lambda item: getattr(item, "index", 0)
    ):
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
    cached_tokens = _detail_tokens(prompt_details, "cached_tokens")
    cache_write_tokens = _detail_tokens(prompt_details, "cache_write_tokens")

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
        cache_creation_tokens=cache_write_tokens,
    )


def _detail_tokens(details: Any, field: str) -> int:
    """Read one token counter from a usage ``*_details`` object or dict."""

    if details is None:
        return 0
    return coerce_non_negative_int(
        details.get(field) if isinstance(details, dict) else getattr(details, field, None)
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
    cached_tokens = _detail_tokens(input_details, "cached_tokens")
    cache_write_tokens = _detail_tokens(input_details, "cache_write_tokens")

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
        cache_creation_tokens=cache_write_tokens,
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
        "name": _sanitize_responses_tool_name(str(function_payload.get("name", "") or "")),
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


def _enforce_strict_json_schema(schema: Any) -> Any:
    """Recursively add ``additionalProperties: false`` to every object schema.

    The Responses API requires ``additionalProperties: false`` on every object
    when ``strict: true``. Pydantic's ``model_json_schema()`` does not emit it
    by default, so the API rejects the schema with
    ``Invalid schema for response_format: 'additionalProperties' is required
    to be supplied and to be false``. Walk the schema and patch every object
    node (``type: object`` or ``$defs`` entries) so it satisfies the strict
    constraint.
    """

    if isinstance(schema, dict):
        if schema.get("type") == "object" and "additionalProperties" not in schema:
            schema["additionalProperties"] = False
        for key, value in schema.items():
            if key == "$defs" and isinstance(value, dict):
                for def_name, def_schema in value.items():
                    value[def_name] = _enforce_strict_json_schema(def_schema)
            elif isinstance(value, (dict, list)):
                schema[key] = _enforce_strict_json_schema(value)
    elif isinstance(schema, list):
        return [_enforce_strict_json_schema(item) for item in schema]
    return schema


def _normalize_openai_response_format(response_format: Any) -> Any:
    """Translate one structured-output hint into Responses API text.format."""

    if is_pydantic_model_class(response_format):
        schema = response_format.model_json_schema()
        _enforce_strict_json_schema(schema)
        return {
            "type": "json_schema",
            "name": response_format.__name__,
            "schema": schema,
            "strict": True,
        }
    if isinstance(response_format, dict):
        return dict(response_format)
    return response_format


def _uses_openai_responses_api(model_def: ModelDescriptor) -> bool:
    completion_api = str(getattr(model_def, "completion_api", "chat_completions") or "")
    return completion_api.strip().lower() == "responses"


# ── Structured-output normalization ──────────────────────────────────────


def _extract_json_object(text: str) -> str:
    """Extract the first likely JSON object from a model response.

    Strips ```` ```json ```` / ```` ``` ```` markdown fences, then narrows to
    the outermost ``{...}`` span. Ported from the legacy host dispatch so
    structured-output operations see clean JSON regardless of whether the
    model wrapped its answer in a fence.
    """

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


def normalize_structured_content(content: str, response_format: Any) -> str:
    """Return re-validated structured JSON when possible, else the original.

    When ``response_format`` is a Pydantic model class, strip any markdown
    fence, validate the payload against the schema, and return the canonical
    ``model_dump_json()`` round-trip. On validation failure the raw content is
    returned unchanged so callers can degrade gracefully instead of raising.
    """

    if (
        response_format is None
        or not content
        or not hasattr(response_format, "model_validate_json")
    ):
        return content
    try:
        parsed = parse_structured_payload(response_format, _extract_json_object(content))
    except ValidationError:
        return content
    normalized = parsed.model_dump_json()
    if not isinstance(normalized, str):
        return content
    return normalized


# ── Gateways ─────────────────────────────────────────────────────────────


def create_azure_openai_client(*, endpoint: str, api_key: str) -> AsyncOpenAI:
    """Construct an Azure v1 client shared by chat, Responses, and embeddings."""
    base_url = endpoint.rstrip("/")
    if not base_url.endswith("/openai/v1"):
        base_url = f"{base_url}/openai/v1"
    return AsyncOpenAI(base_url=f"{base_url}/", api_key=api_key)


class AzureOpenAIGateway:
    """Gateway for Azure OpenAI chat completions, responses, and embeddings."""

    def __init__(
        self,
        *,
        complete_client_factory: Callable[[], AsyncOpenAI],
        event_emitter: ProviderEventEmitter | None = None,
    ) -> None:
        """Construct the Azure gateway with one SDK-owned v1 client factory."""
        self._complete_client_factory = complete_client_factory
        self._event_emitter = event_emitter

    async def complete(self, request: CompletionRequest) -> CompletionResult:
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
                client = self._complete_client_factory()
                name_map = build_responses_tool_name_map(request.tools or [])
                response = await _create_with_cache_fallback(
                    client.responses.create, build_openai_responses_kwargs(request), request
                )
                content = normalize_structured_content(
                    extract_responses_content(response), request.response_format
                )
                # Dotted tool names are sanitized for the wire; restore the
                # originals so the kernel dispatcher can match them.
                tool_calls = restore_responses_tool_names(
                    extract_responses_tool_calls(response), name_map
                )
                usage = extract_responses_usage(response)
            else:
                client = self._complete_client_factory()
                kwargs = build_openai_completion_kwargs(request)
                # Plain ``create`` for structured output too: the strict
                # json_schema wire param is set in ``build_openai_completion_kwargs``,
                # and fence/prose tolerance is handled client-side by
                # ``normalize_structured_content`` instead of raising inside
                # the SDK's ``parse`` helper.
                response = await _create_with_cache_fallback(
                    client.chat.completions.create, kwargs, request
                )
                content = normalize_structured_content(
                    extract_openai_content(response), request.response_format
                )
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
        # Chat-completions responses carry choices[0].finish_reason — the
        # same shape the OpenAI-compatible gateway reads. Responses-API
        # responses use ``status`` instead; left None (unknown) here — the
        # conservative choice — until a truncation contract for that
        # branch is needed. (2026-09-19 narration-loop incident.)
        if _uses_openai_responses_api(request.model_def):
            finish_reason, truncated = None, None
        else:
            finish_reason, truncated = extract_openai_finish_reason(response)
        return CompletionResult(
            content=content,
            provider=request.model_def.provider,
            model_id=request.model_def.model_id,
            display_name=request.model_def.display_name,
            request_id=request_id,
            tool_calls=tool_calls,
            usage=usage,
            reasoning_content=extract_openai_reasoning(response),
            finish_reason=finish_reason,
            truncated=truncated,
            continuation=extract_responses_continuation(response, request.model_def)
            if _uses_openai_responses_api(request.model_def)
            else None,
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
        # Strict OpenAI-compatible backends (OpenRouter upstreams, etc.)
        # reject dotted tool names; sanitize on the wire and restore the
        # originals on extraction so the kernel dispatcher still matches.
        sanitized_tools, name_map = sanitize_chat_tool_names(kwargs.get("tools"))
        if name_map:
            kwargs["tools"] = sanitized_tools
            kwargs["messages"] = sanitize_chat_history_tool_names(kwargs["messages"], name_map)
        await self._emit(
            {
                "type": "provider_call_started",
                "provider": request.model_def.provider,
                "model_id": request.model_def.model_id,
                "display_name": request.model_def.display_name,
            }
        )
        try:
            # Plain ``create`` for structured output too: the strict
            # json_schema wire param is set in ``build_openai_completion_kwargs``,
            # and fence/prose tolerance is handled client-side by
            # ``normalize_structured_content`` instead of raising inside
            # the SDK's ``parse`` helper.
            response = await _create_with_cache_fallback(
                client.chat.completions.create, kwargs, request
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
        # Surface the provider termination signal so the kernel can see
        # truncation (finish_reason='length') on the non-streaming path —
        # the 2026-09-19 narration-loop incident: a length-truncated
        # response cut mid-tool-call was indistinguishable from a natural
        # narration-only completion without these fields.
        finish_reason, truncated = extract_openai_finish_reason(response)
        return CompletionResult(
            content=normalize_structured_content(
                extract_openai_content(response), request.response_format
            ),
            provider=request.model_def.provider,
            model_id=request.model_def.model_id,
            display_name=request.model_def.display_name,
            request_id=request_id,
            tool_calls=restore_chat_tool_names(extract_openai_tool_calls(response), name_map),
            usage=extract_openai_completion_usage(response),
            reasoning_content=extract_openai_reasoning(response),
            finish_reason=finish_reason,
            truncated=truncated,
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

        v1 streamed tokens only; the 2026-10-12 one-pass change adds
        tool-call delta parsing: an OpenAI-style ``delta.tool_calls`` list is
        forwarded per chunk as ``tool_call_deltas`` (index/id/name/argument
        fragments the kernel stitches). Backends that don't stream tool
        calls keep the chunk contract unchanged — the turn runner falls
        back to ``complete()`` for a ``tool_calls`` finish with no deltas.
        """

        client = self._client_factory()
        kwargs = build_openai_completion_kwargs(request)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        # Strict backends validate tool names even on streaming requests;
        # sanitize on the wire and restore the originals on the parsed
        # deltas before yielding (matches the non-streaming path's
        # sanitize/restore pair).
        sanitized_tools, name_map = sanitize_chat_tool_names(kwargs.get("tools"))
        if name_map:
            kwargs["tools"] = sanitized_tools
            kwargs["messages"] = sanitize_chat_history_tool_names(kwargs["messages"], name_map)
        await self._emit(
            {
                "type": "provider_call_started",
                "provider": request.model_def.provider,
                "model_id": request.model_def.model_id,
                "display_name": request.model_def.display_name,
            }
        )
        try:
            stream = await _create_with_cache_fallback(
                client.chat.completions.create, kwargs, request
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
        async for event in stream:
            delta_content = ""
            delta_reasoning = ""
            finish_reason = None
            choices = getattr(event, "choices", None) or []
            tool_call_deltas: list[dict[str, Any]] = []
            if choices:
                choice = choices[0]
                delta = getattr(choice, "delta", None)
                if delta is not None:
                    delta_content = getattr(delta, "content", None) or ""
                    # gpt-5.x reasoning models emit "thinking" as
                    # ``delta.reasoning_content`` (separate from
                    # ``delta.content``). Capture it so the aggregated
                    # completion can surface it as a fallback when the
                    # visible answer is empty. Ollama names the same field
                    # ``delta.reasoning`` — read both so reasoning models
                    # served through Ollama keep their answer visible.
                    delta_reasoning = (
                        getattr(delta, "reasoning_content", None)
                        or getattr(delta, "reasoning", None)
                        or ""
                    )
                for raw_delta in (
                    getattr(delta, "tool_calls", None) if delta is not None else None
                ) or []:
                    function_obj = getattr(raw_delta, "function", None)
                    entry: dict[str, Any] = {
                        "index": getattr(raw_delta, "index", 0) or 0,
                    }
                    call_id = getattr(raw_delta, "id", None)
                    if call_id:
                        entry["id"] = call_id
                    call_type = getattr(raw_delta, "type", None)
                    if call_type:
                        entry["type"] = call_type
                    entry["function"] = {
                        "name": getattr(function_obj, "name", None) or "",
                        "arguments": getattr(function_obj, "arguments", None) or "",
                    }
                    # Restore the sanitized dotted tool name so the kernel
                    # dispatcher matches the non-streaming wire shape.
                    wire_name = entry["function"]["name"]
                    if name_map and wire_name in name_map:
                        entry["function"]["name"] = name_map[wire_name]
                    tool_call_deltas.append(entry)
                finish_reason = getattr(choice, "finish_reason", None)
            usage = extract_openai_completion_usage(event)
            yield StreamedChunk(
                content=delta_content,
                reasoning_content=delta_reasoning,
                finish_reason=finish_reason,
                usage=usage,
                tool_call_deltas=tool_call_deltas,
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

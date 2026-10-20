"""Tests for the Anthropic gateway tool-use path and provider routing.

Covers ``AnthropicGateway.complete`` translation of OpenAI-format messages
and tools into Anthropic format, extraction of ``text``/``tool_use`` content
blocks back into the normalized ``CompletionResult`` shape, and
``SdkProviderGatewayProvider.gateway_for`` routing for ``anthropic``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from kernel.provider_contracts import (
    CompletionRequest,
    CompletionResult,
    ModelDescriptor,
)
from sdk.providers.anthropic_gateway import (
    AnthropicGateway,
    _build_anthropic_messages,
)
from sdk.providers.openai_gateway import (
    _coerce_tool_arguments_object,
    _repair_openai_tool_arguments,
    build_openai_responses_input,
    normalize_openai_messages,
)

# ── Test helpers ──────────────────────────────────────────────────────


def _model_def(provider: str = "anthropic", model_id: str = "claude-test") -> ModelDescriptor:
    return SimpleNamespace(
        provider=provider,
        model_id=model_id,
        display_name="Claude Test",
        reasoning_model=False,
        thinking_model=False,
    )


def _request(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    model_def: ModelDescriptor | None = None,
) -> CompletionRequest:
    return CompletionRequest(
        messages=messages,
        model_def=model_def or _model_def(),
        max_tokens=max_tokens,
        temperature=temperature,
        tools=tools,
    )


def _text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _tool_use_block(block_id: str, name: str, input_obj: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_use", "id": block_id, "name": name, "input": input_obj}


def _response(
    blocks: list[dict[str, Any]],
    *,
    input_tokens: int = 10,
    output_tokens: int = 20,
    response_id: str = "resp_123",
    stop_reason: str | None = "end_turn",
) -> dict[str, Any]:
    """Return an Anthropic Messages JSON payload (the gateway reads dicts)."""

    return {
        "id": response_id,
        "content": blocks,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


class _StubHttpResponse:
    """Minimal stub for a raw ``httpx.Response`` (json/raise_for_status/headers)."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.headers: dict[str, str] = {"request-id": str(payload.get("id") or "")}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _StubHttpClient:
    """Stub raw-httpx client exposing ``post(url, json=..., headers=...)``."""

    def __init__(self, payload: dict[str, Any], captured: dict[str, Any]) -> None:
        self._payload = payload
        self.captured = captured

    async def post(self, url: str, *, json: Any = None, headers: Any = None) -> _StubHttpResponse:
        self.captured["url"] = url
        self.captured["json"] = json
        self.captured["headers"] = headers
        return _StubHttpResponse(self._payload)


# ── Usage extraction: cache sub-counters ─────────────────────────────


def test_anthropic_usage_maps_cache_creation_to_dedicated_counter() -> None:
    """cache_creation_input_tokens rides ``cache_creation_tokens``, not reasoning."""
    from sdk.providers.anthropic_gateway import extract_anthropic_usage

    usage = extract_anthropic_usage(
        {
            "usage": {
                "input_tokens": 200,
                "output_tokens": 40,
                "cache_creation_input_tokens": 150,
                "cache_read_input_tokens": 800,
            }
        }
    )
    assert usage is not None
    assert usage.cache_creation_tokens == 150
    assert usage.cached_tokens == 800
    assert usage.prompt_tokens == 1150
    assert usage.total_tokens == 1190
    # Cache creation is a write-rate cost, not reasoning output.
    assert usage.reasoning_tokens == 0


# ── Cache-aware system prompt segmentation ────────────────────────────


def _segmented_request() -> CompletionRequest:
    """Build a request carrying typed stable/ephemeral prompt segments."""
    from kernel.provider_contracts import SystemPromptSegment

    return CompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        model_def=_model_def(),
        max_tokens=1024,
        temperature=0.0,
        system_prompt_segments=(
            SystemPromptSegment(text="Stable harness + mode", stability="stable"),
            SystemPromptSegment(text="SESSION body", stability="stable"),
            SystemPromptSegment(text="## Routing Hint\n\nhand off", stability="ephemeral"),
            SystemPromptSegment(text="iter 2/50", stability="ephemeral"),
        ),
    )


@pytest.mark.asyncio
async def test_segments_emit_system_blocks_with_ephemeral_cache_breakpoint() -> None:
    """Stable prefix gets one cache_control block; dynamics stay uncached."""
    captured: dict[str, Any] = {}
    gateway = _gateway(_response([_text_block("ok")]), captured)
    await gateway.complete(_segmented_request())
    system = captured["json"]["system"]
    assert isinstance(system, list)
    assert [block["text"] for block in system] == [
        "Stable harness + mode",
        "SESSION body",
        "## Routing Hint\n\nhand off",
        "iter 2/50",
    ]
    # Only the LAST stable block carries the breakpoint — the boundary.
    assert system[1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in system[0]
    assert "cache_control" not in system[2]
    assert "cache_control" not in system[3]


@pytest.mark.asyncio
async def test_segments_without_ephemeral_tail_cache_the_single_stable_block() -> None:
    """Pure-stable payloads put the breakpoint on the final stable block."""
    from kernel.provider_contracts import SystemPromptSegment

    captured: dict[str, Any] = {}
    gateway = _gateway(_response([_text_block("ok")]), captured)
    request = CompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        model_def=_model_def(),
        max_tokens=1024,
        temperature=0.0,
        system_prompt_segments=(SystemPromptSegment(text="Stable only", stability="stable"),),
    )
    await gateway.complete(request)
    system = captured["json"]["system"]
    assert isinstance(system, list)
    assert system[0]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_prompt_cache_disabled_flattens_to_legacy_system_string() -> None:
    """With the host opt-out, segments flatten to the pre-segments string."""
    captured: dict[str, Any] = {}
    client = _StubHttpClient(_response([_text_block("ok")]), captured)
    gateway = AnthropicGateway(
        client_factory=lambda: client,
        api_key="test-key",
        api_version="2023-06-01",
        prompt_cache_enabled=False,
    )
    await gateway.complete(_segmented_request())
    system = captured["json"]["system"]
    # Legacy string form; byte-identical to joining with the canonical "\n\n".
    assert isinstance(system, str)
    assert "Stable harness + mode" in system
    assert "## Routing Hint" in system
    assert "iter 2/50" in system
    assert "cache_control" not in captured["json"]


@pytest.mark.asyncio
async def test_growing_history_uses_automatic_cache_with_stable_system_anchor() -> None:
    captured: dict[str, Any] = {}
    gateway = _gateway(_response([_text_block("ok")]), captured)
    request = _segmented_request()
    await gateway.complete(request)
    assert captured["json"]["cache_control"] == {"type": "ephemeral"}
    request.messages.extend(
        [
            {"role": "assistant", "content": "working"},
            {"role": "user", "content": "continue"},
        ]
    )
    await gateway.complete(request)
    assert captured["json"]["cache_control"] == {"type": "ephemeral"}
    assert captured["json"]["system"][1]["cache_control"] == {"type": "ephemeral"}
    assert len(captured["json"]["messages"]) == 3


@pytest.mark.asyncio
async def test_no_segments_keeps_legacy_string_system_only() -> None:
    """A request without segment metadata keeps the single-string system payload."""
    captured: dict[str, Any] = {}
    gateway = _gateway(_response([_text_block("ok")]), captured)
    request = _request(
        [
            {"role": "system", "content": "Only the legacy system string"},
            {"role": "user", "content": "hi"},
        ]
    )
    await gateway.complete(request)
    assert captured["json"]["system"] == "Only the legacy system string"


def _gateway(
    payload: dict[str, Any],
    captured: dict[str, Any],
) -> AnthropicGateway:
    client = _StubHttpClient(payload, captured)
    return AnthropicGateway(
        client_factory=lambda: client,
        api_key="test-key",
        api_version="2023-06-01",
    )


# ── Text-only response ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_text_only_response() -> None:
    captured: dict[str, Any] = {}
    gateway = _gateway(_response([_text_block("hello world")]), captured)
    result = await gateway.complete(_request([{"role": "user", "content": "hi"}]))
    assert isinstance(result, CompletionResult)
    assert result.content == "hello world"
    assert result.tool_calls == []
    assert result.provider == "anthropic"
    assert result.model_id == "claude-test"
    assert result.request_id == "resp_123"


# ── Tool-use response ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_use_response() -> None:
    captured: dict[str, Any] = {}
    gateway = _gateway(
        _response(
            [
                _text_block("Let me check."),
                _tool_use_block("call_1", "get_weather", {"city": "Seattle"}),
            ]
        ),
        captured,
    )
    result = await gateway.complete(_request([{"role": "user", "content": "weather?"}]))
    assert result.content == "Let me check."
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc["id"] == "call_1"
    assert tc["type"] == "function"
    assert tc["name"] == "get_weather"
    assert tc["arguments"] == '{"city": "Seattle"}'


# ── Tool-use only (no text) ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_use_only_no_text() -> None:
    captured: dict[str, Any] = {}
    gateway = _gateway(
        _response([_tool_use_block("call_2", "search", {"q": "cats"})]),
        captured,
    )
    result = await gateway.complete(_request([{"role": "user", "content": "go"}]))
    assert result.content == ""
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "search"


# ── Multi-block text concatenation ────────────────────────────────────


@pytest.mark.asyncio
async def test_multi_block_text_concatenated() -> None:
    captured: dict[str, Any] = {}
    gateway = _gateway(
        _response([_text_block("part one"), _text_block("part two")]),
        captured,
    )
    result = await gateway.complete(_request([{"role": "user", "content": "x"}]))
    assert result.content == "part one\npart two"


# ── Tool round-trip: tools passed to Anthropic ────────────────────────


@pytest.mark.asyncio
async def test_tools_translated_to_anthropic_format() -> None:
    captured: dict[str, Any] = {}
    gateway = _gateway(_response([_text_block("ok")]), captured)
    openai_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
    await gateway.complete(_request([{"role": "user", "content": "hi"}], tools=[openai_tool]))
    assert captured["json"]["tools"] == [
        {
            "name": "get_weather",
            "description": "Get the weather",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]


# ── Assistant + tool_calls round-trip in messages ─────────────────────


@pytest.mark.asyncio
async def test_assistant_tool_calls_translated_to_anthropic() -> None:
    captured: dict[str, Any] = {}
    gateway = _gateway(_response([_text_block("done")]), captured)
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "Checking.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Seattle"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "sunny",
        },
    ]
    await gateway.complete(_request(list(messages)))
    anthropic_messages = captured["json"]["messages"]
    # user message passes through
    assert anthropic_messages[0] == {"role": "user", "content": "weather?"}
    # assistant message becomes content blocks
    assistant_msg = anthropic_messages[1]
    assert assistant_msg["role"] == "assistant"
    block_types = [b["type"] for b in assistant_msg["content"]]
    assert block_types == ["text", "tool_use"]
    tool_use_block = assistant_msg["content"][1]
    assert tool_use_block["id"] == "call_1"
    assert tool_use_block["name"] == "get_weather"
    assert tool_use_block["input"] == {"city": "Seattle"}
    # tool-role message becomes user with tool_result
    tool_msg = anthropic_messages[2]
    assert tool_msg["role"] == "user"
    assert tool_msg["content"][0]["type"] == "tool_result"
    assert tool_msg["content"][0]["tool_use_id"] == "call_1"
    assert tool_msg["content"][0]["content"] == "sunny"


# ── normalize_openai_messages: strict-backend assistant content fix ──


def test_normalize_assistant_with_tool_calls_and_no_content_defaults_to_empty_string() -> None:
    """Assistant messages with tool_calls and no ``content`` get ``content=""``.

    The turn runner appends ``{"role": "assistant", "tool_calls": [...]}``
    (no ``content`` key) when the model emits tool calls without text. Strict
    OpenAI-compatible backends (Ollama) reject ``content: <nil>`` with
    ``invalid message content type: <nil>``. ``normalize_openai_messages`` is
    the single chokepoint that must repair this for every OpenAI request.
    """

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "remember this"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "synthesize", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]
    normalized = normalize_openai_messages(list(messages))
    assistant = normalized[1]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == ""
    # tool_calls preserved
    assert isinstance(assistant.get("tool_calls"), list)
    assert assistant["tool_calls"][0]["id"] == "call_1"


def test_normalize_assistant_with_explicit_none_content_defaults_to_empty_string() -> None:
    """An assistant message whose content is explicitly ``None`` is repaired."""

    messages: list[dict[str, Any]] = [
        {"role": "assistant", "content": None, "tool_calls": []},
    ]
    normalized = normalize_openai_messages(messages)
    assert normalized[0]["content"] == ""


def test_normalize_preserves_non_assistant_missing_content() -> None:
    """Only assistant messages are repaired; user/tool pass through untouched."""

    messages: list[dict[str, Any]] = [{"role": "user"}]  # no content key on a user message
    normalized = normalize_openai_messages(messages)
    assert "content" not in normalized[0]


def test_normalize_preserves_assistant_text_content() -> None:
    """Assistant messages with real text keep their content unchanged."""

    messages: list[dict[str, Any]] = [{"role": "assistant", "content": "hello"}]
    normalized = normalize_openai_messages(messages)
    assert normalized[0]["content"] == "hello"


# ── normalize_openai_messages: orphaned tool_calls pruning ───────────


def test_normalize_prunes_orphaned_tool_calls_without_tool_result() -> None:
    """An assistant tool_call with no matching tool result is dropped.

    When the turn loop breaks mid-tool-loop (forced plan/fork handoff,
    entry failure, iteration-cap yield), it can leave an assistant message
    declaring ``tool_calls`` with no following ``role: tool`` message for
    that id. Strict OpenAI backends reject this with a 400
    ("...must be followed by tool messages responding to each
    'tool_call_id'"). The normalization chokepoint must prune the orphan so
    the payload stays well-formed.
    """

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "do a multi-step thing"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_orphan",
                    "type": "function",
                    "function": {"name": "transition_stage", "arguments": "{}"},
                }
            ],
        },
        # No tool result for call_orphan — history is dangling.
    ]
    normalized = normalize_openai_messages(messages)
    assistant = normalized[1]
    assert "tool_calls" not in assistant
    assert assistant["content"] == ""
    # The user message is untouched.
    assert normalized[0]["content"] == "do a multi-step thing"


def test_normalize_keeps_answered_tool_calls() -> None:
    """A tool_call paired with a matching tool result is preserved."""

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_ok",
                    "type": "function",
                    "function": {"name": "memory.search", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_ok", "content": "results"},
    ]
    normalized = normalize_openai_messages(messages)
    assistant = normalized[1]
    assert "tool_calls" in assistant
    assert assistant["tool_calls"][0]["id"] == "call_ok"


def test_normalize_prunes_orphan_among_answered_calls() -> None:
    """An orphaned call is dropped while answered calls survive."""

    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_ok", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                {
                    "id": "call_orphan",
                    "type": "function",
                    "function": {"name": "transition_stage", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_ok", "content": "ok"},
    ]
    normalized = normalize_openai_messages(messages)
    assistant = normalized[0]
    assert assistant["tool_calls"] == [
        {"id": "call_ok", "type": "function", "function": {"name": "a", "arguments": "{}"}}
    ]


# ── normalize_openai_messages: Gemini thought_signature pruning ──────


def test_normalize_strips_signature_less_tool_calls_for_gemini() -> None:
    """Signature-less synthesized tool_calls are pruned for Gemini thinking models."""

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "Handing off.",
            "tool_calls": [
                {
                    "id": "call_synth",
                    "type": "function",
                    "function": {"name": "transition_stage", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_synth", "content": "ok"},
    ]
    normalized = normalize_openai_messages(messages, require_thought_signature=True)
    assistant = normalized[1]
    assert "tool_calls" not in assistant
    assert assistant["content"] == "Handing off."
    # orphaned tool result dropped
    assert not [m for m in normalized if m.get("role") == "tool"]


def test_normalize_keeps_signed_calls_when_require_thought_signature_true() -> None:
    """Tool_calls carrying a thought_signature survive the Gemini prune."""

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_signed",
                    "type": "function",
                    "function": {"name": "memory.search", "arguments": "{}"},
                    "thought_signature": "sig_abc",
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_signed", "content": "results"},
    ]
    normalized = normalize_openai_messages(messages, require_thought_signature=True)
    assistant = normalized[1]
    assert "tool_calls" in assistant
    assert assistant["tool_calls"][0]["id"] == "call_signed"


def test_normalize_require_thought_signature_false_keeps_unsigned_calls() -> None:
    """When require_thought_signature is False, unsigned calls are preserved."""

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_synth",
                    "type": "function",
                    "function": {"name": "transition_stage", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_synth", "content": "ok"},
    ]
    normalized = normalize_openai_messages(messages, require_thought_signature=False)
    assistant = normalized[1]
    assert "tool_calls" in assistant
    assert assistant["tool_calls"][0]["id"] == "call_synth"


# ── normalize_openai_messages: Gemini trailing-assistant nudge ──────


def test_normalize_appends_user_nudge_when_gemini_prune_leaves_trailing_assistant() -> None:
    """Gemini rejects requests ending with an assistant turn.

    When signature-less pruning converts a trailing assistant ``tool_calls``
    message into plain text (and drops its orphaned tool result), the
    history now ends with an ``assistant`` message. Google returns
    ``400 Requests ending with a model turn are not supported``. The
    normalize chokepoint must append a synthetic ``user`` nudge so the
    request ends with a ``user`` role.
    """

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "do a multi-step thing"},
        {
            "role": "assistant",
            "content": "Handing off.",
            "tool_calls": [
                {
                    "id": "call_synth",
                    "type": "function",
                    "function": {"name": "transition_stage", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_synth", "content": "ok"},
    ]
    normalized = normalize_openai_messages(messages, require_thought_signature=True)
    # The trailing assistant (now plain text) is no longer the last message.
    assert normalized[-1]["role"] == "user"
    assert normalized[-1]["content"]


def test_normalize_no_trailing_nudge_when_history_ends_with_tool_result() -> None:
    """No nudge is appended when the history ends with a tool result.

    A signed tool call followed by its tool result leaves the last message
    as ``role: tool``; Google accepts that. The trailing-assistant nudge
    must only fire when the last message is actually an assistant turn.
    """

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_signed",
                    "type": "function",
                    "function": {"name": "memory.search", "arguments": "{}"},
                    "thought_signature": "sig_abc",
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_signed", "content": "results"},
    ]
    normalized = normalize_openai_messages(messages, require_thought_signature=True)
    assert normalized[-1]["role"] == "tool"
    assert normalized[-1]["tool_call_id"] == "call_signed"


def test_normalize_no_trailing_nudge_when_require_thought_signature_false() -> None:
    """The trailing-assistant nudge is Gemini-only.

    OpenAI/Azure accept a request ending with an assistant message, so the
    nudge must not fire when ``require_thought_signature`` is False.
    """

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "Done.",
            "tool_calls": [
                {
                    "id": "call_synth",
                    "type": "function",
                    "function": {"name": "transition_stage", "arguments": "{}"},
                }
            ],
        },
    ]
    normalized = normalize_openai_messages(messages, require_thought_signature=False)
    # Trailing assistant with its (orphaned) tool_calls pruned stays last.
    assert normalized[-1]["role"] == "assistant"


# ── normalize_openai_messages: unparseable tool-call arguments repair ──


def test_normalize_repairs_unparseable_tool_call_arguments() -> None:
    """A tool call whose arguments never parsed is replayed as ``{}``.

    Regression for the 2026-10-19 MiniMax M3.1 Flash turn failure: three
    parallel tool calls were stitched into one call whose arguments were
    three JSON objects glued end to end. The kernel refused to dispatch
    it, but the broken call stayed in the message history and the next
    request replayed it. Ollama validates the arguments of every
    assistant ``tool_calls`` entry in the history and answered
    ``400 invalid tool call arguments``, which killed the whole turn.

    The original values do not exist (the payload was cut), so the empty
    object is the honest repair: it keeps the history well-formed and
    lets the model re-issue the call.
    """

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "review the code"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_merged",
                    "type": "function",
                    "function": {
                        "name": "lisa-storage.list_fileslisa-storage.read_file",
                        "arguments": ('{"prefix":"demo"}{"path":"a.md"}{"path":"b.md"}'),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_merged", "content": "truncated"},
    ]
    normalized = normalize_openai_messages(messages)
    call = normalized[1]["tool_calls"][0]
    assert call["function"]["arguments"] == "{}"


def test_normalize_repairs_length_cut_tool_call_arguments() -> None:
    """A length-cut arguments payload is repaired the same way.

    The 2026-09-23 finances incident produced the same shape from a
    different cause: ``finish_reason='length'`` cut the arguments
    mid-payload. The kernel refuses to dispatch such a call, so the
    replay must not kill the turn either.
    """

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "write the ledger"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_cut",
                    "type": "function",
                    "function": {
                        "name": "lisa-storage.write_file",
                        "arguments": '{"path": "ledgers/2024.beancount", "content": "2024-01-02',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_cut", "content": "truncated"},
    ]
    normalized = normalize_openai_messages(messages)
    call = normalized[1]["tool_calls"][0]
    assert call["function"]["arguments"] == "{}"


def test_normalize_repairs_empty_tool_call_arguments_string() -> None:
    """An empty argument string is not valid JSON on the wire.

    The kernel reads an empty string as a legitimate "no parameters"
    call, but the backend parses the string as JSON and rejects it.
    """

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_empty",
                    "type": "function",
                    "function": {"name": "list_threads", "arguments": ""},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_empty", "content": "ok"},
    ]
    normalized = normalize_openai_messages(messages)
    call = normalized[1]["tool_calls"][0]
    assert call["function"]["arguments"] == "{}"


def test_normalize_keeps_parseable_tool_call_arguments_untouched() -> None:
    """A well-formed payload is replayed byte-for-byte."""

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_ok",
                    "type": "function",
                    "function": {
                        "name": "lisa-storage.read_file",
                        "arguments": '{"path": "README.md"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_ok", "content": "ok"},
    ]
    normalized = normalize_openai_messages(messages)
    call = normalized[1]["tool_calls"][0]
    assert call["function"]["arguments"] == '{"path": "README.md"}'


# ── tool-call arguments: strict JSON across every gateway ──


def test_repair_rejects_python_lenient_json_constants() -> None:
    """``NaN``/``Infinity`` are rejected by strict backends, not just by Python.

    ``json.loads`` accepts the non-standard constants and an overflowing
    float such as ``1e999``; strict backends reject all of them and
    ``httpx`` refuses to serialize them at all. A payload Python calls
    valid therefore reached the wire as a 400 (or a client-side crash) and
    the repair skipped it.
    """
    for raw in ('{"x": NaN}', '{"x": Infinity}', '{"x": -Infinity}', '{"x": 1e999}'):
        assert _repair_openai_tool_arguments(raw) == "{}", raw


def test_repair_keeps_a_finite_large_float() -> None:
    """A large but finite float is valid JSON and must survive untouched."""

    assert _repair_openai_tool_arguments('{"x": 1e308}') == '{"x": 1e308}'


def test_repair_rejects_a_non_object_json_payload() -> None:
    """A tool call's arguments must be a JSON object on the wire."""

    for raw in ("[1,2]", "null", "123", '"text"'):
        assert _repair_openai_tool_arguments(raw) == "{}", raw


def test_repair_rejects_a_dict_holding_a_non_finite_float() -> None:
    """A dict argument with ``NaN`` cannot be serialized; it is repaired."""

    assert _repair_openai_tool_arguments({"x": float("nan")}) == "{}"


def test_coerce_tool_arguments_object_rejects_invalid_payloads() -> None:
    """The shared coercion accepts only a JSON object."""

    assert _coerce_tool_arguments_object('{"a": 1}') == {"a": 1}
    for raw in ('{"a":1}{"b":2}', '{"x": NaN}', "[1,2]", "null", "123", ""):
        assert _coerce_tool_arguments_object(raw) == {}, raw


def test_coerce_tool_arguments_object_keeps_a_valid_dict() -> None:
    """A dict argument passes through unchanged."""

    assert _coerce_tool_arguments_object({"a": 1}) == {"a": 1}


def test_anthropic_tool_use_input_is_always_an_object() -> None:
    """The Anthropic path coerces the same way the OpenAI path does.

    The three non-OpenAI gateways each parsed the arguments with their
    own ``except json.JSONDecodeError`` handler, so a payload Python
    accepts but a strict backend rejects (``NaN``) reached the wire and
    crashed ``httpx`` before any request was made.
    """
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_nan",
                    "type": "function",
                    "function": {"name": "a_tool", "arguments": '{"x": NaN}'},
                }
            ],
        },
    ]
    converted = _build_anthropic_messages(messages)
    tool_use = converted[1]["content"][0]
    assert tool_use["type"] == "tool_use"
    assert tool_use["input"] == {}


def test_normalize_repairs_the_flat_tool_call_shape() -> None:
    """The flat tool-call shape the kernel produces is repaired too.

    ``extract_openai_tool_calls`` returns ``{"id", "type", "name",
    "arguments"}`` with no ``function`` wrapper, and that is the shape the
    turn runner records in the assistant message.
    """
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_flat",
                    "type": "function",
                    "name": "a_tool",
                    "arguments": '{"a":1}{"b":2}',
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_flat", "content": "err"},
    ]
    normalized = normalize_openai_messages(messages)
    assert normalized[1]["tool_calls"][0]["function"]["arguments"] == "{}"


def test_responses_input_repairs_unparseable_tool_call_arguments() -> None:
    """The Responses path repairs the same payloads the chat path does."""

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_cut",
                    "type": "function",
                    "function": {
                        "name": "a_tool",
                        "arguments": '{"path": "a.md", "content": "2024-01-02',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_cut", "content": "err"},
    ]
    items = build_openai_responses_input(messages)
    function_calls = [item for item in items if item.get("type") == "function_call"]
    assert len(function_calls) == 1
    assert function_calls[0]["arguments"] == "{}"


def test_repair_rejects_an_integer_beyond_the_float64_range() -> None:
    """An integer too large for a float64 is rejected by a strict backend.

    Python's ``int`` is unbounded, so ``json`` accepts and re-emits a
    310-digit integer happily. A strict backend decodes JSON numbers as
    float64 and rejects anything beyond that range with ``400 invalid
    tool call arguments`` — the exact error the repair exists to
    prevent.
    """
    assert _repair_openai_tool_arguments('{"x": %s}' % ("1" * 310)) == "{}"
    assert _repair_openai_tool_arguments('{"x": %s}' % ("1" * 400)) == "{}"


def test_repair_keeps_an_integer_within_the_float64_range() -> None:
    """A large but representable integer survives untouched."""

    assert _repair_openai_tool_arguments('{"x": 9007199254740991}') == ('{"x": 9007199254740991}')


def test_repair_rejects_a_lone_surrogate() -> None:
    """A lone surrogate crashes the transport's UTF-8 encoder.

    ``httpx`` encodes with ``ensure_ascii=False`` and then UTF-8 encodes
    the result, so a lone surrogate — which a cut multi-byte character
    can produce — raises ``UnicodeEncodeError`` before the request is
    made. A round trip with ``ensure_ascii=True`` does not see it,
    because that encoder escapes the surrogate back into ASCII.
    """
    assert _repair_openai_tool_arguments('{"text": "\\ud800"}') == "{}"
    assert _coerce_tool_arguments_object('{"text": "\\ud800"}') == {}


def test_coerce_tool_arguments_object_rejects_an_oversized_integer() -> None:
    """The shared coercion rejects an integer beyond the float64 range."""

    assert _coerce_tool_arguments_object('{"x": %s}' % ("1" * 310)) == {}


def test_repair_keeps_a_large_64_bit_integer() -> None:
    """A 64-bit id is a legitimate tool argument and must survive.

    The backend's decoder is float64-based, so it rejects an integer
    beyond the float64 range. It does NOT reject a 64-bit id: the
    boundary was observed live at 309 versus 310 digits. Rejecting
    anything above 2**53 silently replaced real arguments — a snowflake
    id, a nanosecond timestamp — with an empty object.
    """
    for value in (
        9007199254740993,
        1234567890123456789,
        1750000000000000000,
        18446744073709551615,
    ):
        payload = f'{{"id": {value}}}'
        assert _repair_openai_tool_arguments(payload) == payload


def test_repair_keeps_an_integer_below_the_float64_boundary() -> None:
    """A 309-digit integer is still inside the float64 range."""

    digits = "1" * 309
    payload = f'{{"id": {digits}}}'
    assert _repair_openai_tool_arguments(payload) == payload


def test_repair_degrades_a_too_deeply_nested_payload() -> None:
    """A payload too deep for the JSON parser is repaired, not raised.

    The request builders run OUTSIDE the gateways' ``try`` blocks, so an
    escaping ``RecursionError`` reached the host as an unhandled error
    instead of a provider error.
    """
    deep = '{"a":' * 20000 + "1" + "}" * 20000
    assert _repair_openai_tool_arguments(deep) == "{}"
    assert _coerce_tool_arguments_object(deep) == {}

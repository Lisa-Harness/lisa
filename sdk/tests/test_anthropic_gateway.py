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
from sdk.providers import SdkProviderGatewayProvider
from sdk.providers.anthropic_gateway import AnthropicGateway
from sdk.providers.openai_gateway import normalize_openai_messages

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


def _text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(block_id: str, name: str, input_obj: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=input_obj)


def _response(
    blocks: list[SimpleNamespace],
    *,
    input_tokens: int = 10,
    output_tokens: int = 20,
    response_id: str = "resp_123",
) -> SimpleNamespace:
    return SimpleNamespace(
        content=blocks,
        id=response_id,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )


class _StubMessages:
    """Minimal stub for the Anthropic client's ``messages.create``."""

    def __init__(self, response: SimpleNamespace, captured: dict[str, Any]) -> None:
        self._response = response
        self.captured = captured

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.captured.update(kwargs)
        return self._response


class _StubAnthropicClient:
    """Stub client exposing ``.messages`` for ``AnthropicGateway``."""

    def __init__(self, response: SimpleNamespace, captured: dict[str, Any]) -> None:
        self.messages = _StubMessages(response, captured)


def _gateway(
    response: SimpleNamespace,
    captured: dict[str, Any],
) -> AnthropicGateway:
    client = _StubAnthropicClient(response, captured)
    return AnthropicGateway(client_factory=lambda: client)


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
    assert "tools" in captured
    assert captured["tools"] == [
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
    anthropic_messages = captured["messages"]
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
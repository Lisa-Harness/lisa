"""Tests for the Anthropic gateway tool-use path and provider routing.

Covers ``AnthropicGateway.complete`` translation of OpenAI-format messages
and tools into Anthropic format, extraction of ``text``/``tool_use`` content
blocks back into the normalized ``CompletionResult`` shape, and
``KernelProviderGatewayProvider.gateway_for`` routing for ``anthropic``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from kernel.context.provider_gateway_provider import KernelProviderGatewayProvider
from kernel.provider_core import (
    AnthropicGateway,
    CompletionRequest,
    CompletionResult,
    ModelDescriptor,
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


# ── Temperature passed through ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_temperature_passed_when_set() -> None:
    captured: dict[str, Any] = {}
    gateway = _gateway(_response([_text_block("ok")]), captured)
    await gateway.complete(_request([{"role": "user", "content": "hi"}], temperature=0.7))
    assert captured["temperature"] == 0.7


@pytest.mark.asyncio
async def test_temperature_omitted_when_zero() -> None:
    captured: dict[str, Any] = {}
    gateway = _gateway(_response([_text_block("ok")]), captured)
    await gateway.complete(_request([{"role": "user", "content": "hi"}], temperature=0.0))
    assert "temperature" not in captured


# ── System message extracted to top-level system param ─────────────────


@pytest.mark.asyncio
async def test_system_message_extracted() -> None:
    captured: dict[str, Any] = {}
    gateway = _gateway(_response([_text_block("ok")]), captured)
    await gateway.complete(
        _request(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"},
            ]
        )
    )
    assert captured["system"] == "You are helpful."
    assert all(m["role"] != "system" for m in captured["messages"])


# ── Usage extraction ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_usage_extracted_from_anthropic_response() -> None:
    captured: dict[str, Any] = {}
    gateway = _gateway(_response([_text_block("ok")], input_tokens=42, output_tokens=58), captured)
    result = await gateway.complete(_request([{"role": "user", "content": "hi"}]))
    assert result.usage is not None
    assert result.usage.prompt_tokens == 42
    assert result.usage.completion_tokens == 58
    assert result.usage.total_tokens == 100


# ── Provider routing ───────────────────────────────────────────────────


def test_gateway_for_anthropic_returns_anthropic_gateway() -> None:
    provider = KernelProviderGatewayProvider()
    gateway_cls = provider.gateway_for("anthropic")
    assert gateway_cls is AnthropicGateway


def test_gateway_for_openai_returns_openai_compatible_gateway() -> None:
    from kernel.provider_core import OpenAICompatibleGateway

    provider = KernelProviderGatewayProvider()
    assert provider.gateway_for("openai") is OpenAICompatibleGateway


def test_gateway_for_unknown_returns_none() -> None:
    provider = KernelProviderGatewayProvider()
    assert provider.gateway_for("not-a-provider") is None

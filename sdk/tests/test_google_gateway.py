"""Tests for the Google Gemini gateway (Part 1.1).

Covers ``GoogleGeminiGateway.complete`` message translation, tool-use
extraction, system-prompt extraction, usage extraction, embed, and
``SdkProviderGatewayProvider.gateway_for`` routing for ``google``.
"""

from __future__ import annotations

from typing import Any

import pytest
from kernel.provider_contracts import (
    CompletionRequest,
    CompletionResult,
)
from pydantic import BaseModel
from sdk.providers import SdkProviderGatewayProvider
from sdk.providers.google_gateway import GoogleGeminiGateway

# ── Test helpers ──────────────────────────────────────────────────────


def _model_def(provider: str = "google", model_id: str = "gemini-2.5-flash") -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        provider=provider,
        model_id=model_id,
        display_name="Gemini Test",
        reasoning_model=False,
        thinking_model=False,
    )


def _request(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    response_format: Any | None = None,
) -> CompletionRequest:
    return CompletionRequest(
        messages=messages,
        model_def=_model_def(),
        max_tokens=max_tokens,
        temperature=temperature,
        tools=tools,
        response_format=response_format,
    )


class _StubHttpxResponse:
    def __init__(self, json_data: dict[str, Any]) -> None:
        self._json_data = json_data

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._json_data


class _StubHttpxClient:
    """Stub httpx.AsyncClient that returns pre-configured responses."""

    def __init__(self, response_data: dict[str, Any]) -> None:
        self._response_data = response_data
        self.captured_url: str = ""
        self.captured_json: dict[str, Any] = {}
        self.captured_headers: dict[str, str] = {}

    async def post(
        self,
        url: str,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> _StubHttpxResponse:
        self.captured_url = url
        self.captured_json = json or {}
        self.captured_headers = headers or {}
        return _StubHttpxResponse(self._response_data)


def _gateway(response_data: dict[str, Any]) -> tuple[GoogleGeminiGateway, _StubHttpxClient]:
    client = _StubHttpxClient(response_data)
    gateway = GoogleGeminiGateway(api_key="test-key", client=client)
    return gateway, client


# ── Text-only response ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_text_only_response() -> None:
    data = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "hello world"}],
                }
            }
        ]
    }
    gateway, _ = _gateway(data)
    result = await gateway.complete(_request([{"role": "user", "content": "hi"}]))
    assert isinstance(result, CompletionResult)
    assert result.content == "hello world"
    assert result.tool_calls == []
    assert result.provider == "google"
    assert result.model_id == "gemini-2.5-flash"


# ── Tool-use response ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_use_response() -> None:
    data = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "Let me check."},
                        {"functionCall": {"name": "get_weather", "args": {"city": "Seattle"}}},
                    ]
                }
            }
        ]
    }
    gateway, _ = _gateway(data)
    result = await gateway.complete(_request([{"role": "user", "content": "weather?"}]))
    assert result.content == "Let me check."
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc["id"] == "gemini_call_1"
    assert tc["type"] == "function"
    assert tc["name"] == "get_weather"
    assert tc["arguments"] == '{"city": "Seattle"}'


# ── Tool-use only (no text) ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_use_only_no_text() -> None:
    data = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"functionCall": {"name": "search", "args": {"q": "cats"}}},
                    ]
                }
            }
        ]
    }
    gateway, _ = _gateway(data)
    result = await gateway.complete(_request([{"role": "user", "content": "go"}]))
    assert result.content == ""
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "search"


# ── Multi-block text concatenation ────────────────────────────────────


@pytest.mark.asyncio
async def test_multi_block_text_concatenated() -> None:
    data = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "part one"},
                        {"text": "part two"},
                    ]
                }
            }
        ]
    }
    gateway, _ = _gateway(data)
    result = await gateway.complete(_request([{"role": "user", "content": "x"}]))
    assert result.content == "part one\npart two"


# ── Tools passed to Gemini as functionDeclarations ────────────────────


@pytest.mark.asyncio
async def test_tools_translated_to_gemini_format() -> None:
    data = {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
    gateway, client = _gateway(data)
    openai_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
    await gateway.complete(_request([{"role": "user", "content": "hi"}], tools=[openai_tool]))
    assert "tools" in client.captured_json
    tools = client.captured_json["tools"]
    assert tools[0]["functionDeclarations"][0]["name"] == "get_weather"
    assert tools[0]["functionDeclarations"][0]["description"] == "Get the weather"


# ── Assistant + tool_calls round-trip in messages ────────────────────


@pytest.mark.asyncio
async def test_assistant_tool_calls_translated_to_gemini() -> None:
    data = {"candidates": [{"content": {"parts": [{"text": "done"}]}}]}
    gateway, client = _gateway(data)
    messages = [
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
            "name": "get_weather",
            "content": "sunny",
        },
    ]
    await gateway.complete(_request(list(messages)))
    contents = client.captured_json["contents"]
    # user message
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"][0]["text"] == "weather?"
    # assistant message becomes model with text + functionCall
    assert contents[1]["role"] == "model"
    part_types = []
    for p in contents[1]["parts"]:
        if "text" in p:
            part_types.append("text")
        elif "functionCall" in p:
            part_types.append("functionCall")
    assert part_types == ["text", "functionCall"]
    fc = contents[1]["parts"][1]["functionCall"]
    assert fc["name"] == "get_weather"
    assert fc["args"] == {"city": "Seattle"}
    # tool-role message becomes user with functionResponse
    assert contents[2]["role"] == "user"
    fr = contents[2]["parts"][0]["functionResponse"]
    assert fr["name"] == "get_weather"
    assert fr["response"]["result"] == "sunny"


# ── thoughtSignature captured from response and echoed on round-trip ─


@pytest.mark.asyncio
async def test_thought_signature_captured_from_response() -> None:
    """Gemini thinking models emit thoughtSignature as a sibling of functionCall.

    The gateway must capture it so the turn runner can echo it back on the
    next request; otherwise Gemini rejects the call with INVALID_ARGUMENT.
    """
    data = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {"name": "search", "args": {"q": "cats"}},
                            "thoughtSignature": "sig-abc-123",
                        }
                    ]
                }
            }
        ]
    }
    gateway, _ = _gateway(data)
    result = await gateway.complete(_request([{"role": "user", "content": "go"}]))
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["thought_signature"] == "sig-abc-123"


@pytest.mark.asyncio
async def test_thought_signature_echoed_in_function_call_part() -> None:
    """When rebuilding assistant→model contents, thoughtSignature must be
    emitted as a sibling of functionCall, not nested inside it.
    """
    data = {"candidates": [{"content": {"parts": [{"text": "done"}]}}]}
    gateway, client = _gateway(data)
    messages = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "Checking.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "thought_signature": "sig-xyz",
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
            "name": "get_weather",
            "content": "sunny",
        },
    ]
    await gateway.complete(_request(list(messages)))
    contents = client.captured_json["contents"]
    fc_part = contents[1]["parts"][1]
    assert "functionCall" in fc_part
    assert fc_part["thoughtSignature"] == "sig-xyz"
    # signature must NOT be nested inside functionCall
    assert "thoughtSignature" not in fc_part["functionCall"]


# ── System message extracted to systemInstruction ─────────────────────


@pytest.mark.asyncio
async def test_system_message_extracted() -> None:
    data = {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
    gateway, client = _gateway(data)
    await gateway.complete(
        _request(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"},
            ]
        )
    )
    assert "systemInstruction" in client.captured_json
    assert client.captured_json["systemInstruction"]["parts"][0]["text"] == "You are helpful."
    # system message should not appear in contents
    for c in client.captured_json["contents"]:
        assert c["role"] != "system"


@pytest.mark.asyncio
async def test_generation_config_and_all_system_messages_are_sent() -> None:
    data = {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
    gateway, client = _gateway(data)
    await gateway.complete(
        _request(
            [
                {"role": "system", "content": "First instruction."},
                {"role": "system", "content": "Second instruction."},
                {"role": "user", "content": "hi"},
            ],
            max_tokens=4096,
            temperature=0.35,
        )
    )
    assert client.captured_json["generationConfig"] == {
        "maxOutputTokens": 4096,
        "temperature": 0.35,
    }
    assert client.captured_json["systemInstruction"]["parts"][0]["text"] == (
        "First instruction.\n\nSecond instruction."
    )
    assert "?key=" not in client.captured_url
    assert client.captured_headers["x-goog-api-key"] == "test-key"


class _StructuredReply(BaseModel):
    answer: str
    count: int | None = None


@pytest.mark.asyncio
async def test_structured_output_uses_json_mode_schema_and_normalizes_content() -> None:
    data = {"candidates": [{"content": {"parts": [{"text": '```json\n{"answer": "ok"}\n```'}]}}]}
    gateway, client = _gateway(data)
    result = await gateway.complete(
        _request(
            [{"role": "user", "content": "hi"}],
            response_format=_StructuredReply,
        )
    )
    generation_config = client.captured_json["generationConfig"]
    assert generation_config["responseMimeType"] == "application/json"
    assert generation_config["responseSchema"] == {
        "type": "OBJECT",
        "properties": {
            "answer": {"type": "STRING"},
            "count": {"type": "INTEGER", "nullable": True},
        },
        "required": ["answer"],
    }
    assert client.captured_json["systemInstruction"]["parts"][0]["text"].endswith(
        "Return only a valid JSON object with no Markdown wrapper."
    )
    assert result.content == '{"answer":"ok","count":null}'


# ── Usage extraction ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_usage_extracted_from_gemini_response() -> None:
    data = {
        "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
        "usageMetadata": {
            "promptTokenCount": 42,
            "candidatesTokenCount": 58,
            "totalTokenCount": 100,
            "thoughtsTokenCount": 13,
        },
    }
    gateway, _ = _gateway(data)
    result = await gateway.complete(_request([{"role": "user", "content": "hi"}]))
    assert result.usage is not None
    assert result.usage.prompt_tokens == 42
    assert result.usage.completion_tokens == 58
    assert result.usage.total_tokens == 100
    assert result.usage.reasoning_tokens == 13


@pytest.mark.asyncio
async def test_finish_reason_and_reasoning_content_are_normalized() -> None:
    data = {
        "candidates": [
            {
                "finishReason": "MAX_TOKENS",
                "content": {
                    "parts": [
                        {"text": "hidden", "thought": True},
                        {"text": "shown"},
                    ]
                },
            }
        ]
    }
    gateway, _ = _gateway(data)
    result = await gateway.complete(_request([{"role": "user", "content": "hi"}]))
    assert result.content == "shown"
    assert result.reasoning_content == "hidden"
    assert result.finish_reason == "length"
    assert result.truncated is True


# ── supports() static method ─────────────────────────────────────────


def test_supports_gemini_model() -> None:
    assert GoogleGeminiGateway.supports("gemini-2.5-flash") is True
    assert GoogleGeminiGateway.supports("gemini-2.5-pro") is True
    assert GoogleGeminiGateway.supports("gpt-4") is False


# ── Provider routing ──────────────────────────────────────────────────


def test_gateway_for_google_returns_gemini_gateway() -> None:
    provider = SdkProviderGatewayProvider()
    assert provider.gateway_for("google") is GoogleGeminiGateway


def test_gateway_for_anthropic_still_returns_anthropic_gateway() -> None:
    from sdk.providers.anthropic_gateway import AnthropicGateway

    provider = SdkProviderGatewayProvider()
    assert provider.gateway_for("anthropic") is AnthropicGateway


@pytest.mark.asyncio
async def test_gemini_tool_call_args_are_coerced_to_an_object() -> None:
    """A payload that is not a usable JSON object never reaches Gemini.

    The three non-OpenAI gateways each parsed the arguments with their
    own ``except json.JSONDecodeError`` handler, so a payload Python
    accepts but a strict backend rejects (``NaN``) reached the wire and
    crashed ``httpx`` before any request was made.
    """
    data = {"candidates": [{"content": {"parts": [{"text": "done"}]}}]}
    gateway, client = _gateway(data)
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_nan",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"x": NaN}'},
                }
            ],
        },
    ]
    await gateway.complete(_request(list(messages)))
    contents = client.captured_json["contents"]
    fc = contents[1]["parts"][0]["functionCall"]
    assert fc["args"] == {}

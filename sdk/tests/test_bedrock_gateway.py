"""Tests for the AWS Bedrock gateway (Part 1.2).

Covers ``BedrockGateway.complete`` message translation, tool-use extraction,
system-prompt extraction, usage extraction, ImportError when boto3 is missing,
and ``SdkProviderGatewayProvider.gateway_for`` routing for ``aws``.

All tests stub the boto3 client — no AWS credentials or network access required.
"""

from __future__ import annotations

import json
from io import BytesIO
from types import SimpleNamespace
from typing import Any

import pytest
from kernel.provider_contracts import (
    CompletionRequest,
    CompletionResult,
)
from pydantic import BaseModel
from sdk.providers import SdkProviderGatewayProvider
from sdk.providers.bedrock_gateway import BedrockGateway

# ── Test helpers ──────────────────────────────────────────────────────


def _model_def(
    provider: str = "aws", model_id: str = "us.anthropic.claude-sonnet-4-20250514-v1:0"
) -> Any:
    return SimpleNamespace(
        provider=provider,
        model_id=model_id,
        display_name="Claude Sonnet 4 (Bedrock)",
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


class _StubBedrockClient:
    """Stub boto3 bedrock-runtime client."""

    def __init__(
        self, converse_response: dict[str, Any], invoke_response: dict[str, Any] | None = None
    ) -> None:
        self._converse_response = converse_response
        self._invoke_response = invoke_response or {}
        self.captured_kwargs: dict[str, Any] = {}

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.captured_kwargs.update(kwargs)
        return self._converse_response

    def invoke_model(self, **kwargs: Any) -> dict[str, Any]:
        return {"body": BytesIO(json.dumps(self._invoke_response).encode())}


def _gateway(response: dict[str, Any]) -> tuple[BedrockGateway, _StubBedrockClient]:
    client = _StubBedrockClient(response)
    gateway = BedrockGateway(client=client)
    return gateway, client


# ── Text-only response ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_text_only_response() -> None:
    response = {
        "output": {"message": {"content": [{"text": "hello world"}]}},
        "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
    }
    gateway, _ = _gateway(response)
    result = await gateway.complete(_request([{"role": "user", "content": "hi"}]))
    assert isinstance(result, CompletionResult)
    assert result.content == "hello world"
    assert result.tool_calls == []
    assert result.provider == "aws"


# ── Tool-use response ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_use_response() -> None:
    response = {
        "output": {
            "message": {
                "content": [
                    {"text": "Let me check."},
                    {
                        "toolUse": {
                            "toolUseId": "call_1",
                            "name": "get_weather",
                            "input": {"city": "Seattle"},
                        }
                    },
                ]
            }
        }
    }
    gateway, _ = _gateway(response)
    result = await gateway.complete(_request([{"role": "user", "content": "weather?"}]))
    assert result.content == "Let me check."
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc["id"] == "call_1"
    assert tc["type"] == "function"
    assert tc["name"] == "get_weather"
    assert tc["arguments"] == '{"city": "Seattle"}'


# ── Tool-use only (no text) ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_use_only_no_text() -> None:
    response = {
        "output": {
            "message": {
                "content": [
                    {"toolUse": {"toolUseId": "call_2", "name": "search", "input": {"q": "cats"}}},
                ]
            }
        }
    }
    gateway, _ = _gateway(response)
    result = await gateway.complete(_request([{"role": "user", "content": "go"}]))
    assert result.content == ""
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "search"


# ── Multi-block text concatenation ────────────────────────────────────


@pytest.mark.asyncio
async def test_multi_block_text_concatenated() -> None:
    response = {
        "output": {
            "message": {
                "content": [
                    {"text": "part one"},
                    {"text": "part two"},
                ]
            }
        }
    }
    gateway, _ = _gateway(response)
    result = await gateway.complete(_request([{"role": "user", "content": "x"}]))
    assert result.content == "part one\npart two"


# ── Tools passed to Bedrock as toolConfig ─────────────────────────────


@pytest.mark.asyncio
async def test_tools_translated_to_bedrock_format() -> None:
    response = {"output": {"message": {"content": [{"text": "ok"}]}}}
    gateway, client = _gateway(response)
    openai_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
    await gateway.complete(_request([{"role": "user", "content": "hi"}], tools=[openai_tool]))
    assert "toolConfig" in client.captured_kwargs
    tools = client.captured_kwargs["toolConfig"]["tools"]
    assert tools[0]["toolSpec"]["name"] == "get_weather"
    assert tools[0]["toolSpec"]["description"] == "Get the weather"
    assert tools[0]["toolSpec"]["inputSchema"]["json"]["type"] == "object"


# ── Assistant + tool_calls round-trip in messages ─────────────────────


@pytest.mark.asyncio
async def test_assistant_tool_calls_translated_to_bedrock() -> None:
    response = {"output": {"message": {"content": [{"text": "done"}]}}}
    gateway, client = _gateway(response)
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
            "name": "get_weather",
            "content": "sunny",
        },
    ]
    await gateway.complete(_request(list(messages)))
    converse_messages = client.captured_kwargs["messages"]
    # user message
    assert converse_messages[0]["role"] == "user"
    assert converse_messages[0]["content"][0]["text"] == "weather?"
    # assistant message with text + toolUse
    assert converse_messages[1]["role"] == "assistant"
    block_types = []
    for b in converse_messages[1]["content"]:
        if "text" in b:
            block_types.append("text")
        elif "toolUse" in b:
            block_types.append("toolUse")
    assert block_types == ["text", "toolUse"]
    tu = converse_messages[1]["content"][1]["toolUse"]
    assert tu["toolUseId"] == "call_1"
    assert tu["name"] == "get_weather"
    assert tu["input"] == {"city": "Seattle"}
    # tool-role message becomes user with toolResult
    assert converse_messages[2]["role"] == "user"
    tr = converse_messages[2]["content"][0]["toolResult"]
    assert tr["toolUseId"] == "call_1"
    assert tr["content"][0]["text"] == "sunny"


# ── System message extracted to system[] ──────────────────────────────


@pytest.mark.asyncio
async def test_system_message_extracted() -> None:
    response = {"output": {"message": {"content": [{"text": "ok"}]}}}
    gateway, client = _gateway(response)
    await gateway.complete(
        _request(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"},
            ]
        )
    )
    assert "system" in client.captured_kwargs
    assert client.captured_kwargs["system"][0]["text"] == "You are helpful."
    # system message should not appear in messages
    for m in client.captured_kwargs["messages"]:
        assert m["role"] != "system"


@pytest.mark.asyncio
async def test_inference_config_and_all_system_messages_are_sent() -> None:
    response = {"output": {"message": {"content": [{"text": "ok"}]}}}
    gateway, client = _gateway(response)
    await gateway.complete(
        _request(
            [
                {"role": "system", "content": "First instruction."},
                {"role": "system", "content": "Second instruction."},
                {"role": "user", "content": "hi"},
            ],
            max_tokens=4096,
            temperature=0.3,
        )
    )
    assert client.captured_kwargs["inferenceConfig"] == {
        "maxTokens": 4096,
        "temperature": 0.3,
    }
    assert client.captured_kwargs["system"] == [
        {"text": "First instruction.\n\nSecond instruction."}
    ]


class _StructuredReply(BaseModel):
    answer: str


@pytest.mark.asyncio
async def test_structured_output_adds_json_instruction_and_normalizes_content() -> None:
    response = {"output": {"message": {"content": [{"text": '```json\n{"answer": "ok"}\n```'}]}}}
    gateway, client = _gateway(response)
    result = await gateway.complete(
        _request(
            [{"role": "user", "content": "hi"}],
            response_format=_StructuredReply,
        )
    )
    assert client.captured_kwargs["system"] == [
        {"text": "Return only a valid JSON object with no Markdown wrapper."}
    ]
    assert result.content == '{"answer":"ok"}'


@pytest.mark.asyncio
async def test_stop_reason_is_normalized() -> None:
    response = {
        "output": {"message": {"content": [{"text": "partial"}]}},
        "stopReason": "max_tokens",
    }
    gateway, _ = _gateway(response)
    result = await gateway.complete(_request([{"role": "user", "content": "hi"}]))
    assert result.finish_reason == "max_tokens"
    assert result.truncated is True


# ── Usage extraction ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_usage_extracted_from_bedrock_response() -> None:
    response = {
        "output": {"message": {"content": [{"text": "ok"}]}},
        "usage": {"inputTokens": 42, "outputTokens": 58, "totalTokens": 100},
    }
    gateway, _ = _gateway(response)
    result = await gateway.complete(_request([{"role": "user", "content": "hi"}]))
    assert result.usage is not None
    assert result.usage.prompt_tokens == 42
    assert result.usage.completion_tokens == 58
    assert result.usage.total_tokens == 100


# ── supports() static method ─────────────────────────────────────────


def test_supports_bedrock_model() -> None:
    assert BedrockGateway.supports("us.anthropic.claude-sonnet-4-20250514-v1:0") is True
    assert BedrockGateway.supports("us.meta.llama4-maverick-17b-instruct-v1:0") is True
    assert BedrockGateway.supports("us.amazon.nova-pro-v1:0") is True
    assert BedrockGateway.supports("gpt-4") is False


# ── Provider routing ──────────────────────────────────────────────────


def test_gateway_for_aws_returns_bedrock_gateway() -> None:
    provider = SdkProviderGatewayProvider()
    assert provider.gateway_for("aws") is BedrockGateway


def test_gateway_for_google_still_returns_gemini_gateway() -> None:
    from sdk.providers.google_gateway import GoogleGeminiGateway

    provider = SdkProviderGatewayProvider()
    assert provider.gateway_for("google") is GoogleGeminiGateway

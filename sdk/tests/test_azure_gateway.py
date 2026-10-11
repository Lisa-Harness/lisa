"""Tests for the Azure OpenAI gateway.

Covers the dual-client split (chat-completions on the GA api-version vs.
the Responses API on a preview api-version), the Responses tool-name
sanitize/restore round-trip, and the ``max_completion_tokens`` special-case
for Azure deployments (which accept it regardless of the reasoning flag).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from kernel.provider_contracts import CompletionRequest
from sdk.providers.openai_gateway import (
    AzureOpenAIGateway,
    build_openai_completion_kwargs,
    build_responses_tool_name_map,
    restore_responses_tool_names,
)


def _model_def(**overrides: Any) -> Any:
    base = {
        "provider": "azure_openai",
        "model_id": "gpt-5.4",
        "display_name": "GPT-5.4",
        "reasoning_model": False,
        "thinking_model": False,
        "completion_api": "chat_completions",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _request(
    messages: list[dict[str, Any]] | None = None,
    *,
    tools: list[dict[str, Any]] | None = None,
    model_def: Any | None = None,
) -> CompletionRequest:
    return CompletionRequest(
        messages=messages or [{"role": "user", "content": "hi"}],
        model_def=model_def or _model_def(),
        max_tokens=1024,
        temperature=0.4,
        tools=tools,
    )


# ── Stub clients ─────────────────────────────────────────────────────


class _StubChatCompletions:
    def __init__(self, response: Any, captured: dict[str, Any]) -> None:
        self._response = response
        self.captured = captured

    async def create(self, **kwargs: Any) -> Any:
        self.captured["chat_kwargs"] = kwargs
        return self._response


class _StubResponses:
    def __init__(self, response: Any, captured: dict[str, Any]) -> None:
        self._response = response
        self.captured = captured

    async def create(self, **kwargs: Any) -> Any:
        self.captured["responses_kwargs"] = kwargs
        return self._response


class _StubAzureClient:
    """Stub exposing only the surface the gateway touches for one path."""

    def __init__(
        self,
        *,
        chat_response: Any = None,
        responses_response: Any = None,
        captured: dict[str, Any] | None = None,
    ) -> None:
        self.captured = captured if captured is not None else {}
        if chat_response is not None:
            self.chat = SimpleNamespace(
                completions=_StubChatCompletions(chat_response, self.captured)
            )
        if responses_response is not None:
            self.responses = _StubResponses(responses_response, self.captured)


def _chat_response(content: str = "ok") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        id="chat-1",
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=7),
    )


def _responses_response(tool_name: str = "get_weather") -> SimpleNamespace:
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                id="fc_1",
                call_id="call_1",
                name=tool_name,
                arguments='{"city": "Seattle"}',
            )
        ],
        id="resp-1",
        usage=SimpleNamespace(input_tokens=5, output_tokens=7),
    )


# ── Dual-client split ────────────────────────────────────────────────


class TestDualClientSplit:
    @pytest.mark.asyncio
    async def test_chat_completions_uses_complete_client(self) -> None:
        captured: dict[str, Any] = {}
        chat_client = _StubAzureClient(chat_response=_chat_response(), captured=captured)
        responses_client = _StubAzureClient(responses_response=_responses_response())
        gateway = AzureOpenAIGateway(
            complete_client_factory=lambda: chat_client,
            responses_client_factory=lambda: responses_client,
        )
        await gateway.complete(_request())
        assert "chat_kwargs" in captured
        assert "responses_kwargs" not in captured

    @pytest.mark.asyncio
    async def test_responses_api_uses_responses_client(self) -> None:
        captured: dict[str, Any] = {}
        chat_client = _StubAzureClient(chat_response=_chat_response(), captured=captured)
        responses_client = _StubAzureClient(
            responses_response=_responses_response(), captured=captured
        )
        model_def = _model_def(completion_api="responses")
        gateway = AzureOpenAIGateway(
            complete_client_factory=lambda: chat_client,
            responses_client_factory=lambda: responses_client,
        )
        result = await gateway.complete(_request(model_def=model_def))
        assert "responses_kwargs" in captured
        assert "chat_kwargs" not in captured
        # The sanitized wire name is restored to the original.
        assert result.tool_calls[0]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_responses_falls_back_to_complete_client_when_no_factory(self) -> None:
        captured: dict[str, Any] = {}
        client = _StubAzureClient(
            chat_response=_chat_response(),
            responses_response=_responses_response(),
            captured=captured,
        )
        gateway = AzureOpenAIGateway(complete_client_factory=lambda: client)
        result = await gateway.complete(_request(model_def=_model_def(completion_api="responses")))
        assert "responses_kwargs" in captured
        assert result.tool_calls[0]["name"] == "get_weather"


# ── Tool-name sanitize/restore round-trip ────────────────────────────


class TestToolNameRoundTrip:
    def test_build_map_sanitizes_dotted_names(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {"name": "lisa-storage.read_file", "description": "", "parameters": {}},
            }
        ]
        name_map = build_responses_tool_name_map(tools)
        assert "lisa-storage_read_file" in name_map
        assert name_map["lisa-storage_read_file"] == "lisa-storage.read_file"

    def test_restore_maps_back_to_original(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {"name": "a.b", "description": "", "parameters": {}},
            }
        ]
        name_map = build_responses_tool_name_map(tools)
        calls = [{"id": "1", "type": "function", "name": "a_b", "arguments": "{}"}]
        restored = restore_responses_tool_names(calls, name_map)
        assert restored[0]["name"] == "a.b"


# ── max_completion_tokens special case ───────────────────────────────


class TestMaxCompletionTokens:
    def test_azure_non_reasoning_gets_max_completion_tokens(self) -> None:
        kwargs = build_openai_completion_kwargs(_request(model_def=_model_def()))
        assert kwargs["max_completion_tokens"] == 1024
        assert "max_tokens" not in kwargs
        # Azure rejects ``temperature`` alongside max_completion_tokens on
        # some deployments — legacy omitted it in this branch.
        assert "temperature" not in kwargs

    def test_public_openai_non_reasoning_gets_max_tokens(self) -> None:
        kwargs = build_openai_completion_kwargs(_request(model_def=_model_def(provider="openai")))
        assert kwargs["max_tokens"] == 1024
        assert "max_completion_tokens" not in kwargs

    def test_public_openai_reasoning_gets_max_completion_tokens(self) -> None:
        kwargs = build_openai_completion_kwargs(
            _request(model_def=_model_def(provider="openai", reasoning_model=True))
        )
        assert kwargs["max_completion_tokens"] == 1024


# ── prompt_cache_key ─────────────────────────────────────────────────


class _RejectingChatCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if "prompt_cache_key" in kwargs:
            raise ValueError("Unrecognized request argument supplied: prompt_cache_key")
        return _chat_response()


class TestPromptCacheKey:
    def test_key_sent_to_azure_and_openai_only(self) -> None:
        def keyed(provider: str) -> CompletionRequest:
            request = _request(model_def=_model_def(provider=provider, model_id="m-key"))
            request.prompt_cache_key = "lisa-abc"
            return request

        assert build_openai_completion_kwargs(keyed("azure_openai"))["prompt_cache_key"] == (
            "lisa-abc"
        )
        assert build_openai_completion_kwargs(keyed("openai"))["prompt_cache_key"] == "lisa-abc"
        assert "prompt_cache_key" not in build_openai_completion_kwargs(keyed("ollama"))
        assert "prompt_cache_key" not in build_openai_completion_kwargs(_request())

    @pytest.mark.asyncio
    async def test_rejecting_deployment_retries_once_then_stops_sending(self) -> None:
        completions = _RejectingChatCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        gateway = AzureOpenAIGateway(complete_client_factory=lambda: client)
        request = _request(model_def=_model_def(model_id="legacy-deployment"))
        request.prompt_cache_key = "lisa-abc"

        result = await gateway.complete(request)
        assert result.content == "ok"
        assert ["prompt_cache_key" in call for call in completions.calls] == [True, False]

        await gateway.complete(request)
        assert "prompt_cache_key" not in completions.calls[-1]
        assert len(completions.calls) == 3

"""Tests for the Azure OpenAI gateway.

Covers the shared Azure v1 client for chat and Responses, the Responses tool-name
sanitize/restore round-trip, and the ``max_completion_tokens`` special-case
for Azure deployments (which accept it regardless of the reasoning flag).
"""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from kernel.provider_contracts import CompletionRequest, SystemPromptSegment
from sdk.providers.google_gateway import _build_gemini_request_body
from sdk.providers.openai_gateway import (
    AzureOpenAIGateway,
    add_prompt_cache_breakpoints,
    build_openai_completion_kwargs,
    build_openai_responses_kwargs,
    build_responses_tool_name_map,
    create_azure_openai_client,
    extract_openai_completion_usage,
    extract_responses_usage,
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


# ── Shared v1 client ─────────────────────────────────────────────────


class TestSharedClient:
    @pytest.mark.asyncio
    async def test_chat_completions_uses_complete_client(self) -> None:
        captured: dict[str, Any] = {}
        client = _StubAzureClient(
            chat_response=_chat_response(),
            responses_response=_responses_response(),
            captured=captured,
        )
        gateway = AzureOpenAIGateway(complete_client_factory=lambda: client)
        await gateway.complete(_request())
        assert "chat_kwargs" in captured
        assert "responses_kwargs" not in captured

    @pytest.mark.asyncio
    async def test_responses_api_uses_shared_client(self) -> None:
        captured: dict[str, Any] = {}
        client = _StubAzureClient(
            chat_response=_chat_response(),
            responses_response=_responses_response(),
            captured=captured,
        )
        model_def = _model_def(completion_api="responses")
        gateway = AzureOpenAIGateway(complete_client_factory=lambda: client)
        result = await gateway.complete(_request(model_def=model_def))
        assert "responses_kwargs" in captured
        assert "chat_kwargs" not in captured
        assert captured["responses_kwargs"]["store"] is False
        assert captured["responses_kwargs"]["include"] == ["reasoning.encrypted_content"]
        # The sanitized wire name is restored to the original.
        assert result.tool_calls[0]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_same_gateway_handles_chat_then_responses(self) -> None:
        captured: dict[str, Any] = {}
        client = _StubAzureClient(
            chat_response=_chat_response(),
            responses_response=_responses_response(),
            captured=captured,
        )
        gateway = AzureOpenAIGateway(complete_client_factory=lambda: client)
        await gateway.complete(_request())
        result = await gateway.complete(_request(model_def=_model_def(completion_api="responses")))
        assert "responses_kwargs" in captured
        assert result.tool_calls[0]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_encrypted_reasoning_replays_before_tools_only_to_its_model() -> None:
    captured: dict[str, Any] = {}
    response = _responses_response()
    reasoning = {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [],
        "encrypted_content": "opaque-encrypted-context",
    }
    response.output.insert(0, SimpleNamespace(**reasoning))
    gateway = AzureOpenAIGateway(
        complete_client_factory=lambda: _StubAzureClient(
            responses_response=response, captured=captured
        )
    )
    model = _model_def(completion_api="responses", reasoning_model=True)
    result = await gateway.complete(_request(model_def=model))
    assert result.continuation is not None
    messages = [
        {"role": "user", "content": "check weather"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                }
            ],
            "provider_continuation": asdict(result.continuation),
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ]
    kwargs = build_openai_responses_kwargs(_request(messages, model_def=model))
    assert [item["type"] for item in kwargs["input"]] == [
        "message",
        "reasoning",
        "function_call",
        "function_call_output",
    ]
    assert kwargs["input"][1] == reasoning
    assert "opaque-encrypted-context" not in result.content
    for other_model in (
        _model_def(model_id="different", completion_api="responses"),
        _model_def(provider="openai", completion_api="responses"),
    ):
        other = build_openai_responses_kwargs(_request(messages, model_def=other_model))
        assert all(item["type"] != "reasoning" for item in other["input"])
    chat = build_openai_completion_kwargs(_request(messages, model_def=model))
    assert all("provider_continuation" not in message for message in chat["messages"])
    gemini = _build_gemini_request_body(
        _request(messages, model_def=_model_def(provider="google", model_id="gemini-test"))
    )
    assert "opaque-encrypted-context" not in str(gemini)
    assert messages[1]["provider_continuation"]["items"] == [reasoning]


@pytest.mark.asyncio
async def test_azure_v1_sends_chat_responses_and_embeddings_without_api_version() -> None:
    captured: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if request.url.path.endswith("/chat/completions"):
            body = {
                "id": "chat",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
            }
        elif request.url.path.endswith("/responses"):
            body = {"id": "resp", "output": []}
        else:
            body = {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}
        return httpx.Response(200, json=body)

    original = create_azure_openai_client(
        endpoint="https://resource.services.ai.azure.com", api_key="test-key"
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport:
        client = original.with_options(http_client=transport)
        try:
            await client.chat.completions.create(model="deployment", messages=[])
            await client.responses.create(model="deployment", input="hi")
            await client.embeddings.create(model="embedding-deployment", input=["hi"])
        finally:
            await client.close()
            await original.close()
    assert [request.url.path for request in captured] == [
        "/openai/v1/chat/completions",
        "/openai/v1/responses",
        "/openai/v1/embeddings",
    ]
    assert all(not request.url.query for request in captured)
    assert all(request.headers["authorization"] == "Bearer test-key" for request in captured)


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


# ── prompt-cache breakpoints (Azure GPT-5.6+) ────────────────────────


_BP = {"mode": "explicit"}


def _breakpointed(message: dict[str, Any]) -> bool:
    content = message.get("content")
    return isinstance(content, list) and any(
        isinstance(part, dict) and "prompt_cache_breakpoint" in part for part in content
    )


def _breakpoint_model(**overrides: Any) -> Any:
    return _model_def(
        **{"model_id": "gpt-5.6-terra", "prompt_cache_breakpoints": True, **overrides}
    )


class TestPromptCacheBreakpoints:
    def test_system_split_at_stable_prefix(self) -> None:
        segments = (
            SystemPromptSegment("persona"),
            SystemPromptSegment("tools guide"),
            SystemPromptSegment("routing hint", stability="ephemeral"),
        )
        messages = [
            {"role": "system", "content": "persona\n\ntools guide\n\nrouting hint"},
            {"role": "user", "content": "hi"},
        ]
        marked = add_prompt_cache_breakpoints(messages, segments)
        blocks = marked[0]["content"]
        assert "".join(block["text"] for block in blocks) == messages[0]["content"]
        assert [block.get("prompt_cache_breakpoint") for block in blocks] == [None, _BP, None]

    def test_system_marked_whole_when_segments_do_not_match(self) -> None:
        messages = [{"role": "system", "content": "other"}, {"role": "user", "content": "hi"}]
        marked = add_prompt_cache_breakpoints(messages, (SystemPromptSegment("persona"),))
        assert marked[0]["content"] == [
            {"type": "text", "text": "other", "prompt_cache_breakpoint": _BP}
        ]

    def test_boundaries_before_each_assistant_and_final_message(self) -> None:
        call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "do it"},
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "c1", "content": "r1"},
            {"role": "tool", "tool_call_id": "c2", "content": "r2"},
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "c3", "content": "r3"},
        ]
        marked = add_prompt_cache_breakpoints(messages)
        assert [_breakpointed(message) for message in marked] == [
            True,
            True,
            False,
            False,
            True,
            False,
            True,
        ]
        # Unmarked messages are passed through untouched.
        assert marked[3] is messages[3]

    def test_boundary_count_capped_to_latest(self) -> None:
        messages: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
        for index in range(60):
            messages.append({"role": "user", "content": f"u{index}"})
            messages.append({"role": "assistant", "content": f"a{index}"})
        messages.append({"role": "user", "content": "last"})
        marked = add_prompt_cache_breakpoints(messages)
        flags = [_breakpointed(message) for message in marked]
        assert flags[0] is True
        assert sum(flags) == 50
        assert flags[-1] is True
        assert flags[1] is False

    def test_only_flagged_openai_family_models_get_breakpoints(self) -> None:
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        flagged = build_openai_completion_kwargs(_request(messages, model_def=_breakpoint_model()))
        assert _breakpointed(flagged["messages"][0])
        plain = build_openai_completion_kwargs(_request(messages))
        assert plain["messages"] == messages
        other = build_openai_completion_kwargs(
            _request(messages, model_def=_breakpoint_model(provider="ollama"))
        )
        assert other["messages"] == messages

    @pytest.mark.asyncio
    async def test_rejected_breakpoints_are_stripped_and_not_resent(self) -> None:
        calls: list[dict[str, Any]] = []

        class _Rejecting:
            async def create(self, **kwargs: Any) -> Any:
                calls.append(kwargs)
                if any(_breakpointed(message) for message in kwargs["messages"]):
                    raise ValueError("Unrecognized parameter: prompt_cache_breakpoint")
                return _chat_response()

        client = SimpleNamespace(chat=SimpleNamespace(completions=_Rejecting()))
        gateway = AzureOpenAIGateway(complete_client_factory=lambda: client)
        request = _request(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
            model_def=_breakpoint_model(model_id="gpt-5.6-rejecting"),
        )

        assert (await gateway.complete(request)).content == "ok"
        assert [any(_breakpointed(m) for m in c["messages"]) for c in calls] == [True, False]

        await gateway.complete(request)
        assert len(calls) == 3
        assert not any(_breakpointed(m) for m in calls[-1]["messages"])


class TestCacheWriteUsage:
    def test_chat_usage_reads_cache_writes(self) -> None:
        usage = SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=10,
            prompt_tokens_details=SimpleNamespace(cached_tokens=600, cache_write_tokens=300),
        )
        parsed = extract_openai_completion_usage(SimpleNamespace(usage=usage))
        assert parsed is not None
        assert parsed.cached_tokens == 600
        assert parsed.cache_creation_tokens == 300

    def test_responses_usage_reads_cache_writes(self) -> None:
        usage = SimpleNamespace(
            input_tokens=1000,
            output_tokens=10,
            input_tokens_details=SimpleNamespace(cached_tokens=400, cache_write_tokens=500),
        )
        parsed = extract_responses_usage(SimpleNamespace(usage=usage))
        assert parsed is not None
        assert parsed.cached_tokens == 400
        assert parsed.cache_creation_tokens == 500

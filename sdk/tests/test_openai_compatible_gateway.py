"""Tests for chat-completions tool-name sanitization in OpenAICompatibleGateway.

Strict OpenAI-compatible backends (OpenRouter upstreams such as Decart)
enforce ``^[a-zA-Z0-9_-]+$`` on function names and reject Lisa's dotted
namespace names with ``400 backend_invalid_argument``. The gateway must
sanitize names on the wire — including historical ``tool_calls`` — and
restore the originals on extraction so the kernel dispatcher still matches.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from kernel.provider_contracts import CompletionRequest
from sdk.providers.openai_gateway import (
    OpenAICompatibleGateway,
    extract_openai_content,
    sanitize_chat_history_tool_names,
    sanitize_chat_tool_names,
)


def _model_def() -> Any:
    return SimpleNamespace(
        provider="openrouter",
        model_id="z-ai/glm-5.2:free",
        display_name="GLM 5.2",
        reasoning_model=False,
        thinking_model=False,
    )


def _request(
    messages: list[dict[str, Any]] | None = None,
    *,
    tools: list[dict[str, Any]] | None = None,
) -> CompletionRequest:
    return CompletionRequest(
        messages=messages or [{"role": "user", "content": "hi"}],
        model_def=_model_def(),
        max_tokens=1024,
        temperature=0.4,
        tools=tools,
    )


_TOOL = {
    "type": "function",
    "function": {
        "name": "lisa-storage.read_file",
        "description": "Read a file.",
        "parameters": {"type": "object", "properties": {}},
    },
}


# ── Unit: helpers ────────────────────────────────────────────────────


class TestSanitizeChatToolNames:
    def test_dotted_name_sanitized_with_reverse_map(self) -> None:
        sanitized, name_map = sanitize_chat_tool_names([_TOOL])
        assert sanitized[0]["function"]["name"] == "lisa-storage_read_file"
        assert name_map == {"lisa-storage_read_file": "lisa-storage.read_file"}

    def test_compliant_name_untouched_and_no_map_entry(self) -> None:
        tool = {"type": "function", "function": {"name": "yield", "parameters": {}}}
        sanitized, name_map = sanitize_chat_tool_names([tool])
        assert sanitized[0]["function"]["name"] == "yield"
        assert name_map == {}

    def test_none_tools_returns_empty(self) -> None:
        assert sanitize_chat_tool_names(None) == ([], {})

    def test_non_function_tools_pass_through(self) -> None:
        tool = {"type": "web_search"}
        sanitized, name_map = sanitize_chat_tool_names([tool])
        assert sanitized == [tool]
        assert name_map == {}


class TestSanitizeChatHistoryToolNames:
    def test_history_tool_calls_rewritten(self) -> None:
        name_map: dict[str, str] = {}
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lisa-storage.read_file",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "data"},
        ]
        sanitized = sanitize_chat_history_tool_names(messages, name_map)
        call = sanitized[0]["tool_calls"][0]
        assert call["function"]["name"] == "lisa-storage_read_file"
        # The reverse map is extended so responses restore correctly.
        assert name_map == {"lisa-storage_read_file": "lisa-storage.read_file"}
        # Non-assistant messages pass through untouched.
        assert sanitized[1] is messages[1]

    def test_clean_history_returns_original(self) -> None:
        messages = [{"role": "user", "content": "hi"}]
        assert sanitize_chat_history_tool_names(messages, {}) is messages


# ── Integration: gateway round-trip ──────────────────────────────────


class _StubChatCompletions:
    def __init__(self, response: Any, captured: dict[str, Any]) -> None:
        self._response = response
        self.captured = captured

    async def create(self, **kwargs: Any) -> Any:
        self.captured["kwargs"] = kwargs
        return self._response


class _StubClient:
    def __init__(self, response: Any, captured: dict[str, Any]) -> None:
        self.chat = SimpleNamespace(completions=_StubChatCompletions(response, captured))


def _tool_call_response(name: str) -> SimpleNamespace:
    function = SimpleNamespace(name=name, arguments='{"path": "x"}')
    call = SimpleNamespace(id="call_1", type="function", function=function)
    message = SimpleNamespace(content=None, tool_calls=[call])
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        id="chat-1",
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=7),
    )


@pytest.mark.asyncio
async def test_complete_sanitizes_wire_names_and_restores_result() -> None:
    captured: dict[str, Any] = {}
    client = _StubClient(_tool_call_response("lisa-storage_read_file"), captured)
    gateway = OpenAICompatibleGateway(client_factory=lambda: client)
    history = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "lisa-storage.list_files",
                        "arguments": "{}",
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": "[]"},
        {"role": "user", "content": "read it"},
    ]
    result = await gateway.complete(
        _request(messages=history, tools=[_TOOL])
    )
    wire_kwargs = captured["kwargs"]
    # Tools and history are sanitized on the wire.
    assert (
        wire_kwargs["tools"][0]["function"]["name"] == "lisa-storage_read_file"
    )
    assert (
        wire_kwargs["messages"][0]["tool_calls"][0]["function"]["name"]
        == "lisa-storage_list_files"
    )
    # The extracted tool call is restored to the dotted original.
    assert result.tool_calls[0]["name"] == "lisa-storage.read_file"


@pytest.mark.asyncio
async def test_complete_stream_sanitizes_wire_names() -> None:
    events = [
        SimpleNamespace(choices=[], usage=None),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None),
                    finish_reason="stop",
                )
            ],
            usage=None,
        ),
    ]

    class _Stream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            if not events:
                raise StopAsyncIteration
            return events.pop(0)

    captured: dict[str, Any] = {}

    class _StreamCompletions:
        async def create(self, **kwargs: Any) -> Any:
            captured["kwargs"] = kwargs
            return _Stream()

    client = SimpleNamespace(chat=SimpleNamespace(completions=_StreamCompletions()))
    gateway = OpenAICompatibleGateway(client_factory=lambda: client)
    async for _ in gateway.complete_stream(_request(tools=[_TOOL])):
        pass
    assert (
        captured["kwargs"]["tools"][0]["function"]["name"]
        == "lisa-storage_read_file"
    )


# ── Reasoning-field fallbacks (2026-09-02 empty-turn incident) ────────


class TestOllamaReasoningFieldFallback:
    """Ollama names the reasoning field ``reasoning``, not ``reasoning_content``.

    The openai SDK preserves the extra attribute (``model_config
    extra="allow"``), so both names must be read. Without the fallback,
    reasoning models served through Ollama that exhaust their completion
    budget on thinking return ``content=""`` and the turn yields
    "empty response" (the 2026-09-02 glm-5.3-flash:cloud incident).
    """

    @staticmethod
    def _ollama_message(reasoning_attr: str) -> SimpleNamespace:
        return SimpleNamespace(
            content="",
            **{reasoning_attr: "The user asked about zeek. I should read the config."},
        )

    def test_extract_prefers_content_over_reasoning(self) -> None:
        message = SimpleNamespace(content="Final answer.", reasoning="thinking")
        response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
        assert extract_openai_content(response) == "Final answer."

    def test_extract_falls_back_to_reasoning_content(self) -> None:
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=self._ollama_message("reasoning_content"))]
        )
        assert "read the config" in extract_openai_content(response)

    def test_extract_falls_back_to_ollama_reasoning(self) -> None:
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=self._ollama_message("reasoning"))]
        )
        assert "read the config" in extract_openai_content(response)

    def test_extract_returns_empty_when_both_absent(self) -> None:
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=""))]
        )
        assert extract_openai_content(response) == ""

    @pytest.mark.asyncio
    async def test_complete_surfaces_ollama_reasoning_as_content(self) -> None:
        # Simulate the exact Ollama shape: extra ``reasoning`` field on the
        # parsed message (as the openai SDK preserves it via extra="allow").
        message = SimpleNamespace(
            content="",
            reasoning="long thinking that consumed the budget",
        )
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            id="chat-1",
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=200, total_tokens=210
            ),
        )
        client = _StubClient(response, {})
        gateway = OpenAICompatibleGateway(client_factory=lambda: client)
        result = await gateway.complete(_request())
        assert result.content == "long thinking that consumed the budget"

    @pytest.mark.asyncio
    async def test_complete_stream_captures_ollama_reasoning_delta(self) -> None:
        events = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=None, reasoning="thinking..."),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=None, reasoning=None),
                        finish_reason="length",
                    )
                ],
                usage=None,
            ),
        ]

        class _Stream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                if not events:
                    raise StopAsyncIteration
                return events.pop(0)

        class _StreamCompletions:
            async def create(self, **kwargs: Any) -> Any:
                return _Stream()

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=_StreamCompletions())
        )
        gateway = OpenAICompatibleGateway(client_factory=lambda: client)
        chunks = [chunk async for chunk in gateway.complete_stream(_request())]
        assert chunks[0].reasoning_content == "thinking..."
        assert chunks[1].finish_reason == "length"


# ── Truncation signals on the non-streaming path ────────────────────


def _completion_response(finish_reason: str | None) -> SimpleNamespace:
    """Build a chat-completions response stub with a finish_reason.

    The real SDK response carries ``choices[0].finish_reason``; stubs read
    the same attribute path via ``getattr``.
    """
    message = SimpleNamespace(content="Partial answer, cut mid-")
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], id="chat-1", usage=None)


class TestCompleteFinishReasonAndTruncated:
    """``complete()`` must surface ``finish_reason``/``truncated``.

    The 2026-09-19 narration-loop incident: the OpenAI-compatible
    ``complete()`` (serving openai/ollama/openrouter) constructed
    ``CompletionResult`` without either field, so a ``finish_reason='length'``
    truncation that cut a response mid-tool-call was invisible to the
    kernel — no truncation guard could fire. Bedrock/Google/Anthropic
    gateways already populate both fields; this is the same contract.
    """

    @pytest.mark.asyncio
    async def test_length_truncation_surfaces_finish_reason_and_flag(self) -> None:
        captured: dict[str, Any] = {}
        client = _StubClient(_completion_response("length"), captured)
        gateway = OpenAICompatibleGateway(client_factory=lambda: client)
        result = await gateway.complete(_request())
        assert result.finish_reason == "length"
        assert result.truncated is True

    @pytest.mark.asyncio
    async def test_clean_stop_surfaces_false_not_none(self) -> None:
        captured: dict[str, Any] = {}
        client = _StubClient(_completion_response("stop"), captured)
        gateway = OpenAICompatibleGateway(client_factory=lambda: client)
        result = await gateway.complete(_request())
        assert result.finish_reason == "stop"
        assert result.truncated is False

    @pytest.mark.asyncio
    async def test_tool_calls_finish_surfaces_false(self) -> None:
        captured: dict[str, Any] = {}
        client = _StubClient(_completion_response("tool_calls"), captured)
        gateway = OpenAICompatibleGateway(client_factory=lambda: client)
        result = await gateway.complete(_request())
        assert result.finish_reason == "tool_calls"
        assert result.truncated is False

    @pytest.mark.asyncio
    async def test_absent_finish_reason_is_unknown_none(self) -> None:
        captured: dict[str, Any] = {}
        client = _StubClient(_completion_response(None), captured)
        gateway = OpenAICompatibleGateway(client_factory=lambda: client)
        result = await gateway.complete(_request())
        assert result.finish_reason is None
        assert result.truncated is None

    @pytest.mark.asyncio
    async def test_empty_choices_surfaces_nothing(self) -> None:
        captured: dict[str, Any] = {}
        client = _StubClient(SimpleNamespace(choices=[], id="chat-1"), captured)
        gateway = OpenAICompatibleGateway(client_factory=lambda: client)
        result = await gateway.complete(_request())
        assert result.finish_reason is None
        assert result.truncated is None

    @pytest.mark.asyncio
    async def test_provider_specific_reason_is_unknown_none(self) -> None:
        captured: dict[str, Any] = {}
        client = _StubClient(_completion_response("content_filter"), captured)
        gateway = OpenAICompatibleGateway(client_factory=lambda: client)
        result = await gateway.complete(_request())
        assert result.finish_reason == "content_filter"
        assert result.truncated is None

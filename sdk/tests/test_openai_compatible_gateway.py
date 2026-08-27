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

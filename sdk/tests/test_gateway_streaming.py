"""Gateway streaming tests — OpenAI and Anthropic complete_stream.

Moved from ``kernel/tests/test_streaming.py`` when the gateways relocated to
``sdk/providers/``. The contract and turn-runner streaming tests remain in
``kernel/tests/test_streaming.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from kernel.provider_contracts import (
    CompletionRequest,
    StreamedChunk,
    _aggregate_streamed_chunks,
)
from sdk.providers.anthropic_gateway import AnthropicGateway
from sdk.providers.openai_gateway import OpenAICompatibleGateway


def _model_def(provider: str = "openai", model_id: str = "test-model") -> Any:
    return SimpleNamespace(
        provider=provider,
        model_id=model_id,
        display_name="Test Model",
        reasoning_model=False,
        thinking_model=False,
    )


def _request(messages: list[dict[str, Any]] | None = None) -> CompletionRequest:
    return CompletionRequest(
        messages=messages or [{"role": "user", "content": "hi"}],
        model_def=_model_def(),
        max_tokens=1024,
        temperature=0.0,
    )


# ── OpenAI gateway streaming ────────────────────────────────────────────


class _StubStreamEvent:
    """Simulates one SSE event from OpenAI streaming."""

    def __init__(
        self, delta_content: str = "", finish_reason: str | None = None, usage_obj: Any = None
    ) -> None:
        self.choices = (
            [
                SimpleNamespace(
                    delta=SimpleNamespace(content=delta_content or None),
                    finish_reason=finish_reason,
                )
            ]
            if (delta_content or finish_reason)
            else []
        )
        self.usage = usage_obj


class _StubAsyncStream:
    """Async iterable that yields pre-defined events."""

    def __init__(self, events: list[_StubStreamEvent]) -> None:
        self._events = events
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._events):
            raise StopAsyncIteration
        event = self._events[self._index]
        self._index += 1
        return event


class _StubOpenAIChatCompletions:
    def __init__(self, events: list[_StubStreamEvent]) -> None:
        self._events = events
        self.captured_kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> _StubAsyncStream:
        self.captured_kwargs.update(kwargs)
        return _StubAsyncStream(self._events)


class _StubOpenAIClient:
    def __init__(self, events: list[_StubStreamEvent]) -> None:
        self.chat = SimpleNamespace(completions=_StubOpenAIChatCompletions(events))


@pytest.mark.asyncio
async def test_openai_gateway_stream_yields_chunks() -> None:
    events = [
        _StubStreamEvent(delta_content="Hello"),
        _StubStreamEvent(delta_content=" world"),
        _StubStreamEvent(finish_reason="stop"),
    ]
    client = _StubOpenAIClient(events)
    gateway = OpenAICompatibleGateway(client_factory=lambda: client)
    chunks = []
    async for chunk in gateway.complete_stream(_request()):
        chunks.append(chunk)
    assert len(chunks) == 3
    assert chunks[0].content == "Hello"
    assert chunks[1].content == " world"
    assert chunks[2].finish_reason == "stop"
    aggregated = _aggregate_streamed_chunks(chunks)
    assert aggregated.content == "Hello world"


@pytest.mark.asyncio
async def test_openai_gateway_stream_passes_stream_flag() -> None:
    events = [_StubStreamEvent(finish_reason="stop")]
    client = _StubOpenAIClient(events)
    gateway = OpenAICompatibleGateway(client_factory=lambda: client)
    async for _ in gateway.complete_stream(_request()):
        pass
    assert client.chat.completions.captured_kwargs.get("stream") is True


# ── Anthropic gateway streaming ─────────────────────────────────────────


class _StubAnthropicStreamEvent:
    def __init__(self, event_type: str, delta_text: str | None = None) -> None:
        self.type = event_type
        if delta_text is not None:
            self.delta = SimpleNamespace(text=delta_text)


class _StubAnthropicStreamContext:
    """Simulates ``client.messages.stream()`` async context manager."""

    def __init__(self, events: list[_StubAnthropicStreamEvent], final_message: Any) -> None:
        self._events = events
        self._final_message = final_message

    async def __aenter__(self) -> _StubAnthropicStreamContext:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)

    async def get_final_message(self) -> Any:
        return self._final_message


class _StubAnthropicMessages:
    def __init__(self, events: list[_StubAnthropicStreamEvent], final_message: Any) -> None:
        self._events = events
        self._final_message = final_message
        self.captured_kwargs: dict[str, Any] = {}

    def stream(self, **kwargs: Any) -> _StubAnthropicStreamContext:
        self.captured_kwargs.update(kwargs)
        return _StubAnthropicStreamContext(list(self._events), self._final_message)


class _StubAnthropicClient:
    def __init__(self, events: list[_StubAnthropicStreamEvent], final_message: Any) -> None:
        self.messages = _StubAnthropicMessages(events, final_message)


@pytest.mark.asyncio
async def test_anthropic_gateway_stream_yields_text_deltas() -> None:
    events = [
        _StubAnthropicStreamEvent("content_block_delta", delta_text="Hello"),
        _StubAnthropicStreamEvent("content_block_delta", delta_text=" world"),
        _StubAnthropicStreamEvent("message_stop"),
    ]
    final_message = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=20,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
    )
    client = _StubAnthropicClient(events, final_message)
    gateway = AnthropicGateway(client_factory=lambda: client)
    chunks = []
    async for chunk in gateway.complete_stream(_request()):
        chunks.append(chunk)
    # 2 text deltas + 1 message_stop + 1 usage chunk
    assert len(chunks) == 4
    assert chunks[0].content == "Hello"
    assert chunks[1].content == " world"
    assert chunks[2].finish_reason == "stop"
    assert chunks[3].usage is not None
    assert chunks[3].usage.prompt_tokens == 10
    assert chunks[3].usage.completion_tokens == 20


@pytest.mark.asyncio
async def test_anthropic_gateway_stream_system_extracted() -> None:
    events = [_StubAnthropicStreamEvent("message_stop")]
    final_message = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=5,
            output_tokens=5,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
    )
    client = _StubAnthropicClient(events, final_message)
    gateway = AnthropicGateway(client_factory=lambda: client)
    async for _ in gateway.complete_stream(
        _request(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"},
            ]
        )
    ):
        pass
    assert client.messages.captured_kwargs.get("system") == "You are helpful."
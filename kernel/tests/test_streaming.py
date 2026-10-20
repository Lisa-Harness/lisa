"""Tests for streaming support (Part 2).

Covers ``StreamedChunk``/``StreamedCompletion`` types,
``_aggregate_streamed_chunks`` helper, the ``StreamedCompletionAdapter``
bridge, and the turn runner's streaming path (``TurnHooks.complete_stream``).

Gateway-specific streaming tests (``OpenAICompatibleGateway.complete_stream``,
``AnthropicGateway.complete_stream``) have moved to
``sdk/tests/test_gateway_streaming.py``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from kernel.provider_contracts import (
    CompletionRequest,
    CompletionUsage,
    StreamedChunk,
    StreamedCompletion,
    _aggregate_streamed_chunks,
)
from kernel.turn_hooks import (
    ModelHooks,
)
from kernel.turn_runner import StreamedCompletionAdapter, TurnHooks, run_turn

_CORE_ROOT: Path = (Path(__file__).resolve().parents[1] / ".." / "sdk" / "packs" / "core").resolve()

# The shared conftest fixture (_mcp_catalog_setup) handles all catalog wiring.


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


# ── Type tests ──────────────────────────────────────────────────────────


def test_streamed_chunk_defaults() -> None:
    chunk = StreamedChunk()
    assert chunk.content == ""
    assert chunk.finish_reason is None
    assert chunk.usage is None


def test_streamed_completion_defaults() -> None:
    sc = StreamedCompletion()
    assert sc.content == ""
    assert sc.tool_calls == []
    assert sc.usage is None
    assert sc.finish_reason == "stop"
    assert sc.chunks == []


# ── Aggregation tests ────────────────────────────────────────────────────


def test_aggregate_basic_text() -> None:
    chunks = [
        StreamedChunk(content="Hello"),
        StreamedChunk(content=" "),
        StreamedChunk(content="world", finish_reason="stop"),
    ]
    result = _aggregate_streamed_chunks(chunks)
    assert result.content == "Hello world"
    assert result.finish_reason == "stop"
    assert result.tool_calls == []
    assert len(result.chunks) == 3


def test_aggregate_empty_chunks() -> None:
    result = _aggregate_streamed_chunks([])
    assert result.content == ""
    assert result.finish_reason == "stop"


def test_aggregate_usage_from_last_chunk() -> None:
    usage = CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    chunks = [
        StreamedChunk(content="a"),
        StreamedChunk(content="b", usage=usage),
    ]
    result = _aggregate_streamed_chunks(chunks)
    assert result.usage is not None
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 20


def test_aggregate_finish_reason_last_wins() -> None:
    chunks = [
        StreamedChunk(content="a", finish_reason=None),
        StreamedChunk(content="b", finish_reason="length"),
        StreamedChunk(content="c", finish_reason="stop"),
    ]
    result = _aggregate_streamed_chunks(chunks)
    assert result.finish_reason == "stop"


# ── StreamedCompletionAdapter ───────────────────────────────────────────


def test_streamed_completion_adapter_fields() -> None:
    usage = CompletionUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15)
    sc = StreamedCompletion(content="hello", usage=usage, finish_reason="stop")
    adapter = StreamedCompletionAdapter(sc)
    assert adapter.content == "hello"
    assert adapter.tool_calls == []
    assert adapter.usage is not None
    assert adapter.usage.prompt_tokens == 5


def test_streamed_completion_adapter_keeps_reasoning_separate() -> None:
    """Reasoning-only stream surfaces EMPTY content + reasoning_content.

    The adapter must not fold reasoning into ``content`` — the kernel's
    ``reasoning_tail_excerpt`` display guard only bounds leaked thinking when
    ``content`` is empty (2026-10-04 non-streaming 119KB monologue incident).
    """
    sc = StreamedCompletion(
        content="",
        reasoning_content="thinking about it",
        finish_reason="length",
    )
    adapter = StreamedCompletionAdapter(sc)
    assert adapter.content == ""
    assert adapter.reasoning_content == "thinking about it"


def test_streamed_completion_adapter_surfaces_finish_reason() -> None:
    """The adapter exposes the aggregated ``finish_reason`` for diagnostics.

    Real gateways set ``finish_reason`` on ``CompletionResult`` so the
    reasoning-tail caveat can add its truncation clause and the turn-loop
    diagnostics can distinguish a budget-starved stop from a clean one
    (2026-10-13 turn 789: the streamed burnout caveat read
    ``finish_reason=unknown`` because the adapter hid the aggregate). The
    completion-loop reads it via ``getattr``, so the adapter needs a real
    attribute, not just the ``__repr__``.
    """
    sc = StreamedCompletion(
        content="",
        reasoning_content="pondering",
        finish_reason="length",
    )
    adapter = StreamedCompletionAdapter(sc)
    assert adapter.finish_reason == "length"


def test_streamed_completion_adapter_truncated_surfaces_aggregate() -> None:
    """The adapter exposes the aggregate's ``finish_reason`` default.

    A stream that never carries an explicit finish aggregates to ``stop``;
    the adapter must surface that default (burnout predicates read
    ``finish_reason`` via ``getattr`` and must not crash on absence).
    """
    sc = StreamedCompletion(content="partial")
    adapter = StreamedCompletionAdapter(sc)
    assert adapter.finish_reason == "stop"


# ── Turn runner streaming path ───────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _StubCompletion:
    content: str
    tool_calls: list[dict[str, Any]] | None = None


def _make_streaming_complete_stream(chunks: list[StreamedChunk]):
    """Return an async generator that yields pre-defined chunks."""

    async def stream_fn(**kwargs: Any):
        for chunk in chunks:
            yield chunk

    return stream_fn


def _make_stub_complete(response: str):
    async def complete_fn(**kwargs: Any):
        return _StubCompletion(content=response)

    return complete_fn


def _build_minimal_hooks(complete_fn=None, stream_fn=None) -> TurnHooks:
    return TurnHooks(
        models=ModelHooks(
            complete=complete_fn,
            complete_stream=stream_fn,
        ),
    )


@pytest.mark.asyncio
async def test_turn_runner_uses_stream_when_set() -> None:
    chunks = [
        StreamedChunk(content="Hello"),
        StreamedChunk(content=" world", finish_reason="stop"),
    ]
    stream_fn = _make_streaming_complete_stream(chunks)
    complete_fn = _make_stub_complete("should not be called")
    hooks = _build_minimal_hooks(complete_fn, stream_fn)
    result = await run_turn(mode_id="default", query="hi", hooks=hooks)
    assert "Hello world" in result.response_text


@pytest.mark.asyncio
async def test_turn_runner_falls_back_to_complete_when_stream_not_set() -> None:
    complete_fn = _make_stub_complete("non-streaming response")
    hooks = _build_minimal_hooks(complete_fn, stream_fn=None)
    result = await run_turn(mode_id="default", query="hi", hooks=hooks)
    assert "non-streaming response" in result.response_text


@pytest.mark.asyncio
async def test_turn_runner_stream_tool_calls_falls_back_to_complete() -> None:
    """When stream finish_reason='tool_calls', turn runner falls back to complete()."""

    stream_chunks = [StreamedChunk(content="", finish_reason="tool_calls")]
    stream_fn = _make_streaming_complete_stream(stream_chunks)

    async def complete_fn(**kwargs: Any):
        return _StubCompletion(content="final answer")

    hooks = _build_minimal_hooks(complete_fn, stream_fn)
    result = await run_turn(mode_id="default", query="hi", hooks=hooks)
    assert "final answer" in result.response_text


# ── Streaming tool-call deltas (2026-10-12 one-pass tool turns) ──


def test_aggregate_stitches_tool_call_deltas_by_index() -> None:
    """Deltas for the same call index merge; names and arguments concatenate.

    Ollama v1 streams a tool call as fragments:
    ``delta.tool_calls=[{index:0, id:'call_1', function:{name:'write'}}]``
    then ``[{index:0, function:{arguments:'{"pa'}}]`` … The stitched result
    must be the same normalized shape ``extract_openai_tool_calls`` returns
    for a non-streaming response.
    """
    chunks = [
        StreamedChunk(
            tool_call_deltas=[
                {"index": 0, "id": "call_1", "function": {"name": "write_file", "arguments": ""}}
            ]
        ),
        StreamedChunk(
            tool_call_deltas=[{"index": 0, "function": {"arguments": '{"path": "README.md", '}}]
        ),
        StreamedChunk(
            tool_call_deltas=[{"index": 0, "function": {"arguments": '"content": "hi"}'}}],
            finish_reason="tool_calls",
        ),
        # A second call starting mid-stream gets its own index.
        StreamedChunk(
            tool_call_deltas=[
                {"index": 1, "id": "call_2", "function": {"name": "read_file", "arguments": "{}"}}
            ]
        ),
    ]
    result = _aggregate_streamed_chunks(chunks)
    assert result.finish_reason == "tool_calls"
    assert len(result.tool_calls) == 2
    first = result.tool_calls[0]
    assert first["id"] == "call_1"
    assert first["type"] == "function"
    assert first["function"]["name"] == "write_file"
    assert json.loads(first["function"]["arguments"]) == {"path": "README.md", "content": "hi"}
    second = result.tool_calls[1]
    assert second["id"] == "call_2"
    assert second["function"]["name"] == "read_file"


def test_aggregate_without_deltas_keeps_empty_tool_calls() -> None:
    """No deltas = no tool calls: today's text-only behavior is unchanged."""
    chunks = [StreamedChunk(content="hi"), StreamedChunk(finish_reason="tool_calls")]
    result = _aggregate_streamed_chunks(chunks)
    assert result.tool_calls == []
    assert result.finish_reason == "tool_calls"


def test_aggregate_separates_parallel_calls_sharing_an_index() -> None:
    """Parallel calls that reuse one index stay separate calls.

    Regression for the 2026-10-19 MiniMax M3.1 Flash turn failure: Ollama
    streams every parallel tool call with ``index: 0`` and tells them
    apart only by ``id``. Stitching by ``index`` alone concatenated them
    into one garbage call named
    ``lisa-storage.list_fileslisa-storage.read_file`` with three JSON
    objects glued end to end. The turn loop then refused the unparseable
    payload, and the next request replayed the broken call and died on
    ``400 invalid tool call arguments``.
    """
    chunks = [
        StreamedChunk(
            tool_call_deltas=[
                {
                    "index": 0,
                    "id": "call_1",
                    "function": {"name": "lisa-storage.list_files", "arguments": '{"path":"."}'},
                }
            ]
        ),
        StreamedChunk(
            tool_call_deltas=[
                {
                    "index": 0,
                    "id": "call_2",
                    "function": {"name": "lisa-storage.read_file", "arguments": '{"path":"a.md"}'},
                }
            ]
        ),
        StreamedChunk(
            tool_call_deltas=[
                {
                    "index": 0,
                    "id": "call_3",
                    "function": {"name": "lisa-storage.read_file", "arguments": '{"path":"b.md"}'},
                }
            ],
            finish_reason="tool_calls",
        ),
    ]
    result = _aggregate_streamed_chunks(chunks)
    assert len(result.tool_calls) == 3, "each parallel call must stay its own call"
    names = [call["function"]["name"] for call in result.tool_calls]
    assert names == [
        "lisa-storage.list_files",
        "lisa-storage.read_file",
        "lisa-storage.read_file",
    ]
    assert [call["id"] for call in result.tool_calls] == ["call_1", "call_2", "call_3"]
    for call, expected in zip(
        result.tool_calls,
        [{"path": "."}, {"path": "a.md"}, {"path": "b.md"}],
        strict=True,
    ):
        assert json.loads(call["function"]["arguments"]) == expected


def test_aggregate_keeps_fragments_of_one_call_together() -> None:
    """Continuation fragments of a single call still merge by index.

    The shared-index split must not break the normal OpenAI shape, where
    one call's arguments arrive across many fragments that all carry the
    same id (or no id at all).
    """
    chunks = [
        StreamedChunk(
            tool_call_deltas=[
                {
                    "index": 0,
                    "id": "call_1",
                    "function": {"name": "lisa-storage.write_file", "arguments": ""},
                }
            ]
        ),
        StreamedChunk(
            tool_call_deltas=[
                {"index": 0, "function": {"arguments": '{"path":"a.md",'}},
            ]
        ),
        StreamedChunk(
            tool_call_deltas=[
                {"index": 0, "function": {"arguments": '"content":"hi"}'}},
            ],
            finish_reason="tool_calls",
        ),
    ]
    result = _aggregate_streamed_chunks(chunks)
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call["id"] == "call_1"
    assert call["function"]["name"] == "lisa-storage.write_file"
    assert json.loads(call["function"]["arguments"]) == {"path": "a.md", "content": "hi"}


def test_aggregate_keeps_indexed_calls_with_distinct_indices_apart() -> None:
    """The ordinary OpenAI shape (one index per call) is unchanged."""

    chunks = [
        StreamedChunk(
            tool_call_deltas=[
                {
                    "index": 0,
                    "id": "call_1",
                    "function": {"name": "a_tool", "arguments": "{}"},
                },
                {
                    "index": 1,
                    "id": "call_2",
                    "function": {"name": "b_tool", "arguments": "{}"},
                },
            ],
            finish_reason="tool_calls",
        ),
    ]
    result = _aggregate_streamed_chunks(chunks)
    assert [call["id"] for call in result.tool_calls] == ["call_1", "call_2"]
    assert [call["function"]["name"] for call in result.tool_calls] == ["a_tool", "b_tool"]


@pytest.mark.asyncio
async def test_turn_runner_stream_tool_deltas_skip_the_fallback_complete() -> None:
    """A tool_calls stream with parsed deltas dispatches without regenerating.

    Regression for the 2026-10-12 performance review: the v1 gateway
    streamed, hit ``finish_reason='tool_calls'``, and the turn loop
    regenerated the entire content non-streaming — a measured 1.21x tax
    on every tool-call iteration (the ollama stream pass is discarded).
    When the stream carries parseable tool-call deltas, the dispatch must
    run off the streamed payload and ``complete()`` must not be called.
    """
    stream_chunks = [
        StreamedChunk(content="Editing."),
        StreamedChunk(
            tool_call_deltas=[
                {
                    "index": 0,
                    "id": "call_1",
                    "function": {
                        "name": "memory.add",
                        "arguments": json.dumps({"content": "edited"}),
                    },
                }
            ],
            finish_reason="tool_calls",
        ),
    ]
    stream_fn = _make_streaming_complete_stream(stream_chunks)
    complete_calls: list[int] = []

    async def complete_fn(**kwargs: Any):
        complete_calls.append(kwargs.get("max_tokens"))
        return _StubCompletion(content="should not be reached")

    hooks = _build_minimal_hooks(complete_fn, stream_fn)
    with patch(
        "kernel.turn_runner._compute_capability_routing_hint",
        return_value="",
    ):
        result = await run_turn(mode_id="default", query="make the edit", hooks=hooks)
    assert complete_calls == [], "complete() must not run when deltas were parsed"
    assert result.response_text != "should not be reached"


@pytest.mark.asyncio
async def test_turn_runner_tool_calls_stream_without_deltas_still_falls_back() -> None:
    """No parseable deltas = unchanged fallback (older backends)."""
    stream_calls: list[int] = []
    stream_fn_source = [StreamedChunk(content="", finish_reason="tool_calls")]

    async def stream_fn(**kwargs: Any):
        stream_calls.append(kwargs.get("max_tokens"))
        for chunk in stream_fn_source:
            yield chunk

    async def complete_fn(**kwargs: Any):
        return _StubCompletion(content="final answer")

    hooks = _build_minimal_hooks(complete_fn, stream_fn)
    result = await run_turn(mode_id="default", query="hi", hooks=hooks)
    assert "final answer" in result.response_text


# ── Truncated-stream rescue (the 2026-09-02 write-announcement loop) ───


@pytest.mark.asyncio
async def test_turn_runner_rescues_truncated_stream_with_tools_offered() -> None:
    """A stream that stops at ``length`` while tools were offered is
    retried once with a doubled budget instead of being accepted as a
    narration-only final completion.

    Regression for the 2026-09-02 write-announcement loop: v1 streaming
    does not parse tool-call deltas, so when glm-5.2:cloud spent its
    4096-token budget narrating and starting a multi-KB ``write_file``
    payload, the stream ended ``finish_reason='length'`` with tool calls
    still pending. The kernel treated the narration as a natural
    completion, the announced write never dispatched, and the user had to
    re-ask "continue" every turn.
    """
    calls: list[int] = []

    def _stream_fn(chunks_by_call: list[list[StreamedChunk]]):
        call_idx = {"n": 0}

        async def stream_fn(**kwargs: Any):
            calls.append(kwargs.get("max_tokens"))
            chunks = chunks_by_call[call_idx["n"]]
            call_idx["n"] += 1
            for chunk in chunks:
                yield chunk

        return stream_fn

    chunks_by_call = [
        # First call: narration streams fine, then truncation mid payload.
        [StreamedChunk(content="Writing both files now."), StreamedChunk(finish_reason="length")],
        # Retry: the same narration plus a tool_calls finish — the payload
        # fits in the doubled budget.
        [
            StreamedChunk(content="Writing both files now."),
            StreamedChunk(finish_reason="tool_calls"),
        ],
    ]
    stream_fn = _stream_fn(chunks_by_call)

    async def complete_fn(**kwargs: Any):
        return _StubCompletion(content="wrote it")

    hooks = _build_minimal_hooks(complete_fn, stream_fn)
    # The query text must NOT trigger the routing-hint forced transition
    # (a "write"-intent gap arms a forced handoff that synthesizes a
    # transition dispatch and consumes a third model call the stubs do
    # not provide). This test pins the rescue contract, not routing —
    # suppress the hint through the documented kernel seam.
    with patch(
        "kernel.turn_runner._compute_capability_routing_hint",
        return_value="",
    ):
        result = await run_turn(mode_id="default", query="write the scripts", hooks=hooks)

    # The retry fired with a doubled budget…
    assert calls == [4096, 8192], f"expected one rescue retry at 2x, got {calls}"
    # …the retried stream ended tool_calls, so the fallback complete() ran
    # and produced the tool-call turn.
    assert "wrote it" in result.response_text


@pytest.mark.asyncio
async def test_turn_runner_truncated_stream_without_tools_is_not_retried() -> None:
    """``finish_reason='length'`` with NO tools offered is a natural
    completion, not a lost tool call — no rescue retry."""

    calls: list[int] = []

    async def stream_fn(**kwargs: Any):
        calls.append(kwargs.get("max_tokens"))
        yield StreamedChunk(content="A plain answer that ran long.")
        yield StreamedChunk(finish_reason="length")

    async def complete_fn(**kwargs: Any):
        raise AssertionError("complete() must not be called on the natural path")

    async def empty_tool_surface(_mode_config: Any, **_kwargs: Any):
        return (
            [],
            {
                "input_count": 0,
                "output_count": 0,
                "dropped_duplicate": 0,
                "dropped_disallowed": 0,
                "dropped_invalid": 0,
            },
        )

    hooks = _build_minimal_hooks(complete_fn, stream_fn)
    # The default catalog loads ambient memory/yield tools; strip the tool
    # surface so this test exercises the genuinely tools-offered=False path.
    with patch("kernel.turn_runner.build_tool_surface", empty_tool_surface):
        result = await run_turn(mode_id="default", query="hello there", hooks=hooks)

    assert calls == [4096], "no retry should fire when tools were not offered"
    assert "plain answer" in result.response_text


@pytest.mark.asyncio
async def test_turn_runner_truncated_stream_retry_respects_context_cap() -> None:
    """The rescue retry budget is bounded by the model's context window."""

    calls: list[int] = []

    async def stream_fn(**kwargs: Any):
        calls.append(kwargs.get("max_tokens"))
        yield StreamedChunk(content="partial")
        yield StreamedChunk(finish_reason="length")

    async def complete_fn(**kwargs: Any):
        return _StubCompletion(content="wrote it")

    hooks = TurnHooks(
        models=ModelHooks(
            complete=complete_fn,
            complete_stream=stream_fn,
            resolve_model=lambda _mid: SimpleNamespace(
                provider="test",
                model_id="small-model",
                display_name="Small Model",
                reasoning_model=False,
                thinking_model=False,
                context_window=9000,
            ),
        ),
    )
    # The retry uses only the space left after input and the safety margin.
    # Suppress the routing hint (same seam as the rescue test above) so
    # the forced transition cannot consume a third stubbed stream call.
    with patch(
        "kernel.turn_runner._compute_capability_routing_hint",
        return_value="",
    ):
        result = await run_turn(
            mode_id="default",
            query="write the scripts",
            hooks=hooks,
        )

    budget = result.model_requests[0]["context_budget"]
    assert calls == [4096, 9000 - budget["input_tokens"] - budget["margin_tokens"]]
    assert "partial" in result.response_text


# ── Truncated-completion rescue (non-streaming path) ──────────────────
# The 2026-09-19 narration-loop incident: `lisa once` defaults to
# streaming="none", so its turn-loop completions take the
# `_gateway.complete()` path. A `finish_reason='length'` truncation cut
# the response mid-tool-call and the narration half was accepted as the
# final answer. These tests pin the same rescue contract as the streaming
# tests above, but on the non-streaming path.


@dataclass(frozen=True, slots=True)
class _RescueStubCompletion:
    """Stub completion carrying the provider truncation signals.

    Mirrors ``CompletionResult``: ``finish_reason`` drives the rescue,
    ``tool_calls`` drives the dispatch path.
    """

    content: str
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str | None = None


@pytest.mark.asyncio
async def test_turn_runner_rescues_truncated_completion_with_tools_offered() -> None:
    """A non-streaming completion that stops at ``length`` while tools
    were offered is retried once with a doubled budget instead of being
    accepted as a truncated narration-only final completion."""

    calls: list[int] = []
    completions = [
        # First call: narration, truncated mid-tool-call payload.
        _RescueStubCompletion(
            content="Let me write the config and the report script now.",
            finish_reason="length",
        ),
        # Retry (doubled budget): the payload fits — the model lands its
        # answer cleanly (same shape as the streaming twin: the retried
        # generation produces the full completion).
        _RescueStubCompletion(
            content="wrote it",
            finish_reason="stop",
        ),
    ]
    call_idx = {"n": 0}

    async def complete_fn(**kwargs: Any):
        calls.append(kwargs.get("max_tokens"))
        completion = completions[call_idx["n"]]
        call_idx["n"] += 1
        return completion

    hooks = _build_minimal_hooks(complete_fn, stream_fn=None)
    with patch(
        "kernel.turn_runner._compute_capability_routing_hint",
        return_value="",
    ):
        result = await run_turn(mode_id="default", query="write the scripts", hooks=hooks)

    # The retry fired with a doubled budget…
    assert calls == [4096, 8192], f"expected one rescue retry at 2x, got {calls}"
    # …and the retried completion's answer is surfaced, not the truncated
    # narration half.
    assert "wrote it" in result.response_text


@pytest.mark.asyncio
async def test_turn_runner_truncated_completion_stripped_tools_retries() -> None:
    """A completion truncated on a tools-stripped iteration still retries.

    Regression for the 2026-10-06 "unfinished turns" incident: turns ended
    blocked with partial reasoning-prose when the final no-tools completion
    (pending_yield = iteration cap, lifecycle _FinishAction, router terminal
    action, approval continuation) was truncated at the cap. The rescue's
    tools-offered predicate read the iteration-local effective_tools, which
    the loop strips on those iterations, so truncated stripped-tools finals
    never got the 2x retry and the turn aborted with a reasoning tail.

    The model had tools earlier in the turn; stripping them on one
    iteration is harness-internal (the loop's choice to force a summary).
    Retry on the doubled budget instead of surfacing the truncation abort.
    """

    calls: list[int] = []
    completions = [
        # First call (tools offered): the model makes no tool calls, runs
        # out of its budget narrating a summary, and truncates.
        _RescueStubCompletion(
            content="The work so far concludes with the change applied, but writing the summary…",
            finish_reason="length",
        ),
        # Retry (doubled budget): completes with a full final answer.
        _RescueStubCompletion(
            content="Conclusion: landed the change, ran the checks, and wrote the summary.",
            finish_reason="stop",
        ),
    ]
    call_idx = {"n": 0}

    async def complete_fn(**kwargs: Any):
        calls.append(kwargs.get("max_tokens"))
        completion = completions[call_idx["n"]]
        call_idx["n"] += 1
        return completion

    hooks = _build_minimal_hooks(complete_fn, stream_fn=None)
    with patch(
        "kernel.turn_runner._compute_capability_routing_hint",
        return_value="",
    ):
        result = await run_turn(mode_id="default", query="finish up the changes.", hooks=hooks)

    # The retry fired with a doubled budget — the tools-stripped final got
    # the same 2x rescue as a tools-offered call.
    assert calls == [4096, 8192], f"expected one rescue retry at 2x, got {calls}"
    # The turn completes with the retried completion's answer, not the
    # truncated "nothing was dispatched" abort message.
    assert "nothing was dispatched" not in result.response_text
    assert "Conclusion" in result.response_text


@pytest.mark.asyncio
async def test_turn_runner_truncated_completion_no_tools_no_retry() -> None:
    """A stripped-tools truncation is retried only when the turn began
    with a tool surface.

    The sibling test ``test_turn_runner_truncated_completion_without_tools_
    is_not_retried`` pins the no-tools-at-all contract: a plain chat turn
    that finishes at the cap is a natural completion, no retry. This case
    fills the other gap: a turn that *started* with tools (a
    ``memory.read`` catalog ambient tool) and then strips them for the
    final iteration because the cap fires must still retry.
    """

    calls: list[int] = []
    completions = [
        _RescueStubCompletion(content="partial", finish_reason="length"),
        _RescueStubCompletion(content="done", finish_reason="stop"),
    ]
    call_idx = {"n": 0}

    async def complete_fn(**kwargs: Any):
        calls.append(kwargs.get("max_tokens"))
        completion = completions[call_idx["n"]]
        call_idx["n"] += 1
        return completion

    hooks = _build_minimal_hooks(complete_fn, stream_fn=None)
    with patch(
        "kernel.turn_runner._compute_capability_routing_hint",
        return_value="",
    ):
        result = await run_turn(mode_id="default", query="summarize the findings", hooks=hooks)

    assert calls == [4096, 8192], f"expected one rescue retry at 2x, got {calls}"
    assert result.response_text

    """``finish_reason='length'`` with NO tools offered is a natural
    completion, not a lost tool call — no rescue retry on the
    non-streaming path either."""

    calls: list[int] = []

    async def complete_fn(**kwargs: Any):
        calls.append(kwargs.get("max_tokens"))
        return _RescueStubCompletion(
            content="A plain answer that ran long.",
            finish_reason="length",
        )

    async def empty_tool_surface(_mode_config: Any, **_kwargs: Any):
        return (
            [],
            {
                "input_count": 0,
                "output_count": 0,
                "dropped_duplicate": 0,
                "dropped_disallowed": 0,
                "dropped_invalid": 0,
            },
        )

    hooks = _build_minimal_hooks(complete_fn, stream_fn=None)
    # Strip the tool surface so this exercises the genuinely
    # tools-offered=False path (the ambient catalog loads memory/yield
    # tools otherwise — same note as the streaming twin above).
    with patch("kernel.turn_runner.build_tool_surface", empty_tool_surface):
        result = await run_turn(mode_id="default", query="hello there", hooks=hooks)

    assert calls == [4096], "no retry should fire when tools were not offered"
    assert "plain answer" in result.response_text


@pytest.mark.asyncio
async def test_turn_runner_truncated_completion_retry_respects_context_cap() -> None:
    """The non-streaming rescue retry budget is bounded by the model's
    context window."""

    calls: list[int] = []
    completions = [
        _RescueStubCompletion(content="partial", finish_reason="length"),
        _RescueStubCompletion(content="complete answer", finish_reason="stop"),
    ]
    call_idx = {"n": 0}

    async def complete_fn(**kwargs: Any):
        calls.append(kwargs.get("max_tokens"))
        completion = completions[call_idx["n"]]
        call_idx["n"] += 1
        return completion

    hooks = TurnHooks(
        models=ModelHooks(
            complete=complete_fn,
            resolve_model=lambda _mid: SimpleNamespace(
                provider="test",
                model_id="small-model",
                display_name="Small Model",
                reasoning_model=False,
                thinking_model=False,
                context_window=9000,
            ),
        ),
    )
    # The retry uses only the space left after input and the safety margin.
    with patch(
        "kernel.turn_runner._compute_capability_routing_hint",
        return_value="",
    ):
        result = await run_turn(
            mode_id="default",
            query="write the scripts",
            hooks=hooks,
        )

    budget = result.model_requests[0]["context_budget"]
    assert calls == [4096, 9000 - budget["input_tokens"] - budget["margin_tokens"]]
    assert "complete answer" in result.response_text


@pytest.mark.asyncio
async def test_turn_runner_rescues_truncated_fallback_after_tool_calls_stream() -> None:
    """A truncated non-streaming fallback after a tool_calls stream is retried.

    Regression for the 2026-10-08 "ran out without doing anything" turns:
    v1 streaming ends ``finish_reason='tool_calls'`` and falls back to the
    non-streaming ``complete()`` to rebuild the structured tool-call payload.
    That regeneration is a fresh generation — a reasoning model can burn its
    whole budget re-deliberating and stop at ``finish_reason='length'`` with
    no tool calls (glm-5.3:cloud in coding_execute: 16384-token loops of
    "Let me output. Here. Go. OK. Writing."). Both sibling paths (the
    truncated-stream rescue and the non-streaming truncation rescue) retry
    at 2x, but this fallback regenerated with no rescue — the turn ended
    with the "reached its token cap before emitting a tool call" abort and
    the degenerate reasoning tail surfaced as the response, three turns in
    a row.
    """

    stream_calls: list[int] = []
    complete_calls: list[int] = []
    stream_idx = {"n": 0}

    async def stream_fn(**kwargs: Any):
        stream_calls.append(kwargs.get("max_tokens"))
        if stream_idx["n"] == 0:
            stream_idx["n"] += 1
            yield StreamedChunk(content="Let me make the edit now.")
            yield StreamedChunk(finish_reason="tool_calls")
        else:
            yield StreamedChunk(content="", finish_reason="stop")

    completions = [
        # First fallback generation: re-deliberates past the budget and
        # truncates before emitting the tool-call payload.
        _RescueStubCompletion(
            content="",
            finish_reason="length",
        ),
        # Retry (doubled budget): the payload fits — the edit lands as a
        # structured tool call.
        _RescueStubCompletion(
            content="",
            finish_reason="stop",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "memory.add",
                        "arguments": json.dumps({"content": "edited"}),
                    },
                }
            ],
        ),
        # Iteration 2 after the dispatch: the fallback path returns a
        # clean final answer (no further truncation).
        _RescueStubCompletion(
            content="Edit applied and recorded.",
            finish_reason="stop",
        ),
    ]
    call_idx = {"n": 0}

    async def complete_fn(**kwargs: Any):
        complete_calls.append(kwargs.get("max_tokens"))
        completion = completions[call_idx["n"]]
        call_idx["n"] += 1
        return completion

    hooks = _build_minimal_hooks(complete_fn, stream_fn)
    with patch(
        "kernel.turn_runner._compute_capability_routing_hint",
        return_value="",
    ):
        result = await run_turn(mode_id="default", query="make the edit", hooks=hooks)

    # The fallback completion was retried at a doubled budget…
    assert complete_calls == [4096, 8192], (
        f"expected one fallback rescue retry at 2x, got {complete_calls}"
    )
    # …and the turn did NOT end on the token-cap abort wrapper.
    assert "reached its token cap before emitting a tool call" not in result.response_text


# ── Reasoning-burnout rescue (finish_reason='stop', 2026-10-13) ────────


@dataclass(frozen=True, slots=True)
class _BurnoutStubCompletion:
    """Stub completion for the clean-stop burnout shape.

    ``glmfash``-style reasoning models bill chain-of-thought against
    ``max_tokens`` and can end with ``finish_reason='stop'`` even after
    the whole budget went to thinking — the stop is not clean, it is the
    reasoning loop giving up mid-sentence. Carries flat token counters so
    the stub gateway's usage normalization keeps them (the counters ride
    ``usage.completion_tokens`` on the normalized ``CompletionResult``).
    """

    content: str
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    reasoning_content: str = ""
    reasoning_tokens: int = 0
    completion_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0


@pytest.mark.asyncio
async def test_turn_runner_rescues_reasoning_burnout_completion() -> None:
    """A non-streaming completion that burns its budget on reasoning and
    ends ``stop`` with empty content and no tool calls is retried once at
    a doubled budget.

    Regression for the 2026-10-13 turn 789 stall (glm-5.3-flash:cloud,
    coding_execute): the model re-deliberated a ``coding_edit_files``
    payload across the entire per-call budget — 16k tokens, then the
    doubled 32k rescue after a sibling truncation — and both rounds
    ended with no visible content, no tool call, and a 121KB reasoning
    tail. The truncation rescues only fired on ``finish_reason='length'``;
    this clean-stop burnout shape passed straight through to the
    reasoning-tail caveat and the turn ended with nothing dispatched —
    the user had to say "continue and finish" to make progress.
    """
    calls: list[int] = []
    completions = [
        # First call: whole budget burned on thinking; clean stop;
        # no content, no tool call — invisible to the length rescues.
        # completion_tokens==4096 matches a real on-the-wire burn (usage
        # is capped at the request's max_tokens; the incident's 16k
        # figure was the doubled budget its rescue had by then).
        _BurnoutStubCompletion(
            content="",
            prompt_tokens=10000,
            total_tokens=14096,
            finish_reason="stop",
            reasoning_content="Let me format carefully — multiline literals: " * 2000,
            reasoning_tokens=4096,
            completion_tokens=4096,
        ),
        # Retry (doubled budget): the payload fits — the edit lands.
        _BurnoutStubCompletion(
            content="",
            finish_reason="stop",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "memory.add",
                        "arguments": json.dumps({"content": "edited"}),
                    },
                }
            ],
        ),
        # Iteration 2 after the dispatch: a clean final answer.
        _BurnoutStubCompletion(content="Edit applied and recorded.", finish_reason="stop"),
    ]
    call_idx = {"n": 0}

    async def complete_fn(**kwargs: Any):
        calls.append(kwargs.get("max_tokens"))
        completion = completions[call_idx["n"]]
        call_idx["n"] += 1
        return completion

    hooks = _build_minimal_hooks(complete_fn, stream_fn=None)
    with patch(
        "kernel.turn_runner._compute_capability_routing_hint",
        return_value="",
    ):
        result = await run_turn(mode_id="default", query="make the edit", hooks=hooks)

    assert calls[:2] == [4096, 8192], f"expected one burnout retry at 2x, got {calls}"
    # The tool-call dispatch from the retried completion runs iteration 2
    # (base budget again), whose final answer lands the turn.
    assert "Edit applied and recorded." in result.response_text


@pytest.mark.asyncio
async def test_turn_runner_reasoning_burnout_below_burn_share_not_retried() -> None:
    """A clean-stop empty completion that did NOT burn near the budget is
    NOT retried — below the share the reasoning-tail caveat ends the turn
    as a natural completion (only a completion with neither content nor
    reasoning reaches the yield path).
    """
    calls: list[int] = []

    async def complete_fn(**kwargs: Any):
        calls.append(kwargs.get("max_tokens"))
        return _BurnoutStubCompletion(
            content="",
            prompt_tokens=10000,
            total_tokens=11200,
            finish_reason="stop",
            reasoning_content="a brief stray thought",
            reasoning_tokens=400,
            completion_tokens=1200,
        )

    hooks = _build_minimal_hooks(complete_fn, stream_fn=None)
    with patch(
        "kernel.turn_runner._compute_capability_routing_hint",
        return_value="",
    ):
        result = await run_turn(mode_id="default", query="hello there", hooks=hooks)

    assert calls == [4096], "no retry for an empty completion far under budget"
    # Below the burn share the caveat content ends the turn as a natural
    # completion (the reasoning-tail caveat IS content), not a yield.
    assert result.yielded is None
    assert "no visible answer" in result.response_text


@pytest.mark.asyncio
async def test_turn_runner_reasoning_burnout_respects_context_cap() -> None:
    """The burnout retry budget is bounded by the model's context window."""
    calls: list[int] = []
    completions = [
        _BurnoutStubCompletion(
            content="",
            prompt_tokens=10000,
            total_tokens=18192,
            finish_reason="stop",
            reasoning_content="burned thinking",
            reasoning_tokens=8192,
            completion_tokens=8192,
        ),
        _BurnoutStubCompletion(content="done at last", finish_reason="stop"),
    ]
    call_idx = {"n": 0}

    async def complete_fn(**kwargs: Any):
        calls.append(kwargs.get("max_tokens"))
        completion = completions[call_idx["n"]]
        call_idx["n"] += 1
        return completion

    hooks = TurnHooks(
        models=ModelHooks(
            complete=complete_fn,
            resolve_model=lambda _mid: SimpleNamespace(
                provider="test",
                model_id="small-model",
                display_name="Small Model",
                reasoning_model=False,
                thinking_model=False,
                context_window=9000,
            ),
        ),
    )
    with patch(
        "kernel.turn_runner._compute_capability_routing_hint",
        return_value="",
    ):
        result = await run_turn(mode_id="default", query="make the edit", hooks=hooks)

    budget = result.model_requests[0]["context_budget"]
    assert calls == [4096, 9000 - budget["input_tokens"] - budget["margin_tokens"]]
    assert "done at last" in result.response_text


@pytest.mark.asyncio
async def test_turn_runner_rescues_reasoning_burnout_stream() -> None:
    """A streamed burnout (stop, empty content, reasoning-only, no tool
    calls, whole budget burned) is retried once at a doubled budget.

    The 2026-10-13 turn 789 iteration ran on the streaming path; the
    burnout rescue must cover it too, or the doubled-budget retry only
    ever fires on the sibling `length` shape.
    """
    calls: list[int] = []

    def _stream_fn(chunks_by_call: list[list[StreamedChunk]]):
        call_idx = {"n": 0}

        async def stream_fn(**kwargs: Any):
            calls.append(kwargs.get("max_tokens"))
            chunks = chunks_by_call[call_idx["n"]]
            call_idx["n"] += 1
            for chunk in chunks:
                yield chunk

        return stream_fn

    chunks_by_call = [
        # First call: reasoning streams the whole budget, no content. The
        # trailing usage chunk is what real Ollama streams send last —
        # the burnout predicate needs it to verify the burn.
        [
            StreamedChunk(reasoning_content="deliberating the payload " * 500),
            StreamedChunk(
                finish_reason="stop",
                usage=CompletionUsage(
                    prompt_tokens=10000, completion_tokens=4096, total_tokens=14096
                ),
            ),
        ],
        # Retry: the payload fits — the edit lands as a tool call.
        [
            StreamedChunk(reasoning_content="brief check"),
            StreamedChunk(finish_reason="tool_calls"),
        ],
    ]
    stream_fn = _stream_fn(chunks_by_call)

    async def complete_fn(**kwargs: Any):
        return _BurnoutStubCompletion(content="", finish_reason="stop")

    hooks = _build_minimal_hooks(complete_fn, stream_fn)
    with patch(
        "kernel.turn_runner._compute_capability_routing_hint",
        return_value="",
    ):
        result = await run_turn(mode_id="default", query="make the edit", hooks=hooks)

    assert calls == [4096, 8192], f"expected one burnout stream retry at 2x, got {calls}"
    # The streamed-path rescue fired on the carried usage (the trailing
    # usage chunk proves the burn); the retried stream ended tool_calls
    # and the fallback complete() produced the turn's response — the turn
    # did not end on the reasoning-tail caveat with nothing dispatched.
    assert "no visible answer" not in result.response_text


@pytest.mark.asyncio
async def test_turn_runner_burnout_rescue_skipped_after_length_rescue() -> None:
    """One rescue per streamed iteration: the burnout block stays dormant
    when the ``length`` rescue already consumed this iteration's retry.

    Chaining a second stream at the same doubled budget would re-spend
    twice with no headroom gained (the length retry above set
    ``fallback_request`` to the doubled request; a second hop targeting
    ``2 * original`` is capped right back at the same size).
    """
    calls: list[int] = []

    def _stream_fn(chunks_by_call: list[list[StreamedChunk]]):
        call_idx = {"n": 0}

        async def stream_fn(**kwargs: Any):
            calls.append(kwargs.get("max_tokens"))
            chunks = chunks_by_call[call_idx["n"]]
            call_idx["n"] += 1
            for chunk in chunks:
                yield chunk

        return stream_fn

    chunks_by_call = [
        # First call: truncated mid payload with tools offered — the
        # length rescue consumes the iteration's one shot.
        [
            StreamedChunk(content="Writing both files now."),
            StreamedChunk(
                finish_reason="length",
                usage=CompletionUsage(
                    prompt_tokens=10000, completion_tokens=4096, total_tokens=14096
                ),
            ),
        ],
        # Length-rescue retry: the aggregate is the burnout shape (clean
        # stop, whole doubled budget burned on reasoning, no tool call),
        # but the burnout block must NOT fire a second stream.
        [
            StreamedChunk(reasoning_content="still deliberating " * 500),
            StreamedChunk(
                finish_reason="stop",
                usage=CompletionUsage(
                    prompt_tokens=10000, completion_tokens=8192, total_tokens=18192
                ),
            ),
        ],
    ]
    stream_fn = _stream_fn(chunks_by_call)

    async def complete_fn(**kwargs: Any):
        return _BurnoutStubCompletion(content="", finish_reason="stop")

    hooks = _build_minimal_hooks(complete_fn, stream_fn)
    with patch(
        "kernel.turn_runner._compute_capability_routing_hint",
        return_value="",
    ):
        result = await run_turn(mode_id="default", query="make the edit", hooks=hooks)

    assert calls == [4096, 8192], f"expected exactly one rescue per iteration, got {calls}"
    # The length-rescue aggregate owns the iteration's end: the reasoning
    # tail surfaces as the (bounded) caveat response.
    assert result.content_source == "reasoning_tail"


# ── on_delta per-chunk forwarding (v1 streaming display) ──────────────


@pytest.mark.asyncio
async def test_turn_runner_forwards_chunks_to_on_delta() -> None:
    """Every streamed chunk's text/reasoning deltas reach ``on_delta``.

    The runtime's ``on_text_chunk`` transport display streams through this
    hook; without it the chunks were collected locally and only the
    aggregated completion reached the host — the 2026-10-04 "spinner with
    nothing streaming" CLI incident.
    """
    chunks = [
        StreamedChunk(content="Hel"),
        StreamedChunk(reasoning_content="thinking..."),
        StreamedChunk(content="lo world", finish_reason="stop"),
    ]
    stream_fn = _make_streaming_complete_stream(chunks)
    received: list[tuple[str, str]] = []

    async def on_delta(text: str, reasoning: str) -> None:
        received.append((text, reasoning))

    hooks = _build_minimal_hooks(_make_stub_complete("unused"), stream_fn)
    hooks.models.on_delta = on_delta
    result = await run_turn(mode_id="default", query="hi", hooks=hooks)

    assert "Hello world" in result.response_text
    assert received == [("Hel", ""), ("", "thinking..."), ("lo world", "")]


@pytest.mark.asyncio
async def test_turn_runner_skips_empty_chunks_in_on_delta() -> None:
    """Bare metadata chunks (finish_reason only) do not reach ``on_delta``."""
    chunks = [
        StreamedChunk(content="answer"),
        StreamedChunk(finish_reason="stop"),
    ]
    stream_fn = _make_streaming_complete_stream(chunks)
    received: list[tuple[str, str]] = []

    async def on_delta(text: str, reasoning: str) -> None:
        received.append((text, reasoning))

    hooks = _build_minimal_hooks(_make_stub_complete("unused"), stream_fn)
    hooks.models.on_delta = on_delta
    await run_turn(mode_id="default", query="hi", hooks=hooks)

    assert received == [("answer", "")]


@pytest.mark.asyncio
async def test_turn_runner_on_delta_absent_is_noop() -> None:
    """``on_delta=None`` (default) leaves the streaming path unchanged."""
    chunks = [StreamedChunk(content="plain stream")]
    stream_fn = _make_streaming_complete_stream(chunks)
    hooks = _build_minimal_hooks(_make_stub_complete("unused"), stream_fn)
    assert hooks.models.on_delta is None
    result = await run_turn(mode_id="default", query="hi", hooks=hooks)
    assert "plain stream" in result.response_text


# ── Stream idle watchdog (the 2026-10-06 mid-stream freeze) ───────────


@pytest.mark.asyncio
async def test_turn_runner_stalled_stream_aborts_after_idle_timeout() -> None:
    """A stream that goes silent mid-generation aborts after the idle
    timeout instead of hanging the turn forever.

    Regression for the 2026-10-06 glm-5.3:cloud marathon: the doubled
    truncation-rescue stream stalled mid-generation (socket byte counters
    frozen for 8+ minutes) and ``_collect_stream``'s bare ``async for``
    blocked indefinitely — no timeout existed between the POST and the
    next lifecycle event. The turn only recovered because the upstream
    happened to resume; a permanent upstream hang left the CLI dead.
    The stalled call raises ``StreamIdleTimeout``; the turn loop's
    provider-failure path turns that into a turn error the host renders.
    """
    import asyncio

    import kernel.turn_runner
    from kernel.turn_runner import _collect_stream

    chunks_yielded: list[StreamedChunk] = []

    def _stalled_stream_fn(idle_s: float):
        async def stream_fn(*_args: Any, **_kwargs: Any):
            # First chunk arrives promptly…
            chunk = StreamedChunk(content="thinking")
            chunks_yielded.append(chunk)
            yield chunk
            # …then the upstream freezes: nothing else ever arrives.
            await asyncio.sleep(idle_s * 10)

        return stream_fn

    request = _request()
    before = time.monotonic()
    with (
        patch(
            "kernel.turn_runner._kernel_settings",
            SimpleNamespace(stream_idle_timeout_s=0.05),
        ),
        pytest.raises(kernel.turn_runner.StreamIdleTimeout),
    ):
        await _collect_stream(_stalled_stream_fn(0.05), request, None)
    elapsed = time.monotonic() - before
    # The watchdog aborted the stalled stream within the timeout budget
    # (not the 0.5s the sleep would have taken).
    assert elapsed < 0.4, f"watchdog did not bound the stall: {elapsed:.2f}s"
    # The chunk that arrived before the stall was produced (the generator
    # ran at least once before freezing).
    assert [c.content for c in chunks_yielded] == ["thinking"]


@pytest.mark.asyncio
async def test_turn_runner_healthy_stream_unaffected_by_watchdog() -> None:
    """A stream that keeps producing within the idle window is consumed
    fully — the watchdog never fires on a live generation, even one
    slower than the timeout in aggregate."""
    import asyncio

    from kernel.turn_runner import _collect_stream

    def _slow_but_alive_stream_fn():
        async def stream_fn(*_args: Any, **_kwargs: Any):
            for _ in range(3):
                yield StreamedChunk(content="x")
                await asyncio.sleep(0.02)  # well under the idle timeout

        return stream_fn

    with patch(
        "kernel.turn_runner._kernel_settings",
        SimpleNamespace(stream_idle_timeout_s=0.5),
    ):
        chunks, _ = await _collect_stream(_slow_but_alive_stream_fn(), _request(), None)
    assert [c.content for c in chunks] == ["x", "x", "x"]


def test_stream_idle_timeout_setting_env_override() -> None:
    """``STREAM_IDLE_TIMEOUT_S`` tunes the watchdog; 0 disables it."""
    from kernel.config import Settings

    assert Settings.model_validate({"stream_idle_timeout_s": "120"}).stream_idle_timeout_s == 120
    assert Settings.model_validate({"stream_idle_timeout_s": "0"}).stream_idle_timeout_s == 0


def test_aggregate_keeps_a_call_whole_when_the_id_repeats() -> None:
    """A backend that repeats the id on every fragment does not split its call.

    The OpenAI contract does not guarantee the id arrives only on the first
    fragment. A backend that shares one index across parallel calls AND
    repeats the id on each fragment would have its own call misread as a
    new call, splitting one call's arguments across two entries and
    leaving two tool calls with the same id — fatal to a replay, because
    a duplicate tool-call id is rejected on its own.
    """
    chunks = [
        StreamedChunk(
            tool_call_deltas=[
                {"index": 0, "id": "call_a", "function": {"name": "a_tool", "arguments": '{"x":'}}
            ]
        ),
        StreamedChunk(
            tool_call_deltas=[
                {"index": 0, "id": "call_b", "function": {"name": "b_tool", "arguments": '{"y":'}}
            ]
        ),
        StreamedChunk(
            tool_call_deltas=[{"index": 0, "id": "call_a", "function": {"arguments": "1}"}}]
        ),
        StreamedChunk(
            tool_call_deltas=[{"index": 0, "id": "call_b", "function": {"arguments": "2}"}}]
        ),
    ]
    result = _aggregate_streamed_chunks(chunks)
    assert len(result.tool_calls) == 2
    ids = [call["id"] for call in result.tool_calls]
    assert ids == ["call_a", "call_b"], "ids must stay unique and in arrival order"
    assert result.tool_calls[0]["function"]["arguments"] == '{"x":1}'
    assert result.tool_calls[1]["function"]["arguments"] == '{"y":2}'


def test_aggregate_synthetic_slot_does_not_collide_with_a_later_index() -> None:
    """A slot the stitcher allocates cannot shadow a real index.

    The stitcher allocates its own slot for a parallel call that reuses an
    occupied index. Allocating in the backend's own index keyspace let
    that synthetic slot collide with a real index arriving later, and the
    collision appended one call's continuation to another call's
    arguments.
    """
    chunks = [
        StreamedChunk(
            tool_call_deltas=[
                {"index": 0, "id": "call_a", "function": {"name": "a_tool", "arguments": "{}"}}
            ]
        ),
        StreamedChunk(
            tool_call_deltas=[
                {"index": 0, "id": "call_b", "function": {"name": "b_tool", "arguments": "{}"}}
            ]
        ),
        StreamedChunk(
            tool_call_deltas=[
                {"index": 1, "id": "call_c", "function": {"name": "c_tool", "arguments": "{}"}}
            ]
        ),
        # A continuation for the call at index 1.
        StreamedChunk(tool_call_deltas=[{"index": 1, "function": {"arguments": '{"w":4}'}}]),
    ]
    result = _aggregate_streamed_chunks(chunks)
    by_id = {call["id"]: call for call in result.tool_calls}
    assert set(by_id) == {"call_a", "call_b", "call_c"}
    assert by_id["call_a"]["function"]["arguments"] == "{}"
    assert by_id["call_b"]["function"]["arguments"] == "{}", (
        "call_b must not receive another call's continuation"
    )
    assert by_id["call_c"]["function"]["arguments"] == '{}{"w":4}', (
        "the continuation belongs to the call at its own index"
    )


def test_aggregate_separates_distinct_indices_without_ids() -> None:
    """Two calls at distinct indices stay two calls even without ids.

    The gateway drops the ``id`` key when a backend sends an empty id, so
    a call can arrive with an index and no id. Merging such a call into
    whichever call started last reproduced the original 2026-10-19
    incident: one tool call with a glued-together name and two JSON
    objects end to end.
    """
    chunks = [
        StreamedChunk(
            tool_call_deltas=[{"index": 0, "function": {"name": "tool_a", "arguments": '{"x":1}'}}]
        ),
        StreamedChunk(
            tool_call_deltas=[{"index": 1, "function": {"name": "tool_b", "arguments": '{"y":2}'}}]
        ),
    ]
    result = _aggregate_streamed_chunks(chunks)
    assert len(result.tool_calls) == 2, "distinct indices must not be merged"
    assert [call["function"]["name"] for call in result.tool_calls] == ["tool_a", "tool_b"]
    assert json.loads(result.tool_calls[0]["function"]["arguments"]) == {"x": 1}
    assert json.loads(result.tool_calls[1]["function"]["arguments"]) == {"y": 2}


def test_aggregate_keeps_one_call_whole_when_the_id_arrives_late() -> None:
    """An id that arrives on a later fragment names the call already staged.

    The OpenAI streaming contract does not guarantee the id arrives on
    the first fragment. When it arrives later, the call it names is the
    one already staged at that index — treating it as a new call split
    the call's arguments across two entries, one with an empty id and one
    with an empty name, neither parseable.
    """
    chunks = [
        StreamedChunk(
            tool_call_deltas=[{"index": 0, "function": {"name": "tool_a", "arguments": '{"x":'}}]
        ),
        StreamedChunk(
            tool_call_deltas=[{"index": 0, "id": "call_a", "function": {"arguments": "1}"}}]
        ),
    ]
    result = _aggregate_streamed_chunks(chunks)
    assert len(result.tool_calls) == 1, "the id must adopt the staged call, not split it"
    call = result.tool_calls[0]
    assert call["id"] == "call_a"
    assert call["function"]["name"] == "tool_a"
    assert json.loads(call["function"]["arguments"]) == {"x": 1}


def test_aggregate_separates_calls_at_a_shared_index_when_the_first_has_no_id() -> None:
    """A call with an id never merges into an id-less call at the same index.

    This is the shape the 2026-10-19 incident actually had: Ollama
    streams every parallel call at ``index: 0``. When the first call
    carried no id and the second carried one, adopting the occupant
    merged them into one call named ``list_filesread_file`` with two
    JSON objects end to end — the glued payload the backend rejects with
    ``400 invalid tool call arguments``.
    """
    chunks = [
        StreamedChunk(
            tool_call_deltas=[
                {"index": 0, "function": {"name": "list_files", "arguments": '{"path":"."}'}}
            ]
        ),
        StreamedChunk(
            tool_call_deltas=[
                {
                    "index": 0,
                    "id": "call_b",
                    "function": {"name": "read_file", "arguments": '{"path":"a.md"}'},
                }
            ]
        ),
    ]
    result = _aggregate_streamed_chunks(chunks)
    assert len(result.tool_calls) == 2, "a call with an id must not merge into an id-less call"
    assert result.tool_calls[0]["function"]["name"] == "list_files"
    assert result.tool_calls[1]["id"] == "call_b"
    assert result.tool_calls[1]["function"]["name"] == "read_file"


def test_aggregate_keeps_one_call_whole_without_an_index() -> None:
    """A continuation with neither an id nor an index continues its call.

    The OpenAI-compatible gateway always supplies an index, but
    ``StreamedChunk.tool_call_deltas`` does not require one. Treating a
    missing index as a new call split the call in two, leaving one entry
    with an empty id and one with an empty name — neither parseable, and
    the empty name produced a 500 on replay.
    """
    chunks = [
        StreamedChunk(
            tool_call_deltas=[
                {"id": "call_a", "function": {"name": "tool_a", "arguments": '{"x":'}}
            ]
        ),
        StreamedChunk(tool_call_deltas=[{"function": {"arguments": "1}"}}]),
    ]
    result = _aggregate_streamed_chunks(chunks)
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call["id"] == "call_a"
    assert call["function"]["name"] == "tool_a"
    assert json.loads(call["function"]["arguments"]) == {"x": 1}


def test_aggregate_a_negative_backend_index_keeps_its_own_call() -> None:
    """A negative backend index cannot land on a stitcher-allocated slot.

    The stitcher allocates slots of its own for calls that share an
    index. Those slots must be unreachable by any index a backend can
    send, including a negative one.
    """
    chunks = [
        StreamedChunk(
            tool_call_deltas=[
                {"index": 0, "id": "call_a", "function": {"name": "a_tool", "arguments": "{}"}}
            ]
        ),
        StreamedChunk(
            tool_call_deltas=[
                {"index": 0, "id": "call_b", "function": {"name": "b_tool", "arguments": "{}"}}
            ]
        ),
        StreamedChunk(
            tool_call_deltas=[
                {"index": -1, "id": "call_c", "function": {"name": "c_tool", "arguments": "{}"}}
            ]
        ),
    ]
    result = _aggregate_streamed_chunks(chunks)
    assert [call["id"] for call in result.tool_calls] == ["call_a", "call_b", "call_c"]
    assert [call["function"]["name"] for call in result.tool_calls] == [
        "a_tool",
        "b_tool",
        "c_tool",
    ]

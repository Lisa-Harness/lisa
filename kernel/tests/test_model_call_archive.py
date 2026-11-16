"""Model-call archive response recording (Spec: MODEL-CALL-ARCHIVE).

Covers the acceptance cases for the response half of the archive: the
request/response pair staged on the turn state, the WARNING-not-raise
posture for a failing recorder, the graceful wall-budget expiry pair, the
streamed truncation-rescue two-pair split, and the off-turn snapshot
shape through the host adapter's audit seam.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest
from kernel.provider_contracts import (
    CompletionRequest,
    CompletionResult,
    CompletionUsage,
    StreamedChunk,
)
from kernel.seams.host_adapter import CompletionAuditContext
from kernel.turn_phases._turn_state import _TurnState
from kernel.turn_runner import (
    _record_request_snapshot,
    _record_response_snapshot,
    _record_unarchived_call,
    _streamed_gateway_completion,
)

LOG = logging.getLogger("test_model_call_archive")


def _hooks(
    request_refs: list[str], response_refs: list[str], *, fail_response: bool = False
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
    """Recorder bundle capturing snapshots; refs handed out in order."""
    requests: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []

    def record_request(snapshot: dict[str, Any]) -> str:
        requests.append(snapshot)
        return request_refs.pop(0)

    def record_response(snapshot: dict[str, Any]) -> str:
        responses.append(snapshot)
        if fail_response:
            raise RuntimeError("recorder exploded")
        return response_refs.pop(0)

    hooks = SimpleNamespace(
        models=SimpleNamespace(
            record_request=record_request,
            record_response=record_response,
            on_delta=None,
        )
    )
    return hooks, requests, responses


def _state() -> _TurnState:
    state = _TurnState()
    state.iteration = 2
    return state


def _completion(**overrides: Any) -> CompletionResult:
    fields: dict[str, Any] = {
        "content": "done",
        "provider": "prov",
        "model_id": "m",
        "display_name": "M",
        "request_id": "req_1",
        "tool_calls": [],
        "usage": CompletionUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
        "duration_ms": 12.0,
        "finish_reason": "stop",
    }
    fields.update(overrides)
    return CompletionResult(**fields)


def test_request_stages_entry_and_response_stamps_ref() -> None:
    hooks, _requests, responses = _hooks(["ra"], ["rr"])
    state = _state()
    call_id = _record_request_snapshot(state, hooks, {"messages": [], "model_id": "m"})
    assert state.current_call_entry is state.model_requests[0]
    assert state.model_requests[0]["call_id"] == call_id

    _record_response_snapshot(
        state=state,
        hooks=hooks,
        completion=_completion(),
        model_def=SimpleNamespace(provider="prov", model_id="m"),
        elapsed_ms=1.0,
    )

    assert state.model_requests[0]["response_ref"] == "rr"
    assert state.current_call_entry is None
    snap = responses[0]
    assert snap["call_id"] == call_id
    assert snap["attempt"] == "primary"
    assert snap["iteration"] == 2
    assert snap["request_ref"] == "ra"
    assert snap["outcome"] == "ok"
    assert snap["content"] == "done"
    assert snap["usage"]["prompt_tokens"] == 5
    assert snap["usage"]["cached_tokens"] is None
    assert snap["duration_ms"] == 12.0
    assert snap["streamed"] is False
    assert "error" not in snap


def test_failing_response_recorder_leaves_countable_gap() -> None:
    hooks, _requests, responses = _hooks(["ra"], [], fail_response=True)
    state = _state()
    _record_request_snapshot(state, hooks, {"messages": [], "model_id": "m"})

    # The WARNING posture: recording failure must not raise.
    _record_response_snapshot(
        state=state,
        hooks=hooks,
        completion=_completion(),
        model_def=SimpleNamespace(provider="prov", model_id="m"),
    )

    assert len(responses) == 1  # the snapshot was attempted
    assert state.model_requests[0]["response_ref"] is None
    assert state.current_call_entry is None


def test_wall_budget_expiry_records_error_pair() -> None:
    hooks, _requests, responses = _hooks(["ra"], ["rr"])
    state = _state()
    _record_request_snapshot(state, hooks, {"messages": [], "model_id": "m"})

    _record_unarchived_call(
        state=state, hooks=hooks, outcome="wall_budget_expired", error=TimeoutError("spent")
    )

    snap = responses[0]
    assert snap["outcome"] == "wall_budget_expired"
    assert snap["content"] == ""
    assert snap["tool_calls"] == []
    assert snap["usage"] is None
    assert snap["error"] == {"class": "TimeoutError", "message": "spent"}
    assert state.model_requests[0]["response_ref"] == "rr"


class _FakeGateway:
    """Streaming gateway stub: scripted streams and scripted completions."""

    def __init__(
        self,
        streams: list[list[StreamedChunk]],
        completions: list[CompletionResult],
    ) -> None:
        self._streams = list(streams)
        self._completions = list(completions)

    async def complete_stream(self, request: CompletionRequest) -> Any:
        for chunk in self._streams.pop(0):
            yield chunk

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        return self._completions.pop(0)


def _stream_request(model: Any) -> CompletionRequest:
    return CompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        model_def=model,
        max_tokens=4096,
        temperature=0.4,
        tools=[{"type": "function", "function": {"name": "t", "parameters": {}}}],
    )


@pytest.mark.asyncio
async def test_streamed_truncation_rescue_archives_two_pairs() -> None:
    model = SimpleNamespace(provider="prov", model_id="m", context_window=131072)
    hooks, _requests, responses = _hooks(["ra", "rb"], ["rr1", "rr2"])
    state = _state()
    # The real flow stages the primary request before dispatching it
    # (_model_completion_for_iteration), so the primary response has an
    # entry to stamp; mirror that here.
    _record_request_snapshot(state, hooks, {"messages": [], "model_id": "m"})
    gateway = _FakeGateway(
        streams=[
            [StreamedChunk(content="half answer", finish_reason="length")],
            [StreamedChunk(content="full answer", finish_reason="stop")],
        ],
        completions=[],
    )

    completion = await _streamed_gateway_completion(
        gateway=gateway,
        hooks=hooks,
        mode_id="coding_execute",
        effective_model_def=model,
        effective_tools=[{"type": "function", "function": {"name": "t", "parameters": {}}}],
        tools_offered_this_turn=True,
        completion_request=_stream_request(model),
        complete_started=0.0,
        log=LOG,
        state=state,
        request_snapshot={"messages": [], "model_id": "m"},
    )

    assert completion.content == "full answer"
    assert len(responses) == 2
    first, second = responses
    assert first["attempt"] == "primary"
    assert first["finish_reason"] == "length"
    assert first["content"] == "half answer"
    assert first["streamed"] is True
    assert first["chunk_count"] == 1
    assert second["attempt"] == "truncation_continuation"
    assert second["content"] == "full answer"
    assert first["call_id"] != second["call_id"]
    assert state.model_requests[0]["response_ref"] == "rr1"
    assert state.model_requests[1]["response_ref"] == "rr2"


def test_off_turn_response_snapshot_shape() -> None:
    from kernel.seams.host_adapter import _CallableHost

    captured: list[dict[str, Any]] = []

    def record_response(snapshot: dict[str, Any]) -> str:
        captured.append(snapshot)
        return "rresp"

    audit = CompletionAuditContext(
        record_request=lambda snapshot: "rreq",
        event_bus=None,
        operation_id="mine_facts",
        turn_index=7,
        user_slug="u",
        thread_store_slug="s",
        call_id="call_1",
        dispatch_id="disp_1",
        record_response=record_response,
    )
    host = _CallableHost(audit=audit)
    request = CompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        model_def=SimpleNamespace(provider="prov", model_id="m"),
        max_tokens=100,
        temperature=0.4,
    )

    ref = host._record_response_snapshot(audit, request, _completion(), "rreq")

    assert ref == "rresp"
    snap = captured[0]
    assert snap["call_id"] == "call_1"
    assert snap["attempt"] is None
    assert snap["iteration"] is None
    assert snap["request_ref"] == "rreq"
    assert snap["provider"] == "prov"
    assert snap["model_id"] == "m"
    assert snap["provider_request_id"] == "req_1"
    assert snap["streamed"] is False
    assert snap["duration_ms"] == 12.0
    assert "error" not in snap

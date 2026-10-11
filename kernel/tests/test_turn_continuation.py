"""Approval must suspend dispatch without rebuilding the model conversation."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest
from kernel.approval import ApprovalInvocation, ApprovalUnavailable, InMemoryApprovalService
from kernel.models import ToolDefinition, ToolResult
from kernel.turn_continuation import TurnContinuation, capture_state, resume_after_pause
from kernel.turn_hooks import ContextHooks, HostServiceHandles, ModelHooks, TurnHooks, TurnIdentity
from kernel.turn_phases._turn_state import _TurnState
from kernel.turn_runner import _project_provider_messages, run_turn
from pydantic import TypeAdapter


class _Allow(InMemoryApprovalService):
    async def request(self, invocation: ApprovalInvocation) -> ApprovalInvocation:
        recorded = await super().request(invocation)
        await self.approve(recorded.request_id, recorded.identity)
        return recorded


@pytest.mark.asyncio
@pytest.mark.parametrize("automatic", [False, True])
async def test_approval_keeps_pending_batch_and_context(
    monkeypatch: pytest.MonkeyPatch, automatic: bool
) -> None:
    executed: list[str] = []
    requests: list[list[dict[str, Any]]] = []

    async def execute(ctx: Any) -> ToolResult:
        executed.append(ctx.tool_id)
        return ToolResult(output_text=f"result:{ctx.tool_id}")

    monkeypatch.setattr(
        "kernel.tools.registry._TOOL_CATALOG",
        {
            name: ToolDefinition(
                tool_id=name,
                pack_id="test",
                effect="write" if name == "write" else "read",
                approval_required=name == "write",
                implementation_module=SimpleNamespace(execute=execute),
            )
            for name in ("read", "write", "after")
        },
    )

    async def complete(**kwargs: Any) -> Any:
        requests.append(copy.deepcopy(kwargs["messages"]))
        if len(requests) == 1:
            return SimpleNamespace(
                content="",
                tool_calls=[
                    {"id": name, "function": {"name": name, "arguments": "{}"}}
                    for name in ("read", "write", "after")
                ],
            )
        return SimpleNamespace(content="finished", tool_calls=[])

    def forbidden_replay() -> list[dict[str, Any]]:
        raise AssertionError("resume reconstructed the turn")

    def forbidden_context(*_: Any) -> str:
        raise AssertionError("resume reassembled context")

    service = _Allow() if automatic else InMemoryApprovalService()
    hooks = TurnHooks(
        models=ModelHooks(complete=complete),
        services=HostServiceHandles(approval_service=service),
        identity=TurnIdentity(workspace_id="w", user_id="u", task_id="task"),
    )
    result = await run_turn(mode_id="default", query="do it", hooks=hooks)
    if not automatic:
        assert executed == ["read"]
        assert result.continuation is not None
        assert result.pending_approval is not None
        await service.approve(result.pending_approval.request_id, result.pending_approval.identity)
        saved = TurnContinuation.model_validate_json(result.continuation.model_dump_json())
        hooks.context = ContextHooks(
            prior_messages=forbidden_replay,
            assemble_user_context=forbidden_context,
        )

        async def unavailable_receipt(*_: Any) -> None:
            return None

        with monkeypatch.context() as scoped:
            scoped.setattr(service, "consume", unavailable_receipt)
            with pytest.raises(ApprovalUnavailable):
                await run_turn(mode_id="default", query="do it", hooks=hooks, continuation=saved)
        assert executed == ["read"]
        assert len(requests) == 1
        result = await run_turn(mode_id="default", query="do it", hooks=hooks, continuation=saved)
    assert result.response_text == "finished"
    assert executed == ["read", "write", "after"]
    assert len(requests) == 2
    assert requests[0] == requests[1][: len(requests[0])]
    assert result.tool_call_count == 3
    if not automatic:
        with pytest.raises(ValueError, match="unexpired, unconsumed"):
            await run_turn(mode_id="default", query="do it", hooks=hooks, continuation=saved)
        assert executed == ["read", "write", "after"]


@pytest.mark.asyncio
async def test_auto_approval_cap_keeps_batch_and_consumed_allowance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("kernel.config.settings.max_auto_approved_actions", 1)
    executed: list[str] = []

    async def execute(ctx: Any) -> ToolResult:
        executed.append(ctx.tool_id)
        return ToolResult(output_text=ctx.tool_id)

    monkeypatch.setattr(
        "kernel.tools.registry._TOOL_CATALOG",
        {
            name: ToolDefinition(
                tool_id=name,
                pack_id="test",
                effect="write",
                approval_required=True,
                implementation_module=SimpleNamespace(execute=execute),
            )
            for name in ("first", "second", "third")
        },
    )
    requests: list[list[dict[str, Any]]] = []

    async def complete(**kwargs: Any) -> Any:
        requests.append(copy.deepcopy(kwargs["messages"]))
        if len(requests) == 1:
            return SimpleNamespace(
                content="",
                tool_calls=[
                    {"id": name, "function": {"name": name, "arguments": "{}"}}
                    for name in ("first", "second", "third")
                ],
            )
        return SimpleNamespace(content="finished", tool_calls=[])

    hooks = TurnHooks(
        models=ModelHooks(complete=complete),
        services=HostServiceHandles(approval_service=_Allow()),
    )
    paused = await run_turn(mode_id="default", query="do it", hooks=hooks)
    assert executed == ["first"]
    assert paused.continuation is not None
    assert paused.continuation.reason == "auto_approval_cap"
    assert paused.continuation.restore_state().auto_approved_actions == 1
    saved = paused.continuation.grant_auto_approval_allowance()
    finished = await run_turn(mode_id="default", query="do it", hooks=hooks, continuation=saved)
    assert finished.response_text == "finished"
    assert finished.tool_call_count == 3
    assert executed == ["first", "second", "third"]
    assert len(requests) == 2
    assert requests[0] == requests[1][: len(requests[0])]


@pytest.mark.asyncio
async def test_request_sink_keeps_only_refs_and_exact_snapshots() -> None:
    snapshots: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []

    def record(request: dict[str, Any]) -> str:
        snapshots.append(copy.deepcopy(request))
        return f"result://snapshot-{len(snapshots)}"

    async def complete(**kwargs: Any) -> Any:
        requests.append(copy.deepcopy(kwargs))
        return SimpleNamespace(content="finished", tool_calls=[])

    result = await run_turn(
        mode_id="default",
        query="do it",
        hooks=TurnHooks(models=ModelHooks(complete=complete, record_request=record)),
    )
    assert result.model_requests == [{"iteration": 1, "ref": "result://snapshot-1"}]
    assert snapshots[0]["messages"] == requests[0]["messages"]
    assert snapshots[0]["tools"] == requests[0]["tools"]


def test_provider_residency_survives_a_serialized_checkpoint() -> None:
    raw = [
        {"role": "system", "content": "stable"},
        {"role": "tool", "content": "large original result", "tool_call_id": "first"},
    ]
    inputs: list[list[dict[str, Any]]] = []

    def project(
        messages: list[dict[str, Any]],
        _task: str,
        *,
        budget_bytes: int,
    ) -> list[dict[str, Any]]:
        assert budget_bytes > 0
        inputs.append(copy.deepcopy(messages))
        projected = copy.deepcopy(messages)
        projected[1]["content"] = "result reference"
        return projected

    state = _TurnState()
    hooks = TurnHooks(context=ContextHooks(project_messages=project))
    _project_provider_messages(state, hooks, raw, SimpleNamespace())
    assert raw[1]["content"] == "large original result"
    restored = TypeAdapter(_TurnState).validate_json(capture_state(state))
    _project_provider_messages(
        restored,
        hooks,
        [*raw, {"role": "user", "content": "new tail"}],
        SimpleNamespace(),
    )
    assert inputs[1][1]["content"] == "result reference"
    assert inputs[1][-1]["content"] == "new tail"


@pytest.mark.asyncio
async def test_cap_pause_resumes_in_place_with_a_bounded_new_allowance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def execute(ctx: Any) -> ToolResult:
        return ToolResult(output_text=f"result:{ctx.tool_id}")

    monkeypatch.setattr(
        "kernel.tools.registry._TOOL_CATALOG",
        {
            "read": ToolDefinition(
                tool_id="read",
                pack_id="test",
                effect="read",
                implementation_module=SimpleNamespace(execute=execute),
            )
        },
    )
    requests: list[list[dict[str, Any]]] = []
    finish_after = 3

    async def complete(**kwargs: Any) -> Any:
        requests.append(copy.deepcopy(kwargs["messages"]))
        if len(requests) >= finish_after:
            return SimpleNamespace(content="finished", tool_calls=[])
        call_id = f"read-{len(requests)}"
        return SimpleNamespace(
            content="",
            tool_calls=[{"id": call_id, "function": {"name": "read", "arguments": "{}"}}],
        )

    def forbidden_replay() -> list[dict[str, Any]]:
        raise AssertionError("resume reconstructed the turn")

    hooks = TurnHooks(
        models=ModelHooks(complete=complete),
        identity=TurnIdentity(workspace_id="w", user_id="u", task_id="task"),
    )
    paused = await run_turn(mode_id="default", query="do it", hooks=hooks, max_tool_iterations=3)
    assert paused.operator_exit == "paused"
    assert len(requests) == 2
    assert paused.continuation is not None
    assert paused.continuation.reason == "cap"
    assert not paused.continuation.requires_approval

    hooks.context = ContextHooks(prior_messages=forbidden_replay)
    saved = resume_after_pause(
        TurnContinuation.model_validate_json(paused.continuation.model_dump_json()),
        "keep going",
    )
    assert saved.max_iterations == 6
    resumed = await run_turn(mode_id="default", query="do it", hooks=hooks, continuation=saved)
    assert resumed.response_text == "finished"
    assert len(requests) == 3
    assert requests[1] == requests[2][: len(requests[1])]
    assert requests[2][-1] == {"role": "user", "content": "keep going"}
    assert resumed.tool_call_count == 2

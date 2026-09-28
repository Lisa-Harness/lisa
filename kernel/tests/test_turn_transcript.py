"""Tests for the verbatim turn transcript, model-request capture, and prior-message replay."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from kernel.approval import InMemoryApprovalService
from kernel.models import ToolDefinition, ToolExecutionContext, ToolResult
from kernel.turn_hooks import ContextHooks, HostServiceHandles, ModelHooks, TurnIdentity
from kernel.turn_runner import TurnHooks, run_turn
from pydantic import BaseModel


class _Parameters(BaseModel):
    path: str


@dataclass(frozen=True)
class _Completion:
    content: str = ""
    tool_calls: list[dict[str, Any]] | None = None


def _hooks(complete: Any, prior: list[dict[str, Any]] | None = None) -> TurnHooks:
    return TurnHooks(
        models=ModelHooks(complete=complete),
        context=ContextHooks(prior_messages=(lambda: prior) if prior is not None else None),
        services=HostServiceHandles(approval_service=InMemoryApprovalService()),
        identity=TurnIdentity(
            workspace_id="workspace",
            user_id="user",
            user_slug="user",
            surface_id="surface",
            session_plan_digest="plan-a",
        ),
    )


def _register_approval_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    async def execute(_ctx: ToolExecutionContext) -> ToolResult:
        return ToolResult(output_text="unexpected")

    definition = ToolDefinition(
        tool_id="write_file",
        description="Write a file.",
        pack_id="some-pack",
        parameters_schema=_Parameters,
        effect="write",
        approval_required=True,
        implementation_module=SimpleNamespace(execute=execute),
    )
    monkeypatch.setattr("kernel.tools.registry._TOOL_CATALOG", {"write_file": definition})


@pytest.mark.asyncio
async def test_approval_paused_turn_carries_transcript_and_model_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_approval_tool(monkeypatch)
    tool_call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "write_file", "arguments": '{"path":"src/a.py"}'},
    }

    async def complete(**_kwargs: Any) -> _Completion:
        return _Completion(tool_calls=[tool_call])

    result = await run_turn(mode_id="default", query="change the file", hooks=_hooks(complete))

    assert result.pending_approval is not None
    assert result.transcript[0]["role"] == "user"
    assert "change the file" in result.transcript[0]["content"]
    assert any(m.get("tool_calls") for m in result.transcript if m["role"] == "assistant")
    assert len(result.model_requests) == 1
    assert result.model_requests[0]["messages"][0]["role"] == "system"


@pytest.mark.asyncio
async def test_prior_messages_replay_before_query_and_stay_out_of_transcript() -> None:
    prior = [
        {"role": "user", "content": "old request"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "t1_a", "type": "function", "function": {"name": "r", "arguments": "{}"}},
                {"id": "t1_b", "type": "function", "function": {"name": "r", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "t1_a", "content": "full file contents"},
        {"role": "tool", "tool_call_id": "orphan", "content": "dropped"},
        {"role": "assistant", "content": "paused"},
    ]

    async def complete(**_kwargs: Any) -> _Completion:
        return _Completion(content="done")

    result = await run_turn(mode_id="default", query="continue", hooks=_hooks(complete, prior))

    sent = result.model_requests[0]["messages"]
    assert [m["role"] for m in sent[:6]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert [c["id"] for c in sent[2]["tool_calls"]] == ["t1_a"]
    assert sent[3]["content"] == "full file contents"
    assert "continue" in sent[5]["content"]
    assert result.transcript[0]["role"] == "user"
    assert "old request" not in str(result.transcript)

"""Tests for the mine_thread_segment operation.

Mocks the OperationHost seam (complete / resolve_model / read_authored_text /
execute_tool) to verify the segment distillation → semantic persistence flow
without a live model or SQLite store.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from kernel.models import OperationExecutionContext, ToolResult
from kernel.seams.host_adapter import build_callable_host
from sdk.packs.memory.operations.memory import mine_thread_segment as op


def _turns() -> list[dict[str, Any]]:
    return [
        {"turn": 10, "query": "how do we start the GCP bind VM?", "response_text": "needs project_id"},
        {"turn": 11, "query": "what model do we prefer?", "response_text": "user prefers Claude"},
        {"turn": 12, "query": "tools were down again", "response_text": "cs_list_vms failed"},
    ]


def _make_ctx(
    *,
    params: dict[str, Any],
    complete_fn: Any,
    execute_tool_fn: Any,
) -> OperationExecutionContext:
    def resolve_model_fn(model_id: str | None) -> dict[str, Any]:
        return {"model_id": "test-model"}

    def read_authored_text_fn(path: str) -> str:
        return "Mining system prompt (stubbed)."

    return OperationExecutionContext(
        operation_id="mine_thread_segment",
        user_slug="adam",
        user_id="adam",
        workspace_id="default",
        params=params,
        host=build_callable_host(
            complete_fn=complete_fn,
            resolve_model_fn=resolve_model_fn,
            read_authored_text_fn=read_authored_text_fn,
            execute_tool_fn=execute_tool_fn,
        ),
    )


@pytest.mark.asyncio
async def test_mine_stores_semantic_records_with_provenance() -> None:
    captured: list[tuple[str, dict[str, Any]]] = []

    async def complete_fn(messages: list[dict[str, Any]], **_: Any) -> str:
        return json.dumps({"facts": ["GCP start-vm requires project_id", "User prefers Claude"]})

    async def execute_tool_fn(tool_id: str, ctx: Any) -> ToolResult:
        params = getattr(ctx, "parameters", {}) or {}
        captured.append((tool_id, dict(params)))
        if tool_id == "memory.add":
            return ToolResult(tool_id=tool_id, output_text="Memory stored with ID 42")
        return ToolResult(tool_id=tool_id, error=f"unknown tool {tool_id}")

    ctx = _make_ctx(
        params={"turns": _turns(), "user_slug": "adam"},
        complete_fn=complete_fn,
        execute_tool_fn=execute_tool_fn,
    )

    result = await op.execute(ctx)

    assert result.extra.get("mined") is True
    assert result.extra.get("memory_ids") == [42, 42]
    assert result.extra.get("thread_turn_lo") == 10
    assert result.extra.get("thread_turn_hi") == 12
    add_calls = [c for c in captured if c[0] == "memory.add"]
    assert len(add_calls) == 2
    add_params = add_calls[0][1]
    assert add_params["memory_kind"] == "semantic"
    assert add_params["operation_id"] == "mine_thread_segment"
    assert add_params["memory_tier"] == "long_term"
    assert add_params["content_json"]["thread_turn_lo"] == 10
    assert add_params["content_json"]["thread_turn_hi"] == 12


@pytest.mark.asyncio
async def test_mining_nothing_durable_stores_nothing() -> None:
    async def complete_fn(messages: list[dict[str, Any]], **_: Any) -> str:
        return json.dumps({"facts": []})

    async def execute_tool_fn(tool_id: str, ctx: Any) -> ToolResult:
        return ToolResult(tool_id=tool_id, error=f"unexpected call {tool_id}")

    ctx = _make_ctx(
        params={"turns": _turns()},
        complete_fn=complete_fn,
        execute_tool_fn=execute_tool_fn,
    )
    result = await op.execute(ctx)
    assert result.extra.get("mined") is False
    assert result.extra.get("memory_ids") == []


@pytest.mark.asyncio
async def test_invalid_params_reports_error() -> None:
    ctx = _make_ctx(
        params={"turns": []},
        complete_fn=None,
        execute_tool_fn=None,
    )
    result = await op.execute(ctx)
    assert "Invalid mining parameters" in result.output_text
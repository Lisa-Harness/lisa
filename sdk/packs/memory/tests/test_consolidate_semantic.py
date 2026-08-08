"""Tests for the consolidate_semantic operation.

Mocks the OperationHost seam (complete / resolve_model / read_authored_text /
execute_tool) to verify the distillation → persist → retire flow without a
live model or SQLite store. The structured-output loop, evidence-id
validation, salience boost, supersession, and retirement are all exercised
end-to-end against the same host adapter the recall tests use.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from kernel.models import OperationExecutionContext, ToolResult
from kernel.seams.host_adapter import build_callable_host
from sdk.packs.memory.operations.memory import consolidate_semantic as op


def _model_def() -> dict[str, Any]:
    return {"model_id": "test-model"}


def _episodic_records() -> list[dict[str, Any]]:
    return [
        {"id": 1, "content": "We chose Postgres for analytics.", "created_at": "2026-07-01T00:00:00Z", "salience": 0.3},
        {"id": 2, "content": "Postgres analytics pipeline ships Friday.", "created_at": "2026-07-02T00:00:00Z", "salience": 0.4},
        {"id": 3, "content": "The analytics team owns the Postgres warehouse.", "created_at": "2026-07-03T00:00:00Z", "salience": 0.2},
    ]


@pytest.mark.asyncio
async def test_consolidate_distills_persists_and_retires() -> None:
    """End-to-end: LLM distillation → memory.add (semantic) → memory.retire × N."""
    tool_calls: list[tuple[str, dict[str, Any]]] = []

    async def complete_fn(messages: list[dict[str, Any]], **_: Any) -> str:
        # Return a valid ConsolidatedSemantic JSON payload.
        return json.dumps({
            "summary": "Postgres was chosen for the analytics warehouse.",
            "confidence": 0.85,
            "item_ids": [1, 2, 3],
        })

    def resolve_model_fn(model_id: str | None) -> dict[str, Any]:
        return _model_def()

    def read_authored_text_fn(path: str) -> str:
        return "Consolidation system prompt (stubbed)."

    async def execute_tool_fn(tool_id: str, ctx: Any) -> ToolResult:
        params = getattr(ctx, "parameters", {}) or {}
        tool_calls.append((tool_id, dict(params)))
        if tool_id == "memory.add":
            # Simulate the "Memory stored with ID N" reply.
            return ToolResult(tool_id=tool_id, output_text="Memory stored with ID 42")
        if tool_id == "memory.retire":
            return ToolResult(tool_id=tool_id, output_text=f"Retired memory {params.get('memory_id')}.")
        return ToolResult(tool_id=tool_id, error=f"unknown tool {tool_id}")

    ctx = OperationExecutionContext(
        operation_id="consolidate_semantic",
        user_slug="adam",
        user_id="adam",
        workspace_id="default",
        params={
            "episodic_records": _episodic_records(),
            "user_slug": "adam",
            "memory_scope": "personal",
            "surface_id": "",
            "workspace_id": "default",
        },
        host=build_callable_host(
            complete_fn=complete_fn,
            resolve_model_fn=resolve_model_fn,
            read_authored_text_fn=read_authored_text_fn,
            execute_tool_fn=execute_tool_fn,
        ),
    )

    result = await op.execute(ctx)

    assert "✅ Consolidated 3 episodic records" in result.output_text
    assert "id=42" in result.output_text
    assert result.extra.get("consolidated") is True
    assert result.extra.get("semantic_id") == 42
    assert result.extra.get("retired_ids") == [1, 2, 3]

    # One memory.add (the semantic record) + three memory.retire calls.
    add_calls = [c for c in tool_calls if c[0] == "memory.add"]
    retire_calls = [c for c in tool_calls if c[0] == "memory.retire"]
    assert len(add_calls) == 1
    assert len(retire_calls) == 3

    # The semantic record must carry the consolidation metadata.
    add_params = add_calls[0][1]
    assert add_params["memory_kind"] == "semantic"
    assert add_params["operation_id"] == "consolidate_semantic"
    assert add_params["supersedes_id"] == 1
    assert add_params["item_ids"] == [1, 2, 3]
    assert add_params["confidence"] == 0.85
    # Salience boosted above the max source (0.4) and floored at 0.8.
    assert add_params["salience"] >= 0.8


@pytest.mark.asyncio
async def test_consolidate_filters_hallucinated_item_ids() -> None:
    """Evidence ids not in the input batch are dropped; falls back to all input ids."""

    async def complete_fn(messages: list[dict[str, Any]], **_: Any) -> str:
        return json.dumps({
            "summary": "Consolidated fact.",
            "confidence": 0.7,
            "item_ids": [1, 999],  # 999 is not in the input batch.
        })

    async def execute_tool_fn(tool_id: str, ctx: Any) -> ToolResult:
        if tool_id == "memory.add":
            return ToolResult(tool_id=tool_id, output_text="Memory stored with ID 7")
        return ToolResult(tool_id=tool_id, output_text="Retired.")

    ctx = OperationExecutionContext(
        operation_id="consolidate_semantic",
        user_slug="adam",
        params={
            "episodic_records": [
                {"id": 1, "content": "fact one", "created_at": "2026-07-01T00:00:00Z"},
                {"id": 2, "content": "fact two", "created_at": "2026-07-02T00:00:00Z"},
            ],
            "user_slug": "adam",
        },
        host=build_callable_host(
            complete_fn=complete_fn,
            resolve_model_fn=lambda _: _model_def(),
            read_authored_text_fn=lambda _: "stub",
            execute_tool_fn=execute_tool_fn,
        ),
    )

    result = await op.execute(ctx)

    # Only id 1 was valid; id 999 was filtered out. With at least one valid
    # id, the fallback does NOT trigger — item_ids is just [1].
    assert result.extra.get("item_ids") == [1]
    assert result.extra.get("retired_ids") == [1]


@pytest.mark.asyncio
async def test_consolidate_falls_back_to_all_input_ids_when_model_returns_none() -> None:
    """When the model returns no valid evidence ids, use the full input set."""

    async def complete_fn(messages: list[dict[str, Any]], **_: Any) -> str:
        return json.dumps({
            "summary": "Consolidated fact.",
            "confidence": 0.5,
            "item_ids": [888, 999],  # none in the input batch.
        })

    async def execute_tool_fn(tool_id: str, ctx: Any) -> ToolResult:
        if tool_id == "memory.add":
            return ToolResult(tool_id=tool_id, output_text="Memory stored with ID 7")
        return ToolResult(tool_id=tool_id, output_text="Retired.")

    ctx = OperationExecutionContext(
        operation_id="consolidate_semantic",
        user_slug="adam",
        params={
            "episodic_records": [
                {"id": 5, "content": "fact one", "created_at": "2026-07-01T00:00:00Z"},
                {"id": 6, "content": "fact two", "created_at": "2026-07-02T00:00:00Z"},
            ],
            "user_slug": "adam",
        },
        host=build_callable_host(
            complete_fn=complete_fn,
            resolve_model_fn=lambda _: _model_def(),
            read_authored_text_fn=lambda _: "stub",
            execute_tool_fn=execute_tool_fn,
        ),
    )

    result = await op.execute(ctx)
    # Fallback: sorted input ids.
    assert result.extra.get("item_ids") == [5, 6]
    assert result.extra.get("retired_ids") == [5, 6]


@pytest.mark.asyncio
async def test_consolidate_rejects_empty_records() -> None:
    """An empty batch is rejected — Pydantic's min_length=1 validator fires first."""
    ctx = OperationExecutionContext(
        operation_id="consolidate_semantic",
        user_slug="adam",
        params={"episodic_records": [], "user_slug": "adam"},
        host=build_callable_host(),
    )
    result = await op.execute(ctx)
    assert "Invalid consolidation parameters" in result.output_text
    assert "episodic_records" in result.output_text
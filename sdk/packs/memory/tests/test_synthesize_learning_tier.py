"""Tests for the synthesize_and_store_learning tier default.

Verifies that the model-called path (no ``memory_tier`` in params) defaults
to ``daily`` — not ``long_term`` — so workflow-state captures expire quickly
rather than polluting long-term recall. The auto-capture path
(``host.turn_hooks._auto_capture``) always supplies an explicit,
salience-graded tier; this default only binds when the model invokes the
operation directly without specifying a tier.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from kernel.models import OperationExecutionContext, ToolResult
from kernel.seams.host_adapter import build_callable_host
from sdk.packs.memory.operations.memory import synthesize_learning as op


def _model_def() -> dict[str, Any]:
    return {"model_id": "test-model"}


def _make_ctx(
    *,
    params: dict[str, Any],
    execute_tool_fn: Any,
) -> OperationExecutionContext:
    async def complete_fn(messages: list[dict[str, Any]], **_: Any) -> str:
        return json.dumps({"facts": ["a distilled fact"]})

    def resolve_model_fn(model_id: str | None) -> dict[str, Any]:
        return _model_def()

    def read_authored_text_fn(path: str) -> str:
        return "Synthesize learning system prompt (stubbed)."

    return OperationExecutionContext(
        operation_id="synthesize_and_store_learning",
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
async def test_model_called_without_tier_defaults_to_daily() -> None:
    """When the model omits ``memory_tier``, the record lands as ``daily``."""
    captured: list[dict[str, Any]] = []

    async def execute_tool_fn(tool_id: str, ctx: Any) -> ToolResult:
        params = getattr(ctx, "parameters", {}) or {}
        if tool_id == "memory.add":
            captured.append(dict(params))
            return ToolResult(tool_id=tool_id, output_text="Memory stored with ID 7")
        return ToolResult(tool_id=tool_id, error=f"unknown tool {tool_id}")

    ctx = _make_ctx(
        params={"content": "some workflow status note", "salience": 0.75},
        execute_tool_fn=execute_tool_fn,
    )

    result = await op.execute(ctx)
    assert "✅ Learning stored" in result.output_text
    assert len(captured) == 1
    assert captured[0]["memory_tier"] == "daily"
    assert captured[0]["salience"] == 0.75
    assert captured[0]["operation_id"] == "save_learning"
    assert captured[0]["confidence"] == 1.0
    assert "content_json" in captured[0]


@pytest.mark.asyncio
async def test_explicit_tier_param_is_preserved() -> None:
    """An explicit ``memory_tier`` in params overrides the default."""
    captured: list[dict[str, Any]] = []

    async def execute_tool_fn(tool_id: str, ctx: Any) -> ToolResult:
        params = getattr(ctx, "parameters", {}) or {}
        if tool_id == "memory.add":
            captured.append(dict(params))
            return ToolResult(tool_id=tool_id, output_text="Memory stored with ID 8")
        return ToolResult(tool_id=tool_id, error=f"unknown tool {tool_id}")

    ctx = _make_ctx(
        params={"content": "a long-lived decision", "memory_tier": "long_term"},
        execute_tool_fn=execute_tool_fn,
    )

    result = await op.execute(ctx)
    assert "✅ Learning stored" in result.output_text
    assert captured[0]["memory_tier"] == "long_term"

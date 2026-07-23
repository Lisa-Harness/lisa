"""Tests for exact-action approval enforcement in the execution registries."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from kernel.approval import (
    ApprovalIdentity,
    ApprovalRequired,
    InMemoryApprovalService,
)
from kernel.models import ToolDefinition, ToolExecutionContext, ToolResult
from kernel.tools.registry import ToolExecutionError, execute_tool


class _Parameters(BaseModel):
    name: str


def _identity() -> ApprovalIdentity:
    return ApprovalIdentity(
        workspace_id="workspace",
        user_id="user",
        user_slug="user",
        surface_id="surface",
        mode_id="coding",
        session_plan_digest="plan-a",
    )


@pytest.mark.asyncio
async def test_approval_is_exact_identity_bound_and_single_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A receipt cannot approve changed parameters or be reused."""

    calls: list[str] = []

    async def execute(ctx: ToolExecutionContext) -> ToolResult:
        calls.append(ctx.parameters["name"])
        return ToolResult(tool_id=ctx.tool_id, output_text="done")

    definition = ToolDefinition(
        tool_id="effectful_tool",
        description="Effectful test tool.",
        pack_id="lisa-pack-test",
        parameters_schema=_Parameters,
        effect="write",
        approval_required=True,
        approval_preview="write the test value",
        implementation_module=SimpleNamespace(execute=execute),
    )
    monkeypatch.setattr("kernel.tools.registry._TOOL_CATALOG", {"effectful_tool": definition})
    service = InMemoryApprovalService()
    context = ToolExecutionContext(
        tool_id="effectful_tool",
        workspace_id="workspace",
        user_id="user",
        user_slug="user",
        surface_id="surface",
        mode_id="coding",
        session_plan_digest="plan-a",
        parameters={"name": "first"},
        approval_service=service,
    )

    with pytest.raises(ApprovalRequired) as required:
        await execute_tool("effectful_tool", context)
    request_id = required.value.invocation.request_id
    assert await service.approve(request_id, _identity()) is not None

    context.approval_request_id = request_id
    assert (await execute_tool("effectful_tool", context)).output_text == "done"
    assert calls == ["first"]

    with pytest.raises(ToolExecutionError, match="no matching usable approval"):
        await execute_tool("effectful_tool", context)

    changed = context.model_copy(
        update={"approval_request_id": request_id, "parameters": {"name": "changed"}}
    )
    with pytest.raises(ToolExecutionError, match="no matching usable approval"):
        await execute_tool("effectful_tool", changed)

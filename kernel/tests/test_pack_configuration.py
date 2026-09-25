"""Focused tests for session-scoped pack configuration at execution boundaries."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from kernel.models import (
    OperationDefinition,
    OperationExecutionContext,
    OperationResult,
    ToolDefinition,
    ToolExecutionContext,
    ToolResult,
)
from kernel.operations.registry import OperationExecutionError, execute_operation
from kernel.pack_configuration import SessionPackConfigurationResolver
from kernel.tools.registry import ToolExecutionError, execute_tool
from pydantic import BaseModel, Field


class _ToolParameters(BaseModel):
    path: str = Field(min_length=1)


class _OperationParameters(BaseModel):
    count: int = Field(ge=1)


@pytest.mark.asyncio
async def test_tool_receives_only_its_owner_pack_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The registry resolves the tool owner config and canonicalizes inputs."""

    captured: dict[str, Any] = {}

    async def execute(ctx: ToolExecutionContext) -> ToolResult:
        captured["config"] = ctx.pack_config
        captured["parameters"] = ctx.parameters
        return ToolResult(tool_id=ctx.tool_id, output_text="ok")

    definition = ToolDefinition(
        tool_id="owner_tool",
        description="Test owner configuration.",
        pack_id="lisa-pack-owner",
        parameters_schema=_ToolParameters,
        mode_allowlist=["coding"],
        implementation_module=SimpleNamespace(execute=execute),
    )
    monkeypatch.setattr("kernel.tools.registry._TOOL_CATALOG", {"owner_tool": definition})
    resolver = SessionPackConfigurationResolver(
        {
            "lisa-pack-owner": {"workspace_root": ".", "checks": [{"id": "unit"}]},
            "lisa-pack-other": {"secret": "not-visible"},
        }
    )

    result = await execute_tool(
        "owner_tool",
        ToolExecutionContext(
            mode_id="coding",
            parameters={"path": "src/main.py"},
            pack_configuration_resolver=resolver,
        ),
    )

    assert result.output_text == "ok"
    assert captured["parameters"] == {"path": "src/main.py"}
    assert captured["config"]["workspace_root"] == "."
    assert "secret" not in captured["config"]
    with pytest.raises(TypeError):
        captured["config"]["workspace_root"] = "elsewhere"


@pytest.mark.asyncio
async def test_registry_rejects_disallowed_or_invalid_tool_invocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime validation applies even when a caller bypasses model schemas."""

    async def execute(_ctx: ToolExecutionContext) -> ToolResult:
        return ToolResult(output_text="unexpected")

    definition = ToolDefinition(
        tool_id="guarded_tool",
        description="Guarded test tool.",
        parameters_schema=_ToolParameters,
        mode_allowlist=["coding"],
        implementation_module=SimpleNamespace(execute=execute),
    )
    monkeypatch.setattr("kernel.tools.registry._TOOL_CATALOG", {"guarded_tool": definition})

    with pytest.raises(ToolExecutionError, match="not allowed"):
        await execute_tool("guarded_tool", ToolExecutionContext(mode_id="default"))
    with pytest.raises(ToolExecutionError, match="invalid parameters"):
        await execute_tool(
            "guarded_tool",
            ToolExecutionContext(mode_id="coding", parameters={"path": ""}),
        )


@pytest.mark.asyncio
async def test_operation_receives_its_owner_pack_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operation execution follows the same target-scoped contract as tools."""

    captured: dict[str, Any] = {}

    async def execute(ctx: OperationExecutionContext) -> OperationResult:
        captured["config"] = ctx.pack_config
        captured["parameters"] = ctx.params
        return OperationResult(output_text="ok")

    definition = OperationDefinition(
        operation_id="owner_operation",
        description="Test owner operation.",
        pack_id="lisa-pack-owner",
        parameters_schema=_OperationParameters,
        mode_allowlist=["coding"],
        implementation_module=SimpleNamespace(execute=execute),
    )
    monkeypatch.setattr(
        "kernel.operations.registry._OPERATION_CATALOG",
        {"owner_operation": definition},
    )
    resolver = SessionPackConfigurationResolver({"lisa-pack-owner": {"limit": 3}})

    result = await execute_operation(
        "owner_operation",
        OperationExecutionContext(
            mode_id="coding",
            params={"count": 2},
            pack_configuration_resolver=resolver,
        ),
    )

    assert result.output_text == "ok"
    assert captured == {"config": {"limit": 3}, "parameters": {"count": 2}}

    with pytest.raises(OperationExecutionError, match="invalid parameters"):
        await execute_operation(
            "owner_operation",
            OperationExecutionContext(
                mode_id="coding",
                params={"count": 0},
                pack_configuration_resolver=resolver,
            ),
        )

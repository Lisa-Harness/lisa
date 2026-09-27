"""Focused tests for session-scoped pack configuration at execution boundaries."""

from __future__ import annotations

from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest
from kernel.capability_binding import CapabilityBinding, CapabilityRef
from kernel.models import (
    OperationDefinition,
    OperationExecutionContext,
    OperationResult,
    ToolDefinition,
    ToolExecutionContext,
    ToolResult,
)
from kernel.operations.registry import OperationExecutionError, execute_operation
from kernel.tools.registry import ToolExecutionError, execute_tool
from pydantic import BaseModel, Field


class _ToolParameters(BaseModel):
    path: str = Field(min_length=1)


class _OperationParameters(BaseModel):
    count: int = Field(ge=1)


class _Bindings:
    def __init__(self, configurations: dict[str, Any]) -> None:
        self.configurations = configurations

    def binding_for(self, capability: CapabilityRef) -> CapabilityBinding:
        return CapabilityBinding(
            owner_id=capability.owner_id,
            configuration=self.configurations.get(capability.owner_id, {}),
        )


class _MismatchedBinding:
    def binding_for(self, capability: CapabilityRef) -> CapabilityBinding:
        return CapabilityBinding(owner_id="lisa-pack-other", configuration={})


@pytest.mark.asyncio
async def test_tool_receives_only_its_owner_pack_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The registry resolves the tool owner config and canonicalizes inputs."""

    captured: dict[str, Any] = {}

    async def execute(ctx: ToolExecutionContext) -> ToolResult:
        captured["config"] = ctx.pack_config
        captured["parameters"] = ctx.parameters
        captured["provider"] = ctx.capability_binding_provider
        return ToolResult(tool_id=ctx.tool_id, output_text="ok")

    definition = ToolDefinition(
        tool_id="owner_tool",
        description="Test owner configuration.",
        owner_id="lisa-pack-owner",
        pack_id="lisa-pack-owner",
        parameters_schema=_ToolParameters,
        mode_allowlist=["coding"],
        implementation_module=SimpleNamespace(execute=execute),
    )
    monkeypatch.setattr("kernel.tools.registry._TOOL_CATALOG", {"owner_tool": definition})
    configs = {
        "lisa-pack-owner": MappingProxyType({"workspace_root": ".", "checks": ({"id": "unit"},)}),
        "lisa-pack-other": MappingProxyType({"secret": "not-visible"}),
    }

    resolver = _Bindings(configs)

    result = await execute_tool(
        "owner_tool",
        ToolExecutionContext(
            mode_id="coding",
            parameters={"path": "src/main.py"},
            capability_binding_provider=resolver,
        ),
    )

    assert result.output_text == "ok"
    assert captured["parameters"] == {"path": "src/main.py"}
    assert captured["config"]["workspace_root"] == "."
    assert "secret" not in captured["config"]
    assert captured["provider"] is None
    with pytest.raises(TypeError):
        captured["config"]["workspace_root"] = "elsewhere"


@pytest.mark.asyncio
async def test_registry_rejects_disallowed_or_invalid_tool_invocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime validation applies even when a caller bypasses model schemas."""


@pytest.mark.asyncio
async def test_registry_rejects_binding_for_wrong_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    async def execute(_ctx: ToolExecutionContext) -> ToolResult:
        nonlocal called
        called = True
        return ToolResult(output_text="unexpected")

    definition = ToolDefinition(
        tool_id="owner_tool",
        description="Owner-bound tool.",
        owner_id="lisa-pack-owner",
        mode_allowlist=["coding"],
        implementation_module=SimpleNamespace(execute=execute),
    )
    monkeypatch.setattr("kernel.tools.registry._TOOL_CATALOG", {"owner_tool": definition})

    with pytest.raises(ToolExecutionError, match="does not match"):
        await execute_tool(
            "owner_tool",
            ToolExecutionContext(
                mode_id="coding",
                capability_binding_provider=_MismatchedBinding(),
            ),
        )
    assert not called

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
        captured["provider"] = ctx.capability_binding_provider
        return OperationResult(output_text="ok")

    definition = OperationDefinition(
        operation_id="owner_operation",
        description="Test owner operation.",
        owner_id="lisa-pack-owner",
        pack_id="lisa-pack-owner",
        parameters_schema=_OperationParameters,
        mode_allowlist=["coding"],
        implementation_module=SimpleNamespace(execute=execute),
    )
    monkeypatch.setattr(
        "kernel.operations.registry._OPERATION_CATALOG",
        {"owner_operation": definition},
    )

    resolver = _Bindings({"lisa-pack-owner": {"limit": 3}})

    result = await execute_operation(
        "owner_operation",
        OperationExecutionContext(
            mode_id="coding",
            params={"count": 2},
            capability_binding_provider=resolver,
        ),
    )

    assert result.output_text == "ok"
    assert captured == {
        "config": {"limit": 3},
        "parameters": {"count": 2},
        "provider": None,
    }

    with pytest.raises(OperationExecutionError, match="invalid parameters"):
        await execute_operation(
            "owner_operation",
            OperationExecutionContext(
                mode_id="coding",
                params={"count": 0},
                capability_binding_provider=resolver,
            ),
        )

"""Required preparation fails closed without changing enrichment defaults."""

from unittest.mock import AsyncMock

import pytest
from kernel.models import (
    OperationExecutionContext,
    OperationResult,
    ToolResult,
    WorkflowModeConfig,
)

from kernel import ipsar_fsm


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["abort", "skip"])
async def test_pre_turn_failure_policy(monkeypatch: pytest.MonkeyPatch, policy: str) -> None:
    """Only an explicit abort contract propagates a preparation failure."""
    mode = WorkflowModeConfig.model_validate(
        {
            "mode_id": "test",
            "pre_turn_operation_ids": ["prepare"],
            "pre_turn_failure_policy": policy,
        }
    )
    monkeypatch.setattr(ipsar_fsm, "get_workflow_catalog", lambda: {"test": mode})
    execute = AsyncMock(side_effect=RuntimeError("preparation failed"))
    monkeypatch.setattr(ipsar_fsm, "_execute_mode_target_ids", execute)
    if policy == "abort":
        with pytest.raises(RuntimeError, match="preparation failed"):
            await ipsar_fsm.run_mode_pre_turn_operations("test", OperationExecutionContext())
    else:
        await ipsar_fsm.run_mode_pre_turn_operations("test", OperationExecutionContext())
    assert execute.call_args.kwargs["failure_policy"] == policy


def _mode_with_pre_turn(target_ids: list[str], policy: str) -> WorkflowModeConfig:
    return WorkflowModeConfig.model_validate(
        {
            "mode_id": "test",
            "pre_turn_operation_ids": target_ids,
            "pre_turn_failure_policy": policy,
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["abort", "skip"])
async def test_unknown_pre_turn_target_policy(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, policy: str
) -> None:
    """A misconfigured target id raises under abort and warns under skip."""
    mode = _mode_with_pre_turn(["no-such-target"], policy)
    monkeypatch.setattr(ipsar_fsm, "get_workflow_catalog", lambda: {"test": mode})
    ctx = OperationExecutionContext()
    if policy == "abort":
        with pytest.raises(RuntimeError, match="no-such-target"):
            await ipsar_fsm.run_mode_pre_turn_operations("test", ctx)
    else:
        result = await ipsar_fsm.run_mode_pre_turn_operations("test", ctx)
        assert isinstance(result, OperationResult)
        assert "no-such-target" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["abort", "skip"])
async def test_pre_turn_tool_error_policy(monkeypatch: pytest.MonkeyPatch, policy: str) -> None:
    """A failed prerequisite tool raises under abort and stays isolated under skip."""
    mode = _mode_with_pre_turn(["flaky-tool"], policy)
    monkeypatch.setattr(ipsar_fsm, "get_workflow_catalog", lambda: {"test": mode})
    monkeypatch.setattr(
        "kernel.tools.registry.get_tool_catalog",
        lambda: {"flaky-tool": object()},
    )
    monkeypatch.setattr(
        "kernel.tools.registry.execute_tool",
        AsyncMock(return_value=ToolResult(tool_id="flaky-tool", error="boom")),
    )
    ctx = OperationExecutionContext()
    if policy == "abort":
        with pytest.raises(RuntimeError, match="boom"):
            await ipsar_fsm.run_mode_pre_turn_operations("test", ctx)
    else:
        result = await ipsar_fsm.run_mode_pre_turn_operations("test", ctx)
        assert isinstance(result, OperationResult)
        assert result.extra == {"tool_error": "boom"}


@pytest.mark.asyncio
async def test_default_policy_keeps_enrichment_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Modes without an explicit abort policy keep swallow-and-continue."""
    mode = WorkflowModeConfig.model_validate(
        {
            "mode_id": "test",
            "pre_turn_operation_ids": ["no-such-target", "flaky-tool"],
        }
    )
    assert mode.pre_turn_failure_policy == "skip"
    monkeypatch.setattr(ipsar_fsm, "get_workflow_catalog", lambda: {"test": mode})
    monkeypatch.setattr(
        "kernel.tools.registry.get_tool_catalog",
        lambda: {"flaky-tool": object()},
    )
    monkeypatch.setattr(
        "kernel.tools.registry.execute_tool",
        AsyncMock(return_value=ToolResult(tool_id="flaky-tool", error="boom")),
    )
    result = await ipsar_fsm.run_mode_pre_turn_operations("test", OperationExecutionContext())
    assert isinstance(result, OperationResult)
    assert result.extra == {"tool_error": "boom"}

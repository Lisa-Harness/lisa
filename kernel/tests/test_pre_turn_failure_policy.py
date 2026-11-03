"""Required preparation fails closed without changing enrichment defaults."""

from unittest.mock import AsyncMock

import pytest
from kernel.models import OperationExecutionContext, WorkflowModeConfig

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

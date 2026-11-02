"""Runtime operation composition retains session-owned child configuration."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from kernel.models import OperationDefinition
from kernel.operations.registry import _wrap_declarative_tool_executor


@pytest.mark.asyncio
async def test_runtime_child_receives_captured_binding() -> None:
    """Runtime calls retain configuration even after the parent context clears it."""
    provider = object()
    observed = []

    async def dispatch(tool_id: str, ctx: SimpleNamespace) -> str:
        observed.append((tool_id, ctx.capability_binding_provider))
        return "result"

    definition = OperationDefinition(
        operation_id="runtime", description="test", execution_kind="runtime"
    )
    execute = _wrap_declarative_tool_executor(definition, dispatch, provider)
    ctx = SimpleNamespace(capability_binding_provider=None)
    assert await execute("child", ctx) == "result"
    assert observed == [("child", provider)]
    assert ctx.capability_binding_provider is None


@pytest.mark.asyncio
async def test_runtime_child_cleanup_after_failure() -> None:
    """A failed nested dispatch does not leave the provider on its context."""
    definition = OperationDefinition(
        operation_id="runtime", description="test", execution_kind="runtime"
    )
    execute = _wrap_declarative_tool_executor(
        definition, AsyncMock(side_effect=ValueError("failure")), object()
    )
    ctx = SimpleNamespace(capability_binding_provider=None)
    with pytest.raises(ValueError, match="failure"):
        await execute("child", ctx)
    assert ctx.capability_binding_provider is None

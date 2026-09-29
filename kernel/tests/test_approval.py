"""Tests for exact-action approval enforcement in the execution registries."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from kernel.approval import (
    ApprovalIdentity,
    ApprovalInvocation,
    ApprovalRequired,
    InMemoryApprovalService,
    build_approval_invocation,
)
from kernel.capability_binding import CapabilityBinding
from kernel.models import (
    OperationDefinition,
    OperationExecutionContext,
    OperationResult,
    ToolDefinition,
    ToolExecutionContext,
    ToolResult,
)
from kernel.operations.registry import execute_operation
from kernel.tools.registry import ToolExecutionError, execute_tool
from kernel.turn_phases._post_turn import _to_tool_context
from pydantic import BaseModel


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


def test_nested_tool_context_keeps_task_identity() -> None:
    operation = OperationExecutionContext(task_id="task-7")

    assert _to_tool_context("coding_create_worktree", operation).task_id == "task-7"


@pytest.mark.asyncio
async def test_nested_tool_uses_its_own_owner_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def parent(ctx: ToolExecutionContext) -> ToolResult:
        assert ctx.pack_config["parent_secret"] == "only parent"
        floor = ctx.execution_floor_for("sandbox")
        assert floor is not None and floor.execution == "sandbox"
        assert not hasattr(floor, "child_secret")
        assert ctx.delegated_tool_dispatch is not None
        return await ctx.delegated_tool_dispatch("sandbox", {"argv": ["true"]})

    async def sandbox(ctx: ToolExecutionContext) -> ToolResult:
        assert ctx.pack_config["child_secret"] == "only child"
        assert "parent_secret" not in ctx.pack_config
        assert ctx.parameters == {"argv": ["true"]}
        return ToolResult(output_text="contained")

    monkeypatch.setattr(
        "kernel.tools.registry._TOOL_CATALOG",
        {
            "parent": ToolDefinition(
                tool_id="parent",
                owner_id="parent-owner",
                implementation_module=SimpleNamespace(execute=parent),
            ),
            "sandbox": ToolDefinition(
                tool_id="sandbox",
                owner_id="child-owner",
                implementation_module=SimpleNamespace(execute=sandbox),
            ),
        },
    )
    bindings = {
        "parent-owner": CapabilityBinding("parent-owner", {"parent_secret": "only parent"}),
        "child-owner": CapabilityBinding(
            "child-owner",
            {"child_secret": "only child", "sandbox": {"floor": {"execution": "sandbox"}}},
        ),
    }
    context = ToolExecutionContext(
        tool_id="parent",
        capability_binding_provider=SimpleNamespace(
            binding_for=lambda ref: bindings.get(ref.owner_id)
        ),
    )

    result = await execute_tool("parent", context)

    assert result.output_text == "contained"


@pytest.mark.asyncio
async def test_operation_sees_child_floor_without_private_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def prewarm(ctx: OperationExecutionContext) -> OperationResult:
        assert ctx.pack_config["parent_secret"] == "only parent"
        assert ctx.execution_floor_for is not None
        floor = ctx.execution_floor_for("sandbox")
        assert floor is not None and floor.persistence == "direct-write"
        assert floor.paths == {"/workspace": "rw"}
        assert not hasattr(floor, "child_secret")
        return OperationResult(output_text="primed")

    monkeypatch.setattr(
        "kernel.operations.registry._OPERATION_CATALOG",
        {
            "prewarm": OperationDefinition(
                operation_id="prewarm",
                owner_id="parent-owner",
                implementation_module=SimpleNamespace(execute=prewarm),
            ),
        },
    )
    monkeypatch.setattr(
        "kernel.tools.registry._TOOL_CATALOG",
        {"sandbox": ToolDefinition(tool_id="sandbox", owner_id="child-owner")},
    )
    bindings = {
        "parent-owner": CapabilityBinding("parent-owner", {"parent_secret": "only parent"}),
        "child-owner": CapabilityBinding(
            "child-owner",
            {
                "child_secret": "only child",
                "sandbox": {
                    "floor": {
                        "execution": "sandbox",
                        "persistence": "direct-write",
                        "paths": {"/workspace": "rw"},
                    }
                },
            },
        ),
    }
    result = await execute_operation(
        "prewarm",
        OperationExecutionContext(
            capability_binding_provider=SimpleNamespace(
                binding_for=lambda ref: bindings.get(ref.owner_id)
            ),
        ),
    )
    assert result.output_text == "primed"


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


def _invocation(
    *,
    parameters: dict[str, str] | None = None,
    identity: ApprovalIdentity | None = None,
    target_id: str = "effectful_tool",
) -> ApprovalInvocation:
    return build_approval_invocation(
        target_kind="tool",
        target_id=target_id,
        pack_id="lisa-pack-test",
        effect="write",
        parameters=parameters or {"name": "first"},
        identity=identity or _identity(),
        pack_config={"workspace_root": "."},
        safe_preview="write the test value",
    )


@pytest.mark.asyncio
async def test_request_is_idempotent_on_action_digest_while_pending() -> None:
    """Re-issuing the same action before approval reuses the pending request_id.

    Reproduces the three-mint approval loop: the model re-issued
    ``coding_create_worktree`` on every "proceed" turn and the service minted
    a fresh ``request_id`` each time, stranding the original pending request
    and forcing the user to approve a different id on every retry.
    """

    service = InMemoryApprovalService()
    first = await service.request(_invocation())
    second = await service.request(_invocation())
    third = await service.request(_invocation())

    assert second.request_id == first.request_id
    assert third.request_id == first.request_id
    pending = await service.list_pending(_identity())
    assert len(pending) == 1
    assert pending[0].invocation.request_id == first.request_id


@pytest.mark.asyncio
async def test_request_mints_new_id_after_consume() -> None:
    """A genuinely new request after consumption mints a new request_id.

    Only the ``pending`` state is idempotent. Once approved and consumed, the
    same logical action is a new approval lifecycle and gets a new id.
    """

    service = InMemoryApprovalService()
    invocation = _invocation()
    first = await service.request(invocation)
    assert await service.approve(first.request_id, _identity()) is not None
    assert await service.consume(first.request_id, invocation) is not None

    second = await service.request(_invocation())
    assert second.request_id != first.request_id


@pytest.mark.asyncio
async def test_request_distinguishes_different_actions() -> None:
    """Different parameters (different action_digest) mint distinct ids."""

    service = InMemoryApprovalService()
    a = await service.request(_invocation(parameters={"name": "a"}))
    b = await service.request(_invocation(parameters={"name": "b"}))
    assert a.request_id != b.request_id
    assert len(await service.list_pending(_identity())) == 2


@pytest.mark.asyncio
async def test_request_distinguishes_different_identities() -> None:
    """Same action under a different owner identity is a separate request."""

    service = InMemoryApprovalService()
    other = ApprovalIdentity(
        workspace_id="workspace",
        user_id="other",
        user_slug="other",
        surface_id="surface",
        mode_id="coding",
        session_plan_digest="plan-a",
    )
    a = await service.request(_invocation())
    b = await service.request(_invocation(identity=other))
    assert a.request_id != b.request_id
    assert len(await service.list_pending(_identity())) == 1
    assert len(await service.list_pending(other)) == 1


@pytest.mark.asyncio
async def test_request_reuses_pending_after_one_is_denied_and_reissued() -> None:
    """A denied request is not reused; a fresh request mints a new id.

    Denial is terminal for that request_id. Re-issuing the same action starts
    a new pending lifecycle rather than resurrecting the denied record.
    """

    service = InMemoryApprovalService()
    first = await service.request(_invocation())
    assert await service.deny(first.request_id, _identity()) is True

    second = await service.request(_invocation())
    assert second.request_id != first.request_id
    assert len(await service.list_pending(_identity())) == 1

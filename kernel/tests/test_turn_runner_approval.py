"""Tests that approval interruptions pause, rather than fail, model turns."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from kernel.approval import (
    ApprovalIdentity,
    ApprovalRequired,
    InMemoryApprovalService,
    build_approval_invocation,
)
from kernel.ipsar_fsm import (
    run_mode_entry_operations,
    run_mode_post_turn_operations,
    run_mode_pre_turn_operations,
)
from kernel.models import (
    OperationDefinition,
    OperationExecutionContext,
    OperationResult,
    ToolDefinition,
    ToolExecutionContext,
    ToolResult,
    WorkflowModeConfig,
)
from kernel.operations.declarative import execute_declarative_operation
from kernel.seams.host_adapter import build_callable_host
from kernel.tools.registry import execute_tool
from kernel.turn_hooks import (
    HostServiceHandles,
    ModelHooks,
    TaskCompletionEvidence,
    TurnIdentity,
)
from kernel.turn_runner import TurnHooks, execute_approved_invocation, run_turn
from pydantic import BaseModel


class _Parameters(BaseModel):
    path: str


@dataclass(frozen=True)
class _Completion:
    content: str = ""
    tool_calls: list[dict[str, Any]] | None = None


@pytest.mark.asyncio
async def test_run_turn_returns_pending_approval_for_effectful_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model turn stops before the side effect and exposes one request ID."""

    executed = False
    started: list[Any] = []

    async def begin(record: Any) -> Any:
        started.append(record)
        return record

    async def settle(_work_item_id: str, _identity: Any, **_kwargs: Any) -> None:
        return None

    async def mark_running_indeterminate(_identity: Any) -> int:
        return 0

    async def execute(_ctx: ToolExecutionContext) -> ToolResult:
        nonlocal executed
        executed = True
        return ToolResult(output_text="unexpected")

    definition = ToolDefinition(
        tool_id="write_file",
        description="Write a file.",
        pack_id="lisa-pack-coding",
        parameters_schema=_Parameters,
        effect="write",
        approval_required=True,
        approval_preview="apply a file change",
        implementation_module=SimpleNamespace(execute=execute),
    )
    monkeypatch.setattr("kernel.tools.registry._TOOL_CATALOG", {"write_file": definition})

    async def complete(**_kwargs: Any) -> _Completion:
        return _Completion(
            tool_calls=[
                {
                    "id": "call-1",
                    "function": {"name": "write_file", "arguments": '{"path":"src/a.py"}'},
                }
            ]
        )

    result = await run_turn(
        mode_id="default",
        query="change the file",
        hooks=TurnHooks(
            models=ModelHooks(
                complete=complete,
            ),
            services=HostServiceHandles(
                approval_service=InMemoryApprovalService(),
                recovery_service=SimpleNamespace(
                    begin=begin,
                    settle=settle,
                    mark_running_indeterminate=mark_running_indeterminate,
                ),
            ),
            identity=TurnIdentity(
                workspace_id="workspace",
                user_id="user",
                user_slug="user",
                surface_id="surface",
                session_plan_digest="plan-a",
            ),
        ),
    )

    assert result.pending_approval is not None
    assert result.pending_approval.target_id == "write_file"
    assert result.pending_approval_tool_call_id == "call-1"
    assert len(started) == 1
    assert result.pending_approval_recovery_item_id == started[0].work_item_id
    assert "/approve " in result.response_text
    assert executed is False


@pytest.mark.asyncio
async def test_approval_preserves_earlier_effect_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def read_file(_ctx: ToolExecutionContext) -> ToolResult:
        return ToolResult(output_text="file contents")

    async def write_file(_ctx: ToolExecutionContext) -> ToolResult:
        raise AssertionError("write ran before approval")

    monkeypatch.setattr(
        "kernel.tools.registry._TOOL_CATALOG",
        {
            "read_file": ToolDefinition(
                tool_id="read_file",
                pack_id="lisa-pack-coding",
                effect="read",
                implementation_module=SimpleNamespace(execute=read_file),
            ),
            "write_file": ToolDefinition(
                tool_id="write_file",
                pack_id="lisa-pack-coding",
                parameters_schema=_Parameters,
                effect="write",
                approval_required=True,
                implementation_module=SimpleNamespace(execute=write_file),
            ),
        },
    )

    async def complete(**_kwargs: Any) -> _Completion:
        return _Completion(
            tool_calls=[
                {"id": "call-1", "function": {"name": "read_file", "arguments": "{}"}},
                {
                    "id": "call-2",
                    "function": {"name": "write_file", "arguments": '{"path":"src/a.py"}'},
                },
            ]
        )

    result = await run_turn(
        mode_id="default",
        query="change the file",
        hooks=TurnHooks(
            models=ModelHooks(complete=complete),
            services=HostServiceHandles(approval_service=InMemoryApprovalService()),
            identity=TurnIdentity(
                workspace_id="workspace",
                user_id="user",
                user_slug="user",
                surface_id="surface",
                session_plan_digest="plan-a",
            ),
        ),
    )

    assert result.pending_approval_tool_call_id == "call-2"
    assert [(receipt.capability_id, receipt.effect) for receipt in result.effect_receipts] == [
        ("read_file", "read")
    ]
    assert result.transcript[-1]["tool_call_id"] == "call-1"


@pytest.mark.asyncio
async def test_run_turn_pauses_when_operation_calls_effectful_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nested tool approval pauses the turn instead of becoming an operation error."""

    executed = False

    async def write_file(_ctx: ToolExecutionContext) -> ToolResult:
        nonlocal executed
        executed = True
        return ToolResult(tool_id="write_file", output_text="unexpected")

    async def nested_operation(ctx: OperationExecutionContext) -> OperationResult:
        assert ctx.host is not None
        await ctx.host.execute_tool(
            "write_file",
            ToolExecutionContext(
                tool_id="write_file",
                workspace_id=ctx.workspace_id,
                user_id=ctx.user_id,
                user_slug=ctx.user_slug,
                surface_type=ctx.surface_type,
                surface_id=ctx.surface_id,
                memory_scope=ctx.memory_scope,
                mode_id=ctx.mode_id,
                parameters={"path": "src/a.py"},
                capability_binding_provider=ctx.capability_binding_provider,
                approval_service=ctx.approval_service,
                session_plan_digest=ctx.session_plan_digest,
            ),
        )
        return OperationResult(output_text="unexpected")

    tool_definition = ToolDefinition(
        tool_id="write_file",
        description="Write a file.",
        pack_id="lisa-pack-coding",
        parameters_schema=_Parameters,
        effect="write",
        approval_required=True,
        implementation_module=SimpleNamespace(execute=write_file),
    )
    operation_definition = OperationDefinition(
        operation_id="nested_operation",
        description="Call a nested effectful tool.",
        mode_allowlist=["default"],
        implementation_module=SimpleNamespace(execute=nested_operation),
    )
    monkeypatch.setattr("kernel.tools.registry._TOOL_CATALOG", {"write_file": tool_definition})
    monkeypatch.setattr(
        "kernel.operations.registry._OPERATION_CATALOG",
        {"nested_operation": operation_definition},
    )

    call_count = 0

    async def complete(**_kwargs: Any) -> _Completion:
        nonlocal call_count
        call_count += 1
        return _Completion(
            tool_calls=[
                {
                    "id": "call-1",
                    "function": {"name": "nested_operation", "arguments": "{}"},
                }
            ]
        )

    result = await run_turn(
        mode_id="default",
        query="change the file",
        hooks=TurnHooks(
            models=ModelHooks(
                complete=complete,
            ),
            services=HostServiceHandles(
                approval_service=InMemoryApprovalService(),
            ),
            identity=TurnIdentity(
                workspace_id="workspace",
                user_id="user",
                user_slug="user",
                surface_id="surface",
                session_plan_digest="plan-a",
            ),
        ),
    )

    assert result.pending_approval is not None
    assert result.pending_approval.target_id == "write_file"
    assert executed is False
    assert call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_policy", ["abort", "skip"])
async def test_lifecycle_entry_tool_rethrows_approval_for_each_failure_policy(
    monkeypatch: pytest.MonkeyPatch,
    failure_policy: str,
) -> None:
    """Entry target approval pauses rather than follows lifecycle failure policy."""

    tool_definition = ToolDefinition(
        tool_id="write_file",
        description="Write a file.",
        pack_id="lisa-pack-coding",
        effect="write",
        approval_required=True,
        implementation_module=SimpleNamespace(execute=lambda _ctx: None),
        mode_allowlist=["approval_action"],
    )
    monkeypatch.setattr("kernel.tools.registry._TOOL_CATALOG", {"write_file": tool_definition})
    monkeypatch.setattr(
        "kernel.ipsar_fsm.get_workflow_catalog",
        lambda: {
            "approval_action": SimpleNamespace(
                entry_operation_ids=["write_file"],
                entry_failure_policy=failure_policy,
            )
        },
    )
    service = InMemoryApprovalService()
    context = OperationExecutionContext(
        workspace_id="workspace",
        user_id="user",
        user_slug="user",
        surface_id="surface",
        mode_id="approval_action",
        session_plan_digest="plan-a",
        approval_service=service,
    )

    with pytest.raises(ApprovalRequired) as required:
        await run_mode_entry_operations("approval_action", context)

    assert required.value.invocation.target_id == "write_file"
    assert required.value.invocation.identity.workspace_id == "workspace"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target_field", "runner"),
    [
        ("pre_turn_operation_ids", run_mode_pre_turn_operations),
        ("post_turn_operation_ids", run_mode_post_turn_operations),
    ],
)
async def test_pre_and_post_lifecycle_tools_rethrow_approval(
    monkeypatch: pytest.MonkeyPatch,
    target_field: str,
    runner: Any,
) -> None:
    """Isolation policy does not convert a required approval into a skipped failure."""

    tool_definition = ToolDefinition(
        tool_id="write_file",
        description="Write a file.",
        pack_id="lisa-pack-coding",
        effect="write",
        approval_required=True,
        implementation_module=SimpleNamespace(execute=lambda _ctx: None),
        mode_allowlist=["approval_chat"],
    )
    monkeypatch.setattr("kernel.tools.registry._TOOL_CATALOG", {"write_file": tool_definition})
    monkeypatch.setattr(
        "kernel.ipsar_fsm.get_workflow_catalog",
        lambda: {"approval_chat": SimpleNamespace(**{target_field: ["write_file"]})},
    )
    context = OperationExecutionContext(
        workspace_id="workspace",
        user_id="user",
        user_slug="user",
        surface_id="surface",
        mode_id="approval_chat",
        session_plan_digest="plan-a",
        approval_service=InMemoryApprovalService(),
    )

    with pytest.raises(ApprovalRequired) as required:
        await runner("approval_chat", context)

    assert required.value.invocation.target_id == "write_file"


@pytest.mark.asyncio
async def test_declarative_operation_rethrows_nested_approval() -> None:
    """Prompt operations preserve approval interruptions from their declared tools."""

    service = InMemoryApprovalService()
    pending = build_approval_invocation(
        target_kind="tool",
        target_id="write_file",
        pack_id="lisa-pack-coding",
        effect="write",
        parameters={"path": "src/a.py"},
        identity=ApprovalIdentity(
            workspace_id="workspace",
            user_id="user",
            user_slug="user",
            surface_id="surface",
            mode_id="coding",
            session_plan_digest="plan-a",
        ),
        pack_config={},
    )
    observed: list[ToolExecutionContext] = []

    async def complete(**_kwargs: Any) -> _Completion:
        return _Completion(
            tool_calls=[
                {
                    "id": "call-1",
                    "function": {"name": "write_file", "arguments": '{"path":"src/a.py"}'},
                }
            ]
        )

    async def execute_nested_tool(_tool_id: str, tool_ctx: ToolExecutionContext) -> ToolResult:
        observed.append(tool_ctx)
        raise ApprovalRequired(pending)

    context = OperationExecutionContext(
        workspace_id="workspace",
        user_id="user",
        user_slug="user",
        surface_id="surface",
        mode_id="coding",
        session_plan_digest="plan-a",
        approval_service=service,
        host=build_callable_host(
            complete_fn=complete,
            resolve_model_fn=lambda _model_id: object(),
            execute_tool_fn=execute_nested_tool,
        ),
    )

    with pytest.raises(ApprovalRequired) as required:
        await execute_declarative_operation(
            context,
            operation_id="declared_operation",
            instruction_markdown_path="operation.md",
            instruction_body="Use the declared tool.",
            max_calls_per_turn=1,
            response_format_model=None,
            response_output_text_field="output_text",
            response_output_json_passthrough={},
            completion_max_tokens=100,
            completion_temperature=0.0,
        )

    assert required.value.invocation.request_id == pending.request_id
    assert observed[0].approval_service is service
    assert observed[0].session_plan_digest == "plan-a"


@pytest.mark.asyncio
async def test_execute_approved_invocation_consumes_the_real_registry_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Approval resume rebuilds the invocation and consumes its one-time receipt."""

    calls: list[str] = []

    async def write_file(ctx: ToolExecutionContext) -> ToolResult:
        calls.append(ctx.parameters["path"])
        return ToolResult(tool_id="write_file", output_text="written")

    definition = ToolDefinition(
        tool_id="write_file",
        description="Write a file.",
        pack_id="lisa-pack-coding",
        parameters_schema=_Parameters,
        effect="write",
        approval_required=True,
        implementation_module=SimpleNamespace(execute=write_file),
    )
    monkeypatch.setattr("kernel.tools.registry._TOOL_CATALOG", {"write_file": definition})
    service = InMemoryApprovalService()
    initial_context = ToolExecutionContext(
        tool_id="write_file",
        workspace_id="workspace",
        user_id="user",
        user_slug="user",
        surface_id="surface",
        mode_id="coding",
        session_plan_digest="plan-a",
        parameters={"path": "src/a.py"},
        approval_service=service,
    )

    with pytest.raises(ApprovalRequired) as required:
        await execute_tool("write_file", initial_context)
    invocation = required.value.invocation
    assert await service.approve(invocation.request_id, invocation.identity) is not None

    result = await execute_approved_invocation(
        invocation=invocation,
        hooks=TurnHooks(
            services=HostServiceHandles(
                approval_service=service,
            ),
            identity=TurnIdentity(
                workspace_id="workspace",
                user_id="user",
                user_slug="user",
                surface_id="surface",
                session_plan_digest="plan-a",
            ),
        ),
    )

    assert result.output_text == "written"
    assert calls == ["src/a.py"]
    record = await service.get(invocation.request_id)
    assert record is not None
    assert record.state == "consumed"


@pytest.mark.asyncio
async def test_checked_commit_runner_uses_host_evidence_before_finishing() -> None:
    catalog = {
        "executor": WorkflowModeConfig(
            mode_id="executor",
            start_stage_id="none",
            mode_kind="chat",
            task_completion="checked_commit",
            task_artifact_tool_id="create_worktree",
        )
    }

    async def complete(**_kwargs: Any) -> _Completion:
        return _Completion(content="Implementation complete.")

    verified = False

    async def verify(request: Any) -> TaskCompletionEvidence:
        assert request.task_id == "task-one"
        assert request.origin_tool_id == "create_worktree"
        return TaskCompletionEvidence(
            verified=verified,
            reason="No current checked commit for this task.",
        )

    hooks = TurnHooks(
        models=ModelHooks(complete=complete),
        services=HostServiceHandles(verify_task_completion=verify),
        identity=TurnIdentity(task_id="task-one"),
    )
    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=catalog),
        patch("kernel.turn_runner.get_tool_catalog", return_value={}),
    ):
        blocked = await run_turn(mode_id="executor", query="implement it", hooks=hooks)
        verified = True
        done = await run_turn(mode_id="executor", query="continue", hooks=hooks)

    assert blocked.blocked is True
    assert "No current checked commit" in blocked.response_text
    assert done.blocked is False
    assert "Implementation complete" in done.response_text


@pytest.mark.asyncio
async def test_checked_commit_backstop_reminds_once_without_fabricating_work() -> None:
    catalog = {
        "executor": WorkflowModeConfig(
            mode_id="executor",
            start_stage_id="none",
            mode_kind="chat",
            task_completion="checked_commit",
            task_artifact_tool_id="create_worktree",
            authored_allowed_tool_ids=["create_worktree"],
            allowed_tool_ids=["create_worktree"],
        )
    }
    tool_catalog = {
        "create_worktree": ToolDefinition(
            tool_id="create_worktree",
            effect="write",
            parameters_schema=_Parameters,
        )
    }
    model_requests: list[str] = []

    async def complete(**kwargs: Any) -> _Completion:
        model_requests.append(str(kwargs))
        return _Completion(content="I will make the change.")

    async def verify(_request: Any) -> TaskCompletionEvidence:
        return TaskCompletionEvidence(verified=False, reason="No current checked commit.")

    hooks = TurnHooks(
        models=ModelHooks(complete=complete),
        services=HostServiceHandles(verify_task_completion=verify),
        identity=TurnIdentity(task_id="task-one"),
    )
    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=catalog),
        patch("kernel.turn_runner.get_tool_catalog", return_value=tool_catalog),
        patch("kernel.tools.registry.get_tool_catalog", return_value=tool_catalog),
    ):
        result = await run_turn(mode_id="executor", query="implement it", hooks=hooks)

    assert len(model_requests) == 3
    assert sum("complete arguments" in request for request in model_requests) == 1
    assert result.tool_call_count == 0
    assert all(not message.get("tool_calls") for message in result.transcript)
    assert result.block_origin == "task_completion"

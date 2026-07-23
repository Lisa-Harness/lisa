"""Tests for host-owned approval commands and file-backed persistence."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from kernel.approval import (
    ApprovalIdentity,
    ApprovalInvocation,
    ApprovalRequired,
    build_approval_invocation,
)
from kernel.models import MemoryScope, ToolDefinition, ToolExecutionContext, ToolResult
from kernel.tools.registry import execute_tool
from kernel.turn_runner import TurnHooks
from pydantic import BaseModel

from hosts.core import HostConfig, HostRuntime, NullTransport
from hosts.core.handle_message import handle_message


class _Parameters(BaseModel):
    path: str


def _identity(runtime: HostRuntime) -> ApprovalIdentity:
    return ApprovalIdentity(
        workspace_id=runtime.session_plan.manifest.name,
        user_id=runtime.config.user_slug,
        user_slug=runtime.config.user_slug,
        surface_type=runtime.config.surface_type,
        surface_id=runtime.config.surface_id,
        memory_scope=runtime.config.memory_scope,
        mode_id=runtime.current_mode_id,
        session_plan_digest=runtime.session_plan.plan_digest,
    )


@pytest.mark.asyncio
async def test_approve_dispatches_only_the_saved_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`/approve` bypasses model routing and calls the exact executor once."""

    runtime = HostRuntime(
        HostConfig.from_env(
            data_dir=str(tmp_path),
            user_slug="user",
            surface_type="personal_dm",
            surface_id="surface",
        )
    )
    identity = _identity(runtime)
    invocation = build_approval_invocation(
        target_kind="tool",
        target_id="coding_apply_patch",
        pack_id="lisa-pack-coding",
        effect="write",
        parameters={"patch": "diff --git a/a b/a"},
        identity=identity,
        pack_config={"workspace_root": "."},
        safe_preview="apply the patch",
    )
    await runtime.approval_service.request(invocation)
    executed: list[str] = []

    async def execute_saved(*, invocation: ApprovalInvocation, hooks: object) -> ToolResult:
        _ = hooks
        executed.append(invocation.request_id)
        return ToolResult(output_text="patch applied")

    handle_message_module = importlib.import_module("hosts.core.handle_message")
    monkeypatch.setattr(handle_message_module, "execute_approved_invocation", execute_saved)

    result = await handle_message(runtime, f"/approve {invocation.request_id}", NullTransport())

    assert result.text == "patch applied"
    assert executed == [invocation.request_id]
    persisted = await runtime.approval_service.get(invocation.request_id)
    assert persisted is not None
    assert persisted.state == "approved"


@pytest.mark.asyncio
async def test_approve_consumes_receipt_through_real_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`/approve` executes only the approved action and consumes its receipt."""

    runtime = HostRuntime(
        HostConfig.from_env(
            data_dir=str(tmp_path),
            user_slug="user",
            surface_type="personal_dm",
            surface_id="surface",
        )
    )
    executed: list[str] = []

    async def write_file(ctx: ToolExecutionContext) -> ToolResult:
        executed.append(ctx.parameters["path"])
        return ToolResult(tool_id="write_file", output_text="file written")

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
    context = ToolExecutionContext(
        tool_id="write_file",
        workspace_id=runtime.session_plan.manifest.name,
        user_id="user",
        user_slug="user",
        surface_type="personal_dm",
        surface_id="surface",
        memory_scope=cast(MemoryScope, runtime.config.memory_scope),
        mode_id=runtime.current_mode_id,
        session_plan_digest=runtime.session_plan.plan_digest,
        parameters={"path": "src/a.py"},
        approval_service=runtime.approval_service,
    )

    with pytest.raises(ApprovalRequired) as required:
        await execute_tool("write_file", context)
    invocation = required.value.invocation

    handle_message_module = importlib.import_module("hosts.core.handle_message")
    monkeypatch.setattr(
        handle_message_module,
        "build_turn_hooks",
        lambda _runtime, **_kwargs: TurnHooks(
            approval_service=runtime.approval_service,
            workspace_id=runtime.session_plan.manifest.name,
            user_id="user",
            user_slug="user",
            surface_type="personal_dm",
            surface_id="surface",
            memory_scope=runtime.config.memory_scope,
            session_plan_digest=runtime.session_plan.plan_digest,
        ),
    )

    result = await handle_message(runtime, f"/approve {invocation.request_id}", NullTransport())

    assert result.text == "file written"
    assert executed == ["src/a.py"]
    persisted = await runtime.approval_service.get(invocation.request_id)
    assert persisted is not None
    assert persisted.state == "consumed"


@pytest.mark.asyncio
async def test_deny_marks_the_saved_invocation_unusable(tmp_path: Path) -> None:
    """`/deny` is a host command and leaves no receipt that can be consumed."""

    runtime = HostRuntime(
        HostConfig.from_env(
            data_dir=str(tmp_path),
            user_slug="user",
            surface_type="personal_dm",
            surface_id="surface",
        )
    )
    identity = _identity(runtime)
    invocation = build_approval_invocation(
        target_kind="tool",
        target_id="coding_cleanup_worktree",
        pack_id="lisa-pack-coding",
        effect="write",
        parameters={"artifact_id": "artifact"},
        identity=identity,
        pack_config={},
    )
    await runtime.approval_service.request(invocation)

    result = await handle_message(runtime, f"/deny {invocation.request_id}", NullTransport())

    assert "Denied approval request" in result.text
    persisted = await runtime.approval_service.get(invocation.request_id)
    assert persisted is not None
    assert persisted.state == "denied"

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


def _make_runtime(tmp_path: Path, **config_kwargs: str) -> HostRuntime:
    """Build a HostRuntime whose data_dir is tmp_path with a minimal models.yaml.

    Seeds a minimal models.yaml so manifest-dependent code paths (capability
    snapshot, model resolution) have at least one model. Mirrors the helper
    in test_core_services.py.
    """

    (tmp_path / "models.yaml").write_text(
        "\n".join(
            [
                "version: 1",
                "mode_models: {default: gemma3:latest}",
                "models:",
                "  - model_id: gemma3:latest",
                "    provider: ollama",
                "  - model_id: deepseek-r1:latest",
                "    provider: ollama",
                "    reasoning_model: true",
            ]
        ),
        encoding="utf-8",
    )
    config = HostConfig.from_env(
        data_dir=str(tmp_path),
        user_slug="user",
        surface_type="personal_dm",
        surface_id="surface",
        **config_kwargs,
    )
    return HostRuntime(config)


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


@pytest.mark.asyncio
async def test_affirmative_resume_approves_the_single_pending_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Typing "proceed" with one pending approval resumes it via /approve.

    Reproduces the user-side of the approval loop: the user typed ``proceed``
    instead of ``/approve <id>`` and the host routed a fresh model turn,
    which re-issued the effectful tool and minted a new request_id. The host
    now treats a short affirmative as an implicit ``/approve`` when exactly
    one approval is pending.
    """

    runtime = _make_runtime(tmp_path)
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

    result = await handle_message(runtime, "proceed", NullTransport())

    assert result.text == "patch applied"
    assert executed == [invocation.request_id]
    persisted = await runtime.approval_service.get(invocation.request_id)
    assert persisted is not None
    assert persisted.state == "approved"


@pytest.mark.asyncio
async def test_affirmative_resume_is_case_insensitive_and_phrase_scoped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Short affirmative variants resume; longer messages do not."""

    runtime = _make_runtime(tmp_path)
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

    result = await handle_message(runtime, "OK", NullTransport())
    assert result.text == "patch applied"
    assert executed == [invocation.request_id]


@pytest.mark.asyncio
async def test_affirmative_resume_does_not_fire_with_multiple_pending(
    tmp_path: Path,
) -> None:
    """With more than one pending approval the user must disambiguate.

    Guessing would silently execute the wrong action, so the affirmative is
    ignored and the message falls through to normal routing.
    """

    runtime = _make_runtime(tmp_path)
    identity = _identity(runtime)
    first = build_approval_invocation(
        target_kind="tool",
        target_id="coding_apply_patch",
        pack_id="lisa-pack-coding",
        effect="write",
        parameters={"patch": "diff --git a/a b/a"},
        identity=identity,
        pack_config={"workspace_root": "."},
        safe_preview="apply the patch",
    )
    second = build_approval_invocation(
        target_kind="tool",
        target_id="coding_cleanup_worktree",
        pack_id="lisa-pack-coding",
        effect="write",
        parameters={"artifact_id": "artifact"},
        identity=identity,
        pack_config={"workspace_root": "."},
        safe_preview="clean up the worktree",
    )
    await runtime.approval_service.request(first)
    await runtime.approval_service.request(second)

    pending = await runtime.approval_service.list_pending(identity)
    assert len(pending) == 2

    # The affirmative resume must not consume either; both stay pending.
    # handle_message would route a model turn, but we only assert that the
    # approval state is untouched (the model-turn path is exercised by the
    # turn-runner tests and is blocked here by the catalog collision in
    # the test environment).
    from hosts.core.handle_message import _try_affirmative_approval_resume

    resume_result = await _try_affirmative_approval_resume(
        runtime, "proceed", "proceed", NullTransport()
    )
    assert resume_result is None
    assert (await runtime.approval_service.get(first.request_id)).state == "pending"
    assert (await runtime.approval_service.get(second.request_id)).state == "pending"


@pytest.mark.asyncio
async def test_affirmative_resume_does_not_fire_without_pending(
    tmp_path: Path,
) -> None:
    """An affirmative with nothing pending is not consumed as a resume."""

    runtime = _make_runtime(tmp_path)

    from hosts.core.handle_message import _try_affirmative_approval_resume

    resume_result = await _try_affirmative_approval_resume(
        runtime, "proceed", "proceed", NullTransport()
    )
    assert resume_result is None


@pytest.mark.asyncio
async def test_non_affirmative_message_is_not_consumed_as_resume(
    tmp_path: Path,
) -> None:
    """A question or long message is never treated as an approval resume."""

    runtime = _make_runtime(tmp_path)
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

    from hosts.core.handle_message import _try_affirmative_approval_resume

    resume_result = await _try_affirmative_approval_resume(
        runtime,
        "can you explain what this will do?",
        "can you explain what this will do?",
        NullTransport(),
    )
    assert resume_result is None
    assert (await runtime.approval_service.get(invocation.request_id)).state == "pending"

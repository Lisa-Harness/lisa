"""Cross-layer conformance tests for the approval↔recovery link.

The kernel and runtime are separate packages linked by string contracts the
import-linter cannot see:

- Thread provenance keys: ``handle_message`` writes
  ``pending_approval_request_id`` / ``pending_approval_recovery_item_id``;
  ``runtime.approval.orchestration`` and ``runtime.thread`` read them to
  find the paused entry and its paired recovery work item. Drift strands
  the paired settlement — the mechanism of the 2026-08-06 zombie-stall
  incident.
- Identity mirroring: ``kernel.turn_runner._build_recovery_identity`` (the
  writer) and ``runtime.approval.orchestration._recovery_identity_for_invocation``
  (the reader) must produce identity scopes that compare equal, because
  ``RecoveryService.get`` / ``settle`` require an exact identity match.

These tests exercise the real writer and real reader end-to-end so a rename
on either side fails here instead of silently orphaning stalls in
production.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from kernel.approval import ApprovalIdentity, build_approval_invocation
from kernel.recovery import (
    RECOVERY_APPROVAL_PENDING_ERROR,
    RecoveryIdentity,
    RecoveryRecord,
)
from kernel.turn_hooks import HostServiceHandles, TurnIdentity
from kernel.turn_runner import TurnHooks
from kernel.turn_runner._defaults import _build_recovery_identity
from runtime.approval.orchestration import (
    _recovery_identity_for_invocation,
    settle_paired_recovery_record,
)
from runtime.recovery import FileRecoveryService
from runtime.thread import append_thread_entry

from runtime import HostConfig, HostRuntime


def _make_runtime(tmp_path: Path) -> HostRuntime:
    return HostRuntime(
        HostConfig.from_env(
            data_dir=str(tmp_path),
            user_slug="user",
            surface_type="personal_dm",
            surface_id="surface",
        )
    )


def _turn_identity(runtime: HostRuntime) -> TurnIdentity:
    return TurnIdentity(
        workspace_id=runtime.session_plan.manifest.name,
        user_id=runtime.config.user_slug,
        user_slug=runtime.config.user_slug,
        surface_type=runtime.config.surface_type,
        surface_id=runtime.config.surface_id,
        memory_scope=runtime.config.memory_scope,
    )


def _kernel_recovery_identity(runtime: HostRuntime) -> RecoveryIdentity:
    """Run the kernel writer against hooks built the way the turn loop builds them."""
    return _build_recovery_identity(
        TurnHooks(services=HostServiceHandles(), identity=_turn_identity(runtime))
    )


def _recovery_service(runtime: HostRuntime) -> FileRecoveryService:
    return cast(FileRecoveryService, runtime.services.recovery_service)


def _shim_invocation(runtime: HostRuntime, target_id: str, request_id: str) -> Any:
    """Build a minimal invocation-shaped object for the runtime reader."""

    class _ShimInvocation:
        identity: ApprovalIdentity
        session_plan_digest: str
        target_id: str
        request_id: str

    shim = _ShimInvocation()
    shim.identity = ApprovalIdentity(
        workspace_id=runtime.session_plan.manifest.name,
        user_id=runtime.config.user_slug,
        user_slug=runtime.config.user_slug,
        surface_type=runtime.config.surface_type,
        surface_id=runtime.config.surface_id,
        memory_scope=runtime.config.memory_scope,
        mode_id="executor",
        session_plan_digest=runtime.session_plan.plan_digest,
    )
    shim.session_plan_digest = runtime.session_plan.plan_digest
    shim.target_id = target_id
    shim.request_id = request_id
    return shim


# ── Identity mirroring (kernel writer ↔ runtime reader) ────────────────────


def test_kernel_identity_scope_matches_runtime_approval_reader(
    tmp_path: Path,
) -> None:
    """The identity scope the kernel writes records under must equal the
    scope ``_recovery_identity_for_invocation`` rebuilds for settlement.

    ``RecoveryService.get``/``settle`` match identity exactly, so field
    drift between the kernel turn-runner writer and the runtime approval
    reader would silently orphan every paired recovery record.
    """
    runtime = _make_runtime(tmp_path)
    kernel_identity = _kernel_recovery_identity(runtime)
    reader_identity = _recovery_identity_for_invocation(
        runtime,
        _shim_invocation(runtime, target_id="coding_apply_patch", request_id="req-1"),
    )
    assert reader_identity is not None
    assert reader_identity == kernel_identity


def test_runtime_identity_reader_rejects_stale_session_plan(
    tmp_path: Path,
) -> None:
    """A stale plan digest must fail scoped, not settle another session's stall."""
    runtime = _make_runtime(tmp_path)
    shim = _shim_invocation(runtime, target_id="coding_apply_patch", request_id="req-1")
    shim.session_plan_digest = "stale-digest"
    assert _recovery_identity_for_invocation(runtime, shim) is None


# ── Provenance keys (handle_message writer ↔ orchestration reader) ─────────


@pytest.mark.asyncio
async def test_paused_approval_provenance_drives_paired_settlement(
    tmp_path: Path,
) -> None:
    """The provenance shape ``handle_message`` writes must let
    ``settle_paired_recovery_record`` find and settle the paired record.

    Guards every key the settlement path reads:
    ``pending_approval_request_id`` (paused-entry lookup),
    ``pending_approval_recovery_item_id`` (paired work item), and the
    approval-identity fields the settlement correlates on.
    """
    runtime: HostRuntime = _make_runtime(tmp_path)
    identity = _kernel_recovery_identity(runtime)
    service = _recovery_service(runtime)

    started = await service.begin(
        RecoveryRecord(
            identity=identity,
            mode_id="executor",
            target_kind="pack_tool",
            target_id="coding_apply_patch",
            effect="write",
            retry_policy="confirm",
        )
    )
    await service.settle(
        started.work_item_id,
        identity,
        state="stalled",
        last_error=RECOVERY_APPROVAL_PENDING_ERROR,
        retry_policy="confirm",
    )

    invocation = build_approval_invocation(
        target_kind="tool",
        target_id="coding_apply_patch",
        pack_id="lisa-pack-coding",
        effect="write",
        parameters={"patch": "diff"},
        identity=cast(
            ApprovalIdentity,
            ApprovalIdentity(
                workspace_id=identity.workspace_id,
                user_id=identity.user_id,
                user_slug=identity.user_slug,
                surface_type=cast(Any, identity.surface_type),
                surface_id=identity.surface_id,
                memory_scope=cast(Any, identity.memory_scope),
                mode_id="executor",
                session_plan_digest=runtime.session_plan.plan_digest,
            ),
        ),
        pack_config={},
    )
    assert invocation.request_id

    # Exactly the provenance shape ``handle_message`` produces for a paused
    # approval turn (see runtime.handle_message thread persistence).
    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        mode_id="executor",
        query="fix it",
        response_text="approval required",
        turn=0,
        provenance={
            "source": "observed",
            "finish_reason": "unknown",
            "model_id": "m",
            "task_id": "task-1",
            "pending_approval_request_id": invocation.request_id,
            "pending_approval_tool_call_id": "call-1",
            "pending_approval_recovery_item_id": started.work_item_id,
        },
    )

    await settle_paired_recovery_record(
        runtime,
        invocation,
        state="succeeded",
        result_preview="patched",
    )

    settled = await service.get(started.work_item_id, identity)
    assert settled is not None
    assert settled.state == "succeeded", (
        "paired settlement could not resolve the provenance keys handle_message writes"
    )


# ── Stall sentinel end-to-end (kernel writer ↔ lifecycle gate) ─────────────


@pytest.mark.asyncio
async def test_kernel_approval_stall_carries_lifecycle_sentinel(
    tmp_path: Path,
) -> None:
    """The writer must stamp the sentinel the lifecycle gate now reads.

    The gate keys on ``last_error == RECOVERY_APPROVAL_PENDING_ERROR``;
    if the dispatch layer's settlement stops writing the sentinel (or
    writes a different string), the workflow-blocking yield silently
    disables itself — the failure mode this contract exists to catch.
    """
    runtime = _make_runtime(tmp_path)
    identity = _kernel_recovery_identity(runtime)
    service = _recovery_service(runtime)

    started = await service.begin(
        RecoveryRecord(
            identity=identity,
            target_kind="pack_tool",
            target_id="coding_apply_patch",
            effect="write",
            retry_policy="confirm",
        )
    )
    await service.settle(
        started.work_item_id,
        identity,
        state="stalled",
        last_error=RECOVERY_APPROVAL_PENDING_ERROR,
        retry_policy="confirm",
    )

    stalled = await service.get(started.work_item_id, identity)
    assert stalled is not None
    assert stalled.state == "stalled"
    assert stalled.last_error == RECOVERY_APPROVAL_PENDING_ERROR

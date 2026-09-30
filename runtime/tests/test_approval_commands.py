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
from kernel.models import (
    MemoryScope,
    OperationResult,
    ToolDefinition,
    ToolExecutionContext,
    ToolResult,
)
from kernel.tools.registry import execute_tool
from kernel.turn_hooks import (
    HostServiceHandles,
    TurnIdentity,
)
from kernel.turn_runner import TurnHooks, TurnResult
from pydantic import BaseModel
from runtime.approvals import FileApprovalService
from runtime.handle_message import handle_message
from runtime.policy import PolicyDocument, PolicyRule
from runtime.policy_service import PolicyApprovalService
from runtime.types import HostCommandResult

from runtime import HostConfig, HostRuntime, NullTransport


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
        mode_id=runtime.session.current_mode_id,
        session_plan_digest=runtime.session_plan.plan_digest,
    )


def _make_runtime(tmp_path: Path, **config_kwargs: str) -> HostRuntime:
    """Build a HostRuntime whose data_dir is tmp_path with a minimal models.yaml.

    Seeds a minimal models.yaml so manifest-dependent code paths (capability
    snapshot, model resolution) have at least one model. Mirrors the helper
    in test_core_services.py.
    """
    from runtime.testing import build_test_runtime

    models_yaml = "\n".join(
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
    )
    runtime = build_test_runtime(
        tmp_path,
        models_yaml=models_yaml,
        user_slug="user",
        surface_type="personal_dm",
        surface_id="surface",
        **config_kwargs,
    )
    manifest = runtime.session_plan.manifest
    packs = manifest.resources.packs
    runtime.session_plan = runtime.session_plan.model_copy(
        update={
            "manifest": manifest.model_copy(
                update={
                    "resources": manifest.resources.model_copy(
                        update={
                            "packs": packs.model_copy(
                                update={"include": (*packs.include, "lisa-pack-bwrap")}
                            )
                        }
                    )
                }
            )
        }
    )
    return runtime


async def _unused_route(
    _runtime: HostRuntime,
    message: str,
    _transport: object,
) -> HostCommandResult:  # pragma: no cover - never invoked in these tests
    """Stand-in ``route_model_turn`` for resume-detection paths that no-op."""
    raise AssertionError(f"route_model_turn must not be called here (got {message!r})")


def _stub_continuation_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    """Stub the post-approval continuation turn and return its queries.

    The approve path now records the executed action and routes a synthesis
    turn in the owning mode; tests stub ``_route_model_turn`` so no model
    call is attempted and the continuation query can be asserted (it must
    carry the action output and the do-not-re-run instruction).
    """

    queries: list[str] = []

    async def _continuation(
        _runtime: HostRuntime,
        message: str,
        _transport: object,
    ) -> HostCommandResult:
        queries.append(message)
        return HostCommandResult(text="continuation synthesized")

    handle_message_module = importlib.import_module("runtime.handle_message")
    monkeypatch.setattr(handle_message_module, "_route_model_turn", _continuation)
    return queries


@pytest.mark.asyncio
async def test_approve_dispatches_only_the_saved_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`/approve` executes the exact saved invocation once, then continues.

    The action output must reach the model: the continuation turn carries
    it as observed data with an explicit do-not-re-run instruction, so the
    model reports the result instead of re-issuing the action.
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
    await runtime.services.approval_service.request(invocation)
    executed: list[str] = []

    async def execute_saved(*, invocation: ApprovalInvocation, hooks: object) -> ToolResult:
        _ = hooks
        executed.append(invocation.request_id)
        return ToolResult(output_text="patch applied")

    approval_module = importlib.import_module("runtime.approval.orchestration")
    monkeypatch.setattr(approval_module, "execute_approved_invocation", execute_saved)
    continuation_queries = _stub_continuation_turn(monkeypatch)

    result = await handle_message(runtime, f"/approve {invocation.request_id}", NullTransport())

    assert executed == [invocation.request_id]
    assert len(continuation_queries) == 1
    assert "patch applied" in continuation_queries[0]
    # The continuation is a report, not an invitation to plan more tool
    # work — a one-shot CLI process would pause for another approval on any
    # new effectful action (the double-approve friction on tumbleweed3).
    assert "Do not call any tools" in continuation_queries[0]
    assert "do not start follow-up commands" in continuation_queries[0]
    # Ambition contract: the report ends with a proposed next step so the
    # synthesis turn stays useful without re-inviting approvals (the 2026-09-17
    # "shell session ready / what next" passivity pattern).
    assert "next step" in continuation_queries[0]
    assert result.text == "continuation synthesized"
    persisted = await runtime.services.approval_service.get(invocation.request_id)
    assert persisted is not None
    assert persisted.state == "approved"


@pytest.mark.asyncio
async def test_pending_approval_handoff_updates_session_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A paused approval carries the mode handoff so the next turn resumes in the owner."""
    runtime = _make_runtime(tmp_path)
    runtime.session.current_mode_id = "default"
    identity = ApprovalIdentity(
        workspace_id=runtime.session_plan.manifest.name,
        user_id=runtime.config.user_slug,
        user_slug=runtime.config.user_slug,
        surface_type=runtime.config.surface_type,
        surface_id=runtime.config.surface_id,
        memory_scope=runtime.config.memory_scope,
        mode_id="executor",
        session_plan_digest=runtime.session_plan.plan_digest,
    )
    invocation = build_approval_invocation(
        target_kind="tool",
        target_id="coding_apply_patch",
        pack_id="lisa-pack-coding",
        effect="write",
        parameters={"patch": "diff --git a/a b/a"},
        identity=identity,
        pack_config={},
    )
    result = TurnResult(
        response_text="approval required", pending_approval=invocation, handoff_to="executor"
    )

    async def _continue(
        _runtime: HostRuntime, _message: str, _transport: object, *, task_id: str = ""
    ) -> HostCommandResult:
        assert _runtime.session.current_mode_id == "executor"
        assert task_id == "task-1"
        return HostCommandResult(text="continuation synthesized")

    monkeypatch.setattr("runtime.handle_message._route_model_turn", _continue)

    response = await importlib.import_module("runtime.handle_message")._handle_pending_approval(
        runtime,
        result,
        mode_id="default",
        query="fix it",
        resolved_model=SimpleNamespace(model_id="m", provider="p", reasoning_model=False),
        turn_index=7,
        task_id="task-1",
        transport=NullTransport(),
    )

    assert response is not None
    assert runtime.session.current_mode_id == "executor"
    assert response.text == "approval required"


@pytest.mark.asyncio
async def test_explicit_pick_up_resumes_pending_task_in_owning_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from runtime.thread import append_thread_entry

    runtime = _make_runtime(tmp_path)
    runtime.session.current_mode_id = "default"
    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        mode_id="default",
        query="fix issue 6",
        response_text="approval required",
        handoff_to="executor",
        provenance={"task_id": "task-1", "pending_approval_request_id": "approval-1"},
    )
    for task_id in ("task-2", "task-3"):
        append_thread_entry(
            thread_path=runtime.paths.thread_path,
            user_slug=runtime.config.user_slug,
            mode_id="default",
            query="lets pick up where we left off.",
            response_text="The patch is still waiting for approval.",
            provenance={"task_id": task_id},
        )
    turn_module = importlib.import_module("runtime.handle_message")
    captured: dict[str, str] = {}

    async def ensure_pool_ready() -> None:
        return None

    async def execute_mode_turn(
        _runtime: HostRuntime,
        _message: str,
        mode_id: str,
        _query: str,
        _transport: object,
        *,
        task_id: str = "",
    ) -> str:
        captured.update(mode_id=mode_id, task_id=task_id)
        return "routed"

    monkeypatch.setattr(runtime, "ensure_pool_ready", ensure_pool_ready)
    monkeypatch.setattr(turn_module, "_execute_mode_turn", execute_mode_turn)

    result = await turn_module._route_model_turn(
        runtime, "lets pick up where we left off.", NullTransport()
    )

    assert result == "routed"
    assert captured == {"mode_id": "executor", "task_id": "task-1"}


@pytest.mark.asyncio
async def test_policy_minted_receipt_executes_without_second_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An auto-approved record executes its receipt without re-approving it."""

    runtime = _make_runtime(tmp_path)
    runtime.services.approval_service = PolicyApprovalService(
        FileApprovalService(tmp_path / "approvals.json"),
        policy=PolicyDocument(allow=[PolicyRule(tool="bash_run")]),
        data_dir=tmp_path,
    )
    invocation = build_approval_invocation(
        target_kind="tool",
        target_id="bash_run",
        pack_id="lisa-pack-bash",
        effect="execute",
        parameters={"command": "pwd"},
        identity=_identity(runtime),
        pack_config={},
        safe_preview="run a shell command",
    )
    await runtime.services.approval_service.request(invocation)
    record = await runtime.services.approval_service.get(invocation.request_id)
    assert record is not None and record.state == "approved" and record.receipt is not None

    async def must_not_reapprove(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("an approved record must execute its existing receipt")

    executed: list[str] = []

    async def execute_saved(*, invocation: ApprovalInvocation, hooks: object) -> ToolResult:
        _ = hooks
        executed.append(invocation.request_id)
        return ToolResult(output_text="/workspace")

    class _RecordingTransport(NullTransport):
        def __init__(self) -> None:
            self.responses: list[tuple[str, dict[str, object] | None]] = []

        async def on_response(self, text: str, data: dict[str, object] | None = None) -> None:
            self.responses.append((text, data))

    approval_module = importlib.import_module("runtime.approval.orchestration")
    monkeypatch.setattr(runtime.services.approval_service, "approve", must_not_reapprove)
    monkeypatch.setattr(approval_module, "execute_approved_invocation", execute_saved)
    _stub_continuation_turn(monkeypatch)
    transport = _RecordingTransport()

    result = await handle_message(runtime, f"/approve {invocation.request_id}", transport)

    assert result.text == "continuation synthesized"
    assert executed == [invocation.request_id]
    assert transport.responses == [("continuation synthesized", None)]


@pytest.mark.asyncio
async def test_approve_consumes_receipt_through_real_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`/approve` executes only the approved action, consumes its receipt, and
    carries the output into the continuation turn."""

    runtime = _make_runtime(tmp_path)
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
        mode_id=runtime.session.current_mode_id,
        session_plan_digest=runtime.session_plan.plan_digest,
        parameters={"path": "src/a.py"},
        approval_service=runtime.services.approval_service,
    )

    with pytest.raises(ApprovalRequired) as required:
        await execute_tool("write_file", context)
    invocation = required.value.invocation

    approval_module = importlib.import_module("runtime.approval.orchestration")
    monkeypatch.setattr(
        approval_module,
        "build_turn_hooks",
        lambda _runtime, **_kwargs: TurnHooks(
            services=HostServiceHandles(
                approval_service=runtime.services.approval_service,
            ),
            identity=TurnIdentity(
                workspace_id=runtime.session_plan.manifest.name,
                user_id="user",
                user_slug="user",
                surface_type="personal_dm",
                surface_id="surface",
                memory_scope=runtime.config.memory_scope,
                session_plan_digest=runtime.session_plan.plan_digest,
            ),
        ),
    )
    continuation_queries = _stub_continuation_turn(monkeypatch)

    result = await handle_message(runtime, f"/approve {invocation.request_id}", NullTransport())

    assert result.text == "continuation synthesized"
    assert executed == ["src/a.py"]
    assert len(continuation_queries) == 1
    assert "file written" in continuation_queries[0]
    persisted = await runtime.services.approval_service.get(invocation.request_id)
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
    await runtime.services.approval_service.request(invocation)

    result = await handle_message(runtime, f"/deny {invocation.request_id}", NullTransport())

    assert "Denied approval request" in result.text
    persisted = await runtime.services.approval_service.get(invocation.request_id)
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
    await runtime.services.approval_service.request(invocation)
    executed: list[str] = []

    async def execute_saved(*, invocation: ApprovalInvocation, hooks: object) -> ToolResult:
        _ = hooks
        executed.append(invocation.request_id)
        return ToolResult(output_text="patch applied")

    approval_module = importlib.import_module("runtime.approval.orchestration")
    monkeypatch.setattr(approval_module, "execute_approved_invocation", execute_saved)
    continuation_queries = _stub_continuation_turn(monkeypatch)

    result = await handle_message(runtime, "proceed", NullTransport())

    assert executed == [invocation.request_id]
    assert len(continuation_queries) == 1
    assert "patch applied" in continuation_queries[0]
    assert result.text == "continuation synthesized"
    persisted = await runtime.services.approval_service.get(invocation.request_id)
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
    await runtime.services.approval_service.request(invocation)
    executed: list[str] = []

    async def execute_saved(*, invocation: ApprovalInvocation, hooks: object) -> ToolResult:
        _ = hooks
        executed.append(invocation.request_id)
        return ToolResult(output_text="patch applied")

    approval_module = importlib.import_module("runtime.approval.orchestration")
    monkeypatch.setattr(approval_module, "execute_approved_invocation", execute_saved)
    continuation_queries = _stub_continuation_turn(monkeypatch)

    result = await handle_message(runtime, "OK", NullTransport())
    assert result.text == "continuation synthesized"
    assert executed == [invocation.request_id]
    assert len(continuation_queries) == 1
    assert "patch applied" in continuation_queries[0]


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
    await runtime.services.approval_service.request(first)
    await runtime.services.approval_service.request(second)

    pending = await runtime.services.approval_service.list_pending(identity)
    assert len(pending) == 2

    # The affirmative resume must not consume either; both stay pending.
    # handle_message would route a model turn, but we only assert that the
    # approval state is untouched (the model-turn path is exercised by the
    # turn-runner tests and is blocked here by the catalog collision in
    # the test environment).
    from runtime.approval import try_affirmative_approval_resume as _try_affirmative_approval_resume

    resume_result = await _try_affirmative_approval_resume(
        runtime, "proceed", "proceed", NullTransport(), route_model_turn=_unused_route
    )
    assert resume_result is None
    assert (await runtime.services.approval_service.get(first.request_id)).state == "pending"
    assert (await runtime.services.approval_service.get(second.request_id)).state == "pending"


@pytest.mark.asyncio
async def test_affirmative_resume_does_not_fire_without_pending(
    tmp_path: Path,
) -> None:
    """An affirmative with nothing pending is not consumed as a resume."""

    runtime = _make_runtime(tmp_path)

    from runtime.approval import try_affirmative_approval_resume as _try_affirmative_approval_resume

    resume_result = await _try_affirmative_approval_resume(
        runtime, "proceed", "proceed", NullTransport(), route_model_turn=_unused_route
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
    await runtime.services.approval_service.request(invocation)

    from runtime.approval import try_affirmative_approval_resume as _try_affirmative_approval_resume

    resume_result = await _try_affirmative_approval_resume(
        runtime,
        "can you explain what this will do?",
        "can you explain what this will do?",
        NullTransport(),
        route_model_turn=_unused_route,
    )
    assert resume_result is None
    assert (await runtime.services.approval_service.get(invocation.request_id)).state == "pending"


# ── Prefix-matching affirmative resume (option 2) ─────────────────────


@pytest.mark.asyncio
async def test_affirmative_resume_prefix_matches_proceed_with_suffix(
    tmp_path: Path,
) -> None:
    """``"proceed with the current step in the plan"`` matches ``"proceed"``.

    Reproduces the 2026-07-31 incident: the user typed a longer resume
    phrase that the old exact-match detector missed. The loosened
    prefix-match detector treats it as an affirmative resume when an
    approval is pending.
    """
    from kernel.routing_vocabulary import (
        register_routing_vocabulary,
        reset_routing_vocabulary,
    )

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
    await runtime.services.approval_service.request(invocation)

    from runtime.approval import is_affirmative_resume as _is_affirmative_resume

    try:
        register_routing_vocabulary(
            resume_phrases=[
                "proceed",
                "continue",
                "go ahead",
                "go for it",
                "do it",
                "carry on",
                "pick up",
                "resume",
                "next step",
                "keep going",
            ],
        )
        assert _is_affirmative_resume("proceed with the current step in the plan")
        assert _is_affirmative_resume("continue with the current step in the current plan")
        assert _is_affirmative_resume("go ahead and run it")
    finally:
        reset_routing_vocabulary()


def test_affirmative_resume_exact_match_still_works() -> None:
    """Conversational affirmatives (``"ok"``, ``"yes"``) still match exactly."""
    from kernel.routing_vocabulary import (
        register_routing_vocabulary,
        reset_routing_vocabulary,
    )
    from runtime.approval import is_affirmative_resume as _is_affirmative_resume

    try:
        register_routing_vocabulary(resume_phrases=["proceed"])
        assert _is_affirmative_resume("ok")
        assert _is_affirmative_resume("yes")
        assert _is_affirmative_resume("approved")
        assert _is_affirmative_resume("yep")
    finally:
        reset_routing_vocabulary()


def test_affirmative_resume_rejects_non_prefix_non_exact() -> None:
    """A message that does not start with a resume phrase and isn't exact is rejected."""
    from kernel.routing_vocabulary import (
        register_routing_vocabulary,
        reset_routing_vocabulary,
    )
    from runtime.approval import is_affirmative_resume as _is_affirmative_resume

    try:
        register_routing_vocabulary(resume_phrases=["proceed", "continue", "go ahead"])
        assert not _is_affirmative_resume("can you explain what this will do?")
        assert not _is_affirmative_resume("ok but let me think")
        assert not _is_affirmative_resume("yes that sounds risky")
    finally:
        reset_routing_vocabulary()


def test_affirmative_resume_uses_session_resume_phrases() -> None:
    """A session-authored resume phrase is honored by the prefix detector."""
    from kernel.routing_vocabulary import (
        register_routing_vocabulary,
        reset_routing_vocabulary,
    )
    from runtime.approval import is_affirmative_resume as _is_affirmative_resume

    try:
        register_routing_vocabulary(resume_phrases=["onward", "carry forward"])
        assert _is_affirmative_resume("onward with the plan")
        assert _is_affirmative_resume("carry forward from step 3")
        # The historical default ``"proceed"`` is NOT mixed in — the session
        # list is authoritative.
        assert not _is_affirmative_resume("proceed with the plan")
    finally:
        reset_routing_vocabulary()


# ── Approval ↔ recovery settlement linkage ────────────────────────────


@pytest.mark.asyncio
async def test_approve_settles_paired_stalled_recovery_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``/approve`` unblocks the paired recovery stall left by the model turn.

    Regression for the 2026-08-06 zombie-stall incident: the model-turn path
    settles a recovery record as ``stalled`` with ``blocks_workflow=True``
    before raising ``ApprovalRequired``. ``/approve`` re-executes through a
    fresh registry path that never touched the recovery service, so the
    stall stayed and re-yielded on every subsequent turn. The fix links the
    two stores by ``target_id`` + identity and settles the paired record on
    successful approve (and on deny).
    """

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
        target_id="coding_create_worktree",
        pack_id="lisa-pack-coding",
        effect="write",
        parameters={"summary": "learn the repo"},
        identity=identity,
        pack_config={"workspace_root": "."},
        safe_preview="create an isolated coding worktree",
    )
    await runtime.services.approval_service.request(invocation)

    # Simulate the model-turn path: it began a recovery record for the same
    # target, then settled it as ``stalled`` / ``blocks_workflow=True`` when
    # the tool raised ``ApprovalRequired``.
    from kernel.recovery import RecoveryIdentity, RecoveryRecord

    recovery_identity = RecoveryIdentity(
        workspace_id=runtime.session_plan.manifest.name,
        user_id="user",
        user_slug="user",
        surface_type=runtime.config.surface_type,
        surface_id=runtime.config.surface_id,
        memory_scope=cast(MemoryScope, runtime.config.memory_scope),
    )
    started = await runtime.services.recovery_service.begin(
        RecoveryRecord(
            identity=recovery_identity,
            mode_id="coding",
            stage_id="remember_1",
            target_kind="pack_tool",
            target_id="coding_create_worktree",
            effect="write",
            parameters={"summary": "learn the repo"},
            retry_policy="confirm",
        )
    )
    await runtime.services.recovery_service.settle(
        started.work_item_id,
        recovery_identity,
        state="stalled",
        last_error="approval required",
        retry_policy="confirm",
    )
    unrelated = await runtime.services.recovery_service.begin(
        RecoveryRecord(
            identity=recovery_identity,
            mode_id="coding",
            target_kind="pack_tool",
            target_id="coding_create_worktree",
            effect="write",
            parameters={"summary": "different request"},
        )
    )
    await runtime.services.recovery_service.settle(
        unrelated.work_item_id, recovery_identity, state="stalled", last_error="approval required"
    )
    from runtime.thread import append_thread_entry

    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        mode_id="coding",
        query="learn the repo",
        response_text="approval required",
        provenance={
            "pending_approval_request_id": invocation.request_id,
            "pending_approval_recovery_item_id": started.work_item_id,
        },
    )

    async def execute_saved(*, invocation: ApprovalInvocation, hooks: object) -> ToolResult:
        _ = hooks
        return ToolResult(output_text="worktree created")

    approval_module = importlib.import_module("runtime.approval.orchestration")
    monkeypatch.setattr(approval_module, "execute_approved_invocation", execute_saved)
    continuation_queries = _stub_continuation_turn(monkeypatch)
    monkeypatch.setattr(
        approval_module,
        "build_turn_hooks",
        lambda _runtime, **_kwargs: TurnHooks(
            services=HostServiceHandles(
                approval_service=runtime.services.approval_service,
            ),
            identity=TurnIdentity(
                workspace_id=runtime.session_plan.manifest.name,
                user_id="user",
                user_slug="user",
                surface_type="personal_dm",
                surface_id="surface",
                memory_scope=runtime.config.memory_scope,
                session_plan_digest=runtime.session_plan.plan_digest,
            ),
        ),
    )

    result = await handle_message(runtime, f"/approve {invocation.request_id}", NullTransport())

    assert result.text == "continuation synthesized"
    assert len(continuation_queries) == 1
    assert "worktree created" in continuation_queries[0]
    # The paired stall must be cleared so subsequent turns route to the model
    # instead of re-yielding the zombie stall.
    settled = await runtime.services.recovery_service.get(started.work_item_id, recovery_identity)
    assert settled is not None
    assert settled.state == "succeeded"
    other_record = await runtime.services.recovery_service.get(
        unrelated.work_item_id, recovery_identity
    )
    assert other_record is not None
    assert other_record.state == "stalled"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action_result",
    [
        ToolResult(error="sandbox unavailable", output_json={"artifact_id": "false-success"}),
        OperationResult(status="degraded", error_detail="partial write"),
    ],
)
async def test_approved_returned_failure_does_not_continue_or_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action_result: ToolResult | OperationResult
) -> None:
    runtime = _make_runtime(tmp_path)
    invocation = build_approval_invocation(
        target_kind="tool",
        target_id="write_file",
        pack_id="lisa-pack-coding",
        effect="write",
        parameters={"path": "src/a.py"},
        identity=_identity(runtime),
        pack_config={},
    )
    await runtime.services.approval_service.request(invocation)
    from kernel.recovery import RecoveryIdentity, RecoveryRecord
    from runtime.thread import append_thread_entry, load_recent_thread_entries

    recovery_identity = RecoveryIdentity(
        workspace_id=invocation.identity.workspace_id,
        user_id=invocation.identity.user_id,
        user_slug=invocation.identity.user_slug,
        surface_type=runtime.config.surface_type,
        surface_id=runtime.config.surface_id,
        memory_scope=cast(MemoryScope, runtime.config.memory_scope),
    )
    started = await runtime.services.recovery_service.begin(
        RecoveryRecord(
            identity=recovery_identity,
            target_kind="pack_tool",
            target_id="write_file",
            effect="write",
        )
    )
    await runtime.services.recovery_service.settle(
        started.work_item_id, recovery_identity, state="stalled", last_error="approval required"
    )
    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        mode_id=invocation.identity.mode_id,
        query="fix issue 6",
        response_text="approval required",
        provenance={
            "pending_approval_request_id": invocation.request_id,
            "pending_approval_recovery_item_id": started.work_item_id,
        },
    )

    async def execute_saved(
        *, invocation: ApprovalInvocation, hooks: object
    ) -> ToolResult | OperationResult:
        return action_result

    approval_module = importlib.import_module("runtime.approval.orchestration")
    monkeypatch.setattr(approval_module, "execute_approved_invocation", execute_saved)
    continuation_queries = _stub_continuation_turn(monkeypatch)

    response = await handle_message(runtime, f"/approve {invocation.request_id}", NullTransport())

    assert "Approved action failed" in response.text
    assert continuation_queries == []
    settled = await runtime.services.recovery_service.get(started.work_item_id, recovery_identity)
    assert settled is not None and settled.state == "stalled"
    entries = load_recent_thread_entries(thread_path=runtime.paths.thread_path)
    failed_entries = [
        entry
        for entry in entries
        if (entry.get("provenance") or {}).get("source") == "host-approved-failed"
    ]
    assert len(failed_entries) == 1
    assert failed_entries[0]["provenance"]["finish_reason"] == "approved-action-failed"
    assert not failed_entries[0].get("effect_receipts")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action_result",
    [ToolResult(output_text="patch applied"), ToolResult(error="invalid patch")],
)
async def test_policy_approved_action_resumes_even_when_mode_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action_result: ToolResult
) -> None:
    from kernel.models import WorkflowModeConfig
    from runtime.approval.orchestration import maybe_auto_continue_approval
    from runtime.thread import append_thread_entry

    runtime = _make_runtime(tmp_path)
    runtime.session.current_mode_id = "executor"
    invocation = build_approval_invocation(
        target_kind="tool",
        target_id="coding_apply_patch",
        pack_id="lisa-pack-coding",
        effect="write",
        parameters={"artifact_id": "artifact-one", "patch": "diff --git a/a b/a"},
        identity=_identity(runtime),
        pack_config={},
    )
    await runtime.services.approval_service.request(invocation)
    await runtime.services.approval_service.approve(invocation.request_id, invocation.identity)
    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        mode_id="executor",
        query="fix issue 6",
        response_text="approval required",
        provenance={"task_id": "task-one", "pending_approval_request_id": invocation.request_id},
    )
    approval_module = importlib.import_module("runtime.approval.orchestration")
    mode = WorkflowModeConfig(mode_id="executor", start_stage_id="none")
    monkeypatch.setattr(approval_module, "get_workflow_catalog", lambda: {"executor": mode})

    async def saved_action(*, invocation: ApprovalInvocation, hooks: object) -> ToolResult:
        return action_result

    monkeypatch.setattr(approval_module, "execute_approved_invocation", saved_action)
    queries: list[str] = []

    async def resume(
        host: HostRuntime, query: str, transport: object, *, task_id: str
    ) -> HostCommandResult:
        assert host.session.current_mode_id == "executor"
        assert task_id == "task-one"
        queries.append(query)
        return HostCommandResult(text="continued")

    result = await maybe_auto_continue_approval(
        runtime, invocation, NullTransport(), route_model_turn=resume
    )

    assert result is not None and result.text == "continued"
    assert len(queries) == 1
    if action_result.error:
        assert "invalid patch" in queries[0]
        assert "Continue from this failure" in queries[0]
    else:
        assert "patch applied" in queries[0]
        assert "Continue that request from where it paused" in queries[0]


@pytest.mark.asyncio
async def test_policy_approved_failure_resumes_with_error_not_success_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kernel.models import WorkflowModeConfig
    from runtime.approval.orchestration import maybe_auto_continue_approval
    from runtime.thread import append_thread_entry, load_recent_thread_entries

    runtime = _make_runtime(tmp_path)
    runtime.session.current_mode_id = "executor"
    invocation = build_approval_invocation(
        target_kind="tool",
        target_id="coding_apply_patch",
        pack_id="lisa-pack-coding",
        effect="write",
        parameters={"artifact_id": "artifact-one", "patch": "*** Begin Patch\n*** End Patch"},
        identity=_identity(runtime),
        pack_config={},
    )
    await runtime.services.approval_service.request(invocation)
    await runtime.services.approval_service.approve(invocation.request_id, invocation.identity)
    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        mode_id="executor",
        query="fix issue 6",
        response_text="approval required",
        provenance={"task_id": "task-one", "pending_approval_request_id": invocation.request_id},
    )
    approval_module = importlib.import_module("runtime.approval.orchestration")
    mode = WorkflowModeConfig(
        mode_id="executor", start_stage_id="none", approval_continuation="resume"
    )
    monkeypatch.setattr(approval_module, "get_workflow_catalog", lambda: {"executor": mode})
    error = "Patch must contain at least one unified diff file path."

    async def failed_action(*, invocation: ApprovalInvocation, hooks: object) -> ToolResult:
        return ToolResult(error=error, output_json={"artifact_id": "not-a-success"})

    monkeypatch.setattr(approval_module, "execute_approved_invocation", failed_action)
    queries: list[str] = []

    async def resume(
        host: HostRuntime, query: str, transport: object, *, task_id: str
    ) -> HostCommandResult:
        assert host.session.current_mode_id == "executor"
        assert task_id == "task-one"
        queries.append(query)
        return HostCommandResult(text="corrected invocation requested")

    result = await maybe_auto_continue_approval(
        runtime, invocation, NullTransport(), route_model_turn=resume
    )

    assert result is not None and result.text == "corrected invocation requested"
    assert len(queries) == 1 and error in queries[0] and "fix issue 6" in queries[0]
    assert "executed successfully" not in queries[0]
    entry = load_recent_thread_entries(thread_path=runtime.paths.thread_path)[-1]
    assert entry["provenance"]["finish_reason"] == "approved-action-failed"
    assert entry["provenance"]["task_id"] == "task-one"
    assert not entry.get("effect_receipts")


@pytest.mark.asyncio
async def test_failed_continuation_quotes_the_task_origin_not_the_previous_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kernel.models import WorkflowModeConfig
    from runtime.approval.orchestration import maybe_auto_continue_approval
    from runtime.thread import append_thread_entry

    runtime = _make_runtime(tmp_path)
    runtime.session.current_mode_id = "executor"
    invocation = build_approval_invocation(
        target_kind="tool",
        target_id="coding_apply_patch",
        pack_id="lisa-pack-coding",
        effect="write",
        parameters={"artifact_id": "artifact-one", "patch": "diff --git a/a b/a"},
        identity=_identity(runtime),
        pack_config={},
    )
    await runtime.services.approval_service.request(invocation)
    await runtime.services.approval_service.approve(invocation.request_id, invocation.identity)
    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        mode_id="executor",
        query="fix issue 6",
        response_text="approval required",
        provenance={"task_id": "task-one", "pending_approval_request_id": "earlier-request"},
    )
    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        mode_id="executor",
        query="/approve earlier-request",
        response_text="Approved action failed: first failure",
        provenance={"task_id": "task-one"},
    )
    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        mode_id="executor",
        query=(
            "The approved action `coding_apply_patch` failed (approval earlier-request):\n\n"
            "first failure\n\nThe original request was:\n\nfix issue 6\n\n"
            "Continue from this failure."
        ),
        response_text="approval required",
        provenance={"task_id": "task-one", "pending_approval_request_id": invocation.request_id},
    )
    approval_module = importlib.import_module("runtime.approval.orchestration")
    mode = WorkflowModeConfig(
        mode_id="executor", start_stage_id="none", approval_continuation="resume"
    )
    monkeypatch.setattr(approval_module, "get_workflow_catalog", lambda: {"executor": mode})

    async def failed_action(*, invocation: ApprovalInvocation, hooks: object) -> ToolResult:
        return ToolResult(error="second failure")

    monkeypatch.setattr(approval_module, "execute_approved_invocation", failed_action)
    queries: list[str] = []

    async def resume(
        host: HostRuntime, query: str, transport: object, *, task_id: str
    ) -> HostCommandResult:
        queries.append(query)
        return HostCommandResult(text="continued")

    await maybe_auto_continue_approval(
        runtime, invocation, NullTransport(), route_model_turn=resume
    )

    assert len(queries) == 1
    assert queries[0].count("The original request was:") == 1
    assert "fix issue 6" in queries[0]
    assert "first failure" not in queries[0]


@pytest.mark.asyncio
async def test_failed_auto_approvals_share_one_user_message_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kernel.models import WorkflowModeConfig
    from kernel.turn_runner import TurnResult

    runtime = _make_runtime(tmp_path)
    turn_module = importlib.import_module("runtime.handle_message")
    approval_module = importlib.import_module("runtime.approval.orchestration")
    mode = WorkflowModeConfig(
        mode_id="default", start_stage_id="none", mode_kind="chat", approval_continuation="resume"
    )
    monkeypatch.setattr(turn_module, "get_workflow_catalog", lambda: {"default": mode})
    monkeypatch.setattr(approval_module, "get_workflow_catalog", lambda: {"default": mode})
    monkeypatch.setattr(approval_module, "AUTO_APPROVE_TURN_CAP", 2)
    task_ids: list[str] = []
    executed: list[str] = []

    async def run_turn(*, hooks: object, **kwargs: object) -> TurnResult:
        task_ids.append(hooks.identity.task_id)
        assert len(task_ids) <= 3, "auto-approval budget reset during continuation"
        invocation = build_approval_invocation(
            target_kind="tool",
            target_id="coding_apply_patch",
            pack_id="lisa-pack-coding",
            effect="write",
            parameters={"patch": f"attempt-{len(task_ids)}"},
            identity=_identity(runtime),
            pack_config={},
        )
        await runtime.services.approval_service.request(invocation)
        await runtime.services.approval_service.approve(invocation.request_id, invocation.identity)
        return TurnResult(response_text="approval required", pending_approval=invocation)

    async def failed_action(*, invocation: ApprovalInvocation, hooks: object) -> ToolResult:
        executed.append(invocation.request_id)
        return ToolResult(error="invalid patch")

    monkeypatch.setattr(turn_module, "run_turn", run_turn)
    monkeypatch.setattr(approval_module, "execute_approved_invocation", failed_action)
    runtime.session.auto_approved_this_turn = 99

    result = await handle_message(runtime, "fix the code", NullTransport())

    assert len(executed) == 2
    assert len(task_ids) == 3 and len(set(task_ids)) == 1
    assert result.data is not None and result.data["request_id"] not in executed
    assert runtime.session.auto_approved_this_turn == 2


@pytest.mark.asyncio
async def test_deny_settles_paired_stalled_recovery_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``/deny`` dismisses the paired recovery stall so the workflow advances."""

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
        target_id="coding_create_worktree",
        pack_id="lisa-pack-coding",
        effect="write",
        parameters={"summary": "learn the repo"},
        identity=identity,
        pack_config={},
        safe_preview="create an isolated coding worktree",
    )
    await runtime.services.approval_service.request(invocation)

    from kernel.recovery import RecoveryIdentity, RecoveryRecord

    recovery_identity = RecoveryIdentity(
        workspace_id=runtime.session_plan.manifest.name,
        user_id="user",
        user_slug="user",
        surface_type=runtime.config.surface_type,
        surface_id=runtime.config.surface_id,
        memory_scope=cast(MemoryScope, runtime.config.memory_scope),
    )
    started = await runtime.services.recovery_service.begin(
        RecoveryRecord(
            identity=recovery_identity,
            mode_id="coding",
            stage_id="remember_1",
            target_kind="pack_tool",
            target_id="coding_create_worktree",
            effect="write",
            parameters={"summary": "learn the repo"},
            retry_policy="confirm",
        )
    )
    await runtime.services.recovery_service.settle(
        started.work_item_id,
        recovery_identity,
        state="stalled",
        last_error="approval required",
        retry_policy="confirm",
    )
    from runtime.thread import append_thread_entry

    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        mode_id="coding",
        query="learn the repo",
        response_text="approval required",
        provenance={
            "pending_approval_request_id": invocation.request_id,
            "pending_approval_recovery_item_id": started.work_item_id,
        },
    )

    approval_module = importlib.import_module("runtime.approval.orchestration")
    monkeypatch.setattr(
        approval_module,
        "build_turn_hooks",
        lambda _runtime, **_kwargs: TurnHooks(
            services=HostServiceHandles(
                approval_service=runtime.services.approval_service,
            ),
            identity=TurnIdentity(
                workspace_id=runtime.session_plan.manifest.name,
                user_id="user",
                user_slug="user",
                surface_type="personal_dm",
                surface_id="surface",
                memory_scope=runtime.config.memory_scope,
                session_plan_digest=runtime.session_plan.plan_digest,
            ),
        ),
    )

    result = await handle_message(runtime, f"/deny {invocation.request_id}", NullTransport())

    assert "Denied approval request" in result.text
    settled = await runtime.services.recovery_service.get(started.work_item_id, recovery_identity)
    assert settled is not None
    assert settled.state == "cancelled"


# ── Recorded-owner identity fallback (process-restart approval resume) ──


def _shell_owner_identity(runtime: HostRuntime) -> ApprovalIdentity:
    """The identity an approval carries when minted in a mid-turn handoff mode.

    Mirrors the 2026-09-04 remote bash_run incident: the turn entered
    ``default``, handed off to ``shell`` mid-turn, and ``bash_run`` minted
    the approval with ``mode_id="shell"``. After the process exited, the next
    ``lisa "/approve <id>"`` rebuilt the identity with ``mode_id="default"``
    and the exact-match stranded the approval.
    """

    return ApprovalIdentity(
        workspace_id=runtime.session_plan.manifest.name,
        user_id=runtime.config.user_slug,
        user_slug=runtime.config.user_slug,
        surface_type=runtime.config.surface_type,
        surface_id=runtime.config.surface_id,
        memory_scope=runtime.config.memory_scope,
        mode_id="shell",
        session_plan_digest=runtime.session_plan.plan_digest,
    )


def _reset_runtime_mode_to_start_mode(runtime: HostRuntime) -> None:
    """Simulate a fresh host process whose mode state reset to the start mode."""

    runtime.session.current_mode_id = runtime.session_plan.manifest.routing.start_mode or "default"


@pytest.mark.asyncio
async def test_approve_after_process_restart_matches_recorded_owner_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`/approve` resumes an approval minted in a mid-turn handoff mode.

    Reproduces the 2026-09-04 remote bash_run incident: the pending
    invocation's identity carries the *owning* mode (``shell``, set after the
    default → shell handoff), but every CLI invocation is a fresh process, so
    ``current_mode_id`` reset to the start mode (``default``). The live
    identity rebuild no longer matched and the approval stranded with an
    ambiguous "not pending" message. The handler now falls back to the
    recorded owner identity when the session scope still agrees.
    """

    runtime = _make_runtime(tmp_path)
    invocation = build_approval_invocation(
        target_kind="tool",
        target_id="bash_run",
        pack_id="lisa-pack-bash",
        effect="execute",
        parameters={"command": "hostname"},
        identity=_shell_owner_identity(runtime),
        pack_config={},
        safe_preview="run a shell command",
    )
    await runtime.services.approval_service.request(invocation)

    # Fresh process: the mode state reset to the start mode, so the live
    # identity rebuild diverges from the recorded owner exactly as observed.
    _reset_runtime_mode_to_start_mode(runtime)
    # lazy: the private helper is asserted against, not imported at module
    # scope like the public surface above.
    from runtime.approval import approval_identity as _host_approval_identity

    assert _host_approval_identity(runtime).mode_id != "shell"

    executed: list[str] = []

    async def execute_saved(*, invocation: ApprovalInvocation, hooks: object) -> ToolResult:
        _ = hooks
        executed.append(invocation.request_id)
        return ToolResult(output_text="hostname ok")

    approval_module = importlib.import_module("runtime.approval.orchestration")
    monkeypatch.setattr(approval_module, "execute_approved_invocation", execute_saved)
    continuation_queries = _stub_continuation_turn(monkeypatch)

    result = await handle_message(runtime, f"/approve {invocation.request_id}", NullTransport())

    assert executed == [invocation.request_id]
    assert len(continuation_queries) == 1
    assert "hostname ok" in continuation_queries[0]
    assert result.text == "continuation synthesized"
    persisted = await runtime.services.approval_service.get(invocation.request_id)
    assert persisted is not None
    assert persisted.state == "approved"


@pytest.mark.asyncio
async def test_approve_records_action_in_thread_with_effect_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The executed action lands in the thread with its receipt before the
    continuation turn.

    Reproduces the tumbleweed3 re-issue loop: the approval executed and
    returned ``root``, but no thread entry recorded it, so the next turn's
    context excerpt ended at the approval request and the model re-planned
    from the stale prompt — re-issuing the exact command it had already run.
    With the receipt in the ledger, the completion contract counts the
    action as done and later turns read an honest history.
    """

    runtime = _make_runtime(tmp_path)
    identity = _shell_owner_identity(runtime)
    invocation = build_approval_invocation(
        target_kind="tool",
        target_id="bash_run",
        pack_id="lisa-pack-bash",
        effect="execute",
        parameters={"command": "whoami"},
        identity=identity,
        pack_config={},
        safe_preview="run a shell command",
    )
    await runtime.services.approval_service.request(invocation)

    async def execute_saved(*, invocation: ApprovalInvocation, hooks: object) -> ToolResult:
        _ = hooks
        return ToolResult(output_text="root", output_json={"artifact_id": "artifact-7"})

    approval_module = importlib.import_module("runtime.approval.orchestration")
    monkeypatch.setattr(approval_module, "execute_approved_invocation", execute_saved)
    _stub_continuation_turn(monkeypatch)

    await handle_message(runtime, f"/approve {invocation.request_id}", NullTransport())

    # lazy: reads back the appended entry after the approve turn ran
    from runtime.thread import load_recent_thread_entries

    entries = load_recent_thread_entries(thread_path=runtime.paths.thread_path, limit=5)
    action_entries = [
        e for e in entries if e.get("provenance", {}).get("source") == "host-approved"
    ]
    assert len(action_entries) == 1
    entry = action_entries[0]
    # The receipt carries the executed capability and its effect kind so the
    # completion contract counts the action as done.
    assert entry["effect_receipts"] == [
        {"capability_id": "bash_run", "effect": "execute", "artifact_id": "artifact-7"}
    ]
    assert entry["response_text"] == "root"
    assert entry["provenance"]["approval_request_id"] == invocation.request_id
    # The entry is stamped with the owning mode, not the reset start mode.
    assert entry["mode_id"] == "shell"


async def _continuation_for_declared_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    approval_continuation: str,
) -> str:
    from runtime.thread import append_thread_entry

    runtime = _make_runtime(tmp_path)
    runtime.session.current_mode_id = "worker"
    invocation = build_approval_invocation(
        target_kind="tool",
        target_id="some_action",
        pack_id="some-pack",
        effect="write",
        parameters={},
        identity=_identity(runtime),
        pack_config={},
    )
    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        segment_turns=runtime.config.thread_segment_turns,
        mode_id="worker",
        query="fix issue 6",
        response_text="needs approval",
        provenance={"source": "observed", "pending_approval_request_id": invocation.request_id},
    )
    approval_module = importlib.import_module("runtime.approval.orchestration")
    monkeypatch.setattr(
        approval_module,
        "get_workflow_catalog",
        lambda: {"worker": SimpleNamespace(approval_continuation=approval_continuation)},
    )
    queries: list[str] = []

    async def route_model_turn(_runtime: HostRuntime, query: str, _t: object) -> HostCommandResult:
        queries.append(query)
        return HostCommandResult(text="continued")

    await approval_module._record_and_continue_after_approval(
        runtime,
        invocation,
        ToolResult(output_text="action output"),
        transport=NullTransport(),
        route_model_turn=route_model_turn,
    )
    assert len(queries) == 1
    return queries[0]


@pytest.mark.asyncio
async def test_approved_worktree_receipt_keeps_paused_task_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from runtime.thread import append_thread_entry, load_recent_thread_entries

    runtime = _make_runtime(tmp_path)
    runtime.session.current_mode_id = "executor"
    invocation = build_approval_invocation(
        target_kind="tool",
        target_id="coding_create_worktree",
        pack_id="lisa-pack-coding",
        effect="write",
        parameters={},
        identity=_identity(runtime),
        pack_config={},
    )
    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        segment_turns=1,
        mode_id="executor",
        query="fix the bug",
        response_text="needs approval",
        provenance={
            "source": "observed",
            "task_id": "task-original",
            "pending_approval_request_id": invocation.request_id,
        },
    )
    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        segment_turns=1,
        mode_id="executor",
        query="other request",
        response_text="old result",
        provenance={"source": "observed", "task_id": "other-task"},
    )
    approval_module = importlib.import_module("runtime.approval.orchestration")
    monkeypatch.setattr(
        approval_module,
        "get_workflow_catalog",
        lambda: {"executor": SimpleNamespace(approval_continuation="resume")},
    )
    resumed_ids: list[str] = []

    async def route_model_turn(
        _runtime: HostRuntime, _query: str, _transport: object, *, task_id: str = ""
    ) -> HostCommandResult:
        resumed_ids.append(task_id)
        return HostCommandResult(text="resumed")

    await approval_module._record_and_continue_after_approval(
        runtime,
        invocation,
        ToolResult(output_json={"artifact_id": "owned-worktree"}),
        transport=NullTransport(),
        route_model_turn=route_model_turn,
    )
    approved = load_recent_thread_entries(thread_path=runtime.paths.thread_path, limit=1)[0]
    assert approved["provenance"]["task_id"] == "task-original"
    assert approved["effect_receipts"][0]["artifact_id"] == "owned-worktree"
    assert resumed_ids == ["task-original"]


@pytest.mark.asyncio
async def test_approved_tool_executes_with_original_task_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from runtime.thread import append_thread_entry

    runtime = _make_runtime(tmp_path)
    runtime.session.current_mode_id = "executor"
    invocation = build_approval_invocation(
        target_kind="tool",
        target_id="coding_create_worktree",
        pack_id="lisa-pack-coding",
        effect="write",
        parameters={},
        identity=_identity(runtime),
        pack_config={},
    )
    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        mode_id="executor",
        query="fix the bug",
        response_text="needs approval",
        provenance={
            "source": "observed",
            "task_id": "task-original",
            "pending_approval_request_id": invocation.request_id,
        },
    )
    approval_module = importlib.import_module("runtime.approval.orchestration")
    seen_task_ids: list[str] = []

    async def execute_saved(*, invocation: ApprovalInvocation, hooks: object) -> ToolResult:
        seen_task_ids.append(hooks.identity.task_id)
        return ToolResult(output_text="worktree created")

    async def settle(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(approval_module, "execute_approved_invocation", execute_saved)
    monkeypatch.setattr(approval_module, "settle_paired_recovery_record", settle)
    await approval_module._execute_approved_with_settlement(
        runtime,
        SimpleNamespace(invocation=invocation),
        request_id=invocation.request_id,
        transport=NullTransport(),
    )
    assert seen_task_ids == ["task-original"]


@pytest.mark.asyncio
async def test_resume_mode_continuation_carries_paused_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    query = await _continuation_for_declared_mode(tmp_path, monkeypatch, "resume")

    assert "action output" in query
    assert "fix issue 6" in query
    assert "Do not re-run the approved action" in query
    assert "Do not call any tools" not in query


@pytest.mark.asyncio
async def test_report_mode_continuation_stays_report_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    query = await _continuation_for_declared_mode(tmp_path, monkeypatch, "report")

    assert "Do not call any tools" in query
    assert "fix issue 6" not in query


@pytest.mark.asyncio
async def test_approve_reports_decision_state_for_already_denied_request(
    tmp_path: Path,
) -> None:
    """Re-approving a decided request names the decision, not "not pending"."""

    runtime = _make_runtime(tmp_path)
    identity = _identity(runtime)
    invocation = build_approval_invocation(
        target_kind="tool",
        target_id="bash_run",
        pack_id="lisa-pack-bash",
        effect="execute",
        parameters={"command": "hostname"},
        identity=identity,
        pack_config={},
    )
    await runtime.services.approval_service.request(invocation)
    assert await runtime.services.approval_service.deny(invocation.request_id, identity) is True

    result = await handle_message(runtime, f"/approve {invocation.request_id}", NullTransport())

    assert "already denied" in result.text
    assert "not pending" not in result.text


@pytest.mark.asyncio
async def test_approve_reports_expiry_for_expired_request(
    tmp_path: Path,
) -> None:
    """An expired approval reports expiry and clears its paired recovery stall."""

    runtime = _make_runtime(tmp_path)
    identity = _identity(runtime)
    invocation = build_approval_invocation(
        target_kind="tool",
        target_id="bash_run",
        pack_id="lisa-pack-bash",
        effect="execute",
        parameters={"command": "hostname"},
        identity=identity,
        pack_config={},
    )
    await runtime.services.approval_service.request(invocation)

    from kernel.recovery import RecoveryIdentity, RecoveryRecord

    recovery_identity = RecoveryIdentity(
        workspace_id=runtime.session_plan.manifest.name,
        user_id=runtime.config.user_slug,
        user_slug=runtime.config.user_slug,
        surface_type=runtime.config.surface_type,
        surface_id=runtime.config.surface_id,
        memory_scope=cast(MemoryScope, runtime.config.memory_scope),
    )
    started = await runtime.services.recovery_service.begin(
        RecoveryRecord(
            identity=recovery_identity,
            mode_id="shell",
            stage_id="run",
            target_kind="pack_tool",
            target_id="bash_run",
            effect="execute",
            parameters={"command": "hostname"},
            retry_policy="confirm",
        )
    )
    await runtime.services.recovery_service.settle(
        started.work_item_id,
        recovery_identity,
        state="stalled",
        last_error="approval required",
        retry_policy="confirm",
    )
    from runtime.thread import append_thread_entry

    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        mode_id="shell",
        query="run hostname",
        response_text="approval required",
        provenance={
            "pending_approval_request_id": invocation.request_id,
            "pending_approval_recovery_item_id": started.work_item_id,
        },
    )

    # Backdate past the TTL so is_expired() fires inside approve().
    import json as _json  # local to this one-off store backdate

    store_path = Path(runtime.paths.state_dir) / "approvals.json"
    payload = _json.loads(store_path.read_text())
    payload[invocation.request_id]["invocation"]["expires_at"] = "2020-01-01T00:00:00Z"
    store_path.write_text(_json.dumps(payload))

    result = await handle_message(runtime, f"/approve {invocation.request_id}", NullTransport())

    assert "expired" in result.text
    assert "not pending" not in result.text
    assert "fresh approval" in result.text
    settled = await runtime.services.recovery_service.get(started.work_item_id, recovery_identity)
    assert settled is not None
    assert settled.state == "cancelled"


@pytest.mark.asyncio
async def test_approve_rejects_cross_user_request_from_other_session(
    tmp_path: Path,
) -> None:
    """A request id belonging to another user is never approvable."""

    runtime = _make_runtime(tmp_path)
    other_owner = ApprovalIdentity(
        workspace_id=runtime.session_plan.manifest.name,
        user_id="someone-else",
        user_slug="someone-else",
        surface_type=runtime.config.surface_type,
        surface_id=runtime.config.surface_id,
        memory_scope=runtime.config.memory_scope,
        mode_id="shell",
        session_plan_digest=runtime.session_plan.plan_digest,
    )
    invocation = build_approval_invocation(
        target_kind="tool",
        target_id="bash_run",
        pack_id="lisa-pack-bash",
        effect="execute",
        parameters={"command": "hostname"},
        identity=other_owner,
        pack_config={},
    )
    await runtime.services.approval_service.request(invocation)

    result = await handle_message(runtime, f"/approve {invocation.request_id}", NullTransport())

    assert "different session" in result.text
    persisted = await runtime.services.approval_service.get(invocation.request_id)
    assert persisted is not None
    assert persisted.state == "pending"

"""Regression tests for approval interruptions across a mid-turn mode handoff.

When a turn enters ``default`` and the model hands off mid-turn to a mode
that owns an effectful tool (e.g. ``coding``), the tool's
``ApprovalRequired`` carries the *owning* mode in its identity. The host
advances ``current_mode_id`` from ``TurnResult.handoff_to`` — so the
approval-resume path (which rebuilds the identity from
``current_mode_id``) can only match the pending record when the turn
result propagates the owning mode as ``handoff_to``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from kernel.approval import InMemoryApprovalService
from kernel.ipsar_fsm import synthesize_routing_transitions
from kernel.models import (
    ToolDefinition,
    ToolExecutionContext,
    ToolResult,
    WorkflowModeConfig,
)
from kernel.turn_runner import TurnHooks, run_turn
from kernel.turn_hooks import (
    ModelHooks,
    ContextHooks,
    OperationHooks,
    LifecycleHooks,
    WorkflowHooks,
    HostServiceHandles,
    TurnIdentity,
    RecallHooks,
    ToolHooks,
)


class _Parameters(BaseModel):
    path: str


@dataclass(frozen=True)
class _Completion:
    content: str = ""
    tool_calls: list[dict[str, Any]] | None = None


def _mode_config(mode_id: str, *, satisfies_intents: list[str] | None = None) -> WorkflowModeConfig:
    return WorkflowModeConfig(
        mode_id=mode_id,
        start_stage_id="none",
        mode_kind="chat",
        routable=True,
        satisfies_intents=list(satisfies_intents) if satisfies_intents else [],
        required_capabilities=[],
        routing_intent_tags=[],
        routing_intent_keywords={},
    )


@pytest.mark.asyncio
async def test_run_turn_approval_after_mid_turn_handoff_propagates_owning_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An approval paused in a mid-turn-handed-off mode surfaces that mode
    as ``handoff_to`` so the host advances ``current_mode_id`` before the
    user resumes.

    Regression for the 2026-07-28 glm-5.2 incident: the turn entered
    ``default``, handed off to ``coding``, and ``coding_apply_patch``
    raised ``ApprovalRequired``. The exception path returned
    ``handoff_to=None``, so ``current_mode_id`` stayed ``default``; the
    subsequent ``/approve`` rebuilt the identity with ``mode_id=default``
    and the exact-match in ``_identity_matches`` failed, stranding the
    approval. The host then re-ran the whole workflow as a fresh model
    turn.
    """
    from unittest.mock import patch

    executed = False

    async def apply_patch(_ctx: ToolExecutionContext) -> ToolResult:
        nonlocal executed
        executed = True
        return ToolResult(output_text="unexpected")

    tool_definition = ToolDefinition(
        tool_id="coding_apply_patch",
        description="Apply a patch.",
        pack_id="lisa-pack-coding",
        parameters_schema=_Parameters,
        effect="write",
        approval_required=True,
        approval_preview="apply the patch",
        implementation_module=SimpleNamespace(execute=apply_patch),
    )
    monkeypatch.setattr(
        "kernel.tools.registry._TOOL_CATALOG", {"coding_apply_patch": tool_definition}
    )

    patched_catalog = {
        "default": _mode_config("default"),
        "coding": _mode_config("coding", satisfies_intents=["write"]),
    }

    call_log: list[str] = []
    # State-driven completion: the first model call (in default) hands off
    # to coding; the second (in coding) calls the approval-gated tool.
    # Parsing the system prompt is fragile because the stub catalog's
    # ``get_mode_system_intro`` returns empty and the handoff-context
    # block is keyed on the *from* mode, not the target.
    handed_off = False

    async def complete(**kwargs: Any) -> _Completion:
        nonlocal handed_off
        _ = kwargs
        if not handed_off:
            call_log.append("model_call:default")
            handed_off = True
            return _Completion(
                tool_calls=[
                    {
                        "id": "call-default-to-coding",
                        "function": {
                            "name": "transition_stage",
                            "arguments": json.dumps(
                                {"target_stage_id": "coding", "rationale": "write intent"}
                            ),
                        },
                    }
                ]
            )
        call_log.append("model_call:coding")
        return _Completion(
            tool_calls=[
                {
                    "id": "call-coding-1",
                    "function": {
                        "name": "coding_apply_patch",
                        "arguments": '{"path":"src/a.py"}',
                    },
                }
            ]
        )

    service = InMemoryApprovalService()
    hooks = TurnHooks(
    models=ModelHooks(
        complete=complete,
    ),
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
)

    # Patch the binding where the behavior is defined: the transition
    # interception's routability gate (``is_routable_mode``) and the FSM
    # target list (``get_routing_targets`` →
    # ``synthesize_routing_transitions``) read the catalog through
    # ``kernel.ipsar_fsm.get_workflow_catalog``, NOT the turn_runner
    # re-export. Patching only ``kernel.turn_runner.get_workflow_catalog``
    # left the real catalog (``default``-only in tests) in charge of the
    # gate: the handoff fell through to ``Unknown transition target`` and
    # the approval minted in the entry mode — the exact-match test below
    # then compared ``default`` vs ``coding`` and failed.
    transitions = synthesize_routing_transitions(patched_catalog)
    with (
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=patched_catalog),
        patch(
            "kernel.ipsar_fsm.synthesize_routing_transitions",
            lambda catalog=None: (
                synthesize_routing_transitions(patched_catalog) if catalog is None else transitions
            ),
        ),
        patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
    ):
        result = await run_turn(mode_id="default", query="patch the file", hooks=hooks)

    # The tool paused before the side effect.
    assert executed is False
    assert result.pending_approval is not None
    assert result.pending_approval.target_id == "coding_apply_patch"
    # The owning mode is captured on the pending invocation (coding, not
    # the turn's entry mode default).
    assert result.pending_approval.identity.mode_id == "coding"
    # The turn result propagates the owning mode as handoff_to so the
    # host advances current_mode_id before the user resumes. Without
    # this, /approve rebuilds the identity with mode_id=default and the
    # exact-match stranding the approval.
    assert result.handoff_to == "coding"
    # The model actually handed off into coding before pausing.
    assert "model_call:default" in call_log
    assert "model_call:coding" in call_log


@pytest.mark.asyncio
async def test_run_turn_approval_in_entry_mode_leaves_handoff_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An approval paused in the turn's entry mode (no mid-turn handoff)
    does not fabricate a handoff. ``handoff_to`` stays ``None`` and the
    host leaves ``current_mode_id`` unchanged — the resume identity will
    match because the invocation was captured in the same entry mode.
    """

    async def write_file(_ctx: ToolExecutionContext) -> ToolResult:
        return ToolResult(output_text="unexpected")

    tool_definition = ToolDefinition(
        tool_id="write_file",
        description="Write a file.",
        pack_id="lisa-pack-coding",
        parameters_schema=_Parameters,
        effect="write",
        approval_required=True,
        approval_preview="write the file",
        implementation_module=SimpleNamespace(execute=write_file),
    )
    monkeypatch.setattr("kernel.tools.registry._TOOL_CATALOG", {"write_file": tool_definition})

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
        query="write the file",
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
    assert result.pending_approval.identity.mode_id == "default"
    assert result.handoff_to is None

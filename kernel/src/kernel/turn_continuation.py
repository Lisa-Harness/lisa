"""Data-only suspension of the existing turn loop."""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from kernel.config import settings
from kernel.execution_policy import ExecutionPolicy
from kernel.models import TurnBudget
from kernel.system_segments import SystemPromptSegments
from kernel.turn_hooks import TurnIdentity
from kernel.turn_phases._turn_state import _TurnState

_STATE = TypeAdapter(_TurnState)
# Pauses at an iteration boundary: no pending tool call and no approval.
PAUSE_REASONS = frozenset({"cap", "pause"})
_SEGMENTS = TypeAdapter(SystemPromptSegments)


class TurnContinuation(BaseModel):
    """A saved dispatch/iteration boundary, not a new request or workflow."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1] = 1
    turn_id: str = Field(default_factory=lambda: uuid4().hex)
    identity: TurnIdentity
    mode_id: str
    query: str
    state_json: str
    context_json: str
    mode_config_json: str
    segments_json: str
    preferred_model_id: str | None
    resident_model_id: str = ""
    max_iterations: int
    per_step_max_iterations: int
    per_step_max_tool_calls: int
    turn_budget: TurnBudget | None
    execution_policy: ExecutionPolicy | None
    suppress_routing_hint: bool
    delegate_depth: int
    mode_instruction: str
    chain_guidance: str = ""
    stalled_guidance: str = ""
    routing_hint: str = ""
    pre_handoff_text: str = ""
    pending_calls: list[dict[str, Any]] = Field(default_factory=list)
    pending_index: int = 0
    pending_tool_name: str = ""
    pending_tool_arguments: dict[str, Any] = Field(default_factory=dict)
    pending_tool_kind: str = ""
    pending_tool_call_id: str = ""
    pending_recovery_item_id: str | None = None
    pending_recovery_retry_policy: str = ""
    approval_request_id: str = ""
    reason: str = "approval"
    active_wall_ms: float = 0
    # The allowance the turn started with; each cap resume grants it again.
    base_max_iterations: int = 0
    base_turn_budget: TurnBudget | None = None

    @property
    def requires_approval(self) -> bool:
        return self.reason not in PAUSE_REASONS

    def restore_state(self) -> _TurnState:
        return _STATE.validate_json(self.state_json)

    def restore_segments(self) -> SystemPromptSegments:
        return _SEGMENTS.validate_json(self.segments_json)

    def grant_auto_approval_allowance(self) -> TurnContinuation:
        """An explicit operator decision grants another allowance without resetting usage."""
        state = self.restore_state()
        state.auto_approval_limit += settings.max_auto_approved_actions
        return self.model_copy(update={"state_json": capture_state(state)})


def resume_after_pause(saved: TurnContinuation, note: str) -> TurnContinuation:
    """Continue a paused loop in place with the operator's reply appended.

    A cap pause grants one more of the turn's original allowances plus a
    fresh stage budget; the cumulative counters stay, so every resume is
    bounded and accounted.
    """
    state = saved.restore_state()
    if note.strip() and state.messages is not None:
        state.messages.append({"role": "user", "content": note.strip()})
    update: dict[str, Any] = {}
    if saved.reason == "cap":
        # The pause snapshot leaves the stage-stall counter one below the
        # cap; restored as-is, the resumed loop's first-pass increment
        # re-hits the stuck-in-stage cap before any model call — the
        # operator's note never reaches the model and the harness stamps a
        # second cap diagnostic onto a turn that did no work (the
        # 2026-10-25 turns 970/971 incident). A fresh allowance comes with
        # a fresh stage budget, matching a new turn's counter reset.
        state.step_iteration = 0
        state.step_tool_calls = 0
        base = saved.base_turn_budget or saved.turn_budget
        update["max_iterations"] = saved.max_iterations + (
            saved.base_max_iterations or saved.max_iterations
        )
        if saved.turn_budget is not None and base is not None:
            update["turn_budget"] = saved.turn_budget.model_copy(
                update={
                    field: getattr(saved.turn_budget, field) + getattr(base, field)
                    for field in TurnBudget.model_fields
                    if getattr(saved.turn_budget, field)
                }
            )
    update["state_json"] = capture_state(state)
    return saved.model_copy(update=update)


def capture_state(state: _TurnState) -> str:
    """Exclude live model descriptors and already-archived request snapshots."""
    return _STATE.dump_json(
        state,
        exclude={"mode_model_def", "effective_model", "pending_router_decision", "model_requests"},
    ).decode("utf-8")


def capture_segments(segments: SystemPromptSegments) -> str:
    return _SEGMENTS.dump_json(segments).decode("utf-8")

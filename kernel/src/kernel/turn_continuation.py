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

    def restore_state(self) -> _TurnState:
        return _STATE.validate_json(self.state_json)

    def restore_segments(self) -> SystemPromptSegments:
        return _SEGMENTS.validate_json(self.segments_json)

    def grant_auto_approval_allowance(self) -> TurnContinuation:
        """An explicit operator decision grants another allowance without resetting usage."""
        state = self.restore_state()
        state.auto_approval_limit += settings.max_auto_approved_actions
        return self.model_copy(update={"state_json": capture_state(state)})


def capture_state(state: _TurnState) -> str:
    """Exclude live model descriptors and already-archived request snapshots."""
    return _STATE.dump_json(
        state,
        exclude={"mode_model_def", "effective_model", "pending_router_decision", "model_requests"},
    ).decode("utf-8")


def capture_segments(segments: SystemPromptSegments) -> str:
    return _SEGMENTS.dump_json(segments).decode("utf-8")

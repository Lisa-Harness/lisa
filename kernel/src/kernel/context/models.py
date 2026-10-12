"""Immutable contracts for the session-owned context mapper."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kernel.session_plan import ContextPlan

__all__ = [
    "ContextBudgetSnapshot",
    "ContextCandidate",
    "ContextDecision",
    "ContextDecisionTrace",
    "ContextMapperRequest",
    "ContextMapperResult",
    "ContextSelection",
]

ContextProvenance = Literal[
    "authoritative",
    "inferred",
    "speculative",
    "user_asserted",
    "system_observed",
    "recalled_memory",
    "diagnostic",
]
ContextSensitivity = Literal["public", "private", "restricted"]
ContextConflictRisk = Literal["low", "medium", "high"]
ContextDecisionOutcome = Literal["selected", "dropped", "gated"]


class _FrozenContextModel(BaseModel):
    """Strict immutable base for mapper-facing contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class ContextCandidate(_FrozenContextModel):
    """One raw, prompt-eligible context item before mapping or ranking."""

    candidate_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_ref: str = Field(min_length=1)
    source_kind: str = Field(min_length=1)
    evidence_type: str = ""
    title: str = ""
    content: str = Field(min_length=1)
    pack_id: str = ""
    provenance: ContextProvenance = "authoritative"
    sensitivity: ContextSensitivity = "private"
    authority: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    salience: float = Field(default=0.5, ge=0.0, le=1.0)
    continuity: float = Field(default=0.0, ge=0.0, le=1.0)
    user_priority: float = Field(default=0.0, ge=0.0, le=1.0)
    collected_at: datetime | None = None
    expires_at: datetime | None = None
    tags: tuple[str, ...] = ()
    mode_ids: tuple[str, ...] = ()
    stage_ids: tuple[str, ...] = ()
    conflict_key: str = ""
    conflict_risk: ContextConflictRisk = "low"
    explicitly_invoked: bool = False
    required: bool = False
    pinned: bool = False
    token_estimate: int = Field(default=0, ge=0)
    content_hash: str = ""

    @model_validator(mode="after")
    def _validate_time_bounds(self) -> ContextCandidate:
        if (
            self.collected_at is not None
            and self.expires_at is not None
            and self.expires_at < self.collected_at
        ):
            raise ValueError("expires_at cannot precede collected_at")
        return self


class ContextBudgetSnapshot(_FrozenContextModel):
    """The model-aware budget available to mapper-selected context."""

    model_context_window_tokens: int = Field(ge=1)
    output_reserve_tokens: int = Field(ge=0)
    safety_margin_tokens: int = Field(ge=0)
    available_context_tokens: int = Field(ge=0)
    used_context_tokens: int = Field(default=0, ge=0)
    lane_used_tokens: dict[str, int] = Field(default_factory=dict)


class ContextDecision(_FrozenContextModel):
    """One explainable mapper decision, including gate and score details."""

    candidate_id: str
    source_ref: str
    outcome: ContextDecisionOutcome
    matched_rule_ids: tuple[str, ...] = ()
    lane_id: str = ""
    gate_passed: bool = False
    component_scores: dict[str, float] = Field(default_factory=dict)
    base_score: float = 0.0
    mmr_score: float | None = None
    token_estimate: int = 0
    allocated_tokens: int = 0
    selection_order: int | None = None
    drop_reason: str = ""
    duplicate_of: str = ""


class ContextSelection(_FrozenContextModel):
    """One selected candidate in its mapper-assigned lane."""

    candidate: ContextCandidate
    lane_id: str
    token_count: int = Field(ge=0)
    base_score: float
    mmr_score: float
    selection_order: int = Field(ge=0)


class ContextDecisionTrace(_FrozenContextModel):
    """Persistable explanation of all mapper inclusion and drop decisions."""

    decisions: tuple[ContextDecision, ...]
    budget: ContextBudgetSnapshot
    selected_candidate_ids: tuple[str, ...] = ()


class ContextMapperRequest(_FrozenContextModel):
    """All static-plan and live-turn input required for one mapper pass."""

    context_plan: ContextPlan
    query: str = ""
    mode_id: str = "default"
    stage_id: str = ""
    candidates: tuple[ContextCandidate, ...] = ()
    active_packs: frozenset[str] = frozenset()
    allowed_sensitivities: frozenset[ContextSensitivity] = frozenset(
        {"public", "private", "restricted"}
    )
    model_context_window_tokens: int = Field(ge=1)
    now: datetime | None = None

    @model_validator(mode="after")
    def _validate_candidate_ids(self) -> ContextMapperRequest:
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("context candidate IDs must be unique")
        return self


class ContextMapperResult(_FrozenContextModel):
    """Selected, provenance-rendered context and its full decision trace."""

    selections: tuple[ContextSelection, ...] = ()
    rendered_context: str = ""
    trace: ContextDecisionTrace

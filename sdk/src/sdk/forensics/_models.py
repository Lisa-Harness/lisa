"""Typed forensic evidence records.

Every field here is traceable to a file on disk: a thread entry, a
``turn.decision`` line, a ``tool.telemetry`` line, or a ``token_usage`` line.
Nothing in this module infers, summarizes, or guesses — the join
(:mod:`sdk.forensics._join`) and the renderer
(:mod:`sdk.forensics.render`) are where interpretation happens, and they are
kept separate so a claim can always be traced back to a record.

The central concept is *attribution*. A record carries a ``turn_index`` when
it can be joined to a thread turn, and ``None`` when it cannot. ``None`` is
meaningful: off-turn slice work and thread-less hosts produce telemetry that
belongs to no turn. Collapsing ``None`` into ``0`` would attribute background
work to the thread's first turn, which is the single most damaging failure
mode for a forensic reader — it manufactures plausible-looking evidence.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "BundleSources",
    "CostRecord",
    "DecisionRecord",
    "DispatchRecord",
    "ForensicBundle",
    "TurnEvidence",
]


class _Record(BaseModel):
    """Shared configuration for evidence records."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=False)


class DecisionRecord(_Record):
    """One ``turn.decision`` line: why the loop chose what it chose."""

    turn_index: int | None = None
    iteration: int = 0
    turns_in_stage: int = 0
    mode_id: str = ""
    action_type: str = ""
    action_reason: str = ""
    total_tool_calls: int = 0
    write_receipts: int = 0
    read_receipts: int = 0
    logged_at: str = ""

    @property
    def is_blocking(self) -> bool:
        """Return whether the action halted progress (yield/finish with reason)."""
        return bool(self.action_reason) and "finish" in self.action_type.lower()


class DispatchRecord(_Record):
    """One ``tool.telemetry`` line: what ran, how long, and whether it failed."""

    turn_index: int | None = None
    target_id: str = ""
    target_type: str = ""
    status: str = ""
    duration_ms: int = 0
    error_message: str = ""
    logged_at: str = ""

    @property
    def failed(self) -> bool:
        """Return whether the dispatch failed or timed out."""
        return self.status in {"error", "timeout"}


class CostRecord(_Record):
    """One ``token_usage`` line, reduced to what a forensic read needs."""

    turn_index: int | None = None
    mode_id: str = ""
    model_id: str = ""
    call_site: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    emitted_at: str = ""

    @property
    def billed_tokens(self) -> int:
        """Return provider-total tokens minus cache hits (the billable view)."""
        return max(0, self.total_tokens - self.cached_tokens)


class TurnEvidence(_Record):
    """One thread turn plus every record that joins to it.

    ``turn_index`` is the thread's own key and is never ``None`` for an entry
    read out of the store; the optional sub-records are what may be missing.
    ``provisional_turn`` is the index the turn ran under when it differs
    from the committed one (overlapping turns commit in completion order);
    the turn's in-flight telemetry carries that index.
    """

    turn_index: int
    provisional_turn: int | None = None
    timestamp_utc: str = ""
    mode_id: str = ""
    query_preview: str = ""
    response_preview: str = ""
    handoff_to: str = ""
    transitioned_to: str = ""
    blocked: bool = False
    block_origin: str = ""
    content_source: str = ""
    provenance_source: str = ""
    task_id: str = ""
    receipt_capability_ids: list[str] = Field(default_factory=list)
    decisions: list[DecisionRecord] = Field(default_factory=list)
    dispatches: list[DispatchRecord] = Field(default_factory=list)
    costs: list[CostRecord] = Field(default_factory=list)

    @property
    def telemetry_turn_index(self) -> int:
        """Return the index the turn's in-flight telemetry was stamped with."""
        return self.turn_index if self.provisional_turn is None else self.provisional_turn

    @property
    def total_tokens(self) -> int:
        """Return the turn's summed token total across model calls."""
        return sum(record.total_tokens for record in self.costs)

    @property
    def total_cost_usd(self) -> float:
        """Return the turn's summed cost estimate."""
        return round(sum(record.cost_usd for record in self.costs), 6)

    @property
    def failed_dispatches(self) -> list[DispatchRecord]:
        """Return the dispatches that failed or timed out."""
        return [record for record in self.dispatches if record.failed]

    @property
    def has_cost_evidence(self) -> bool:
        """Return whether any cost record joined to this turn."""
        return bool(self.costs)


class BundleSources(BaseModel):
    """Which evidence files were read, and which were missing.

    A forensic read must never present an absent file as an empty one: "no
    errors" and "I could not read the error log" are different answers, and
    only one of them is safe to act on.

    Deliberately mutable while a read is in progress (the reader records each
    source as it opens it) and frozen once it is published inside a
    :class:`ForensicBundle`.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    thread_store: bool = False
    decisions_log: bool = False
    token_usage_log: bool = False
    runtime_log: bool = False
    missing: list[str] = Field(default_factory=list)


class ForensicBundle(_Record):
    """One forensic read: the requested turns and their joined evidence."""

    generated_at: str
    turns: list[TurnEvidence] = Field(default_factory=list)
    sources: BundleSources = Field(default_factory=BundleSources)
    #: Records whose ``turn_index`` matched no requested turn. Surfaced rather
    #: than dropped: an unattributable record is itself a finding.
    unattributed_decisions: list[DecisionRecord] = Field(default_factory=list)
    unattributed_dispatches: list[DispatchRecord] = Field(default_factory=list)
    #: Log lines at warning/error level inside the window, newest last.
    log_errors: list[dict[str, Any]] = Field(default_factory=list)
    #: Human-readable notes about bounded or degraded reads (truncation,
    #: missing files, caps applied). Never silently omitted.
    notes: list[str] = Field(default_factory=list)

    @property
    def is_degraded(self) -> bool:
        """Return whether the bundle lost a source or hit a cap."""
        return bool(self.sources.missing) or bool(self.notes)

    @property
    def total_tokens(self) -> int:
        """Return the summed token total across every attributed turn."""
        return sum(turn.total_tokens for turn in self.turns)

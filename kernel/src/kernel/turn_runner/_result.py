"""TurnResult — the result type of one ``run_turn()`` call."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from kernel.approval import ApprovalInvocation
from kernel.models import BlockDiagnostic, EffectReceipt, ForkChildResult, YieldAction
from kernel.slice import SliceOutcome


# ── Public types ───────────────────────────────────────────────────────────


@dataclass
class TurnResult:
    """Result of one call to `run_turn()`."""

    response_text: str
    tool_call_count: int = 0
    called_tool_names: set[str] = field(default_factory=set)
    """Names of every tool actually dispatched this turn. Used by the
    fork-child validator to confirm an action mode's child invoked at
    least one of its authored action tools, not just read-only fetches."""
    effect_receipts: list[EffectReceipt] = field(default_factory=list)
    """``EffectReceipt`` for every successfully-dispatched capability this
    turn. Write/external_write receipts satisfy a mode's write contract;
    read/execute receipts prove real read-only work. The child-turn
    completion contract checks these receipts (not tool-name heuristics) so
    a child that called a ``mode_allowlist``-admitted write operation is
    correctly counted. See ``kernel.models.EffectReceipt``."""
    iteration_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    #: Provider-reported prompt-cache *creation/population* tokens (Anthropic
    #: ``cache_creation_input_tokens``). 0 when the provider does not report it.
    cache_creation_tokens: int = 0
    duration_ms: float = 0.0
    latency_ms: int = 0
    slice_outcomes: list[SliceOutcome] = field(default_factory=list)
    """SliceOutcome records for every opportunistic slice that ran
    during this turn. Populated only when the host has wired the
    corresponding slice slot. Empty list when opportunistic
    house-keeping is off (``slice_tuning=None`` or all triggers set
    to ``never``)."""
    blocked: bool = False
    """True when an input or output guard blocked the turn. Input blocks
    short-circuit before the model call; output blocks replace the response
    with a safe fallback and emit a ``guardrail.audit`` event."""
    block_origin: str = ""
    """When ``blocked`` is True, which gate produced the block:
    ``"capability_gate"``, ``"input_guard"``, or ``"output_guard"``. Empty
    when the block came from a non-guard lifecycle failure (entry/plan/fork).
    The child-turn classifier reads this to build an accurate
    ``BlockDiagnostic`` instead of assuming every block is a capability gate."""
    block_diagnostic: BlockDiagnostic | None = None
    """Structured reason a turn was blocked, when the block came from a gate
    that produces one (currently the capability gate). Carried on the
    ``TurnResult`` so the child-turn classifier and hosts surface the
    structured truth (resolved model, missing capabilities) instead of
    flattening it into response text."""
    transitioned_to: str | None = None
    """When the model called ``transition_stage`` during this turn, the new
    ``stage_id``. None when no transition occurred or the mode is not a
    workflow. The host reads this to persist the updated session and update
    its UI route preview."""
    handoff_to: str | None = None
    """When the model called ``transition_stage`` with a ``mode_id`` target
    (routing FSM handoff), the new ``mode_id``. The host reads this to update
    ``current_mode_id`` and persist the mode switch. None when no handoff
    occurred. See ``docs/ROUTING_RFC_HANDOFF.md``."""
    fork_results: list[ForkChildResult] | None = None
    """When the ``fork`` operation ran this turn, the collected results
    from all parallel child runs. None when no fork occurred. The host
    and subsequent stages read this to synthesize from the fan-out
    outputs."""
    workflow_terminated: bool = False
    """True when the transition moved the session to a terminal stage. The
    loop forces one final no-tools completion using the terminal stage's
    instruction as the system prompt so the model produces a closing summary."""
    operator_exit: Literal["", "paused", "stopped", "terminated"] = ""
    """When the turn ended via an operator signal (pause/stop/terminate) or a
    cap path that degrades to one, the discriminator the transport reads to
    decide whether to return the prompt.

    - ``""``: not an operator exit (normal finish, model yield, or error).
    - ``"paused"``: operator paused OR a cap path froze the turn. The
      transport holds the turn open — no prompt returned. Resume re-enters
      the same turn against the preserved ``WorkflowPosition`` with a fresh
      iteration budget.
    - ``"stopped"``: operator stopped. Same checkpoint as pause, but the
      transport returns the prompt.
    - ``"terminated"``: operator terminated OR an action-mode cap last
      resort. The workflow is marked complete; the transport returns the
      prompt.
    """
    yielded: YieldAction | None = None
    """The unified-lifecycle ``YieldAction`` when this turn terminated by
    yielding (asking the user a question), ``None`` when it finished cleanly.

    Surfaced to the host so the transport can distinguish a yield from a
    finish and render the question with its context — ``YieldAction.context``
    carries the reason ("model-emitted yield", "turn budget exhausted …",
    "stuck in stage …", "recovery record blocks …") and ``target_mode_id``
    names the mode the yield routes to. ``TurnResult.yielded`` is the
    non-durable mirror for in-process transports; the thread entry records
    ``yielded=True`` + ``yield_question`` for durable resume via ``/resume``.
    See :class:`kernel.models.YieldAction`."""
    pending_approval: ApprovalInvocation | None = None
    """The exact action that paused this turn, when an approval is required."""
    per_model_usage: list[dict[str, Any]] = field(default_factory=list)
    """Per-model token breakdown for every model that was actually called
    during this turn. Each entry is keyed by the model's ``model_id`` and
    carries ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens`` /
    ``cached_tokens`` / ``reasoning_tokens`` / ``duration_ms`` / ``latency_ms``.
    Entries for the same model_id across iterations are merged into one row.
    The host's token accounting reads this to populate ``model_token_totals``
    so ``/tokens`` reports per-model usage rather than attributing the whole
    turn to the default mode's resolved model. Empty when no model was
    called (e.g. input-guard block, pure operation mode)."""
    per_mode_usage: list[dict[str, Any]] = field(default_factory=list)
    """Per-mode token breakdown for every mode that actually ran during this
    turn. Mirrors ``per_model_usage`` but keyed by ``mode_id``. The host's
    token accounting reads this to populate ``mode_token_totals`` so
    ``/tokens`` reports the actual mode(s) that did the work rather than
    attributing the whole turn to the entry mode — the previous behavior
    misattributed kernel-driven recursive handoffs (plan → search → coding)
    to the entry mode. Empty when ``record_model_usage`` was never called
    with a mode_id, or no model was called this turn."""

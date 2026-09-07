"""Mutable turn-state container and closure-free helpers for ``_run_turn_inner``.

This module owns the per-turn mutable state that was previously captured by
nested closures inside ``kernel.turn_runner._run_turn_inner``.  The five
closures (``_emit_stage``, ``_mcp_tool_executor``, ``_record_model_usage``,
``_budget_forces_final``, ``_resolve_iteration_model``) have been promoted to
module-level functions that accept the state explicitly as a ``_TurnState``
dataclass.  This makes them independently testable and eliminates the closure
capture of mutable parent state.

DESIGN — explicit state passing, not global:
    ``_TurnState`` is instantiated once per turn and threaded through every
    helper call.  No module-level mutable singletons.  The turn runner
    mutates fields on the instance (e.g. ``state.total_tool_calls += 1``) and
    reads them back in the loop.  The state object is never shared across
    turns or across concurrent fork children.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from kernel.models import EffectReceipt, ForkChildResult, OperationResult, ToolResult
from kernel.seams.protocols import RouterModelRef
from kernel.tools.registry import execute_tool

if TYPE_CHECKING:
    from kernel.models import TurnBudget
    from kernel.tools.mcp_client import MCPToolPool

logger = logging.getLogger(__name__)


@dataclass
class ForcedAction:
    """A deterministic routing decision the kernel will execute itself if
    the model refuses to.

    One enforcer for both shapes (Phase 5): a fan-out (``kind == "fork"``,
    several targets, expected tool ``fork``) or a single handoff
    (``kind == "transition"``, one target, expected tool
    ``transition_stage``). The dispatch loop increments
    ``non_compliant_dispatches`` on every tool call that is not
    ``expected_tool``; at ``forced_action_non_compliant_cap`` the kernel
    synthesizes the call itself. The kind is *derived* from the target
    list — ``targets`` is the only state.
    """

    targets: list[str]
    non_compliant_dispatches: int = 0

    @property
    def kind(self) -> str:
        """``"fork"`` for a multi-target fan-out, else ``"transition"``."""
        return "fork" if len(self.targets) > 1 else "transition"

    @property
    def expected_tool(self) -> str:
        """The tool the model is expected to call for this routing shape."""
        return "fork" if self.kind == "fork" else "transition_stage"

    @property
    def target_csv(self) -> str:
        """The targets as a comma-joined string (fork payload building)."""
        return ",".join(self.targets)


@dataclass
class _TurnState:
    """Mutable per-turn counters and accumulators.

    Instantiated once at the top of ``_run_turn_inner`` and threaded through
    every helper call.  Fields are intentionally mutable (no ``frozen=True``)
    — the turn loop updates them in place.
    """

    # ── Token / usage accumulators ───────────────────────────────────
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cached_tokens: int = 0
    total_reasoning_tokens: int = 0
    total_duration_ms: float = 0.0
    total_latency_ms: int = 0
    total_tool_calls: int = 0
    per_model_usage: dict[str, dict[str, Any]] = field(default_factory=dict)
    per_mode_usage: dict[str, dict[str, Any]] = field(default_factory=dict)

    # ── Budget caps (stamped on every emitted stage payload) ───────────
    # Set once at turn start from ``kernel.config.Settings`` and
    # ``TurnBudget``. The transport reads them to render graphical
    # counters (pip rows, sparklines) against the correct denominator.
    # 0 means "unlimited" (``TurnBudget`` convention); the transport
    # falls back to a saturating counter in that case.
    max_iterations: int = 0
    """Global per-turn iteration cap (``max_tool_iterations``). 0 = unlimited."""
    max_tool_calls: int = 0
    """Per-turn tool-call cap (``TurnBudget.max_tool_calls``). 0 = unlimited."""
    max_handoffs: int = 0
    """Per-turn mode-handoff cap (``max_handoffs_per_turn``). 0 = unlimited."""

    # ── Router-relevant state ────────────────────────────────────────
    last_tool_result_text: str | None = None
    # The last in-process ``OperationResult`` dispatched this turn, or
    # ``None`` when the last dispatch was an MCP tool or a low-level
    # pack tool. The cap-pause diagnostic reads ``extra`` for structured
    # context (e.g. continuation attempt counts, truncation reason) so
    # the user sees *why* an operation kept failing, not just *that* the
    # turn paused. Only set for the in-process operation dispatch branch.
    last_operation_result: OperationResult | None = None
    # Count of tools that have returned a result this turn. Drives
    # ``RouterRequest.has_tool_result`` so router rules see turn-level
    # "any tool returned" state rather than just the last result text
    # (which may be empty even when earlier tools returned results).
    tool_results_received: int = 0
    required_capabilities: tuple[str, ...] = ()
    # Metadata of the model effective in the preceding iteration. Set after
    # the first completion; on iteration 1 the router receives the static
    # preferred model via RouterRequest.effective_model.
    effective_model: Any = None
    # When the router returns a terminal action (finish / transition_to),
    # resolve_iteration_model stores it here so the turn loop can act on
    # it after the model-selection path. Cleared at the start of each
    # iteration.
    pending_router_decision: Any = None

    # ── Content / outcome accumulators ───────────────────────────────
    content: str = ""
    transitioned_to: str | None = None
    handoff_to: str | None = None
    fork_results: list[ForkChildResult] | None = None
    workflow_terminated: bool = False
    iteration: int = 0
    # ── Per-step iteration sub-budget ────────────────────────────────
    # Iterations and tool calls consumed by the *current* workflow step
    # only, reset to 0 on every successful ``transition_stage`` (the same
    # site as the durable instance advancement). The global ``iteration``
    # counter is NOT reset on handoffs — it bounds the whole turn — but
    # ``step_iteration`` bounds a single step so a long chained workflow
    # (default→plan→search→ponder→coding→markdown) does not starve its
    # final step on the shared ``max_tool_iterations`` budget. When the
    # per-step cap binds before the global cap, the kernel synthesizes a
    # handoff to the next remaining step instead of running the generic
    # empty-content final completion (see Phase 3 of the durable-workflow
    # plan, 2026-07-29). When no ``WorkflowInstance`` is wired, these
    # counters are unused and the global cap alone governs the loop.
    step_iteration: int = 0
    """Model-completion iterations consumed by the current workflow step.
    Reset to 0 on every ``transition_stage``; bounded by
    ``per_step_max_iterations`` from ``kernel.config.Settings``."""
    step_tool_calls: int = 0
    """Tool calls dispatched during the current workflow step. Reset to 0
    on every ``transition_stage``; bounded by ``per_step_max_tool_calls``
    as a secondary guard against a model stuck in a tool-calling rut."""
    call_ledger_counts: dict[str, int] = field(default_factory=dict)
    """Snapshot of the per-target ``CallLedger.counts`` for the budget
    briefing. Synced from the ledger after each ``ledger.record`` so the
    per-iteration budget briefing can report per-target remaining calls
    without holding a reference to the ledger itself."""

    # ── Chain routing accumulator ────────────────────────────────────
    chain_satisfied: set[str] = field(default_factory=set)

    # ── Deferred ad-hoc workflow-instance steps ───────────────────────
    # Compiled at turn start from the routing hint's multi-step chain,
    # but NOT persisted until the first successful ``transition_stage``
    # handoff. Deferring persistence makes ad-hoc instance creation
    # transactional with the first real handoff: if the turn fails
    # before any handoff succeeds (router TypeError, fork NameError,
    # entry-op failure, user Ctrl+C), the steps die with ``_state`` and
    # no zombie ``WorkflowInstance`` is left in the durable store to
    # hijack the next turn (the 2026-08-06 incident: a failed fork at
    # 17:58 persisted an instance that drove every subsequent turn's
    # "plan-driven next step" against the user's actual intent). The
    # 2026-08-01 yield→resume fix is preserved because a yield happens
    # mid-turn, after at least one handoff has already persisted the
    # instance. Empty tuple when no ad-hoc chain was compiled or after
    # the deferred persist has fired.
    pending_adhoc_steps: tuple[Any, ...] = field(default_factory=tuple)

    # ── Hard-failure / loop-guard state ───────────────────────────────
    entry_failed: bool = False
    """True when an entry operation for a handoff target failed this turn
    (e.g. ``web_search`` returned no content). Once set, the turn must not
    continue in the data-less target mode — the model would drift into an
    action mode and hallucinate a solution (the 2026-07-25 ``crew/``
    incident). The handoff handler sets this and breaks the loop with a
    deterministic user-facing failure instead of ``continue``-ing."""
    entry_failed_mode: str = ""
    """The mode_id whose entry operation failed, for the user-facing
    failure message (e.g. ``"search"``)."""
    entry_failed_reason: str = ""
    """Short human-readable failure reason from the caught exception,
    surfaced to the user alongside the mode name."""
    handoff_count: int = 0
    """Number of mid-turn mode handoffs executed this turn. Bounded by
    ``max_handoffs_per_turn`` (``kernel.config.Settings``) as a hard
    backstop against routing-hint loops that the ``chain_satisfied``
    accumulator fails to break (e.g. when a pack omits
    ``routing_intent_tags`` or ``_detect_query_intents`` over-detects)."""
    last_hint_text: str = ""
    """The most recently injected routing-hint text. Used by the
    duplicate-hint dedup guard: when the recomputed post-handoff hint is
    byte-identical to the previous one for >=2 consecutive handoffs, the
    injection is suppressed and the max-hops guard takes over."""
    duplicate_hint_count: int = 0
    """Consecutive count of identical routing hints. Resets to 0 on a
    distinct hint; when it reaches ``duplicate_hint_limit``
    (``kernel.config.Settings``) the hint is suppressed."""
    hint_target_history: list[str] = field(default_factory=list)
    """Ordered list of routing-hint target mode ids injected this turn
    (e.g. ``["ponder", "markdown", "ponder"]``). Used by the hint-cycle
    guard to detect A→B→A 2-cycles that the byte-identical dedup misses:
    a 2-cycle produces two *different* hint strings, so
    ``duplicate_hint_count`` never increments. When a cycle is detected
    the hint is suppressed so the ``handoff_count`` backstop fires
    immediately instead of wasting model round-trips on the oscillation."""
    pending_planned_handoff: str = ""
    """A ``target_stage_id`` produced by a planner action mode's entry
    operation (e.g. ``plan_workflow``) — either at Phase 1 (turn start) or
    during a mid-turn handoff. When set, the turn loop synthesizes a
    ``transition_stage`` tool call for this target at the top of the next
    iteration — before calling the model — so the kernel drives the next
    handoff deterministically instead of begging the model to call
    ``transition_stage`` (the 2026-07-28 stall: the model in ``plan`` mode
    read the plan result and stopped). Cleared after the synthesized
    dispatch."""
    plan_prefix: str = ""
    """The rendered plan block (``## Plan ...``) staged by the Phase-1
    planner path when the planner returned a steps array. Prepended to the
    turn's final content so the user sees the plan before the first step's
    output — the in-loop replacement for the deleted recursive call's
    ``step_result.response_text`` prepend."""
    plan_failed: bool = False
    """True when a planner action mode's entry operation returned an empty
    ``target_stage_id`` (parse failure, model error, no wiring). Once set,
    the turn loop must NOT continue routing — the post-handoff hint
    generator would re-hallucinate a target and the model would improvise
    around it (the 2026-07-30 plan→ponder→markdown bounce). The turn
    terminates with a structured recovery prompt that tells the user plan
    failed and offers concrete next steps. Distinct from ``entry_failed``
    (which covers a *non-planner* entry op raising) — a planner that
    returns cleanly with no target is not an exception, it is a routing
    dead end."""
    plan_failed_mode: str = ""
    """The mode_id whose entry planner returned no target, for the
    user-facing recovery prompt (e.g. ``"plan"``)."""
    plan_failed_reason: str = ""
    """Short human-readable failure reason from the planner's
    ``output_text`` (e.g. ``"parse failure"``), surfaced to the user
    alongside the mode name."""
    hint_override_count: int = 0
    """Number of times this turn the model called ``transition_stage`` to a
    target that *differs* from the routing hint's suggested target. When
    this reaches ``hint_override_limit`` (``kernel.config.Settings``) the
    hint layer stops injecting
    further hints for the rest of the turn — the model has demonstrated it
    has its own routing judgment, and continuing to hint is noise the model
    rationally routes around (the 2026-07-30 plan→ponder→markdown bounce:
    the hint said ``discuss`` twice, the model went to ``ponder`` twice)."""
    forced_action: ForcedAction | None = None
    """The one forced-action enforcer (Phase 5). Set whenever a routing
    hint encodes a deterministic routing decision the kernel will execute
    itself if the model refuses: a ``fork:<csv>`` target (kind ``fork``)
    or any single-handoff target — ``plan``, ``search``, ``remember`` …
    (kind ``transition``). The dispatch loop increments
    ``non_compliant_dispatches`` on every tool call that is not the
    expected tool (``fork`` / ``transition_stage``); at
    ``forced_action_non_compliant_cap`` the kernel synthesizes the call
    itself. ``None`` when there is no pending deterministic routing
    decision. Replaces the pre-Phase-5 ``forced_fork_target`` /
    ``non_fork_tool_dispatches`` / ``forced_transition_target`` /
    ``non_transition_tool_dispatches`` quartet — two enforcers that
    differed only in tool name and target arity."""
    called_tool_names: set[str] = field(default_factory=set)
    """Names of every tool actually dispatched this turn (operations,
    pack tools, MCP tools). Used by the fork-child validator to confirm
    that an action mode's child invoked at least one of its authored
    action tools — not just read-only fetches. A child that did 4
    ``fetch_page`` calls but never called ``markdown_forge_document``
    hallucinated its success (the 2026-07-27 follow-on incident)."""
    effect_receipts: list[EffectReceipt] = field(default_factory=list)
    """``EffectReceipt`` instances collected at dispatch time for every
    successfully-dispatched capability. Write/external_write receipts prove
    a mode's write contract was satisfied; read/execute receipts prove real
    read-only work — the completion contract distinguishes an honest
    read-only turn from a hallucinated success by filtering the ledger on
    ``effect`` (not on emptiness). The child-turn completion contract checks
    these receipts (not tool-name heuristics) so a child that called a
    ``mode_allowlist``-admitted write operation invisible to the old
    ``authored_allowed_*``-only detector is correctly counted as having
    completed. See ``kernel.models.EffectReceipt``."""
    forced_work_tool: str = ""
    """A single action-tool id the kernel will synthesize when the model
    returns an empty completion on a plan-self-handoff iteration (the
    2026-08-15 ``no_tools`` block: the routing hint and the plan-driven
    hint both correctly go silent when the next step is the current mode,
    leaving a guidance-needing model with no directive and zero tool
    calls). Set only when the effective model declared ``needs_guidance``
    and the plan's next step mode equals the current mode. Distinct from
    ``forced_action`` (a *routing* decision with ``fork`` /
    ``transition_stage`` as the expected tool); this is a *do-the-work*
    backstop whose expected tool is a mode action tool. Cleared as soon as
    any tool is dispatched this iteration so it cannot loop."""

    # ── Sub-turn (fork / delegate child) contract state ─────────────
    # Set by the caller when this turn is a sub-turn (a fork child, or a
    # delegate child once the delegate operation lands). ``None`` for the
    # parent turn. The budget briefing uses this to emit a distinct
    # ``## Sub-Turn Contract`` block that frames the budget as a
    # write-deadline and names the action tools the child must invoke to
    # avoid a ``no_tools`` abort — the 2026-08-03 incident: a fork child
    # saw only ``iters: 49/50`` and read-looped to the cap without ever
    # writing. The seam is generic (``str``) so delegate adds
    # ``"delegate"`` without reworking the briefing.
    sub_turn_kind: str | None = None
    """``"fork"`` for fork children, ``"delegate"`` for delegate children
    (planned), ``None`` for the parent turn. Drives the sub-turn contract
    block in the budget briefing and the midpoint write-nudge below."""
    action_tool_ids: set[str] = field(default_factory=set)
    """The side-effecting action tools the current mode declares (from
    ``mode_completion_effects``), resolved at turn start for parent and
    sub-turns alike and refreshed on every mid-turn mode handoff. Empty
    for modes with no write contract. Sub-turns: the briefing names these
    as the success criterion; the midpoint nudge fires when the child is
    past its iteration midpoint with zero of these called. Parent turns:
    the Phase-3 completion-contract check (Phase 8) blocks a clean exit
    whose receipt ledger has no entry for these tools (``no_tools`` —
    hallucinated success)."""
    write_nudge_fired: bool = False
    """One-shot guard for the midpoint write-nudge. Set True after the
    nudge is injected once so it does not repeat on every subsequent
    iteration. Reset to False only by a fresh ``_TurnState`` (one per
    turn / fork child), so each child gets exactly one nudge
    opportunity."""
    stalled_write_rescue_fired: bool = False
    """One-shot guard for the stalled-write rescue. Set True when a
    natural completion announced a write (``_STALLED_WRITE_PATTERNS``)
    while the receipt ledger shows zero write attempts — the rescue
    injects a one-shot dispatch directive into the next iteration's
    lifecycle briefing and must not repeat: a model that genuinely
    declines to write gets nudged once, then the turn is accepted
    honestly. Reset to False only by a fresh ``_TurnState``."""
    consecutive_tool_errors: int = 0
    """Count of trailing dispatches to the same tool that returned an
    error (``ToolResult.error`` non-empty, ``OperationResult`` not
    success, or ``MCPToolResult.is_error``). Reset to 0 when a different
    tool is dispatched or the same tool succeeds. Used by the stage
    nudge to tell the model to synthesize what it has and diagnose the
    failure rather than retrying blindly."""
    last_error_tool: str = ""
    """The tool name that most recently returned an error. Paired with
    ``consecutive_tool_errors`` so the stage nudge can name the failing
    tool. Cleared when any tool succeeds."""
    routing_misfire_prose: bool = False
    """Set True by the synthesized-call required-params guard when the
    abort happened on a turn that dispatched ZERO real tools (the
    2026-09-02 executor misroute: a conversational question routed into a
    write-contract mode, the model answered in prose, the write-contract
    backstop synthesized its first action tool with ``{}``, and the guard
    aborted). The post-loop ``_parent_no_tools`` check reads this to
    surface the model's prose with the no-write warning footnote instead
    of replacing it with the hard ``no_tools`` block — the prose may be a
    genuine answer to the user's question, and the guard's own contract
    is "surfacing model prose". A turn that dispatched real tools never
    sets this (failed writes still block strictly)."""

    # ── Within-turn read cache (Level C) ─────────────────────────────
    # Per-turn cache of read-tool call signatures → one-line abbreviations.
    # On a cache hit (the model re-requests the same path/line-range), the
    # dispatch path returns the abbreviation instead of re-running the tool
    # and appending a second full result to ``messages``. This is the
    # within-turn analog of cross-turn conversation compaction: it prevents
    # the linear prompt-token accumulation that burned 1.4M tokens on the
    # 2026-08-03 code review. The cache is keyed by a stable signature
    # derived from the tool id + sorted params (built by the caller); only
    # tools with ``effect == "read"`` and an authored
    # ``abbreviation_template`` are cached, so the kernel never names tool
    # ids. One cache per turn (no cross-turn leakage); reset by the fresh
    # ``_TurnState`` every turn and every fork child.
    read_cache: dict[str, str] = field(default_factory=dict)
    """Per-turn cache of read-tool signatures → abbreviations. Prevents
    re-reads from re-accumulating full tool output in the message
    history."""


# ── Closure-free helper functions ──────────────────────────────────────────


async def emit_stage(
    state: _TurnState,
    on_stage_hook: Any,
    log: logging.Logger,
    label: str,
    data: dict[str, Any] | None = None,
) -> None:
    """Wrap ``hooks.on_stage`` with None-check and try/except.

    Failures are logged and swallowed — a transport error never breaks the
    turn.
    """
    if on_stage_hook is None:
        return
    payload = dict(data or {})
    payload.setdefault("iteration", state.iteration)
    payload.setdefault("_state.iteration", state.iteration)
    # Stamp the budget caps so the transport can render graphical
    # counters (pip rows / sparklines) against the correct denominator.
    # Always present; 0 means unlimited (the transport falls back to a
    # saturating counter in that case).
    payload.setdefault("max_iterations", state.max_iterations)
    payload.setdefault("max_tool_calls", state.max_tool_calls)
    payload.setdefault("max_handoffs", state.max_handoffs)
    try:
        await on_stage_hook(label, payload)
    except Exception as stage_exc:  # noqa: BLE001  # pylint: disable=broad-except
        log.debug("on_stage(%s) hook failed: %s", label, stage_exc, exc_info=True)


async def mcp_tool_executor(
    tool_id: str,
    _ctx: Any,
    pool: MCPToolPool,
) -> ToolResult:
    """Bridge ``OperationExecutionContext.host.execute_tool`` to in-process tools.

    Catalog-driven: if the tool_id is in the internal tool catalog (discovered
    from packs), dispatch in-process via ``tools.registry.execute_tool``.
    Otherwise fall through to the MCP pool.  Declarative operations that need
    to call tools route through here.
    """
    from kernel.tools.registry import get_tool_catalog as _get_tools  # noqa: PLC0415
    from kernel.turn_phases._post_turn import _to_tool_context  # noqa: PLC0415

    tool_catalog = _get_tools()
    definition = tool_catalog.get(tool_id)
    if definition is not None and definition.implementation_module is not None:
        tool_ctx = _to_tool_context(tool_id, _ctx)
        return await execute_tool(tool_id, tool_ctx)

    params = getattr(_ctx, "parameters", {}) or {}
    mcp_result = await pool.call_tool(tool_id, params)
    return ToolResult(
        tool_id=tool_id,
        output_text=mcp_result.output_text,
        error=mcp_result.output_text if mcp_result.is_error else "",
    )


def record_model_usage(
    state: _TurnState,
    model_def: Any,
    *,
    p: int,
    c: int,
    t: int,
    ct: int,
    rt: int,
    d: float,
    lat: int,
    mode_id: str = "",
) -> None:
    """Merge one completion's counters into the per-model accumulator.

    When ``mode_id`` is non-empty, also accumulate the same counters into
    ``per_mode_usage`` so the host's token accounting can attribute each
    model call to the mode that actually ran it — not just the turn's
    entry mode. This mirrors ``per_model_usage``; without it, a kernel-
    driven recursive handoff (plan → search → coding) merges every inner
    turn's tokens into the outer result and the host attributes them to
    the entry mode.
    """
    mid = (getattr(model_def, "model_id", "") or "-").strip()
    entry = state.per_model_usage.setdefault(mid, {"model_id": mid})
    entry["prompt_tokens"] = entry.get("prompt_tokens", 0) + p
    entry["completion_tokens"] = entry.get("completion_tokens", 0) + c
    entry["total_tokens"] = entry.get("total_tokens", 0) + t
    entry["cached_tokens"] = entry.get("cached_tokens", 0) + ct
    entry["reasoning_tokens"] = entry.get("reasoning_tokens", 0) + rt
    entry["duration_ms"] = entry.get("duration_ms", 0.0) + d
    entry["latency_ms"] = entry.get("latency_ms", 0) + lat

    mid_mode = (mode_id or "").strip()
    if mid_mode:
        mode_entry = state.per_mode_usage.setdefault(mid_mode, {"mode_id": mid_mode})
        mode_entry["prompt_tokens"] = mode_entry.get("prompt_tokens", 0) + p
        mode_entry["completion_tokens"] = mode_entry.get("completion_tokens", 0) + c
        mode_entry["total_tokens"] = mode_entry.get("total_tokens", 0) + t
        mode_entry["cached_tokens"] = mode_entry.get("cached_tokens", 0) + ct
        mode_entry["reasoning_tokens"] = mode_entry.get("reasoning_tokens", 0) + rt
        mode_entry["duration_ms"] = mode_entry.get("duration_ms", 0.0) + d
        mode_entry["latency_ms"] = mode_entry.get("latency_ms", 0) + lat


def merge_child_usage(
    state: _TurnState,
    *,
    per_model_usage: list[dict[str, Any]] | None,
    per_mode_usage: list[dict[str, Any]] | None,
) -> None:
    """Merge a fork/delegate child's per-model/per-mode usage into the parent.

    Fork and delegate operations return ``ForkResult`` / ``DelegationResult``
    carrying the child turn's ``per_model_usage`` / ``per_mode_usage`` (every
    model / mode the child actually called). The parent turn loop calls this
    after the operation returns so ``/tokens`` attributes the child's tokens
    to the child's models/modes — not just the parent's resolved default.

    Without this, a multi-model fork or delegate shows only the parent's
    model in the per-model breakdown (the 2026-08-06 regression). Empty /
    ``None`` lists are no-ops (timeout / error / cancelled children, or
    operations that do not carry usage).
    """
    for entry in per_model_usage or []:
        mid = str(entry.get("model_id", "") or "-").strip() or "-"
        target = state.per_model_usage.setdefault(mid, {"model_id": mid})
        target["prompt_tokens"] = target.get("prompt_tokens", 0) + _int(entry.get("prompt_tokens", 0))
        target["completion_tokens"] = target.get("completion_tokens", 0) + _int(entry.get("completion_tokens", 0))
        target["total_tokens"] = target.get("total_tokens", 0) + _int(entry.get("total_tokens", 0))
        target["cached_tokens"] = target.get("cached_tokens", 0) + _int(entry.get("cached_tokens", 0))
        target["reasoning_tokens"] = target.get("reasoning_tokens", 0) + _int(entry.get("reasoning_tokens", 0))
        target["duration_ms"] = target.get("duration_ms", 0.0) + _float(entry.get("duration_ms", 0.0))
        target["latency_ms"] = target.get("latency_ms", 0) + _int(entry.get("latency_ms", 0))
    for entry in per_mode_usage or []:
        mid = str(entry.get("mode_id", "") or "-").strip() or "-"
        target = state.per_mode_usage.setdefault(mid, {"mode_id": mid})
        target["prompt_tokens"] = target.get("prompt_tokens", 0) + _int(entry.get("prompt_tokens", 0))
        target["completion_tokens"] = target.get("completion_tokens", 0) + _int(entry.get("completion_tokens", 0))
        target["total_tokens"] = target.get("total_tokens", 0) + _int(entry.get("total_tokens", 0))
        target["cached_tokens"] = target.get("cached_tokens", 0) + _int(entry.get("cached_tokens", 0))
        target["reasoning_tokens"] = target.get("reasoning_tokens", 0) + _int(entry.get("reasoning_tokens", 0))
        target["duration_ms"] = target.get("duration_ms", 0.0) + _float(entry.get("duration_ms", 0.0))
        target["latency_ms"] = target.get("latency_ms", 0) + _int(entry.get("latency_ms", 0))


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def budget_forces_final(
    state: _TurnState,
    turn_budget: TurnBudget | None,
    turn_start_mono: float,
) -> bool:
    """Return True when the turn budget is exhausted.

    A budget field set to 0 means unlimited and never forces.  When any
    nonzero field is exhausted, the model gets one more call — but with no
    tools advertised so it must produce a final text response.
    """
    if turn_budget is None:
        return False
    if turn_budget.max_tool_calls and state.total_tool_calls >= turn_budget.max_tool_calls:
        return True
    # Completion-based token rail: measures actual model work (output tokens),
    # not context re-send. ``max_total_tokens`` is the cumulative fallback for
    # hosts/sessions that still want a whole-turn envelope.
    if turn_budget.max_completion_tokens and state.total_completion_tokens >= (
        turn_budget.max_completion_tokens
    ):
        return True
    if turn_budget.max_total_tokens and state.total_tokens >= turn_budget.max_total_tokens:
        return True
    if turn_budget.max_wall_ms:
        elapsed_ms = int((time.monotonic() - turn_start_mono) * 1000)
        if elapsed_ms >= turn_budget.max_wall_ms:
            return True
    return False


def resolve_iteration_model(
    state: _TurnState,
    iteration_num: int,
    hooks: Any,
    preferred_model_id: str,
    resolve_model: Any,
    model_def: Any,
    query: str,
    mode_id: str,
    turn_budget: TurnBudget | None,
    log: logging.Logger,
) -> Any:
    """Three-tier per-iteration model selection.

    1. ``ContextSeams.model_router`` — the formal ``ModelRouter`` Protocol
       with a widened ``RouterRequest`` (now including ``effective_model``).
    2. ``ModelPolicy.choose`` — the legacy duck-typed optional method.
    3. Static ``model_def`` — the once-resolved ``preferred_model_id``.

    When the router returns a terminal action (``finish`` or
    ``transition_to``), the decision is stored on
    ``state.pending_router_decision`` and the static ``model_def`` is
    returned — the turn loop inspects the pending decision after this call
    and forces a final completion or a workflow transition accordingly.
    """
    from kernel.seams.protocols import (  # noqa: PLC0415
        RouterDecision,
        RouterRequest,
    )

    state.pending_router_decision = None

    router = hooks.context_seams.model_router
    if router is not None:
        effective_ref = _effective_model_ref(state, model_def, preferred_model_id)
        req = RouterRequest(
            default_model_id=preferred_model_id,
            iteration=iteration_num,
            tool_calls_so_far=state.total_tool_calls,
            budget_remaining=(
                None
                if turn_budget is None
                else max(0, turn_budget.max_tool_calls - state.total_tool_calls)
                if turn_budget.max_tool_calls
                else None
            ),
            query=query,
            last_tool_result=state.last_tool_result_text,
            has_tool_result=state.tool_results_received > 0,
            required_capabilities=state.required_capabilities,
            mode_id=mode_id,
            effective_model=effective_ref,
        )
        try:
            decision = router.choose(req)
        except Exception:  # pylint: disable=broad-except
            log.warning(
                "model_router.choose failed (iteration=%d)",
                iteration_num,
                exc_info=True,
            )
            decision = None
        if isinstance(decision, RouterDecision):
            if decision.finish.strip():
                state.pending_router_decision = decision
                return model_def
            if decision.transition_to.strip():
                state.pending_router_decision = decision
                return model_def
            if decision.model_id.strip():
                resolved = resolve_model(decision.model_id.strip())
                if resolved is not None:
                    state.effective_model = resolved
                    return resolved
                log.warning(
                    "model_router returned unknown model_id %r; falling back",
                    decision.model_id,
                )
        elif isinstance(decision, str) and decision.strip():
            # Tolerate routers that return a bare model_id string.
            resolved = resolve_model(decision.strip())
            if resolved is not None:
                state.effective_model = resolved
                return resolved
    # Legacy duck-typed ``choose`` on ModelPolicy.
    policy = hooks.context_seams.model_policy
    choose_fn = getattr(policy, "choose", None)
    if callable(choose_fn):
        chosen_id = policy.choose(
            default_model_id=preferred_model_id,
            iteration=iteration_num,
            tool_calls_so_far=state.total_tool_calls,
            budget_remaining=(
                None
                if turn_budget is None
                else max(0, turn_budget.max_tool_calls - state.total_tool_calls)
                if turn_budget.max_tool_calls
                else None
            ),
        )
        if isinstance(chosen_id, str) and chosen_id.strip():
            resolved = resolve_model(chosen_id.strip())
            if resolved is not None:
                state.effective_model = resolved
                return resolved
    state.effective_model = model_def
    return model_def


def _effective_model_ref(
    state: _TurnState,
    model_def: Any,
    preferred_model_id: str,
) -> RouterModelRef:
    """Build a RouterModelRef from the effective/preceding model.

    On iteration 1, ``state.effective_model`` is None (nothing has run yet),
    so we use ``model_def`` — the statically-resolved preferred model. On
    later iterations, ``state.effective_model`` carries the model actually
    selected in the preceding iteration.
    """
    source = state.effective_model if state.effective_model is not None else model_def
    if source is None:
        return RouterModelRef(model_id=preferred_model_id)
    return RouterModelRef(
        model_id=getattr(source, "model_id", "") or preferred_model_id,
        provider=getattr(source, "provider", "") or "",
        cost_tier=getattr(source, "cost_tier", "") or "",
        latency_tier=getattr(source, "latency_tier", "") or "",
        capabilities=tuple(getattr(source, "capabilities", ()) or ()),
        reasoning_model=bool(getattr(source, "reasoning_model", False)),
        thinking_model=bool(getattr(source, "thinking_model", False)),
        structured_reliability=_coerce_float(getattr(source, "structured_reliability", 0.0)),
        needs_guidance=bool(getattr(source, "needs_guidance", False)),
    )


def _coerce_float(value: Any) -> float:
    """Coerce a possibly-missing reliability value to a bounded float."""
    try:
        number = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number

"""Intra-turn context compaction: live-array estimator, epoch evictor, gates.

The governing contracts are ``docs/TOKEN_BUDGET.md`` and
``docs/CONTEXT_LIFECYCLE.md``. Motivated by the 2026-10-31 turn-1116 incident:
61 tool dispatches kept
full-text results resident in the live provider message array (~144K-token
average calls), output headroom collapsed, an edit batch was cut mid-JSON,
and the starved turn died in a hallucination spiral.

Three kernel-owned pieces:

- :class:`LiveArrayEstimator` — an incremental, conservatively biased
  token estimate of the live message array, anchored on the kernel's own
  per-request token count (``RequestContextBudget.input_tokens``) and
  advanced with byte deltas between requests. Under-counting invites
  output truncation; over-counting only forces an early compaction, so
  every uncertainty rounds UP.
- :func:`maybe_run_soft_epoch` — the mid-turn governor's soft path.
  A mode's working set lowers the trigger and sizes the epoch's tool-result
  cap, while the allocator still checks the real model window. Growth
  hysteresis keeps the live
  array append-only between epochs, so normal dispatches extend the
  provider's cached prefix instead of rewriting it. The epoch uses the
  host projector's ref-backed previews and evidence inventories, then
  synchronizes the live and projected arrays as one coherent state.
- :func:`hard_ceiling_stop_reason` — the hard ceiling. When the estimate
  crosses the configured share of the window, large tool dispatches are
  denied (gates live at the prepare-dispatch boundary) and the loop is
  pointed at a clean finish at the next dispatch boundary. Termination
  deliberately does NOT depend on the in-flight loop-control adapter:
  falling back to ``finish`` degrades gracefully whether or not the
  mid-dispatch break vocabulary lands.

One-way ref-out is enforced by construction: the epoch replaces evicted
content with its storage ref and never re-inlines it; recovery from a
ref goes through the thread tools only, so the oscillation that would
double the cache-rewrite tax cannot occur.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from kernel.config import settings as _kernel_settings
from kernel.context.request_budget import (
    RequiredContextBudgetError,
    ResultProjector,
    estimate_request_tokens,
    fit_request_context,
)
from kernel.provider_contracts import CompletionRequest, RequestTokenCount
from kernel.turn_phases import _token_accounting


@dataclass
class IntraTurnCompactionState:
    """Per-turn governor config snapshot plus epoch/breaker flags.

    Mirrors the kernel settings once per turn so every governor decision
    inside one turn reads consistent thresholds even if the process-wide
    settings object is reloaded mid-turn. ``epoch_estimate`` records the
    last attempted epoch so the next rewrite waits for a new crossing or
    meaningful growth, rather than retrying a failed projector every call.
    """

    enabled: bool = True
    soft_trigger_share: float = 0.5
    hard_ceiling_share: float = 0.85
    envelope_window_share: float = 0.8
    turn_token_envelope: int = 0
    preview_repeat_limit: int = 3
    history_epoch_min_tokens: int = 12_000
    cache_read_share: float = 0.1
    cache_write_share: float = 1.25
    epoch_estimate: int = 0
    epoch_failures: int = 0
    finish_now: bool = False

    @classmethod
    def from_settings(cls) -> IntraTurnCompactionState:
        """Snapshot the intra-turn compaction settings block."""
        return cls(
            enabled=bool(_kernel_settings.intra_turn_compaction_enabled),
            soft_trigger_share=float(_kernel_settings.intra_turn_soft_trigger_share),
            hard_ceiling_share=float(_kernel_settings.intra_turn_hard_ceiling_share),
            envelope_window_share=float(_kernel_settings.intra_turn_envelope_window_share),
            preview_repeat_limit=int(_kernel_settings.intra_turn_spill_preview_repeat_limit),
            history_epoch_min_tokens=int(_kernel_settings.history_epoch_min_tokens),
            cache_read_share=float(_kernel_settings.intra_turn_cache_read_share),
            cache_write_share=float(_kernel_settings.intra_turn_cache_write_share),
        )


@dataclass
class LiveArrayEstimator:
    """Incremental, over-count-biased token estimate of the live array.

    A dataclass (not a slotted plain class) so the turn-state serializer
    can generate a pydantic schema for it: ``kernel.turn_continuation``
    builds ``TypeAdapter(_TurnState)`` at import time and every
    ``_TurnState`` field type must be schema-generatable.

    The anchor is authoritative: the first observation adopts the kernel's
    counted input tokens for the projected request (which contains every
    live message plus framing), so the estimator never starts below the
    real prompt. Between requests the estimate advances with byte deltas
    from appended live messages, converted at a conservative 3 bytes per
    token (dense code and JSON tokenize well below the 4-chars-per-token
    prose average — the same bias ``estimate_request_tokens`` uses).
    Every estimate is multiplied by a small safety factor so borderline
    arrays trigger compaction slightly early.
    """

    _estimate: int = 0
    _live_bytes: int = 0
    _observed: bool = False

    def observe_request_tokens(self, counted_input_tokens: int) -> None:
        """Adopt a new request anchor; appended bytes are already in it."""
        self._estimate = max(self._estimate, int(counted_input_tokens))
        self._live_bytes = 0
        self._observed = True

    def observe_append(self, message: dict[str, Any]) -> None:
        """Advance the byte baseline by one appended live message."""
        self._live_bytes += len(str(message.get("content") or "").encode("utf-8"))

    def estimate(self) -> int:
        """Return the current estimate, rounded up conservatively."""
        if not self._observed:
            return 0
        delta_tokens = -(-self._live_bytes // 3)
        return int((self._estimate + delta_tokens) * 1.05)

    def release(self, *, bytes_removed: int) -> None:
        """Record bytes leaving the live array (history folds).

        The byte baseline drops in full; the adopted request-token anchor
        drops at a quarter weight so the estimate stays biased high — the
        same under-subtraction the fold governor prices its rewrites with.
        """

        self._live_bytes = max(0, self._live_bytes - bytes_removed)
        self._estimate = max(0, self._estimate - bytes_removed // 4)


def _epoch_projector(hooks: Any) -> ResultProjector:
    """Adapt the host projector to the allocator's typed callback contract."""
    project = hooks.context.project_messages

    def project_results(
        messages: list[dict[str, Any]], *, budget_bytes: int
    ) -> list[dict[str, Any]]:
        # ``hooks`` is ``Any`` here, so the projector call below is typed
        # ``Any`` even though the host-side contract is the typed
        # ``ResultProjector`` used everywhere else; the cast keeps the
        # adapter's declared return type without changing runtime behavior.
        return cast(
            "list[dict[str, Any]]",
            project(messages, hooks.identity.task_id, budget_bytes=budget_bytes),
        )

    return project_results


def build_epoch_request(
    *,
    messages: list[dict[str, Any]],
    model_def: Any,
    tools: list[dict[str, Any]] | None,
    max_tokens: int,
    prompt_cache_key: str,
) -> CompletionRequest:
    """Mirror the next normal request while preserving its real model window."""
    return CompletionRequest(
        messages=messages,
        model_def=model_def,
        max_tokens=max_tokens,
        temperature=0.4,
        tools=tools,
        prompt_cache_key=prompt_cache_key,
    )


def run_compaction_epoch(
    *,
    state: Any,
    hooks: Any,
    messages: list[dict[str, Any]],
    model_def: Any,
    tools: list[dict[str, Any]] | None,
    max_tokens: int,
    tool_budget_bytes: int | None,
    prompt_cache_key: str,
    log: logging.Logger,
) -> bool:
    """Apply the host's ref-backed projection at an optional residency cap.

    The model's real window still bounds required input and output. The cap
    limits only replaceable tool results; it never pretends that a large
    required prefix can be compacted by changing the provider's window.
    """
    if hooks.context.project_messages is None:
        log.debug("compaction epoch skipped: no host projector wired")
        return False
    request = build_epoch_request(
        messages=messages,
        model_def=model_def,
        tools=tools,
        max_tokens=max_tokens,
        prompt_cache_key=prompt_cache_key,
    )
    try:
        fitted = fit_request_context(
            request,
            _epoch_projector(hooks),
            estimate_request_tokens,
            tool_budget_bytes=tool_budget_bytes,
        )
    except Exception as exc:  # pylint: disable=broad-except
        cfg = state.intra_turn_compaction
        cfg.epoch_failures += 1
        log.warning(
            "compaction epoch failed; keeping live array unchanged (failure=%d): %s",
            cfg.epoch_failures,
            exc,
        )
        return False
    budget = fitted.context_budget
    if budget is None:
        log.warning("compaction epoch produced no context budget; keeping live array unchanged")
        return False
    reduced = fitted.messages
    before = sum(
        len(str(m.get("content") or "").encode("utf-8"))
        for m in messages
        if m.get("role") == "tool"
    )
    after = sum(
        len(str(m.get("content") or "").encode("utf-8")) for m in reduced if m.get("role") == "tool"
    )
    cfg = state.intra_turn_compaction
    if reduced == messages:
        log.debug("compaction epoch: no residency change; array unchanged")
        return False
    messages[:] = reduced
    # Resync the estimator: the epoch's reduced request is the new anchor.
    estimator = state.intra_turn_compaction_estimator
    old_estimate = estimator.estimate()
    estimator._estimate = max(budget.input_tokens, 0)
    estimator._live_bytes = 0
    estimator._observed = True
    # Resync the projection copy so the next request build appends the
    # post-epoch messages instead of resurrecting evicted full text from
    # the pre-epoch projection snapshot.
    state.provider_messages = copy.deepcopy(reduced)
    state.projected_message_count = len(messages)
    state.request_context_budget = budget
    cfg.epoch_failures = 0
    log.info(
        "compaction epoch: ref-out %d -> %d tool bytes (budget=%s, estimate %d -> %d)",
        before,
        after,
        tool_budget_bytes if tool_budget_bytes is not None else "window",
        old_estimate,
        budget.input_tokens,
    )
    return True


def maybe_run_soft_epoch(
    *,
    state: Any,
    hooks: Any,
    messages: list[dict[str, Any]],
    model_def: Any,
    tools: list[dict[str, Any]] | None,
    max_tokens: int,
    mode_id: str,
    prompt_cache_key: str,
    log: logging.Logger,
) -> bool:
    """Compact when the live array crosses its trigger; re-arm after growth."""
    cfg = state.intra_turn_compaction
    if not cfg.enabled:
        return False
    estimator = state.intra_turn_compaction_estimator
    estimate = estimator.estimate()
    if estimate <= 0:
        return False
    trigger_tokens = soft_trigger_tokens(state=state, model_def=model_def)
    if trigger_tokens <= 0:
        return False
    mode_config = getattr(state, "mode_config", None)
    working_set = int(getattr(mode_config, "working_set_tokens", 0) or 0)
    window = effective_window_tokens(state=state, model_def=model_def) or trigger_tokens
    if estimate < trigger_tokens:
        cfg.epoch_estimate = 0
        return False
    if cfg.epoch_estimate > 0 and estimate < cfg.epoch_estimate * 1.10:
        return False
    cfg.epoch_estimate = estimate
    log.info(
        "soft trigger crossed (estimate=%d >= %d of window=%d); running "
        "batched ref-out epoch (mode=%s)",
        estimate,
        trigger_tokens,
        window,
        mode_id,
    )
    # Residency target for replaceable tool results. A mode working set is
    # authoritative; otherwise keep half the trigger so the post-epoch
    # array has room to grow append-only before the next crossing. Passing
    # no budget would let the projector keep every byte (a no-op epoch on
    # large-window models).
    residency_tokens = working_set if working_set > 0 else trigger_tokens // 2
    return run_compaction_epoch(
        state=state,
        hooks=hooks,
        messages=messages,
        model_def=model_def,
        tools=tools,
        max_tokens=max_tokens,
        tool_budget_bytes=residency_tokens * 3,
        prompt_cache_key=prompt_cache_key,
        log=log,
    )


def effective_window_tokens(*, state: Any, model_def: Any | None = None) -> int:
    """Return the window the soft/hard shares apply to.

    The model window comes from the fitted request budget when available
    (same window the allocator just enforced), else the model definition.
    A turn processed-token envelope caps it two ways:

    - **Static cap** — ``envelope * envelope_window_share`` bounds residency
      by what the turn could afford at its start.
    - **Remaining cap** — the *unspent* portion of the envelope, times the
      same share, tightens as the turn spends uncached input and output.
      At turn start the two caps are equal; as the turn works, the remaining
      cap is strictly tighter, so a high-output or cache-cold turn compacts
      (and then finishes) instead of dispatching past its envelope.

    Both caps use the same processed-token accounting as the budget rail
    (``_token_accounting.processed_tokens``), so the compaction governor and
    ``budget_forces_final`` can never disagree about what was spent.
    """
    window = 0
    budget = state.request_context_budget
    if budget is not None:
        window = int(getattr(budget, "window_tokens", 0) or 0)
    if window <= 0 and model_def is not None:
        from kernel.context.context_budget import model_context_window_tokens

        window = model_context_window_tokens(model_def)
    if window <= 0:
        return 0
    cfg = state.intra_turn_compaction
    if cfg.turn_token_envelope > 0:
        static_cap = int(cfg.turn_token_envelope * cfg.envelope_window_share)
        processed = _token_accounting.processed_tokens(
            int(getattr(state, "total_tokens", 0) or 0),
            int(getattr(state, "total_cached_tokens", 0) or 0),
        )
        remaining = _token_accounting.remaining_envelope(cfg.turn_token_envelope, processed)
        if remaining is not None:
            remaining_cap = int(remaining * cfg.envelope_window_share)
            window = min(window, static_cap, remaining_cap)
        else:
            window = min(window, static_cap)
    return window


def soft_trigger_tokens(*, state: Any, model_def: Any | None = None) -> int:
    """Return the active soft-trigger token threshold for the turn."""
    window = effective_window_tokens(state=state, model_def=model_def)
    if window <= 0:
        return 0
    mode_config = getattr(state, "mode_config", None)
    working_set = int(getattr(mode_config, "working_set_tokens", 0) or 0)
    return min(int(window * state.intra_turn_compaction.soft_trigger_share), working_set or window)


@dataclass(frozen=True)
class CompactionDiagnostics:
    """Numeric-only snapshot of the compaction envelope for diagnostics.

    Read-only: this never feeds a budget decision. The cache hit rate is
    provider-reported usage, not a guarantee of future cache hits, and is
    ``None`` when the provider did not report prompt tokens.
    """

    mode_id: str = ""
    raw_window: int = 0
    effective_window: int = 0
    envelope: int = 0
    processed: int = 0
    remaining: int | None = None
    cached: int = 0
    prompt: int = 0
    cache_hit_rate: float | None = None
    share: float = 0.0


def compaction_diagnostics(
    *,
    state: Any,
    model_def: Any | None = None,
    mode_id: str = "",
) -> CompactionDiagnostics:
    """Build the numeric-only diagnostics snapshot for the current turn."""
    cfg = state.intra_turn_compaction
    raw_window = 0
    budget = state.request_context_budget
    if budget is not None:
        raw_window = int(getattr(budget, "window_tokens", 0) or 0)
    if raw_window <= 0 and model_def is not None:
        from kernel.context.context_budget import model_context_window_tokens

        raw_window = model_context_window_tokens(model_def)
    processed = _token_accounting.processed_tokens(
        int(getattr(state, "total_tokens", 0) or 0),
        int(getattr(state, "total_cached_tokens", 0) or 0),
    )
    remaining = _token_accounting.remaining_envelope(cfg.turn_token_envelope, processed)
    cached = int(getattr(state, "total_cached_tokens", 0) or 0)
    prompt = int(getattr(state, "total_prompt_tokens", 0) or 0)
    return CompactionDiagnostics(
        mode_id=str(mode_id),
        raw_window=raw_window,
        effective_window=effective_window_tokens(state=state, model_def=model_def),
        envelope=int(cfg.turn_token_envelope),
        processed=processed,
        remaining=remaining,
        cached=cached,
        prompt=prompt,
        cache_hit_rate=_token_accounting.cache_hit_rate(cached, prompt),
        share=float(cfg.envelope_window_share),
    )


def log_compaction_diagnostics(
    *,
    state: Any,
    model_def: Any | None = None,
    mode_id: str = "",
    log: logging.Logger,
) -> None:
    """Emit one numeric-only DEBUG record for the compaction envelope.

    The caller owns the one-shot flag (reset on resume) so this helper stays
    pure. Diagnostics failures are swallowed: they must never change the
    turn outcome.
    """
    try:
        d = compaction_diagnostics(state=state, model_def=model_def, mode_id=mode_id)
        log.debug(
            "compaction envelope: mode=%s raw_window=%d effective_window=%d "
            "envelope=%d processed=%d remaining=%s cached=%d prompt=%d "
            "cache_hit_rate=%s share=%.2f",
            d.mode_id,
            d.raw_window,
            d.effective_window,
            d.envelope,
            d.processed,
            d.remaining,
            d.cached,
            d.prompt,
            (f"{d.cache_hit_rate:.3f}" if d.cache_hit_rate is not None else "unknown"),
            d.share,
        )
    except Exception:  # pylint: disable=broad-except
        log.debug("compaction envelope diagnostics failed; ignoring", exc_info=True)


def should_spill_on_arrival(*, state: Any, model_def: Any | None = None) -> bool:
    """Return True when delivery-time spill should apply for this turn.

    Spill-on-arrival is pressure-gated: below the soft trigger we keep tool
    results inline so the model can inspect fresh evidence without immediate
    dereference churn. At/above the trigger we spill large outputs to keep
    live-array growth append-friendly.
    """
    if not state.intra_turn_compaction.enabled:
        return False
    trigger = soft_trigger_tokens(state=state, model_def=model_def)
    if trigger <= 0:
        return False
    return bool(state.intra_turn_compaction_estimator.estimate() >= trigger)


def hard_ceiling_stop_reason(*, state: Any) -> str:
    """Return the hard-ceiling stop reason, or ``""`` when under the ceiling.

    The window comes from the live request budget: ``fit_request_context``
    sizes every provider call against the active model's window and stores
    it on the turn state, so a dispatch boundary can check the ceiling
    without a model definition in hand.
    """
    cfg = state.intra_turn_compaction
    if not cfg.enabled:
        return ""
    budget = state.request_context_budget
    if budget is None or budget.window_tokens <= 0:
        return ""
    window = effective_window_tokens(state=state)
    estimate = state.intra_turn_compaction_estimator.estimate()
    if estimate >= int(window * cfg.hard_ceiling_share):
        return "context_hard_ceiling"
    return ""


def tool_dispatch_refusal(
    *,
    state: Any,
    messages: list[dict[str, Any]],
    model_def: Any,
    tools: list[dict[str, Any]] | None,
    max_tokens: int,
    projector: ResultProjector | None = None,
    counter: Callable[[CompletionRequest], RequestTokenCount] = estimate_request_tokens,
) -> str:
    """Return a refusal when the fitted proposed request crosses the ceiling.

    The current assistant message already contains the proposed tool-call
    arguments, so fitting ``messages`` accounts for the exact dispatch payload
    plus retained context using the same allocator as the next provider call.
    Read-only tools bypass this helper at the caller.
    """
    cfg = state.intra_turn_compaction
    if not cfg.enabled:
        return ""
    window = effective_window_tokens(state=state, model_def=model_def)
    if window <= 0:
        return ""
    request = CompletionRequest(
        messages=messages,
        model_def=model_def,
        max_tokens=max_tokens,
        temperature=0.4,
        tools=tools,
    )
    try:
        fitted = fit_request_context(request, projector, counter)
    except RequiredContextBudgetError as exc:
        return (
            "Dispatch refused: the proposed request cannot fit with its "
            f"required completion reserve ({exc}) under the hard-ceiling "
            "policy. Finish the turn now with the results already in hand; "
            "do not retry this dispatch."
        )
    budget = fitted.context_budget
    if budget is None:
        return (
            "Dispatch refused: the proposed request has no fitted context "
            "budget. Finish the turn now with the results already in hand."
        )
    hard_ceiling = int(window * cfg.hard_ceiling_share)
    if budget.input_tokens < hard_ceiling:
        return ""
    return (
        "Dispatch refused: the fitted proposed request uses "
        f"{budget.input_tokens} input tokens against a {hard_ceiling}-token "
        "hard ceiling (context_hard_ceiling). Finish the turn now with the "
        "results already in hand; do not retry this dispatch."
    )

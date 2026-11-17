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
from dataclasses import dataclass
from typing import Any, cast

from kernel.config import settings as _kernel_settings
from kernel.context.request_budget import (
    ResultProjector,
    estimate_request_tokens,
    fit_request_context,
)
from kernel.provider_contracts import CompletionRequest


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
    hard_ceiling_share: float = 0.65
    large_dispatch_min_chars: int = 2000
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
            large_dispatch_min_chars=int(_kernel_settings.intra_turn_large_dispatch_min_chars),
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
    window = state.request_context_budget.window_tokens if state.request_context_budget else 0
    if window <= 0:
        window = trigger_tokens
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
    return run_compaction_epoch(
        state=state,
        hooks=hooks,
        messages=messages,
        model_def=model_def,
        tools=tools,
        max_tokens=max_tokens,
        tool_budget_bytes=working_set * 3 if working_set > 0 else None,
        prompt_cache_key=prompt_cache_key,
        log=log,
    )


def soft_trigger_tokens(*, state: Any, model_def: Any | None = None) -> int:
    """Return the active soft-trigger token threshold for the turn.

    Uses the fitted request budget window when available (same window the
    allocator just enforced). Falls back to the model definition when the
    turn has not fitted a request yet.
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
    mode_config = getattr(state, "mode_config", None)
    working_set = int(getattr(mode_config, "working_set_tokens", 0) or 0)
    return min(int(window * state.intra_turn_compaction.soft_trigger_share), working_set or window)


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
    return state.intra_turn_compaction_estimator.estimate() >= trigger


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
    estimate = state.intra_turn_compaction_estimator.estimate()
    if estimate >= int(budget.window_tokens * cfg.hard_ceiling_share):
        return "context_hard_ceiling"
    return ""


def tool_size_refusal(
    *,
    state: Any,
    tool_arguments: Any,
) -> str:
    """Return a refusal message for a large dispatch under the hard ceiling.

    Read-only tools are never gated (the model may still need to inspect
    its own results); large-argument dispatches are denied with a
    model-visible message directing a clean finish at the next boundary.
    """
    cfg = state.intra_turn_compaction
    if not cfg.enabled:
        return ""
    try:
        serialized = len(str(tool_arguments).encode("utf-8"))
    except Exception:  # pylint: disable=broad-except
        return ""
    if serialized < cfg.large_dispatch_min_chars:
        return ""
    stop = hard_ceiling_stop_reason(state=state)
    if not stop:
        return ""
    return (
        "Dispatch refused: the live context is at the hard ceiling "
        f"({stop}). Finish the turn now with the results already in hand; "
        "do not retry this or other large calls this turn."
    )

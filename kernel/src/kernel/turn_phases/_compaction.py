"""Intra-turn context compaction: live-array estimator, epoch evictor, gates.

Design: ``docs/explainer/intra-turn-compaction-design.md`` (mechanism) and
``docs/design/intra-turn-compaction-cache-policy.md`` (cache policy layer).
Motivated by the 2026-10-31 turn-1116 incident: 61 tool dispatches kept
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
- :func:`compact_if_soft_triggered` — the mid-turn governor's soft path.
  At most ONE batched ref-out epoch per soft-trigger crossing
  (hysteresis): between epochs the live array is append-only, so normal
  dispatches extend the provider's cached prefix instead of rewriting it.
  The epoch forces the existing ``fit_request_context`` reduction (the
  proven host-projector path: newest-first near-tail eviction with
  dereferenceable result refs) at a window fraction well below the
  reactive limit, then rewrites the live array in place and resyncs the
  projection bookkeeping.
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

import logging
from dataclasses import dataclass, replace
from typing import Any

from kernel.config import settings as _kernel_settings
from kernel.context.request_budget import estimate_request_tokens, fit_request_context
from kernel.provider_contracts import CompletionRequest


@dataclass
class IntraTurnCompactionState:
    """Per-turn governor config snapshot plus epoch/breaker flags.

    Mirrors the kernel settings once per turn so every governor decision
    inside one turn reads consistent thresholds even if the process-wide
    settings object is reloaded mid-turn. ``epoch_done_for_crossing``
    implements the hysteresis: at most ONE batched ref-out epoch per
    soft-trigger crossing.
    """

    enabled: bool = True
    soft_trigger_share: float = 0.5
    hard_ceiling_share: float = 0.65
    large_dispatch_min_chars: int = 2000
    preview_repeat_limit: int = 3
    history_epoch_min_tokens: int = 2_000
    cache_read_share: float = 0.1
    cache_write_share: float = 1.25
    epoch_done_for_crossing: bool = False
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
        """Adopt the kernel's counted input tokens for the last request."""
        self._estimate = max(self._estimate, int(counted_input_tokens))
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


class _WindowFitter:
    """Callable shim that fits a request to a fraction of the model window.

    ``fit_request_context`` reads the window from the model definition; the
    epoch needs the SAME reduction path aimed at a lower effective window
    (soft trigger fraction below 1.0), so the model def is wrapped for the
    duration of the call. The wrapper object escapes only inside this
    module — it is never stored on state or handed to a provider.
    """

    __slots__ = ("_model_def", "_window")

    def __init__(self, model_def: Any, window: int) -> None:
        self._model_def = model_def
        self._window = int(window)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model_def, name)

    @property
    def context_window(self) -> int:
        return self._window


def _windowed_request(request: CompletionRequest, effective_window: int) -> CompletionRequest:
    wrapped = _WindowFitter(request.model_def, effective_window)
    return replace(request, model_def=wrapped)


def build_epoch_request(
    *,
    messages: list[dict[str, Any]],
    model_def: Any,
    tools: list[dict[str, Any]] | None,
    max_tokens: int,
    soft_window: int,
    prompt_cache_key: str,
) -> CompletionRequest:
    """Assemble the epoch's provider request at the soft-trigger window.

    The request mirrors what the next normal iteration would send (same
    messages, model, tools) so the epoch's persisted results are exactly
    the ones future calls need; only the effective window is lowered so
    ``fit_request_context`` performs the batched ref-out now instead of
    at the reactive limit.
    """
    return _windowed_request(
        CompletionRequest(
            messages=messages,
            model_def=model_def,
            max_tokens=max_tokens,
            temperature=0.4,
            tools=tools,
            prompt_cache_key=prompt_cache_key,
        ),
        soft_window,
    )


def run_compaction_epoch(
    *,
    state: Any,
    hooks: Any,
    messages: list[dict[str, Any]],
    model_def: Any,
    tools: list[dict[str, Any]] | None,
    max_tokens: int,
    soft_window: int,
    prompt_cache_key: str,
    log: logging.Logger,
) -> bool:
    """Run ONE batched ref-out epoch against the live array. Returns True on success.

    The epoch drives the existing host projector (spill-to-store + preview
    + ``r<sha256>`` ref) through ``fit_request_context`` at the soft window,
    then rewrites the live array in place with the reduced message list.
    Evicted content is never re-inlined — refs are one-way by construction.
    A failed epoch is logged and swallowed: the reactive reduction remains
    as the safety net, so a projector failure never breaks the turn.
    """
    if hooks.context.project_messages is None:
        log.debug("compaction epoch skipped: no host projector wired")
        return False
    request = build_epoch_request(
        messages=messages,
        model_def=model_def,
        tools=tools,
        max_tokens=max_tokens,
        soft_window=soft_window,
        prompt_cache_key=prompt_cache_key,
    )
    try:
        fitted = fit_request_context(
            request,
            (
                lambda source, *, budget_bytes: hooks.context.project_messages(
                    source, hooks.identity.task_id, budget_bytes=budget_bytes
                )
            ),
            estimate_request_tokens,
        )
    except Exception as exc:  # pylint: disable=broad-except
        log.warning("compaction epoch failed; keeping live array unchanged: %s", exc)
        return False
    budget = fitted.context_budget
    if budget is None:
        log.warning("compaction epoch produced no context budget; keeping live array unchanged")
        return False
    reduced = fitted.messages
    if len(reduced) != len(messages):
        log.warning(
            "compaction epoch changed message count (%d -> %d); keeping live array unchanged",
            len(messages),
            len(reduced),
        )
        return False
    before = sum(
        len(str(m.get("content") or "").encode("utf-8"))
        for m in messages
        if m.get("role") == "tool"
    )
    after = sum(
        len(str(m.get("content") or "").encode("utf-8")) for m in reduced if m.get("role") == "tool"
    )
    messages[:] = reduced
    # Resync the estimator: the epoch's reduced request is the new anchor.
    estimator = state.intra_turn_compaction_estimator
    old_estimate = estimator.estimate()
    estimator._estimate = max(budget.input_tokens, 0)
    estimator._live_bytes = sum(len(str(m.get("content") or "").encode("utf-8")) for m in messages)
    estimator._observed = True
    # Resync the projection copy so the next request build appends the
    # post-epoch messages instead of resurrecting evicted full text from
    # the pre-epoch projection snapshot.
    state.provider_messages = [
        {**m, "content": ""} if m.get("role") == "tool" else {**m} for m in reduced
    ]
    state.projected_message_count = len(messages)
    log.info(
        "compaction epoch: ref-out %d -> %d tool bytes (window=%d, estimate %d -> %d)",
        before,
        after,
        soft_window,
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
    """Run at most one epoch per soft-trigger crossing. True when one ran."""
    cfg = state.intra_turn_compaction
    if not cfg.enabled:
        return False
    estimator = state.intra_turn_compaction_estimator
    estimate = estimator.estimate()
    if estimate <= 0:
        return False
    from kernel.context.context_budget import model_context_window_tokens

    window = model_context_window_tokens(model_def)
    soft_tokens = int(window * cfg.soft_trigger_share)
    # A mode's ``working_set_tokens`` caps the trigger: a coding loop needs a
    # bounded working set, not half of a million-token window.
    mode_config = getattr(state, "mode_config", None)
    working_set = int(getattr(mode_config, "working_set_tokens", 0) or 0)
    if working_set > 0:
        soft_tokens = min(soft_tokens, working_set)
    crossing = estimate >= soft_tokens and not cfg.epoch_done_for_crossing
    if not crossing:
        return False
    # The gate records the epoch BEFORE running it: a failed epoch is not
    # retried every subsequent dispatch boundary (that would spam a broken
    # projector once per call); the reactive reduction remains the net.
    cfg.epoch_done_for_crossing = True
    log.info(
        "soft trigger crossed (estimate=%d >= %d of window=%d); running "
        "batched ref-out epoch (mode=%s)",
        estimate,
        soft_tokens,
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
        soft_window=soft_tokens,
        prompt_cache_key=prompt_cache_key,
        log=log,
    )


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

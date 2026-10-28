"""Durable token-usage telemetry handler.

This module provides ``TokenUsageTelemetryHandler``, the canonical consumer
for ``TokenUsageEvent`` payloads. It appends every event as one JSONL line
to a host-supplied ``token-usage.jsonl`` file so token accounting survives
process restarts.

Design notes:
    - Runtime-neutral: the handler takes a ``Path`` and an optional event
      filter. It does not import app-layer config or Azure exporters.
    - Idempotency: the ``TokenUsageEvent.idempotency_key`` is set to a
      timestamp + nanosecond suffix when empty, so retries and stream-final
      events always append without collapsing (per the event docstring).
    - Atomic appends: each line is written with ``"a"`` mode under a
      ``mkdir(parents=True, exist_ok=True)`` so the file appears lazily.
    - Failures are swallowed: telemetry must never break the turn loop.
      Errors are logged through ``logging`` and dropped.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from pathlib import Path
from typing import Any

from kernel.models import TokenUsageEvent, resolve_turn_index

logger = logging.getLogger(__name__)

#: Event-bus topic every ``TokenUsageEvent`` is published on. The host
#: subscribes its durable ``TokenUsageTelemetryHandler`` to this topic;
#: headless contexts (no bus, no subscriber) drop the events silently.
TOKEN_USAGE_TOPIC = "lisa.token_usage"

#: Strong references to in-flight token-usage publish tasks (RUF006). Tasks
#: are fire-and-forget telemetry; the set keeps them garbage-collectable
#: only after completion and is pruned on each publish.
_PENDING_PUBLISH_TASKS: set[asyncio.Task[None]] = set()

__all__ = [
    "TOKEN_USAGE_TOPIC",
    "TokenUsageTelemetryHandler",
    "build_token_usage_event",
    "coerce_non_negative_float",
    "coerce_non_negative_int",
    "ensure_usage_entry",
    "estimate_call_cost_usd",
    "publish_token_usage",
]


def coerce_non_negative_int(value: Any) -> int:
    """Safely convert *value* to a non-negative int (0 on failure)."""
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def publish_token_usage(bus: Any, event: TokenUsageEvent) -> None:
    """Publish one ``TokenUsageEvent`` to the host's event bus, best-effort.

    The turn runner calls this at every point where a completion's usage
    has been normalized (per-iteration model calls, forced-final, and
    terminal-stage completions). The host wires its durable
    :class:`TokenUsageTelemetryHandler` as a subscriber so every event
    survives process restarts; headless contexts (no bus) drop the event.

    Strictly best-effort per the telemetry contract: a publish failure is
    swallowed and must never break the model-call path. When no event loop
    is running (sync test contexts) the publish is dropped rather than
    blocking.
    """
    if bus is None or not hasattr(bus, "publish"):
        return
    with contextlib.suppress(Exception):
        task = asyncio.create_task(bus.publish(TOKEN_USAGE_TOPIC, event.model_dump(mode="json")))
        _PENDING_PUBLISH_TASKS.add(task)
        task.add_done_callback(_PENDING_PUBLISH_TASKS.discard)


def coerce_non_negative_float(value: Any) -> float:
    """Safely convert *value* to a non-negative float (0.0 on failure)."""
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def ensure_usage_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Ensure one breakdown dict has every token/time field.

    Pre-existing entries (from earlier sessions or tests with the old
    3-key schema) are upgraded in place so the accumulator math never
    raises ``KeyError`` on a missing ``cached_tokens`` etc.
    """
    entry.setdefault("prompt_tokens", 0)
    entry.setdefault("completion_tokens", 0)
    entry.setdefault("total_tokens", 0)
    entry.setdefault("cached_tokens", 0)
    entry.setdefault("reasoning_tokens", 0)
    entry.setdefault("cache_creation_tokens", 0)
    entry.setdefault("duration_ms", 0.0)
    entry.setdefault("latency_ms", 0)
    return entry


class TokenUsageTelemetryHandler:
    """Append every ``TokenUsageEvent`` to a durable JSONL file.

    Hosts construct one instance with the resolved ``token-usage.jsonl``
    path and register it as an event-bus consumer for the ``token_used``
    event type. The handler is async-publishable (``async def publish``)
    so it slots into the existing ``EventBus`` protocol, but the file
    write itself is synchronous and fast (one ``open(..., "a")`` per
    event).

    Idempotency key:
        When the incoming event has an empty ``idempotency_key``, the
        handler stamps it with ``emitted_at`` plus a nanosecond suffix
        derived from ``time.time_ns()``. This guarantees uniqueness
        across retries and stream-final events without collapsing
        legitimate duplicate calls.

    Args:
        path: Absolute path to the ``token-usage.jsonl`` file. Parent
            directories are created lazily on first write.
        filter_fn: Optional predicate that receives the event and
            returns ``True`` to persist, ``False`` to skip. Hosts use
            this to drop events below a severity threshold or to
            redact sensitive fields before persistence.
    """

    def __init__(
        self,
        path: Path,
        *,
        filter_fn: Any | None = None,
    ) -> None:
        self._path = path
        self._filter_fn = filter_fn

    @property
    def path(self) -> Path:
        """Return the resolved JSONL path."""
        return self._path

    async def publish(self, event: Any) -> None:
        """Persist one ``TokenUsageEvent`` as a single JSONL line.

        Accepts either a ``TokenUsageEvent`` instance or a payload dict
        (the shape the ``EventBus`` contract delivers to subscribers —
        publishers serialize events via ``model_dump(mode="json")``).
        Dicts are revalidated into the typed event; records that fail
        validation are dropped with a warning so one malformed payload
        cannot wedge the subscription.

        Silently returns for other payloads so the handler can be wired
        into a bus that fans out to many event types without per-type
        filtering at the call site.
        """
        if not isinstance(event, TokenUsageEvent):
            if isinstance(event, dict):
                try:
                    event = TokenUsageEvent.model_validate(event)
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning("token-usage dropped unvalidatable dict payload: %s", exc)
                    return
            else:
                return
        if self._filter_fn is not None:
            try:
                if not self._filter_fn(event):
                    return
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("token-usage filter raised; persisting event: %s", exc)
        self._append(event)

    def _append(self, event: TokenUsageEvent) -> None:
        if not event.idempotency_key:
            event.idempotency_key = f"{event.emitted_at}:{_time_ns_suffix()}"
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(event.model_dump_json() + "\n")
        except OSError as exc:
            logger.warning("token-usage append failed for %s: %s", self._path, exc)


def _time_ns_suffix() -> str:
    """Return a short strictly-monotonic nanosecond suffix for idempotency keys.

    Windows wall clocks tick at ~15.6ms, so raw ``time.time_ns()`` returns
    the identical value for events published within one tick and collapsed
    three legitimate events onto one idempotency key (2026-10-13 windows CI:
    ``assert 1 == 3``). Track the last issued value and always advance past
    it so each key in a process is unique.
    """
    global _LAST_IDEMPOTENCY_SUFFIX  # noqa: PLW0603
    value = time.time_ns() % 1_000_000_000
    if value <= _LAST_IDEMPOTENCY_SUFFIX:
        value = _LAST_IDEMPOTENCY_SUFFIX + 1
    _LAST_IDEMPOTENCY_SUFFIX = value
    return str(value)


_LAST_IDEMPOTENCY_SUFFIX = -1


def build_token_usage_event(
    *,
    provider: str,
    model_id: str,
    kind: str = "chat_completion",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    cached_tokens: int = 0,
    reasoning_tokens: int = 0,
    cache_creation_tokens: int = 0,
    duration_ms: float = 0.0,
    latency_ms: int = 0,
    cost_usd: float = 0.0,
    time_cost_usd: float = 0.0,
    provider_cost_usd: float = 0.0,
    request_id: str = "",
    mode_id: str = "",
    stage_id: str = "",
    turn_index: int | None = None,
    call_site: str = "",
    conversation_id: str = "",
    user_slug: str = "default",
    surface_id: str = "",
    extra: dict[str, Any] | None = None,
) -> TokenUsageEvent:
    """Build one ``TokenUsageEvent`` with normalized defaults.

    Hosts call this from each LLM call site so every provider call produces
    exactly one durable record. The
    ``total_tokens`` field falls out as ``prompt + completion`` when
    the caller does not supply it directly.

    The ``kind`` literal discriminates billing shape:
        - ``chat_completion`` / ``embedding`` / ``image_generation`` /
          ``stream_chunk``: token-based billing.
        - ``time_billed``: duration-based billing (e.g. Ollama local
          GPU time). ``duration_ms`` and ``time_cost_usd`` carry the
          billable signal; token counters may still be populated for
          observability even when cost is time-derived.
    """
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    return TokenUsageEvent(
        provider=provider,
        model_id=model_id,
        kind=kind,  # type: ignore[arg-type]
        prompt_tokens=int(prompt_tokens or 0),
        completion_tokens=int(completion_tokens or 0),
        total_tokens=int(total_tokens or 0),
        cached_tokens=int(cached_tokens or 0),
        reasoning_tokens=int(reasoning_tokens or 0),
        cache_creation_tokens=int(cache_creation_tokens or 0),
        duration_ms=float(duration_ms or 0.0),
        latency_ms=int(latency_ms or 0),
        cost_usd=float(cost_usd or 0.0),
        time_cost_usd=float(time_cost_usd or 0.0),
        provider_cost_usd=float(provider_cost_usd or 0.0),
        request_id=str(request_id or ""),
        mode_id=str(mode_id or ""),
        stage_id=str(stage_id or ""),
        turn_index=resolve_turn_index(turn_index),
        call_site=str(call_site or ""),
        conversation_id=str(conversation_id or ""),
        user_slug=str(user_slug or "default"),
        surface_id=str(surface_id or ""),
        extra=dict(extra) if extra else {},
    )


def estimate_call_cost_usd(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    cache_creation_tokens: int = 0,
    prompt_price: float = 0.0,
    completion_price: float = 0.0,
    cache_read_price: float = 0.0,
    cache_write_price: float = 0.0,
) -> float:
    """Price one completion from per-token USD rates (0.0 when unpriced).

    This is the *canonical* cost model every host shares: the same math the
    CLI footer used to re-derive locally, but with the prompt-cache split
    applied to both sides of the cache rather than just reads.

    The prompt side is a three-way split, because a cache hit is billed at
    the provider's discounted read rate while a cache *population* is billed
    at a premium write rate — pricing writes at the ordinary prompt rate
    silently understates turns that warm a cache. ``cached_tokens`` and
    ``cache_creation_tokens`` are the provider-reported sub-counters, and
    ``prompt_tokens`` already includes both (OpenRouter/OpenAI report the
    total prompt size), so the uncached remainder is what is left after
    removing them.

    Rates of 0.0 mean "provider does not price this", in which case the
    corresponding tokens fall back to ``prompt_price`` (the pre-split
    behaviour) so an incomplete manifest never produces a *free* call.
    Returns 0.0 overall when the provider publishes no prices at all.
    """

    prompt = max(0, int(prompt_tokens or 0))
    completion = max(0, int(completion_tokens or 0))
    cached = min(max(0, int(cached_tokens or 0)), prompt)
    # Cache writes are a subset of the prompt too. A provider that reports a
    # write count larger than the remaining prompt (Anthropic reports
    # creation and read counts for the same prefix across calls) must not
    # push the split negative; clamp to what is left after reads.
    writes = min(max(0, int(cache_creation_tokens or 0)), prompt - cached)
    uncached = prompt - cached - writes

    prompt_rate = float(prompt_price or 0.0)
    read_rate = float(cache_read_price or 0.0) or prompt_rate
    write_rate = float(cache_write_price or 0.0) or prompt_rate
    completion_rate = float(completion_price or 0.0)

    return max(
        0.0,
        uncached * prompt_rate
        + cached * read_rate
        + writes * write_rate
        + completion * completion_rate,
    )

"""Durable token-usage telemetry handler.

This module provides ``TokenUsageTelemetryHandler``, the canonical consumer
for ``TokenUsageEvent`` payloads. It appends every event as one JSONL line
to a host-supplied ``token-usage.jsonl`` file so token accounting survives
process restarts.

Design notes:
    - Runtime-neutral: the handler takes a ``Path`` and an optional event
      filter. It does not import app-layer config or Azure exporters.
    - Idempotency: the ``TokenUsageEvent.idempotency_key`` includes the
      handler instance and sequence as well as a timestamp suffix, so
      separate writers cannot collapse events on coarse-grained clocks.
    - Atomic appends: each line is written with ``"a"`` mode under a
      ``mkdir(parents=True, exist_ok=True)`` so the file appears lazily.
    - Failures are swallowed: telemetry must never break the turn loop.
      Errors are logged through ``logging`` and dropped.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import logging
import math
import os
import time
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from kernel.models import TokenUsageEvent, resolve_turn_index

logger = logging.getLogger(__name__)

#: Typed fields the durable ledger persists, in order — the privacy
#: allowlist. ``_ledger_dump`` writes exactly these keys: prompt/tool
#: content and the raw ``thread_store_slug`` never reach disk, and the
#: untyped ``extra`` dict (which any publisher can stuff prompt-shaped
#: payloads into) is excluded by construction. ``proxy_telemetry`` is
#: allowlisted as a typed field so pack-owned relay counters survive;
#: new durable facts join here as typed fields, never via ``extra``.
_LEDGER_FIELDS: tuple[str, ...] = (
    "schema_version",
    "event_type",
    "conversation_id",
    "user_slug",
    "surface_type",
    "surface_id",
    "memory_scope",
    "mode_id",
    "stage_id",
    "operation_id",
    "turn_index",
    "call_site",
    "provider",
    "model_id",
    "request_id",
    "kind",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cached_tokens",
    "reasoning_tokens",
    "cache_creation_tokens",
    "cost_usd",
    "provider_cost_usd",
    "duration_ms",
    "time_cost_usd",
    "latency_ms",
    "proxy_telemetry",
    "effective_model_data_source",
    "effective_model_data_version",
    "upstream_route",
    "cache_policy_fingerprint",
    "cache_ttl_seconds",
    "cache_min_prefix_chars",
    "metric_origins",
    "extra_context_appended",
    "sequence",
    "ledger_instance",
    "thread_pseudonym_id",
    "idempotency_key",
    "emitted_at",
)


def _ledger_dump(event: TokenUsageEvent) -> str:
    """Serialize an event to one JSONL line under the privacy allowlist.

    Redaction by construction: only the allowlisted typed fields are
    serialized — the raw thread-store slug and the untyped ``extra`` dict
    are dropped here, so no downstream writer can leak them. Unknown
    (``None``) fields and empty containers are omitted outright so legacy
    readers see absent keys, not nulls, and rows stay compact.
    """

    source = event.model_dump(mode="json")
    record: dict[str, Any] = {}
    for field in _LEDGER_FIELDS:
        value = source.get(field)
        if value is None or value in ({}, []):
            continue
        record[field] = value
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"))


#: Event-bus topic every ``TokenUsageEvent`` is published on. The host
#: subscribes its durable ``TokenUsageTelemetryHandler`` to this topic;
#: headless contexts (no bus, no subscriber) drop the events silently.
TOKEN_USAGE_TOPIC = "lisa.token_usage"

__all__ = [
    "PSEUDONYM_SECRET_FILENAME",
    "TOKEN_USAGE_TOPIC",
    "TokenUsageTelemetryHandler",
    "build_token_usage_event",
    "coerce_non_negative_float",
    "coerce_non_negative_int",
    "ensure_usage_entry",
    "estimate_call_cost_usd",
    "nullable_float",
    "nullable_int",
    "publish_token_usage",
    "thread_pseudonym",
]


def coerce_non_negative_int(value: Any) -> int:
    """Safely convert *value* to a non-negative int (0 on failure)."""
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def nullable_int(value: Any) -> int | None:
    """Convert *value* to a non-negative int, or ``None`` when unknown.

    Telemetry counterpart to :func:`coerce_non_negative_int` for the
    cache-efficiency v2 contract: an absent, ``None``, or malformed
    provider value is *unknown* and stays ``None`` instead of being
    reported as a measured zero. A present finite non-negative number —
    including a real ``0`` — is preserved as measured. Negative or
    non-finite values are rejected as malformed (``None``).
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def nullable_float(value: Any) -> float | None:
    """Convert *value* to a non-negative float, or ``None`` when unknown.

    Same unknown-versus-zero split as :func:`nullable_int` for the
    nullable cost/latency fields on ``TokenUsageEvent``. ``bool`` is not a
    measured value; negative or non-finite numbers are malformed.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0.0:
        return None
    return number


def publish_token_usage(bus: Any, event: TokenUsageEvent) -> None:
    """Publish one ``TokenUsageEvent`` to the host's event bus, best-effort.

    The turn runner calls this at every point where a completion's usage
    has been normalized (per-iteration model calls, forced-final, and
    terminal-stage completions). The host wires its durable
    :class:`TokenUsageTelemetryHandler` as a subscriber so every event
    survives process restarts; headless contexts (no bus) drop the event.

    Strictly best-effort per the telemetry contract: a publish failure is
    never allowed to break the model-call path, but it is no longer silent —
    :func:`schedule_telemetry_publish` logs a WARNING so a lost durable
    token row is visible in ``runtime.jsonl`` instead of vanishing. When no
    event loop is running (sync test contexts) the publish is dropped with
    the same warning rather than blocking.
    """
    from kernel.telemetry import schedule_telemetry_publish

    if bus is None or not hasattr(bus, "publish"):
        return
    schedule_telemetry_publish(
        bus,
        TOKEN_USAGE_TOPIC,
        event.model_dump(mode="json"),
        source=f"token_usage:{event.call_site or 'unknown'}",
    )


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


#: Secret-file name under the data root. Created once per data root with
#: owner-only permissions; losing it permanently severs pseudonym joins to
#: already-persisted rows (documented consequence, not an error).
PSEUDONYM_SECRET_FILENAME = ".token-usage-pseudonym-key"

#: Domain-separation prefix for the thread-pseudonym HMAC, plus the on-disk
#: pseudonym prefix. A v2 scheme change (algorithm or domain) would ship as
#: ``v2:`` without reinterpreting ``v1:`` rows.
_PSEUDONYM_DOMAIN = b"lisa-token-usage-thread-v1"
_PSEUDONYM_PREFIX = "v1:"


def _resolve_pseudonym_secret(data_dir: Path) -> bytes | None:
    """Return the data-root pseudonym secret, creating it atomically if absent.

    The key file is created with ``O_CREAT | O_EXCL`` and mode ``0o600`` so
    concurrent processes sharing one data root cannot race out two secrets
    (the loser re-reads the winner's key) and the secret is never
    world-readable. Any failure (read, create, or permission oddity)
    returns ``None`` and the handler then emits *unattributed* rows rather
    than failing the turn loop.
    """

    key_path = data_dir / PSEUDONYM_SECRET_FILENAME
    try:
        return key_path.read_bytes()
    except FileNotFoundError:
        pass
    except OSError:
        return None
    secret = os.urandom(32)
    try:
        fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        # Another process won the race; reuse its key so every writer in
        # one data root produces the same pseudonym space.
        try:
            return key_path.read_bytes()
        except OSError:
            return None
    except OSError:
        return None
    with os.fdopen(fd, "wb") as handle:
        handle.write(secret)
    return secret


def thread_pseudonym(thread_store_slug: str, secret: bytes) -> str:
    """Return the stable data-root-scoped pseudonym for one thread slug.

    ``HMAC-SHA256(secret, domain || slug)`` hex, ``v1:``-prefixed. Distinct
    slugs map to distinct pseudonyms within one data root; different data
    roots are unlinkable because each generates its own secret.
    """

    digest = hmac.new(
        secret, _PSEUDONYM_DOMAIN + thread_store_slug.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return _PSEUDONYM_PREFIX + digest


class TokenUsageTelemetryHandler:
    """Append every ``TokenUsageEvent`` to a durable JSONL file.

    Hosts construct one instance with the resolved ``token-usage.jsonl``
    path and register it as an event-bus consumer for the ``token_used``
    event type. The handler is async-publishable (``async def publish``)
    so it slots into the existing ``EventBus`` protocol, but the file
    write itself is synchronous and fast (one ``open(..., "a")`` per
    event).

    Durability contract (cache-efficiency telemetry slice):

    - The in-memory ``thread_store_slug`` rides the bus event for joins,
      but the persisted row carries only ``thread_pseudonym_id`` (a
      data-root-scoped HMAC of the slug). Raw thread identity never
      reaches disk.
    - ``user_slug`` persists so cross-user rows stay separable on a shared
      data root; it is an existing scoping key, not new exposure.
    - Numerics persist as ``null`` when unknown; the serializer never
      coerces ``None`` into a measured zero.

    Idempotency key:
        When the incoming event has an empty ``idempotency_key``, the
        handler stamps it with ``emitted_at``, this handler's instance and
        sequence, and a monotonic nanosecond suffix. The instance and
        sequence keep separate processes from colliding when their clocks
        return the same timestamp.

    Args:
        path: Absolute path to the ``token-usage.jsonl`` file. Parent
            directories are created lazily on first write.
        data_dir: Optional host data root. When supplied, the handler
            loads (or atomically creates) the per-root pseudonym secret
            and stamps ``thread_pseudonym_id`` on every row. ``None``
            keeps the handler unattributed: rows persist with
            ``thread_pseudonym_id=None`` and no raw slug.
        filter_fn: Optional predicate that receives the event and
            returns ``True`` to persist, ``False`` to skip. Hosts use
            this to drop events below a severity threshold or to
            redact sensitive fields before persistence.
        lock_factory: Optional callable returning a context manager that
            serializes the append across processes sharing one file (the
            runtime injects ``sdk.filelock.file_lock`` over a sidecar
            ``.lock``). ``None`` keeps appends unsynchronized, as before.
    """

    def __init__(
        self,
        path: Path,
        *,
        data_dir: Path | None = None,
        filter_fn: Any | None = None,
        lock_factory: Callable[[], AbstractContextManager[None]] | None = None,
    ) -> None:
        self._path = path
        self._filter_fn = filter_fn
        self._lock_factory = lock_factory or contextlib.nullcontext
        # Pseudonym secret: loaded once per handler instance. ``None`` =
        # unavailable; rows persist unattributed (fail closed) rather than
        # carrying a raw identity or a fabricated pseudonym.
        self._pseudonym_secret = _resolve_pseudonym_secret(data_dir) if data_dir else None
        # Per-instance sequence + identity: the token ledger is shared by
        # every process writing one data dir, so a global counter is not
        # comparable across writers. ``sequence`` orders this instance's
        # rows and ``ledger_instance`` names this instance, letting a
        # forensic reader detect dropped rows *within* one writer's run
        # (a restart or a second process starts a new instance at 1).
        self._sequence = 0
        self._instance = uuid.uuid4().hex[:12]

    @property
    def path(self) -> Path:
        """Return the resolved JSONL path."""
        return self._path

    @property
    def instance(self) -> str:
        """Return this handler's instance id (stamped on every row)."""
        return self._instance

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
        self._sequence += 1
        if not event.idempotency_key:
            event.idempotency_key = (
                f"{event.emitted_at}:{self._instance}:{self._sequence}:{_time_ns_suffix()}"
            )
        # Handler-stamped loss-detection fields, same posture as the
        # idempotency key: publishers never set them. A reader detects
        # dropped rows by finding sequence gaps within one
        # ``ledger_instance``.
        event.sequence = self._sequence
        event.ledger_instance = self._instance
        # Pseudonymous attribution: stamp the durable pseudonym when the
        # per-root key is available, then strip the raw slug so it never
        # reaches disk. No key -> the row stays unattributed (``None``),
        # never carrying a raw identity or a fabricated pseudonym.
        if self._pseudonym_secret is not None and event.thread_store_slug:
            event.thread_pseudonym_id = thread_pseudonym(
                event.thread_store_slug, self._pseudonym_secret
            )
        event.thread_store_slug = ""
        try:
            with self._lock_factory():
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(_ledger_dump(event) + "\n")
        except (OSError, TimeoutError) as exc:
            # TimeoutError covers the host-injected lock timing out (dropping
            # one line is better than stalling the turn).
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
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    cached_tokens: int | None = None,
    reasoning_tokens: int | None = None,
    cache_creation_tokens: int | None = None,
    duration_ms: float | None = None,
    latency_ms: int | None = None,
    cost_usd: float | None = None,
    time_cost_usd: float | None = None,
    provider_cost_usd: float | None = None,
    effective_model_data_source: str | None = None,
    effective_model_data_version: str | None = None,
    request_id: str = "",
    mode_id: str = "",
    stage_id: str = "",
    turn_index: int | None = None,
    call_site: str = "",
    conversation_id: str = "",
    user_slug: str = "default",
    thread_store_slug: str = "",
    surface_id: str = "",
    proxy_telemetry: dict[str, int] | None = None,
    extra: dict[str, Any] | None = None,
    metric_origins: dict[str, str] | None = None,
) -> TokenUsageEvent:
    """Build one ``TokenUsageEvent`` preserving unknown-versus-zero.

    Hosts call this from each LLM call site so every provider call produces
    exactly one durable record. v2 semantics: a ``None`` argument means the
    provider never surfaced that value and the field persists as unknown;
    an explicit ``0`` is a real measured zero and is kept. ``total_tokens``
    is derived from ``prompt + completion`` only when both are known and no
    explicit total was supplied (a supplied total, including ``0``, is
    authoritative); the derivation is recorded in ``metric_origins``.

    ``proxy_telemetry`` carries the typed pack-owned proxy counters (e.g.
    ``{"tokens_saved": 40000}``); they persist through the durable
    allowlist, unlike the untyped ``extra`` bag, which is dropped at
    append time and exists only for in-process joins.

    ``effective_model_data_source`` / ``effective_model_data_version``
    stamp where the effective model data for this call came from and its
    immutable version (v3 provenance fields, allowlisted since v3 but
    never populated until Workstream 1). ``None`` / empty stays unknown:
    the factory never invents a source or a version, and producers pass
    values only from the model-composition seam when it actually knows
    them.

    The ``kind`` literal discriminates billing shape:
        - ``chat_completion`` / ``embedding`` / ``image_generation`` /
          ``stream_chunk``: token-based billing.
        - ``time_billed``: duration-based billing (e.g. Ollama local
          GPU time). ``duration_ms`` and ``time_cost_usd`` carry the
          billable signal; token counters may still be populated for
          observability even when cost is time-derived.
    """
    origins: dict[str, str] = dict(metric_origins or {})
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
        origins.setdefault("total_tokens", "derived")
    return TokenUsageEvent(
        provider=provider,
        model_id=model_id,
        kind=kind,  # type: ignore[arg-type]
        prompt_tokens=nullable_int(prompt_tokens),
        completion_tokens=nullable_int(completion_tokens),
        total_tokens=nullable_int(total_tokens),
        cached_tokens=nullable_int(cached_tokens),
        reasoning_tokens=nullable_int(reasoning_tokens),
        cache_creation_tokens=nullable_int(cache_creation_tokens),
        duration_ms=nullable_float(duration_ms),
        latency_ms=nullable_int(latency_ms),
        cost_usd=nullable_float(cost_usd),
        time_cost_usd=nullable_float(time_cost_usd),
        provider_cost_usd=nullable_float(provider_cost_usd),
        effective_model_data_source=effective_model_data_source or None,
        effective_model_data_version=effective_model_data_version or None,
        proxy_telemetry=dict(proxy_telemetry) if proxy_telemetry else {},
        metric_origins=origins,
        request_id=str(request_id or ""),
        mode_id=str(mode_id or ""),
        stage_id=str(stage_id or ""),
        turn_index=resolve_turn_index(turn_index),
        call_site=str(call_site or ""),
        conversation_id=str(conversation_id or ""),
        user_slug=str(user_slug or "default"),
        thread_store_slug=str(thread_store_slug or ""),
        surface_id=str(surface_id or ""),
        extra=dict(extra) if extra else {},
    )


def _cache_split(
    prompt: int, cached_tokens: int | None, cache_creation_tokens: int | None
) -> tuple[int, int, int] | None:
    """Resolve the prompt-side split to ``(uncached, cached, writes)``.

    Sub-counters already ride inside ``prompt_tokens``, so the uncached
    remainder is what is left after removing them. A wholly missing split
    is just ``(prompt, 0, 0)`` — the provider gave no basis for a discount,
    so the full prompt bills at the prompt rate. ``None`` flags inconsistent
    provider data (the split exceeds the prompt): refuse rather than clamp.
    """

    cached = max(0, int(cached_tokens or 0))
    writes = max(0, int(cache_creation_tokens or 0))
    if cached + writes > prompt:
        return None
    return prompt - cached - writes, cached, writes


def _cache_rates(
    cached: int, writes: int, read_rate: float | None, write_rate: float | None
) -> tuple[float, float] | None:
    """Resolve the cache rates for a reported split, or ``None`` if unknown.

    A positive count with an unknown rate makes the whole call unpriced:
    pricing it at the prompt rate fabricates the very discount the split
    exists to measure. Zero-count components never block — nothing
    cost-bearing exists there — and an explicit ``0.0`` is a real free
    rate, not missing data.
    """

    if (cached and read_rate is None) or (writes and write_rate is None):
        return None
    return read_rate or 0.0, write_rate or 0.0


def estimate_call_cost_usd(
    *,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    cached_tokens: int | None = None,
    cache_creation_tokens: int | None = None,
    prompt_price: float | None = None,
    completion_price: float | None = None,
    cache_read_price: float | None = None,
    cache_write_price: float | None = None,
) -> float | None:
    """Price one completion from per-token USD rates, or ``None`` when unpriced.

    This is the *canonical* cost model every host shares: the same math the
    CLI footer used to re-derive locally, but with the prompt-cache split
    applied to both sides of the cache rather than just reads.

    §4 sentinel semantics: rates are ``float | None`` — ``None`` means the
    provider published no rate (unknown) and an explicit ``0.0`` means a
    genuinely free tier. The return preserves that distinction:

    - Any cost-bearing component whose rate is unknown makes the whole
      call unknown (``None``). A missing cache rate is NEVER fallen back
      to the prompt rate: the discounted/premium cache basis is exactly
      what the split exists to bill, so guessing would fabricate a number.
    - Explicit ``0.0`` rates price a real ``0.0`` — a free tier is a real
      answer, not missing data.
    - A zero-token call is ``0.0`` regardless of rates: nothing
      cost-bearing exists, so the rates are moot.

    The prompt side is a three-way split: a cache hit bills at the read
    rate while a cache *population* bills at a premium write rate.
    ``cached_tokens``/``cache_creation_tokens`` are provider-reported
    sub-counters already included in ``prompt_tokens``, so the uncached
    remainder is what is left after removing them. A missing split (both
    sub-counters unknown) prices the full prompt at the prompt rate — the
    provider gave no basis for a discount. A split exceeding the prompt is
    inconsistent provider data and refuses to price rather than clamping.
    """

    if prompt_tokens is None or completion_tokens is None:
        return None
    prompt = max(0, int(prompt_tokens))
    completion = max(0, int(completion_tokens))
    if prompt == 0 and completion == 0:
        # Nothing cost-bearing exists; unknown rates are moot, not blocking.
        return 0.0
    if (prompt and prompt_price is None) or (completion and completion_price is None):
        # A component with a positive token count needs a sourced rate;
        # unknown cannot yield a cost — not a zero, not a partial. A
        # zero-count component bills nothing, so its rate never blocks.
        return None

    split = _cache_split(prompt, cached_tokens, cache_creation_tokens)
    if split is None:
        # Inconsistent provider data: the split exceeds the prompt. Refuse
        # to price rather than silently clamping and calling it an estimate.
        return None
    uncached, cached, writes = split

    rates = _cache_rates(cached, writes, cache_read_price, cache_write_price)
    if rates is None:
        return None
    read_rate, write_rate = rates

    return max(
        0.0,
        # The ``or 0.0`` below only ever multiplies a zero-count component
        # (a positive count with an unknown rate returned None above), so
        # it zeroes nothing that bills — not the v1 sentinel collapse.
        uncached * (prompt_price or 0.0)
        + completion * (completion_price or 0.0)
        + cached * read_rate
        + writes * write_rate,
    )

"""Report-time cache measurement over the durable token-usage ledger.

Runtime half of SPEC Workstream 1: ``/tokens`` (and only ``/tokens``)
calls :func:`report_cache`, which incrementally ingests the ledger under
integrity checks, folds it into honest per-partition cache metrics, and
renders a focused report section. Nothing here runs inside a model turn.

Integrity model (see the Workstream 1 design):

- The checkpoint stores the consumed byte offset, the ledger size and
  inode it was taken against, and the SHA-256 of every consumed byte.
  Each report re-hashes the consumed prefix (O(offset) I/O — the price
  of detecting same-size rewrites) and resets on any mismatch.
- Rotation/truncation/copytruncate invalidates the checkpoint; the new
  file is replayed from zero and the lost window is reported as
  continuity-incomplete, never silently merged.
- New state is written atomically (temp file + rename) only after a
  successful read; a crash leaves the previous checkpoint intact.
- Every failure degrades to ``unavailable``/``incomplete`` status lines,
  never to fabricated zeros and never to a turn-blocking error.

Privacy: the checkpoint stores only counts, partition labels, nullable
sums, window timestamps and idempotency keys — no prompts, no tool
arguments, no raw thread identity.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from kernel.cache_metrics import (
    CACHE_HIT_METRIC,
    MAX_WILSON_HALF_WIDTH,
    MIN_ELIGIBLE_SAMPLES,
    OBSERVATION_WINDOW_DAYS,
    SNAPSHOT_EXPIRY_DAYS,
    CacheMetricsInputError,
    PartitionKey,
    PartitionMetrics,
    build_partition_report,
    classify_usage_record,
)
from sdk.filelock import atomic_write_text, file_lock, sidecar_lock_path

logger = logging.getLogger(__name__)
_STATE_SCHEMA_VERSION = 1
_MEASURED_STORE_SCHEMA_VERSION = 1
_REPORT_LOCK_TIMEOUT_S = 5.0
_LEDGER_LOCK_TIMEOUT_S = 5.0
#: Hard ceiling for incremental mode. A ledger larger than this is
#: reported ``unavailable`` rather than half-verified; realistic ledgers
#: are orders of magnitude below it.
_MAX_INCREMENTAL_BYTES = 1 << 30
_READ_CHUNK = 1 << 20

__all__ = [
    "report_cache",
    "update_measured_store",
]


# ── Time helpers ─────────────────────────────────────────────────────────────


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_utc(value: Any) -> datetime | None:
    """Parse one ISO-8601 timestamp; naive values are read as UTC."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


# ── Checkpoint state ─────────────────────────────────────────────────────────


@dataclass(slots=True)
class _Checkpoint:
    """Validated incremental-reader state (also the on-disk JSON shape)."""

    offset: int = 0
    source_size: int = 0
    prefix_sha256: str = ""
    contributions: list[dict[str, Any]] = field(default_factory=list)
    excluded: dict[str, dict[str, int]] = field(default_factory=dict)
    diagnostics: dict[str, int] = field(default_factory=dict)
    continuity_lost_at: str | None = None
    last_read_utc: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {
                "schema_version": _STATE_SCHEMA_VERSION,
                "offset": self.offset,
                "source_size": self.source_size,
                "prefix_sha256": self.prefix_sha256,
                "contributions": self.contributions,
                "excluded": self.excluded,
                "diagnostics": self.diagnostics,
                "continuity_lost_at": self.continuity_lost_at,
                "last_read_utc": self.last_read_utc,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )


_STATE_KEYS = {
    "offset",
    "source_size",
    "prefix_sha256",
    "contributions",
    "excluded",
    "diagnostics",
    "continuity_lost_at",
    "last_read_utc",
}


def _checkpoint_is_valid(raw: Any) -> bool:
    """Structural validation for one deserialized checkpoint document."""
    if not isinstance(raw, dict) or raw.get("schema_version") != _STATE_SCHEMA_VERSION:
        return False
    if not _STATE_KEYS.issubset(raw):
        return False
    if not isinstance(raw["offset"], int) or raw["offset"] < 0:
        return False
    if not isinstance(raw["contributions"], list) or not isinstance(raw["excluded"], dict):
        return False
    return isinstance(raw["diagnostics"], dict) and isinstance(raw["prefix_sha256"], str)


def _load_checkpoint(path: Path) -> _Checkpoint | None:
    """Load a checkpoint; corrupt/incompatible state rebuilds (``None``)."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not _checkpoint_is_valid(raw):
        return None
    assert isinstance(raw, dict)
    return _Checkpoint(
        offset=raw["offset"],
        source_size=raw["source_size"],
        prefix_sha256=raw["prefix_sha256"],
        contributions=raw["contributions"],
        excluded=raw["excluded"],
        diagnostics=raw["diagnostics"],
        continuity_lost_at=raw["continuity_lost_at"],
        last_read_utc=raw["last_read_utc"],
    )


def _save_checkpoint(path: Path, state: _Checkpoint) -> None:
    atomic_write_text(path, state.to_json() + "\n")


def _stat_identity(path: Path) -> tuple[int, int, int] | None:
    """Return ``(st_dev, st_ino, st_size)``, or ``None`` when unavailable."""
    try:
        info = path.stat()
    except OSError:
        return None
    return (info.st_dev, info.st_ino, info.st_size)


# ── Ingestion ────────────────────────────────────────────────────────────────


def partition_key_of(record: dict[str, Any]) -> PartitionKey:
    """Re-exported kernel partitioner (kept local for a narrow import)."""
    from kernel.cache_metrics import partition_key_from_record

    return partition_key_from_record(record)


def _key_tuple(
    key: PartitionKey,
) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    return (key.provider, key.model_id, key.actual_route, key.cache_policy, key.prefix_id)


def _key_label(key: PartitionKey) -> str:
    """Canonical, privacy-safe partition label (provider/model/policy)."""
    provider = key.provider or "unknown"
    model = key.model_id or "unknown"
    policy = key.cache_policy or "unknown"
    return f"{provider}/{model}/policy={policy}"


def _row_contribution(
    record: dict[str, Any], row: Any, event_time: datetime
) -> dict[str, Any] | None:
    """Reduce an eligible row to its minimal window contribution."""
    key = partition_key_of(record)
    return {
        "t": _iso(event_time),
        "pr": key.provider,
        "mo": key.model_id,
        "pol": key.cache_policy,
        "pt": row.prompt_tokens,
        "ct": row.cached_tokens,
        "c": row.cost_usd,
        "pc": row.provider_cost_usd,
        "a": 1 if row.ambiguous_provenance else 0,
        "k": str(record.get("idempotency_key") or ""),
    }


def _excluded_bucket(record: dict[str, Any], row: Any) -> tuple[str, str]:
    key = partition_key_of(record)
    return (_key_label(key), row.exclusion_reason or "unknown")


def _resume_offset(data: bytes, start_offset: int, diagnostics: dict[str, int]) -> tuple[int, int]:
    """Return the record-aligned resume cursor and the complete-data end.

    A clean boundary sits at a newline-terminated record edge; a mid-line
    start (a truncated previous read) rewinds to the last record boundary
    before it, and that one record is consumed again. Idempotency keys
    de-duplicate keyed rows; keyless rows would be counted twice, which
    the diagnostics make visible.
    """
    complete_end = data.rfind(b"\n") + 1
    cursor = min(start_offset, complete_end)
    if cursor > 0 and data[cursor - 1 : cursor] != b"\n":
        cursor = data.rfind(b"\n", 0, cursor) + 1
        diagnostics["rewound_partial_lines"] = diagnostics.get("rewound_partial_lines", 0) + 1
    return cursor, complete_end


def _classify_line(raw_line: bytes) -> tuple[dict[str, Any] | None, str | None]:
    """Decode and validate one ledger line.

    Returns ``(record, None)`` when the line is a usable usage record and
    ``(None, diagnostic_name)`` when it is blank/undecodable/malformed
    (with the diagnostic counter to bump). A line that is a valid JSON
    object but not interpretable under the measurement contract is also
    malformed by definition — the caller counts it and marks the window
    incomplete.
    """
    try:
        text = raw_line.decode("utf-8")
    except UnicodeDecodeError:
        return None, "invalid_utf8_lines"
    text = text.strip()
    if not text:
        return None, "blank_lines"
    try:
        record = json.loads(text)
    except ValueError:
        return None, "malformed_lines"
    if not isinstance(record, dict):
        return None, "malformed_lines"
    try:
        classify_usage_record(record)
    except CacheMetricsInputError:
        return None, "malformed_lines"
    return record, None


def _absorb_row(
    state: _Checkpoint,
    record: dict[str, Any],
    diagnostics: dict[str, int],
    seen_keys: set[str],
    now: datetime,
) -> tuple[datetime | None, str | None]:
    """Fold one already-classified usable row into the checkpoint.

    Returns ``(event_time, exclusion_or_none)``: ``event_time`` is set
    only when the row became a window contribution; the second element
    names the exclusion bucket for non-eligible rows (also ``None`` for
    duplicates, which are counted as diagnostics instead).
    """
    event_time = _parse_utc(record.get("emitted_at"))
    if event_time is None:
        diagnostics["invalid_timestamp_rows"] = diagnostics.get("invalid_timestamp_rows", 0) + 1
        return None, "__invalid_time"
    if event_time > now + timedelta(minutes=5):
        diagnostics["future_timestamp_rows"] = diagnostics.get("future_timestamp_rows", 0) + 1
        return None, "__future_time"
    row = classify_usage_record(record)
    if not row.eligible:
        bucket, reason = _excluded_bucket(record, row)
        buckets = state.excluded.setdefault(bucket, {})
        buckets[reason] = buckets.get(reason, 0) + 1
        return event_time, reason
    key = str(record.get("idempotency_key") or "")
    if key:
        if key in seen_keys:
            diagnostics["duplicate_keys"] = diagnostics.get("duplicate_keys", 0) + 1
            return event_time, None
        seen_keys.add(key)
    state.contributions.append(_row_contribution(record, row, event_time))
    return event_time, None


def _ingest_lines(
    state: _Checkpoint,
    data: bytes,
    start_offset: int,
    now: datetime,
) -> tuple[int, bool, datetime | None, datetime | None, dict[str, int]]:
    """Parse complete newline-terminated records from ``data``.

    Returns the new offset, whether the source stayed complete, the
    min/max contribution timestamps observed in this read, and the
    per-call diagnostic counters. Malformed lines, bad UTF-8, and
    unusable timestamps are counted and skipped — they make the affected
    window incomplete, never a zero.
    """
    diagnostics: dict[str, int] = {}
    source_complete = True
    earliest: datetime | None = None
    latest: datetime | None = None
    cursor, complete_end = _resume_offset(data, start_offset, diagnostics)
    offset = cursor
    seen_keys = {
        item.get("k")
        for item in state.contributions
        if isinstance(item.get("k"), str) and item.get("k")
    }
    while cursor < complete_end:
        end = data.find(b"\n", cursor)
        if end == -1:  # pragma: no cover - complete_end bounds this loop
            break
        raw_line = data[cursor:end]
        cursor = end + 1
        offset = cursor
        record, diagnostic = _classify_line(raw_line)
        if record is None:
            assert diagnostic is not None
            diagnostics[diagnostic] = diagnostics.get(diagnostic, 0) + 1
            if diagnostic != "blank_lines":
                source_complete = False
            continue
        event_time, _excluded = _absorb_row(state, record, diagnostics, seen_keys, now)
        if event_time is None:
            continue
        if earliest is None or event_time < earliest:
            earliest = event_time
        if latest is None or event_time > latest:
            latest = event_time
    return offset, source_complete, earliest, latest, diagnostics


@dataclass(slots=True)
class _Ingestion:
    status: str
    source_complete: bool
    partitions: dict[tuple[Any, ...], PartitionMetrics] = field(default_factory=dict)
    keys: dict[tuple[Any, ...], PartitionKey] = field(default_factory=dict)
    diagnostics: dict[str, int] = field(default_factory=dict)
    excluded: dict[str, dict[str, int]] = field(default_factory=dict)
    window_start: datetime | None = None
    window_end: datetime | None = None
    reason: str = ""


def _fold_contributions(
    state: _Checkpoint,
) -> tuple[
    dict[tuple[Any, ...], PartitionMetrics],
    dict[tuple[Any, ...], PartitionKey],
    datetime | None,
    datetime | None,
]:
    partitions: dict[tuple[Any, ...], PartitionMetrics] = {}
    keys: dict[tuple[Any, ...], PartitionKey] = {}
    earliest: datetime | None = None
    latest: datetime | None = None
    for item in state.contributions:
        key = PartitionKey(
            provider=item.get("pr"),
            model_id=item.get("mo"),
            actual_route=None,
            cache_policy=item.get("pol"),
            prefix_id=None,
        )
        tuple_key = _key_tuple(key)
        keys.setdefault(tuple_key, key)
        metrics = partitions.setdefault(tuple_key, PartitionMetrics())
        metrics.row_count += 1
        metrics.eligible_count += 1
        cached = item.get("ct") or 0
        prompt = item.get("pt") or 0
        if cached > 0:
            metrics.hits += 1
        else:
            metrics.observed_zero_count += 1
        metrics.prompt_sum += prompt
        metrics.cached_sum += cached
        if item.get("a"):
            metrics.ambiguous_rows += 1
        cost = item.get("c")
        if cost is not None:
            metrics.cost_known_count += 1
            metrics.cost_sum += float(cost)
        provider_cost = item.get("pc")
        if provider_cost is not None:
            metrics.provider_cost_known_count += 1
            metrics.provider_cost_sum += float(provider_cost)
        event_time = _parse_utc(item.get("t"))
        if event_time is not None:
            if earliest is None or event_time < earliest:
                earliest = event_time
            if latest is None or event_time > latest:
                latest = event_time
    return partitions, keys, earliest, latest


def _guarded_unavailable(reason: str) -> _Ingestion:
    """Uniform degraded result: never a figure, never a zero."""
    return _Ingestion(status="unavailable", source_complete=False, reason=reason)


def _locked_read(ledger_path: Path, st_dev: int, st_ino: int) -> tuple[bytes, int] | _Ingestion:
    """Read the ledger under the writer's timed sidecar lock.

    Returns ``(data, size)`` or a degraded :class:`_Ingestion` when the
    lock times out, the identity shifts under the lock, or the read
    fails. The lock is held only for the bounded read, never during
    parsing, so appends never wait on report rendering.
    """
    try:
        with file_lock(sidecar_lock_path(ledger_path), timeout=_LEDGER_LOCK_TIMEOUT_S):
            locked_identity = _stat_identity(ledger_path)
            if locked_identity is None or locked_identity[:2] != (st_dev, st_ino):
                return _guarded_unavailable("ledger_changed")
            locked_size = locked_identity[2]
            if locked_size > _MAX_INCREMENTAL_BYTES:
                return _guarded_unavailable("ledger_too_large")
            with ledger_path.open("rb") as handle:
                return handle.read(locked_size), locked_size
    except TimeoutError:
        return _guarded_unavailable("ledger_lock_timeout")
    except OSError:
        return _guarded_unavailable("ledger_unreadable")


def _capture_stable_view(
    ledger_path: Path,
) -> tuple[bytes, tuple[int, int], int] | _Ingestion:
    """Capture a bounded, lock-held view of the ledger.

    Returns ``(data, (st_dev, st_ino), size)`` or a degraded
    :class:`_Ingestion` when the source cannot be captured stably. The
    file identity is re-checked after the lock releases: an inconsistent
    post-state proves the view unreliable and forbids advancing the
    checkpoint.
    """
    identity = _stat_identity(ledger_path)
    if identity is None:
        return _guarded_unavailable("ledger_unreadable")
    st_dev, st_ino, pre_size = identity
    if pre_size > _MAX_INCREMENTAL_BYTES:
        return _guarded_unavailable("ledger_too_large")
    locked = _locked_read(ledger_path, st_dev, st_ino)
    if isinstance(locked, _Ingestion):
        return locked
    data, size = locked
    post = _stat_identity(ledger_path)
    if post is None or post[:2] != (st_dev, st_ino) or post[2] != size:
        return _Ingestion(
            status="incomplete", source_complete=False, reason="source_changed_during_read"
        )
    return data, (st_dev, st_ino), size


def _verify_consumed_prefix(
    ledger_path: Path,
    state: _Checkpoint | None,
    size: int,
    diagnostics: dict[str, int],
) -> _Checkpoint | None:
    """Re-hash the consumed prefix; an unproven prefix resets the state.

    Same-size rewrites and corrupted prefixes cannot hide from the
    digest; truncation and rotation were already ruled out by the
    identity/size guards before this runs.
    """
    if state is None:
        return None
    if state.offset == 0:
        return state
    if not _state_prefix_is_current(state, size):
        return None
    digest = hashlib.sha256()
    remaining = state.offset
    try:
        with ledger_path.open("rb") as handle:
            while remaining > 0:
                chunk = handle.read(min(_READ_CHUNK, remaining))
                if not chunk:
                    break
                digest.update(chunk)
                remaining -= len(chunk)
    except OSError:
        diagnostics["prefix_rewrites"] = diagnostics.get("prefix_rewrites", 0) + 1
        return None
    if remaining != 0 or digest.hexdigest() != state.prefix_sha256:
        diagnostics["prefix_rewrites"] = diagnostics.get("prefix_rewrites", 0) + 1
        return None
    return state


def _state_prefix_is_current(state: _Checkpoint, size: int) -> bool:
    """False when the consumed prefix provably no longer exists."""
    if state.source_size and size < state.offset:
        return False  # truncation or copytruncate reset the consumed prefix
    return state.offset <= size


def _advance_checkpoint(
    state: _Checkpoint,
    state_path: Path,
    *,
    offset: int,
    size: int,
    prefix_sha256: str,
    now: datetime,
    continuity_lost_at: str | None,
    diagnostics: dict[str, int],
) -> None:
    """Age the window, fold diagnostics, and persist state atomically."""
    cutoff = now - timedelta(days=OBSERVATION_WINDOW_DAYS)
    kept = [item for item in state.contributions if (_parse_utc(item.get("t")) or now) >= cutoff]
    if len(kept) != len(state.contributions):
        diagnostics["aged_out_rows"] = diagnostics.get("aged_out_rows", 0) + (
            len(state.contributions) - len(kept)
        )
        state.contributions = kept
    state.offset = offset
    state.source_size = size
    state.prefix_sha256 = prefix_sha256
    state.continuity_lost_at = continuity_lost_at
    for name, count in diagnostics.items():
        state.diagnostics[name] = state.diagnostics.get(name, 0) + count
    _save_checkpoint(state_path, state)


def _incremental_read(
    ledger_path: Path,
    state_path: Path,
    now: datetime,
) -> _Ingestion:
    """Run one bounded, integrity-checked incremental read."""
    if _stat_identity(ledger_path) is None:
        return _guarded_unavailable("ledger_unreadable")
    identity = _stat_identity(ledger_path)
    assert identity is not None
    size = identity[2]
    if size == 0:
        _save_checkpoint(state_path, _Checkpoint(last_read_utc=_iso(now)))
        return _Ingestion(status="ok", source_complete=True, reason="empty_ledger")

    state = _load_checkpoint(state_path)
    diagnostics: dict[str, int] = {}
    continuity_lost_at = state.continuity_lost_at if state else None

    captured = _capture_stable_view(ledger_path)
    if isinstance(captured, _Ingestion):
        return captured
    data, _, size = captured

    had_checkpoint = state is not None
    state = _verify_consumed_prefix(ledger_path, state, size, diagnostics)
    fresh = state is None
    if fresh:
        state = _Checkpoint(last_read_utc=_iso(now))
        if had_checkpoint:
            # A verified checkpoint failed its integrity check (rotation,
            # truncation, rewrite): the prior file's rows are gone, so the
            # 14-day window coverage is incomplete until a full window
            # passes over the new file. A first-ever read sees the whole
            # current file and is not a continuity loss.
            continuity_lost_at = continuity_lost_at or _iso(now)
    else:
        assert state is not None
        state.last_read_utc = _iso(now)

    offset, lines_complete, earliest, latest, per_call = _ingest_lines(
        state, data, state.offset, now
    )
    for name, count in per_call.items():
        diagnostics[name] = diagnostics.get(name, 0) + count
    source_complete = lines_complete

    _advance_checkpoint(
        state,
        state_path,
        offset=offset,
        size=size,
        prefix_sha256=hashlib.sha256(data[:offset]).hexdigest(),
        now=now,
        continuity_lost_at=continuity_lost_at,
        diagnostics=diagnostics,
    )

    partitions, keys, win_start, win_end = _fold_contributions(state)
    window_lost = continuity_lost_at is not None and (
        _parse_utc(continuity_lost_at) or now
    ) > now - timedelta(days=OBSERVATION_WINDOW_DAYS)
    return _Ingestion(
        status="ok",
        source_complete=source_complete and not window_lost,
        partitions=partitions,
        keys=keys,
        diagnostics=dict(state.diagnostics),
        excluded=dict(state.excluded),
        window_start=win_start or earliest,
        window_end=win_end or latest,
    )


def ingest_ledger(
    ledger_path: Path, state_path: Path, *, now: datetime | None = None
) -> _Ingestion:
    """Public entry: one incremental, locked, integrity-checked read."""
    now = now or _utc_now()
    try:
        with file_lock(sidecar_lock_path(state_path), timeout=_REPORT_LOCK_TIMEOUT_S):
            return _incremental_read(ledger_path, state_path, now)
    except TimeoutError:
        return _Ingestion(status="unavailable", source_complete=False, reason="state_lock_timeout")
    except OSError:
        return _Ingestion(status="unavailable", source_complete=False, reason="state_unwritable")
    except Exception:  # pylint: disable=broad-except
        logger.exception("cache report ingestion failed")
        return _Ingestion(status="unavailable", source_complete=False, reason="internal_error")


# ── Measured-layer store ─────────────────────────────────────────────────────


def load_measured_store(path: Path) -> dict[str, Any]:
    """Load the measured-store document, or a fresh empty one."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"schema_version": _MEASURED_STORE_SCHEMA_VERSION, "entries": []}
    if not isinstance(raw, dict) or raw.get("schema_version") != _MEASURED_STORE_SCHEMA_VERSION:
        return {"schema_version": _MEASURED_STORE_SCHEMA_VERSION, "entries": []}
    if not isinstance(raw.get("entries"), list):
        return {"schema_version": _MEASURED_STORE_SCHEMA_VERSION, "entries": []}
    return raw


def update_measured_store(
    path: Path,
    entries: list[dict[str, Any]],
    *,
    now: datetime | None = None,
) -> None:
    """Merge qualified observations into the measured store atomically.

    Only callers that already hold qualified reports (none in this slice —
    the ledger has no prefix identity) produce entries. Expired entries
    are pruned on every successful update. Unqualified diagnostics never
    reach this store.

    ``entries`` carrying ``provider=None``/``model_id=None`` identities
    are rejected: unknown dimensions are never persisted as comparable
    observations.
    """
    now = now or _utc_now()
    for entry in entries:
        if not entry.get("provider") or not entry.get("model_id"):
            raise ValueError("measured-layer entries require known provider and model_id")
    with file_lock(sidecar_lock_path(path), timeout=_REPORT_LOCK_TIMEOUT_S):
        doc = load_measured_store(path)
        kept = [
            entry
            for entry in doc["entries"]
            if isinstance(entry, dict) and (_parse_utc(entry.get("expires_at_utc")) or now) > now
        ]
        merged = {json.dumps(_entry_identity(entry), sort_keys=True): entry for entry in kept}
        for entry in entries:
            # Incoming already-expired observations are dropped, not stored.
            if (_parse_utc(entry.get("expires_at_utc")) or now) <= now:
                continue
            merged[json.dumps(_entry_identity(entry), sort_keys=True)] = entry
        doc["entries"] = list(merged.values())
        doc["updated_at_utc"] = _iso(now)
        atomic_write_text(path, json.dumps(doc, ensure_ascii=False, indent=1) + "\n")


def _entry_identity(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": entry.get("provider"),
        "model_id": entry.get("model_id"),
        "actual_route": entry.get("actual_route"),
        "cache_policy": entry.get("cache_policy"),
        "prefix_id": entry.get("prefix_id"),
        "metric": entry.get("metric"),
    }


# ── Rendering ────────────────────────────────────────────────────────────────


def _fmt_pct(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "n/a"


def render_cache_section(ingestion: _Ingestion, now: datetime) -> list[str]:
    """Render the focused cache section for ``/tokens``."""
    lines = ["", "### Cache (last 14 days)"]
    if ingestion.status != "ok":
        lines.append(f"- Status: unavailable ({ingestion.reason}); no figures invented")
        return lines
    status = "complete" if ingestion.source_complete else "incomplete"
    start_label = _iso(ingestion.window_start) if ingestion.window_start else "n/a"
    end_label = _iso(ingestion.window_end) if ingestion.window_end else "n/a"
    window = f"{start_label} → {end_label}"
    lines.append(f"- Window: {window} ({status} source)")
    diagnostics = ingestion.diagnostics
    if diagnostics:
        shown = ", ".join(f"{name}={count}" for name, count in sorted(diagnostics.items()) if count)
        if shown:
            lines.append(f"- Diagnostics: {shown}")
    for label, buckets in sorted(ingestion.excluded.items()):
        shown = ", ".join(f"{reason}:{count}" for reason, count in sorted(buckets.items()))
        if shown:
            lines.append(f"- Excluded for {label}: {shown}")
    if not ingestion.partitions:
        lines.append("- No eligible cache observations in the window yet")
        lines.append("- Qualified measured layer: no writes (prefix identity unknown)")
        return lines
    lines.append("")
    lines.append("| Partition | Hit proxy | Token share | Ambiguous | Excluded |")
    lines.append("| --- | --- | --- | --- | --- |")
    for tuple_key in sorted(ingestion.partitions, key=lambda k: _key_label(ingestion.keys[k])):
        key = ingestion.keys[tuple_key]
        metrics = ingestion.partitions[tuple_key]
        hit = _fmt_pct(metrics.request_hit_rate)
        share = _fmt_pct(metrics.token_share)
        excluded = (
            ", ".join(
                f"{reason}:{count}" for reason, count in sorted(metrics.excluded_counts.items())
            )
            or "—"
        )
        ambiguous = str(metrics.ambiguous_rows) if metrics.ambiguous_rows else "—"
        lines.append(
            f"| {_key_label(key)} | {metrics.hits}/{metrics.eligible_count} ({hit}) "
            f"| {metrics.cached_sum:,}/{metrics.prompt_sum:,} ({share}) "
            f"| {ambiguous} | {excluded} |"
        )
    unqualified = any(
        metrics.eligible_count < MIN_ELIGIBLE_SAMPLES
        or (metrics.wilson_half_width or 1.0) > MAX_WILSON_HALF_WIDTH
        or not key.identity_known
        for key, metrics in (
            (ingestion.keys[k], ingestion.partitions[k]) for k in ingestion.partitions
        )
    )
    if unqualified:
        lines.append("- Qualified measured layer: no writes (gates unmet or identity unknown)")
    else:  # pragma: no cover - unreachable while prefix identity is absent
        lines.append("- Qualified measured layer: no writes in this slice")
    return lines


def _qualified_entries(ingestion: _Ingestion, now: datetime) -> list[dict[str, Any]]:
    """Qualified measured-layer observations, or (usually) none.

    A partition qualifies only with full provider/model/route/prefix
    identity plus the sample and interval gates over a proven-complete
    source. The ledger supplies no route or prefix identity, so this is
    empty for the whole current ledger — the function keeps the gate
    logic and the store write on one tested path instead of a
    speculative separate one.
    """
    entries: list[dict[str, Any]] = []
    for tuple_key, metrics in ingestion.partitions.items():
        key = ingestion.keys[tuple_key]
        report = build_partition_report(key, metrics, source_complete=ingestion.source_complete)
        if report.status != "qualified":
            continue
        hit_rate = metrics.request_hit_rate
        if hit_rate is None or ingestion.window_end is None:
            continue
        entries.append(
            {
                "provider": key.provider,
                "model_id": key.model_id,
                "actual_route": key.actual_route,
                "cache_policy": key.cache_policy,
                "prefix_id": key.prefix_id,
                "metric": CACHE_HIT_METRIC,
                "method_version": report.method_version,
                "estimate": hit_rate,
                "units": "probability",
                "sample_count": metrics.eligible_count,
                "excluded_counts": dict(metrics.excluded_counts),
                "window_start_utc": _iso(ingestion.window_start)
                if ingestion.window_start
                else None,
                "window_end_utc": _iso(ingestion.window_end),
                "expires_at_utc": _iso(ingestion.window_end + timedelta(days=SNAPSHOT_EXPIRY_DAYS)),
                "generated_at_utc": _iso(now),
            }
        )
    return entries


def report_cache(
    token_usage_path: Path,
    state_path: Path,
    measured_store_path: Path,
) -> list[str]:
    """Produce the ``/tokens`` cache section lines. Never raises."""
    now = _utc_now()
    ingestion = ingest_ledger(token_usage_path, state_path, now=now)
    try:
        entries = _qualified_entries(ingestion, now)
        if entries:
            update_measured_store(measured_store_path, entries, now=now)
        return render_cache_section(ingestion, now)
    except Exception:  # pylint: disable=broad-except
        logger.exception("cache report rendering failed")
        return ["", "### Cache (last 14 days)", "- Status: unavailable (internal error)"]

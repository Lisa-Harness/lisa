"""Workstream 1: runtime ingestion, gating, and /tokens rendering.

Exercises the report-time pipeline: incremental digest-verified
checkpointing, rotation/truncation/rewrite handling, unknown-preserving
metrics, and the focused cache section. All filesystem access runs
through real temp directories; the clock is injected.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from runtime.cache_report import (
    _Ingestion,
    ingest_ledger,
    render_cache_section,
    report_cache,
    update_measured_store,
)


def _ts(days_ago: float = 0.0) -> str:
    now = datetime(2026, 11, 13, 12, 0, 0, tzinfo=UTC)
    return (now - timedelta(days=days_ago)).isoformat().replace("+00:00", "Z")


def _record(
    *,
    cached: int | None = 40,
    prompt: int | None = 100,
    model: str = "m1",
    provider: str = "openai",
    schema: int | None = 3,
    call_site: str = "turn_loop",
    days_ago: float = 0.0,
    kind: str = "chat_completion",
    key: str = "",
    **extra: object,
) -> dict:
    record = {
        "emitted_at": _ts(days_ago),
        "kind": kind,
        "provider": provider,
        "model_id": model,
        "call_site": call_site,
        "schema_version": schema,
        "prompt_tokens": prompt,
        "cached_tokens": cached,
    }
    if key:
        record["idempotency_key"] = key
    record.update(extra)
    return record


def _write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


_NOW = datetime(2026, 11, 13, 12, 0, 0, tzinfo=UTC)


# ── Ingestion ──────────────────────────────────────────────────────────────────


def test_incremental_read_is_idempotent(tmp_path: Path) -> None:
    ledger = tmp_path / "token-usage.jsonl"
    state = tmp_path / "state.json"
    _write(ledger, [_record(key="k1"), _record(key="k2")])

    first = ingest_ledger(ledger, state, now=_NOW)
    assert first.status == "ok"
    assert first.source_complete
    (p1,) = first.partitions.values()
    assert p1.eligible_count == 2

    # No new lines: the second read adds nothing (no double counting).
    second = ingest_ledger(ledger, state, now=_NOW)
    (p2,) = second.partitions.values()
    assert p2.eligible_count == 2


def test_incremental_read_consumes_only_new_lines(tmp_path: Path) -> None:
    ledger = tmp_path / "token-usage.jsonl"
    state = tmp_path / "state.json"
    _write(ledger, [_record(key="k1")])
    ingest_ledger(ledger, state, now=_NOW)

    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_record(key="k2", cached=0, prompt=50)) + "\n")
    result = ingest_ledger(ledger, state, now=_NOW)
    (p,) = result.partitions.values()
    assert p.eligible_count == 2
    assert p.hits == 1
    assert p.observed_zero_count == 1


def test_truncation_resets_and_replays(tmp_path: Path) -> None:
    ledger = tmp_path / "token-usage.jsonl"
    state = tmp_path / "state.json"
    _write(ledger, [_record(key=f"k{i}") for i in range(5)])
    ingest_ledger(ledger, state, now=_NOW)

    # copytruncate-style rewrite: fewer rows than the checkpoint saw.
    _write(ledger, [_record(key="fresh")])
    result = ingest_ledger(ledger, state, now=_NOW)
    (p,) = result.partitions.values()
    assert p.eligible_count == 1
    assert not result.source_complete
    assert "continuity" in (result.reason or "") or not result.source_complete


def test_same_size_rewrite_detected_by_prefix_digest(tmp_path: Path) -> None:
    """A same-size in-place rewrite invalidates the checkpoint via digest."""
    ledger = tmp_path / "token-usage.jsonl"
    state = tmp_path / "state.json"
    _write(ledger, [_record(key="k1", cached=40)])
    ingest_ledger(ledger, state, now=_NOW)

    # Rewrite in place with same byte length but different content.
    replacement = _record(key="k1", cached=41)
    _write(ledger, [replacement])
    result = ingest_ledger(ledger, state, now=_NOW)
    (p,) = result.partitions.values()
    assert p.cached_sum == 41


def test_malformed_lines_make_window_incomplete(tmp_path: Path) -> None:
    ledger = tmp_path / "token-usage.jsonl"
    state = tmp_path / "state.json"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        json.dumps(_record(key="k1")) + "\n" + "{not json}\n",
        encoding="utf-8",
    )
    result = ingest_ledger(ledger, state, now=_NOW)
    assert not result.source_complete
    assert result.diagnostics.get("malformed_lines") == 1


def test_missing_ledger_is_unavailable(tmp_path: Path) -> None:
    result = ingest_ledger(tmp_path / "absent.jsonl", tmp_path / "state.json", now=_NOW)
    assert result.status == "unavailable"


def test_duplicate_keys_counted_once(tmp_path: Path) -> None:
    ledger = tmp_path / "token-usage.jsonl"
    state = tmp_path / "state.json"
    dup = _record(key="k1")
    _write(ledger, [dup, dup])
    result = ingest_ledger(ledger, state, now=_NOW)
    (p,) = result.partitions.values()
    assert p.eligible_count == 1
    assert result.diagnostics.get("duplicate_keys") == 1


def test_aging_prunes_old_contributions(tmp_path: Path) -> None:
    ledger = tmp_path / "token-usage.jsonl"
    state = tmp_path / "state.json"
    _write(
        ledger,
        [
            _record(key="old", days_ago=20),
            _record(key="new", days_ago=1),
        ],
    )
    result = ingest_ledger(ledger, state, now=_NOW)
    (p,) = result.partitions.values()
    assert p.eligible_count == 1


def test_future_timestamps_excluded(tmp_path: Path) -> None:
    ledger = tmp_path / "token-usage.jsonl"
    state = tmp_path / "state.json"
    future = (_NOW + timedelta(days=2)).isoformat().replace("+00:00", "Z")
    rec = _record(key="k1")
    rec["emitted_at"] = future
    _write(ledger, [rec])
    result = ingest_ledger(ledger, state, now=_NOW)
    assert not result.partitions
    assert result.diagnostics.get("future_timestamp_rows") == 1


def test_unknown_and_legacy_cache_rows_excluded_as_coverage(tmp_path: Path) -> None:
    ledger = tmp_path / "token-usage.jsonl"
    state = tmp_path / "state.json"
    _write(
        ledger,
        [
            _record(key="k1", cached=None),
            _record(key="k2", schema=None, cached=0),
            _record(key="k3"),
        ],
    )
    result = ingest_ledger(ledger, state, now=_NOW)
    (p,) = result.partitions.values()
    assert p.eligible_count == 1
    buckets = result.excluded["openai/m1/policy=unknown"]
    assert buckets.get("cache_unknown") == 1
    assert buckets.get("cache_untrusted_zero") == 1


def test_report_renders_unknown_dimension_labels(tmp_path: Path) -> None:
    ledger = tmp_path / "token-usage.jsonl"
    state = tmp_path / "state.json"
    measured = tmp_path / "measured.json"
    _write(ledger, [_record(key="k1", cached=40)])
    lines = report_cache(ledger, state, measured)
    text = "\n".join(lines)
    assert "### Cache (last 14 days)" in text
    assert "openai/m1/policy=unknown" in text
    assert "1/1 (100.0%)" in text
    assert "40/100 (40.0%)" in text


def test_report_handles_unwritable_state_dir(tmp_path: Path) -> None:
    ledger = tmp_path / "logs" / "token-usage.jsonl"
    _write(ledger, [_record(key="k1")])
    blocked = tmp_path / "blocked"
    blocked.write_text("", encoding="utf-8")  # a file, not a directory
    lines = report_cache(ledger, blocked / "state.json", tmp_path / "measured.json")
    text = "\n".join(lines)
    assert "unavailable" in text


# ── Measured store ──────────────────────────────────────────────────────────────


def test_measured_store_merge_and_expiry(tmp_path: Path) -> None:
    store = tmp_path / "measured.json"
    entry = {
        "provider": "p",
        "model_id": "m",
        "actual_route": "direct",
        "cache_policy": "pol",
        "prefix_id": "pre",
        "metric": "cache_request_hit_proxy",
        "estimate": 0.5,
        "window_start_utc": _ts(3),
        "window_end_utc": _ts(1),
        "expires_at_utc": (_NOW - timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
    }
    update_measured_store(store, [entry], now=_NOW)
    doc = json.loads(store.read_text(encoding="utf-8"))
    assert doc["entries"] == []  # expired on arrival → pruned


def test_measured_store_keeps_live_entries(tmp_path: Path) -> None:
    store = tmp_path / "measured.json"
    entry = {
        "provider": "p",
        "model_id": "m",
        "actual_route": "direct",
        "cache_policy": "pol",
        "prefix_id": "pre",
        "metric": "cache_request_hit_proxy",
        "estimate": 0.5,
        "window_start_utc": _ts(3),
        "window_end_utc": _ts(1),
        "expires_at_utc": (_NOW + timedelta(days=1)).isoformat().replace("+00:00", "Z"),
    }
    update_measured_store(store, [entry], now=_NOW)
    doc = json.loads(store.read_text(encoding="utf-8"))
    assert len(doc["entries"]) == 1


# ── Renderer details ────────────────────────────────────────────────────────────


def test_render_empty_ingestion_shows_no_writes() -> None:
    ingestion = _Ingestion(status="ok", source_complete=True)
    lines = render_cache_section(ingestion, _NOW)
    assert "No eligible cache observations in the window yet" in "\n".join(lines)
    assert "no writes" in "\n".join(lines)

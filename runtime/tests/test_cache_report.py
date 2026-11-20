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
from types import SimpleNamespace

import pytest
from runtime.cache_report import (
    _Ingestion,
    ingest_ledger,
    render_cache_section,
    report_cache,
    update_measured_store,
)

from runtime import cache_report


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


def test_parser_and_checkpoint_validation_cover_degraded_inputs(tmp_path: Path) -> None:
    assert cache_report._parse_utc(None) is None
    assert cache_report._parse_utc("not-a-timestamp") is None
    naive = cache_report._parse_utc("2026-11-13T12:00:00")
    assert naive is not None
    assert naive.tzinfo is UTC

    raw = {"schema_version": cache_report._STATE_SCHEMA_VERSION}
    assert cache_report._checkpoint_is_valid(raw) is False
    state_path = tmp_path / "state.json"
    state_path.write_text("{bad json", encoding="utf-8")
    assert cache_report._load_checkpoint(state_path) is None

    diagnostics: dict[str, int] = {}
    assert cache_report._resume_offset(b"abc\n", 2, diagnostics) == (0, 4)
    assert diagnostics["rewound_partial_lines"] == 1
    assert cache_report._classify_line(b"\xff")[1] == "invalid_utf8_lines"
    assert cache_report._classify_line(b"   ")[1] == "blank_lines"
    assert cache_report._classify_line(b"[]")[1] == "malformed_lines"
    assert cache_report._classify_line(b"{bad")[1] == "malformed_lines"
    record, diagnostic = cache_report._classify_line(
        b'{"kind":"chat_completion","prompt_tokens":"bad"}'
    )
    assert record is not None
    assert diagnostic is None


def test_absorb_row_rejects_invalid_and_future_timestamps(tmp_path: Path) -> None:
    del tmp_path
    state = cache_report._Checkpoint()
    diagnostics: dict[str, int] = {}

    event_time, exclusion = cache_report._absorb_row(
        state,
        {"emitted_at": "invalid"},
        diagnostics,
        set(),
        _NOW,
    )
    assert event_time is None
    assert exclusion == "__invalid_time"
    assert diagnostics["invalid_timestamp_rows"] == 1

    event_time, exclusion = cache_report._absorb_row(
        state,
        {"emitted_at": (_NOW + timedelta(days=1)).isoformat()},
        diagnostics,
        set(),
        _NOW,
    )
    assert event_time is None
    assert exclusion == "__future_time"
    assert diagnostics["future_timestamp_rows"] == 1


def test_empty_ledger_saves_checkpoint_and_returns_ok(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    state = tmp_path / "state.json"
    ledger.write_bytes(b"")

    result = ingest_ledger(ledger, state, now=_NOW)

    assert result.status == "ok"
    assert result.source_complete is True
    assert result.reason == "empty_ledger"
    assert state.exists()


def test_capture_stable_view_covers_missing_and_lock_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    absent = tmp_path / "absent.jsonl"
    captured = cache_report._capture_stable_view(absent)
    assert isinstance(captured, _Ingestion)
    assert captured.reason == "ledger_unreadable"

    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps(_record()) + "\n", encoding="utf-8")

    def _timeout(*_args: object, **_kwargs: object) -> None:
        raise TimeoutError

    monkeypatch.setattr(cache_report, "file_lock", _timeout)
    captured = cache_report._capture_stable_view(ledger)
    assert isinstance(captured, _Ingestion)
    assert captured.reason == "ledger_lock_timeout"


def test_capture_stable_view_covers_too_large_and_changed_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        cache_report,
        "_stat_identity",
        lambda _path: (1, 2, cache_report._MAX_INCREMENTAL_BYTES + 1),
    )
    captured = cache_report._capture_stable_view(ledger)
    assert isinstance(captured, _Ingestion)
    assert captured.reason == "ledger_too_large"

    identities = iter([(1, 2, 3), (9, 9, 3)])
    monkeypatch.setattr(cache_report, "_stat_identity", lambda _path: next(identities))
    monkeypatch.setattr(cache_report, "_locked_read", lambda *_args: (b"x\n", 2))
    captured = cache_report._capture_stable_view(ledger)
    assert isinstance(captured, _Ingestion)
    assert captured.reason == "source_changed_during_read"


def test_verify_consumed_prefix_handles_read_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("x", encoding="utf-8")
    state = cache_report._Checkpoint(
        offset=1,
        prefix_sha256="a" * 64,
        source_size=1,
    )

    def _open_error(*_args: object, **_kwargs: object) -> object:
        raise OSError("read failed")

    monkeypatch.setattr(Path, "open", _open_error)
    diagnostics: dict[str, int] = {}

    assert cache_report._verify_consumed_prefix(ledger, state, 1, diagnostics) is None
    assert diagnostics["prefix_rewrites"] == 1


def test_ingest_lock_fallbacks_cover_timeout_oserror_and_internal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    state = tmp_path / "state.json"
    ledger.write_text(json.dumps(_record()) + "\n", encoding="utf-8")

    def _timeout(*_args: object, **_kwargs: object) -> None:
        raise TimeoutError

    monkeypatch.setattr(cache_report, "file_lock", _timeout)
    result = ingest_ledger(ledger, state, now=_NOW)
    assert result.reason == "state_lock_timeout"

    def _oserror(*_args: object, **_kwargs: object) -> None:
        raise OSError("state failed")

    monkeypatch.setattr(cache_report, "file_lock", _oserror)
    result = ingest_ledger(ledger, state, now=_NOW)
    assert result.reason == "state_unwritable"

    def _internal(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(cache_report, "file_lock", _internal)
    result = ingest_ledger(ledger, state, now=_NOW)
    assert result.reason == "internal_error"


def test_load_measured_store_covers_invalid_shapes(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    assert cache_report.load_measured_store(missing) == {
        "schema_version": cache_report._MEASURED_STORE_SCHEMA_VERSION,
        "entries": [],
    }

    wrong_schema = tmp_path / "wrong-schema.json"
    wrong_schema.write_text(json.dumps({"schema_version": 999, "entries": []}), encoding="utf-8")
    assert cache_report.load_measured_store(wrong_schema)["entries"] == []

    wrong_entries = tmp_path / "wrong-entries.json"
    wrong_entries.write_text(
        json.dumps({"schema_version": cache_report._MEASURED_STORE_SCHEMA_VERSION, "entries": {}}),
        encoding="utf-8",
    )
    assert cache_report.load_measured_store(wrong_entries)["entries"] == []


def test_update_measured_store_rejects_unknown_identity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="known provider and model_id"):
        update_measured_store(
            tmp_path / "measured.json",
            [{"provider": None, "model_id": None}],
            now=_NOW,
        )


def test_qualified_entries_skips_missing_window_or_hit_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = cache_report.PartitionKey(
        provider="openai",
        model_id="m1",
        actual_route=None,
        cache_policy="unknown",
        prefix_id=None,
    )
    tuple_key = cache_report._key_tuple(key)
    metrics = cache_report.PartitionMetrics()
    ingestion = _Ingestion(
        status="ok",
        source_complete=True,
        partitions={tuple_key: metrics},
        keys={tuple_key: key},
        window_end=None,
    )

    monkeypatch.setattr(
        cache_report,
        "build_partition_report",
        lambda *_args, **_kwargs: SimpleNamespace(status="qualified", method_version="test"),
    )
    assert cache_report._qualified_entries(ingestion, _NOW) == []


def test_renderer_includes_diagnostics_and_excluded_buckets() -> None:
    ingestion = _Ingestion(
        status="ok",
        source_complete=True,
        diagnostics={"malformed_lines": 2},
        excluded={"openai/m1/policy=unknown": {"cache_unknown": 3}},
    )

    text = "\n".join(render_cache_section(ingestion, _NOW))

    assert "Diagnostics: malformed_lines=2" in text
    assert "Excluded for openai/m1/policy=unknown" in text
    assert "cache_unknown:3" in text


def test_report_cache_returns_unavailable_on_render_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    state = tmp_path / "state.json"
    measured = tmp_path / "measured.json"
    _write(ledger, [_record()])

    monkeypatch.setattr(
        cache_report,
        "_qualified_entries",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    lines = report_cache(ledger, state, measured)

    assert "- Status: unavailable (internal error)" in "\n".join(lines)


def test_checkpoint_validation_covers_all_structural_branches() -> None:
    valid = {
        "schema_version": cache_report._STATE_SCHEMA_VERSION,
        "offset": 0,
        "source_size": 0,
        "prefix_sha256": "",
        "contributions": [],
        "excluded": {},
        "diagnostics": {},
        "continuity_lost_at": None,
        "last_read_utc": "",
    }
    assert cache_report._checkpoint_is_valid(valid) is True
    assert cache_report._checkpoint_is_valid({**valid, "offset": -1}) is False
    assert cache_report._checkpoint_is_valid({**valid, "contributions": {}}) is False
    assert cache_report._checkpoint_is_valid({**valid, "diagnostics": []}) is False
    assert cache_report._checkpoint_is_valid({**valid, "prefix_sha256": 123}) is False


def test_classify_line_records_cache_metrics_input_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(_record: object) -> object:
        raise cache_report.CacheMetricsInputError("bad row")

    monkeypatch.setattr(cache_report, "classify_usage_record", _raise)
    record, diagnostic = cache_report._classify_line(b'{"kind":"chat_completion"}')
    assert record is None
    assert diagnostic == "malformed_lines"


def test_fold_contributions_covers_optional_cost_and_ambiguity() -> None:
    state = cache_report._Checkpoint(
        contributions=[
            {
                "pr": "openai",
                "mo": "m1",
                "pol": "unknown",
                "ct": 4,
                "pt": 10,
                "a": True,
                "c": 0.25,
                "pc": 0.30,
                "t": _ts(),
            }
        ]
    )

    partitions, _keys, earliest, latest = cache_report._fold_contributions(state)

    (metrics,) = partitions.values()
    assert metrics.ambiguous_rows == 1
    assert metrics.cost_known_count == 1
    assert metrics.cost_sum == 0.25
    assert metrics.provider_cost_known_count == 1
    assert metrics.provider_cost_sum == 0.30
    assert earliest == _NOW
    assert latest == _NOW


def test_locked_read_covers_identity_size_and_lock_fallbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("x\n", encoding="utf-8")

    class _Lock:
        def __enter__(self) -> _Lock:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(cache_report, "file_lock", lambda *_args, **_kwargs: _Lock())
    monkeypatch.setattr(cache_report, "_stat_identity", lambda _path: (1, 2, 1))

    changed = cache_report._locked_read(ledger, 9, 9)
    assert isinstance(changed, _Ingestion)
    assert changed.reason == "ledger_changed"

    monkeypatch.setattr(
        cache_report,
        "_stat_identity",
        lambda _path: (1, 2, cache_report._MAX_INCREMENTAL_BYTES + 1),
    )
    too_large = cache_report._locked_read(ledger, 1, 2)
    assert isinstance(too_large, _Ingestion)
    assert too_large.reason == "ledger_too_large"

    monkeypatch.setattr(
        cache_report, "file_lock", lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError())
    )
    lock_timeout = cache_report._locked_read(ledger, 1, 2)
    assert isinstance(lock_timeout, _Ingestion)
    assert lock_timeout.reason == "ledger_lock_timeout"


def test_verify_consumed_prefix_covers_offset_zero_and_partial_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("x", encoding="utf-8")

    state = cache_report._Checkpoint(offset=0)
    assert cache_report._verify_consumed_prefix(ledger, state, 1, {}) is state

    state = cache_report._Checkpoint(offset=1, source_size=1, prefix_sha256="a" * 64)

    class _ShortHandle:
        def __enter__(self) -> _ShortHandle:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _size: int) -> bytes:
            return b""

    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: _ShortHandle())
    diagnostics: dict[str, int] = {}
    assert cache_report._verify_consumed_prefix(ledger, state, 1, diagnostics) is None
    assert diagnostics["prefix_rewrites"] == 1


def test_load_measured_store_covers_valid_document(tmp_path: Path) -> None:
    store = tmp_path / "measured.json"
    store.write_text(
        json.dumps(
            {
                "schema_version": cache_report._MEASURED_STORE_SCHEMA_VERSION,
                "entries": [{"provider": "p", "model_id": "m"}],
            }
        ),
        encoding="utf-8",
    )
    doc = cache_report.load_measured_store(store)
    assert len(doc["entries"]) == 1


def test_qualified_entries_builds_identity_bound_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = cache_report.PartitionKey(
        provider="openai",
        model_id="m1",
        actual_route="direct",
        cache_policy="policy",
        prefix_id="prefix",
    )
    tuple_key = cache_report._key_tuple(key)
    metrics = cache_report.PartitionMetrics(
        row_count=1,
        eligible_count=1,
        hits=1,
        prompt_sum=100,
        cached_sum=40,
    )
    ingestion = _Ingestion(
        status="ok",
        source_complete=True,
        partitions={tuple_key: metrics},
        keys={tuple_key: key},
        window_start=_NOW - timedelta(days=1),
        window_end=_NOW,
    )
    monkeypatch.setattr(
        cache_report,
        "build_partition_report",
        lambda *_args, **_kwargs: SimpleNamespace(
            status="qualified",
            method_version="test",
        ),
    )

    entries = cache_report._qualified_entries(ingestion, _NOW)

    assert len(entries) == 1
    assert entries[0]["provider"] == "openai"
    assert entries[0]["model_id"] == "m1"
    assert entries[0]["estimate"] == 1.0

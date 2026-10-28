"""Tests for the rotation-aware log reader.

The reader is the foundation of every forensic answer, so its failure modes
are tested rather than assumed: rotation, a byte budget that lands mid-record,
malformed lines, and timestamps that do not parse.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sdk.forensics import (
    DEFAULT_MAX_BYTES,
    filter_window,
    in_window,
    parse_ts,
    read_jsonl_tail,
    record_ts,
    rotated_segments,
)


def _write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_rotated_segments_orders_oldest_first(tmp_path: Path) -> None:
    """The rotation set is returned oldest-first, ending with the active file."""
    active = tmp_path / "runtime.jsonl"
    active.write_text("", encoding="utf-8")
    (tmp_path / "runtime.jsonl.1").write_text("", encoding="utf-8")
    (tmp_path / "runtime.jsonl.2").write_text("", encoding="utf-8")

    assert [p.name for p in rotated_segments(active)] == [
        "runtime.jsonl.2",
        "runtime.jsonl.1",
        "runtime.jsonl",
    ]


def test_rotated_segments_skips_absent_backups(tmp_path: Path) -> None:
    """A missing backup is not an error; only existing files are returned."""
    active = tmp_path / "runtime.jsonl"
    active.write_text("", encoding="utf-8")
    assert [p.name for p in rotated_segments(active)] == ["runtime.jsonl"]


def test_rotated_segments_none_discovers_all_segments_oldest_first(tmp_path: Path) -> None:
    """backup_count=None returns append-only sealed segments ascending (.1 oldest)."""
    active = tmp_path / "decisions.jsonl"
    active.write_text("", encoding="utf-8")
    for n in (1, 2, 3, 5, 7):
        (tmp_path / f"decisions.jsonl.{n}").write_text("", encoding="utf-8")

    assert [p.name for p in rotated_segments(active, None)] == [
        "decisions.jsonl.1",
        "decisions.jsonl.2",
        "decisions.jsonl.3",
        "decisions.jsonl.5",
        "decisions.jsonl.7",
        "decisions.jsonl",
    ]


def test_append_only_budget_keeps_the_newest_sealed_segment(tmp_path: Path) -> None:
    """A tight budget over sealed segments keeps the most recent records."""
    active = tmp_path / "decisions.jsonl"
    _write(tmp_path / "decisions.jsonl.1", [{"seq": 1}])
    _write(tmp_path / "decisions.jsonl.2", [{"seq": 2}])
    _write(active, [{"seq": 3}])
    budget = (tmp_path / "decisions.jsonl.2").stat().st_size + active.stat().st_size

    records = read_jsonl_tail(active, max_bytes=budget, backup_count=None).records

    assert [r["seq"] for r in records] == [2, 3]


def test_rotated_segments_int_keeps_newest_three(tmp_path: Path) -> None:
    """backup_count=3 returns only the newest 3 plus the active file."""
    active = tmp_path / "runtime.jsonl"
    active.write_text("", encoding="utf-8")
    for n in (1, 2, 3, 4, 5):
        (tmp_path / f"runtime.jsonl.{n}").write_text("", encoding="utf-8")

    assert [p.name for p in rotated_segments(active, 3)] == [
        "runtime.jsonl.3",
        "runtime.jsonl.2",
        "runtime.jsonl.1",
        "runtime.jsonl",
    ]


def test_read_jsonl_tail_spans_rotation_in_order(tmp_path: Path) -> None:
    """Records from a rotated segment precede the active file's records."""
    active = tmp_path / "runtime.jsonl"
    _write(tmp_path / "runtime.jsonl.1", [{"seq": 1}, {"seq": 2}])
    _write(active, [{"seq": 3}])

    assert [r["seq"] for r in read_jsonl_tail(active).records] == [1, 2, 3]


def test_read_jsonl_tail_honors_byte_budget_from_the_newest_end(tmp_path: Path) -> None:
    """An oversized history degrades to the most recent window, not to OOM."""
    active = tmp_path / "runtime.jsonl"
    _write(active, [{"seq": n, "pad": "x" * 200} for n in range(200)])

    records = read_jsonl_tail(active, max_bytes=2000).records
    assert records, "the budget must still yield the newest records"
    assert records[-1]["seq"] == 199
    assert len(records) < 200


def test_byte_budget_never_parses_a_partial_line(tmp_path: Path) -> None:
    """A mid-record seek drops the leading fragment instead of parsing it."""
    active = tmp_path / "runtime.jsonl"
    _write(active, [{"seq": n, "pad": "y" * 100} for n in range(50)])

    # A budget that lands mid-line must still produce only valid records.
    records = read_jsonl_tail(active, max_bytes=250).records
    assert all(isinstance(record.get("seq"), int) for record in records)


def test_read_jsonl_tail_skips_malformed_and_non_mapping_lines(tmp_path: Path) -> None:
    """A torn write must not hide the records around it."""
    active = tmp_path / "runtime.jsonl"
    active.write_text(
        '{"seq": 1}\n{"seq": 2, "truncated"\n[1, 2, 3]\n\n{"seq": 3}\n',
        encoding="utf-8",
    )

    assert [r["seq"] for r in read_jsonl_tail(active).records] == [1, 3]


def test_complete_read_is_not_truncated(tmp_path: Path) -> None:
    """Reading the whole rotation set within budget reports full coverage."""
    active = tmp_path / "runtime.jsonl"
    _write(tmp_path / "runtime.jsonl.1", [{"seq": 1}])
    _write(active, [{"seq": 2}])

    assert read_jsonl_tail(active).truncated is False


def test_budget_exhaustion_is_reported_as_truncation(tmp_path: Path) -> None:
    """A read the budget cut short says so instead of passing as complete."""
    active = tmp_path / "runtime.jsonl"
    _write(tmp_path / "runtime.jsonl.1", [{"seq": n, "pad": "x" * 200} for n in range(50)])
    _write(active, [{"seq": 100 + n, "pad": "x" * 200} for n in range(5)])

    within_active = read_jsonl_tail(active, max_bytes=active.stat().st_size)
    mid_record = read_jsonl_tail(active, max_bytes=500)

    assert within_active.truncated is True
    assert [r["seq"] for r in within_active.records] == [100, 101, 102, 103, 104]
    assert mid_record.truncated is True


def _stamped(start: datetime, count: int, *, step: timedelta) -> list[dict]:
    return [
        {"seq": n, "ts": (start + step * n).isoformat(), "pad": "z" * 100} for n in range(count)
    ]


def test_since_stops_at_the_window_without_reading_older_segments(tmp_path: Path) -> None:
    """A windowed read pays for its window, not for the whole history."""
    now = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)
    active = tmp_path / "runtime.jsonl"
    old = tmp_path / "runtime.jsonl.2"
    _write(old, _stamped(now - timedelta(days=30), 10, step=timedelta(minutes=1)))
    _write(
        tmp_path / "runtime.jsonl.1",
        _stamped(now - timedelta(days=2), 10, step=timedelta(minutes=1)),
    )
    _write(active, _stamped(now - timedelta(hours=1), 10, step=timedelta(minutes=1)))
    old.chmod(0)  # Reading it would raise; the window must make it unnecessary.

    try:
        tail = read_jsonl_tail(active, since=now - timedelta(hours=24))
    finally:
        old.chmod(0o644)

    assert tail.truncated is False
    stamps = [parse_ts(r["ts"]) for r in tail.records]
    assert all(ts is not None for ts in stamps)
    assert sum(1 for ts in stamps if ts and ts >= now - timedelta(hours=24)) == 10


def test_since_stops_inside_a_large_segment(tmp_path: Path) -> None:
    """Inside one big file, reading stops near the window start, not at byte 0."""
    now = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)
    active = tmp_path / "token-usage.jsonl"
    records = _stamped(now - timedelta(days=60), 20_000, step=timedelta(minutes=4))
    _write(active, records)
    since = now - timedelta(hours=24)

    tail = read_jsonl_tail(active, since=since)

    in_window = [r for r in tail.records if (parse_ts(r["ts"]) or since) >= since]
    expected = [r for r in records if (parse_ts(r["ts"]) or since) >= since]
    assert [r["seq"] for r in in_window] == [r["seq"] for r in expected]
    assert tail.truncated is False
    assert len(tail.records) < len(records) // 10


def test_budget_inside_the_window_is_truncation(tmp_path: Path) -> None:
    """Running out of budget before the window start is a partial window."""
    now = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)
    active = tmp_path / "runtime.jsonl"
    _write(active, _stamped(now - timedelta(hours=2), 500, step=timedelta(seconds=10)))

    tail = read_jsonl_tail(active, since=now - timedelta(hours=24), max_bytes=4096)

    assert tail.truncated is True


def test_default_budget_covers_the_full_runtime_rotation_set() -> None:
    """The default read spans every runtime log segment the handler keeps."""
    from sdk import logging_setup

    rotation_set = logging_setup._ROTATE_MAX_BYTES * (logging_setup._ROTATE_BACKUP_COUNT + 1)
    assert rotation_set <= DEFAULT_MAX_BYTES


def test_read_jsonl_tail_yields_nothing_when_absent(tmp_path: Path) -> None:
    """A missing file is empty evidence, not an exception."""
    assert read_jsonl_tail(tmp_path / "nope.jsonl").records == []
    assert read_jsonl_tail(None).records == []


def test_parse_ts_accepts_iso_z_suffix() -> None:
    """The writers emit trailing-Z ISO stamps; those must parse."""
    parsed = parse_ts("2026-10-27T12:00:00Z")
    assert parsed == datetime(2026, 10, 27, 12, 0, tzinfo=UTC)


def test_parse_ts_accepts_epoch_numbers() -> None:
    """A numeric epoch is a supported timestamp shape."""
    parsed = parse_ts(1_700_000_000)
    assert parsed is not None
    assert parsed.tzinfo is not None


def test_parse_ts_rejects_garbage_without_defaulting_to_now() -> None:
    """An unparseable stamp must not manufacture recency."""
    assert parse_ts("not a timestamp") is None
    assert parse_ts("") is None
    assert parse_ts(None) is None
    assert parse_ts(True) is None
    assert parse_ts({}) is None


def test_record_ts_reads_each_writer_field() -> None:
    """Every writer's timestamp field resolves through one function."""
    stamp = "2026-10-27T12:00:00+00:00"
    assert record_ts({"ts": stamp}) is not None
    assert record_ts({"emitted_at": stamp}) is not None
    assert record_ts({"timestamp_utc": stamp}) is not None
    assert record_ts({"logged_at": stamp}) is not None
    assert record_ts({}) is None


def test_in_window_admits_unknown_timestamps() -> None:
    """Undated evidence stays: a broken writer must not hide a live failure."""
    cutoff = datetime.now(UTC) - timedelta(hours=1)
    assert in_window(None, cutoff) is True
    assert in_window(datetime.now(UTC), cutoff) is True
    assert in_window(datetime.now(UTC) - timedelta(hours=5), cutoff) is False


def test_filter_window_keeps_only_recent_records() -> None:
    """The window is applied on the record's own timestamp."""
    now = datetime.now(UTC)
    records = [
        {"emitted_at": (now - timedelta(days=2)).isoformat(), "seq": 1},
        {"emitted_at": now.isoformat(), "seq": 2},
        {"seq": 3},
    ]

    windowed = filter_window(records, now=now, window_seconds=86_400)
    assert [r["seq"] for r in windowed] == [2, 3]

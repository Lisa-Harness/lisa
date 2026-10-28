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
    filter_window,
    in_window,
    iter_jsonl_tail,
    parse_ts,
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

    records = iter_jsonl_tail(active, max_bytes=budget, backup_count=None)

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


def test_iter_jsonl_tail_spans_rotation_in_order(tmp_path: Path) -> None:
    """Records from a rotated segment precede the active file's records."""
    active = tmp_path / "runtime.jsonl"
    _write(tmp_path / "runtime.jsonl.1", [{"seq": 1}, {"seq": 2}])
    _write(active, [{"seq": 3}])

    assert [r["seq"] for r in iter_jsonl_tail(active)] == [1, 2, 3]


def test_iter_jsonl_tail_honors_byte_budget_from_the_newest_end(tmp_path: Path) -> None:
    """An oversized history degrades to the most recent window, not to OOM."""
    active = tmp_path / "runtime.jsonl"
    _write(active, [{"seq": n, "pad": "x" * 200} for n in range(200)])

    records = list(iter_jsonl_tail(active, max_bytes=2000))
    assert records, "the budget must still yield the newest records"
    assert records[-1]["seq"] == 199
    assert len(records) < 200


def test_byte_budget_never_parses_a_partial_line(tmp_path: Path) -> None:
    """A mid-record seek drops the leading fragment instead of parsing it."""
    active = tmp_path / "runtime.jsonl"
    _write(active, [{"seq": n, "pad": "y" * 100} for n in range(50)])

    # A budget that lands mid-line must still produce only valid records.
    records = list(iter_jsonl_tail(active, max_bytes=250))
    assert all(isinstance(record.get("seq"), int) for record in records)


def test_iter_jsonl_tail_skips_malformed_and_non_mapping_lines(tmp_path: Path) -> None:
    """A torn write must not hide the records around it."""
    active = tmp_path / "runtime.jsonl"
    active.write_text(
        '{"seq": 1}\n{"seq": 2, "truncated"\n[1, 2, 3]\n\n{"seq": 3}\n',
        encoding="utf-8",
    )

    assert [r["seq"] for r in iter_jsonl_tail(active)] == [1, 3]


def test_iter_jsonl_tail_yields_nothing_when_absent(tmp_path: Path) -> None:
    """A missing file is empty evidence, not an exception."""
    assert list(iter_jsonl_tail(tmp_path / "nope.jsonl")) == []
    assert list(iter_jsonl_tail(None)) == []


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

"""Tests for :class:`sdk.forensics.ForensicReader` against real files.

These exercise the composition end-to-end: real JSONL on disk, a real
segmented thread store, real redaction. The contracts under test are the ones
that would make a forensic read dangerous if broken — path derivation,
degradation reporting, caps, and redaction.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sdk.forensics import ForensicReader, render_bundle
from sdk.thread import ThreadRedactionPolicy
from sdk.thread._backend import _segment_paths


def _write_thread_entry(thread: Path, entry: dict) -> None:
    paths = _segment_paths(thread)
    paths.current.parent.mkdir(parents=True, exist_ok=True)
    with paths.current.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _seed(
    tmp_path: Path,
    *,
    user: str = "u",
    turns: int = 3,
    thread_store_slug: str = "",
) -> tuple[Path, Path]:
    """Create a data root with a thread store and telemetry for ``turns`` turns.

    ``thread_store_slug`` stamps the ledger rows with the thread's store stem
    when given, as the writers do. Left empty, the rows carry no stem.
    """
    data_dir = tmp_path / "data"
    thread_dir = data_dir / "data" / "threads"
    thread = thread_dir / (thread_store_slug or user)
    now = datetime.now(UTC).isoformat()

    for turn in range(turns):
        _write_thread_entry(
            thread,
            {
                "timestamp_utc": now,
                "mode_id": "coding_execute",
                "query": f"query {turn}",
                "response_text": f"response {turn}",
                "turn": turn,
                "provenance": {"source": "observed", "task_id": "t-1"},
            },
        )

    ledger = data_dir / "logs" / "decisions.jsonl"
    _write_jsonl(
        ledger,
        [
            record
            for turn in range(turns)
            for record in (
                {
                    "topic": "turn.decision",
                    "turn_index": turn,
                    "user_slug": user,
                    "thread_store_slug": thread_store_slug,
                    "action_type": "ContinueAction",
                    "iteration": 1,
                },
                {
                    "topic": "tool.telemetry",
                    "turn_index": turn,
                    "user_slug": user,
                    "thread_store_slug": thread_store_slug,
                    "target_id": "read_file",
                    "target_type": "tool",
                    "status": "success",
                },
            )
        ],
    )
    return data_dir, thread_dir


def test_reader_reads_the_latest_turn_by_default(tmp_path: Path) -> None:
    """``turns="last"`` selects the most recent recorded turn."""
    data_dir, thread_dir = _seed(tmp_path, turns=3)
    reader = ForensicReader(data_dir, thread_dir=thread_dir, user_slug="u")

    bundle = reader.bundle()

    assert [turn.turn_index for turn in bundle.turns] == [2]
    assert bundle.turns[0].decisions[0].action_type == "ContinueAction"
    assert bundle.turns[0].dispatches[0].target_id == "read_file"
    assert bundle.sources.thread_store
    assert bundle.sources.decisions_log


def test_reader_parses_an_explicit_range(tmp_path: Path) -> None:
    """``turns="0-1"`` reads an inclusive range."""
    data_dir, thread_dir = _seed(tmp_path, turns=3)
    reader = ForensicReader(data_dir, thread_dir=thread_dir, user_slug="u")

    bundle = reader.bundle(turns="0-1")

    assert [turn.turn_index for turn in bundle.turns] == [0, 1]


def test_reader_reports_missing_logs_rather_than_empty_ones(tmp_path: Path) -> None:
    """An absent ledger yields notes, because 'unknown' is not 'clean'."""
    thread_dir = tmp_path / "threads"
    _write_thread_entry(thread_dir / "u", {"mode_id": "default", "query": "q", "turn": 0})

    reader = ForensicReader(tmp_path, thread_dir=thread_dir, user_slug="u")
    bundle = reader.bundle()

    assert not bundle.sources.decisions_log
    assert any("decisions.jsonl" in note for note in bundle.notes)
    assert "unavailable" in render_bundle(bundle)


def test_reader_scopes_to_the_injected_user_slug(tmp_path: Path) -> None:
    """The store is selected by constructor state, never by a tool argument."""
    data_dir, thread_dir = _seed(tmp_path, user="alice", turns=1)
    _write_thread_entry(
        thread_dir / "bob", {"mode_id": "default", "query": "bob secret", "turn": 0}
    )

    bundle = ForensicReader(data_dir, thread_dir=thread_dir, user_slug="alice").bundle()

    rendered = render_bundle(bundle)
    assert "query 0" in rendered
    assert "bob secret" not in rendered


def test_reader_applies_redaction_to_thread_text(tmp_path: Path) -> None:
    """Threads persist raw text; a forensic read must honor the policy."""
    thread_dir = tmp_path / "threads"
    _write_thread_entry(
        thread_dir / "u",
        {
            "mode_id": "default",
            "query": "why",
            "response_text": "I've hit my iteration budget in explainer (cap 12)",
            "turn": 0,
        },
    )
    policy = ThreadRedactionPolicy.from_patterns((r"I've hit my iteration budget[^\n]*",))
    reader = ForensicReader(tmp_path, thread_dir=thread_dir, user_slug="u", redaction=policy)

    rendered = render_bundle(reader.bundle())

    assert "hit my iteration budget" not in rendered


def test_reader_caps_turns_and_says_so(tmp_path: Path) -> None:
    """A cap that silently truncates would hide the reason for the read."""
    data_dir, thread_dir = _seed(tmp_path, turns=6)
    reader = ForensicReader(data_dir, thread_dir=thread_dir, user_slug="u", max_turns=2)

    bundle = reader.bundle(turns="0-5")

    assert len(bundle.turns) == 2
    assert any("read cap is 2" in note for note in bundle.notes)
    # The most recent turns are kept, not the oldest.
    assert [turn.turn_index for turn in bundle.turns] == [4, 5]


def test_reader_surfaces_log_errors_in_the_window(tmp_path: Path) -> None:
    """Warning/error lines accompany the bundle as evidence."""
    data_dir, thread_dir = _seed(tmp_path, turns=1)
    now = datetime.now(UTC).isoformat()
    _write_jsonl(
        data_dir / "logs" / "runtime.jsonl",
        [
            {"ts": now, "level": "info", "msg": "noise"},
            {"ts": now, "level": "warning", "msg": "pack outdated"},
        ],
    )

    bundle = ForensicReader(data_dir, thread_dir=thread_dir, user_slug="u").bundle()

    assert [record["msg"] for record in bundle.log_errors] == ["pack outdated"]
    assert bundle.sources.runtime_log


def test_reader_handles_an_empty_data_root(tmp_path: Path) -> None:
    """Nothing to read is a stated empty answer, not a crash."""
    bundle = ForensicReader(tmp_path, user_slug="u").bundle()

    assert bundle.turns == []
    assert bundle.notes
    assert "No turn entries matched" in render_bundle(bundle)


@pytest.mark.parametrize("selection", ["nonsense", "5-1", "-2", "x-y"])
def test_reader_rejects_invalid_selections_with_a_note(tmp_path: Path, selection: str) -> None:
    """A bad range is reported; it never silently reads something else."""
    data_dir, thread_dir = _seed(tmp_path, turns=2)
    bundle = ForensicReader(data_dir, thread_dir=thread_dir, user_slug="u").bundle(turns=selection)
    assert bundle.turns == []
    assert any("range" in note or "turn" in note for note in bundle.notes)


def test_reader_render_is_bounded_for_a_real_turn(tmp_path: Path) -> None:
    """A single busy turn stays a summary rather than a thread dump."""
    data_dir, thread_dir = _seed(tmp_path, turns=1)
    _write_jsonl(
        data_dir / "logs" / "decisions.jsonl",
        [
            {
                "topic": "turn.decision",
                "user_slug": "u",
                "turn_index": 0,
                "action_type": "ContinueAction",
                "iteration": n,
            }
            for n in range(1, 41)
        ]
        + [
            {
                "topic": "tool.telemetry",
                "user_slug": "u",
                "turn_index": 0,
                "target_id": "read_file",
                "target_type": "tool",
                "status": "success",
            }
            for _ in range(40)
        ],
    )

    rendered = render_bundle(
        ForensicReader(data_dir, thread_dir=thread_dir, user_slug="u").bundle()
    )

    assert len(rendered) < 6000
    assert "elided" in rendered


def test_reader_reads_decisions_beyond_old_three_segment_cap(tmp_path: Path) -> None:
    """A record in a segment beyond the old 3-segment cap is read and attributed.

    The original bug: ``ForensicReader`` read only 3 segments, so a record
    living in ``decisions.jsonl.5`` was silently dropped. With ``backup_count=None``
    the full segment set is discovered.
    """
    data_dir = tmp_path / "data"
    thread_dir = data_dir / "data" / "threads"
    thread = thread_dir / "u"
    now = datetime.now(UTC).isoformat()

    for turn in range(6):
        _write_thread_entry(
            thread,
            {
                "timestamp_utc": now,
                "mode_id": "coding_execute",
                "query": f"query {turn}",
                "response_text": f"response {turn}",
                "turn": turn,
                "provenance": {"source": "observed", "task_id": "t-1"},
            },
        )

    logs = data_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    # One distinct turn per segment; turn_index=4 lives only in .5 (beyond the
    # old 3-segment cap of .1/.2/.3).
    segment_turns = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    for seg, turn in segment_turns.items():
        _write_jsonl(
            logs / f"decisions.jsonl.{seg}",
            [
                {
                    "topic": "turn.decision",
                    "user_slug": "u",
                    "turn_index": turn,
                    "action_type": "ContinueAction",
                    "iteration": 1,
                }
            ],
        )
    _write_jsonl(
        logs / "decisions.jsonl",
        [
            {
                "topic": "turn.decision",
                "user_slug": "u",
                "turn_index": 5,
                "action_type": "ContinueAction",
                "iteration": 1,
            }
        ],
    )

    reader = ForensicReader(data_dir, thread_dir=thread_dir, user_slug="u", max_bytes=1024 * 1024)
    bundle = reader.bundle(turns="4")

    assert bundle.sources.decisions_log
    assert len(bundle.turns) == 1
    assert bundle.turns[0].turn_index == 4
    assert len(bundle.turns[0].decisions) == 1
    assert bundle.turns[0].decisions[0].turn_index == 4
    # Not reported as unattributed or missing.
    assert all(d.turn_index != 4 for d in bundle.unattributed_decisions)
    assert not any("decisions.jsonl" in note and "no records" in note for note in bundle.notes)


def test_turn_cap_is_applied_before_the_thread_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wide range decodes only the turns it will return."""
    from sdk.thread import ThreadStore

    data_dir, thread_dir = _seed(tmp_path, turns=10)
    calls: list[tuple[int, int]] = []
    original = ThreadStore.iter_range

    def _spy(self: ThreadStore, *, start_turn: int, end_turn: int):  # type: ignore[no-untyped-def]
        calls.append((start_turn, end_turn))
        return original(self, start_turn=start_turn, end_turn=end_turn)

    monkeypatch.setattr(ThreadStore, "iter_range", _spy)
    reader = ForensicReader(data_dir, thread_dir=thread_dir, user_slug="u", max_turns=3)

    bundle = reader.bundle(turns="0-9")

    assert [turn.turn_index for turn in bundle.turns] == [7, 8, 9]
    assert (7, 9) in calls
    assert all(start >= 6 for start, _ in calls)
    assert any("10 turns requested but the read cap is 3" in note for note in bundle.notes)


def test_requested_turn_dispatches_survive_a_long_newer_ledger(tmp_path: Path) -> None:
    """The record cap never drops the evidence for the turns asked about."""
    data_dir, thread_dir = _seed(tmp_path, turns=3)
    ledger = data_dir / "logs" / "decisions.jsonl"
    newer = [
        {
            "user_slug": "u",
            "topic": "tool.telemetry",
            "turn_index": 2,
            "target_id": f"t{n}",
            "status": "success",
        }
        for n in range(30)
    ]
    with ledger.open("a", encoding="utf-8") as handle:
        handle.writelines(json.dumps(record) + "\n" for record in newer)

    bundle = ForensicReader(data_dir, thread_dir=thread_dir, user_slug="u").bundle(
        turns="0", max_records=5
    )

    assert [d.target_id for d in bundle.turns[0].dispatches] == ["read_file"]
    assert len(bundle.unattributed_dispatches) == 5
    assert any("32 dispatch records named no requested turn" in n for n in bundle.notes)


def test_unattributed_records_are_bounded_in_the_rendered_bundle(tmp_path: Path) -> None:
    """A long ledger of other turns cannot swamp the answer."""
    data_dir, thread_dir = _seed(tmp_path, turns=1)
    ledger = data_dir / "logs" / "decisions.jsonl"
    with ledger.open("a", encoding="utf-8") as handle:
        for n in range(1000):
            record = {
                "topic": "turn.decision",
                "user_slug": "u",
                "turn_index": 500 + n,
                "action_type": "ContinueAction",
                "action_reason": "r" * 150,
            }
            handle.write(json.dumps(record) + "\n")

    bundle = ForensicReader(data_dir, thread_dir=thread_dir, user_slug="u").bundle()
    rendered = render_bundle(bundle)

    assert len(bundle.unattributed_decisions) == 50
    assert len(rendered) < 20_000
    assert "1000 decision records named no requested turn" in rendered


def test_budget_truncation_is_reported_as_a_lower_bound(tmp_path: Path) -> None:
    """A read that ran out of budget says its counts are partial."""
    data_dir, thread_dir = _seed(tmp_path, turns=1)
    now = datetime.now(UTC).isoformat()
    _write_jsonl(
        data_dir / "logs" / "runtime.jsonl",
        [{"ts": now, "level": "warning", "msg": "x" * 200} for _ in range(100)],
    )

    bundle = ForensicReader(data_dir, thread_dir=thread_dir, user_slug="u", max_bytes=4096).bundle()

    assert any("runtime.jsonl read stopped" in n and "lower bound" in n for n in bundle.notes)


def test_costs_join_for_turns_older_than_the_log_window(tmp_path: Path) -> None:
    """Per-turn cost is bounded by the turn's own time, not a 24h window."""
    data_dir = tmp_path / "data"
    thread_dir = data_dir / "data" / "threads"
    old = datetime(2026, 1, 1, tzinfo=UTC)
    for turn in range(2):
        _write_thread_entry(
            thread_dir / "u",
            {"timestamp_utc": old.isoformat(), "mode_id": "m", "query": "q", "turn": turn},
        )
    _write_jsonl(
        data_dir / "logs" / "token-usage.jsonl",
        [
            {
                "user_slug": "u",
                "turn_index": 1,
                "total_tokens": 7,
                "emitted_at": "2025-06-01T00:00:00+00:00",
            },
            {"user_slug": "u", "turn_index": 1, "total_tokens": 40, "emitted_at": old.isoformat()},
        ],
    )

    bundle = ForensicReader(data_dir, thread_dir=thread_dir, user_slug="u").bundle(turns="1")

    # The stale record from an earlier thread with a colliding index is excluded.
    assert bundle.turns[0].total_tokens == 40


def test_shared_ledgers_do_not_leak_another_users_turns(tmp_path: Path) -> None:
    """Bob's dispatches and spend on his turn 1 never join onto Alice's turn 1.

    The ledgers live under the shared data root and turn indices are
    per-thread, so a join on ``turn_index`` alone attributes another user's
    activity (and private model names) to the requesting user.
    """
    data_dir, thread_dir = _seed(tmp_path, user="alice", turns=2)
    _write_thread_entry(
        thread_dir / "bob",
        {"timestamp_utc": datetime.now(UTC).isoformat(), "turn": 1, "query": "bob secret"},
    )
    ledger = data_dir / "logs" / "decisions.jsonl"
    with ledger.open("a", encoding="utf-8") as handle:
        for record in (
            {
                "topic": "tool.telemetry",
                "turn_index": 1,
                "user_slug": "bob",
                "target_id": "bob_private_tool",
                "status": "success",
            },
            {
                "topic": "turn.decision",
                "turn_index": 1,
                "user_slug": "bob",
                "action_type": "BobOnlyAction",
            },
        ):
            handle.write(json.dumps(record) + "\n")
    now = datetime.now(UTC).isoformat()
    _write_jsonl(
        data_dir / "logs" / "token-usage.jsonl",
        [
            {"user_slug": "alice", "turn_index": 1, "total_tokens": 10, "emitted_at": now},
            {
                "user_slug": "bob",
                "turn_index": 1,
                "total_tokens": 4321,
                "estimated_cost_usd": 1.23,
                "model_id": "bob-private-model",
                "emitted_at": now,
            },
        ],
    )

    bundle = ForensicReader(data_dir, thread_dir=thread_dir, user_slug="alice").bundle(turns="1")
    rendered = render_bundle(bundle)

    turn = bundle.turns[0]
    assert [d.target_id for d in turn.dispatches] == ["read_file"]
    assert [d.action_type for d in turn.decisions] == ["ContinueAction"]
    for leaked in ("bob_private_tool", "BobOnlyAction", "4321", "1.23", "bob-private-model"):
        assert leaked not in rendered
    assert not any("predate per-user attribution" in note for note in bundle.notes)


def test_unowned_legacy_records_are_excluded_with_a_note(tmp_path: Path) -> None:
    """Records written before attribution cannot be scoped, so they are dropped."""
    data_dir, thread_dir = _seed(tmp_path, turns=1)
    ledger = data_dir / "logs" / "decisions.jsonl"
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"topic": "tool.telemetry", "turn_index": 0, "target_id": "legacy_tool"})
            + "\n"
        )

    bundle = ForensicReader(data_dir, thread_dir=thread_dir, user_slug="u").bundle()

    assert [d.target_id for d in bundle.turns[0].dispatches] == ["read_file"]
    assert any(
        "1 decisions.jsonl records predate per-user attribution" in note for note in bundle.notes
    )


def test_a_stem_scopes_an_unscoped_owner_read(tmp_path: Path) -> None:
    """A stem narrows the read even when the owner is not scoped.

    A stem names one thread, so honouring it is what the caller asked for.
    Ignoring it would join every user's rows onto one thread's turns.
    """
    from sdk.forensics import filter_owned

    records = [
        {"user_slug": "u", "thread_store_slug": "u--a"},
        {"user_slug": "v", "thread_store_slug": "v--b"},
    ]

    owned = filter_owned(records, None, "u--a")

    assert [r["user_slug"] for r in owned.records] == ["u"]


def test_unstemmed_records_are_excluded_with_a_note(tmp_path: Path) -> None:
    """Owned records written before thread attribution are dropped, and said.

    A row can name its owner but carry no thread stem, written before the
    stem existed. Dropping it silently would make the bundle look complete
    while evidence is missing, so the count is reported.
    """
    data_dir, thread_dir = _seed(tmp_path, turns=1, thread_store_slug="u--conv")
    ledger = data_dir / "logs" / "decisions.jsonl"
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "topic": "tool.telemetry",
                    "turn_index": 0,
                    "user_slug": "u",
                    "target_id": "legacy_tool",
                }
            )
            + "\n"
        )

    bundle = ForensicReader(
        data_dir, thread_dir=thread_dir, user_slug="u", thread_store_slug="u--conv"
    ).bundle()

    assert [d.target_id for d in bundle.turns[0].dispatches] == ["read_file"]
    assert any(
        "1 decisions.jsonl records predate per-thread attribution" in note for note in bundle.notes
    )

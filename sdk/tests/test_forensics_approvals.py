"""ForensicReader joins approval-ledger evidence onto requested turns.

Approval transitions are one row per state change (requested / approved /
denied / consumed / expired), stamped with the request's ``turn_index`` at
issue time. The reader routes rows for requested turns onto their turn,
surfaces rows for other turns as unattributed findings, notes an absent
ledger, and counts sequence gaps like the other durable ledgers.
"""

from __future__ import annotations

import json
from pathlib import Path

from sdk.forensics import ForensicReader, render_bundle


def _seed_thread(threads: Path, turns: int, user: str = "adam") -> None:
    threads.mkdir(parents=True, exist_ok=True)
    entries = [
        {
            "timestamp_utc": f"2026-10-30T1{i}:00:00+00:00",
            "turn": i,
            "user_slug": user,
            "mode_id": "coding",
            "query": f"q{i}",
            "response_text": "r",
        }
        for i in range(turns)
    ]
    (threads / f"{user}.current.jsonl").write_text(
        "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8"
    )


def _row(
    state: str,
    *,
    turn_index: int,
    sequence: int,
    request_id: str = "abc1234567890123",
    target: str = "apply_patch",
    by: str = "",
    source: str = "",
) -> dict:
    row = {
        "state": state,
        "request_id": request_id,
        "target_id": target,
        "target_kind": "tool",
        "effect": "write",
        "turn_index": turn_index,
        "user_slug": "adam",
        "thread_store_slug": "adam",
        "sequence": sequence,
        "ledger_instance": "i1",
        "logged_at": 1793376030.0 + sequence * 30,
    }
    if by:
        row["decision_by"] = by
    if source:
        row["decision_source"] = source
    return row


def _write_ledger(logs: Path, rows: list[dict]) -> None:
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "approvals.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


def _bundle(data_dir: Path):
    reader = ForensicReader(
        data_dir,
        thread_dir=data_dir / "data" / "threads",
        user_slug="adam",
        resolve_redaction_from_env=False,
    )
    return reader.bundle(turns="last")


def test_approval_transitions_join_to_their_turn(tmp_path: Path) -> None:
    threads = tmp_path / "data" / "threads"
    _seed_thread(threads, turns=4)
    _write_ledger(
        tmp_path / "logs",
        [
            _row("requested", turn_index=3, sequence=1),
            _row("approved", turn_index=3, sequence=2, by="adam", source="operator"),
            _row("consumed", turn_index=3, sequence=3, by="adam", source="operator"),
        ],
    )
    bundle = _bundle(tmp_path)
    turn = bundle.turns[0]
    assert turn.turn_index == 3
    assert [a.state for a in turn.approvals] == ["requested", "approved", "consumed"]
    assert bundle.sources.approvals_log is True
    assert not bundle.unattributed_approvals


def test_policy_source_renders_distinct_from_operator(tmp_path: Path) -> None:
    threads = tmp_path / "data" / "threads"
    _seed_thread(threads, turns=2)
    _write_ledger(
        tmp_path / "logs",
        [
            _row("requested", turn_index=1, sequence=1),
            _row("approved", turn_index=1, sequence=2, source="policy"),
        ],
    )
    bundle = _bundle(tmp_path)
    out = render_bundle(bundle)
    assert "approvals (1 request(s))" in out
    assert "by policy" in out


def test_other_turns_rows_surface_as_unattributed(tmp_path: Path) -> None:
    threads = tmp_path / "data" / "threads"
    _seed_thread(threads, turns=4)
    _write_ledger(
        tmp_path / "logs",
        [
            _row("requested", turn_index=3, sequence=1),
            _row(
                "requested",
                turn_index=12,
                sequence=2,
                request_id="def9876543210987",
                target="send_email",
            ),
        ],
    )
    bundle = _bundle(tmp_path)
    assert len(bundle.turns[0].approvals) == 1
    assert len(bundle.unattributed_approvals) == 1
    assert bundle.unattributed_approvals[0].turn_index == 12
    out = render_bundle(bundle)
    assert "approval (turn 12): send_email → requested (by unknown)" in out


def test_absent_ledger_is_noted_not_silent(tmp_path: Path) -> None:
    threads = tmp_path / "data" / "threads"
    _seed_thread(threads, turns=2)
    (tmp_path / "logs").mkdir(parents=True)
    bundle = _bundle(tmp_path)
    assert not bundle.turns[0].approvals
    assert bundle.sources.approvals_log is False
    assert any("approvals.jsonl has no approval transitions" in n for n in bundle.notes)


def test_denial_reason_is_rendered(tmp_path: Path) -> None:
    threads = tmp_path / "data" / "threads"
    _seed_thread(threads, turns=2)
    _write_ledger(
        tmp_path / "logs",
        [
            _row("denied", turn_index=1, sequence=1, by="adam", source="operator"),
        ],
    )
    bundle = _bundle(tmp_path)
    assert bundle.turns[0].approvals[0].state == "denied"
    assert bundle.turns[0].approvals[0].decision_by == "adam"


def test_another_users_rows_do_not_join(tmp_path: Path) -> None:
    threads = tmp_path / "data" / "threads"
    _seed_thread(threads, turns=2)
    row = _row("approved", turn_index=1, sequence=1, by="mallory", source="operator")
    row["user_slug"] = "mallory"
    _write_ledger(tmp_path / "logs", [row])
    bundle = _bundle(tmp_path)
    # Another user's row is excluded by the scoped read — silently, because
    # rows outside the requested window are not findings (only rows that
    # *cannot* be attributed at all are noted).
    assert not bundle.turns[0].approvals
    assert not bundle.unattributed_approvals


def test_ledger_sequence_gaps_are_counted(tmp_path: Path) -> None:
    threads = tmp_path / "data" / "threads"
    _seed_thread(threads, turns=2)
    _write_ledger(
        tmp_path / "logs",
        [
            _row("requested", turn_index=1, sequence=1),
            # 2 dropped
            _row("consumed", turn_index=1, sequence=3),
        ],
    )
    bundle = _bundle(tmp_path)
    assert any(
        "accounts for 1 dropped row(s)" in n and "approvals.jsonl" in n for n in bundle.notes
    ), bundle.notes
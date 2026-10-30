"""ForensicReader detects dropped ledger rows from sequence numbering.

Both durable ledger handlers stamp each row with a per-instance
``sequence`` and the writer's ``ledger_instance``; the reader partitions on
the instance and reports gaps within one writer's run as an honest read
note (a lower bound — proof rows went missing, not which rows). Companion
to ``kernel/tests/test_telemetry_loss_observability.py`` (the kernel-side
stamps and publish warnings); lives here because the reader is SDK-owned
and kernel tests must not import the SDK.
"""

from __future__ import annotations

import json
from pathlib import Path

from sdk.forensics import ForensicReader


def _read_notes(data_dir: Path) -> list[str]:
    reader = ForensicReader(
        data_dir,
        thread_dir=data_dir / "data" / "threads",
        user_slug="",
        resolve_redaction_from_env=False,
    )
    bundle = reader.bundle(turns="last")
    return list(bundle.notes)


def _write_ledgers(logs: Path, *, decisions: str, token_usage: str) -> None:
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "decisions.jsonl").write_text(decisions, encoding="utf-8")
    (logs / "token-usage.jsonl").write_text(token_usage, encoding="utf-8")


def _rows(records: list[dict]) -> str:
    return "\n".join(json.dumps(r) for r in records) + "\n"


_SHORT = {"turn_index": 0, "ledger_instance": "a1"}


def _row(topic: str, sequence: int) -> dict:
    return {"topic": topic, "sequence": sequence, **_SHORT}


def test_reader_reports_sequence_gaps(tmp_path: Path) -> None:
    """A gap within one writer's numbering becomes an honest read note."""
    _write_ledgers(
        tmp_path / "logs",
        decisions=_rows(
            [
                _row("turn.decision", 1),
                # 2 and 3 dropped (lock timeout / failed publish)
                _row("turn.decision", 4),
                _row("turn.decision", 5),
                # A second writer's restart-at-1 is NOT a gap.
                {"topic": "turn.decision", "sequence": 1, **_SHORT, "ledger_instance": "b2"},
            ]
        ),
        token_usage="",
    )
    notes = _read_notes(tmp_path)
    assert any(
        "accounts for 2 dropped row(s)" in note and "decisions.jsonl" in note
        for note in notes
    ), notes


def test_reader_silent_when_contiguous(tmp_path: Path) -> None:
    _write_ledgers(
        tmp_path / "logs",
        decisions=_rows(
            [
                _row("turn.decision", 1),
                _row("turn.decision", 2),
                _row("tool.telemetry", 3),
            ]
        ),
        token_usage="",
    )
    notes = _read_notes(tmp_path)
    assert not any("dropped row" in note for note in notes), notes


def test_reader_skips_unstamped_rows(tmp_path: Path) -> None:
    """Rows predating the instance stamps are skipped, not reported."""
    _write_ledgers(
        tmp_path / "logs",
        decisions=_rows(
            [
                {"topic": "turn.decision", "turn_index": 0},  # legacy: no stamps
                {"topic": "turn.decision", "turn_index": 1},  # legacy
            ]
        ),
        token_usage="",
    )
    notes = _read_notes(tmp_path)
    assert not any("dropped row" in note for note in notes), notes


def test_reader_reports_token_ledger_gaps(tmp_path: Path) -> None:
    _write_ledgers(
        tmp_path / "logs",
        decisions="",
        token_usage=_rows(
            [
                {
                    "turn_index": 0,
                    "sequence": 1,
                    "ledger_instance": "a1",
                    "prompt_tokens": 10,
                    "emitted_at": "2026-10-30T12:00:00+00:00",
                },
                {
                    "turn_index": 0,
                    "sequence": 3,
                    "ledger_instance": "a1",
                    "prompt_tokens": 5,
                    "emitted_at": "2026-10-30T12:01:00+00:00",
                },
            ]
        ),
    )
    notes = _read_notes(tmp_path)
    assert any(
        "accounts for 1 dropped row(s)" in note and "token-usage.jsonl" in note
        for note in notes
    ), notes

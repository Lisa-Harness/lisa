"""The forensic read's optional chain-verification note.

``bundle(verify_chain=True)`` appends exactly one integrity status note —
verified, broken, or cannot-run — and the default read carries no chain
note at all: the verification decompresses every sealed segment, so it is
an explicit audit, never part of the ordinary read path. The operator
``/forensics --verify-chain`` flag and the model-facing
``turn_forensics(verify_chain=true)`` parameter route through the same
reader.
"""

from __future__ import annotations

import json
from pathlib import Path

from sdk.forensics import ForensicReader


def _seed(base_dir: Path, turns: int = 4) -> Path:
    threads = base_dir / "data" / "threads"
    threads.mkdir(parents=True, exist_ok=True)
    entries = [
        {
            "timestamp_utc": f"2026-10-30T1{i}:00:00+00:00",
            "turn": i,
            "user_slug": "u",
            "mode_id": "default",
            "query": f"q{i}",
            "response_text": "r",
        }
        for i in range(turns)
    ]
    (threads / "u.current.jsonl").write_text(
        "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8"
    )
    return threads


def _reader(base_dir: Path, threads: Path) -> ForensicReader:
    return ForensicReader(
        base_dir,
        thread_dir=threads,
        user_slug="u",
        resolve_redaction_from_env=False,
    )


def test_default_read_carries_no_chain_note(tmp_path: Path) -> None:
    threads = _seed(tmp_path)
    bundle = _reader(tmp_path, threads).bundle(turns="last")
    assert not any("chain" in note.lower() for note in bundle.notes)


def test_verified_note_names_the_chained_counts(tmp_path: Path) -> None:
    threads = _seed(tmp_path)
    bundle = _reader(tmp_path, threads).bundle(turns="last", verify_chain=True)
    chain_notes = [n for n in bundle.notes if "chain" in n.lower()]
    assert len(chain_notes) == 1
    # Legacy-shaped store (no links): reported as honestly unchained.
    assert "thread chain verified" in chain_notes[0]
    assert "pre-chain" in chain_notes[0]


def _seed_chained(base_dir: Path, turns: int = 4) -> Path:
    """Seed a store whose entries carry real chain links."""
    import hashlib

    threads = base_dir / "data" / "threads"
    threads.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    prev = ""
    for i in range(turns):
        entry = {
            "timestamp_utc": f"2026-10-30T1{i}:00:00+00:00",
            "turn": i,
            "user_slug": "u",
            "mode_id": "default",
            "query": f"q{i}",
            "response_text": "r",
        }
        if prev:
            entry["prev_entry_sha256"] = prev
        line = json.dumps(entry, ensure_ascii=False)
        lines.append(line)
        prev = hashlib.sha256(line.encode("utf-8")).hexdigest()
    (threads / "u.current.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return threads


def test_broken_note_names_the_first_divergence(tmp_path: Path) -> None:
    threads = _seed_chained(tmp_path)
    current = threads / "u.current.jsonl"
    lines = current.read_text().splitlines()
    entry = json.loads(lines[0])
    entry["response_text"] = "rewritten after the fact"
    lines[0] = json.dumps(entry, ensure_ascii=False)
    current.write_text("\n".join(lines) + "\n")

    bundle = _reader(tmp_path, threads).bundle(turns="last", verify_chain=True)
    broken = [n for n in bundle.notes if "THREAD CHAIN BROKEN" in n]
    assert len(broken) == 1
    assert "chain link does not match" in broken[0]


def test_unreadable_store_reports_cannot_run(tmp_path: Path) -> None:
    base_dir = tmp_path
    threads = base_dir / "data" / "threads"
    threads.mkdir(parents=True, exist_ok=True)
    # A directory where the store file should be makes the read fail loudly
    # rather than fabricating a verification result.
    (threads / "u.current.jsonl").mkdir()
    bundle = _reader(base_dir, threads).bundle(turns="last", verify_chain=True)
    assert any(
        "chain verification could not run" in note for note in bundle.notes
    ), bundle.notes
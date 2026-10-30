"""Thread-store hash-chain verification (tamper evidence).

Tests the reader side against stores built to the on-disk spec: entry
``prev_entry_sha256`` links, sealed-segment ``tail_sha256`` index fields,
pre-chain legacy history, and the divergence reports for each tamper
shape. The writer side (runtime appends/stamps) is covered by
``runtime/tests/test_thread_chain_writes.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sdk.thread import ChainVerification, verify_chain


def _entry_line(turn: int, prev: str = "", **overrides: object) -> str:
    entry: dict = {
        "timestamp_utc": f"2026-10-30T1{turn}:00:00+00:00",
        "turn": turn,
        "user_slug": "u",
        "mode_id": "default",
        "query": f"q{turn}",
        "response_text": f"r{turn}",
    }
    if prev:
        entry["prev_entry_sha256"] = prev
    entry.update(overrides)
    return json.dumps(entry, ensure_ascii=False)


def _active_sha(line: str) -> str:
    import hashlib

    return hashlib.sha256(line.encode("utf-8")).hexdigest()


def _write_active(base: Path, lines: list[str]) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    (Path(f"{base}.current.jsonl")).write_text(
        "\n".join(lines) + "\n" if lines else "", encoding="utf-8"
    )


def _seal(base: Path, seg_id: str, lines: list[str]) -> str:
    """Write a fake sealed segment + index line; return the tail hash."""
    import hashlib

    import zstandard

    raw = ("\n".join(lines) + "\n").encode("utf-8")
    Path(f"{base}.{seg_id}.jsonl.zst").write_bytes(
        zstandard.ZstdCompressor().compress(raw)
    )
    tail = hashlib.sha256(lines[-1].encode("utf-8")).hexdigest()
    record = {
        "seg_id": seg_id,
        "turn_lo": 0,
        "turn_hi": len(lines) - 1,
        "ts_lo": "",
        "ts_hi": "",
        "bytes_raw": len(raw),
        "bytes_zst": 0,
        "tail_sha256": tail,
    }
    index_path = Path(f"{base}.index.jsonl")
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return tail


# ── Clean stores ──────────────────────────────────────────────────────


def test_clean_active_chain(tmp_path: Path) -> None:
    base = tmp_path / "u"
    line0 = _entry_line(0)  # chain head: no link
    line1 = _entry_line(1, prev=_active_sha(line0))
    line2 = _entry_line(2, prev=_active_sha(line1))
    _write_active(base, [line0, line1, line2])

    result = verify_chain(base)
    assert result.ok
    assert result.chained_entries == 2
    assert result.unchained_entries == 1  # the head is honestly unchained
    assert result.head_sha256 == _active_sha(line2)
    assert result.first_divergence == ""


def test_clean_chain_across_seal_boundary(tmp_path: Path) -> None:
    base = tmp_path / "u"
    line0 = _entry_line(0)
    line1 = _entry_line(1, prev=_active_sha(line0))
    tail = _seal(base, "0001", [line0, line1])
    line2 = _entry_line(2, prev=tail)  # post-seal link comes from the index
    _write_active(base, [line2])

    result = verify_chain(base)
    assert result.ok
    assert result.chained_entries == 2
    assert result.verified_seals == 1
    assert result.unchained_seals == 0


def test_legacy_pre_chain_store_is_honestly_unchained(tmp_path: Path) -> None:
    base = tmp_path / "u"
    _write_active(base, [_entry_line(0), _entry_line(1), _entry_line(2)])

    result = verify_chain(base)
    assert result.ok  # predating the chain is not tampering
    assert result.chained_entries == 0
    assert result.unchained_entries == 3


def test_empty_store(tmp_path: Path) -> None:
    result = verify_chain(tmp_path / "u")
    assert result.ok
    assert result.head_sha256 == ""


# ── Tamper shapes ─────────────────────────────────────────────────────


def test_edited_active_entry_breaks_the_next_link(tmp_path: Path) -> None:
    base = tmp_path / "u"
    line0 = _entry_line(0)
    line1 = _entry_line(1, prev=_active_sha(line0))
    line2 = _entry_line(2, prev=_active_sha(line1))
    _write_active(base, [line0, line1, line2])

    tampered = _entry_line(1, prev=_active_sha(line0), response_text="rewritten")
    assert tampered != line1
    _write_active(base, [line0, tampered, line2])

    result = verify_chain(base)
    assert not result.ok
    assert "chain link does not match" in result.first_divergence


def test_deleted_active_entry_is_detected(tmp_path: Path) -> None:
    base = tmp_path / "u"
    line0 = _entry_line(0)
    line1 = _entry_line(1, prev=_active_sha(line0))
    line2 = _entry_line(2, prev=_active_sha(line1))
    _write_active(base, [line0, line1, line2])

    _write_active(base, [line0, line2])  # entry 1 removed

    result = verify_chain(base)
    assert not result.ok
    assert result.first_divergence


def test_edited_sealed_entry_is_detected(tmp_path: Path) -> None:
    base = tmp_path / "u"
    line0 = _entry_line(0)
    line1 = _entry_line(1, prev=_active_sha(line0))
    tail = _seal(base, "0001", [line0, line1])
    line2 = _entry_line(2, prev=tail)
    _write_active(base, [line2])

    # Rewrite entry 0 inside the sealed segment (recompress coherently).
    import zstandard

    tampered = _entry_line(0, query="falsified history")
    raw = ("\n".join([tampered, line1]) + "\n").encode("utf-8")
    Path(f"{base}.0001.jsonl.zst").write_bytes(
        zstandard.ZstdCompressor().compress(raw)
    )

    result = verify_chain(base)
    assert not result.ok
    assert "chain link does not match" in result.first_divergence


def test_edited_index_terminal_hash_is_detected(tmp_path: Path) -> None:
    base = tmp_path / "u"
    line0 = _entry_line(0)
    line1 = _entry_line(1, prev=_active_sha(line0))
    _seal(base, "0001", [line0, line1])
    _write_active(base, [])

    index_path = Path(f"{base}.index.jsonl")
    record = json.loads(index_path.read_text().splitlines()[0])
    record["tail_sha256"] = "f" * 64
    index_path.write_text(json.dumps(record) + "\n")

    result = verify_chain(base)
    assert not result.ok
    assert "index terminal hash" in result.first_divergence


def test_stripped_link_mid_chain_is_detected(tmp_path: Path) -> None:
    base = tmp_path / "u"
    line0 = _entry_line(0)
    line1 = _entry_line(1, prev=_active_sha(line0))
    line2 = _entry_line(2, prev=_active_sha(line1))
    _write_active(base, [line0, line1, line2])

    stripped = _entry_line(1)  # attacker removes the link field
    _write_active(base, [line0, stripped, line2])

    result = verify_chain(base)
    assert not result.ok
    # Detection lands on the successor's now-dangling link.
    assert "chain link does not match" in result.first_divergence


def test_verification_continues_after_first_divergence(tmp_path: Path) -> None:
    base = tmp_path / "u"
    line0 = _entry_line(0)
    line1 = _entry_line(1, prev=_active_sha(line0))
    line2 = _entry_line(2, prev=_active_sha(line1))
    # Two independent rewrites: entry 1's content changed (breaking the
    # link its successor recorded) and entry 0's content changed (breaking
    # the link entry 1 recorded) — one store, two divergences to report.
    tampered1 = _entry_line(1, prev=_active_sha(line0), response_text="rewritten")
    tampered0 = _entry_line(0, query="falsified")
    _write_active(base, [tampered0, tampered1, line2])

    result = verify_chain(base)
    assert not result.ok
    assert len(result.divergences) == 2
    assert result.first_divergence == result.divergences[0]


# ── Model shape ────────────────────────────────────────────────────────


def test_verification_result_is_frozen_dataclass(tmp_path: Path) -> None:
    from dataclasses import FrozenInstanceError

    result = verify_chain(tmp_path / "u")
    assert isinstance(result, ChainVerification)
    with pytest.raises(FrozenInstanceError):  # frozen: audit results are immutable
        result.ok = False  # type: ignore[misc]


def test_clean_store_notarizable_head(tmp_path: Path) -> None:
    """The head value an external notary would capture."""
    base = tmp_path / "u"
    line0 = _entry_line(0)
    line1 = _entry_line(1, prev=_active_sha(line0))
    _write_active(base, [line0, line1])
    result = verify_chain(base)
    assert result.head_sha256 == _active_sha(line1)
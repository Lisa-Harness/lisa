"""Hash-chain stamping on the thread write path.

The writer side of tamper evidence: every appended entry carries
``prev_entry_sha256`` (SHA-256 of the previous stored raw line), seals
record their terminal hash in the index, and the chain continues across
seal boundaries without decompressing anything at append time. The reader
side (divergence reports, legacy honesty) is covered by
``sdk/tests/test_thread_chain_verify.py``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from runtime.thread import append_thread_entry
from runtime.thread._store import _read_index, _segment_paths


def _append(base: Path, n: int, segment_turns: int = 256) -> int | None:
    return append_thread_entry(
        thread_path=base,
        user_slug="u",
        mode_id="default",
        query=f"q{n}",
        response_text=f"r{n}",
        turn=n,
        segment_turns=segment_turns,
    )


def _sha(line: str) -> str:
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


def _active_lines(base: Path) -> list[str]:
    return [
        line
        for line in Path(f"{base}.current.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_first_entry_is_the_chain_head(tmp_path: Path) -> None:
    base = tmp_path / "u"
    assert _append(base, 0) == 0
    entry = json.loads(_active_lines(base)[0])
    assert "prev_entry_sha256" not in entry  # the head fakes no link


def test_each_entry_links_to_the_previous_raw_line(tmp_path: Path) -> None:
    base = tmp_path / "u"
    _append(base, 0)
    _append(base, 1)
    _append(base, 2)
    lines = _active_lines(base)
    entries = [json.loads(line) for line in lines]
    assert entries[1]["prev_entry_sha256"] == _sha(lines[0])
    assert entries[2]["prev_entry_sha256"] == _sha(lines[1])


def test_seal_records_terminal_hash_in_the_index(tmp_path: Path) -> None:
    base = tmp_path / "u"
    _append(base, 0, segment_turns=2)
    _append(base, 1, segment_turns=2)
    _append(base, 2, segment_turns=2)  # triggers the seal of entries 0-1

    paths = _segment_paths(base)
    records = _read_index(paths.index_path)
    assert len(records) == 1
    # The recorded terminal hash is the sealed content's true last line…
    import zstandard

    sealed_lines = zstandard.ZstdDecompressor().decompress(
        Path(f"{base}.0001.jsonl.zst").read_bytes()
    ).decode().splitlines()
    assert records[0].tail_sha256 == _sha(sealed_lines[-1])
    # …and the first post-seal entry links through the index value.
    post = json.loads(_active_lines(base)[0])
    assert post["prev_entry_sha256"] == records[0].tail_sha256


def test_chain_continues_across_multiple_seals(tmp_path: Path) -> None:
    base = tmp_path / "u"
    for n in range(5):
        _append(base, n, segment_turns=2)
    paths = _segment_paths(base)
    records = _read_index(paths.index_path)
    assert len(records) == 2
    # Entry 4 (first of the second post-seal active span) links to seal 2.
    lines = _active_lines(base)
    entry4 = json.loads(lines[0])
    assert entry4["prev_entry_sha256"] == records[1].tail_sha256


def test_prev_hash_resolution_never_decompresses(tmp_path: Path) -> None:
    """After a seal, the link comes from the index, not the sealed bytes.

    Guards the cost contract: append latency does not grow with sealed
    history. If the writer ever resolved the link by decompressing the
    newest sealed segment, this would still pass functionally but the
    ``tail_sha256`` path would be dead — the index field is asserted
    directly instead.
    """
    base = tmp_path / "u"
    _append(base, 0, segment_turns=1)
    _append(base, 1, segment_turns=1)
    records = _read_index(_segment_paths(base).index_path)
    assert records[0].tail_sha256 != ""
    entry1 = json.loads(_active_lines(base)[0])
    assert entry1["prev_entry_sha256"] == records[0].tail_sha256


def test_pre_chain_seal_then_chain_restarts(tmp_path: Path) -> None:
    """A legacy sealed segment (no tail hash) restarts chaining honestly.

    Simulates upgrading a store that predates the chain: the old seal's
    index line carries no ``tail_sha256``, so the next append becomes a
    new chain head rather than fabricating a link it cannot verify.
    """
    base = tmp_path / "u"
    base.parent.mkdir(parents=True, exist_ok=True)

    import zstandard

    legacy_entry = json.dumps(
        {
            "timestamp_utc": "2026-10-01T00:00:00+00:00",
            "turn": 0,
            "user_slug": "u",
            "mode_id": "default",
            "query": "old",
            "response_text": "old",
        }
    )
    raw = (legacy_entry + "\n").encode("utf-8")
    Path(f"{base}.0001.jsonl.zst").write_bytes(zstandard.ZstdCompressor().compress(raw))
    record = {
        "seg_id": "0001",
        "turn_lo": 0,
        "turn_hi": 0,
        "ts_lo": "",
        "ts_hi": "",
        "bytes_raw": len(raw),
        "bytes_zst": 0,
        # no tail_sha256: the pre-chain shape
    }
    with Path(f"{base}.index.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")

    _append(base, 1)
    entry = json.loads(_active_lines(base)[0])
    assert entry["turn"] == 1
    assert "prev_entry_sha256" not in entry  # new head, honestly unlinked
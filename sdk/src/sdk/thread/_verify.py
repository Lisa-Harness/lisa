"""Thread-store hash-chain verification (tamper evidence).

Every appended thread entry (after the chain feature) carries
``prev_entry_sha256`` — the SHA-256 of the previous stored entry's raw
line — and every sealed segment's index line carries the SHA-256 of its
terminal raw line, so the chain continues across seal boundaries. An
attacker who edits or removes any stored entry breaks the link in the
next entry; an attacker who rewrites the next entry too must also rewrite
every subsequent entry and every later seal's index line.

Honest threat model: this detects *tampering after the fact*, not a
full adversary who rewrites the entire store and the index coherently,
and not a writer who strips the fields before first write. Sealed
segments written before the chain existed carry no ``tail_sha256``;
entries written before it carry no ``prev_entry_sha256`` — the verifier
reports where chaining begins rather than pretending to check history it
cannot. For stronger guarantees (repudiation, deletion), export the
chain head periodically to an external notary or WORM store; the
``head_sha256`` this check returns is the value to notarize.

The verification walks the whole store (every sealed segment is
decompressed), so it is an on-demand integrity audit — never part of a
per-read path.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sdk.thread._backend import _read_index, _segment_paths

__all__ = ["ChainVerification", "verify_chain"]


@dataclass(frozen=True)
class ChainVerification:
    """The result of one hash-chain verification over a thread store.

    ``ok`` is ``True`` only when every verifiable link checked out.
    A store that predates the chain feature is *not* an error: its
    pre-chain entries simply do not participate, and ``unchained_entries``
    names how many were skipped.
    """

    ok: bool = True
    #: Chain head: SHA-256 of the newest stored entry's raw line, or ""
    #: when the store is empty. The value to notarize externally.
    head_sha256: str = ""
    #: Entries checked against a predecessor link (or recognized as head).
    chained_entries: int = 0
    #: Entries predating the chain feature (no ``prev_entry_sha256``).
    unchained_entries: int = 0
    #: Sealed segments whose ``tail_sha256`` matched their raw terminal line.
    verified_seals: int = 0
    #: Sealed index lines predating the chain feature.
    unchained_seals: int = 0
    #: First divergence found, if any: human-readable, names the turn.
    first_divergence: str = ""
    #: Every divergence found (verification continues after the first so a
    #: single audit reports the full damage, bounded by the store size).
    divergences: list[str] = field(default_factory=list)


def _entry_sha256(raw_line: str) -> str:
    return hashlib.sha256(raw_line.encode("utf-8")).hexdigest()


def _iter_sealed_lines(base: Path, seg_id: str) -> list[str]:
    """Decompress one sealed segment and return its raw non-empty lines."""
    sealed_path = Path(f"{base}.{seg_id}.jsonl.zst")
    try:
        compressed = sealed_path.read_bytes()
    except OSError:
        return []
    try:
        import zstandard  # lazy: defers optional dep startup cost

        raw = zstandard.ZstdDecompressor().decompress(compressed)
    except Exception:  # pylint: disable=broad-except
        return []
    return [line for line in raw.decode("utf-8", errors="replace").splitlines() if line.strip()]


def _active_lines(current_path: Path) -> list[str]:
    """Return the active segment's raw non-empty lines."""
    if not current_path.exists():
        return []
    return [
        line
        for line in current_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]


def verify_chain(base_path: Path) -> ChainVerification:
    """Verify the tamper-evidence chain over an entire thread store.

    Walks every sealed segment in index order, then the active segment:
    re-hashes each raw line, checks each entry's ``prev_entry_sha256``
    link (including across seal boundaries, via the index's
    ``tail_sha256``), and confirms each seal's recorded terminal hash
    against its raw content. Malformed lines are a divergence, not a skip:
    the lossless contract says every line is one entry.
    """
    paths = _segment_paths(base_path)
    records = _read_index(paths.index_path)
    walk = _ChainWalk()

    # Sealed segments in index order. The previous line's hash threads
    # across boundaries, so the first entry of each later segment is
    # checked against the previous segment's terminal hash exactly like
    # an in-segment link.
    for record in records:
        lines = _iter_sealed_lines(paths.base, record.seg_id)
        if not lines and (record.bytes_raw or record.bytes_zst):
            walk.diverge(f"sealed segment {record.seg_id}: unreadable or empty")
            continue
        walk.entries(lines, f"sealed segment {record.seg_id}")
        walk.seal(record, lines)

    # Active segment last.
    walk.entries(_active_lines(paths.current), "active segment")
    return walk.result()


class _ChainWalk:
    """Mutable verification state threaded across one store's segments.

    The previous raw line's hash is always tracked — it is what the
    current entry's link must commit to, whether or not that predecessor
    itself carried a link (the chain head is exactly what its successor
    committed to).
    """

    def __init__(self) -> None:
        self._ok = True
        self._head_sha256 = ""
        self._chained_entries = 0
        self._unchained_entries = 0
        self._verified_seals = 0
        self._unchained_seals = 0
        self._divergences: list[str] = []
        self._prev_hash: str | None = None
        self._seen_any_chained = False

    def diverge(self, message: str) -> None:
        self._ok = False
        self._divergences.append(message)

    def entries(self, lines: list[str], where: str) -> None:
        """Check one segment's raw lines against their committed links."""
        for position, raw_line in enumerate(lines, start=1):
            line_hash = _entry_sha256(raw_line)
            self._head_sha256 = line_hash
            try:
                entry = json.loads(raw_line)
                if not isinstance(entry, dict):
                    raise ValueError("not a JSON object")
            except ValueError:
                self.diverge(f"{where} line {position}: malformed entry (raw bytes changed?)")
                self._prev_hash = line_hash
                continue
            link = str(entry.get("prev_entry_sha256") or "")
            turn = entry.get("turn")
            if not link:
                if self._seen_any_chained:
                    self.diverge(
                        f"{where} line {position} (turn {turn}): chain link missing "
                        "mid-history — entry removed or rewritten"
                    )
                else:
                    self._unchained_entries += 1
            else:
                self._seen_any_chained = True
                if self._prev_hash is not None and link != self._prev_hash:
                    self.diverge(
                        f"{where} line {position} (turn {turn}): chain link does not "
                        f"match predecessor's hash (expected {self._prev_hash[:12]}…, "
                        f"found {link[:12]}…)"
                    )
                self._chained_entries += 1
            self._prev_hash = line_hash

    def seal(self, record: Any, lines: list[str]) -> None:
        """Check one seal's index-recorded terminal hash."""
        if not record.tail_sha256:
            self._unchained_seals += 1
            return
        if lines and _entry_sha256(lines[-1]) != record.tail_sha256:
            self.diverge(
                f"sealed segment {record.seg_id}: index terminal hash does not "
                "match the segment's raw terminal line"
            )
            return
        self._verified_seals += 1

    def result(self) -> ChainVerification:
        return ChainVerification(
            ok=self._ok,
            head_sha256=self._head_sha256,
            chained_entries=self._chained_entries,
            unchained_entries=self._unchained_entries,
            verified_seals=self._verified_seals,
            unchained_seals=self._unchained_seals,
            first_divergence=self._divergences[0] if self._divergences else "",
            divergences=list(self._divergences),
        )
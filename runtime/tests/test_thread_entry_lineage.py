"""What a persisted thread entry actually carries for execution identity.

The envelope contract for the decisions ledger is pinned in
``kernel/tests/test_evidence_envelope.py``; this is its thread-store twin. A
thread entry records the same three keys (``turn_id`` plus the two lineage
edges), and the failure modes are the same: a value threaded through contexts
but never written, or written as an empty string so "no parent" and "parent
unknown" become indistinguishable. Asserting on the entry read back off disk
is what makes those visible.

Evidence Envelope spec §4.
"""

from __future__ import annotations

import json
from pathlib import Path

from runtime.thread import append_thread_entry

#: Populated identities, shaped like ``mint_turn_id`` / ``mint_dispatch_id``
#: output so a prefix-reading consumer cannot mistake them for unset values.
_TURN_ID = "turn-" + "a" * 32
_PARENT_TURN_ID = "turn-" + "b" * 32
_PARENT_DISPATCH_ID = "dispatch-" + "c" * 32


def _persisted_entries(thread_path: Path) -> list[dict[str, object]]:
    """Return every entry written for ``thread_path``, oldest first.

    Reads the segment files the store wrote rather than assuming one name; the
    lock sidecar is skipped because it is not an entry file.
    """
    rows: list[dict[str, object]] = []
    for candidate in sorted(thread_path.parent.glob("*.jsonl")):
        if candidate.name.endswith(".lock"):
            continue
        text = candidate.read_text(encoding="utf-8")
        rows.extend(json.loads(line) for line in text.splitlines() if line.strip())
    return rows


def _append(thread_path: Path, **lineage: object) -> dict[str, object]:
    """Append one entry through the real writer and return it as persisted."""
    committed = append_thread_entry(
        thread_path=thread_path,
        user_slug="local-user",
        mode_id="default",
        query="what is the status",
        response_text="all clear",
        turn=0,
        run_ref="run-1",
        **lineage,
    )
    assert committed == 0, f"expected the first committed turn, got {committed}"
    entries = _persisted_entries(thread_path)
    assert len(entries) == 1, f"expected one persisted entry, got {len(entries)}"
    return entries[0]


def test_a_top_level_thread_entry_carries_its_turn_id_and_no_parent_keys(
    tmp_path: Path,
) -> None:
    """A top-level turn records the execution id and states no parent edge.

    The keys are *absent*, not empty: ``_stamp_lineage`` omits unknown lineage so
    the delegation tree is built from real edges rather than a graph of empty
    strings. This pins that choice so a later "always write the keys" change has
    to come with a reader-side decision.
    """
    entry = _append(tmp_path / "thread.jsonl", turn_id=_TURN_ID)

    assert entry["turn_id"] == _TURN_ID
    assert "parent_turn_id" not in entry
    assert "parent_dispatch_id" not in entry


def test_a_child_thread_entry_carries_both_parent_keys(tmp_path: Path) -> None:
    """A child turn records both lineage edges, so the tree is reconstructible.

        ``parent_dispatch_id`` is what disambiguates which of several spawn
    dispatches produced this child; ``parent_turn_id`` alone collapses a fan-out
        into one indistinguishable group.
    """
    entry = _append(
        tmp_path / "thread.jsonl",
        turn_id=_TURN_ID,
        parent_turn_id=_PARENT_TURN_ID,
        parent_dispatch_id=_PARENT_DISPATCH_ID,
    )

    assert entry["turn_id"] == _TURN_ID
    assert entry["parent_turn_id"] == _PARENT_TURN_ID
    assert entry["parent_dispatch_id"] == _PARENT_DISPATCH_ID


def test_thread_entry_carries_top_level_task_id(tmp_path: Path) -> None:
    entry = _append(tmp_path / "thread.jsonl", task_id="task-1")
    assert entry["task_id"] == "task-1"


def test_resumed_child_identity_preserves_parent_keys(tmp_path: Path) -> None:
    entry = _append(
        tmp_path / "thread.jsonl",
        turn_id=_TURN_ID,
        parent_turn_id=_PARENT_TURN_ID,
        parent_dispatch_id=_PARENT_DISPATCH_ID,
        task_id="task-1",
    )
    assert entry["turn_id"] == _TURN_ID
    assert entry["parent_turn_id"] == _PARENT_TURN_ID
    assert entry["parent_dispatch_id"] == _PARENT_DISPATCH_ID
    assert entry["task_id"] == "task-1"


def test_child_thread_entry_carries_task_and_parent_identity(tmp_path: Path) -> None:
    entry = _append(
        tmp_path / "thread.jsonl",
        task_id="task-1",
        turn_id=_TURN_ID,
        parent_turn_id=_PARENT_TURN_ID,
        parent_dispatch_id=_PARENT_DISPATCH_ID,
    )
    assert entry["task_id"] == "task-1"
    assert entry["parent_turn_id"] == _PARENT_TURN_ID
    assert entry["parent_dispatch_id"] == _PARENT_DISPATCH_ID


def test_an_entry_written_without_a_turn_id_omits_the_key(tmp_path: Path) -> None:
    """A turn with no execution identity says so by omission, not an empty string.

    A blank string would join against nothing and still look populated to a
    truthiness-blind reader; the writer's contract is absence.
    """
    entry = _append(tmp_path / "thread.jsonl")

    assert "turn_id" not in entry

"""Tests for the turn join and the forensic renderer.

The join is where a forensic claim is born, so the tests are about
attribution rather than formatting: an exact match joins, ``None`` never
joins, and anything unmatched is reported instead of quietly dropped.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sdk.forensics import (
    BundleSources,
    ForensicBundle,
    build_bundle,
    cost_from_record,
    decision_from_record,
    dispatch_from_record,
    render_bundle,
    turn_from_entry,
)

_NOW = datetime.now(UTC).isoformat()


def _entry(turn: int, **overrides: object) -> tuple[int, dict]:
    entry = {
        "timestamp_utc": _NOW,
        "mode_id": "coding_execute",
        "query": "do the thing",
        "response_text": "did the thing",
        "turn": turn,
        "provenance": {"source": "observed", "task_id": "task-1"},
        "effect_receipts": [{"capability_id": "read_file", "effect": "read"}],
    }
    entry.update(overrides)
    return turn, entry


def _bundle(
    *,
    turns: list[tuple[int, dict]] | None = None,
    decisions: list[dict] | None = None,
    dispatches: list[dict] | None = None,
    costs: list[dict] | None = None,
    notes: list[str] | None = None,
) -> ForensicBundle:
    return build_bundle(
        generated_at=_NOW,
        turns=[turn_from_entry(turn, entry) for turn, entry in (turns or [_entry(1)])],
        decisions=decisions or [],
        dispatches=dispatches or [],
        costs=costs or [],
        log_errors=[],
        sources=BundleSources(thread_store=True),
        notes=notes or [],
    )


# ── Projection ─────────────────────────────────────────────────────────


def test_turn_from_entry_reads_thread_fields() -> None:
    """The projection copies the thread's own fields, including receipts."""
    turn = turn_from_entry(
        *(_entry(7, handoff_to="coding_review", content_source="reasoning_tail"))
    )
    assert turn.turn_index == 7
    assert turn.mode_id == "coding_execute"
    assert turn.handoff_to == "coding_review"
    assert turn.content_source == "reasoning_tail"
    assert turn.receipt_capability_ids == ["read_file"]
    assert turn.task_id == "task-1"


def test_turn_from_entry_previews_long_text() -> None:
    """A forensic read is a summary: long fields are bounded."""
    turn = turn_from_entry(1, {"query": "q" * 5000})
    assert len(turn.query_preview) < 500
    assert turn.query_preview.endswith("…")


def test_decision_and_dispatch_projection_preserve_turn_index() -> None:
    """Attribution survives projection; nothing re-derives the index."""
    decision = decision_from_record({"turn_index": 12, "action_type": "FinishAction"})
    dispatch = dispatch_from_record({"turn_index": 12, "target_id": "read_file", "status": "error"})
    cost = cost_from_record({"turn_index": 12, "total_tokens": 10, "cached_tokens": 4})

    assert decision.turn_index == 12
    assert dispatch.turn_index == 12
    assert dispatch.failed is True
    assert cost.turn_index == 12
    assert cost.billed_tokens == 6


@pytest.mark.parametrize("value", [None, "x", -1, True, {}])
def test_unusable_turn_index_becomes_none_not_zero(value: object) -> None:
    """A bad index is unattributed; it must never masquerade as turn 0."""
    assert decision_from_record({"turn_index": value}).turn_index is None


# ── Join ───────────────────────────────────────────────────────────────


def test_records_join_on_exact_turn_index() -> None:
    """Only an exact index match attaches evidence to a turn."""
    bundle = _bundle(
        turns=[_entry(3), _entry(4)],
        decisions=[
            {"turn_index": 3, "action_type": "ContinueAction"},
            {"turn_index": 4, "action_type": "FinishAction"},
        ],
        dispatches=[{"turn_index": 4, "target_id": "write_file"}],
        costs=[{"turn_index": 3, "total_tokens": 100}],
    )

    by_turn = {turn.turn_index: turn for turn in bundle.turns}
    assert [d.action_type for d in by_turn[3].decisions] == ["ContinueAction"]
    assert by_turn[4].dispatches[0].target_id == "write_file"
    assert by_turn[3].total_tokens == 100
    assert by_turn[4].total_tokens == 0


def test_none_index_never_joins_a_turn() -> None:
    """Off-turn work must not be attributed to the first turn."""
    bundle = _bundle(
        turns=[_entry(0)],
        decisions=[{"turn_index": None, "action_type": "ContinueAction"}],
        dispatches=[{"turn_index": None, "target_id": "slice_work"}],
    )

    turn = bundle.turns[0]
    assert turn.decisions == []
    assert turn.dispatches == []
    # ...and it is still reported, because unattributed activity is a finding.
    assert len(bundle.unattributed_decisions) == 1
    assert len(bundle.unattributed_dispatches) == 1


def test_records_outside_the_requested_range_are_surfaced() -> None:
    """A record naming an unrequested turn is reported, not silently dropped."""
    bundle = _bundle(
        turns=[_entry(3)],
        dispatches=[{"turn_index": 99, "target_id": "elsewhere"}],
    )

    assert bundle.turns[0].dispatches == []
    assert [d.turn_index for d in bundle.unattributed_dispatches] == [99]
    rendered = render_bundle(bundle)
    assert "Unattributed records" in rendered
    assert "elsewhere" in rendered


def test_no_fuzzy_matching_across_neighbouring_turns() -> None:
    """A record for turn 5 never attaches to turn 4 or 6."""
    bundle = _bundle(turns=[_entry(4), _entry(6)], dispatches=[{"turn_index": 5}])
    assert bundle.turns[0].dispatches == []
    assert bundle.turns[1].dispatches == []


def test_turn_aggregates_sum_only_joined_costs() -> None:
    """Per-turn cost is the sum of records that actually joined."""
    bundle = _bundle(
        turns=[_entry(2)],
        costs=[
            {"turn_index": 2, "total_tokens": 100, "cost_usd": 0.25},
            {"turn_index": 2, "total_tokens": 50, "cost_usd": 0.1},
            {"turn_index": 3, "total_tokens": 999, "cost_usd": 9.0},
        ],
    )
    turn = bundle.turns[0]
    assert turn.total_tokens == 150
    assert turn.total_cost_usd == pytest.approx(0.35)


def test_failed_dispatches_are_isolated() -> None:
    """Failure detection is by status, not by message text."""
    bundle = _bundle(
        turns=[_entry(1)],
        dispatches=[
            {"turn_index": 1, "target_id": "a", "status": "success"},
            {"turn_index": 1, "target_id": "b", "status": "timeout"},
            {"turn_index": 1, "target_id": "c", "status": "error"},
        ],
    )
    assert [d.target_id for d in bundle.turns[0].failed_dispatches] == ["b", "c"]


# ── Renderer ───────────────────────────────────────────────────────────


def test_render_states_missing_sources_explicitly() -> None:
    """An unreadable log is 'unknown', never a clean result."""
    bundle = ForensicBundle(
        generated_at=_NOW,
        turns=[],
        sources=BundleSources(missing=["decisions.jsonl"]),
    )
    rendered = render_bundle(bundle)
    assert "decisions log: unavailable" in rendered
    assert "**missing:** decisions.jsonl" in rendered
    assert "not absent" in rendered


def test_render_marks_blocked_turns_with_origin() -> None:
    """A blocked turn states why, because that is the forensic question."""
    bundle = _bundle(
        turns=[
            _entry(
                1057,
                provenance={
                    "source": "observed",
                    "blocked": True,
                    "block_origin": "task_completion",
                },
            )
        ]
    )
    rendered = render_bundle(bundle)
    assert "Turn 1057" in rendered
    assert "blocked" in rendered
    assert "task_completion" in rendered


def test_render_says_when_cost_did_not_join() -> None:
    """Absence of cost evidence is stated, not shown as zero spend."""
    rendered = render_bundle(_bundle(turns=[_entry(1)]))
    assert "no token record joined this turn" in rendered


def test_render_dedupes_receipts_with_counts() -> None:
    """Forty identical reads become one counted entry."""
    bundle = _bundle(
        turns=[
            _entry(
                1,
                effect_receipts=[{"capability_id": "read_file", "effect": "read"}] * 5
                + [{"capability_id": "write_file", "effect": "write"}],
            )
        ]
    )
    rendered = render_bundle(bundle)
    assert "read_file×5" in rendered
    assert "write_file" in rendered


def test_render_bounds_long_decision_lists() -> None:
    """A 40-iteration turn keeps its framing and its ending, not the middle."""
    decisions = [
        {"turn_index": 1, "iteration": n, "action_type": "ContinueAction"} for n in range(1, 41)
    ]
    rendered = render_bundle(_bundle(turns=[_entry(1)], decisions=decisions))
    assert "iter 1:" in rendered
    assert "iter 40:" in rendered
    assert "elided" in rendered
    assert "iter 20:" not in rendered


def test_render_elides_older_turns_but_counts_them() -> None:
    """A wide range stays readable without hiding how much was summarized."""
    turns = [_entry(turn) for turn in range(12)]
    rendered = render_bundle(_bundle(turns=turns))
    assert "4 earlier turn(s) summarized" in rendered


def test_render_collapses_repeated_log_messages() -> None:
    """One repeated warning is one finding with a count, not N lines."""
    bundle = build_bundle(
        generated_at=_NOW,
        turns=[turn_from_entry(*_entry(1))],
        decisions=[],
        dispatches=[],
        costs=[],
        log_errors=[{"level": "warning", "msg": "same warning", "ts": _NOW}] * 9,
        sources=BundleSources(runtime_log=True),
        notes=[],
    )
    rendered = render_bundle(bundle)
    assert "same warning" in rendered
    assert rendered.count("same warning") == 1
    assert "×9" in rendered
    assert "9 line(s), 1 distinct" in rendered


def test_render_keeps_log_detail_when_messages_differ() -> None:
    """Distinct messages are never merged, even at the same level."""
    bundle = build_bundle(
        generated_at=_NOW,
        turns=[turn_from_entry(*_entry(1))],
        decisions=[],
        dispatches=[],
        costs=[],
        log_errors=[
            {"level": "warning", "msg": "first", "ts": _NOW},
            {"level": "warning", "msg": "second", "ts": _NOW},
        ],
        sources=BundleSources(runtime_log=True),
        notes=[],
    )
    rendered = render_bundle(bundle)
    assert "first" in rendered
    assert "second" in rendered


def test_render_is_deterministic() -> None:
    """The same bundle renders byte-identically (no time or dict-order drift)."""
    bundle = _bundle(turns=[_entry(1)], dispatches=[{"turn_index": 1, "target_id": "x"}])
    assert render_bundle(bundle) == render_bundle(bundle)


def test_render_always_points_back_to_the_evidence() -> None:
    """Every bundle ends by telling the reader how to verify a claim."""
    rendered = render_bundle(_bundle())
    assert "thread_read_range" in rendered


def test_join_matches_telemetry_by_the_index_the_turn_ran_under() -> None:
    """Overlapping turns that commit out of start order keep their telemetry.

    Turn A ran as 7 but committed second (8); turn B ran as 8 but committed
    first (7). Mid-turn records carry the provisional index, so joining by
    the committed index would swap the two turns' evidence.
    """
    bundle = _bundle(
        turns=[
            _entry(7, provisional_turn=8, mode_id="b_mode"),
            _entry(8, provisional_turn=7, mode_id="a_mode"),
        ],
        decisions=[
            {"turn_index": 7, "mode_id": "a_mode", "action_type": "finish"},
            {"turn_index": 8, "mode_id": "b_mode", "action_type": "finish"},
        ],
        costs=[{"turn_index": 7, "mode_id": "a_mode", "total_tokens": 10}],
    )
    by_index = {turn.turn_index: turn for turn in bundle.turns}
    assert [d.mode_id for d in by_index[7].decisions] == ["b_mode"]
    assert [d.mode_id for d in by_index[8].decisions] == ["a_mode"]
    assert by_index[8].total_tokens == 10
    assert not bundle.unattributed_decisions
    assert "ran as turn 8" in render_bundle(bundle)


def test_join_keys_by_committed_index_without_provisional_turn() -> None:
    turn = turn_from_entry(*_entry(3))
    assert turn.provisional_turn is None
    assert turn.telemetry_turn_index == 3

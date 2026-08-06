"""Tests for ``kernel.turn_phases._turn_state.merge_child_usage``.

The parent turn loop calls this after a fork/delegate operation returns to
merge the children's per-model / per-mode token usage into the parent
``_TurnState`` ledger. Without this merge, ``/tokens`` attributes a
multi-model fork's tokens only to the parent's resolved default model —
the 2026-08-06 regression.
"""

from __future__ import annotations

from kernel.turn_phases._turn_state import _TurnState, merge_child_usage


def test_merge_child_usage_accumulates_per_model_by_model_id() -> None:
    """Two children calling the same model merge into one row; distinct
    models stay separate."""
    state = _TurnState()
    merge_child_usage(
        state,
        per_model_usage=[
            {
                "model_id": "child-model",
                "prompt_tokens": 40,
                "completion_tokens": 20,
                "total_tokens": 60,
                "cached_tokens": 0,
                "reasoning_tokens": 0,
                "duration_ms": 10.0,
                "latency_ms": 0,
            },
            {
                "model_id": "child-model",
                "prompt_tokens": 30,
                "completion_tokens": 15,
                "total_tokens": 45,
                "cached_tokens": 5,
                "reasoning_tokens": 0,
                "duration_ms": 20.0,
                "latency_ms": 0,
            },
            {
                "model_id": "other-model",
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cached_tokens": 0,
                "reasoning_tokens": 0,
                "duration_ms": 5.0,
                "latency_ms": 0,
            },
        ],
        per_mode_usage=None,
    )
    assert state.per_model_usage["child-model"]["total_tokens"] == 105
    assert state.per_model_usage["child-model"]["cached_tokens"] == 5
    assert state.per_model_usage["child-model"]["duration_ms"] == 30.0
    assert state.per_model_usage["other-model"]["total_tokens"] == 15
    assert "default-mode" not in state.per_mode_usage


def test_merge_child_usage_accumulates_per_mode_by_mode_id() -> None:
    """Per-mode usage merges by ``mode_id``."""
    state = _TurnState()
    merge_child_usage(
        state,
        per_model_usage=None,
        per_mode_usage=[
            {
                "mode_id": "search",
                "prompt_tokens": 80,
                "completion_tokens": 50,
                "total_tokens": 130,
                "cached_tokens": 0,
                "reasoning_tokens": 0,
                "duration_ms": 400.0,
                "latency_ms": 0,
            },
            {
                "mode_id": "coding",
                "prompt_tokens": 100,
                "completion_tokens": 60,
                "total_tokens": 160,
                "cached_tokens": 0,
                "reasoning_tokens": 0,
                "duration_ms": 600.0,
                "latency_ms": 0,
            },
        ],
    )
    assert state.per_mode_usage["search"]["total_tokens"] == 130
    assert state.per_mode_usage["coding"]["total_tokens"] == 160


def test_merge_child_usage_is_noop_on_empty_or_none() -> None:
    """Empty / None lists leave the state ledger untouched."""
    state = _TurnState()
    state.per_model_usage["existing"] = {"model_id": "existing", "total_tokens": 5}
    merge_child_usage(state, per_model_usage=None, per_mode_usage=None)
    merge_child_usage(state, per_model_usage=[], per_mode_usage=[])
    assert state.per_model_usage["existing"]["total_tokens"] == 5
    assert len(state.per_model_usage) == 1


def test_merge_child_usage_coerces_garbage_entries_to_zero() -> None:
    """Malformed entries (missing / non-numeric fields) coerce to 0
    instead of raising so a bad child payload never corrupts the ledger."""
    state = _TurnState()
    merge_child_usage(
        state,
        per_model_usage=[
            {"model_id": "loose-model", "total_tokens": "not-a-number"},
            {"model_id": "loose-model", "total_tokens": 20},
        ],
        per_mode_usage=None,
    )
    assert state.per_model_usage["loose-model"]["total_tokens"] == 20


def test_merge_child_usage_merges_with_existing_state_entries() -> None:
    """A child calling the parent's already-recorded model adds into the
    existing row rather than overwriting it (no double-count, no drop)."""
    state = _TurnState()
    state.per_model_usage["shared-model"] = {
        "model_id": "shared-model",
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "duration_ms": 100.0,
        "latency_ms": 0,
    }
    merge_child_usage(
        state,
        per_model_usage=[
            {
                "model_id": "shared-model",
                "prompt_tokens": 40,
                "completion_tokens": 20,
                "total_tokens": 60,
                "cached_tokens": 0,
                "reasoning_tokens": 0,
                "duration_ms": 10.0,
                "latency_ms": 0,
            },
        ],
        per_mode_usage=None,
    )
    assert state.per_model_usage["shared-model"]["total_tokens"] == 210
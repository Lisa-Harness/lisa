"""Regression tests for intra-turn compaction (estimator, epoch, gates, spill)."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import Any

from kernel.provider_contracts import RequestContextBudget
from kernel.turn_phases._compaction import (
    IntraTurnCompactionState,
    LiveArrayEstimator,
    hard_ceiling_stop_reason,
    maybe_run_soft_epoch,
    tool_size_refusal,
)
from kernel.turn_phases._spill import maybe_spill_tool_result


def _est(tokens: int) -> LiveArrayEstimator:
    estimator = LiveArrayEstimator()
    estimator.observe_request_tokens(tokens)
    return estimator


def _state(cfg: IntraTurnCompactionState | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        intra_turn_compaction=cfg if cfg is not None else IntraTurnCompactionState(),
        intra_turn_compaction_estimator=LiveArrayEstimator(),
        provider_messages=[],
        projected_message_count=0,
        request_context_budget=None,
    )


def _budget(window: int) -> RequestContextBudget:
    return RequestContextBudget(
        window_tokens=window,
        requested_completion_tokens=100,
        completion_tokens=100,
        margin_tokens=50,
        pressure_headroom_tokens=0,
        required_tokens=10,
        input_tokens=0,
        tool_budget_bytes=0,
        counting_method="utf8_bytes_div_3_estimate",
        reduced=False,
    )


def _shortening_projector(
    messages: list[dict[str, Any]], task_id: str, *, budget_bytes: int
) -> list[dict[str, Any]]:
    del task_id
    return [
        {**m, "content": str(m.get("content") or "")[:budget_bytes]}
        if m.get("role") == "tool"
        else dict(m)
        for m in messages
    ]


def _hooks_with(projector: Any) -> SimpleNamespace:
    return SimpleNamespace(
        context=SimpleNamespace(project_messages=projector),
        identity=SimpleNamespace(task_id="task-1"),
    )


def _quiet_log() -> SimpleNamespace:
    return SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )


# ── Estimator ──────────────────────────────────────────────────────────


def test_estimator_unobserved_returns_zero() -> None:
    assert LiveArrayEstimator().estimate() == 0


def test_estimator_anchors_on_counted_tokens_and_over_counts() -> None:
    estimator = _est(1000)
    estimator.observe_append({"content": "x" * 300})
    estimate = estimator.estimate()
    # Over-count bias: never below the authoritative anchor, and the
    # 300 appended bytes convert at 3 bytes/token (100 tokens) times the
    # 1.05 safety factor.
    assert estimate >= 1000 + 100


def test_estimator_adopted_after_epoch_resync_stays_consistent() -> None:
    estimator = _est(5000)
    estimator.observe_request_tokens(700)
    assert estimator.estimate() >= 700


# ── Soft-trigger epoch ────────────────────────────────────────────────


def _epoch_fixture(
    *, window: int, estimate: int
) -> tuple[SimpleNamespace, SimpleNamespace, list[dict[str, Any]]]:
    state = _state()
    state.request_context_budget = _budget(window)
    state.intra_turn_compaction_estimator.observe_request_tokens(estimate)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "tool", "tool_call_id": "c1", "name": "read", "content": "y" * 24000},
    ]
    return state, _hooks_with(_shortening_projector), messages


def test_soft_epoch_fires_once_per_crossing_and_rewrites_array() -> None:
    state, hooks, messages = _epoch_fixture(window=1000, estimate=900)
    ran = maybe_run_soft_epoch(
        state=state,
        hooks=hooks,
        messages=messages,
        model_def=SimpleNamespace(context_window=1000),
        tools=None,
        max_tokens=100,
        mode_id="coding_execute",
        prompt_cache_key="k",
        log=_quiet_log(),
    )
    assert ran is True
    assert state.intra_turn_compaction.epoch_done_for_crossing is True
    # The live array was rewritten in place, same element count, smaller tool body.
    assert len(messages) == 3
    assert len(str(messages[2]["content"])) < 24000


def test_soft_epoch_hysteresis_blocks_second_run() -> None:
    state, hooks, messages = _epoch_fixture(window=1000, estimate=900)
    common = dict(
        hooks=hooks,
        messages=messages,
        model_def=SimpleNamespace(context_window=1000),
        tools=None,
        max_tokens=100,
        mode_id="m",
        prompt_cache_key="k",
        log=_quiet_log(),
    )
    assert maybe_run_soft_epoch(state=state, **common) is True
    assert maybe_run_soft_epoch(state=state, **common) is False


def test_soft_epoch_inert_when_disabled_or_unobserved() -> None:
    cfg = IntraTurnCompactionState(enabled=False)
    state, hooks, messages = _epoch_fixture(window=1000, estimate=900)
    state.intra_turn_compaction = cfg
    common = dict(
        hooks=hooks,
        messages=messages,
        model_def=SimpleNamespace(context_window=1000),
        tools=None,
        max_tokens=100,
        mode_id="m",
        prompt_cache_key="k",
        log=_quiet_log(),
    )
    # Disabled config: inert.
    assert maybe_run_soft_epoch(state=state, **common) is False
    # Unobserved estimator: inert.
    fresh = _state()
    fresh.request_context_budget = _budget(1000)
    assert maybe_run_soft_epoch(state=fresh, **common) is False


def test_epoch_failure_keeps_array_and_sets_hysteresis_flag() -> None:
    state, _, messages = _epoch_fixture(window=1000, estimate=900)

    def _raising_projector(
        messages: list[dict[str, Any]], task_id: str, *, budget_bytes: int
    ) -> list[dict[str, Any]]:
        raise RuntimeError("projector exploded")

    ran = maybe_run_soft_epoch(
        state=state,
        hooks=_hooks_with(_raising_projector),
        messages=messages,
        model_def=SimpleNamespace(context_window=1000),
        tools=None,
        max_tokens=100,
        mode_id="m",
        prompt_cache_key="k",
        log=_quiet_log(),
    )
    assert ran is False
    assert str(messages[2]["content"]) == "y" * 24000
    # A failed epoch is not retried every dispatch boundary.
    assert state.intra_turn_compaction.epoch_done_for_crossing is True


# ── Hard ceiling + dispatch gate ──────────────────────────────────────


def test_hard_ceiling_reads_window_from_request_budget() -> None:
    state = _state()
    state.request_context_budget = _budget(1000)
    state.intra_turn_compaction_estimator.observe_request_tokens(700)
    assert hard_ceiling_stop_reason(state=state) == "context_hard_ceiling"
    low = _state()
    low.request_context_budget = _budget(1000)
    low.intra_turn_compaction_estimator.observe_request_tokens(600)
    assert hard_ceiling_stop_reason(state=low) == ""


def test_hard_ceiling_inert_without_budget_or_flag() -> None:
    state = _state()
    state.intra_turn_compaction_estimator.observe_request_tokens(9999)
    assert hard_ceiling_stop_reason(state=state) == ""
    state.request_context_budget = _budget(1000)
    state.intra_turn_compaction = IntraTurnCompactionState(enabled=False)
    assert hard_ceiling_stop_reason(state=state) == ""


def test_tool_size_refusal_gates_only_large_dispatches_under_ceiling() -> None:
    state = _state()
    state.request_context_budget = _budget(1000)
    state.intra_turn_compaction_estimator.observe_request_tokens(700)
    small = "x" * 100
    assert tool_size_refusal(state=state, tool_arguments={"path": small}) == ""
    big = "x" * 3000
    refusal = tool_size_refusal(state=state, tool_arguments={"content": big})
    assert "hard ceiling" in refusal
    below = _state()
    below.request_context_budget = _budget(1000)
    below.intra_turn_compaction_estimator.observe_request_tokens(100)
    assert tool_size_refusal(state=below, tool_arguments={"content": big}) == ""


# ── Spill-on-arrival ──────────────────────────────────────────────────


def _persist_capture(store: list[dict[str, Any]]) -> Any:
    def _persist(record: dict[str, Any]) -> str:
        store.append(record)
        return (
            "r"
            + hashlib.sha256(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        )

    return _persist


def test_spill_replaces_large_result_with_preview_and_ref() -> None:
    store: list[dict[str, Any]] = []
    message = {
        "role": "tool",
        "tool_call_id": "c1",
        "name": "read_file",
        "content": "z" * 10000,
    }
    spilled = maybe_spill_tool_result(
        result_message=message,
        persist=_persist_capture(store),
        task_id="task-1",
        preview_repeat_limit=3,
    )
    assert spilled is True
    content = str(message["content"])
    assert content.startswith("[spilled tool result (10000 chars)")
    assert store and store[0]["content"] == "z" * 10000
    assert store[0]["tool_name"] == "read_file"
    assert store[0]["task_id"] == "task-1"
    assert content.count("r") >= 1


def test_spill_ignores_small_and_already_ref_content() -> None:
    store: list[dict[str, Any]] = []
    small = {"role": "tool", "content": "z" * 100}
    assert (
        maybe_spill_tool_result(
            result_message=small,
            persist=_persist_capture(store),
            task_id="t",
            preview_repeat_limit=3,
        )
        is False
    )
    assert small["content"] == "z" * 100
    refed = {"role": "tool", "content": "r" + "a" * 64}
    assert (
        maybe_spill_tool_result(
            result_message=refed,
            persist=_persist_capture(store),
            task_id="t",
            preview_repeat_limit=3,
        )
        is False
    )
    assert not store


def test_spill_preview_collapses_repeated_lines() -> None:
    store: list[dict[str, Any]] = []
    line = "ValidationError: field 'edits.0' missing expected_sha256"
    # Enough total bytes to cross SPILL_MIN_BYTES so the spill path runs.
    # The repeated block sits beyond the head excerpt: the head shows the
    # line once, and the inventory must not re-add it, so the final
    # preview holds exactly one occurrence.
    filler = ("z" * 120 + "\n") * 35
    content = "\n".join(["header line of the result", filler, *([line] * 200), "trailer"])
    message = {"role": "tool", "content": content, "name": "edit_files"}
    assert (
        maybe_spill_tool_result(
            result_message=message,
            persist=_persist_capture(store),
            task_id="t",
            preview_repeat_limit=3,
        )
        is True
    )
    preview = str(message["content"])
    assert "repeated lines (collapsed; full text via ref):" in preview
    assert preview.count(line) == 1
    assert content.count(line) == 200


def test_spill_noop_without_persist_closure() -> None:
    message = {"role": "tool", "content": "z" * 10000}
    assert (
        maybe_spill_tool_result(
            result_message=message,
            persist=None,
            task_id="t",
            preview_repeat_limit=3,
        )
        is False
    )
    assert message["content"] == "z" * 10000


# ── Config snapshot + breaker defaults ────────────────────────────────


def test_compaction_state_defaults_and_snapshot() -> None:
    cfg = IntraTurnCompactionState()
    assert cfg.enabled is True
    assert cfg.epoch_done_for_crossing is False
    assert cfg.finish_now is False
    snap = IntraTurnCompactionState.from_settings()
    assert snap.soft_trigger_share > 0.0
    assert snap.hard_ceiling_share > snap.soft_trigger_share

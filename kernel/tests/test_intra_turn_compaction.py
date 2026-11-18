"""Regression tests for intra-turn compaction (estimator, epoch, gates, spill)."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from types import SimpleNamespace
from typing import Any

import pytest
from kernel.provider_contracts import RequestContextBudget
from kernel.turn_phases._compaction import (
    IntraTurnCompactionState,
    LiveArrayEstimator,
    compaction_diagnostics,
    effective_window_tokens,
    hard_ceiling_stop_reason,
    log_compaction_diagnostics,
    maybe_run_soft_epoch,
    should_spill_on_arrival,
    soft_trigger_tokens,
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


def test_estimator_does_not_recount_bytes_after_a_new_request_anchor() -> None:
    estimator = _est(1000)
    estimator.observe_append({"content": "x" * 300})
    assert estimator.estimate() == 1155
    estimator.observe_request_tokens(1100)
    assert estimator.estimate() == 1155
    estimator.observe_append({"content": "y" * 300})
    assert estimator.estimate() == 1260


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
    assert state.intra_turn_compaction.epoch_estimate > 0
    # The live array was rewritten in place, same element count, smaller tool body.
    assert len(messages) == 3
    assert len(str(messages[2]["content"])) < 24000


def test_soft_epoch_hysteresis_blocks_second_run() -> None:
    state, hooks, messages = _epoch_fixture(window=1000, estimate=900)
    common: dict[str, Any] = {
        "hooks": hooks,
        "messages": messages,
        "model_def": SimpleNamespace(context_window=1000),
        "tools": None,
        "max_tokens": 100,
        "mode_id": "m",
        "prompt_cache_key": "k",
        "log": _quiet_log(),
    }
    assert maybe_run_soft_epoch(state=state, **common) is True
    assert maybe_run_soft_epoch(state=state, **common) is False


def test_soft_epoch_inert_when_disabled_or_unobserved() -> None:
    cfg = IntraTurnCompactionState(enabled=False)
    state, hooks, messages = _epoch_fixture(window=1000, estimate=900)
    state.intra_turn_compaction = cfg
    common: dict[str, Any] = {
        "hooks": hooks,
        "messages": messages,
        "model_def": SimpleNamespace(context_window=1000),
        "tools": None,
        "max_tokens": 100,
        "mode_id": "m",
        "prompt_cache_key": "k",
        "log": _quiet_log(),
    }
    # Disabled config: inert.
    assert maybe_run_soft_epoch(state=state, **common) is False
    # Unobserved estimator: inert.
    fresh = _state()
    fresh.request_context_budget = _budget(1000)
    assert maybe_run_soft_epoch(state=fresh, **common) is False


def test_epoch_failure_keeps_array_and_allows_later_retry() -> None:
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
    assert state.intra_turn_compaction.epoch_estimate > 0


def test_soft_epoch_rearms_after_falling_below_trigger() -> None:
    state, hooks, messages = _epoch_fixture(window=1000, estimate=900)
    common: dict[str, Any] = {
        "hooks": hooks,
        "messages": messages,
        "model_def": SimpleNamespace(context_window=1000),
        "tools": None,
        "max_tokens": 100,
        "mode_id": "m",
        "prompt_cache_key": "k",
        "log": _quiet_log(),
    }
    assert maybe_run_soft_epoch(state=state, **common) is True
    state.intra_turn_compaction_estimator = _est(100)
    assert maybe_run_soft_epoch(state=state, **common) is False
    state.intra_turn_compaction_estimator.observe_request_tokens(900)
    messages[2]["content"] = "x" * 24000
    assert maybe_run_soft_epoch(state=state, **common) is True


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
    assert cfg.epoch_estimate == 0
    assert cfg.finish_now is False
    snap = IntraTurnCompactionState.from_settings()
    assert snap.soft_trigger_share > 0.0
    assert snap.hard_ceiling_share > snap.soft_trigger_share


def test_mode_working_set_caps_soft_trigger() -> None:
    common: dict[str, Any] = {
        "model_def": SimpleNamespace(context_window=10_000),
        "tools": None,
        "max_tokens": 100,
        "mode_id": "coding_execute",
        "prompt_cache_key": "k",
        "log": _quiet_log(),
    }
    # 900 tokens is far under half of a 10K window: no epoch by share alone.
    state, hooks, messages = _epoch_fixture(window=10_000, estimate=900)
    assert maybe_run_soft_epoch(state=state, hooks=hooks, messages=messages, **common) is False
    # A mode working set of 800 tokens pulls the trigger down under the estimate.
    state, hooks, messages = _epoch_fixture(window=10_000, estimate=900)
    state.mode_config = SimpleNamespace(working_set_tokens=800)
    assert maybe_run_soft_epoch(state=state, hooks=hooks, messages=messages, **common) is True
    assert len(str(messages[2]["content"])) < 24000


def test_soft_trigger_tokens_uses_working_set_and_window_budget() -> None:
    state = _state()
    state.request_context_budget = _budget(10_000)
    assert soft_trigger_tokens(state=state) == 5_000
    state.mode_config = SimpleNamespace(working_set_tokens=800)
    assert soft_trigger_tokens(state=state) == 800


def test_turn_token_envelope_caps_large_model_window() -> None:
    # Turn 1359 shape: 1M window, 1M processed-token envelope, ~330K
    # resident prompt. The raw window put the soft trigger at 524K, so
    # the prompt never compacted while cache-cold resends drained the
    # envelope. The envelope bounds the effective window to 400K.
    state = _state(IntraTurnCompactionState(turn_token_envelope=1_000_000))
    state.request_context_budget = _budget(1_048_576)
    assert soft_trigger_tokens(state=state) == 200_000
    state.intra_turn_compaction_estimator.observe_request_tokens(313_978)
    assert should_spill_on_arrival(state=state) is True
    assert hard_ceiling_stop_reason(state=state) == "context_hard_ceiling"


def test_envelope_never_raises_a_smaller_model_window() -> None:
    state = _state(IntraTurnCompactionState(turn_token_envelope=1_000_000))
    state.request_context_budget = _budget(128_000)
    assert soft_trigger_tokens(state=state) == 64_000


def test_spill_on_arrival_is_gated_by_soft_trigger_pressure() -> None:
    state = _state()
    state.request_context_budget = _budget(10_000)
    state.intra_turn_compaction_estimator.observe_request_tokens(4_700)
    assert should_spill_on_arrival(state=state) is False
    state.intra_turn_compaction_estimator.observe_request_tokens(4_800)
    assert should_spill_on_arrival(state=state) is True


def test_soft_epoch_accepts_ref_backed_inventory_projection() -> None:
    state, _, messages = _epoch_fixture(window=1000, estimate=900)

    def _inventory_projector(
        source: list[dict[str, Any]], task_id: str, *, budget_bytes: int
    ) -> list[dict[str, Any]]:
        del task_id, budget_bytes
        return [
            {
                "role": "assistant",
                "content": json.dumps({"inventory_ref": "r" + "a" * 64}),
            },
            {"role": "tool", "tool_call_id": "recent", "content": "recent evidence"},
        ]

    assert (
        maybe_run_soft_epoch(
            state=state,
            hooks=_hooks_with(_inventory_projector),
            messages=messages,
            model_def=SimpleNamespace(context_window=1000),
            tools=None,
            max_tokens=100,
            mode_id="coding_execute",
            prompt_cache_key="k",
            log=_quiet_log(),
        )
        is True
    )
    assert len(messages) == 2
    assert messages[0]["role"] == "assistant"
    assert state.projected_message_count == len(messages)
    assert len(state.provider_messages) == len(messages)


def test_soft_epoch_preserves_projection_prefix_and_resume_state() -> None:
    state, hooks, messages = _epoch_fixture(window=1000, estimate=900)
    state.provider_messages = copy.deepcopy(messages)
    state.projected_message_count = len(messages)
    messages.extend(
        [
            {"role": "assistant", "content": "planning the next read"},
            {"role": "tool", "tool_call_id": "new", "content": "new evidence"},
        ]
    )
    assert (
        maybe_run_soft_epoch(
            state=state,
            hooks=hooks,
            messages=messages,
            model_def=SimpleNamespace(context_window=1000),
            tools=None,
            max_tokens=100,
            mode_id="m",
            prompt_cache_key="k",
            log=_quiet_log(),
        )
        is True
    )
    assert state.provider_messages == messages


def test_noop_epoch_does_not_consume_the_crossing() -> None:
    state, hooks, messages = _epoch_fixture(window=1000, estimate=900)
    messages[2]["content"] = "small result"
    common: dict[str, Any] = {
        "hooks": hooks,
        "messages": messages,
        "model_def": SimpleNamespace(context_window=1000),
        "tools": None,
        "max_tokens": 100,
        "mode_id": "m",
        "prompt_cache_key": "k",
        "log": _quiet_log(),
    }
    assert maybe_run_soft_epoch(state=state, **common) is False
    first_estimate = state.intra_turn_compaction.epoch_estimate
    assert first_estimate > 0
    state.intra_turn_compaction_estimator.observe_request_tokens(950)
    assert maybe_run_soft_epoch(state=state, **common) is False
    state.intra_turn_compaction_estimator.observe_request_tokens(first_estimate + 200)
    messages[2]["content"] = "x" * 24000
    assert maybe_run_soft_epoch(state=state, **common) is True


def test_failed_epoch_can_retry_after_recovery() -> None:
    state, _, messages = _epoch_fixture(window=1000, estimate=900)
    calls = 0

    def _recovering_projector(
        source: list[dict[str, Any]], task_id: str, *, budget_bytes: int
    ) -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary projector failure")
        return _shortening_projector(source, task_id, budget_bytes=budget_bytes)

    common: dict[str, Any] = {
        "hooks": _hooks_with(_recovering_projector),
        "messages": messages,
        "model_def": SimpleNamespace(context_window=1000),
        "tools": None,
        "max_tokens": 100,
        "mode_id": "m",
        "prompt_cache_key": "k",
        "log": _quiet_log(),
    }
    assert maybe_run_soft_epoch(state=state, **common) is False
    failed_estimate = state.intra_turn_compaction.epoch_estimate
    assert failed_estimate > 0
    assert maybe_run_soft_epoch(state=state, **common) is False
    state.intra_turn_compaction_estimator.observe_request_tokens(failed_estimate + 100)
    assert maybe_run_soft_epoch(state=state, **common) is True


def test_working_set_limits_results_without_shrinking_the_model_window() -> None:
    state, hooks, messages = _epoch_fixture(window=1_000_000, estimate=80_000)
    state.mode_config = SimpleNamespace(working_set_tokens=800)
    assert (
        maybe_run_soft_epoch(
            state=state,
            hooks=hooks,
            messages=messages,
            model_def=SimpleNamespace(context_window=1_000_000),
            tools=None,
            max_tokens=16384,
            mode_id="coding_execute",
            prompt_cache_key="k",
            log=_quiet_log(),
        )
        is True
    )
    assert state.request_context_budget is not None
    assert state.request_context_budget.window_tokens == 1_000_000
    assert len(str(messages[2]["content"])) < 24000


def test_envelope_epoch_evicts_without_a_mode_working_set() -> None:
    # No working set + a 1M window used to pass no residency budget, so the
    # projector kept every byte. The envelope-derived trigger now sizes it.
    state, hooks, messages = _epoch_fixture(window=1_000_000, estimate=300_000)
    state.intra_turn_compaction.turn_token_envelope = 1_000_000
    messages[2]["content"] = "y" * 900_000
    assert (
        maybe_run_soft_epoch(
            state=state,
            hooks=hooks,
            messages=messages,
            model_def=SimpleNamespace(context_window=1_000_000),
            tools=None,
            max_tokens=16384,
            mode_id="coding_review",
            prompt_cache_key="k",
            log=_quiet_log(),
        )
        is True
    )
    assert len(str(messages[2]["content"])) <= 100_000 * 3


def test_compaction_state_snapshots_history_epoch_setting() -> None:
    assert IntraTurnCompactionState.from_settings().history_epoch_min_tokens == 12_000


# ── Dynamic remaining-envelope cap ─────────────────────────────────────


def _envelope_state(
    *,
    envelope: int,
    window: int,
    total_tokens: int = 0,
    total_cached_tokens: int = 0,
    total_prompt_tokens: int = 0,
) -> SimpleNamespace:
    cfg = IntraTurnCompactionState()
    cfg.turn_token_envelope = envelope
    return SimpleNamespace(
        intra_turn_compaction=cfg,
        intra_turn_compaction_estimator=LiveArrayEstimator(),
        provider_messages=[],
        projected_message_count=0,
        request_context_budget=_budget(window),
        total_tokens=total_tokens,
        total_cached_tokens=total_cached_tokens,
        total_prompt_tokens=total_prompt_tokens,
    )


def test_effective_window_at_turn_start_matches_static_cap() -> None:
    # E=1M, W=1,048,576, P=0: effective 400K, soft 200K, hard 260K.
    state = _envelope_state(envelope=1_000_000, window=1_048_576)
    assert effective_window_tokens(state=state) == 400_000
    assert soft_trigger_tokens(state=state) == 200_000
    assert hard_ceiling_stop_reason(state=state) == ""


def test_effective_window_decays_with_processed_tokens() -> None:
    # E=1M, P=800K: R=200K, effective 80K, soft 40K, hard 52K.
    state = _envelope_state(envelope=1_000_000, window=1_048_576, total_tokens=800_000)
    assert effective_window_tokens(state=state) == 80_000
    assert soft_trigger_tokens(state=state) == 40_000


def test_effective_window_high_cache_leaves_more_remaining() -> None:
    # total=900K, cached=800K: P=100K, R=900K — a larger effective window
    # than the cold shape, but never above the static/model cap.
    state = _envelope_state(
        envelope=1_000_000,
        window=1_048_576,
        total_tokens=900_000,
        total_cached_tokens=800_000,
    )
    assert effective_window_tokens(state=state) == 360_000
    cold = _envelope_state(envelope=1_000_000, window=1_048_576, total_tokens=900_000)
    assert effective_window_tokens(state=cold) < effective_window_tokens(state=state)


def test_effective_window_small_model_window_unchanged() -> None:
    # W=128K with no spend: effective 128K, soft 64K; the envelope never
    # raises a small model window.
    state = _envelope_state(envelope=1_000_000, window=128_000)
    assert effective_window_tokens(state=state) == 128_000
    assert soft_trigger_tokens(state=state) == 64_000


def test_effective_window_no_envelope_returns_raw_window() -> None:
    # E=0: raw window behavior is unchanged.
    state = _envelope_state(envelope=0, window=1_048_576)
    assert effective_window_tokens(state=state) == 1_048_576


def test_effective_window_exhausted_envelope_collapses_to_zero() -> None:
    # R=0: no negative thresholds; the effective window collapses so the
    # existing finish path ends the turn.
    state = _envelope_state(envelope=1_000_000, window=1_048_576, total_tokens=1_000_000)
    assert effective_window_tokens(state=state) == 0
    assert soft_trigger_tokens(state=state) == 0


def test_processed_token_accounting_matches_budget_rail() -> None:
    # The compaction cap and budget_forces_final must use identical
    # processed-token arithmetic.
    from kernel.turn_phases import _token_accounting

    assert _token_accounting.processed_tokens(900_000, 800_000) == 100_000
    assert _token_accounting.processed_tokens(100_000, 200_000) == 0  # clamped
    assert _token_accounting.remaining_envelope(1_000_000, 100_000) == 900_000
    assert _token_accounting.remaining_envelope(0, 100_000) is None
    assert _token_accounting.remaining_envelope(1_000_000, 1_000_000) == 0


def test_cache_hit_rate_is_diagnostic_only() -> None:
    from kernel.turn_phases import _token_accounting

    assert _token_accounting.cache_hit_rate(800_000, 900_000) == pytest.approx(0.8889, abs=1e-3)
    assert _token_accounting.cache_hit_rate(0, 0) is None
    assert _token_accounting.cache_hit_rate(100, 0) is None


def test_compaction_diagnostics_snapshot_is_numeric_only() -> None:
    state = _envelope_state(
        envelope=1_000_000,
        window=1_048_576,
        total_tokens=900_000,
        total_cached_tokens=800_000,
        total_prompt_tokens=900_000,
    )
    d = compaction_diagnostics(state=state, mode_id="coding_execute")
    assert d.mode_id == "coding_execute"
    assert d.raw_window == 1_048_576
    assert d.effective_window == 360_000
    assert d.envelope == 1_000_000
    assert d.processed == 100_000
    assert d.remaining == 900_000
    assert d.cached == 800_000
    assert d.prompt == 900_000
    assert d.cache_hit_rate == pytest.approx(0.8889, abs=1e-3)
    assert d.share == 0.4


def test_compaction_diagnostics_log_is_numeric_only(caplog: pytest.LogCaptureFixture) -> None:
    state = _envelope_state(
        envelope=1_000_000,
        window=1_048_576,
        total_tokens=900_000,
        total_cached_tokens=800_000,
        total_prompt_tokens=900_000,
    )
    with caplog.at_level(logging.DEBUG, logger="kernel.turn_phases._compaction"):
        log_compaction_diagnostics(
            state=state,
            mode_id="coding_execute",
            log=logging.getLogger("kernel.turn_phases._compaction"),
        )
    assert "compaction envelope" in caplog.text
    # No content leakage: the log line carries only numbers and the mode id.
    assert "query" not in caplog.text.lower()
    assert "content" not in caplog.text.lower()

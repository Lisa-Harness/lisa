"""Tests for shared execution-policy runtime ledger behavior."""

from __future__ import annotations

from kernel.execution_ledger import ExecutionLedger
from kernel.execution_policy import CapabilityExecutionPolicy, ExecutionPolicy


def test_reserve_model_call_honors_cap() -> None:
    ledger = ExecutionLedger(ExecutionPolicy(max_model_calls=2), started_mono=100.0)

    assert ledger.reserve_model_call() is True
    assert ledger.reserve_model_call() is True
    assert ledger.reserve_model_call() is False
    assert ledger.model_calls_used == 2


def test_reserve_capability_call_honors_global_cap() -> None:
    ledger = ExecutionLedger(ExecutionPolicy(max_capability_calls=2), started_mono=0.0)

    assert ledger.reserve_capability_call("fetch_page") is True
    assert ledger.reserve_capability_call("web_search") is True
    assert ledger.reserve_capability_call("fetch_page") is False
    assert ledger.capability_calls_used == 2


def test_reserve_capability_call_honors_per_capability_cap() -> None:
    policy = ExecutionPolicy(
        capability_limits={
            "fetch_page": CapabilityExecutionPolicy(max_invocations_per_turn=2),
        }
    )
    ledger = ExecutionLedger(policy, started_mono=0.0)

    assert ledger.reserve_capability_call("fetch_page") is True
    assert ledger.reserve_capability_call("fetch_page") is True
    assert ledger.reserve_capability_call("fetch_page") is False
    assert ledger.capability_calls_used == 2
    assert ledger.capability_count("fetch_page") == 2
    assert ledger.exhausted_capabilities() == {"fetch_page": 2}


def test_token_and_wall_exhaustion_snapshot() -> None:
    policy = ExecutionPolicy(max_total_tokens=100, max_wall_ms=1500)
    ledger = ExecutionLedger(policy, started_mono=10.0)

    ledger.record_tokens(40)
    ledger.record_tokens(-1)
    ledger.record_tokens(60)

    assert ledger.token_limit_exhausted() is True
    assert ledger.wall_limit_exhausted(now_mono=11.6) is True

    snapshot = ledger.snapshot(now_mono=11.6)
    assert snapshot.model_calls_used == 0
    assert snapshot.capability_calls_used == 0
    assert snapshot.total_tokens_used == 100
    assert snapshot.elapsed_wall_ms in (1599, 1600)


def test_unbounded_policy_never_blocks_reservations() -> None:
    ledger = ExecutionLedger(ExecutionPolicy(), started_mono=0.0)

    for _ in range(20):
        assert ledger.reserve_model_call() is True
        assert ledger.reserve_capability_call("fetch_page") is True

    assert ledger.model_calls_used == 20
    assert ledger.capability_calls_used == 20
    assert ledger.token_limit_exhausted() is False
    assert ledger.wall_limit_exhausted(now_mono=9999.0) is False

"""Tests for kernel.execution_policy helper semantics."""

from __future__ import annotations

from kernel.execution_policy import (
    CapabilityExecutionPolicy,
    ExecutionPolicy,
    resolve_capability_invocation_limit,
    tighten_counter_limit,
    tighten_limit,
)


def test_tighten_limit_handles_optional_unbounded_values() -> None:
    assert tighten_limit(None, None) is None
    assert tighten_limit(10, None) == 10
    assert tighten_limit(None, 7) == 7
    assert tighten_limit(10, 7) == 7


def test_tighten_counter_limit_uses_zero_as_unlimited() -> None:
    assert tighten_counter_limit(0, None) == 0
    assert tighten_counter_limit(0, 5) == 5
    assert tighten_counter_limit(10, None) == 10
    assert tighten_counter_limit(10, 4) == 4


def test_resolve_capability_invocation_limit_no_policy_returns_authored() -> None:
    assert resolve_capability_invocation_limit(None, "web_search", 0) == 0
    assert resolve_capability_invocation_limit(None, "web_search", 3) == 3


def test_resolve_capability_invocation_limit_tightens_when_policy_present() -> None:
    policy = ExecutionPolicy(
        capability_limits={
            "web_search": CapabilityExecutionPolicy(max_invocations_per_turn=2),
        }
    )

    assert resolve_capability_invocation_limit(policy, "web_search", 0) == 2
    assert resolve_capability_invocation_limit(policy, "web_search", 5) == 2
    assert resolve_capability_invocation_limit(policy, "fetch_page", 5) == 5

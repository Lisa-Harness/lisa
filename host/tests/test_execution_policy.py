"""Tests for host execution policy defaults and resolution."""

from __future__ import annotations

from host.execution_policy import (
    default_interactive_execution_policy,
    resolve_execution_policy,
)
from kernel.execution_policy import CapabilityExecutionPolicy, ExecutionPolicy
from kernel.session_plan import SessionCapabilityExecutionRequest, SessionExecutionRequest


def test_default_interactive_execution_policy_matches_contract() -> None:
    policy = default_interactive_execution_policy()

    assert policy.max_model_calls == 500
    assert policy.max_capability_calls == 500
    assert policy.max_total_tokens == 1_000_000
    assert policy.max_wall_ms == 3_600_000
    assert policy.force_final_on_exhausted is True


def test_resolve_execution_policy_tightens_global_and_capability_limits() -> None:
    base = ExecutionPolicy(
        max_model_calls=20,
        max_capability_calls=30,
        max_total_tokens=50_000,
        max_wall_ms=300_000,
        capability_limits={
            "fetch_page": CapabilityExecutionPolicy(max_invocations_per_turn=10),
        },
    )
    request = SessionExecutionRequest(
        max_model_calls=12,
        max_capability_calls=25,
        max_total_tokens=40_000,
        max_wall_ms=250_000,
        capabilities={
            "fetch_page": SessionCapabilityExecutionRequest(max_invocations_per_turn=6),
            "web_search": SessionCapabilityExecutionRequest(max_internal_tool_calls=2),
        },
    )

    resolved = resolve_execution_policy(default_policy=base, session_request=request)

    assert resolved.max_model_calls == 12
    assert resolved.max_capability_calls == 25
    assert resolved.max_total_tokens == 40_000
    assert resolved.max_wall_ms == 250_000
    assert resolved.capability_limits["fetch_page"].max_invocations_per_turn == 6
    assert resolved.capability_limits["web_search"].max_internal_tool_calls == 2


def test_resolve_execution_policy_cannot_loosen_host_limits() -> None:
    base = ExecutionPolicy(max_model_calls=8, max_capability_calls=9)
    request = SessionExecutionRequest(max_model_calls=20, max_capability_calls=30)

    resolved = resolve_execution_policy(default_policy=base, session_request=request)

    assert resolved.max_model_calls == 8
    assert resolved.max_capability_calls == 9

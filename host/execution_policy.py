"""Host-side execution policy defaults and resolution helpers."""

from __future__ import annotations

from kernel.execution_policy import (
    CapabilityExecutionPolicy,
    ExecutionPolicy,
    tighten_limit,
)
from kernel.session_plan import SessionExecutionRequest


def default_interactive_execution_policy() -> ExecutionPolicy:
    """Return the default interactive host execution envelope."""

    return ExecutionPolicy(
        max_model_calls=500,
        max_capability_calls=500,
        max_total_tokens=1_000_000,
        max_wall_ms=3_600_000,
        force_final_on_exhausted=True,
    )


def resolve_execution_policy(
    *,
    default_policy: ExecutionPolicy,
    session_request: SessionExecutionRequest,
) -> ExecutionPolicy:
    """Resolve a tighten-only execution policy from host defaults + session.

    Session requests may only tighten limits. ``None`` means "no request".
    """

    capability_limits = dict(default_policy.capability_limits)
    for capability_id, requested in session_request.capabilities.items():
        base = capability_limits.get(capability_id, CapabilityExecutionPolicy())
        capability_limits[capability_id] = CapabilityExecutionPolicy(
            max_invocations_per_turn=tighten_limit(
                base.max_invocations_per_turn,
                requested.max_invocations_per_turn,
            ),
            max_internal_tool_calls=tighten_limit(
                base.max_internal_tool_calls,
                requested.max_internal_tool_calls,
            ),
        )

    force_final = (
        default_policy.force_final_on_exhausted
        if session_request.force_final_on_exhausted is None
        else session_request.force_final_on_exhausted
    )

    return ExecutionPolicy(
        max_model_calls=tighten_limit(
            default_policy.max_model_calls, session_request.max_model_calls
        ),
        max_capability_calls=tighten_limit(
            default_policy.max_capability_calls,
            session_request.max_capability_calls,
        ),
        max_total_tokens=tighten_limit(
            default_policy.max_total_tokens,
            session_request.max_total_tokens,
        ),
        max_completion_tokens=tighten_limit(
            default_policy.max_completion_tokens,
            session_request.max_completion_tokens,
        ),
        max_wall_ms=tighten_limit(default_policy.max_wall_ms, session_request.max_wall_ms),
        force_final_on_exhausted=force_final,
        capability_limits=capability_limits,
    )

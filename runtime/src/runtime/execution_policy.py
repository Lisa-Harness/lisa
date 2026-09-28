"""Host-side execution policy defaults and resolution helpers."""

from __future__ import annotations

from kernel.execution_policy import (
    CapabilityExecutionPolicy,
    ExecutionPolicy,
    tighten_limit,
)
from kernel.session_plan import SessionExecutionLimits, SessionExecutionRequest


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
    scope_ids: tuple[str, ...] = (),
) -> ExecutionPolicy:
    """Resolve a tighten-only execution policy from host defaults + session.

    Session requests may only tighten limits. ``None`` means "no request".
    """

    capability_limits = dict(default_policy.capability_limits)

    def apply_limits(
        policy: ExecutionPolicy,
        requested: SessionExecutionLimits,
    ) -> ExecutionPolicy:
        nonlocal capability_limits
        for capability_id, capability_request in requested.capabilities.items():
            base = capability_limits.get(capability_id, CapabilityExecutionPolicy())
            capability_limits[capability_id] = CapabilityExecutionPolicy(
                max_invocations_per_turn=tighten_limit(
                    base.max_invocations_per_turn,
                    capability_request.max_invocations_per_turn,
                ),
                max_internal_tool_calls=tighten_limit(
                    base.max_internal_tool_calls,
                    capability_request.max_internal_tool_calls,
                ),
            )
        force_final = (
            policy.force_final_on_exhausted
            if requested.force_final_on_exhausted is None
            else requested.force_final_on_exhausted
        )
        return ExecutionPolicy(
            max_model_calls=tighten_limit(policy.max_model_calls, requested.max_model_calls),
            max_capability_calls=tighten_limit(
                policy.max_capability_calls, requested.max_capability_calls
            ),
            max_total_tokens=tighten_limit(policy.max_total_tokens, requested.max_total_tokens),
            max_completion_tokens=tighten_limit(
                policy.max_completion_tokens, requested.max_completion_tokens
            ),
            max_wall_ms=tighten_limit(policy.max_wall_ms, requested.max_wall_ms),
            force_final_on_exhausted=force_final,
            capability_limits=capability_limits,
        )

    resolved = apply_limits(default_policy, session_request)
    for scope_id in scope_ids:
        requested = session_request.scopes.get(scope_id)
        if requested is not None:
            resolved = apply_limits(resolved, requested)
    return resolved

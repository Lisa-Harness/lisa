"""Execution policy contracts for host-owned turn budgeting.

This module defines immutable policy inputs and mutable accounting snapshots
that hosts can pass to the turn runner. The initial integration keeps legacy
budget enforcement in place while migrating call sites to these types.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ExecutionStopReason = Literal[
    "completed",
    "model_call_limit",
    "capability_call_limit",
    "token_limit",
    "wall_time_limit",
    "guard_blocked",
    "approval_required",
    "error",
    "cancelled",
    "unknown",
]


class _StrictExecutionModel(BaseModel):
    """Base model for execution-policy contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class CapabilityExecutionPolicy(_StrictExecutionModel):
    """Optional per-capability limits resolved by the host.

    ``None`` means unlimited.
    """

    max_invocations_per_turn: int | None = Field(default=None, ge=1)
    max_internal_tool_calls: int | None = Field(default=None, ge=1)


class ExecutionPolicy(_StrictExecutionModel):
    """Host-resolved execution envelope for one turn.

    ``None`` means unlimited for numeric limits.
    """

    max_model_calls: int | None = Field(default=None, ge=1)
    max_capability_calls: int | None = Field(default=None, ge=1)
    max_total_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    max_wall_ms: int | None = Field(default=None, ge=1)
    force_final_on_exhausted: bool = True
    capability_limits: dict[str, CapabilityExecutionPolicy] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_capability_keys(self) -> ExecutionPolicy:
        for capability_id in self.capability_limits:
            if not capability_id.strip():
                raise ValueError("capability_limits keys must be non-empty capability IDs")
        return self


class ExecutionSnapshot(_StrictExecutionModel):
    """Structured usage snapshot attached to turn results and telemetry."""

    model_calls_used: int = Field(default=0, ge=0)
    capability_calls_used: int = Field(default=0, ge=0)
    total_tokens_used: int = Field(default=0, ge=0)
    elapsed_wall_ms: int = Field(default=0, ge=0)
    exhausted_capabilities: dict[str, int] = Field(default_factory=dict)


def tighten_limit(base: int | None, requested: int | None) -> int | None:
    """Return the tighter of two optional limits.

    ``None`` means unlimited. If both are finite, the lower value wins.
    """

    if requested is None:
        return base
    if base is None:
        return requested
    return min(base, requested)


def tighten_counter_limit(base: int, requested: int | None) -> int:
    """Return the tighter counter-style limit where 0 means unlimited.

    ``base`` and the returned value use ``0`` as the unlimited sentinel.
    ``requested`` follows execution-policy semantics where ``None`` means
    "no request".
    """

    if requested is None:
        return base
    if base <= 0:
        return requested
    return min(base, requested)


def resolve_capability_invocation_limit(
    policy: ExecutionPolicy | None,
    capability_id: str,
    authored_limit: int,
) -> int:
    """Resolve the effective per-turn invocation cap for one capability.

    ``authored_limit`` uses ``0`` for unlimited (catalog semantics). Host
    policy may tighten that limit via ``capability_limits``.
    """

    if policy is None:
        return authored_limit
    capability_policy = policy.capability_limits.get(capability_id)
    if capability_policy is None:
        return authored_limit
    return tighten_counter_limit(authored_limit, capability_policy.max_invocations_per_turn)

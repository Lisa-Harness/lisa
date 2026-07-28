"""Shared runtime ledger for host-owned execution policy accounting.

The ledger is a small mutable state container intended to be shared by one
root turn and any fork children so they consume the same policy envelope.
This module is additive: it does not yet replace existing turn-runner flow.
"""

from __future__ import annotations

import time

from kernel.execution_policy import ExecutionPolicy, ExecutionSnapshot

__all__ = ["ExecutionLedger"]


class ExecutionLedger:
    """Track turn usage against one resolved :class:`ExecutionPolicy`."""

    def __init__(
        self,
        policy: ExecutionPolicy,
        *,
        started_mono: float | None = None,
    ) -> None:
        self.policy = policy
        self.started_mono = time.monotonic() if started_mono is None else started_mono

        self.model_calls_used = 0
        self.capability_calls_used = 0
        self.total_tokens_used = 0
        self._capability_counts: dict[str, int] = {}
        self._exhausted_capabilities: dict[str, int] = {}

    def reserve_model_call(self) -> bool:
        """Reserve one model call if policy allows it.

        Returns ``True`` when the call is admitted and increments usage.
        Returns ``False`` when the model-call limit is already exhausted.
        """

        cap = self.policy.max_model_calls
        if cap is not None and self.model_calls_used >= cap:
            return False
        self.model_calls_used += 1
        return True

    def reserve_capability_call(self, capability_id: str) -> bool:
        """Reserve one capability/tool call for ``capability_id``.

        Checks both the global capability-call limit and an optional
        per-capability invocation limit.
        """

        global_cap = self.policy.max_capability_calls
        if global_cap is not None and self.capability_calls_used >= global_cap:
            return False

        capability_policy = self.policy.capability_limits.get(capability_id)
        per_cap = (
            capability_policy.max_invocations_per_turn
            if capability_policy is not None
            else None
        )
        used = self._capability_counts.get(capability_id, 0)
        if per_cap is not None and used >= per_cap:
            self._exhausted_capabilities[capability_id] = per_cap
            return False

        self.capability_calls_used += 1
        self._capability_counts[capability_id] = used + 1
        return True

    def record_tokens(self, total_tokens: int) -> None:
        """Add provider-reported token usage for one completion.

        Non-positive values are ignored.
        """

        if total_tokens > 0:
            self.total_tokens_used += total_tokens

    def elapsed_wall_ms(self, *, now_mono: float | None = None) -> int:
        """Return elapsed wall time in milliseconds since ledger creation."""

        now = time.monotonic() if now_mono is None else now_mono
        return max(0, int((now - self.started_mono) * 1000))

    def token_limit_exhausted(self) -> bool:
        """Return ``True`` when the total-token cap is exhausted."""

        cap = self.policy.max_total_tokens
        return cap is not None and self.total_tokens_used >= cap

    def wall_limit_exhausted(self, *, now_mono: float | None = None) -> bool:
        """Return ``True`` when the wall-clock cap is exhausted."""

        cap = self.policy.max_wall_ms
        return cap is not None and self.elapsed_wall_ms(now_mono=now_mono) >= cap

    def exhausted_capabilities(self) -> dict[str, int]:
        """Return a copy of currently exhausted per-capability limits."""

        return dict(self._exhausted_capabilities)

    def capability_count(self, capability_id: str) -> int:
        """Return accepted-call count for one capability id."""

        return self._capability_counts.get(capability_id, 0)

    def snapshot(self, *, now_mono: float | None = None) -> ExecutionSnapshot:
        """Build a stable structured usage snapshot."""

        return ExecutionSnapshot(
            model_calls_used=self.model_calls_used,
            capability_calls_used=self.capability_calls_used,
            total_tokens_used=self.total_tokens_used,
            elapsed_wall_ms=self.elapsed_wall_ms(now_mono=now_mono),
            exhausted_capabilities=self.exhausted_capabilities(),
        )

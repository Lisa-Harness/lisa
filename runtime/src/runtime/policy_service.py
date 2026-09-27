"""Policy-decorating approval service.

:class:`PolicyApprovalService` wraps a durable :class:`FileApprovalService`
with the host policy engine (:mod:`runtime.policy`). It implements the very same
:class:`~kernel.approval.ApprovalService` protocol the kernel and execution
registries target, so it drops into the host at exactly one seam:

    # host/runtime.py
    self.approval_service = PolicyApprovalService(
        FileApprovalService(self.paths.state_dir / "approvals.json"),
        policy=load_policy(self.data_dir),
        data_dir=self.data_dir,
    )

Only ``request`` is intercepted — the method every approval gate calls before
side effects. Everything else (``approve``/``deny``/``consume``/``get``/
``list_pending``) delegates straight through, so the exact-action,
single-use, TTL-bound receipt contract on disk is preserved verbatim.

Decision mapping (from :func:`~runtime.policy.decide`):

- **ask** — today's path: persist the request as pending and let the caller
  raise ``ApprovalRequired`` so the operator sees the prompt.
- **allow** — persist the record, then self-approve so a *real receipt* is
  minted. The caller still raises ``ApprovalRequired`` (the registry flow is
  unchanged), but the record is now ``approved`` — the host's post-turn hook
  notices this and drives the approve-execute-continue loop itself, with a
  per-turn cap. Auto-approvals are therefore auditable, never a silent skip.
- **deny** — raise ``ToolExecutionError``. The registry surfaces it to the
  model as a tool error so it reroutes, instead of stalling the turn on a
  prompt the operator will only decline. Denial is loud, never silent.

The engine is re-read from disk on every ``request`` so an operator editing
``policy.yaml`` mid-session (or the host writing a rule via ``--remember``)
takes effect immediately without a process restart.
"""

from __future__ import annotations

import logging
from pathlib import Path

from kernel.approval import (
    ApprovalIdentity,
    ApprovalInvocation,
    ApprovalReceipt,
    ApprovalRecord,
)
from kernel.tools.registry import ToolExecutionError

from runtime.approvals import FileApprovalService
from runtime.policy import (
    VERDICT_ALLOW,
    VERDICT_DENY,
    PolicyDecision,
    PolicyDocument,
    decide,
    load_policy,
)

__all__ = ["PolicyApprovalService"]

logger = logging.getLogger(__name__)


class PolicyApprovalService:
    """An ``ApprovalService`` decorator that consults the host policy engine.

    Parameters
    ----------
    store:
        The durable ``FileApprovalService`` that owns on-disk state.
    policy:
        The policy document to evaluate. When ``None``, it is re-loaded from
        ``data_dir`` on every request (so operator edits and ``--remember``
        writes take effect without a restart).
    data_dir:
        The host data root. Used both to re-load the policy and to resolve
        the sandbox containment fact for the sandbox-boundary downgrade.
    """

    def __init__(
        self,
        store: FileApprovalService,
        *,
        policy: PolicyDocument | None = None,
        data_dir: Path,
    ) -> None:
        self._store = store
        self._static_policy = policy
        self._data_dir = Path(data_dir)

    # ── Policy resolution ────────────────────────────────────────────────

    def _policy(self) -> PolicyDocument:
        """Return the active policy, reloading from disk when not pinned."""
        if self._static_policy is not None:
            return self._static_policy
        return load_policy(self._data_dir)

    def decide(self, invocation: ApprovalInvocation) -> PolicyDecision:
        """Evaluate the policy for an invocation (exposed for tests/UIs)."""
        return decide(policy=self._policy(), invocation=invocation, data_dir=self._data_dir)

    # ── ApprovalService protocol ─────────────────────────────────────────

    async def request(self, invocation: ApprovalInvocation) -> ApprovalInvocation:
        """Record a pending invocation, pre-decided by policy.

        ``allow`` mints a receipt immediately (the record transitions to
        ``approved``); the caller still raises ``ApprovalRequired`` and the
        host's post-turn hook drives the continuation. ``deny`` raises a tool
        error so the model reroutes. ``ask`` is the unchanged operator path.
        """
        decision = self.decide(invocation)

        if decision.verdict == VERDICT_DENY:
            logger.info(
                "policy: denied %s (%s) — %s",
                invocation.target_id,
                invocation.request_id,
                decision.reason,
            )
            # Loud denial: a tool error the model sees and can reroute from,
            # not a prompt it will only decline. Never a silent skip.
            raise ToolExecutionError(f"Action denied by policy: {decision.reason}")

        # Persist the request exactly as the store does today (idempotent on
        # action_digest). For allow we then self-approve to mint the receipt.
        recorded = await self._store.request(invocation)

        if decision.verdict == VERDICT_ALLOW:
            receipt = await self._store.approve(recorded.request_id, recorded.identity)
            if receipt is not None:
                logger.info(
                    "policy: auto-approved %s (receipt %s) — %s",
                    invocation.target_id,
                    recorded.request_id[:8],
                    decision.reason,
                )

        return recorded

    # Thin passthroughs — the receipt, expiry, and persistence semantics are
    # entirely the store's; the engine only ever pre-decides ``request``.

    async def approve(self, request_id: str, identity: ApprovalIdentity) -> ApprovalReceipt | None:
        return await self._store.approve(request_id, identity)

    async def deny(
        self,
        request_id: str,
        identity: ApprovalIdentity,
        *,
        reason: str = "",
    ) -> bool:
        return await self._store.deny(request_id, identity, reason=reason)

    async def consume(
        self, request_id: str, invocation: ApprovalInvocation
    ) -> ApprovalReceipt | None:
        return await self._store.consume(request_id, invocation)

    async def get(self, request_id: str) -> ApprovalRecord | None:
        return await self._store.get(request_id)

    async def list_pending(self, identity: ApprovalIdentity) -> list[ApprovalRecord]:
        return await self._store.list_pending(identity)

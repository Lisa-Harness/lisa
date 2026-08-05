"""Kernel contract for durable workflow-instance storage.

The kernel defines only the seam: ``WorkflowInstanceStore`` is a Protocol
the host implements. The file-backed implementation lives in
``host/workflow_instance_store.py`` — same split as
``kernel/recovery.py`` (contracts) vs ``host/recovery.py``
(``FileRecoveryService``).

The kernel never touches the store directly — the host owns the instance
lifecycle (create on turn 1, advance on handoff, clear on completion,
load on boot). The Protocol mirrors ``RecoveryService`` so the two
stores compose the same way.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from kernel.recovery import RecoveryIdentity
from kernel.workflow_instance import WorkflowInstance

__all__ = ["WorkflowInstanceStore"]


@runtime_checkable
class WorkflowInstanceStore(Protocol):
    """Host-owned durable store for workflow instances.

    The kernel never touches the store directly — the host owns the
    instance lifecycle (create on turn 1, advance on handoff, clear on
    completion, load on boot). The Protocol mirrors
    ``RecoveryService`` so the two stores compose the same way.
    """

    async def save(
        self, instance: WorkflowInstance, identity: RecoveryIdentity | None = None
    ) -> WorkflowInstance:
        """Persist (create or update) one instance.

        ``identity`` scopes the instance to one user/surface so
        ``load_active`` returns the right workflow per identity. When
        ``None`` the instance is stored unscoped (kernel-only tests).
        """

    async def load(self, instance_id: str, identity: RecoveryIdentity) -> WorkflowInstance | None:
        """Return one instance by id when its identity matches exactly."""

    async def load_active(self, identity: RecoveryIdentity) -> WorkflowInstance | None:
        """Return the single non-complete instance for this identity, if any."""

    async def list_open(self, identity: RecoveryIdentity) -> list[WorkflowInstance]:
        """List non-complete instances for this identity, newest first."""

    async def clear(self, instance_id: str, identity: RecoveryIdentity) -> bool:
        """Remove one instance. Return True when an instance was removed."""

"""File-backed host implementation of the kernel workflow-instance store.

Mirrors ``hosts/core/recovery.py`` (``FileRecoveryService``): the same
JSON-file-with-identity-scope pattern, the same atomic-write discipline
(``os.replace`` of a temp file), and the same ``asyncio.Lock`` for
concurrent access. No new persistence infrastructure — one more file in
the host's ``data_dir``, alongside ``recovery-items.json`` and
``approvals.json``.

The store keys instances by ``instance_id`` (a uuid hex stable across
restarts) and scopes the *active* instance lookup by ``RecoveryIdentity``
(the same identity scope recovery records use). One user has at most one
active (non-complete) instance per surface; ``load_active`` returns it so
the host can resume mid-flight workflows on boot.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from kernel.recovery import RecoveryIdentity
from kernel.workflow_instance import WorkflowInstance, is_complete

__all__ = ["FileWorkflowInstanceStore"]


class FileWorkflowInstanceStore:
    """Atomically persist workflow instances beneath the host data directory.

    The on-disk shape is a JSON object keyed by ``instance_id``; each value
    is a JSON object carrying the serialized ``WorkflowInstance`` plus an
    ``_identity`` field (the ``RecoveryIdentity`` this instance belongs to).
    The identity is stored flat (not nested) so a reload can cache it in
    ``self._identities`` without a nested model round-trip, mirroring how
    recovery records carry their identity inline.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()
        # In-memory identity map for instances saved this session. The
        # on-disk record carries ``_identity`` flat (see ``_entry_for_instance``);
        # this map avoids re-reading it on every save when the host passes
        # the identity explicitly via ``save_with_identity``.
        self._identities: dict[str, RecoveryIdentity] = {}

    async def save(
        self, instance: WorkflowInstance, identity: RecoveryIdentity | None = None
    ) -> WorkflowInstance:
        async with self._lock:
            records = self._load_records()
            if identity is not None:
                self._identities[instance.instance_id] = identity
            records[instance.instance_id] = instance
            self._store_records(records)
            return instance

    async def load(self, instance_id: str, identity: RecoveryIdentity) -> WorkflowInstance | None:
        async with self._lock:
            instance = self._load_records().get(instance_id)
            if instance is None:
                return None
            entry_identity = self._identities.get(instance_id)
            if entry_identity is None or entry_identity != identity:
                return None
            return instance

    def load_active_sync(self, identity: RecoveryIdentity) -> WorkflowInstance | None:
        """Synchronous active-instance lookup for boot-load (no lock).

        ``HostRuntime.__init__`` runs before the host's event loop is running,
        so it cannot ``await`` the async ``load_active`` (the store's
        ``asyncio.Lock`` is bound to the host loop). This sync path does the
        same file read + identity match without the lock — safe because
        boot-load is single-threaded and no concurrent writes are possible
        before the loop starts.
        """
        records = self._load_records()
        active: list[WorkflowInstance] = []
        for instance_id, instance in records.items():
            entry_identity = self._identities.get(instance_id)
            if entry_identity is None or entry_identity != identity:
                continue
            if not is_complete(instance):
                active.append(instance)
        if not active:
            return None
        active.sort(key=lambda inst: inst.last_progressed_at, reverse=True)
        return active[0]

    def save_sync(
        self, instance: WorkflowInstance, identity: RecoveryIdentity | None = None
    ) -> None:
        """Synchronous save for host-side callers already inside the event loop.

        The kernel calls ``workflow_instance_updater`` from within the turn
        loop, so the host cannot ``await`` the async ``save`` (the store's
        ``asyncio.Lock`` is bound to the loop and would deadlock). This sync
        path does the same file write without the lock — safe because the
        turn loop is single-threaded within a turn.
        """
        records = self._load_records()
        if identity is not None:
            self._identities[instance.instance_id] = identity
        records[instance.instance_id] = instance
        self._store_records(records)

    def clear_sync(self, instance_id: str, identity: RecoveryIdentity) -> bool:
        """Synchronous clear for host-side callers inside the event loop."""
        records = self._load_records()
        instance = records.get(instance_id)
        if instance is None:
            return False
        entry_identity = self._identities.get(instance_id)
        if entry_identity is None or entry_identity != identity:
            return False
        del records[instance_id]
        self._identities.pop(instance_id, None)
        self._store_records(records)
        return True

    async def load_active(self, identity: RecoveryIdentity) -> WorkflowInstance | None:
        async with self._lock:
            records = self._load_records()
            active: list[WorkflowInstance] = []
            for instance_id, instance in records.items():
                entry_identity = self._identities.get(instance_id)
                if entry_identity is None or entry_identity != identity:
                    continue
                if not is_complete(instance):
                    active.append(instance)
            if not active:
                return None
            # Newest first: the most recently progressed active instance
            # wins when (unusually) more than one is open.
            active.sort(key=lambda inst: inst.last_progressed_at, reverse=True)
            return active[0]

    async def list_open(self, identity: RecoveryIdentity) -> list[WorkflowInstance]:
        async with self._lock:
            records = self._load_records()
            open_instances: list[WorkflowInstance] = []
            for instance_id, instance in records.items():
                entry_identity = self._identities.get(instance_id)
                if entry_identity is None or entry_identity != identity:
                    continue
                if not is_complete(instance):
                    open_instances.append(instance)
            open_instances.sort(key=lambda inst: inst.last_progressed_at, reverse=True)
            return open_instances

    async def clear(self, instance_id: str, identity: RecoveryIdentity) -> bool:
        async with self._lock:
            records = self._load_records()
            instance = records.get(instance_id)
            if instance is None:
                return False
            entry_identity = self._identities.get(instance_id)
            if entry_identity is None or entry_identity != identity:
                return False
            del records[instance_id]
            self._identities.pop(instance_id, None)
            self._store_records(records)
            return True

    def _load_records(self) -> dict[str, WorkflowInstance]:
        if not self._path.is_file():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(raw, dict):
            return {}
        records: dict[str, WorkflowInstance] = {}
        for instance_id, value in raw.items():
            if not isinstance(value, dict):
                continue
            instance = self._instance_from_entry(value)
            if instance is not None:
                records[str(instance_id)] = instance
        return records

    def _store_records(self, records: dict[str, WorkflowInstance]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, dict[str, object]] = {}
        for instance_id, instance in records.items():
            identity = self._identities.get(instance_id)
            payload[instance_id] = self._entry_for_instance(instance, identity)
        temp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temp_path, self._path)

    def _entry_for_instance(
        self,
        instance: WorkflowInstance,
        identity: RecoveryIdentity | None,
    ) -> dict[str, object]:
        """Serialize an instance with its identity stored flat for matching.

        The identity is stored alongside the instance fields (under
        ``_identity``) so a reload (``_instance_from_entry``) can cache it
        in ``self._identities`` without a nested model round-trip. The host
        passes the identity via ``save``; when absent (kernel-only tests),
        the entry has no identity and matches no scoped lookup.
        """
        data = instance.model_dump(mode="json")
        if identity is not None:
            data["_identity"] = {
                "workspace_id": identity.workspace_id,
                "user_id": identity.user_id,
                "user_slug": identity.user_slug,
                "surface_type": identity.surface_type,
                "surface_id": identity.surface_id,
                "memory_scope": identity.memory_scope,
                "session_plan_digest": identity.session_plan_digest,
            }
        return data

    def _instance_from_entry(self, entry: dict[str, object]) -> WorkflowInstance | None:
        """Rebuild an instance from a stored entry, caching its identity.

        The ``_identity`` field is popped from the entry before validation
        (``WorkflowInstance`` is ``extra="forbid"``) and cached in
        ``self._identities`` so a subsequent ``save`` round-trips it without
        the caller re-supplying it.
        """
        entry_copy = dict(entry)
        identity_dict = entry_copy.pop("_identity", None)
        try:
            instance = WorkflowInstance.model_validate(entry_copy)
        except Exception:  # pylint: disable=broad-except
            return None
        if isinstance(identity_dict, dict):
            self._identities[instance.instance_id] = _entry_identity(identity_dict)
        return instance


def _entry_identity(identity_dict: dict[str, object]) -> RecoveryIdentity:
    """Rebuild a ``RecoveryIdentity`` from a stored entry's flat identity."""
    return RecoveryIdentity(
        workspace_id=str(identity_dict.get("workspace_id", "")),
        user_id=str(identity_dict.get("user_id", "")),
        user_slug=str(identity_dict.get("user_slug", "")),
        surface_type=str(identity_dict.get("surface_type", "")),  # type: ignore[arg-type]
        surface_id=str(identity_dict.get("surface_id", "")),
        memory_scope=str(identity_dict.get("memory_scope", "")),  # type: ignore[arg-type]
        session_plan_digest=str(identity_dict.get("session_plan_digest", "")),
    )

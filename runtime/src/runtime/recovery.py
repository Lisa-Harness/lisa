"""File-backed host implementation of the kernel recovery service."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from kernel.recovery import RecoveryIdentity, RecoveryRecord, open_state_records
from sdk.filelock import (
    async_file_lock,
    atomic_write_text,
    is_process_alive,
    sidecar_lock_path,
)

__all__ = ["FileRecoveryService", "RecoveryStoreCorruptError"]

logger = logging.getLogger(__name__)


class RecoveryStoreCorruptError(RuntimeError):
    """The recovery store exists but cannot be read.

    Raised instead of treating corruption as an empty store: a later
    ``settle``/``begin`` would write the misread subset back and erase
    open recovery records. The original file is never rewritten by this
    path — an operator must repair or remove the file explicitly.
    """


class FileRecoveryService:
    """Atomically persist recovery records beneath the host data directory."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()
        self._lock_path = sidecar_lock_path(path)

    async def begin(self, record: RecoveryRecord) -> RecoveryRecord:
        async with self._lock, async_file_lock(self._lock_path):
            records = self._load_records()
            now = datetime.now(UTC)
            updates: dict[str, object] = {"state": "running", "updated_at": now}
            if record.owner_pid is None:
                updates["owner_pid"] = os.getpid()
            running = record.model_copy(update=updates)
            records[running.work_item_id] = running
            self._store_records(records)
            return running

    async def settle(
        self,
        work_item_id: str,
        identity: RecoveryIdentity,
        *,
        state: str,
        last_error: str = "",
        result_preview: str = "",
        retry_policy: str | None = None,
    ) -> RecoveryRecord | None:
        async with self._lock, async_file_lock(self._lock_path):
            records = self._load_records()
            record = records.get(work_item_id)
            if record is None or record.identity != identity:
                return None
            now = datetime.now(UTC)
            update: dict[str, object] = {
                "state": state,
                "updated_at": now,
                "last_error": last_error.strip(),
                "result_preview": result_preview.strip(),
                "settled_at": now,
            }
            if retry_policy is not None:
                update["retry_policy"] = retry_policy
            updated = record.model_copy(update=update)
            records[work_item_id] = updated
            self._store_records(records)
            return updated

    async def get(self, work_item_id: str, identity: RecoveryIdentity) -> RecoveryRecord | None:
        async with self._lock, async_file_lock(self._lock_path):
            record = self._load_records().get(work_item_id)
            if record is None or record.identity != identity:
                return None
            return record

    async def list_open(self, identity: RecoveryIdentity) -> list[RecoveryRecord]:
        async with self._lock, async_file_lock(self._lock_path):
            records = self._load_records()
            return sorted(
                open_state_records(records, identity),
                key=lambda item: item.updated_at,
                reverse=True,
            )

    async def mark_running_indeterminate(self, identity: RecoveryIdentity) -> int:
        async with self._lock, async_file_lock(self._lock_path):
            records = self._load_records()
            now = datetime.now(UTC)
            current_pid = os.getpid()
            changed = 0
            for work_item_id, record in list(records.items()):
                if record.identity != identity or record.state != "running":
                    continue
                if (
                    record.owner_pid is not None
                    and record.owner_pid != current_pid
                    and is_process_alive(record.owner_pid)
                ):
                    continue
                records[work_item_id] = record.model_copy(
                    update={
                        "state": "indeterminate",
                        "updated_at": now,
                        "settled_at": now,
                        "last_error": (
                            record.last_error.strip()
                            or "Recovered after host restart while item was running."
                        ),
                        "retry_policy": "confirm",
                    }
                )
                changed += 1
            if changed:
                self._store_records(records)
            return changed

    def _load_records(self) -> dict[str, RecoveryRecord]:
        """Load the recovery store, failing closed on corruption.

        File-absent is the clean-slate case and returns ``{}``. A file
        that exists but is unreadable, is not a JSON object, or contains
        an unparsable record raises :class:`RecoveryStoreCorruptError` —
        proceeding on a partial view would let the next ``_store_records``
        persist that subset and erase the dropped records (the 2026-11-08
        review, finding 3). The original bytes stay untouched.
        """
        if not self._path.is_file():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RecoveryStoreCorruptError(
                f"recovery store {self._path} is unreadable: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise RecoveryStoreCorruptError(f"recovery store {self._path} is not a JSON object")
        records: dict[str, RecoveryRecord] = {}
        for work_item_id, value in raw.items():
            try:
                records[str(work_item_id)] = RecoveryRecord.model_validate(value)
            except Exception as exc:  # pylint: disable=broad-except
                raise RecoveryStoreCorruptError(
                    f"recovery store {self._path} has an invalid record {work_item_id!r}: {exc}"
                ) from exc
        return records

    def _store_records(self, records: dict[str, RecoveryRecord]) -> None:
        payload = {
            work_item_id: record.model_dump(mode="json") for work_item_id, record in records.items()
        }
        atomic_write_text(
            self._path,
            json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
        )

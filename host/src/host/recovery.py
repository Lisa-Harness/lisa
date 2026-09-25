"""File-backed host implementation of the kernel recovery service."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from kernel.recovery import RecoveryIdentity, RecoveryRecord, open_state_records

__all__ = ["FileRecoveryService"]


class FileRecoveryService:
    """Atomically persist recovery records beneath the host data directory."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()

    async def begin(self, record: RecoveryRecord) -> RecoveryRecord:
        async with self._lock:
            records = self._load_records()
            now = datetime.now(UTC)
            running = record.model_copy(update={"state": "running", "updated_at": now})
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
        async with self._lock:
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
        async with self._lock:
            record = self._load_records().get(work_item_id)
            if record is None or record.identity != identity:
                return None
            return record

    async def list_open(self, identity: RecoveryIdentity) -> list[RecoveryRecord]:
        async with self._lock:
            records = self._load_records()
            return sorted(
                open_state_records(records, identity),
                key=lambda item: item.updated_at,
                reverse=True,
            )

    async def mark_running_indeterminate(self, identity: RecoveryIdentity) -> int:
        async with self._lock:
            records = self._load_records()
            now = datetime.now(UTC)
            changed = 0
            for work_item_id, record in list(records.items()):
                if record.identity != identity or record.state != "running":
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
        if not self._path.is_file():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(raw, dict):
            return {}
        records: dict[str, RecoveryRecord] = {}
        for work_item_id, value in raw.items():
            try:
                records[str(work_item_id)] = RecoveryRecord.model_validate(value)
            except Exception:  # pylint: disable=broad-except
                continue
        return records

    def _store_records(self, records: dict[str, RecoveryRecord]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            work_item_id: record.model_dump(mode="json") for work_item_id, record in records.items()
        }
        temp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temp_path, self._path)

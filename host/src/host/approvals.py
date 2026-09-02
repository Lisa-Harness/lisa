"""File-backed host implementation of the kernel approval service."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from kernel.approval import (
    ApprovalDecision,
    ApprovalIdentity,
    ApprovalInvocation,
    ApprovalReceipt,
    ApprovalRecord,
    _find_pending_for_action,
    is_expired,
)

__all__ = ["FileApprovalService"]


class FileApprovalService:
    """Atomically persist approval state beneath the host's data directory."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()

    async def request(self, invocation: ApprovalInvocation) -> ApprovalInvocation:
        async with self._lock:
            records = self._load_records()
            existing = _find_pending_for_action(records, invocation, now=datetime.now(UTC))
            if existing is not None:
                return existing.invocation
            records.setdefault(invocation.request_id, ApprovalRecord(invocation=invocation))
            self._store_records(records)
            return records[invocation.request_id].invocation

    async def approve(
        self,
        request_id: str,
        identity: ApprovalIdentity,
    ) -> ApprovalReceipt | None:
        async with self._lock:
            records = self._load_records()
            record = records.get(request_id)
            if record is None or record.invocation.identity != identity:
                return None
            if is_expired(record.invocation):
                record.state = "expired"
                self._store_records(records)
                return None
            if record.state != "pending":
                return None

            receipt = ApprovalReceipt(
                request_id=request_id,
                action_digest=record.invocation.action_digest,
                identity=identity,
                approved_at=datetime.now(UTC),
                expires_at=record.invocation.expires_at,
            )
            record.state = "approved"
            record.receipt = receipt
            record.decision = ApprovalDecision(
                request_id=request_id,
                state="approved",
                decided_at=receipt.approved_at,
                identity=identity,
            )
            self._store_records(records)
            return receipt

    async def deny(
        self,
        request_id: str,
        identity: ApprovalIdentity,
        *,
        reason: str = "",
    ) -> bool:
        async with self._lock:
            records = self._load_records()
            record = records.get(request_id)
            if (
                record is None
                or record.invocation.identity != identity
                or record.state != "pending"
            ):
                return False

            record.state = "denied"
            record.decision = ApprovalDecision(
                request_id=request_id,
                state="denied",
                decided_at=datetime.now(UTC),
                identity=identity,
                reason=reason.strip(),
            )
            self._store_records(records)
            return True

    async def consume(
        self,
        request_id: str,
        invocation: ApprovalInvocation,
    ) -> ApprovalReceipt | None:
        async with self._lock:
            records = self._load_records()
            record = records.get(request_id)
            if record is None:
                return None
            if is_expired(record.invocation):
                record.state = "expired"
                self._store_records(records)
                return None
            if (
                record.state != "approved"
                or record.receipt is None
                or record.receipt.action_digest != invocation.action_digest
                or record.invocation.identity != invocation.identity
            ):
                return None

            receipt = record.receipt
            record.state = "consumed"
            record.decision = ApprovalDecision(
                request_id=request_id,
                state="consumed",
                decided_at=datetime.now(UTC),
                identity=invocation.identity,
            )
            self._store_records(records)
            return receipt

    async def get(self, request_id: str) -> ApprovalRecord | None:
        async with self._lock:
            return self._load_records().get(request_id)

    async def list_pending(self, identity: ApprovalIdentity) -> list[ApprovalRecord]:
        async with self._lock:
            now = datetime.now(UTC)
            return [
                record
                for record in self._load_records().values()
                if record.state == "pending"
                and not is_expired(record.invocation, now=now)
                and record.invocation.identity == identity
            ]

    def _load_records(self) -> dict[str, ApprovalRecord]:
        if not self._path.is_file():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(raw, dict):
            return {}
        records: dict[str, ApprovalRecord] = {}
        for request_id, value in raw.items():
            try:
                records[str(request_id)] = ApprovalRecord.model_validate(value)
            except Exception:  # pylint: disable=broad-except
                continue
        return records

    def _store_records(self, records: dict[str, ApprovalRecord]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            request_id: record.model_dump(mode="json") for request_id, record in records.items()
        }
        temp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temp_path, self._path)

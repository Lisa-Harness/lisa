"""File-backed host implementation of the kernel approval service."""

from __future__ import annotations

import asyncio
import json
import logging
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
from sdk.filelock import async_file_lock, atomic_write_text, sidecar_lock_path

from runtime.approval_ledger import ApprovalLedger

__all__ = ["FileApprovalService"]

logger = logging.getLogger(__name__)


class FileApprovalService:
    """Atomically persist approval state beneath the host's data directory.

    ``approvals.json`` (the *working store*) holds each request's latest
    state for recovery and receipt validation. When a ledger path is
    supplied, every state transition also appends one row to the
    append-only approval ledger — the evidence stream that answers "who
    authorized what, when" — inside the same critical section that
    committed the working store. The working store stays authoritative: a
    ledger append failure logs a WARNING and never blocks the transition.
    """

    def __init__(self, path: Path, *, ledger_path: Path | None = None) -> None:
        self._path = path
        self._lock = asyncio.Lock()
        self._lock_path = sidecar_lock_path(path)
        self._ledger = ApprovalLedger(ledger_path) if ledger_path is not None else None

    async def request(self, invocation: ApprovalInvocation) -> ApprovalInvocation:
        async with self._lock, async_file_lock(self._lock_path):
            records = self._load_records()
            existing = _find_pending_for_action(records, invocation, now=datetime.now(UTC))
            if existing is not None:
                return existing.invocation
            records.setdefault(invocation.request_id, ApprovalRecord(invocation=invocation))
            self._store_records(records)
            # Only a genuinely new request is evidence: the dedup path
            # returns the pending record the operator already saw.
            if self._ledger is not None:
                self._ledger.record_transition(invocation=invocation, state="requested")
            return records[invocation.request_id].invocation

    async def approve(
        self,
        request_id: str,
        identity: ApprovalIdentity,
        *,
        decision_source: str = "operator",
    ) -> ApprovalReceipt | None:
        async with self._lock, async_file_lock(self._lock_path):
            records = self._load_records()
            record = records.get(request_id)
            if record is None or record.invocation.identity != identity:
                return None
            if is_expired(record.invocation):
                record.state = "expired"
                self._store_records(records)
                if self._ledger is not None:
                    self._ledger.record_transition(
                        invocation=record.invocation,
                        state="expired",
                        decision_source="expiry",
                    )
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
            if self._ledger is not None:
                self._ledger.record_transition(
                    invocation=record.invocation,
                    state="approved",
                    decision_by=identity.user_slug,
                    decision_source=decision_source,
                )
            return receipt

    async def deny(
        self,
        request_id: str,
        identity: ApprovalIdentity,
        *,
        reason: str = "",
    ) -> bool:
        async with self._lock, async_file_lock(self._lock_path):
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
            if self._ledger is not None:
                self._ledger.record_transition(
                    invocation=record.invocation,
                    state="denied",
                    decision_by=identity.user_slug,
                    decision_source="operator",
                    reason=reason.strip(),
                )
            return True

    async def consume(
        self,
        request_id: str,
        invocation: ApprovalInvocation,
    ) -> ApprovalReceipt | None:
        async with self._lock, async_file_lock(self._lock_path):
            records = self._load_records()
            record = records.get(request_id)
            if record is None:
                return None
            if is_expired(record.invocation):
                record.state = "expired"
                self._store_records(records)
                if self._ledger is not None:
                    self._ledger.record_transition(
                        invocation=record.invocation,
                        state="expired",
                        decision_source="expiry",
                    )
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
            if self._ledger is not None:
                self._ledger.record_transition(
                    invocation=record.invocation,
                    state="consumed",
                    decision_by=invocation.identity.user_slug,
                    decision_source="operator",
                )
            return receipt

    async def get(self, request_id: str) -> ApprovalRecord | None:
        async with self._lock, async_file_lock(self._lock_path):
            return self._load_records().get(request_id)

    async def list_pending(self, identity: ApprovalIdentity) -> list[ApprovalRecord]:
        async with self._lock, async_file_lock(self._lock_path):
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
            logger.warning("approvals: failed to load %s", self._path, exc_info=True)
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
        payload = {
            request_id: record.model_dump(mode="json") for request_id, record in records.items()
        }
        atomic_write_text(
            self._path,
            json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
        )

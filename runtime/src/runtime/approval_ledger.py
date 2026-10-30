"""Append-only approval decision ledger.

``approvals.json`` is the *working store*: the authoritative, mutable
state a restart recovers pending decisions from, replaced atomically on
every transition. This module owns the *evidence*: one JSONL row per
approval state transition (requested, approved, denied, consumed,
expired), appended in the same critical section that mutated the working
store, sealed-never-deleted like the decisions and token-usage ledgers.

Why both: the working store answers "can this action proceed right now?"
and holds only each request's *latest* state. The ledger answers "show me
who authorized what, and when" — every transition, in order, with the
operator identity and the turn attribution — without which an enterprise
audit has the outcome but not the authorizing decisions.

A ledger append failure never blocks the state transition it describes:
the working store remains authoritative, and the loss is visible as a
WARNING (mirroring the telemetry loss-observability contract; rows also
carry per-instance ``sequence`` and ``ledger_instance`` so a forensic read
can count dropped rows).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from kernel.approval import ApprovalInvocation
from sdk.filelock import file_lock, sidecar_lock_path

__all__ = ["ApprovalLedger"]

logger = logging.getLogger(__name__)


class ApprovalLedger:
    """Append one JSON row per approval state transition to a JSONL ledger.

    Rows are flat, self-describing, and carry the join keys:

    - ``request_id`` / ``action_digest`` / ``target_id`` / ``effect``:
      the canonical action (same identity fields the registries stamped).
    - ``turn_index``: the thread turn the request was issued under
      (``None`` = off-turn; never collapsed to ``0``).
    - ``user_slug`` / ``thread_store_slug``: scoping for shared-ledger
      reads, matching the decision and token ledgers.
    - ``state``: the transition this row records.
    - ``decision_by`` / ``decision_source``: who decided (operator
      identity's ``user_slug``) and what mechanism (``operator`` for an
      explicit ``/approve``-``/deny``, ``policy`` for an auto-allow, and
      ``expiry`` for a timeout transition).
    - ``sequence`` / ``ledger_instance``: per-writer loss detection.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock_path = sidecar_lock_path(path)
        self._sequence = 0
        self._instance = uuid.uuid4().hex[:12]

    @property
    def path(self) -> Path:
        """Return the resolved ledger path."""
        return self._path

    @property
    def instance(self) -> str:
        """Return this writer's instance id (stamped on every row)."""
        return self._instance

    def record_transition(
        self,
        *,
        invocation: ApprovalInvocation,
        state: str,
        decision_by: str = "",
        decision_source: str = "",
        reason: str = "",
    ) -> None:
        """Append one transition row for ``invocation`` (best-effort).

        ``state`` is the *new* state after the working store's transition.
        A failed append logs a WARNING and never raises: the working store
        already committed the transition, and the ledger is the secondary
        evidence stream.
        """
        self._sequence += 1
        row: dict[str, Any] = {
            "logged_at": time.time(),
            "state": state,
            "request_id": invocation.request_id,
            "action_digest": invocation.action_digest,
            "target_kind": invocation.target_kind,
            "target_id": invocation.target_id,
            "pack_id": invocation.pack_id,
            "effect": invocation.effect,
            "turn_index": invocation.turn_index,
            "user_slug": invocation.identity.user_slug,
            "thread_store_slug": invocation.thread_store_slug,
            "sequence": self._sequence,
            "ledger_instance": self._instance,
        }
        if decision_by:
            row["decision_by"] = decision_by
        if decision_source:
            row["decision_source"] = decision_source
        if reason:
            row["reason"] = reason
        try:
            with file_lock(self._lock_path):
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        except (OSError, TimeoutError) as exc:
            logger.warning(
                "approval ledger append failed for %s (%s): %s",
                invocation.request_id,
                state,
                exc,
            )
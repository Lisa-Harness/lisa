"""Durable recovery contracts for interrupted work items.

The kernel records dispatch lifecycle events for effectful and read-only
work units so hosts can surface deterministic retry controls after failures
or process restarts.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from kernel.approval import EffectKind
from kernel.models import ConversationSurface, MemoryScope

__all__ = [
    "RecoveryFailureScope",
    "RecoveryIdentity",
    "RecoveryRecord",
    "RecoveryRetryPolicy",
    "RecoveryService",
    "RecoveryState",
    "RecoveryTargetKind",
]

RecoveryTargetKind = Literal["operation", "pack_tool", "mcp_tool", "transition"]
RecoveryState = Literal[
    "running",
    "succeeded",
    "stalled",
    "indeterminate",
    "skipped",
    "cancelled",
]
RecoveryRetryPolicy = Literal["safe", "confirm", "never"]
RecoveryFailureScope = Literal["item", "workflow"]
_OPEN_STATES: tuple[RecoveryState, ...] = ("running", "stalled", "indeterminate")


class _FrozenRecoveryModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RecoveryIdentity(_FrozenRecoveryModel):
    """Exact ownership scope for recovery records."""

    workspace_id: str
    user_id: str
    user_slug: str
    surface_type: ConversationSurface
    surface_id: str
    memory_scope: MemoryScope
    session_plan_digest: str = ""


class RecoveryRecord(_FrozenRecoveryModel):
    """Durable state for one dispatch work item."""

    work_item_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    identity: RecoveryIdentity
    conversation_id: str = ""
    mode_id: str = ""
    stage_id: str = ""
    target_kind: RecoveryTargetKind
    target_id: str
    effect: EffectKind = "read"
    parameters: dict[str, Any] = Field(default_factory=dict)
    state: RecoveryState = "running"
    failure_scope: RecoveryFailureScope = "item"
    retry_policy: RecoveryRetryPolicy = "safe"
    blocks_workflow: bool = False
    attempt: int = 1
    last_error: str = ""
    result_preview: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    settled_at: datetime | None = None

    def open_for_operator(self) -> bool:
        return self.state in _OPEN_STATES


@runtime_checkable
class RecoveryService(Protocol):
    """Host-owned durable service for interrupted dispatch work."""

    async def begin(self, record: RecoveryRecord) -> RecoveryRecord:
        """Persist a work item in ``running`` state before dispatch."""

    async def settle(
        self,
        work_item_id: str,
        identity: RecoveryIdentity,
        *,
        state: RecoveryState,
        last_error: str = "",
        result_preview: str = "",
        failure_scope: RecoveryFailureScope | None = None,
        retry_policy: RecoveryRetryPolicy | None = None,
        blocks_workflow: bool | None = None,
    ) -> RecoveryRecord | None:
        """Transition one record from open to terminal or stalled state."""

    async def get(self, work_item_id: str, identity: RecoveryIdentity) -> RecoveryRecord | None:
        """Return one record by id when identity matches exactly."""

    async def list_open(self, identity: RecoveryIdentity) -> list[RecoveryRecord]:
        """List unresolved records for this identity."""

    async def mark_running_indeterminate(self, identity: RecoveryIdentity) -> int:
        """Convert stale ``running`` records to ``indeterminate`` on startup."""


def open_state_records(
    records: Mapping[str, RecoveryRecord],
    identity: RecoveryIdentity,
) -> list[RecoveryRecord]:
    """Return open records for one exact identity."""

    return [
        record
        for record in records.values()
        if record.identity == identity and record.open_for_operator()
    ]

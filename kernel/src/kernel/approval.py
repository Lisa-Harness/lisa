"""Exact-action approval contracts for catalog-backed effects.

The kernel owns digest construction and receipt validation. Hosts own durable
storage and approval UX through the :class:`ApprovalService` protocol.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ApprovalDecision",
    "ApprovalIdentity",
    "ApprovalInvocation",
    "ApprovalReceipt",
    "ApprovalRecord",
    "ApprovalRequired",
    "ApprovalService",
    "ApprovalState",
    "EffectKind",
    "InMemoryApprovalService",
    "build_approval_invocation",
    "canonical_json",
    "is_expired",
    "render_approval_request",
]

EffectKind = Literal["read", "write", "execute", "external_write"]
ApprovalState = Literal["pending", "approved", "denied", "consumed", "expired"]

_DEFAULT_APPROVAL_TTL = timedelta(minutes=30)


class _FrozenApprovalModel(BaseModel):
    """Base contract for immutable approval values."""

    model_config = ConfigDict(frozen=True)


class ApprovalIdentity(_FrozenApprovalModel):
    """Identity and session boundaries that scope an approval receipt."""

    workspace_id: str = "default"
    user_id: str = ""
    user_slug: str = "default"
    surface_type: str = "personal_dm"
    surface_id: str = ""
    memory_scope: str = "personal"
    mode_id: str = ""
    session_plan_digest: str = ""


class ApprovalInvocation(_FrozenApprovalModel):
    """A canonical effectful invocation that can be approved exactly once."""

    request_id: str
    target_kind: Literal["tool", "operation"]
    target_id: str
    pack_id: str = ""
    effect: EffectKind
    parameters: dict[str, Any] = Field(default_factory=dict)
    identity: ApprovalIdentity
    pack_config_fingerprint: str
    session_plan_digest: str
    action_digest: str
    safe_preview: str
    created_at: datetime
    expires_at: datetime


class ApprovalReceipt(_FrozenApprovalModel):
    """Evidence that one exact pending invocation was approved."""

    request_id: str
    action_digest: str
    identity: ApprovalIdentity
    approved_at: datetime
    expires_at: datetime


class ApprovalDecision(_FrozenApprovalModel):
    """An operator decision recorded against an approval request."""

    request_id: str
    state: ApprovalState
    decided_at: datetime
    identity: ApprovalIdentity
    reason: str = ""


class ApprovalRecord(BaseModel):
    """Durable state for one requested action, owned by a host service."""

    invocation: ApprovalInvocation
    state: ApprovalState = "pending"
    receipt: ApprovalReceipt | None = None
    decision: ApprovalDecision | None = None


class ApprovalUnavailable(RuntimeError):
    """The receipt failed authoritative consumption; do not continue its saved batch."""


class ApprovalRequired(RuntimeError):
    """Interrupt execution until the host has approved the exact action."""

    def __init__(self, invocation: ApprovalInvocation) -> None:
        self.invocation = invocation
        super().__init__(f"Approval required: {invocation.request_id}")


@runtime_checkable
class ApprovalService(Protocol):
    """Host-owned durable approval service used by execution registries."""

    async def request(self, invocation: ApprovalInvocation) -> ApprovalInvocation:
        """Persist a new pending invocation and return it."""

    async def approve(
        self,
        request_id: str,
        identity: ApprovalIdentity,
    ) -> ApprovalReceipt | None:
        """Approve a matching pending invocation, or return ``None``."""

    async def deny(
        self,
        request_id: str,
        identity: ApprovalIdentity,
        *,
        reason: str = "",
    ) -> bool:
        """Deny a matching pending invocation."""

    async def consume(
        self,
        request_id: str,
        invocation: ApprovalInvocation,
    ) -> ApprovalReceipt | None:
        """Atomically consume a matching approved invocation once."""

    async def get(self, request_id: str) -> ApprovalRecord | None:
        """Return the persisted request state without changing it."""

    async def list_pending(self, identity: ApprovalIdentity) -> list[ApprovalRecord]:
        """Return non-expired pending records owned by ``identity``.

        Hosts use this to detect affirmative-resume conditions (e.g. the
        user typing "proceed" while a paused action awaits approval) and
        to surface pending approvals in UIs. The identity match is exact.
        """


def _find_pending_for_action(
    records: Mapping[str, ApprovalRecord],
    invocation: ApprovalInvocation,
    *,
    now: datetime,
) -> ApprovalRecord | None:
    """Return a non-expired pending record matching action_digest + identity.

    The exact-action contract approves one canonical invocation once. When
    a caller re-issues the same effectful action (same target, parameters,
    pack config fingerprint, and owner identity) before the user has
    decided, the service must not mint a fresh ``request_id`` — that would
    strand the original pending request and force the user to re-approve a
    new id on every retry. Returning the existing pending record preserves
    the id the user already saw so a prior or future ``/approve <id>`` still
    satisfies it. Only the ``pending`` state is idempotent; once approved or
    consumed, a genuinely new request mints a new id.
    """

    for record in records.values():
        if record.state != "pending":
            continue
        if is_expired(record.invocation, now=now):
            continue
        if record.invocation.action_digest == invocation.action_digest and _identity_matches(
            record.invocation.identity, invocation.identity
        ):
            return record
    return None


def _canonicalize(value: Any) -> Any:
    """Normalize supported JSON-like values into stable, safe digest input."""

    if isinstance(value, BaseModel):
        return _canonicalize(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(item) for key, item in sorted(value.items())}
    if isinstance(value, list | tuple):
        return [_canonicalize(item) for item in value]
    if isinstance(value, set | frozenset):
        normalized = [_canonicalize(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
    return _canonicalize_scalar(value)


def _canonicalize_scalar(value: Any) -> Any:
    """Normalize a scalar leaf (datetime / None / str / int / float / bool)."""
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def canonical_json(value: Any) -> str:
    """Return a deterministic JSON representation suitable for SHA-256 input."""

    return json.dumps(
        _canonicalize(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def build_approval_invocation(
    *,
    target_kind: Literal["tool", "operation"],
    target_id: str,
    pack_id: str,
    effect: EffectKind,
    parameters: Mapping[str, Any],
    identity: ApprovalIdentity,
    pack_config: Mapping[str, Any],
    safe_preview: str = "",
    now: datetime | None = None,
    ttl: timedelta = _DEFAULT_APPROVAL_TTL,
) -> ApprovalInvocation:
    """Build an exact-action request from canonical validated inputs."""

    created_at = now or datetime.now(UTC)
    normalized_parameters = _canonicalize(parameters)
    if not isinstance(normalized_parameters, dict):
        normalized_parameters = {}
    config_fingerprint = _digest(pack_config)
    action_payload = {
        "target_kind": target_kind,
        "target_id": target_id,
        "pack_id": pack_id,
        "effect": effect,
        "parameters": normalized_parameters,
        "identity": identity,
        "pack_config_fingerprint": config_fingerprint,
        "session_plan_digest": identity.session_plan_digest,
    }
    return ApprovalInvocation(
        request_id=uuid.uuid4().hex,
        target_kind=target_kind,
        target_id=target_id,
        pack_id=pack_id,
        effect=effect,
        parameters=normalized_parameters,
        identity=identity,
        pack_config_fingerprint=config_fingerprint,
        session_plan_digest=identity.session_plan_digest,
        action_digest=_digest(action_payload),
        safe_preview=safe_preview.strip() or f"{effect.replace('_', ' ')} {target_id}",
        created_at=created_at,
        expires_at=created_at + ttl,
    )


def is_expired(invocation: ApprovalInvocation, *, now: datetime | None = None) -> bool:
    """Return whether an approval request has passed its expiry time."""

    return (now or datetime.now(UTC)) >= invocation.expires_at


def render_approval_request(invocation: ApprovalInvocation) -> str:
    """Render the host-neutral user prompt for a paused action.

    The kernel states the situation without command syntax: which action is
    paused and that an operator decision is pending. Hosts own the *how* —
    the runtime builds the structured ``data["approval"]`` payload (with
    ``request_id``) for panel transports and appends the installed command
    hints to the text for plain transports.
    """

    return (
        f"Approval required for {invocation.safe_preview}.\n\n"
        f"This action is paused until you approve or deny it "
        f"(request {invocation.request_id})."
    )


def _identity_matches(expected: ApprovalIdentity, actual: ApprovalIdentity) -> bool:
    return expected == actual


class InMemoryApprovalService:
    """Lock-protected approval service used by kernel tests and lightweight hosts."""

    def __init__(self) -> None:
        self._records: dict[str, ApprovalRecord] = {}
        self._lock = asyncio.Lock()

    async def request(self, invocation: ApprovalInvocation) -> ApprovalInvocation:
        async with self._lock:
            existing = _find_pending_for_action(self._records, invocation, now=datetime.now(UTC))
            if existing is not None:
                return existing.invocation
            self._records.setdefault(invocation.request_id, ApprovalRecord(invocation=invocation))
            return self._records[invocation.request_id].invocation

    async def approve(
        self,
        request_id: str,
        identity: ApprovalIdentity,
    ) -> ApprovalReceipt | None:
        async with self._lock:
            record = self._records.get(request_id)
            if record is None or not _identity_matches(record.invocation.identity, identity):
                return None
            if is_expired(record.invocation):
                record.state = "expired"
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
            return receipt

    async def deny(
        self,
        request_id: str,
        identity: ApprovalIdentity,
        *,
        reason: str = "",
    ) -> bool:
        async with self._lock:
            record = self._records.get(request_id)
            if record is None or not _identity_matches(record.invocation.identity, identity):
                return False
            if record.state != "pending":
                return False
            record.state = "denied"
            record.decision = ApprovalDecision(
                request_id=request_id,
                state="denied",
                decided_at=datetime.now(UTC),
                identity=identity,
                reason=reason.strip(),
            )
            return True

    async def consume(
        self,
        request_id: str,
        invocation: ApprovalInvocation,
    ) -> ApprovalReceipt | None:
        async with self._lock:
            record = self._records.get(request_id)
            if record is None:
                return None
            if is_expired(record.invocation):
                record.state = "expired"
                return None
            if (
                record.state != "approved"
                or record.receipt is None
                or record.receipt.action_digest != invocation.action_digest
                or not _identity_matches(record.invocation.identity, invocation.identity)
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
            return receipt

    async def get(self, request_id: str) -> ApprovalRecord | None:
        async with self._lock:
            return self._records.get(request_id)

    async def list_pending(self, identity: ApprovalIdentity) -> list[ApprovalRecord]:
        async with self._lock:
            now = datetime.now(UTC)
            return [
                record
                for record in self._records.values()
                if record.state == "pending"
                and not is_expired(record.invocation, now=now)
                and _identity_matches(record.invocation.identity, identity)
            ]

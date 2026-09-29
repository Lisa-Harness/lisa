"""Pack-neutral WorkProducts: durable, approvable units of work in the SDK.

A :class:`WorkProduct` is the typed execution-state contract that *two or more
packs* agree on without importing each other. It started life as the kernel's
code-change artifact; it was renamed and promoted to the SDK because it is not
code-specific — it is the host-mediated, durable record of a unit of work a
pack produced and another pack may carry further (a patched worktree → commit
→ push → PR; a captured sandbox layer → a staged change set; an authored doc).

Why this lives in the SDK, not in the kernel or a pack:

The kernel only ever threads an *opaque* artifact store through execution
contexts (an ``Any``-typed field); it never binds the concrete type. The packs
layer forbids cross-pack imports (``sdk/packs/**`` architecture rule), so the
shared contract must live on neutral ground that every pack already depends on
— the SDK. The kernel declares no artifact type; the SDK declares the typed
models and the :class:`WorkProductStore` Protocol. The host owns the concrete
implementation (a file-based store) and injects it; packs and SDK code depend
on the Protocol, never on the host's file format. This is the same
dependency-inversion seam as the SDK's state providers.

The field names retain the code-change lineage (``worktree_path``,
``base_sha``, ``commit_sha``, ``pr_url`` …) because those are the fields a
publication pipeline operates on today; a WorkProduct for a non-code artifact
simply leaves the VCS fields empty.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

__all__ = [
    "WorkProduct",
    "WorkProductCheck",
    "WorkProductCheckStatus",
    "WorkProductStatus",
    "WorkProductStore",
]

WorkProductCheckStatus = Literal["pending", "passed", "failed", "timed_out"]
WorkProductStatus = Literal[
    "created",
    "patched",
    "checked",
    "committed",
    "pushed",
    "pr_opened",
    "cleaned",
]


class WorkProductCheck(BaseModel):
    """One named finite check executed against a WorkProduct revision."""

    check_id: str
    revision: int
    checked_tree_sha: str = ""
    status: WorkProductCheckStatus = "pending"
    argv: list[str] = Field(default_factory=list)
    duration_ms: int = 0
    stdout: str = ""
    stderr: str = ""
    completed_at: datetime | None = None


class WorkProduct(BaseModel):
    """Durable state shared between a producing pack and a publishing pack."""

    artifact_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    task_id: str = ""
    workspace_id: str
    user_id: str
    user_slug: str
    surface_type: str
    surface_id: str
    session_plan_digest: str
    source_root: str
    worktree_path: str = ""
    base_sha: str = ""
    head_sha: str = ""
    branch: str = ""
    revision: int = 0
    changed_paths: list[str] = Field(default_factory=list)
    checks: dict[str, WorkProductCheck] = Field(default_factory=dict)
    required_check_ids: list[str] = Field(default_factory=list)
    commit_sha: str = ""
    pushed_sha: str = ""
    pr_url: str = ""
    pr_number: int | None = None
    status: WorkProductStatus = "created"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    safe_summary: str = ""

    def belongs_to(self, ctx: object) -> bool:
        """Return whether this WorkProduct is owned by the current host identity.

        The single canonical ownership check for the artifact handoff: an exact
        match on workspace, user, surface, and session plan digest. Both the
        producing pack (coding) and the publishing pack (github) call this so
        the identity rule cannot drift between them. ``ctx`` is a
        ``kernel.models.ToolExecutionContext``; typed as ``object`` here because
        the SDK must not import the kernel's higher-level models.
        """
        return (
            self.workspace_id == getattr(ctx, "workspace_id", None)
            and self.user_id == getattr(ctx, "user_id", None)
            and self.user_slug == getattr(ctx, "user_slug", None)
            and self.surface_type == getattr(ctx, "surface_type", None)
            and self.surface_id == getattr(ctx, "surface_id", None)
            and self.session_plan_digest == getattr(ctx, "session_plan_digest", None)
        )


@runtime_checkable
class WorkProductStore(Protocol):
    """Host-owned persistence boundary for WorkProducts."""

    async def create(self, artifact: WorkProduct) -> WorkProduct:
        """Persist a new WorkProduct and return it."""

    async def get(self, artifact_id: str) -> WorkProduct | None:
        """Read one WorkProduct by stable ID."""

    async def list_owned(self, ctx: object) -> list[WorkProduct]:
        """List WorkProducts owned by the exact execution identity in ``ctx``."""

    async def save(self, artifact: WorkProduct) -> WorkProduct:
        """Atomically replace one WorkProduct's state."""

    async def delete(self, artifact_id: str) -> bool:
        """Delete a WorkProduct only after its owned worktree is cleaned."""

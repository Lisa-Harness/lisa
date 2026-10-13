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
from sdk.identity import execution_scope

__all__ = [
    "WorkProduct",
    "WorkProductCheck",
    "WorkProductCheckStatus",
    "WorkProductPreparation",
    "WorkProductPreparationStatus",
    "WorkProductStatus",
    "WorkProductStore",
]

WorkProductCheckStatus = Literal["pending", "passed", "failed", "timed_out"]
WorkProductPreparationStatus = Literal["pending", "ready", "failed", "stale"]
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


class WorkProductPreparation(BaseModel):
    """Evidence that a WorkProduct environment was prepared for execution."""

    status: WorkProductPreparationStatus = "pending"
    recipe_id: str = ""
    recipe_digest: str = ""
    dependency_manifest_digest: str = ""
    environment_path: str = ""
    python_path: str = ""
    baseline_tree_sha: str = ""
    baseline_checks: dict[str, WorkProductCheck] = Field(default_factory=dict)
    probe_stdout: str = ""
    probe_stderr: str = ""
    completed_at: datetime | None = None


class WorkProduct(BaseModel):
    """Durable state shared between a producing pack and a publishing pack."""

    artifact_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    #: The task currently responsible for this WorkProduct (see :meth:`claim_for_task`).
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
    preparation: WorkProductPreparation | None = None
    required_check_ids: list[str] = Field(default_factory=list)
    commit_sha: str = ""
    pushed_sha: str = ""
    pr_url: str = ""
    pr_number: int | None = None
    #: Optional base tip for import-anchored worktree creation; empty for
    #: ordinary HEAD-based worktrees. The worktree therefore branches at this
    #: commit's parent, and this commit is the only one the artifact may import.
    import_commit: str = ""
    #: Upstream commits imported into this artifact, each recorded at its
    #: worktree creation base sha; cleared when the artifact commits, so a
    #: commit carries the imported content out of the pending-import state.
    imported_commits: dict[str, str] = Field(default_factory=dict)
    status: WorkProductStatus = "created"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    safe_summary: str = ""

    def belongs_to(self, ctx: object) -> bool:
        """Return whether this WorkProduct is owned by the current host identity.

        The single canonical ownership check for the artifact handoff. Both the
        producing pack (coding) and the publishing pack (github) call this so
        the identity rule cannot drift between them. Scope comes from
        :func:`sdk.identity.execution_scope`, which documents why
        ``session_plan_digest`` is not part of it; use
        :meth:`session_plan_matches` to report that drift instead.
        """
        return execution_scope(self) == execution_scope(ctx)

    def session_plan_matches(self, ctx: object) -> bool:
        """Return whether this WorkProduct was produced under ``ctx``'s session plan."""
        return self.session_plan_digest == getattr(ctx, "session_plan_digest", None)

    def claim_for_task(self, ctx: object) -> bool:
        """Make ``ctx``'s task responsible for this WorkProduct; return whether it moved.

        Tasks are per-intent and short-lived; WorkProducts outlive them. A
        producing pack calls this whenever a task advances an owned
        WorkProduct, so completion is judged against the task doing the work
        rather than the one that happened to create it. Callers must have
        already checked :meth:`belongs_to`.
        """
        task_id = str(getattr(ctx, "task_id", "") or "")
        if not task_id or task_id == self.task_id:
            return False
        self.task_id = task_id
        return True


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

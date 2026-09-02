"""Pack-neutral artifacts that describe an approved local code change.

Why this lives in the kernel, not in a pack or the host:

This module defines the shared execution-state contract that *two or more
packs* must agree on without importing each other. A local-coding pack
creates and patches a :class:`CodeChangeArtifact`; a publication pack
reads the same artifact to commit, push, and open a PR. The packs layer
forbids cross-pack imports (``sdk/packs/**`` architecture rule), so the
handoff state cannot live in either pack — it must be promoted to
neutral ground.

The kernel owns only the typed models and the
:class:`CodeChangeArtifactStore` Protocol. The host owns the concrete
implementation (``host/code_artifacts.FileCodeChangeArtifactStore``),
wired in ``host/runtime.py``. This is the standard dependency-inversion
seam: packs and kernel code depend on the Protocol, never on the
host's file format, and the kernel never imports host code
(``kernel/**`` architecture rule).

This is the same pattern as :mod:`kernel.state_providers` and the
:class:`~kernel.seams.protocols.StateProvider` Protocol — the kernel
declares the contract; the host injects the wiring.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

__all__ = [
    "CodeChangeArtifact",
    "CodeChangeArtifactStore",
    "CodeChangeStatus",
    "CodeCheckRecord",
    "CodeCheckStatus",
]

CodeCheckStatus = Literal["pending", "passed", "failed", "timed_out"]
CodeChangeStatus = Literal[
    "created",
    "patched",
    "checked",
    "committed",
    "pushed",
    "pr_opened",
    "cleaned",
]


class CodeCheckRecord(BaseModel):
    """One named finite check executed against an artifact revision."""

    check_id: str
    revision: int
    status: CodeCheckStatus = "pending"
    argv: list[str] = Field(default_factory=list)
    duration_ms: int = 0
    stdout: str = ""
    stderr: str = ""
    completed_at: datetime | None = None


class CodeChangeArtifact(BaseModel):
    """Durable state shared between local coding and remote publication packs."""

    artifact_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
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
    checks: dict[str, CodeCheckRecord] = Field(default_factory=dict)
    required_check_ids: list[str] = Field(default_factory=list)
    commit_sha: str = ""
    pushed_sha: str = ""
    pr_url: str = ""
    pr_number: int | None = None
    status: CodeChangeStatus = "created"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    safe_summary: str = ""


@runtime_checkable
class CodeChangeArtifactStore(Protocol):
    """Host-owned persistence boundary for code change artifacts."""

    async def create(self, artifact: CodeChangeArtifact) -> CodeChangeArtifact:
        """Persist a new artifact and return it."""

    async def get(self, artifact_id: str) -> CodeChangeArtifact | None:
        """Read one artifact by stable ID."""

    async def save(self, artifact: CodeChangeArtifact) -> CodeChangeArtifact:
        """Atomically replace one artifact state."""

    async def delete(self, artifact_id: str) -> bool:
        """Delete an artifact only after its owned worktree is cleaned."""

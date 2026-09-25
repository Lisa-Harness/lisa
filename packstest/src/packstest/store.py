"""In-memory ``WorkProductStore`` double.

Pack tools ``cast()`` ``ctx.code_change_artifact_store`` to the SDK's
``WorkProductStore`` Protocol. The host's file-backed implementation lives
behind the host boundary (packs must never import it), so the harness owns a
minimal in-memory double that satisfies the same Protocol. Fidelity rule:
implement exactly what pack code exercises (create/get/save/delete) and no
more; if pack code starts using a new method, add it here with a test.
"""

from __future__ import annotations

from typing import Any

from sdk.workproducts import WorkProduct

__all__ = ["MemoryWorkProductStore"]


class MemoryWorkProductStore:
    """Deterministic in-memory ``WorkProductStore`` for tests."""

    def __init__(self) -> None:
        self._artifacts: dict[str, WorkProduct] = {}

    async def create(self, artifact: WorkProduct) -> WorkProduct:
        """Persist a new WorkProduct and return it."""

        self._artifacts[artifact.artifact_id] = artifact.model_copy(deep=True)
        return artifact

    async def get(self, artifact_id: str) -> WorkProduct | None:
        """Read one WorkProduct by stable ID."""

        artifact = self._artifacts.get(artifact_id)
        return artifact.model_copy(deep=True) if artifact is not None else None

    async def save(self, artifact: WorkProduct) -> WorkProduct:
        """Atomically replace one WorkProduct's state."""

        self._artifacts[artifact.artifact_id] = artifact.model_copy(deep=True)
        return artifact

    async def delete(self, artifact_id: str) -> bool:
        """Delete a WorkProduct only after its owned worktree is cleaned."""

        return self._artifacts.pop(artifact_id, None) is not None

    # ── Test-only surface (not part of the Protocol) ────────────────────────

    def all(self) -> list[WorkProduct]:
        """Return deep copies of every stored artifact (test assertions)."""

        return [a.model_copy(deep=True) for a in self._artifacts.values()]

    def count(self) -> int:
        """Return the number of stored artifacts."""

        return len(self._artifacts)

    def seed(self, artifact: WorkProduct) -> WorkProduct:
        """Insert an artifact without going through ``create``."""

        self._artifacts[artifact.artifact_id] = artifact.model_copy(deep=True)
        return artifact

    def make_artifact(self, **overrides: Any) -> WorkProduct:
        """Build an artifact owned by the harness's default identity.

        Reads the default identity lazily: ``context`` imports this module,
        so a top-level import here would be a cycle.
        """

        from packstest.context import (  # lazy: breaks cycle with packstest.context
            DEFAULT_IDENTITY,
        )

        fields = {
            "workspace_id": DEFAULT_IDENTITY["workspace_id"],
            "user_id": DEFAULT_IDENTITY["user_id"],
            "user_slug": DEFAULT_IDENTITY["user_slug"],
            "surface_type": DEFAULT_IDENTITY["surface_type"],
            "surface_id": DEFAULT_IDENTITY["surface_id"],
            "session_plan_digest": DEFAULT_IDENTITY["session_plan_digest"],
            "source_root": "",
        }
        fields.update(overrides)
        return WorkProduct(**fields)

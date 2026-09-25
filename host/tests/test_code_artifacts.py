"""Tests for host-owned code change artifact persistence."""

from __future__ import annotations

from pathlib import Path

import pytest
from host.code_artifacts import FileCodeChangeArtifactStore
from sdk.workproducts import WorkProduct, WorkProductCheck


@pytest.mark.asyncio
async def test_file_store_round_trips_updated_artifact(tmp_path: Path) -> None:
    """The host store persists an artifact independently of either pack."""

    store = FileCodeChangeArtifactStore(tmp_path / "code-changes.json")
    artifact = WorkProduct(
        workspace_id="workspace",
        user_id="user",
        user_slug="user",
        surface_type="personal_dm",
        surface_id="surface",
        session_plan_digest="plan",
        source_root="C:/repo",
        branch="lisa/change",
        required_check_ids=["unit"],
    )

    await store.create(artifact)
    artifact.revision = 1
    artifact.changed_paths = ["src/main.py"]
    artifact.checks["unit"] = WorkProductCheck(
        check_id="unit",
        revision=1,
        status="passed",
        argv=["uv", "run", "pytest"],
    )
    await store.save(artifact)

    loaded = await store.get(artifact.artifact_id)
    assert loaded is not None
    assert loaded.revision == 1
    assert loaded.checks["unit"].status == "passed"
    assert await store.delete(artifact.artifact_id) is True
    assert await store.get(artifact.artifact_id) is None

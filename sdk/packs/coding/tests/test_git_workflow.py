"""Temporary-repository integration tests for the coding worktree workflow."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from hosts.core.code_artifacts import FileCodeChangeArtifactStore
from kernel.models import ToolExecutionContext
from sdk.packs.coding.tools.apply_patch.apply_patch import execute as apply_patch
from sdk.packs.coding.tools.cleanup_worktree.cleanup_worktree import execute as cleanup_worktree
from sdk.packs.coding.tools.commit.commit import execute as commit
from sdk.packs.coding.tools.create_worktree.create_worktree import execute as create_worktree
from sdk.packs.coding.tools.push.push import execute as push
from sdk.packs.coding.tools.run_check.run_check import execute as run_check
from sdk.packs.coding.tools.status.status import execute as status


async def _git(root: Path, *args: str) -> None:
    process = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=str(root),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    assert process.returncode == 0, stderr.decode("utf-8", errors="replace")


async def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    root.mkdir()
    await _git(root, "init")
    await _git(root, "config", "user.email", "test@example.invalid")
    await _git(root, "config", "user.name", "Lisa Test")
    (root / "app.txt").write_text("hello\n", encoding="utf-8")
    await _git(root, "add", "app.txt")
    await _git(root, "commit", "-m", "initial")
    remote = tmp_path / "remote.git"
    process = await asyncio.create_subprocess_exec(
        "git",
        "init",
        "--bare",
        str(remote),
        cwd=str(tmp_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    assert process.returncode == 0, stderr.decode("utf-8", errors="replace")
    await _git(root, "remote", "add", "origin", str(remote))
    return root


def _context(
    root: Path,
    store: FileCodeChangeArtifactStore,
    params: dict[str, object],
) -> ToolExecutionContext:
    return ToolExecutionContext(
        tool_id="test",
        workspace_id="workspace",
        user_id="user",
        user_slug="user",
        surface_id="surface",
        mode_id="coding",
        session_plan_digest="plan",
        parameters=params,
        code_change_artifact_store=store,
        pack_config={
            "workspace_root": str(root),
            "checks": [
                {
                    "id": "unit",
                    "argv": [sys.executable, "-c", "print('checked')"],
                    "timeout_seconds": 30,
                }
            ],
            "required_check_ids": ["unit"],
        },
    )


@pytest.mark.asyncio
async def test_patch_invalidates_checks_and_status_tracks_owned_worktree(tmp_path: Path) -> None:
    """A worktree patch receives a new revision and only configured checks can run."""

    root = await _repository(tmp_path)
    store = FileCodeChangeArtifactStore(tmp_path / "artifacts.json")
    created = await create_worktree(_context(root, store, {"summary": "change greeting"}))
    assert created.error == ""
    artifact_id = created.output_json["artifact_id"]
    patch = """diff --git a/app.txt b/app.txt
--- a/app.txt
+++ b/app.txt
@@ -1 +1 @@
-hello
+hello world
"""
    patched = await apply_patch(_context(root, store, {"artifact_id": artifact_id, "patch": patch}))
    assert patched.error == ""
    assert patched.output_json["revision"] == 1

    checked = await run_check(
        _context(root, store, {"artifact_id": artifact_id, "check_id": "unit"})
    )
    assert checked.output_json["status"] == "passed"

    committed = await commit(
        _context(root, store, {"artifact_id": artifact_id, "message": "Change greeting"})
    )
    assert committed.error == ""
    pushed = await push(_context(root, store, {"artifact_id": artifact_id}))
    assert pushed.error == ""

    inspected = await status(_context(root, store, {"artifact_id": artifact_id}))
    assert inspected.error == ""
    assert inspected.output_json["revision"] == 1
    assert inspected.output_json["status"] == "pushed"

    cleaned = await cleanup_worktree(_context(root, store, {"artifact_id": artifact_id}))
    assert cleaned.error == ""


@pytest.mark.asyncio
async def test_patch_rejects_traversal_and_allows_owned_new_file(tmp_path: Path) -> None:
    """A patch cannot escape the worktree but may create a contained source file."""

    root = await _repository(tmp_path)
    store = FileCodeChangeArtifactStore(tmp_path / "artifacts.json")
    created = await create_worktree(_context(root, store, {"summary": "add source file"}))
    artifact_id = created.output_json["artifact_id"]
    traversal_patch = """diff --git a/../outside.txt b/../outside.txt
--- a/../outside.txt
+++ b/../outside.txt
@@ -0,0 +1 @@
+outside
"""

    rejected = await apply_patch(
        _context(root, store, {"artifact_id": artifact_id, "patch": traversal_patch})
    )

    assert "Patch path is not allowed" in rejected.error
    new_file_patch = """diff --git a/new.txt b/new.txt
new file mode 100644
index 0000000..3e75765
--- /dev/null
+++ b/new.txt
@@ -0,0 +1 @@
+new source file
"""
    applied = await apply_patch(
        _context(root, store, {"artifact_id": artifact_id, "patch": new_file_patch})
    )

    assert applied.error == ""
    assert applied.output_json["changed_paths"] == ["new.txt"]


@pytest.mark.asyncio
async def test_stale_checks_block_commit_and_dirty_worktree_cleanup(tmp_path: Path) -> None:
    """A later patch invalidates checks and prevents both commit and cleanup."""

    root = await _repository(tmp_path)
    store = FileCodeChangeArtifactStore(tmp_path / "artifacts.json")
    created = await create_worktree(_context(root, store, {"summary": "two changes"}))
    artifact_id = created.output_json["artifact_id"]
    first_patch = """diff --git a/app.txt b/app.txt
--- a/app.txt
+++ b/app.txt
@@ -1 +1 @@
-hello
+hello one
"""
    second_patch = """diff --git a/app.txt b/app.txt
--- a/app.txt
+++ b/app.txt
@@ -1 +1 @@
-hello one
+hello two
"""
    first_applied = await apply_patch(
        _context(root, store, {"artifact_id": artifact_id, "patch": first_patch})
    )
    checked = await run_check(
        _context(root, store, {"artifact_id": artifact_id, "check_id": "unit"})
    )
    second_applied = await apply_patch(
        _context(root, store, {"artifact_id": artifact_id, "patch": second_patch})
    )
    assert first_applied.error == ""
    assert checked.error == ""
    assert second_applied.error == ""

    committed = await commit(
        _context(root, store, {"artifact_id": artifact_id, "message": "Change twice"})
    )
    cleaned = await cleanup_worktree(_context(root, store, {"artifact_id": artifact_id}))

    assert committed.error == "Every required check must pass at the current artifact revision."
    assert "use --force" in cleaned.error


@pytest.mark.asyncio
async def test_commit_and_push_reject_invalid_artifact_states(tmp_path: Path) -> None:
    """No-change, protected-branch, and uncommitted artifacts cannot publish."""

    root = await _repository(tmp_path)
    store = FileCodeChangeArtifactStore(tmp_path / "artifacts.json")
    created = await create_worktree(_context(root, store, {"summary": "guard states"}))
    artifact_id = created.output_json["artifact_id"]
    checked = await run_check(
        _context(root, store, {"artifact_id": artifact_id, "check_id": "unit"})
    )
    assert checked.error == ""

    no_change_commit = await commit(
        _context(root, store, {"artifact_id": artifact_id, "message": "No change"})
    )
    no_commit_push = await push(_context(root, store, {"artifact_id": artifact_id}))
    artifact = await store.get(artifact_id)
    assert artifact is not None
    artifact.branch = "main"
    await store.save(artifact)
    protected_commit = await commit(
        _context(root, store, {"artifact_id": artifact_id, "message": "Protected branch"})
    )

    assert no_change_commit.error == "Owned worktree has no changes to commit."
    assert no_commit_push.error == "Artifact has no committed revision to push."
    assert protected_commit.error == "Artifact branch is not an owned unprotected feature branch."

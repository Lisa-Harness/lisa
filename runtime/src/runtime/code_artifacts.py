"""Atomic file-backed storage for pack-neutral WorkProducts.

The typed contract (``WorkProduct`` / ``WorkProductStore`` Protocol) lives in
``sdk.workproducts``; the host only owns this concrete file store and injects
it into the opaque ``code_change_artifact_store`` execution-context field."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kernel.turn_hooks import TaskCompletionEvidence, TaskCompletionRequest
from sdk.filelock import async_file_lock, atomic_write_text, sidecar_lock_path
from sdk.workproducts import WorkProduct

__all__ = ["FileCodeChangeArtifactStore", "verify_checked_commit"]

logger = logging.getLogger(__name__)


async def _task_artifact_ids(
    request: TaskCompletionRequest, *, store: FileCodeChangeArtifactStore, identity: Any
) -> list[str]:
    """Select WorkProducts this task created or claimed (the store is the durable record)."""
    ids = dict.fromkeys(request.receipt_artifact_ids)
    for artifact in await store.list_owned(identity):
        if artifact.task_id == request.task_id and artifact.status != "cleaned":
            ids.setdefault(artifact.artifact_id)
    return list(ids)


async def _git_read(root: Path, *args: str) -> str | None:
    try:
        process = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            str(root),
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=10)
    except (OSError, TimeoutError):
        if "process" in locals() and process.returncode is None:
            process.kill()
            await process.wait()
        return None
    if process.returncode != 0:
        return None
    return stdout.decode("utf-8", errors="replace").rstrip("\n")


async def _committed_tree(artifact: WorkProduct) -> str | None:
    """Read the owned worktree's unchanged commit tree from Git itself.

    A reused worktree may stack several task commits on the recorded base, so
    the base must be an ancestor of HEAD (not necessarily its parent) and the
    committed paths span every commit since the base.
    """
    source = Path(artifact.source_root).resolve()
    worktree = Path(artifact.worktree_path).resolve()
    if not source.is_dir() or not worktree.is_dir() or not (worktree / ".git").exists():
        return None
    source_common = await _git_read(source, "rev-parse", "--git-common-dir")
    worktree_common = await _git_read(worktree, "rev-parse", "--git-common-dir")
    if not source_common or not worktree_common:
        return None
    if (source / source_common).resolve() != (worktree / worktree_common).resolve():
        return None

    head = await _git_read(worktree, "rev-parse", "HEAD")
    tree = await _git_read(worktree, "rev-parse", "HEAD^{tree}")
    parent_tree = await _git_read(worktree, "rev-parse", "HEAD~1^{tree}")
    branch = await _git_read(worktree, "branch", "--show-current")
    dirty = await _git_read(worktree, "status", "--porcelain", "--untracked-files=all")
    base_is_ancestor = await _git_read(
        worktree, "merge-base", "--is-ancestor", artifact.base_sha, "HEAD"
    )
    if (
        head != artifact.commit_sha
        or base_is_ancestor is None
        or not tree
        or tree == parent_tree
        or branch != artifact.branch
        or dirty != ""
    ):
        return None
    committed_paths = await _git_read(
        worktree, "diff", "--name-only", "-z", artifact.base_sha, "HEAD", "--"
    )
    if committed_paths is None or not set(artifact.changed_paths).issubset(
        set(committed_paths.rstrip("\0").split("\0"))
    ):
        return None
    return tree


def _record_problem(artifact: WorkProduct, *, task_id: str, identity: Any) -> str:
    """Return why the stored record cannot be this task's commit, or ``""``."""
    artifact_id = artifact.artifact_id
    if not artifact.belongs_to(identity):
        return f"Worktree {artifact_id} is not owned by this session."
    if artifact.task_id != task_id:
        return f"Worktree {artifact_id} has been claimed by another task."
    if artifact.status not in ("committed", "pushed", "pr_opened"):
        return f"Worktree {artifact_id} is {artifact.status}, not committed."
    if artifact.revision < 1 or not artifact.changed_paths:
        return f"Worktree {artifact_id} has no changes."
    if (
        not artifact.commit_sha
        or artifact.commit_sha == artifact.base_sha
        or artifact.head_sha != artifact.commit_sha
        or not artifact.branch
        or not artifact.worktree_path
    ):
        return f"Worktree {artifact_id} has no recorded commit for its changes."
    return ""


def _unchecked_required(artifact: WorkProduct, *, tree: str) -> str:
    """Return the first required check that did not pass on ``tree``, or ``""``."""
    for check_id in artifact.required_check_ids:
        check = artifact.checks.get(check_id)
        if (
            check is None
            or check.check_id != check_id
            or check.status != "passed"
            or check.revision != artifact.revision
            or check.checked_tree_sha != tree
        ):
            return check_id
    return ""


async def verify_checked_commit(
    request: TaskCompletionRequest,
    *,
    store: FileCodeChangeArtifactStore,
    identity: Any,
) -> TaskCompletionEvidence:
    """Verify a real, clean committed tree for exactly this task's owned artifact."""

    if not request.task_id or not request.origin_tool_id:
        reason = "Task identity is unavailable, so no worktree can be verified."
        return TaskCompletionEvidence(verified=False, reason=reason)
    artifact_ids = await _task_artifact_ids(request, store=store, identity=identity)
    artifact = await store.get(artifact_ids[0]) if len(artifact_ids) == 1 else None
    if not artifact_ids:
        reason = "This task has not created or worked on a worktree."
    elif len(artifact_ids) > 1:
        reason = (
            f"This task is working on {len(artifact_ids)} worktrees "
            f"({', '.join(artifact_ids)}); finish or clean up all but one."
        )
    elif artifact is None:
        reason = f"Worktree {artifact_ids[0]} is not owned by this session."
    else:
        reason = _record_problem(artifact, task_id=request.task_id, identity=identity)
    if reason or artifact is None:
        return TaskCompletionEvidence(verified=False, reason=reason)

    tree = await _committed_tree(artifact)
    if tree is None:
        reason = (
            f"Worktree {artifact.artifact_id} does not match its recorded commit "
            "(uncommitted edits, a moved HEAD or branch, or a history that no "
            "longer contains the recorded base commit)."
        )
    elif check_id := _unchecked_required(artifact, tree=tree):
        reason = (
            f"Required check '{check_id}' has not passed on the committed tree "
            f"of worktree {artifact.artifact_id}."
        )
    if reason:
        return TaskCompletionEvidence(verified=False, reason=reason)
    return TaskCompletionEvidence(verified=True, artifact_id=artifact.artifact_id)


class FileCodeChangeArtifactStore:
    """Persist WorkProducts under a host-controlled data directory."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()
        self._lock_path = sidecar_lock_path(path)

    async def create(self, artifact: WorkProduct) -> WorkProduct:
        async with self._lock, async_file_lock(self._lock_path):
            artifacts = self._load()
            if artifact.artifact_id in artifacts:
                raise ValueError(f"Artifact '{artifact.artifact_id}' already exists.")
            artifact.updated_at = datetime.now(UTC)
            artifacts[artifact.artifact_id] = artifact
            self._store(artifacts)
            return artifact

    async def get(self, artifact_id: str) -> WorkProduct | None:
        async with self._lock, async_file_lock(self._lock_path):
            return self._load().get(artifact_id)

    async def list_owned(self, ctx: object) -> list[WorkProduct]:
        async with self._lock, async_file_lock(self._lock_path):
            return [artifact for artifact in self._load().values() if artifact.belongs_to(ctx)]

    async def save(self, artifact: WorkProduct) -> WorkProduct:
        async with self._lock, async_file_lock(self._lock_path):
            artifacts = self._load()
            if artifact.artifact_id not in artifacts:
                raise KeyError(f"Unknown artifact '{artifact.artifact_id}'.")
            artifact.updated_at = datetime.now(UTC)
            artifacts[artifact.artifact_id] = artifact
            self._store(artifacts)
            return artifact

    async def delete(self, artifact_id: str) -> bool:
        async with self._lock, async_file_lock(self._lock_path):
            artifacts = self._load()
            if artifact_id not in artifacts:
                return False
            del artifacts[artifact_id]
            self._store(artifacts)
            return True

    def _load(self) -> dict[str, WorkProduct]:
        if not self._path.is_file():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("code_artifacts: failed to load %s", self._path, exc_info=True)
            return {}
        if not isinstance(raw, dict):
            return {}
        artifacts: dict[str, WorkProduct] = {}
        for artifact_id, value in raw.items():
            try:
                artifacts[str(artifact_id)] = WorkProduct.model_validate(value)
            except Exception:  # pylint: disable=broad-except
                continue
        return artifacts

    def _store(self, artifacts: dict[str, WorkProduct]) -> None:
        payload = {
            artifact_id: artifact.model_dump(mode="json")
            for artifact_id, artifact in artifacts.items()
        }
        atomic_write_text(
            self._path,
            json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
        )

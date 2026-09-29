"""Atomic file-backed storage for pack-neutral WorkProducts.

The typed contract (``WorkProduct`` / ``WorkProductStore`` Protocol) lives in
``sdk.workproducts``; the host only owns this concrete file store and injects
it into the opaque ``code_change_artifact_store`` execution-context field."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kernel.turn_hooks import TaskCompletionEvidence, TaskCompletionRequest
from sdk.workproducts import WorkProduct

from runtime.thread import load_thread_range, next_turn_index

__all__ = ["FileCodeChangeArtifactStore", "verify_checked_commit"]


def _task_artifact_ids(
    request: TaskCompletionRequest, *, thread_path: Path | None
) -> set[str]:
    """Read only exact task-linked producer receipts from durable thread entries."""
    ids = set(request.receipt_artifact_ids)
    if thread_path is None:
        return ids
    entries = load_thread_range(
        thread_path=thread_path,
        start_turn=0,
        end_turn=next_turn_index(thread_path=thread_path),
    )
    pending = {
        str(provenance["pending_approval_request_id"]): provenance
        for entry in entries
        if isinstance(provenance := entry.get("provenance"), dict)
        and provenance.get("task_id") == request.task_id
        and provenance.get("pending_approval_request_id")
    }
    for entry in entries:
        provenance = entry.get("provenance") or {}
        if provenance.get("task_id") != request.task_id:
            continue
        source = provenance.get("source")
        if source == "host-approved":
            approval_id = str(provenance.get("approval_request_id") or "")
            if (
                approval_id not in pending
                or provenance.get("target_kind") != "tool"
                or provenance.get("target_id") != request.origin_tool_id
            ):
                continue
        elif source != "observed":
            continue
        for receipt in entry.get("effect_receipts") or []:
            if (
                isinstance(receipt, dict)
                and receipt.get("capability_id") == request.origin_tool_id
                and receipt.get("effect") in ("write", "external_write")
                and receipt.get("artifact_id")
            ):
                ids.add(str(receipt["artifact_id"]))
    return ids


async def _git_read(root: Path, *args: str) -> str | None:
    try:
        process = await asyncio.create_subprocess_exec(
            "git", "-C", str(root), *args,
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
    """Read the owned worktree's unchanged commit tree from Git itself."""
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
    parent = await _git_read(worktree, "rev-parse", "HEAD^")
    tree = await _git_read(worktree, "rev-parse", "HEAD^{tree}")
    parent_tree = await _git_read(worktree, "rev-parse", "HEAD~1^{tree}")
    branch = await _git_read(worktree, "branch", "--show-current")
    dirty = await _git_read(worktree, "status", "--porcelain", "--untracked-files=all")
    if (
        head != artifact.commit_sha
        or parent != artifact.base_sha
        or not tree
        or tree == parent_tree
        or branch != artifact.branch
        or dirty != ""
    ):
        return None
    committed_paths = await _git_read(worktree, "diff", "--name-only", "-z", "HEAD^", "HEAD", "--")
    if committed_paths is None or not set(artifact.changed_paths).issubset(
        set(committed_paths.rstrip("\0").split("\0"))
    ):
        return None
    return tree


async def verify_checked_commit(
    request: TaskCompletionRequest,
    *,
    store: FileCodeChangeArtifactStore,
    thread_path: Path | None,
    identity: Any,
) -> TaskCompletionEvidence:
    """Verify a real, clean committed tree for exactly this task's owned artifact."""
    failure = TaskCompletionEvidence(
        verified=False,
        reason="This task has no owned worktree with a current checked commit.",
    )
    if not request.task_id or not request.origin_tool_id:
        return failure
    artifact_ids = _task_artifact_ids(request, thread_path=thread_path)
    if len(artifact_ids) != 1:
        return failure
    artifact_id = next(iter(artifact_ids))
    artifact = await store.get(artifact_id)
    if (
        artifact is None
        or not artifact.belongs_to(identity)
        or artifact.task_id != request.task_id
        or artifact.status not in ("committed", "pushed", "pr_opened")
        or artifact.revision < 1
        or not artifact.changed_paths
        or not artifact.commit_sha
        or artifact.commit_sha == artifact.base_sha
        or artifact.head_sha != artifact.commit_sha
        or not artifact.branch
        or not artifact.worktree_path
    ):
        return failure

    tree = await _committed_tree(artifact)
    if tree is None:
        return failure
    for check_id in artifact.required_check_ids:
        check = artifact.checks.get(check_id)
        if (
            check is None
            or check.check_id != check_id
            or check.status != "passed"
            or check.revision != artifact.revision
            or check.checked_tree_sha != tree
        ):
            return failure
    return TaskCompletionEvidence(verified=True, artifact_id=artifact_id)


class FileCodeChangeArtifactStore:
    """Persist WorkProducts under a host-controlled data directory."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()

    async def create(self, artifact: WorkProduct) -> WorkProduct:
        async with self._lock:
            artifacts = self._load()
            if artifact.artifact_id in artifacts:
                raise ValueError(f"Artifact '{artifact.artifact_id}' already exists.")
            artifact.updated_at = datetime.now(UTC)
            artifacts[artifact.artifact_id] = artifact
            self._store(artifacts)
            return artifact

    async def get(self, artifact_id: str) -> WorkProduct | None:
        async with self._lock:
            return self._load().get(artifact_id)

    async def list_owned(self, ctx: object) -> list[WorkProduct]:
        async with self._lock:
            return [artifact for artifact in self._load().values() if artifact.belongs_to(ctx)]

    async def save(self, artifact: WorkProduct) -> WorkProduct:
        async with self._lock:
            artifacts = self._load()
            if artifact.artifact_id not in artifacts:
                raise KeyError(f"Unknown artifact '{artifact.artifact_id}'.")
            artifact.updated_at = datetime.now(UTC)
            artifacts[artifact.artifact_id] = artifact
            self._store(artifacts)
            return artifact

    async def delete(self, artifact_id: str) -> bool:
        async with self._lock:
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
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            artifact_id: artifact.model_dump(mode="json")
            for artifact_id, artifact in artifacts.items()
        }
        temporary_path = self._path.with_suffix(self._path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary_path, self._path)

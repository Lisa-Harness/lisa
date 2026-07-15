"""Bounded local Git workflow helpers for the coding pack."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from kernel.code_artifacts import CodeChangeArtifact, CodeChangeArtifactStore
from kernel.models import ToolExecutionContext
from sdk.packs.coding.config import CodingPackConfig
from sdk.packs.coding.local_files import CodingPathError, resolve_relative_path

__all__ = [
    "CodingWorkflowError",
    "ProcessResult",
    "artifact_branch_is_allowed",
    "artifact_has_current_required_checks",
    "artifact_owner_matches",
    "artifact_worktree",
    "changed_paths",
    "load_owned_artifact",
    "patch_paths",
    "run_git",
    "run_process",
]

_GIT_STATUS_PATH_OFFSET = 3


class CodingWorkflowError(RuntimeError):
    """Raised for a failed coding workflow precondition or bounded command."""


@dataclass(frozen=True, slots=True)
class ProcessResult:
    """Bounded process result; output is never unbounded host memory."""

    return_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    output_truncated: bool = False


async def _drain_limited(
    stream: asyncio.StreamReader,
    *,
    limit: int,
) -> tuple[bytes, bool]:
    """Drain a subprocess stream while retaining no more than ``limit`` bytes."""

    retained = bytearray()
    overflowed = False
    while chunk := await stream.read(8192):
        remaining = limit - len(retained)
        if remaining > 0:
            retained.extend(chunk[:remaining])
        if len(chunk) > remaining:
            overflowed = True
    return bytes(retained), overflowed


async def run_process(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    max_output_bytes: int,
    stdin_text: str | None = None,
) -> ProcessResult:
    """Execute an explicit argument vector without a shell or unbounded output."""

    if not argv or not argv[0]:
        raise CodingWorkflowError("A non-empty executable argument vector is required.")
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(cwd),
        stdin=asyncio.subprocess.PIPE if stdin_text is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    stdout_task = asyncio.create_task(_drain_limited(process.stdout, limit=max_output_bytes))
    stderr_task = asyncio.create_task(_drain_limited(process.stderr, limit=max_output_bytes))
    if stdin_text is not None:
        assert process.stdin is not None
        process.stdin.write(stdin_text.encode("utf-8"))
        await process.stdin.drain()
        process.stdin.close()
    timed_out = False
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
    except TimeoutError:
        timed_out = True
        process.kill()
        await process.wait()
    stdout, stdout_overflowed = await stdout_task
    stderr, stderr_overflowed = await stderr_task
    return ProcessResult(
        return_code=process.returncode or 0,
        stdout=stdout.decode("utf-8", errors="replace"),
        stderr=stderr.decode("utf-8", errors="replace"),
        timed_out=timed_out,
        output_truncated=stdout_overflowed or stderr_overflowed,
    )


async def run_git(
    root: Path,
    args: Sequence[str],
    *,
    config: CodingPackConfig,
    stdin_text: str | None = None,
) -> ProcessResult:
    """Run Git with a fixed executable and bounded output."""

    return await run_process(
        ("git", *args),
        cwd=root,
        timeout_seconds=config.git_timeout_seconds,
        max_output_bytes=config.max_command_output_bytes,
        stdin_text=stdin_text,
    )


def artifact_owner_matches(artifact: CodeChangeArtifact, ctx: ToolExecutionContext) -> bool:
    """Return whether an artifact belongs to the current exact host identity."""

    return (
        artifact.workspace_id == ctx.workspace_id
        and artifact.user_id == ctx.user_id
        and artifact.user_slug == ctx.user_slug
        and artifact.surface_type == ctx.surface_type
        and artifact.surface_id == ctx.surface_id
        and artifact.session_plan_digest == ctx.session_plan_digest
    )


def artifact_worktree(artifact: CodeChangeArtifact, *, config: CodingPackConfig) -> Path:
    """Return a validated owned worktree that remains outside the source root."""

    source_root = config.resolve_workspace_root()
    if Path(artifact.source_root).resolve() != source_root:
        raise CodingWorkflowError("Artifact source root does not match the configured workspace.")
    if not artifact.worktree_path:
        raise CodingWorkflowError("Artifact has no owned worktree.")
    worktree = Path(artifact.worktree_path).resolve()
    try:
        worktree.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise CodingWorkflowError("Artifact worktree must live outside the source workspace.")
    if not worktree.is_dir() or not (worktree / ".git").exists():
        raise CodingWorkflowError("Artifact worktree is not available as a Git worktree.")
    return worktree


async def load_owned_artifact(
    ctx: ToolExecutionContext,
    *,
    artifact_id: str,
    config: CodingPackConfig,
    require_worktree: bool = True,
) -> CodeChangeArtifact:
    """Load an artifact only when the host and workspace ownership match."""

    store = ctx.code_change_artifact_store
    if store is None:
        raise CodingWorkflowError("No code change artifact store is available.")
    artifact_store = cast(CodeChangeArtifactStore, store)
    artifact = await artifact_store.get(artifact_id)
    if artifact is None:
        raise CodingWorkflowError(f"Unknown code change artifact '{artifact_id}'.")
    if not artifact_owner_matches(artifact, ctx):
        raise CodingWorkflowError("Artifact does not belong to this workspace, user, or session.")
    if require_worktree:
        artifact_worktree(artifact, config=config)
    return artifact


def artifact_branch_is_allowed(artifact: CodeChangeArtifact, config: CodingPackConfig) -> bool:
    """Return whether the artifact branch is owned and outside protection policy."""

    return (
        bool(artifact.branch)
        and artifact.branch.startswith(config.branch_prefix)
        and artifact.branch not in config.protected_branches
    )


def artifact_has_current_required_checks(artifact: CodeChangeArtifact) -> bool:
    """Require every configured mandatory check to pass at the current revision."""

    return all(
        (record := artifact.checks.get(check_id)) is not None
        and record.status == "passed"
        and record.revision == artifact.revision
        for check_id in artifact.required_check_ids
    )


def patch_paths(patch: str, *, config: CodingPackConfig) -> list[str]:
    """Extract and validate every non-dev-null path mentioned by a unified diff."""

    paths: list[str] = []
    for line in patch.splitlines():
        if not line.startswith(("--- ", "+++ ")):
            continue
        raw = line[4:].split("\t", maxsplit=1)[0].strip()
        if raw == "/dev/null":
            continue
        if raw.startswith(("a/", "b/")):
            raw = raw[2:]
        if not raw:
            raise CodingWorkflowError("Patch contains an empty file path.")
        try:
            resolve_relative_path(config, raw)
        except CodingPathError as exc:
            raise CodingWorkflowError(f"Patch path is not allowed: {raw}") from exc
        if raw not in paths:
            paths.append(raw)
    if not paths:
        raise CodingWorkflowError("Patch must contain at least one unified diff file path.")
    return paths


async def changed_paths(worktree: Path, *, config: CodingPackConfig) -> list[str]:
    """Return bounded changed paths from Git's machine-readable status output."""

    result = await run_git(worktree, ("status", "--porcelain", "-z"), config=config)
    if result.return_code != 0:
        raise CodingWorkflowError(result.stderr.strip() or "git status failed")
    paths: list[str] = []
    for record in result.stdout.split("\0"):
        if len(record) <= _GIT_STATUS_PATH_OFFSET:
            continue
        path = record[_GIT_STATUS_PATH_OFFSET:]
        if path and path not in paths:
            paths.append(path)
    return paths

"""Apply a bounded unified diff only within an owned coding worktree."""

from __future__ import annotations

from pydantic import BaseModel, Field

from kernel.models import ToolExecutionContext, ToolResult
from sdk.packs.coding.config import get_coding_config
from sdk.packs.coding.workflow import (
    CodingWorkflowError,
    artifact_worktree,
    load_owned_artifact,
    patch_paths,
    run_git,
)

TOOL_ID = "coding_apply_patch"


class Parameters(BaseModel):
    """Structured patch application request."""

    artifact_id: str = Field(min_length=1, max_length=100)
    patch: str = Field(min_length=1, max_length=2_000_000)


async def execute(ctx: ToolExecutionContext) -> ToolResult:
    """Check and apply a unified diff, invalidating stale checks afterward."""

    params = Parameters.model_validate(ctx.parameters)
    try:
        config = get_coding_config(ctx)
        if len(params.patch.encode("utf-8")) > config.max_patch_bytes:
            raise CodingWorkflowError("Patch exceeds the configured byte limit.")
        affected_paths = patch_paths(params.patch, config=config)
        artifact = await load_owned_artifact(
            ctx,
            artifact_id=params.artifact_id,
            config=config,
        )
        worktree = artifact_worktree(artifact, config=config)
        checked = await run_git(
            worktree,
            ("apply", "--check", "--whitespace=nowarn", "-"),
            config=config,
            stdin_text=params.patch,
        )
        if checked.return_code != 0:
            raise CodingWorkflowError(checked.stderr.strip() or "git apply --check failed")
        applied = await run_git(
            worktree,
            ("apply", "--whitespace=nowarn", "-"),
            config=config,
            stdin_text=params.patch,
        )
        if applied.return_code != 0:
            raise CodingWorkflowError(applied.stderr.strip() or "git apply failed")
        artifact.revision += 1
        artifact.changed_paths = affected_paths
        artifact.checks = {}
        artifact.status = "patched"
        await ctx.code_change_artifact_store.save(artifact)
    except (CodingWorkflowError, OSError, ValueError) as exc:
        return ToolResult(tool_id=TOOL_ID, error=str(exc))

    return ToolResult(
        tool_id=TOOL_ID,
        output_text=(
            f"Applied patch to {len(affected_paths)} paths at revision {artifact.revision}."
        ),
        output_json={
            "artifact_id": artifact.artifact_id,
            "revision": artifact.revision,
            "changed_paths": affected_paths,
        },
    )

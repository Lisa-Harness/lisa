"""Create one owned feature worktree without mutating the source checkout."""

from __future__ import annotations

from pydantic import BaseModel, Field

from kernel.code_artifacts import CodeChangeArtifact
from kernel.models import ToolExecutionContext, ToolResult
from sdk.packs.coding.config import get_coding_config
from sdk.packs.coding.workflow import CodingWorkflowError, run_git

TOOL_ID = "coding_create_worktree"


class Parameters(BaseModel):
    """Structured worktree creation request."""

    summary: str = Field(default="", max_length=240)


def _branch_name(prefix: str, artifact_id: str) -> str:
    normalized = prefix if prefix.endswith("/") else f"{prefix}-"
    return f"{normalized}{artifact_id[:12]}"


async def execute(ctx: ToolExecutionContext) -> ToolResult:
    """Create and persist an isolated worktree rooted outside the source repository."""

    params = Parameters.model_validate(ctx.parameters)
    try:
        config = get_coding_config(ctx)
        source_root = config.resolve_workspace_root()
        store = ctx.code_change_artifact_store
        if store is None:
            raise CodingWorkflowError("No code change artifact store is available.")
        base = await run_git(source_root, ("rev-parse", "HEAD"), config=config)
        if base.return_code != 0:
            raise CodingWorkflowError(base.stderr.strip() or "Cannot resolve the source HEAD.")
        artifact = CodeChangeArtifact(
            workspace_id=ctx.workspace_id,
            user_id=ctx.user_id,
            user_slug=ctx.user_slug,
            surface_type=ctx.surface_type,
            surface_id=ctx.surface_id,
            session_plan_digest=ctx.session_plan_digest,
            source_root=str(source_root),
            base_sha=base.stdout.strip(),
            head_sha=base.stdout.strip(),
            branch=_branch_name(config.branch_prefix, "pending"),
            required_check_ids=list(config.required_check_ids),
            safe_summary=params.summary.strip(),
        )
        artifact.branch = _branch_name(config.branch_prefix, artifact.artifact_id)
        if artifact.branch in config.protected_branches:
            raise CodingWorkflowError(
                "Configured feature branch conflicts with a protected branch."
            )
        worktree_parent = source_root.parent / ".lisa-worktrees"
        worktree_path = (worktree_parent / artifact.artifact_id).resolve()
        try:
            worktree_path.relative_to(source_root)
        except ValueError:
            pass
        else:
            raise CodingWorkflowError("Derived worktree path must be outside the source workspace.")
        created = await run_git(
            source_root,
            ("worktree", "add", "-b", artifact.branch, str(worktree_path), artifact.base_sha),
            config=config,
        )
        if created.return_code != 0:
            raise CodingWorkflowError(created.stderr.strip() or "git worktree add failed")
        artifact.worktree_path = str(worktree_path)
        await store.create(artifact)
    except (CodingWorkflowError, OSError, ValueError) as exc:
        return ToolResult(tool_id=TOOL_ID, error=str(exc))

    return ToolResult(
        tool_id=TOOL_ID,
        output_text=f"Created worktree {artifact.artifact_id} on branch {artifact.branch}.",
        output_json={
            "artifact_id": artifact.artifact_id,
            "branch": artifact.branch,
            "worktree_path": artifact.worktree_path,
            "base_sha": artifact.base_sha,
        },
    )

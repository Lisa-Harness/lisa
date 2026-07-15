"""List direct child entries from the configured GitHub repository."""

from __future__ import annotations

import httpx
from pydantic import BaseModel, Field

from kernel.models import ToolExecutionContext, ToolResult
from lisa_pack_github.client import GitHubRepoLookupError, is_configured, list_tree

TOOL_ID = "repo_list_tree"


class Parameters(BaseModel):
    """Structured request for one bounded GitHub tree listing."""

    root: str = ""
    ref: str = ""
    max_entries: int = Field(default=40, ge=1, le=80)


async def execute(ctx: ToolExecutionContext) -> ToolResult:
    """List one repository root or subdirectory at the requested ref."""

    params = Parameters.model_validate(ctx.parameters)
    if not is_configured():
        return ToolResult(
            tool_id=TOOL_ID,
            error="GitHub repo tree listing is not configured. Set GITHUB_TOKEN and GITHUB_REPO.",
        )
    try:
        response = await list_tree(params.root, ref=params.ref, max_entries=params.max_entries)
    except (GitHubRepoLookupError, httpx.HTTPError) as exc:
        return ToolResult(tool_id=TOOL_ID, error=str(exc))

    return ToolResult(
        tool_id=TOOL_ID,
        output_text=f"Listed {len(response.entries)} entries under {response.root or '/'}.",
        output_json=response.model_dump(mode="json"),
    )
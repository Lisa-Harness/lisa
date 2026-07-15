"""Search the configured GitHub repository's default branch."""

from __future__ import annotations

import httpx
from pydantic import BaseModel, Field

from kernel.models import ToolExecutionContext, ToolResult
from lisa_pack_github.client import GitHubRepoLookupError, is_configured, search_code

TOOL_ID = "repo_search_code"


class Parameters(BaseModel):
    """Structured request for an indexed GitHub code search."""

    query: str = ""
    path_hints: list[str] = Field(default_factory=list)
    max_results: int = Field(default=5, ge=1, le=8)
    ref: str = ""


async def execute(ctx: ToolExecutionContext) -> ToolResult:
    """Search candidate files without representing search as arbitrary-ref evidence."""

    params = Parameters.model_validate(ctx.parameters)
    query = (params.query or ctx.query).strip()
    if not query:
        return ToolResult(tool_id=TOOL_ID, error="Repo code search requires a query.")
    if not is_configured():
        return ToolResult(
            tool_id=TOOL_ID,
            error="GitHub repo search is not configured. Set GITHUB_TOKEN and GITHUB_REPO.",
        )
    try:
        response = await search_code(
            query,
            path_hints=params.path_hints,
            max_results=params.max_results,
            ref=params.ref,
        )
    except (GitHubRepoLookupError, httpx.HTTPError) as exc:
        return ToolResult(tool_id=TOOL_ID, error=str(exc))

    return ToolResult(
        tool_id=TOOL_ID,
        output_text=f"Found {len(response.matches)} code-search candidates for {query!r}.",
        output_json=response.model_dump(mode="json"),
    )
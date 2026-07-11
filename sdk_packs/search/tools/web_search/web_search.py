"""Generic web_search stub for the bundled search pack.

The bundled search pack keeps the generic ``web_search`` contract but does
not ship a public web provider implementation. Install an external provider
pack (for example ``lisa-pack-brave-search``) to replace this stub.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from kernel.models import ToolExecutionContext, ToolResult

__all__ = ["Parameters", "execute", "is_configured", "search"]

_UNAVAILABLE_MESSAGE = (
    "Web search provider is unavailable. Install lisa-pack-brave-search and enable it with "
    "/pack enable brave-search."
)


class Parameters(BaseModel):
    """Structured input parameters for the web_search tool."""

    query: str = Field(..., description="The search query to execute.")
    count: int = Field(default=5, ge=1, le=20, description="Max results per query.")


def is_configured() -> bool:
    """The bundled provider is intentionally unavailable by default."""
    return False


async def search(query: str) -> dict[str, object]:
    """Fail with a deterministic provider-not-installed message."""
    _ = query
    raise RuntimeError(_UNAVAILABLE_MESSAGE)


async def execute(ctx: ToolExecutionContext) -> ToolResult:
    """Execute the web_search tool — called by the tool registry dispatcher."""
    from pydantic import ValidationError

    try:
        params = Parameters.model_validate(ctx.parameters or {})
        query = params.query.strip()
    except ValidationError:
        query = ""
    if not query:
        # Fall back to ctx.query if the model didn't supply a query argument
        query = (ctx.query or "").strip()
    if not query:
        return ToolResult(tool_id=ctx.tool_id or "web_search", error="web_search requires a query.")

    return ToolResult(
        tool_id=ctx.tool_id or "web_search",
        error=_UNAVAILABLE_MESSAGE,
    )

"""SDK-owned domain models for MCP servers and extensions.

These models are MCP-domain contracts — they live in lisa-sdk so both the
kernel (MCP adapters) and MCP servers can import them without circular deps.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from kernel.models import ConversationSurface, MemoryScope

__all__ = [
    "WebSearchResponse",
    "WebSearchResult",
]


# ---------------------------------------------------------------------------
# Web search (Brave Search API)
# ---------------------------------------------------------------------------


class WebSearchResult(BaseModel):
    name: str
    url: str
    snippet: str = ""
    display_url: str = ""


class WebSearchResponse(BaseModel):
    query: str
    generated_at: str
    results: list[WebSearchResult] = Field(default_factory=list)

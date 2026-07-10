"""Web search tool models for the search pack.

Moved from ``lisa_sdk.models`` — the search pack owns its own response models
so it has zero SDK/kernel config linkage.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["WebSearchResponse", "WebSearchResult"]


class WebSearchResult(BaseModel):
    """One Brave Search result item."""

    name: str
    url: str
    snippet: str = ""
    display_url: str = ""


class WebSearchResponse(BaseModel):
    """Structured response from one Brave Web Search query."""

    query: str
    generated_at: str
    results: list[WebSearchResult] = Field(default_factory=list)
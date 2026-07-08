"""SDK-owned domain models for MCP servers and extensions.

These models are MCP-domain contracts — they live in lisa-sdk so both the
kernel (MCP adapters) and MCP servers can import them without circular deps.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from kernel.models import ConversationSurface, MemoryScope

__all__ = [
    "SurfCommentaryEntry",
    "SurfLinkCandidate",
    "SurfPageCapture",
    "SurfSessionState",
    "SurfTrace",
    "WebSearchResponse",
    "WebSearchResult",
]


class SurfTrace(BaseModel):
    run_at: str
    requested_url: str
    final_url: str = ""
    page_title: str = ""
    status: str = "fail"  # "pass" | "fail"
    http_status: int = 0
    duration_ms: int = 0
    used_persisted_session: bool = False
    saved_persisted_session: bool = False
    screenshot_blob_path: str = ""
    text_excerpt: str = ""
    console_errors: list[str] = Field(default_factory=list)
    error: str = ""


class SurfLinkCandidate(BaseModel):
    """One follow-on navigation candidate extracted from a surfed page.

    Stored in the durable surf workflow session so later turns can reference
    links by index or fuzzy label instead of restating a full absolute URL.
    """

    url: str
    label: str = ""


class SurfPageCapture(BaseModel):
    """Structured snapshot of one page visited during an iterative surf session.

    Produced by the browser runtime after a successful navigation and persisted
    in the surf session blob so Lisa can keep page-by-page continuity across
    turns and container restarts.
    """

    trace: SurfTrace
    artifact_prefix: str = ""
    link_candidates: list[SurfLinkCandidate] = Field(default_factory=list)


class SurfCommentaryEntry(BaseModel):
    """One assistant commentary entry tied to a surf session page state."""

    created_at: str
    page_url: str = ""
    user_query: str = ""
    commentary: str = ""


class SurfSessionState(BaseModel):
    """Durable page-by-page surf workflow state for one conversation surface.

    The surf workflow stores this object in blob storage under the state
    container so an active browsing session can survive process-local cache
    loss and container restarts.
    """

    session_id: str
    workspace_id: str = "default"
    surface_id: str
    surface_type: ConversationSurface = "personal_dm"
    memory_scope: MemoryScope = "personal"
    mode_id: str = "surf"
    created_at: str
    updated_at: str
    expires_at: str
    last_user_query: str = ""
    last_action: str = ""
    current_page_url: str = ""
    current_page_title: str = ""
    summary: str = ""
    pages: list[SurfPageCapture] = Field(default_factory=list)
    commentary: list[SurfCommentaryEntry] = Field(default_factory=list)


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

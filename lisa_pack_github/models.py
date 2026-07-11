"""GitHub extension models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GitHubStatus(BaseModel):
    collected_at: str
    status: str = "ok"  # "ok" | "degraded" | "unconfigured" | "error"
    latest_commit_sha: str = ""
    latest_commit_message: str = ""
    latest_commit_author: str = ""
    tracked_branch: str = ""
    main_workflow_status: str = "unknown"  # "success" | "failure" | "in_progress"
    open_prs: int = 0
    failing_checks: list[str] = Field(default_factory=list)
    latest_release_tag: str = ""


class GitHubCodeSearchMatch(BaseModel):
    """One read-only code-search match returned from GitHub."""

    path: str
    fragment: str = ""


class GitHubCodeSearchResponse(BaseModel):
    """Normalized response for one GitHub code-search query."""

    repo: str
    ref: str = ""
    git_sha: str = ""
    query: str = ""
    path_hints: list[str] = Field(default_factory=list)
    matches: list[GitHubCodeSearchMatch] = Field(default_factory=list)


class GitHubFileExcerpt(BaseModel):
    """One read-only file excerpt fetched from GitHub by ref."""

    repo: str
    ref: str = ""
    git_sha: str = ""
    path: str
    start_line: int = 1
    end_line: int = 1
    total_lines: int = 0
    excerpt: str = ""


class GitHubTreeEntry(BaseModel):
    """One direct child entry in a repo tree listing."""

    path: str
    entry_type: Literal["blob", "tree"] = "blob"
    size: int | None = None


class GitHubTreeResponse(BaseModel):
    """Direct-child tree listing for a repo root or subdirectory."""

    repo: str
    ref: str = ""
    git_sha: str = ""
    root: str = ""
    truncated: bool = False
    entries: list[GitHubTreeEntry] = Field(default_factory=list)

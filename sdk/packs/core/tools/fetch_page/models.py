"""Fetched-page model for the core-pack ``fetch_page`` tool.

Owned locally by the core pack so the tool has zero cross-pack config
linkage and is self-contained for externalization.
"""

from __future__ import annotations

from pydantic import BaseModel

__all__ = ["FetchedPage"]


class FetchedPage(BaseModel):
    """One fetched page rendered as plain text for model consumption."""

    url: str
    status_code: int
    title: str = ""
    text: str = ""
    content_type: str = ""
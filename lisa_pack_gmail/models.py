"""Pydantic models for the Gmail pack."""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["GmailDraftResult"]


class GmailDraftResult(BaseModel):
    """Structured result returned by the ``gmail_create_draft`` tool.

    Mirrors the :class:`~kernel.models.ToolResult` ``output_json`` payload so
    callers (operations, tests) can type the tool's structured output.
    """

    draft_id: str = Field(default="", description="Gmail draft resource id.")
    to: str = Field(default="", description="Recipient email address.")
    subject: str = Field(default="", description="Email subject line.")
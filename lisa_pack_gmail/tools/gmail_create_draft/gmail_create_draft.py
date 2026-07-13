"""``gmail_create_draft`` tool implementation.

Creates a Gmail draft by calling the Gmail REST API ``drafts.create``
endpoint with a base64url-encoded RFC 2822 message. The pack owns no token
acquisition — the host injects a valid OAuth2 access token via the
``GMAIL_TOKEN`` environment variable (see :mod:`lisa_pack_gmail.config`).
"""

from __future__ import annotations

import base64

import httpx
from pydantic import BaseModel, Field, ValidationError

from kernel.models import ToolExecutionContext, ToolResult
from lisa_pack_gmail.config import gmail_settings

__all__ = ["Parameters", "execute"]

_UNAVAILABLE_MESSAGE = (
    "Gmail is not configured. Set GMAIL_TOKEN (an OAuth2 access token with "
    "gmail.compose scope) to enable draft creation."
)


class Parameters(BaseModel):
    """Structured input parameters for the ``gmail_create_draft`` tool."""

    to: str = Field(..., description="Recipient email address.")
    subject: str = Field(..., description="Email subject line.")
    body: str = Field(..., description="Email body as plain text.")
    cc: str | None = Field(
        default=None,
        description="Optional cc addresses, comma-separated.",
    )


def _encode_raw_message(params: Parameters) -> str:
    """Encode a draft message as base64url RFC 2822, per the Gmail API."""

    lines: list[str] = [
        f"To: {params.to}",
        f"Subject: {params.subject}",
    ]
    if params.cc:
        lines.append(f"Cc: {params.cc}")
    lines.extend(["Content-Type: text/plain; charset=utf-8", "", params.body])
    raw_message = "\r\n".join(lines).encode("utf-8")
    return base64.urlsafe_b64encode(raw_message).decode("ascii")


async def execute(ctx: ToolExecutionContext) -> ToolResult:
    """Execute the ``gmail_create_draft`` tool."""

    tool_id = ctx.tool_id or "gmail_create_draft"

    try:
        params = Parameters.model_validate(ctx.parameters or {})
    except ValidationError as exc:
        return ToolResult(tool_id=tool_id, error=str(exc))

    settings = gmail_settings()
    if not settings.is_configured:
        return ToolResult(tool_id=tool_id, error=_UNAVAILABLE_MESSAGE)

    encoded = _encode_raw_message(params)
    headers = {"Authorization": f"Bearer {settings.token}"}
    payload = {"message": {"raw": encoded}}

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{settings.api_base}/users/me/drafts",
                headers=headers,
                json=payload,
            )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        return ToolResult(
            tool_id=tool_id,
            error=f"Gmail API call failed: {type(exc).__name__}: {exc}",
        )

    data = resp.json()
    draft_id = data.get("id", "")
    return ToolResult(
        tool_id=tool_id,
        output_text=f"Draft created (id: {draft_id}).",
        output_json={
            "draft_id": draft_id,
            "to": params.to,
            "subject": params.subject,
        },
    )
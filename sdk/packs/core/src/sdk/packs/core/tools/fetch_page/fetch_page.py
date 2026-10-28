"""Fetch page tool — pack-contributed internal tool.

Retrieves a URL and extracts its main visible text so research loops can
synthesize from full page content instead of search snippets alone.

The network fetch and cache belong to the session's ``fetch`` capability:
the tool calls ``fetch.fetch_url`` through ``ToolExecutionContext.mcp_tool_call``,
so ``providers.capabilities.fetch`` selects the implementation (the built-in
Python ``lisa-fetch`` server or an alternate such as the Go binary). The tool
keeps URL validation, text extraction, and its output contract. There is no
in-process fallback: a context without the MCP seam is an error.

HTML-to-text extraction uses the standard library only — no new heavy
dependency. The extractor strips tags, scripts, styles, and collapses
whitespace, which is sufficient for grounded synthesis.
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from html import unescape

from kernel.models import ToolExecutionContext, ToolResult
from pydantic import BaseModel, Field, ValidationError
from sdk.content import validate_url
from sdk.packs.core.tools.fetch_page._models import FetchedPage

__all__ = ["FETCH_TOOL_ID", "Parameters", "execute", "fetch", "is_configured"]

#: Published id of the fetch capability tool this tool delegates to.
FETCH_TOOL_ID = "fetch.fetch_url"
_TTL_SECONDS: float = 86_400.0  # fetch_page content is research-grade; 24h is fine
_DEFAULT_MAX_BYTES: int = 200_000
_MAX_TEXT_CHARS: int = 12_000
# Upper bound the tool advertises in its parameter schema. The extraction
# ceiling is applied per call, so a request above this is rejected by the
# schema before the fetch rather than silently truncated afterwards.
_MAX_CHARS_LIMIT: int = 20_000
_BLOCK_TAGS = re.compile(
    r"<(script|style|noscript|template|svg|head)\b[\s\S]*?</\1\s*>",
    re.IGNORECASE,
)
_TAG_STRIP = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_TITLE_TAG = re.compile(r"<title\b[^>]*>([\s\S]*?)</title\s*>", re.IGNORECASE)
# Both fetch implementations report an HTTP error status as
# ``HTTP <code> (<reason>) for url '<url>'``; the code is recovered from that
# message so a site refusal reads differently from a transport failure.
_STATUS_ERROR = re.compile(r"\bHTTP (\d{3})\b[^\n]*? for url '")

# Content types whose payload is not HTML or plain text. Stripping tags from
# a PDF (or any other binary) yields unreadable bytes, so these are reported
# as "not extractable" instead of being passed off as text.
_TEXTUAL_CONTENT_TYPES = ("text/", "application/json", "application/xml", "application/xhtml")


class FetchError(RuntimeError):
    """The fetch capability failed; ``status_code`` is set for HTTP refusals."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class Parameters(BaseModel):
    """Structured input parameters for the fetch_page tool."""

    url: str = Field(..., description="The absolute http(s) URL to fetch.")
    max_chars: int = Field(
        default=_MAX_TEXT_CHARS,
        ge=500,
        le=_MAX_CHARS_LIMIT,
        description=(
            "Max extracted text characters to return. Must be between "
            "500 and 20000 (values outside the range are rejected before "
            "the fetch). To read a long document, fetch it in slices "
            "rather than exceeding the cap."
        ),
    )


def _get_max_bytes() -> int:
    raw = os.environ.get("FETCH_PAGE_MAX_BYTES", "").strip()
    if not raw:
        return _DEFAULT_MAX_BYTES
    try:
        return max(1024, int(raw))
    except ValueError:
        return _DEFAULT_MAX_BYTES


def is_configured() -> bool:
    """Return True — fetch_page needs no API key, only network access.

    Kept for API symmetry with other tools so the host can gate the tool
    uniformly.
    """
    return True


def _strip_html(html: str, *, max_chars: int) -> tuple[str, str]:
    """Extract visible text and a title from raw HTML.

    Returns ``(text, title)``. The text is collapsed-whitespace plain text
    capped at ``max_chars``.
    """
    title_match = _TITLE_TAG.search(html)
    title = unescape(title_match.group(1)).strip() if title_match else ""

    body = _BLOCK_TAGS.sub("", html)
    text = _TAG_STRIP.sub("", body)
    text = unescape(text)
    text = _WS.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "\u2026"
    return text, title


async def fetch(
    ctx: ToolExecutionContext,
    url: str,
    *,
    force_refresh: bool = False,
    max_chars: int = _MAX_TEXT_CHARS,
) -> FetchedPage:
    """Fetch one URL through the session's fetch capability and extract text.

    Raises:
        FetchError: when the context has no MCP seam, the capability reports
            an error (``status_code`` set for HTTP refusals), or its payload
            is not the ``fetch_url`` JSON contract.
    """
    call = ctx.mcp_tool_call
    if call is None:
        raise FetchError(
            f"fetch_page requires the fetch capability ({FETCH_TOOL_ID}), but no MCP "
            "tool pool is attached to this tool context."
        )
    arguments: dict[str, object] = {
        "url": url,
        "ttl_seconds": _TTL_SECONDS,
        "max_bytes": _get_max_bytes(),
    }
    if force_refresh:
        arguments["force_refresh"] = True
    result = await call(FETCH_TOOL_ID, arguments)
    if result.error:
        match = _STATUS_ERROR.search(result.error)
        raise FetchError(result.error, status_code=int(match.group(1)) if match else None)
    try:
        payload = json.loads(result.output_text)
        raw_text = str(payload["text"])
        page_url = str(payload.get("url") or url)
        status_code = int(payload["status_code"])
        content_type = str(payload.get("content_type") or "")
    except (ValueError, TypeError, KeyError) as exc:
        raise FetchError(
            f"{FETCH_TOOL_ID} returned an unexpected payload: {result.output_text[:200]!r}"
        ) from exc
    text, title = _strip_html(raw_text, max_chars=max_chars)
    return FetchedPage(
        url=page_url,
        status_code=status_code,
        title=title,
        text=text,
        content_type=content_type,
    )


def _is_textual(content_type: str) -> bool:
    """Return True when the payload is HTML/text/XML/JSON we can extract from.

    A PDF or any other binary payload is not extractable by tag-stripping:
    the "text" would be unreadable bytes. Those are reported as
    not-extractable rather than passed off as prose.
    """
    ctype = (content_type or "").split(";", 1)[0].strip().lower()
    if not ctype:
        # No content type: treat as text and let the extractor decide.
        return True
    return ctype.startswith(_TEXTUAL_CONTENT_TYPES)


async def execute(ctx: ToolExecutionContext) -> ToolResult:
    """Execute the fetch_page tool — called by the tool registry dispatcher."""
    try:
        params = Parameters.model_validate(ctx.parameters or {})
        url = params.url
        max_chars = params.max_chars
    except ValidationError:
        url = (ctx.parameters or {}).get("url", "")
        max_chars = _MAX_TEXT_CHARS

    # Validate first so SSRF rejections surface as tool errors, not exceptions.
    _, reason = validate_url(url)
    if reason is not None:
        return ToolResult(tool_id=ctx.tool_id or "fetch_page", error=reason)

    try:
        page = await fetch(ctx, url, max_chars=max_chars)
    except FetchError as exc:
        if exc.status_code is not None:
            # A status error is a refusal by the site (403/429/5xx), not a
            # transport failure. Report the status plainly so the model can
            # try a different source instead of retrying the same URL.
            return ToolResult(
                tool_id=ctx.tool_id or "fetch_page",
                error=(
                    f"Fetch failed: HTTP {exc.status_code} for {url}. "
                    "The site refused this request; try a different source."
                ),
            )
        return ToolResult(tool_id=ctx.tool_id or "fetch_page", error=f"Fetch failed: {exc}")

    if not _is_textual(page.content_type):
        return ToolResult(
            tool_id=ctx.tool_id or "fetch_page",
            output_text=(
                f"No extractable text at {url} "
                f"(status {page.status_code}, type {page.content_type}). "
                "This content type is not plain text or HTML; fetch a "
                "text/HTML source instead."
            ),
            output_json={"page": page.model_dump(mode="json")},
        )

    if not page.text:
        return ToolResult(
            tool_id=ctx.tool_id or "fetch_page",
            output_text=(
                f"No extractable text at {url} "
                f"(status {page.status_code}, type {page.content_type})."
            ),
            output_json={"page": page.model_dump(mode="json")},
        )

    # Enforce the per-call bound at the tool boundary as well as inside
    # ``fetch``: the tool's contract is the advertised maximum, whatever
    # the fetch implementation returns.
    if len(page.text) > max_chars:
        page = page.model_copy(update={"text": page.text[:max_chars].rsplit(" ", 1)[0] + "\u2026"})

    timestamp = datetime.now(UTC).isoformat()
    header = f"# {page.title or url}\nSource: {url} \u2022 {page.status_code} \u2022 {timestamp}\n"
    return ToolResult(
        tool_id=ctx.tool_id or "fetch_page",
        output_text=header + page.text,
        output_json={"page": page.model_dump(mode="json")},
    )

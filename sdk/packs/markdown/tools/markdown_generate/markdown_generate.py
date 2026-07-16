"""Outline-markdown tool — structural analysis via ``markdown-it-py``.

This is the pack's "markdown library as a tool": it parses a markdown string
with ``markdown-it-py`` and returns a normalized structural outline (heading
hierarchy, ordered/unordered list depth, and code-span/code-block presence).
The operation uses the outline to validate the LLM-authored document before
persisting it.

The tool is a pure read effect — it never touches the filesystem. It accepts
the markdown text on ``ctx.parameters["markdown"]`` (or ``ctx.query`` as a
fallback) and returns the outline on ``output_json`` plus a one-line summary
on ``output_text``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from kernel.models import ToolExecutionContext, ToolResult

try:
    from markdown_it import MarkdownIt
except ImportError as exc:  # pragma: no cover - dependency declared in pyproject
    raise RuntimeError(
        "markdown_generate requires the 'markdown-it-py' package; "
        "install lisa-pack-markdown or add markdown-it-py to the environment."
    ) from exc

TOOL_ID = "markdown_generate"

# Markdown-it tokens that mark inline code or fenced code blocks.
_CODE_TOKENS = frozenset(
    {
        "code_inline",
        "code_block",
        "fence",
    }
)


class Parameters(BaseModel):
    """Structured request for the outline tool."""

    markdown: str = Field(
        default="",
        description="The markdown text to parse. Falls back to ctx.query when empty.",
    )
    max_headings: int = Field(
        default=120,
        ge=1,
        le=400,
        description="Upper bound on returned heading entries to keep output bounded.",
    )


def _heading_level(token: Any) -> int | None:
    """Return the heading level (1-6) for a ``heading_open`` token, else None."""
    tag = getattr(token, "tag", "") or ""
    if not tag.startswith("h") or len(tag) < 2:
        return None
    try:
        level = int(tag[1:])
    except ValueError:
        return None
    return level if 1 <= level <= 6 else None


def _collect_outline(tokens: list[Any], max_headings: int) -> list[dict[str, Any]]:
    """Walk the token stream and collect headings with their inline text.

    markdown-it emits headings as flat sibling triples
    (``heading_open``, ``inline``, ``heading_close``) — they are not nested.
    Only ``heading_open`` tokens start a heading; ``heading_close`` shares
    the same ``tag`` but must not be counted.
    """
    headings: list[dict[str, Any]] = []
    n = len(tokens)
    for i, token in enumerate(tokens):
        if token.type != "heading_open":
            continue
        level = _heading_level(token)
        if level is None:
            continue
        text = ""
        # The inline content token immediately follows heading_open (optionally
        # separated by other inline tokens). Scan forward to the matching
        # heading_close and gather inline content.
        for j in range(i + 1, n):
            nxt = tokens[j]
            if nxt.type == "heading_close":
                break
            if nxt.type == "inline":
                text = (text + " " + (nxt.content or "")).strip()
        headings.append({"level": level, "text": text[:200]})
        if len(headings) >= max_headings:
            break
    return headings


def _count_lists(tokens: list[Any]) -> dict[str, int]:
    """Return ordered/unordered list counts and max nesting depth."""
    ordered = 0
    unordered = 0
    max_depth = 0
    depth = 0
    for token in tokens:
        t = token.type
        if t == "ordered_list_open":
            ordered += 1
            depth += 1
            max_depth = max(max_depth, depth)
        elif t == "bullet_list_open":
            unordered += 1
            depth += 1
            max_depth = max(max_depth, depth)
        elif t in {"ordered_list_close", "bullet_list_close"}:
            depth = max(0, depth - 1)
    return {"ordered": ordered, "unordered": unordered, "max_depth": max_depth}


def _count_code(tokens: list[Any]) -> dict[str, int]:
    """Count fenced code blocks and inline code spans."""
    fences = 0
    inline = 0
    for token in tokens:
        t = token.type
        if t == "fence" or t == "code_block":
            fences += 1
        elif t == "code_inline":
            inline += 1
    return {"fenced_blocks": fences, "inline_spans": inline}


def _outline_of(markdown: str, max_headings: int) -> dict[str, Any]:
    """Parse ``markdown`` and return a bounded structural outline dict."""
    md = MarkdownIt("commonmark", {"html": False}).enable("table")
    tokens = md.parse(markdown or "")
    headings = _collect_outline(tokens, max_headings)
    lists = _count_lists(tokens)
    code = _count_code(tokens)
    return {
        "heading_count": len(headings),
        "headings": headings,
        "lists": lists,
        "code": code,
        "has_content": bool(markdown.strip()),
    }


async def execute(ctx: ToolExecutionContext) -> ToolResult:
    """Parse the supplied markdown and return a structural outline."""
    params = Parameters.model_validate(ctx.parameters)
    markdown = params.markdown or ctx.query or ""
    if not markdown.strip():
        return ToolResult(
            tool_id=TOOL_ID,
            error="markdown_generate received empty markdown.",
        )
    try:
        outline = _outline_of(markdown, params.max_headings)
    except Exception as exc:  # pylint: disable=broad-except
        return ToolResult(tool_id=TOOL_ID, error=f"markdown parse failed: {exc}")

    summary = (
        f"Outline: {outline['heading_count']} headings, "
        f"{outline['lists']['ordered']} ordered / {outline['lists']['unordered']} unordered lists, "
        f"{outline['code']['fenced_blocks']} fenced code blocks."
    )
    return ToolResult(
        tool_id=TOOL_ID,
        output_text=summary,
        output_json=outline,
    )

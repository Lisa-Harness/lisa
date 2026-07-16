"""Tests for the markdown_generate tool (markdown-it-py backed)."""

from __future__ import annotations

import pytest

from kernel.models import ToolExecutionContext, ToolResult
from sdk.packs.markdown.tools.markdown_generate.markdown_generate import (
    TOOL_ID,
    execute,
)


def _ctx(markdown: str) -> ToolExecutionContext:
    return ToolExecutionContext(
        tool_id=TOOL_ID,
        mode_id="markdown",
        parameters={"markdown": markdown},
    )


@pytest.mark.asyncio
async def test_outlines_headings_and_lists() -> None:
    md = (
        "# Title\n"
        "Intro paragraph.\n"
        "## Section A\n"
        "- item 1\n"
        "- item 2\n"
        "### Subsection\n"
        "1. first\n"
        "2. second\n"
        "## Section B\n"
        "```python\nprint('hi')\n```\n"
    )
    result = await execute(_ctx(md))
    assert isinstance(result, ToolResult)
    assert result.error == ""
    outline = result.output_json
    assert outline["heading_count"] == 4
    texts = [h["text"] for h in outline["headings"]]
    assert "Title" in texts
    assert "Section A" in texts
    assert "Subsection" in texts
    assert outline["lists"]["unordered"] == 1
    assert outline["lists"]["ordered"] == 1
    assert outline["code"]["fenced_blocks"] == 1


@pytest.mark.asyncio
async def test_empty_markdown_returns_error() -> None:
    result = await execute(_ctx("   "))
    assert result.error != ""
    assert "empty" in result.error.lower()


@pytest.mark.asyncio
async def test_no_headings_yields_zero() -> None:
    result = await execute(_ctx("Just a paragraph with no headings.\n"))
    assert result.error == ""
    assert result.output_json["heading_count"] == 0


@pytest.mark.asyncio
async def test_falls_back_to_query_when_markdown_param_absent() -> None:
    ctx = ToolExecutionContext(tool_id=TOOL_ID, mode_id="markdown", query="# Q\nBody")
    result = await execute(ctx)
    assert result.error == ""
    assert result.output_json["heading_count"] == 1


@pytest.mark.asyncio
async def test_summary_text_includes_counts() -> None:
    md = "# H1\n## H2\n- a\n- b\n```python\nx\n```\n"
    result = await execute(_ctx(md))
    assert "2 headings" in result.output_text
    assert "1 fenced" in result.output_text
"""Catalog discovery test for the markdown pack."""

from __future__ import annotations

import json

import sdk.packs.markdown as markdown
from sdk.content import (
    build_mode_catalog_json,
    build_operation_catalog_json,
    build_tool_catalog_json,
)


def test_forge_document_discovered() -> None:
    root = markdown.get_content_root() / "operations"
    catalog = json.loads(build_operation_catalog_json([root]))
    assert "markdown_forge_document" in catalog
    op = catalog["markdown_forge_document"]
    assert op["description"]
    assert "markdown" in op["mode_allowlist"]


def test_forge_document_tool_ids_declared() -> None:
    root = markdown.get_content_root() / "operations"
    catalog = json.loads(build_operation_catalog_json([root]))
    tool_ids = set(catalog["markdown_forge_document"]["tool_ids"])
    assert {
        "markdown_generate",
        "write_file",
        "read_file",
        "memory.search",
    } <= tool_ids


def test_outline_markdown_tool_discovered() -> None:
    root = markdown.get_content_root() / "tools"
    catalog = json.loads(build_tool_catalog_json([root]))
    assert "markdown_generate" in catalog
    tool = catalog["markdown_generate"]
    assert tool["description"]
    assert tool["pack_id"] == "lisa-pack-markdown"


def test_markdown_mode_exposes_tools_directly() -> None:
    """The mode exposes storage/memory/outline tools, not the operation."""
    root = markdown.get_content_root() / "modes"
    catalog = json.loads(build_mode_catalog_json([root]))
    mode = catalog["markdown"]
    assert mode["mode_kind"] == "chat"
    assert mode["allowed_operation_ids"] == []
    assert set(mode["allowed_tool_ids"]) >= {
        "markdown_generate",
        "write_file",
        "read_file",
        "memory.search",
    }
"""Catalog discovery test for the docforge pack."""

from __future__ import annotations

import json

import sdk.packs.docforge as docforge
from sdk.content import (
    build_mode_catalog_json,
    build_operation_catalog_json,
    build_tool_catalog_json,
)


def test_forge_document_discovered() -> None:
    root = docforge.get_content_root() / "operations"
    catalog = json.loads(build_operation_catalog_json([root]))
    assert "docforge_forge_document" in catalog
    op = catalog["docforge_forge_document"]
    assert op["description"]
    assert "docforge" in op["mode_allowlist"]


def test_forge_document_tool_ids_declared() -> None:
    root = docforge.get_content_root() / "operations"
    catalog = json.loads(build_operation_catalog_json([root]))
    tool_ids = set(catalog["docforge_forge_document"]["tool_ids"])
    assert {
        "docforge_outline_markdown",
        "write_file",
        "read_file",
        "memory.search",
    } <= tool_ids


def test_outline_markdown_tool_discovered() -> None:
    root = docforge.get_content_root() / "tools"
    catalog = json.loads(build_tool_catalog_json([root]))
    assert "docforge_outline_markdown" in catalog
    tool = catalog["docforge_outline_markdown"]
    assert tool["description"]
    assert tool["pack_id"] == "lisa-pack-docforge"


def test_docforge_mode_exposes_tools_directly() -> None:
    """The mode exposes storage/memory/outline tools, not the operation."""
    root = docforge.get_content_root() / "modes"
    catalog = json.loads(build_mode_catalog_json([root]))
    mode = catalog["docforge"]
    assert mode["mode_kind"] == "chat"
    assert mode["allowed_operation_ids"] == []
    assert set(mode["allowed_tool_ids"]) >= {
        "docforge_outline_markdown",
        "write_file",
        "read_file",
        "memory.search",
    }
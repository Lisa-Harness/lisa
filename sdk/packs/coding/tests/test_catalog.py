"""Catalog tests for the bundled coding pack."""

from __future__ import annotations

import json

from sdk.content import (
    build_mode_catalog_json,
    build_tool_catalog_json,
    resolve_mode_roots,
    resolve_tool_roots,
)
from sdk.packs.coding import get_content_root


def test_coding_pack_catalog_exposes_read_tools_and_workflow_mode() -> None:
    """The pack owns a catalog-native coding mode and bounded read tools."""

    root = get_content_root()
    tool_catalog = json.loads(build_tool_catalog_json(resolve_tool_roots([root])))
    assert set(tool_catalog) >= {
        "coding_list_directory",
        "coding_find_files",
        "coding_read_file",
        "coding_search_code",
        "coding_outline_source",
    }
    assert tool_catalog["coding_read_file"]["pack_id"] == "lisa-pack-coding"

    mode_catalog = json.loads(build_mode_catalog_json(resolve_mode_roots([root])))
    assert mode_catalog["coding"]["tools_authoritative"] is True
    assert "coding_apply_patch" in mode_catalog["coding"]["allowed_tool_ids"]

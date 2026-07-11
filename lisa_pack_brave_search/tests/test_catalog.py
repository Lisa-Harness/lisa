from __future__ import annotations

import json

from sdk.lisa_content._catalog import build_tool_catalog_json, resolve_tool_roots

from lisa_pack_brave_search import get_content_root


def test_brave_pack_web_search_tool_discovered() -> None:
    roots = resolve_tool_roots([get_content_root()])
    assert roots, "No tools/ directory found in brave provider pack"

    catalog = json.loads(build_tool_catalog_json(roots))

    assert "web_search" in catalog
    tool = catalog["web_search"]
    assert tool["tool_id"] == "web_search"
    assert tool["module_path"].startswith("lisa_pack_brave_search.tools.")

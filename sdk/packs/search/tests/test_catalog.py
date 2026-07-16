"""Tests for the search pack's catalog discovery.

Verifies that the ``lisa-content`` server discovers the web_search tool,
the two operations, and the three modes (discuss, search, thinkabout) from
the search pack's content root.
"""

from __future__ import annotations

import json

from sdk.content import (
    build_mode_catalog_json,
    build_tool_catalog_json,
    resolve_mode_roots,
    resolve_tool_roots,
)
from sdk.packs.search import get_content_root


def test_web_search_tool_discovered() -> None:
    """The web_search tool is discovered from the pack's tools/ directory."""
    roots = resolve_tool_roots([get_content_root()])
    assert roots, "No tools/ directory found in search pack"

    catalog_json = build_tool_catalog_json(roots)
    catalog = json.loads(catalog_json)

    assert "web_search" in catalog
    tool = catalog["web_search"]
    assert tool["tool_id"] == "web_search"
    assert tool["description"]
    assert tool["module_path"]
    assert tool["module_path"].startswith("sdk.packs.search.tools.")


def test_search_pack_modes_discovered() -> None:
    """The search pack owns discuss, search, and thinkabout modes.

    Discuss was absorbed from the core pack so each conversational/research
    mode is co-located with its enrichment operations in the search pack.
    """
    roots = resolve_mode_roots([get_content_root()])
    assert roots, "No modes/ directory found in search pack"

    catalog = json.loads(build_mode_catalog_json(roots))

    assert set(catalog) >= {"discuss", "search", "thinkabout"}, catalog.keys()

    discuss = catalog["discuss"]
    assert discuss["mode_kind"] == "chat"
    assert discuss["pre_turn_operation_ids"] == ["discussion_briefing"]
    assert discuss["command_pattern"].startswith("^(?:/)?discuss")


def test_search_pack_models_importable() -> None:
    """The pack's response models are importable via the new module path."""
    from sdk.packs.search.models import WebSearchResponse, WebSearchResult
    from sdk.packs.search.operations.research.web_search import WebSearchStructuredOutput
    from sdk.packs.search.operations.research.discussion_briefing import (
        DiscussionBriefingStructuredOutput,
    )

    assert WebSearchResponse is not None
    assert WebSearchResult is not None
    assert WebSearchStructuredOutput is not None
    assert DiscussionBriefingStructuredOutput is not None

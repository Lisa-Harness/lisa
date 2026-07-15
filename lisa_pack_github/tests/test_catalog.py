"""Tests for the GitHub pack's catalog discovery.

Verifies that the content root is importable, modes are discoverable,
operations are discoverable, and operation allowlists match discovered modes.
"""

from __future__ import annotations

import json

from lisa_pack_github import get_content_root
from sdk.content._catalog import (
    build_mode_catalog_json,
    build_operation_catalog_json,
    build_tool_catalog_json,
    resolve_mode_roots,
    resolve_operation_roots,
    resolve_tool_roots,
)


def test_github_pack_content_root_is_directory() -> None:
    """The pack's content root must be a real directory."""
    root = get_content_root()
    assert root.is_dir(), f"Content root {root} is not a directory"


def test_github_mode_discovered() -> None:
    """The github mode is discovered from the pack's modes/ directory."""
    roots = resolve_mode_roots([get_content_root()])
    assert roots, "No modes/ directory found in github pack"

    catalog = json.loads(build_mode_catalog_json(roots))
    assert "github" in catalog, f"github mode not in catalog: {list(catalog)}"

    mode = catalog["github"]
    assert mode["mode_kind"] == "action"
    assert mode["entry_operation_ids"] == ["github_status"]
    assert mode["command_pattern"].startswith("^(?:/)?github")


def test_github_operation_discovered() -> None:
    """The github_status operation is discovered from the pack's operations/ directory."""
    roots = resolve_operation_roots([get_content_root()])
    assert roots, "No operations/ directory found in github pack"

    catalog = json.loads(build_operation_catalog_json(roots))
    assert "github_status" in catalog, f"github_status not in catalog: {list(catalog)}"

    op = catalog["github_status"]
    assert op["operation_id"] == "github_status"
    assert op["mode_allowlist"] == ["github"]


def test_repo_navigation_catalog_is_discovered() -> None:
    """The GitHub pack owns the provider-neutral repository navigation flow."""

    root = get_content_root()
    mode_catalog = json.loads(build_mode_catalog_json(resolve_mode_roots([root])))
    operation_catalog = json.loads(build_operation_catalog_json(resolve_operation_roots([root])))
    tool_catalog = json.loads(build_tool_catalog_json(resolve_tool_roots([root])))

    assert mode_catalog["repo"]["entry_operation_ids"] == ["navigate_repo"]
    assert operation_catalog["navigate_repo"]["tool_ids"] == [
        "repo_list_tree",
        "repo_search_code",
        "repo_read_file",
        "repo_lookup",
    ]
    assert set(tool_catalog) >= {
        "repo_list_tree",
        "repo_search_code",
        "repo_read_file",
        "repo_lookup",
    }
    assert tool_catalog["repo_lookup"]["pack_id"] == "lisa-pack-github"
    assert tool_catalog["repo_lookup"]["effect"] == "read"
    assert tool_catalog["github_open_pull_request"]["approval_required"] is True
    assert tool_catalog["github_open_pull_request"]["effect"] == "external_write"


def test_github_operation_allowlist_matches_modes() -> None:
    """Every operation's mode_allowlist entries must exist in the mode catalog."""
    mode_roots = resolve_mode_roots([get_content_root()])
    op_roots = resolve_operation_roots([get_content_root()])
    if not mode_roots or not op_roots:
        return

    mode_catalog = json.loads(build_mode_catalog_json(mode_roots))
    op_catalog = json.loads(build_operation_catalog_json(op_roots))

    for op_id, op in op_catalog.items():
        for mode_id in op.get("mode_allowlist", []):
            assert mode_id in mode_catalog, (
                f"Operation {op_id} references mode '{mode_id}' "
                f"which is not in the mode catalog: {list(mode_catalog)}"
            )
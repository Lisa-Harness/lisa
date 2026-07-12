"""Tests for the Cloudflare pack's catalog discovery.

Verifies that the content root is importable, modes are discoverable,
operations are discoverable, and operation allowlists match discovered modes.
"""

from __future__ import annotations

import json

from lisa_pack_cloudflare import get_content_root
from sdk.content._catalog import (
    build_mode_catalog_json,
    build_operation_catalog_json,
    resolve_mode_roots,
    resolve_operation_roots,
)


def test_cloudflare_pack_content_root_is_directory() -> None:
    """The pack's content root must be a real directory."""
    root = get_content_root()
    assert root.is_dir(), f"Content root {root} is not a directory"


def test_cloudflare_mode_discovered() -> None:
    """The cloudflare mode is discovered from the pack's modes/ directory."""
    roots = resolve_mode_roots([get_content_root()])
    assert roots, "No modes/ directory found in cloudflare pack"

    catalog = json.loads(build_mode_catalog_json(roots))
    assert "cloudflare" in catalog, f"cloudflare mode not in catalog: {list(catalog)}"

    mode = catalog["cloudflare"]
    assert mode["mode_kind"] == "action"
    assert mode["entry_operation_ids"] == ["cloudflare_status"]
    assert mode["command_pattern"].startswith("^(?:/)?cloudflare")


def test_cloudflare_operation_discovered() -> None:
    """The cloudflare_status operation is discovered from the pack's operations/ directory."""
    roots = resolve_operation_roots([get_content_root()])
    assert roots, "No operations/ directory found in cloudflare pack"

    catalog = json.loads(build_operation_catalog_json(roots))
    assert "cloudflare_status" in catalog, f"cloudflare_status not in catalog: {list(catalog)}"

    op = catalog["cloudflare_status"]
    assert op["operation_id"] == "cloudflare_status"
    assert op["mode_allowlist"] == ["cloudflare"]


def test_cloudflare_operation_allowlist_matches_modes() -> None:
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
"""Tests for the Stripe pack's catalog discovery.

Verifies that the content root is importable, modes are discoverable,
operations are discoverable, operation allowlists match discovered modes,
and the anomaly detector module is importable.
"""

from __future__ import annotations

import json

from lisa_pack_stripe import get_content_root
from sdk.lisa_content._catalog import (
    build_mode_catalog_json,
    build_operation_catalog_json,
    resolve_mode_roots,
    resolve_operation_roots,
)


def test_stripe_pack_content_root_is_directory() -> None:
    """The pack's content root must be a real directory."""
    root = get_content_root()
    assert root.is_dir(), f"Content root {root} is not a directory"


def test_stripe_mode_discovered() -> None:
    """The stripe mode is discovered from the pack's modes/ directory."""
    roots = resolve_mode_roots([get_content_root()])
    assert roots, "No modes/ directory found in stripe pack"

    catalog = json.loads(build_mode_catalog_json(roots))
    assert "stripe" in catalog, f"stripe mode not in catalog: {list(catalog)}"

    mode = catalog["stripe"]
    assert mode["mode_kind"] == "action"
    assert mode["entry_operation_ids"] == ["stripe_status"]
    assert mode["command_pattern"].startswith("^(?:/)?stripe")


def test_stripe_operation_discovered() -> None:
    """The stripe_status operation is discovered from the pack's operations/ directory."""
    roots = resolve_operation_roots([get_content_root()])
    assert roots, "No operations/ directory found in stripe pack"

    catalog = json.loads(build_operation_catalog_json(roots))
    assert "stripe_status" in catalog, f"stripe_status not in catalog: {list(catalog)}"

    op = catalog["stripe_status"]
    assert op["operation_id"] == "stripe_status"
    assert op["mode_allowlist"] == ["stripe"]


def test_stripe_operation_allowlist_matches_modes() -> None:
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


def test_stripe_processing_loop_importable() -> None:
    """The Stripe processing loop module is importable and exposes the loop."""
    from lisa_pack_stripe.processing import StripeProcessingLoop  # noqa: PLC0415  — lazy import for boundary test

    loop = StripeProcessingLoop()
    assert loop.loop_id == "stripe"
    assert callable(loop.process)
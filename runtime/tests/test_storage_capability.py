"""The model-facing storage capability comes directly from ``lisa-storage``."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from kernel.mcp_launch.server_manifest import load_mcp_server_manifest
from mcp_server.storage.server import mcp as storage_mcp

from runtime import mcp_pool

_HOST_MANIFEST = Path(mcp_pool.__file__).with_name("mcp_servers.yaml")


def _storage_spec() -> Any:
    specs = [
        spec
        for spec in load_mcp_server_manifest(_HOST_MANIFEST)
        if spec.server_id == "lisa-storage"
    ]
    assert len(specs) == 1
    return specs[0]


def test_host_manifest_exposes_storage_tools_directly() -> None:
    """File moves stay in the storage capability, with no pack wrapper."""
    spec = _storage_spec()
    served = {tool.name for tool in asyncio.run(storage_mcp.list_tools())}
    ambient = mcp_pool.resolve_default_mcp_tool_ids(
        host_specs=[spec],
        pack_specs=[],
        user_specs=[],
    )

    assert set(spec.capability_tools) == served
    assert "move_file" in spec.default_available_tools
    assert "storage.move_file" in ambient
    assert "storage.move_directory" not in ambient

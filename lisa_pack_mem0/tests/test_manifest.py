from __future__ import annotations

from kernel.mcp_launch.server_manifest import load_mcp_server_manifest

from lisa_pack_mem0 import get_content_root


def test_mem0_pack_manifest_loads_and_owns_memory() -> None:
    root = get_content_root()
    specs, memory = load_mcp_server_manifest(root / "mcp_servers.yaml")

    assert memory.default_backend == "mem0"
    assert any(backend.id == "mem0" for backend in memory.backends)
    assert any(spec.server_id == "mem0" for spec in specs)

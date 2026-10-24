"""Capability-based MCP server selection tests.

Covers the compile-time resolution that lets built-in (host-manifest)
Python servers and pack-contributed alternates (e.g. the Go Connector SDK
binaries) compete for a capability, with the winner publishing identical
model-facing ids regardless of implementation:

- explicit session selection (``providers.capabilities.<name>``) wins;
- no selection → the built-in candidate wins (loading a pack alone never
  changes the tool ids);
- no built-in → a unique candidate wins; multiple raise;
- the winner keeps its real server_id (in-process + scoped dispatch key
  on it) while the catalog publishes ``<capability>.<tool>`` ids;
- ``tool_map`` publishes canonical names over native wire names.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from kernel.mcp_launch.policy import MCPServerRegistry, MCPServerSpec, RuntimeKind

from runtime import capability_servers, mcp_pool


def _spec(server_id: str, capability: str | None = None, **kw) -> MCPServerSpec:
    default_available_tools = kw.pop("default_available_tools", ())
    capability_tools = kw.pop(
        "capability_tools",
        tuple(default_available_tools) or (("example",) if capability else ()),
    )
    capability_optional_tools = kw.pop("capability_optional_tools", ())
    return MCPServerSpec(
        server_id=server_id,
        capability=capability,
        capability_tools=capability_tools,
        capability_optional_tools=capability_optional_tools,
        default_available_tools=default_available_tools,
        runtime_kind=RuntimeKind.BINARY,
        command_path=kw.pop("command_path", f"/bin/{server_id}"),
        **kw,
    )


def _candidate(
    server_id: str,
    capability: str,
    source: str = "pack",
    pack_slug: str | None = None,
    **kw,
) -> capability_servers.CapabilityCandidate:
    return capability_servers.CapabilityCandidate(
        capability=capability,
        spec=_spec(server_id, capability, **kw),
        source=source,
        pack_slug=pack_slug,
        manifest_path="/nowhere",
    )


# ── select_capability_owner ────────────────────────────────────────


def test_explicit_selection_wins_over_builtin() -> None:
    builtin = _candidate("lisa-storage", "storage", source="builtin")
    go = _candidate("lisa-storage-go", "storage", pack_slug="go-servers")
    selection = capability_servers.select_capability_owner(
        "storage", [builtin, go], selected_provider="go-servers"
    )
    assert selection.winner is go
    assert selection.losers == (builtin,)


def test_no_selection_prefers_builtin() -> None:
    builtin = _candidate("lisa-storage", "storage", source="builtin")
    go = _candidate("lisa-storage-go", "storage", pack_slug="go-servers")
    selection = capability_servers.select_capability_owner("storage", [builtin, go])
    assert selection.winner is builtin
    assert selection.losers == (go,)


def test_unique_candidate_wins_without_builtin() -> None:
    go = _candidate("lisa-thread-go", "thread", pack_slug="go-servers")
    selection = capability_servers.select_capability_owner("thread", [go])
    assert selection.winner is go


def test_contested_without_builtin_raises() -> None:
    a = _candidate("a-server", "fetch", pack_slug="pack-a")
    b = _candidate("b-server", "fetch", pack_slug="pack-b")
    with pytest.raises(capability_servers.CapabilitySelectionError, match="providers"):
        capability_servers.select_capability_owner("fetch", [a, b])


def test_selection_matching_server_id() -> None:
    """Explicit selections may name the server id, not only the pack."""
    go = _candidate("lisa-storage-go", "storage", pack_slug="go-servers")
    builtin = _candidate("lisa-storage", "storage", source="builtin")
    selection = capability_servers.select_capability_owner(
        "storage", [builtin, go], selected_provider="lisa-storage-go"
    )
    assert selection.winner is go


def test_selection_without_match_raises() -> None:
    builtin = _candidate("lisa-storage", "storage", source="builtin")
    with pytest.raises(capability_servers.CapabilitySelectionError, match="does not match"):
        capability_servers.select_capability_owner(
            "storage", [builtin], selected_provider="nonexistent"
        )


def test_two_builtins_raise() -> None:
    a = _candidate("lisa-storage", "storage", source="builtin")
    b = _candidate("other-storage", "storage", source="builtin")
    with pytest.raises(capability_servers.CapabilitySelectionError, match="built-in"):
        capability_servers.select_capability_owner("storage", [a, b])


def test_winning_spec_keeps_real_server_id() -> None:
    """The winner registers under its real id; publishing is a catalog concern."""
    go = _candidate("lisa-storage-go", "storage")
    spec = capability_servers.winning_spec_as_capability(go)
    assert spec.server_id == "lisa-storage-go"
    assert spec.capability == "storage"


# ── _resolve_capability_owners (registry integration) ──────────────


def test_resolve_registers_winner_only() -> None:
    registry = MCPServerRegistry()
    builtin = _candidate("lisa-storage", "storage", source="builtin")
    go = _candidate("lisa-storage-go", "storage", pack_slug="go-servers")
    mcp_pool._resolve_capability_owners(registry, [builtin, go], selected_capability_providers={})
    assert [spec.server_id for spec in registry.all_specs()] == ["lisa-storage"]
    assert registry.get("lisa-storage") is not None
    assert registry.get("lisa-storage-go") is None


def test_resolve_honors_explicit_selection() -> None:
    registry = MCPServerRegistry()
    builtin = _candidate("lisa-storage", "storage", source="builtin")
    go = _candidate("lisa-storage-go", "storage", pack_slug="go-servers")
    mcp_pool._resolve_capability_owners(
        registry, [builtin, go], selected_capability_providers={"storage": "go-servers"}
    )
    assert [spec.server_id for spec in registry.all_specs()] == ["lisa-storage-go"]


def test_resolve_rejects_different_capability_contracts() -> None:
    builtin = _candidate(
        "lisa-storage",
        "storage",
        source="builtin",
        capability_tools=("read_file",),
    )
    go = _candidate(
        "lisa-storage-go",
        "storage",
        pack_slug="go-servers",
        capability_tools=("read_file", "write_file"),
    )
    with pytest.raises(
        capability_servers.CapabilitySelectionError, match="different capability_tools"
    ):
        mcp_pool._resolve_capability_owners(
            MCPServerRegistry(),
            [builtin, go],
            selected_capability_providers={},
        )


def test_resolve_rejects_different_ambient_tool_contracts() -> None:
    builtin = _candidate(
        "lisa-storage",
        "storage",
        source="builtin",
        capability_tools=("read_file", "write_file"),
        default_available_tools=("read_file",),
    )
    go = _candidate(
        "lisa-storage-go",
        "storage",
        pack_slug="go-servers",
        capability_tools=("read_file", "write_file"),
        default_available_tools=("write_file",),
    )
    with pytest.raises(
        capability_servers.CapabilitySelectionError, match="different default_available_tools"
    ):
        mcp_pool._resolve_capability_owners(
            MCPServerRegistry(),
            [builtin, go],
            selected_capability_providers={},
        )


def test_declare_server_or_candidate_collects_capability() -> None:
    registry = MCPServerRegistry()
    candidates: list[capability_servers.CapabilityCandidate] = []
    plain = _spec("lisa-guard")
    claiming = _spec("lisa-storage", "storage")
    mcp_pool._declare_server_or_candidate(
        registry, plain, candidates, source="builtin", manifest_path="m"
    )
    mcp_pool._declare_server_or_candidate(
        registry, claiming, candidates, source="builtin", manifest_path="m"
    )
    assert registry.get("lisa-guard") is not None
    assert registry.get("lisa-storage") is None
    assert [c.spec.server_id for c in candidates] == ["lisa-storage"]
    assert candidates[0].source == "builtin"


# ── compile integration (capability flows into MCPClientConfig) ─────


def test_compiled_config_carries_capability_and_tool_map(tmp_path: Path) -> None:
    from kernel.mcp_launch.compiler import compile_registry

    registry = MCPServerRegistry()
    registry.register(
        MCPServerSpec(
            server_id="lisa-memory",
            capability="memory",
            capability_tools=("add", "search"),
            capability_optional_tools=("get", "retire"),
            runtime_kind=RuntimeKind.BINARY,
            command_path="/bin/lisa-memory",
            tool_map={"add": "add_memory", "search": "search_memory"},
        )
    )
    configs = compile_registry(registry, data_dir=str(tmp_path))
    config = next(c for c in configs if c.server_id == "lisa-memory")
    assert config.capability == "memory"
    assert config.tool_map == {"add": "add_memory", "search": "search_memory"}


# ── ambient id resolution (published prefix) ───────────────────────


def test_collect_default_tool_ids_uses_capability_prefix() -> None:
    builtin = _spec("lisa-storage", "storage", default_available_tools=("read_file", "list_files"))
    go = _spec("lisa-storage-go", "storage", default_available_tools=("read_file", "list_files"))
    ids = mcp_pool._collect_default_tool_ids([builtin, go])
    assert ids == {"storage.read_file", "storage.list_files"}


def test_collect_default_tool_ids_plain_server_unchanged() -> None:
    plain = _spec("lisa-guard", None, default_available_tools=("check",))
    ids = mcp_pool._collect_default_tool_ids([plain])
    assert ids == {"lisa-guard.check"}

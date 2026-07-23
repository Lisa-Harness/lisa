"""Tests for memory alias registration — generic ``memory.*`` router only.

Verifies that:
- Only the three generic ``memory.*`` aliases are registered (no per-backend
  explicit surfaces — native tools are available by their FQN).
- When no backends are declared, the implicit lisa_memory backend is the
  default for the generic router.
- mem0 arg maps translate ``content`` → ``text`` while lisa_memory passes
  args through unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hosts.core.memory_aliases import register_memory_aliases


@dataclass
class FakeSpec:
    """Mirrors MemoryBackendSpec's shape (id, kind, tool_names — no server_id).

    The server id is derived from the backend id at registration time:
    lisa_memory → lisa-memory; everything else uses its own id.
    """

    id: str = "lisa_memory"
    kind: str = "stdio"
    tool_names: dict[str, str] = field(
        default_factory=lambda: {
            "add": "add_memory",
            "search": "search_memory",
            "list": "list_memories",
        }
    )
    result_maps: dict[str, str] = field(default_factory=dict)
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    agent_id: str = "lisa"


@dataclass
class FakeMemoryConfig:
    default_backend: str | None = None
    backends: tuple[Any, ...] = ()


class FakePool:
    """Captures synthetic tool registrations without a real MCP connection."""

    def __init__(self) -> None:
        self.registered: list[dict[str, Any]] = []
        self.ambient: dict[str, Any] = {}

    def set_ambient_resolver(self, key: str, resolver: Any) -> None:
        self.ambient[key] = resolver

    def add_synthetic_tools(self, tools: list[Any]) -> None:
        for tool in tools:
            # SyntheticToolSpec has name, description, input_schema, handler.
            # target_tool and target_server_id are captured in the handler's
            # closure (by mcp_alias_tool), so we inspect __closure__ + co_freevars
            # to extract them for test assertions.
            name = getattr(tool, "name", None)
            handler = getattr(tool, "handler", None)
            closure_vars: dict[str, Any] = {}
            if handler is not None and hasattr(handler, "__closure__") and handler.__closure__:
                freevars = handler.__code__.co_freevars
                for var_name, cell in zip(freevars, handler.__closure__):
                    closure_vars[var_name] = cell.cell_contents
            self.registered.append({
                "name": name,
                "target_tool": closure_vars.get("target_tool"),
                "target_server_id": closure_vars.get("target_server_id"),
                "arg_map": closure_vars.get("arg_map"),
                "result_map": closure_vars.get("result_map"),
                "description": getattr(tool, "description", ""),
            })


def _registered_names(pool: FakePool) -> list[str]:
    return [r["name"] for r in pool.registered]


class TestImplicitLisaMemoryOnly:
    """When no backends are declared, only the generic router is registered."""

    def test_registers_only_generic_router(self) -> None:
        pool = FakePool()
        register_memory_aliases(
            pool,
            memory_config=FakeMemoryConfig(),
            memory_config_pack_slug=None,
            user_slug="alice",
            current_mode_id="default",
            app_id="test",
        )
        names = _registered_names(pool)
        # Only the generic router — no per-backend synthetics.
        assert names == ["memory.add", "memory.search", "memory.list"]
        # The generic router targets the implicit lisa_memory backend.
        add = next(r for r in pool.registered if r["name"] == "memory.add")
        assert add["target_tool"] == "add_memory"
        assert add["target_server_id"] == "lisa-memory"
        assert add["arg_map"] is None
        assert add["result_map"] is None

    def test_ambient_resolvers_set(self) -> None:
        pool = FakePool()
        register_memory_aliases(
            pool,
            memory_config=FakeMemoryConfig(),
            memory_config_pack_slug=None,
            user_slug="alice",
            current_mode_id="default",
            app_id="test",
        )
        assert pool.ambient["user_slug"]() == "alice"
        assert pool.ambient["mode_id"]() == "default"
        assert pool.ambient["app_id"]() == "test"


class TestMem0AndLisaMemoryDeclared:
    """When both mem0 and lisa_memory are declared, only the generic router
    is registered — pinned to the default backend."""

    def test_registers_only_generic_router_pinned_to_default(self) -> None:
        mem0_spec = FakeSpec(
            id="mem0",
            kind="http",
            tool_names={"add": "add_memory", "search": "search_memories", "list": "get_memories"},
        )
        lisa_spec = FakeSpec(id="lisa_memory")
        pool = FakePool()
        register_memory_aliases(
            pool,
            memory_config=FakeMemoryConfig(
                default_backend="mem0",
                backends=(mem0_spec, lisa_spec),
            ),
            memory_config_pack_slug="lisa-pack-mem0",
            user_slug="alice",
            current_mode_id="default",
            app_id="test",
        )
        names = _registered_names(pool)
        # Only the generic router — no per-backend synthetics.
        assert names == ["memory.add", "memory.search", "memory.list"]

    def test_generic_router_targets_default_backend(self) -> None:
        mem0_spec = FakeSpec(
            id="mem0",
            kind="http",
            tool_names={"add": "add_memory", "search": "search_memories", "list": "get_memories"},
        )
        lisa_spec = FakeSpec(id="lisa_memory")
        pool = FakePool()
        register_memory_aliases(
            pool,
            memory_config=FakeMemoryConfig(
                default_backend="mem0",
                backends=(mem0_spec, lisa_spec),
            ),
            memory_config_pack_slug="lisa-pack-mem0",
            user_slug="alice",
            current_mode_id="default",
            app_id="test",
        )
        # The generic memory.add targets the mem0 server with the
        # mem0 arg map (content→text).
        generic_add = next(
            r for r in pool.registered if r["name"] == "memory.add"
        )
        assert generic_add["target_tool"] == "add_memory"
        assert generic_add["target_server_id"] == "lisa-pack-mem0.mem0"
        assert generic_add["arg_map"] is not None

    def test_generic_router_targets_lisa_memory_when_default(self) -> None:
        lisa_spec = FakeSpec(id="lisa_memory")
        pool = FakePool()
        register_memory_aliases(
            pool,
            memory_config=FakeMemoryConfig(
                default_backend="lisa_memory",
                backends=(lisa_spec,),
            ),
            memory_config_pack_slug=None,
            user_slug="alice",
            current_mode_id="default",
            app_id="test",
        )
        generic_add = next(
            r for r in pool.registered if r["name"] == "memory.add"
        )
        assert generic_add["arg_map"] is None
        assert generic_add["target_server_id"] == "lisa-memory"


class TestMem0ArgMap:
    """Verify the mem0 arg map (on the generic router) translates generic→native fields."""

    def test_generic_add_arg_map_translates_content_to_text_when_mem0_default(self) -> None:
        mem0_spec = FakeSpec(
            id="mem0",
            kind="http",
            tool_names={"add": "add_memory", "search": "search_memories", "list": "get_memories"},
        )
        pool = FakePool()
        register_memory_aliases(
            pool,
            memory_config=FakeMemoryConfig(
                default_backend="mem0",
                backends=(mem0_spec,),
            ),
            memory_config_pack_slug=None,
            user_slug="alice",
            current_mode_id="default",
            app_id="test",
        )
        memory_add = next(
            r for r in pool.registered if r["name"] == "memory.add"
        )
        arg_map = memory_add["arg_map"]
        assert arg_map is not None
        mapped = arg_map({"content": "hello", "user_slug": "bob"})
        assert mapped["text"] == "hello"
        assert mapped["user_slug"] == "bob"
        # content should not leak through
        assert "content" not in mapped


class TestBackendResultMaps:
    """Verify _backend_result_maps resolves dotted import paths into callables."""

    def test_lisa_memory_has_no_result_maps(self) -> None:
        """First-party lisa_memory (no result_maps declared) returns all None."""
        from hosts.core.memory_aliases import _backend_result_maps

        spec = FakeSpec(id="lisa_memory", kind="stdio")
        add_map, search_map, list_map = _backend_result_maps(spec)
        assert add_map is None
        assert search_map is None
        assert list_map is None

    def test_mem0_result_maps_resolve_to_callables(self) -> None:
        """Mem0 result_maps import paths resolve to the actual adapter functions."""
        from hosts.core.memory_aliases import _backend_result_maps

        mem0_spec = FakeSpec(
            id="mem0",
            kind="http",
            tool_names={"add": "add_memory", "search": "search_memories", "list": "get_memories"},
            result_maps={
                "add": "lisa_packs.mem0.result_maps.mem0_add_result_map",
                "search": "lisa_packs.mem0.result_maps.mem0_search_result_map",
                "list": "lisa_packs.mem0.result_maps.mem0_list_result_map",
            },
        )
        add_map, search_map, list_map = _backend_result_maps(mem0_spec)
        assert callable(add_map)
        assert callable(search_map)
        assert callable(list_map)
        # Verify they are the actual functions from the module
        from lisa_packs.mem0.result_maps import (
            mem0_add_result_map,
            mem0_list_result_map,
            mem0_search_result_map,
        )

        assert add_map is mem0_add_result_map
        assert search_map is mem0_search_result_map
        assert list_map is mem0_list_result_map

    def test_bad_import_path_falls_back_to_none(self) -> None:
        """A bad import path is logged and falls back to None (raw pass-through)."""
        from hosts.core.memory_aliases import _backend_result_maps

        spec = FakeSpec(
            id="bad",
            kind="http",
            result_maps={"add": "nonexistent.module.bad_function"},
        )
        add_map, search_map, list_map = _backend_result_maps(spec)
        assert add_map is None
        assert search_map is None
        assert list_map is None
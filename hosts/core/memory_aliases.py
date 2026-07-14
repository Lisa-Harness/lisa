"""Register the generic ``memory.*`` synthetic aliases on a started MCP pool.

Transport hosts (cli, web, vscode) all need the same wiring after
``MCPToolPool.start()`` succeeds:

1. Set ambient resolvers on the pool so the proxy layer can inject
   runtime scope (user_slug, mode_id, app_id) into tool arguments at
   dispatch time.
2. Register the three generic aliases (``memory.add`` / ``memory.search``
   / ``memory.list``) that route through the pool to the default memory
   backend selected by capability-provider policy.

Both steps are transport-neutral: they depend only on the runtime's
``memory_config`` / ``memory_config_pack_slug`` (set by
``mcp_pool.compile_mcp_configs``) and the runtime's ``user_slug`` /
``current_mode_id``. They do NOT depend on any ``_RUNTIME_STATE`` dict,
so every transport host gets identical memory behavior by calling
:func:`register_memory_aliases` from its pool-start path.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def register_memory_aliases(
    pool: Any,
    *,
    memory_config: Any,
    memory_config_pack_slug: str | None,
    user_slug: str,
    current_mode_id: str,
    app_id: str,
) -> None:
    """Set ambient resolvers + register the ``memory.*`` synthetic aliases.

    Must be called after ``pool.start()`` so the alias handlers forward to
    live server sessions and the proxy's ambient resolvers are in place
    before the first turn.

    Args:
        pool: The started :class:`~kernel.tools.mcp_client.MCPToolPool`.
        memory_config: The :class:`~kernel.mcp_launch.server_manifest.MemoryBackendConfig`
            selected by capability-provider policy (``runtime.memory_config``).
            May be ``None`` when no pack contributes a ``memory:`` section —
            in that case the first-party ``lisa-memory`` server is the
            default and its tool names are the generic aliases themselves.
        memory_config_pack_slug: The owning pack's slug
            (``runtime.memory_config_pack_slug``). Pack-contributed servers
            are namespaced as ``<pack_slug>.<server_id>``; the alias pins
            dispatch to that namespaced id. ``None`` means the default
            backend is the bare first-party ``lisa-memory`` (registered in
            code, not via a pack).
        user_slug: The active user slug for ambient injection.
        current_mode_id: The active mode id for ambient injection.
        app_id: The host app id for ambient injection (e.g. ``"little-lisa"``).
    """
    _set_ambient_resolvers(pool, user_slug=user_slug, current_mode_id=current_mode_id, app_id=app_id)
    _register_memory_synthetic_tools(
        pool,
        memory_config=memory_config,
        memory_config_pack_slug=memory_config_pack_slug,
    )


def _set_ambient_resolvers(
    pool: Any,
    *,
    user_slug: str,
    current_mode_id: str,
    app_id: str,
) -> None:
    """Register ambient resolvers so the proxy layer can inject runtime scope.

    The proxy strips ambient fields (user_id, agent_id, app_id) from the
    schema the model sees and injects the resolved values before forwarding
    to the MCP server. Without these resolvers, mem0's ``add_memory`` /
    ``search_memories`` / ``get_memories`` would 202 (async job accepted)
    but the memory wouldn't be scoped to a user, and the operation would
    surface "storage unavailable".
    """
    set_ambient_resolver = getattr(pool, "set_ambient_resolver", None)
    if set_ambient_resolver is None:
        logger.debug("pool has no set_ambient_resolver; skipping ambient injection")
        return
    set_ambient_resolver("user_slug", lambda: user_slug)
    set_ambient_resolver("mode_id", lambda: current_mode_id)
    set_ambient_resolver("app_id", lambda: app_id)


def _register_memory_synthetic_tools(
    pool: Any,
    *,
    memory_config: Any,
    memory_config_pack_slug: str | None,
) -> None:
    """Register explicit memory synthetics on a started pool.

    Each declared memory backend gets its own explicit synthetic surface
    (``mem0.add`` / ``mem0.search`` / ``mem0.list``, or
    ``lisa_memory.add`` / ``lisa_memory.search`` / ``lisa_memory.list``)
    so the model can address a specific backend by name rather than through
    an opaque generic alias.

    The generic ``memory.*`` aliases are also registered as a convenience
    router to the default backend, preserving backwards compatibility with
    existing authored prompts and packs that call ``memory.add``.

    Ambient scope (user_id) is injected by the proxy layer, not by these
    aliases — they only do generic→native field renaming (e.g.
    ``content``→``text`` for mem0).
    """
    from kernel.tools.synthetic import mcp_alias_tool  # noqa: PLC0415

    default_backend_id = getattr(memory_config, "default_backend", None)
    declared = list(getattr(memory_config, "backends", ()) or ())
    prefix = f"{memory_config_pack_slug}." if memory_config_pack_slug else ""

    # Always include the first-party lisa_memory backend, even when it is
    # not explicitly declared in the manifest — it is the implicit default.
    declared_ids = {s.id for s in declared}
    if "lisa_memory" not in declared_ids:
        declared = [
            *_LISA_MEMORY_IMPLICIT_SPEC,
            *declared,
        ]

    # Register an explicit synthetic surface per declared backend.
    for spec in declared:
        _register_backend_surface(
            pool,
            spec=spec,
            prefix=prefix,
            mcp_alias_tool=mcp_alias_tool,
            is_default=(spec.id == default_backend_id) or (
                default_backend_id is None and spec.id == "lisa_memory"
            ),
        )

    # Register the generic memory.* router → default backend.
    default_spec = next(
        (s for s in declared if s.id == (default_backend_id or "lisa_memory")),
        declared[0] if declared else None,
    )
    if default_spec is not None:
        _register_generic_router(
            pool,
            default_spec=default_spec,
            prefix=prefix,
            mcp_alias_tool=mcp_alias_tool,
        )


# Implicit first-party lisa_memory backend spec, used when the manifest does
# not declare a memory.backends list. Mirrors MemoryBackendSpec's shape.
_LISA_MEMORY_IMPLICIT_SPEC = [
    type(
        "MemoryBackendSpec",
        (),
        {
            "id": "lisa_memory",
            "server_id": "lisa-memory",
            "tool_names": {
                "add": "add_memory",
                "search": "search_memory",
                "list": "list_memories",
            },
        },
    )()
]


def _register_backend_surface(
    pool: Any,
    *,
    spec: Any,
    prefix: str,
    mcp_alias_tool: Any,
    is_default: bool,
) -> None:
    """Register one backend's explicit synthetic surface (``<backend>.add`` …)."""
    backend_id = spec.id
    server_id = (
        f"{prefix}{spec.server_id}"
        if backend_id != "lisa_memory"
        else f"{prefix}lisa-memory"
    )
    tool_names = dict(spec.tool_names)
    add_arg_map, search_arg_map, list_arg_map = _backend_arg_maps(backend_id)

    add_schema, search_schema, list_schema = _memory_schemas()
    default_note = " (default backend)" if is_default else ""

    pool.add_synthetic_tools([
        mcp_alias_tool(
            name=f"{backend_id}.add",
            description=f"Persist a memory record via the {backend_id} backend{default_note}.",
            pool=pool,
            target_tool=tool_names["add"],
            input_schema=add_schema,
            target_server_id=server_id,
            arg_map=add_arg_map,
        ),
        mcp_alias_tool(
            name=f"{backend_id}.search",
            description=f"Recall memories matching a query via the {backend_id} backend{default_note}.",
            pool=pool,
            target_tool=tool_names["search"],
            input_schema=search_schema,
            target_server_id=server_id,
            arg_map=search_arg_map,
        ),
        mcp_alias_tool(
            name=f"{backend_id}.list",
            description=f"List recent memory records via the {backend_id} backend{default_note}.",
            pool=pool,
            target_tool=tool_names["list"],
            input_schema=list_schema,
            target_server_id=server_id,
            arg_map=list_arg_map,
        ),
    ])


def _register_generic_router(
    pool: Any,
    *,
    default_spec: Any,
    prefix: str,
    mcp_alias_tool: Any,
) -> None:
    """Register the generic ``memory.*`` aliases → default backend."""
    backend_id = default_spec.id
    server_id = (
        f"{prefix}lisa-memory"
        if backend_id == "lisa_memory"
        else f"{prefix}{default_spec.server_id}"
    )
    tool_names = dict(default_spec.tool_names)
    add_arg_map, search_arg_map, list_arg_map = _backend_arg_maps(backend_id)

    add_schema, search_schema, list_schema = _memory_schemas()

    pool.add_synthetic_tools([
        mcp_alias_tool(
            name="memory.add",
            description="Persist a memory record via the default memory backend.",
            pool=pool,
            target_tool=tool_names["add"],
            input_schema=add_schema,
            target_server_id=server_id,
            arg_map=add_arg_map,
        ),
        mcp_alias_tool(
            name="memory.search",
            description="Recall memories matching a free-text query via the default backend.",
            pool=pool,
            target_tool=tool_names["search"],
            input_schema=search_schema,
            target_server_id=server_id,
            arg_map=search_arg_map,
        ),
        mcp_alias_tool(
            name="memory.list",
            description="List recent memory records for a user scope via the default backend.",
            pool=pool,
            target_tool=tool_names["list"],
            input_schema=list_schema,
            target_server_id=server_id,
            arg_map=list_arg_map,
        ),
    ])


def _backend_arg_maps(backend_id: str) -> tuple[Any, Any, Any]:
    """Return (add_arg_map, search_arg_map, list_arg_map) for a backend."""
    if backend_id == "mem0":
        def _mem0_add_arg_map(args: dict[str, Any]) -> dict[str, Any]:
            mapped = {
                "text": args.get("content", ""),
                "user_slug": args.get("user_slug", "default"),
                "memory_scope": args.get("memory_scope", "personal"),
                "memory_tier": args.get("memory_tier", "daily"),
            }
            return {k: v for k, v in mapped.items() if v is not None}

        def _mem0_search_or_list_arg_map(args: dict[str, Any]) -> dict[str, Any]:
            return {k: v for k, v in args.items() if k != "user_slug"}

        return _mem0_add_arg_map, _mem0_search_or_list_arg_map, _mem0_search_or_list_arg_map
    return None, None, None


def _memory_schemas() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return the (add, search, list) input schemas shared by all surfaces."""
    add_schema = {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Memory content to store."},
            "user_slug": {"type": "string", "default": "default"},
            "memory_scope": {"type": "string", "default": "personal"},
            "memory_tier": {"type": "string", "default": "daily"},
        },
        "required": ["content"],
    }
    search_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "user_slug": {"type": "string", "default": "default"},
            "limit": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    }
    list_schema = {
        "type": "object",
        "properties": {
            "user_slug": {"type": "string", "default": "default"},
            "limit": {"type": "integer", "default": 20},
        },
    }
    return add_schema, search_schema, list_schema
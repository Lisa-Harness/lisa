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
    """Register the generic ``memory.*`` synthetic tools on a started pool.

    The three generic aliases route through the pool to whichever MCP server
    the manifest wired as the default memory backend (``lisa-memory`` by
    default, ``mem0`` when its HTTP endpoint is configured). The default
    backend's tool names come from the manifest's ``memory.tool_names``
    mapping. Ambient scope (user_id) is injected by the proxy layer, not by
    these aliases — they only do generic→native field renaming (e.g.
    ``content``→``text`` for mem0).
    """
    from kernel.tools.synthetic import mcp_alias_tool  # noqa: PLC0415

    default_backend_id = getattr(memory_config, "default_backend", None)
    declared = getattr(memory_config, "backends", ()) or ()
    default_spec = next((s for s in declared if s.id == default_backend_id), None)
    prefix = f"{memory_config_pack_slug}." if memory_config_pack_slug else ""
    _BACKEND_ID_TO_SERVER_ID = {"lisa_memory": f"{prefix}lisa-memory"}

    if default_spec is None:
        # No external default declared: lisa-memory is the first-party default.
        tool_names = {"add": "add_memory", "search": "search_memory", "list": "list_memories"}
        target_server_id = f"{prefix}lisa-memory"
        add_arg_map = None
        search_arg_map = None
        list_arg_map = None
    else:
        tool_names = dict(default_spec.tool_names)
        lisa_memory_server_id = _BACKEND_ID_TO_SERVER_ID.get("lisa_memory", "lisa-memory")
        target_server_id = (
            lisa_memory_server_id
            if default_spec.id == "lisa_memory"
            else f"{prefix}{default_spec.id}"
        )
        if default_spec.id == "mem0":
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

            add_arg_map = _mem0_add_arg_map
            search_arg_map = _mem0_search_or_list_arg_map
            list_arg_map = _mem0_search_or_list_arg_map
        else:
            add_arg_map = None
            search_arg_map = None
            list_arg_map = None

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

    pool.add_synthetic_tools([
        mcp_alias_tool(
            name="memory.add",
            description="Persist a memory record via the default memory backend.",
            pool=pool,
            target_tool=tool_names["add"],
            input_schema=add_schema,
            target_server_id=target_server_id,
            arg_map=add_arg_map,
        ),
        mcp_alias_tool(
            name="memory.search",
            description="Recall memories matching a free-text query via the default backend.",
            pool=pool,
            target_tool=tool_names["search"],
            input_schema=search_schema,
            target_server_id=target_server_id,
            arg_map=search_arg_map,
        ),
        mcp_alias_tool(
            name="memory.list",
            description="List recent memory records for a user scope via the default backend.",
            pool=pool,
            target_tool=tool_names["list"],
            input_schema=list_schema,
            target_server_id=target_server_id,
            arg_map=list_arg_map,
        ),
    ])
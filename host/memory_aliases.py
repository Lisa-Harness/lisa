"""Register the generic ``memory.*`` synthetic aliases on a started MCP pool.

Transport hosts (cli, web) both need the same wiring after
``MCPToolPool.start()`` succeeds:

1. Set ambient resolvers on the pool so the proxy layer can inject
   runtime scope (user_slug, mode_id, app_id) into tool arguments at
   dispatch time.
2. Register the five generic aliases (``memory.add`` / ``memory.search``
   / ``memory.list`` / ``memory.get`` / ``memory.delete``) that route
   through the pool to the default memory backend selected by
   capability-provider policy.

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
    _set_ambient_resolvers(
        pool, user_slug=user_slug, current_mode_id=current_mode_id, app_id=app_id
    )
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
    """Register the generic ``memory.*`` synthetic aliases on a started pool.

    Only the five generic aliases (``memory.add`` / ``memory.search`` /
    ``memory.list`` / ``memory.get`` / ``memory.delete``) are registered. They route through the pool to the
    default memory backend selected by capability-provider policy.

    Per-backend explicit surfaces (``mem0.add``, ``lisa_memory.add``, …)
    are NOT registered — the native tools are already available to the model
    under their fully-qualified names (``mem0.add_memory``,
    ``lisa-memory.search_memory``, …) with ambient fields stripped by the
    proxy layer and injected at dispatch time. The generic aliases exist to
    give packs/prompts a stable, backend-agnostic name that follows the
    configured default backend.

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
# not declare a memory.backends list. Mirrors MemoryBackendSpec's shape
# (id, kind, tool_names — no server_id field; server id is derived from id).
_LISA_MEMORY_IMPLICIT_SPEC = [
    type(
        "MemoryBackendSpec",
        (),
        {
            "id": "lisa_memory",
            "kind": "stdio",
            "tool_names": {
                "add": "add_memory",
                "search": "search_memory",
                "list": "list_memories",
                "get": "get_memory",
                "delete": "delete_memory",
            },
        },
    )()
]


# Map backend id → MCP server id. ``lisa_memory`` uses the canonical
# ``lisa-memory`` server id; all other backends use their own id as the
# server id (prefixed by the owning pack slug at call sites).
_BACKEND_ID_TO_SERVER_ID: dict[str, str] = {"lisa_memory": "lisa-memory"}


def _register_generic_router(
    pool: Any,
    *,
    default_spec: Any,
    prefix: str,
    mcp_alias_tool: Any,
) -> None:
    """Register the generic ``memory.*`` aliases → default backend."""
    backend_id = default_spec.id
    raw_server_id = _BACKEND_ID_TO_SERVER_ID.get(backend_id, backend_id)
    server_id = f"{prefix}{raw_server_id}"
    tool_names = dict(default_spec.tool_names)
    add_arg_map, search_arg_map, list_arg_map, get_arg_map, delete_arg_map = _backend_arg_maps(
        backend_id
    )
    add_result_map, search_result_map, list_result_map, get_result_map, delete_result_map = (
        _backend_result_maps(default_spec)
    )

    add_schema, search_schema, list_schema, get_schema, delete_schema = _memory_schemas()

    # Build the alias list dynamically, skipping any operation the backend
    # doesn't declare a tool name for. This keeps the router robust against
    # partial/legacy manifests (e.g. a mem0 config that only declares
    # add/search/list) — the missing aliases simply aren't registered.
    alias_defs = [
        (
            "memory.add",
            "Persist a memory record via the default memory backend.",
            "add",
            add_schema,
            add_arg_map,
            add_result_map,
        ),
        (
            "memory.search",
            "Recall memories matching a free-text query via the default backend.",
            "search",
            search_schema,
            search_arg_map,
            search_result_map,
        ),
        (
            "memory.list",
            "List recent memory records for a user scope via the default backend.",
            "list",
            list_schema,
            list_arg_map,
            list_result_map,
        ),
        (
            "memory.get",
            "Fetch a single memory record by id via the default backend.",
            "get",
            get_schema,
            get_arg_map,
            get_result_map,
        ),
        (
            "memory.delete",
            "Delete a single memory record by id via the default backend.",
            "delete",
            delete_schema,
            delete_arg_map,
            delete_result_map,
        ),
    ]
    aliases = []
    for name, desc, op, schema, arg_map, result_map in alias_defs:
        target = tool_names.get(op)
        if not target:
            continue
        aliases.append(
            mcp_alias_tool(
                name=name,
                description=desc,
                pool=pool,
                target_tool=target,
                input_schema=schema,
                target_server_id=server_id,
                arg_map=arg_map,
                result_map=result_map,
            )
        )
    pool.add_synthetic_tools(aliases)


def _backend_arg_maps(backend_id: str) -> tuple[Any, Any, Any, Any, Any]:
    """Return (add, search, list, get, delete) arg maps for a backend."""
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

        def _mem0_get_or_delete_arg_map(args: dict[str, Any]) -> dict[str, Any]:
            # mem0's get/delete take the memory id under ``memory_id``.
            mapped = {
                "memory_id": args.get("memory_id"),
                "user_id": args.get("user_slug", "default"),
            }
            return {k: v for k, v in mapped.items() if v is not None}

        return (
            _mem0_add_arg_map,
            _mem0_search_or_list_arg_map,
            _mem0_search_or_list_arg_map,
            _mem0_get_or_delete_arg_map,
            _mem0_get_or_delete_arg_map,
        )
    return None, None, None, None, None


def _backend_result_maps(
    default_spec: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    """Resolve the backend's ``result_maps`` dotted import paths into callables.

    Returns ``(add, search, list, get, delete)`` result maps — each is a
    ``Callable[[str], str]`` or ``None``. When the backend declares no
    ``result_maps`` (e.g. first-party lisa_memory which already returns the
    canonical ``list[MemoryRecord]`` JSON array), all five are ``None`` and
    the alias passes ``output_text`` through unchanged.

    Import paths are resolved lazily at registration time (not at manifest
    parse time) so a manifest can declare a pack that isn't installed yet
    without crashing the parse step. A bad import path is logged as a warning
    and the corresponding result_map falls back to ``None`` (raw pass-through)
    so the alias still works — it just doesn't normalize.
    """
    result_maps = getattr(default_spec, "result_maps", None) or {}
    if not result_maps:
        return None, None, None, None, None

    def _resolve(path: str | None) -> Any:
        if not path:
            return None
        try:
            module_path, _, func_name = path.rpartition(".")
            if not module_path or not func_name:
                logger.warning("result_maps entry %r is not a valid dotted path", path)
                return None
            import importlib

            mod = importlib.import_module(module_path)
            fn = getattr(mod, func_name, None)
            if not callable(fn):
                logger.warning("result_maps entry %r: %s is not callable", path, func_name)
                return None
            return fn
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("result_maps entry %r failed to import: %s", path, exc)
            return None

    return (
        _resolve(result_maps.get("add")),
        _resolve(result_maps.get("search")),
        _resolve(result_maps.get("list")),
        _resolve(result_maps.get("get")),
        _resolve(result_maps.get("delete")),
    )


def _memory_schemas() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    """Return the (add, search, list, get, delete) input schemas shared by all surfaces.

    These are the model-facing schemas for the generic ``memory.*`` aliases.
    ``user_slug`` is intentionally ABSENT from every schema: it is an ambient
    identity field, not a model-controllable parameter. The host registers a
    resolver for it (``register_memory_aliases`` → ``set_ambient_resolver``)
    and the proxy injects the active user at dispatch time. Exposing it here
    would force the model to invent a slug (it has no source of truth) — the
    bug Lisa hit when she complained "I need a user_slug to run
    memory.search". The native backend tools carry the same field and the
    per-server ``proxy_policy.ambient`` strips it from their schemas too, so
    no memory path leaks the ambient field to the model.
    """
    add_schema = {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Memory content to store."},
            "memory_scope": {"type": "string", "default": "personal"},
            "memory_tier": {"type": "string", "default": "daily"},
        },
        "required": ["content"],
    }
    search_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    }
    list_schema = {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "default": 20},
        },
    }
    get_schema = {
        "type": "object",
        "properties": {
            "memory_id": {"type": "integer", "description": "The memory record id."},
        },
        "required": ["memory_id"],
    }
    delete_schema = {
        "type": "object",
        "properties": {
            "memory_id": {"type": "integer", "description": "The memory record id to delete."},
        },
        "required": ["memory_id"],
    }
    return add_schema, search_schema, list_schema, get_schema, delete_schema

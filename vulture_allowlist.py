"""Vulture false-positive allowlist for lisa.

These symbols are live at runtime but invisible to static analysis because
they are reached via dynamic dispatch (entry-point loading, getattr, importlib)
or are public API surface declared in ``__all__`` but never imported internally.

Every entry carries a ``# reason:`` comment explaining why vulture should
ignore it.  Run::

    uv run vulture kernel sdk mcp_server host vulture_allowlist.py
"""

# ────────────────────────────────────────────────────────────────────────
# Console-script entry points — called by pip-installed [project.scripts].
# ────────────────────────────────────────────────────────────────────────
# reason: console_script target — lisa-memory MCP server
mcp_server.memory.server.main  # noqa: Vulture

# reason: console_script target — lisa-guard MCP server
mcp_server.guard.server.main  # noqa: Vulture

# reason: console_script target — lisa-content MCP server
mcp_server.content.server.main  # noqa: Vulture

# reason: console_script target — lisa-catalog MCP server
mcp_server.catalog.server.main  # noqa: Vulture

# reason: console_script target — lisa-storage MCP server
mcp_server.storage.server.main  # noqa: Vulture

# ──────────────────────────────────────────────────────────────────────────
# importlib.metadata entry-point targets — loaded via ep.load().
# ──────────────────────────────────────────────────────────────────────────
# reason: lisa.extensions.content_roots entry-point target (sdk._content_root:get_content_root)
sdk._content_root.get_content_root  # noqa: Vulture

# reason: lisa.extensions.content_roots entry-point target (sdk.packs.core:get_content_root)
sdk.packs.core.get_content_root  # noqa: Vulture

# reason: lisa.extensions.host_commands entry-point target (sdk.packs.checkpoint.commands:get_host_commands)
sdk.packs.checkpoint.commands.get_host_commands  # noqa: Vulture

# reason: lisa.extensions.model_routers entry-point target (sdk.router:get_router_factory)
sdk.router.get_router_factory  # noqa: Vulture

# reason: lisa.extensions.model_router_configs entry-point target (sdk.router.default_config:get_default_router_config)
sdk.router.default_config.get_default_router_config  # noqa: Vulture

# ──────────────────────────────────────────────────────────────────────────
# Operation execute() functions — dispatched via getattr(module, "execute")
# at kernel/operations/registry.py.
# ──────────────────────────────────────────────────────────────────────────
# reason: operation dispatch via getattr(module, "execute")
sdk.packs.memory.operations.memory.recall.execute  # noqa: Vulture

# reason: operation dispatch via getattr(module, "execute")
sdk.packs.memory.operations.memory.synthesize_learning.execute  # noqa: Vulture

# reason: operation dispatch via getattr(module, "execute")
sdk.packs.introspect.operations.self_model.self_model.execute  # noqa: Vulture

# ──────────────────────────────────────────────────────────────────────────
# Pydantic structured-output models — loaded via _resolve_model_class()
# string dispatch at kernel/context/mcp_catalog.py.
# ──────────────────────────────────────────────────────────────────────────
# reason: response-format model loaded via _resolve_model_class string dispatch
sdk.packs.introspect.operations.self_model.reflect_self.ReflectSelfOutput  # noqa: Vulture

# reason: operation parameter model loaded via _resolve_model_class string dispatch
sdk.packs.memory.operations.memory.recall.Parameters  # noqa: Vulture

# reason: operation parameter model loaded via _resolve_model_class string dispatch
sdk.packs.memory.operations.memory.synthesize_learning.Parameters  # noqa: Vulture

# reason: operation result model loaded via _resolve_model_class string dispatch
sdk.packs.memory.operations.memory.synthesize_learning.DistilledLearning  # noqa: Vulture

# ──────────────────────────────────────────────────────────────────────────
# Protocol / ABC method parameters — vulture cannot see that these are
# interface contracts.  The parameters are part of the signature callers
# must satisfy, even if the Protocol/ABC body (``...`` / ``raise`` / pass)
# never reads them.
# ──────────────────────────────────────────────────────────────────────────
# reason: OutputGuard.validate Protocol parameter — interface contract
kernel.seams.protocols.OutputGuard.validate.surface  # noqa: Vulture

# reason: StateProvider Protocol parameter — interface contract
kernel.state_providers.BlankStateProvider.__call__  # noqa: Vulture

# reason: EmbeddingProvider Protocol parameter — interface contract
sdk.memory.store.EmbeddingProvider.embed.texts  # noqa: Vulture

# ──────────────────────────────────────────────────────────────────────────
# MCP lifespan context-manager parameters — the ``app`` argument is part of
# the ``@asynccontextmanager`` lifespan signature MCPServer hands us; the
# catalog/content/fetch/guard/memory/storage servers don't read it.
# ──────────────────────────────────────────────────────────────────────────
# reason: MCPServer lifespan signature requires ``app``; value not read
mcp_server.catalog.server._lifespan.app  # noqa: Vulture

# reason: MCPServer lifespan signature requires ``app``; value not read
mcp_server.content.server._lifespan.app  # noqa: Vulture

# reason: MCPServer lifespan signature requires ``app``; value not read
mcp_server.fetch.server._lifespan.app  # noqa: Vulture

# reason: MCPServer lifespan signature requires ``app``; value not read
mcp_server.guard.server._lifespan.app  # noqa: Vulture

# reason: MCPServer lifespan signature requires ``app``; value not read
mcp_server.memory.server._lifespan.app  # noqa: Vulture

# reason: MCPServer lifespan signature requires ``app``; value not read
mcp_server.storage.server._lifespan.app  # noqa: Vulture

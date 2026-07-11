"""Vulture false-positive allowlist for lisa_framework.

These symbols are live at runtime but invisible to static analysis because
they are reached via dynamic dispatch (entry-point loading, getattr, importlib)
or are public API surface declared in ``__all__`` but never imported internally.

Every entry carries a ``# reason:`` comment explaining why vulture should
ignore it.  Run::

    uv run vulture kernel sdk sdk_packs hosts/little_lisa vulture_allowlist.py
"""

# ──────────────────────────────────────────────────────────────────────────
# Console-script entry points — called by pip-installed [project.scripts].
# ──────────────────────────────────────────────────────────────────────────
# reason: console_script target — little-lisa CLI
hosts.little_lisa.main.main  # noqa: Vulture

# reason: console_script target — lisa-context MCP server
sdk.lisa_context.server.main  # noqa: Vulture

# reason: console_script target — lisa-memory MCP server
sdk.lisa_memory.server.main  # noqa: Vulture

# reason: console_script target — lisa-guard MCP server
sdk.lisa_guard.server.main  # noqa: Vulture

# reason: console_script target — lisa-content MCP server
sdk.lisa_content.server.main  # noqa: Vulture

# ──────────────────────────────────────────────────────────────────────────
# importlib.metadata entry-point targets — loaded via ep.load().
# ──────────────────────────────────────────────────────────────────────────
# reason: lisa.extensions.content_roots entry-point target
sdk.get_content_root  # noqa: Vulture

# reason: lisa.extensions.content_roots entry-point target
sdk_packs.core.get_content_root  # noqa: Vulture

# reason: lisa.extensions.signal_connectors entry-point target (cloudflare)
lisa_pack_cloudflare.connector.CloudflareSignalCollector  # noqa: Vulture

# reason: lisa.extensions.signal_connectors entry-point target (github)
lisa_pack_github.connector.GitHubSignalCollector  # noqa: Vulture

# reason: lisa.extensions.signal_connectors entry-point target (stripe)
lisa_pack_stripe.connector.StripeSignalCollector  # noqa: Vulture

# ──────────────────────────────────────────────────────────────────────────
# Operation execute() functions — dispatched via getattr(module, "execute")
# at kernel/operations/registry.py.
# ──────────────────────────────────────────────────────────────────────────
# reason: operation dispatch via getattr(module, "execute")
sdk_packs.learning.operations.memory.recall.execute  # noqa: Vulture

# reason: operation dispatch via getattr(module, "execute")
sdk_packs.learning.operations.memory.synthesize_learning.execute  # noqa: Vulture

# reason: operation dispatch via getattr(module, "execute")
sdk_packs.core.operations.research.workshop_run.execute  # noqa: Vulture

# reason: operation dispatch via getattr(module, "execute")
sdk_packs.core.operations.self_model.self_model.execute  # noqa: Vulture

# ──────────────────────────────────────────────────────────────────────────
# Pydantic structured-output models — loaded via _resolve_model_class()
# string dispatch at kernel/context/mcp_catalog.py.
# ──────────────────────────────────────────────────────────────────────────
# reason: response-format model loaded via _resolve_model_class string dispatch
sdk_packs.search.operations.research.discussion_briefing.DiscussionBriefingStructuredOutput  # noqa: Vulture

# reason: response-format model loaded via _resolve_model_class string dispatch
sdk_packs.search.operations.research.web_search.WebSearchStructuredOutput  # noqa: Vulture

# reason: response-format model loaded via _resolve_model_class string dispatch
sdk_packs.core.operations.research.workshop_run.WorkshopSynthesis  # noqa: Vulture

# reason: response-format model loaded via _resolve_model_class string dispatch
sdk_packs.core.operations.self_model.architecture_atlas.ArchitectureAtlasOutput  # noqa: Vulture

# reason: response-format model loaded via _resolve_model_class string dispatch
sdk_packs.core.operations.self_model.reflect_self.ReflectSelfOutput  # noqa: Vulture

# reason: operation parameter model loaded via _resolve_model_class string dispatch
sdk_packs.learning.operations.memory.recall.Parameters  # noqa: Vulture

# reason: operation parameter model loaded via _resolve_model_class string dispatch
sdk_packs.learning.operations.memory.synthesize_learning.Parameters  # noqa: Vulture

# reason: operation result model loaded via _resolve_model_class string dispatch
sdk_packs.learning.operations.memory.synthesize_learning.DistilledLearning  # noqa: Vulture

# ──────────────────────────────────────────────────────────────────────────
# Test-seam re-exports — imported into main.py so monkeypatch.setattr
# can target them on the module surface.  Not called from main.py itself.
# ──────────────────────────────────────────────────────────────────────────
# reason: test-seam re-export for monkeypatch.setattr(little_lisa, ...)
hosts.little_lisa.main._DEFAULT_MEMORY_MAX_RECORDS  # noqa: Vulture

# ──────────────────────────────────────────────────────────────────────────
# Protocol / ABC method parameters — vulture cannot see that these are
# interface contracts.  The parameters are part of the signature callers
# must satisfy, even if the Protocol/ABC body (``...`` / ``raise`` / pass)
# never reads them.
# ──────────────────────────────────────────────────────────────────────────
# reason: DataSanitizer.redact Protocol parameter — interface contract
kernel.context.data_sanitizer.PassThroughSanitizer.redact.scope  # noqa: Vulture

# reason: OutputGuard.validate Protocol parameter — interface contract
kernel.seams.protocols.OutputGuard.validate.surface  # noqa: Vulture

# reason: StateProvider Protocol parameter — interface contract
kernel.state_providers.BlankStateProvider.__call__  # noqa: Vulture

# reason: EmbeddingProvider Protocol parameter — interface contract
sdk.lisa_memory.store.EmbeddingProvider.embed.texts  # noqa: Vulture

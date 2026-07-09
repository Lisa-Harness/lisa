"""Vulture false-positive allowlist for lisa_framework.

These symbols are live at runtime but invisible to static analysis because
they are reached via dynamic dispatch (entry-point loading, getattr, importlib)
or are public API surface declared in ``__all__`` but never imported internally.

Every entry carries a ``# reason:`` comment explaining why vulture should
ignore it.  Run::

    uv run vulture kernel lisa_sdk sdk_extensions sdk_packs examples/little_lisa vulture_allowlist.py
"""

# ──────────────────────────────────────────────────────────────────────────
# Console-script entry points — called by pip-installed [project.scripts].
# ──────────────────────────────────────────────────────────────────────────
# reason: console_script target — little-lisa CLI
examples.little_lisa.main.main  # noqa: Vulture

# reason: console_script target — lisa-context MCP server
lisa_sdk.lisa_context.server.main  # noqa: Vulture

# reason: console_script target — lisa-memory MCP server
lisa_sdk.lisa_memory.server.main  # noqa: Vulture

# reason: console_script target — lisa-search MCP server
lisa_sdk.lisa_search.server.main  # noqa: Vulture

# reason: console_script target — lisa-guard MCP server
lisa_sdk.lisa_guard.server.main  # noqa: Vulture

# reason: console_script target — lisa-content MCP server
lisa_sdk.lisa_content.server.main  # noqa: Vulture

# ──────────────────────────────────────────────────────────────────────────
# importlib.metadata entry-point targets — loaded via ep.load().
# ──────────────────────────────────────────────────────────────────────────
# reason: lisa.extensions.content_roots entry-point target
lisa_sdk.get_content_root  # noqa: Vulture

# reason: lisa.extensions.content_roots entry-point target
sdk_packs.core.get_content_root  # noqa: Vulture

# reason: lisa.extensions.signal_connectors entry-point target (cloudflare)
sdk_extensions.cloudflare.connector.CloudflareSignalConnector  # noqa: Vulture

# reason: lisa.extensions.signal_connectors entry-point target (github)
sdk_extensions.github.connector.GitHubSignalConnector  # noqa: Vulture

# reason: lisa.extensions.signal_connectors entry-point target (stripe)
sdk_extensions.stripe.connector.StripeSignalConnector  # noqa: Vulture

# ──────────────────────────────────────────────────────────────────────────
# Operation execute() functions — dispatched via getattr(module, "execute")
# at kernel/operations/registry.py:352.
# ──────────────────────────────────────────────────────────────────────────
# reason: operation dispatch via getattr(module, "execute")
sdk_packs.core.operations.memory.recall.execute  # noqa: Vulture

# reason: operation dispatch via getattr(module, "execute")
sdk_packs.core.operations.memory.synthesize_learning.execute  # noqa: Vulture

# reason: operation dispatch via getattr(module, "execute")
sdk_packs.core.operations.research.workshop_run.execute  # noqa: Vulture

# reason: operation dispatch via getattr(module, "execute")
sdk_packs.core.operations.self_model.self_model.execute  # noqa: Vulture

# ──────────────────────────────────────────────────────────────────────────
# Pydantic structured-output models — loaded via _resolve_model_class()
# string dispatch at kernel/context/mcp_catalog.py:321.
# ──────────────────────────────────────────────────────────────────────────
# reason: response-format model loaded via _resolve_model_class string dispatch
sdk_packs.core.operations.research.discussion_briefing.DiscussionBriefingStructuredOutput  # noqa: Vulture

# reason: response-format model loaded via _resolve_model_class string dispatch
sdk_packs.core.operations.research.web_search.WebSearchStructuredOutput  # noqa: Vulture

# reason: response-format model loaded via _resolve_model_class string dispatch
sdk_packs.core.operations.research.workshop_run.WorkshopSynthesis  # noqa: Vulture

# reason: response-format model loaded via _resolve_model_class string dispatch
sdk_packs.core.operations.self_model.architecture_atlas.ArchitectureAtlasOutput  # noqa: Vulture

# reason: response-format model loaded via _resolve_model_class string dispatch
sdk_packs.core.operations.self_model.reflect_self.ReflectSelfOutput  # noqa: Vulture

# reason: operation parameter model loaded via _resolve_model_class string dispatch
sdk_packs.core.operations.memory.recall.Parameters  # noqa: Vulture

# reason: operation parameter model loaded via _resolve_model_class string dispatch
sdk_packs.core.operations.memory.synthesize_learning.Parameters  # noqa: Vulture

# reason: operation result model loaded via _resolve_model_class string dispatch
sdk_packs.core.operations.memory.synthesize_learning.DistilledLearning  # noqa: Vulture

# ──────────────────────────────────────────────────────────────────────────
# Test-seam re-exports — imported into main.py so monkeypatch.setattr
# can target them on the module surface.  Not called from main.py itself.
# ──────────────────────────────────────────────────────────────────────────
# reason: test-seam re-export for monkeypatch.setattr(little_lisa, ...)
examples.little_lisa.main._DEFAULT_MEMORY_MAX_RECORDS  # noqa: Vulture

# ──────────────────────────────────────────────────────────────────────────
# Protocol / ABC method parameters — vulture cannot see that these are
# interface contracts.  The parameters are part of the signature callers
# must satisfy, even if the Protocol/ABC body (``...`` / ``raise`` / pass)
# never reads them.
# ──────────────────────────────────────────────────────────────────────────
# reason: StateBuilder.__call__ Protocol parameter — interface contract
kernel.extension_runtime.StateBuilder.__call__.collections  # noqa: Vulture

# reason: StateBuilder.__call__ Protocol parameter — interface contract
kernel.extension_runtime.StateBuilder.__call__.analysis  # noqa: Vulture

# reason: StageChainExecutor.__call__ Protocol parameter — interface contract
kernel.extension_runtime.StageChainExecutor.__call__.collection_scope  # noqa: Vulture

# reason: StageChainExecutor.__call__ Protocol parameter — interface contract
kernel.extension_runtime.StageChainExecutor.__call__.directional_query  # noqa: Vulture

# reason: abstract method parameter — interface contract for data sources
kernel.intake.knowledge.collect_artifact.http_client  # noqa: Vulture

# reason: abstract method parameter — interface contract for data sources
kernel.intake.sources.execute_cycle.http_client  # noqa: Vulture

# reason: OutboxReader Protocol parameter — interface contract
kernel.seams.protocols.OutboxReader.read_outbox_entries.max_bytes  # noqa: Vulture

# reason: SynthesisRuntime Protocol parameters — interface contracts
kernel.synthesis.runtime.run_local_analysis.collection_scope  # noqa: Vulture

# reason: SynthesisRuntime Protocol parameters — interface contracts
kernel.synthesis.runtime.run_local_analysis.raw_items  # noqa: Vulture

# reason: SynthesisRuntime Protocol parameters — interface contracts
kernel.synthesis.runtime.run_local_analysis.directional_query  # noqa: Vulture

# reason: SynthesisRuntime Protocol parameters — interface contracts
kernel.synthesis.runtime.run_synthesis.collection_scope  # noqa: Vulture

# reason: SynthesisRuntime Protocol parameters — interface contracts
kernel.synthesis.runtime.run_synthesis.local_analysis  # noqa: Vulture

# reason: SynthesisRuntime Protocol parameters — interface contracts
kernel.synthesis.runtime.build_memory_artifact.collection_scope  # noqa: Vulture

# reason: SynthesisRuntime Protocol parameters — interface contracts
kernel.synthesis.runtime.build_memory_artifact.local_analysis  # noqa: Vulture

# reason: SynthesisRuntime Protocol parameters — interface contracts
kernel.synthesis.runtime.build_memory_artifact.synthesis  # noqa: Vulture

# reason: EmbeddingProvider Protocol parameter — interface contract
lisa_sdk.lisa_memory.store.EmbeddingProvider.embed.texts  # noqa: Vulture
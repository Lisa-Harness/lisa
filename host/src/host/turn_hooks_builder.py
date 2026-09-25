"""Build ``TurnHooks`` from a ``HostRuntime``.

This module wires the ``HostRuntime`` to the kernel's ``TurnHooks`` dataclass,
replacing the closure-based ``_build_turn_hooks`` in ``main.py`` with a
testable, module-level function.

All dependencies are resolved from the runtime — no ``_RUNTIME_STATE`` reads,
no global module-level caches. The returned ``TurnHooks`` is ready to pass to
``kernel.turn_runner.run_turn``.

Public surface
--------------
- :func:`build_turn_hooks`
- :func:`content_loader`
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, cast

from kernel.model_router_discovery import discover_model_router_factories
from kernel.models import OperationExecutionContext, OperationResult, ToolResult
from kernel.operations.registry import execute_operation
from kernel.pack_configuration import SessionPackConfigurationResolver
from kernel.provider_gateway_discovery import discover_provider_gateway_factories
from kernel.seams.host_adapter import build_callable_host
from kernel.seams.protocols import ContentRegistry
from kernel.slice import SliceOutcome, SliceTuning
from kernel.token_usage import TOKEN_USAGE_TOPIC
from kernel.turn_hooks import (
    ContextHooks,
    HostServiceHandles,
    LifecycleHooks,
    ModelHooks,
    OperationHooks,
    RecallHooks,
    ToolHooks,
    TurnHooks,
    TurnIdentity,
    WorkflowHooks,
)
from kernel.turn_runner import ContextSeams
from sdk.context.extension_item_builder import (
    EntryPointExtensionItemBuilder,
)

from host.context_assembly import (
    assemble_dynamic_user_context,
    assemble_system_prompt,
    context_budget_values_from_config,
)
from host.introspect import gather_introspection
from host.model_runtime import StubModelDefinition, load_manifest, resolve_model
from host.state_provider import FileBackedStateProvider, build_unified_state
from host.thread import build_recent_thread_excerpt

if TYPE_CHECKING:
    from host.callbacks import TransportCallbacks
    from host.runtime import HostRuntime

logger = logging.getLogger(__name__)


# ── Content loader ────────────────────────────────────────────────────


def content_loader(runtime: HostRuntime) -> Callable[[str], str]:
    """Return a callable that reads authored text through the kernel's content registry.

    The content registry is set by ``setup_framework(runtime)`` at host startup.
    When it is not set, this raises ``RuntimeError`` so the caller knows the
    framework was not initialized.
    """
    _ = runtime  # runtime is part of the API for future use; registry is module-level
    registry = content_registry()
    if registry is None:
        raise RuntimeError("Content registry not set. Call setup_framework(runtime) first.")
    loader: Callable[[str], str] = registry.read_authored_text
    return loader


def content_registry() -> ContentRegistry | None:
    """Return the module-level content registry, or ``None`` when unset.

    The registry is set by ``setup_framework(runtime)`` at host startup.
    Operations that need authored text (e.g. ``synthesize_and_store_learning``)
    reach it via ``TurnHooks.context_seams.content_registry`` so the turn
    runner's ``_build_minimal_context`` can resolve ``read_authored_text_fn``.
    When ``None``, those operations fail at call time with their authored-text
    error (graceful failure); non-authored operations still run normally.
    """
    from kernel.context.context_frames import _content_registry

    return _content_registry


# ── Turn-hooks builder ────────────────────────────────────────────────


class _SliceProviderHooksView:
    """Minimal read-only view passed to a pack slice-provider factory.

    Exposes only the seams a utilization-aware slice needs:
    ``last_context_snapshot`` (read from the runtime; the snapshot sink
    keeps it current each turn), ``execute_operation`` (the kernel operation
    dispatcher), and ``execute_tool`` (the MCP tool dispatcher). This
    avoids handing the factory the full ``TurnHooks`` before it is fully
    constructed. ``execute_tool`` is wired so the memory-consolidation
    slice can call ``memory.list`` to count and load episodic records.
    """

    __slots__ = ("_runtime", "execute_operation", "execute_tool")

    def __init__(
        self,
        *,
        runtime: HostRuntime,
    ) -> None:
        self._runtime = runtime
        self.execute_operation: Any = None
        self.execute_tool: Any = None
        # ``build_turn_hooks`` assigns both dispatchers before any factory
        # invocation; the slots make the surface explicit rather than a
        # monkey-patch fallback (the previous ``__getattr__`` always raised).

    @property
    def last_context_snapshot(self) -> Any:
        return self._runtime.session.last_context_snapshot


def _derive_view(runtime: HostRuntime) -> Any:
    """Derive the workflow view from the thread tail.

    A pure read from the lossless thread log — no mutable position store.
    Returns a ``WorkflowView`` or ``None`` when no active workflow exists.
    """
    from kernel.workflow_view import derive_workflow_view

    from host.thread import load_recent_thread_entries

    entries = load_recent_thread_entries(thread_path=runtime.paths.thread_path, limit=30)
    return derive_workflow_view(entries)


def build_turn_hooks(
    runtime: HostRuntime,
    context: dict[str, Any] | None = None,
    transport: TransportCallbacks | None = None,
) -> TurnHooks:
    """Build a ``TurnHooks`` dataclass wired to the host runtime.

    Args:
        runtime: The typed host runtime with config, manifest, MCP pool.
        context: Optional mutable turn-context dict. When provided, the
            ``on_turn_complete`` hook updates it with completion metadata.
            When ``None``, a fresh dict is created internally.
        transport: Optional transport callbacks for streaming/debug events.

    Returns:
        A ``TurnHooks`` ready to pass to ``kernel.turn_runner.run_turn``.
    """
    from host.gateway_provider import HostGatewayProvider

    ctx_dict = context if context is not None else {}
    pack_configuration_resolver = SessionPackConfigurationResolver(
        runtime.session_plan.manifest.resources.packs.configuration
    )
    approval_service = runtime.services.approval_service

    # ── Slice tuning ──────────────────────────────────────────────
    slice_config = runtime.config.slice_config
    slice_tuning = SliceTuning.from_mapping(slice_config.to_tuning_dict())

    # ── State provider (FileBackedStateProvider) ──────────────────
    # Post-realignment this provider is a no-op slot: history is never
    # compacted (the thread is lossless; sealing is storage management) and
    # distillation is owned by the memory pack's consolidation slice.
    file_state_provider = FileBackedStateProvider(
        thread_path=runtime.paths.thread_path,
    )

    # ── Model resolution ──────────────────────────────────────────
    def _resolve_model(model_id: str) -> StubModelDefinition:
        if runtime.session.manifest is None:
            runtime.session.manifest = load_manifest(
                {}, data_dir=runtime.paths.data_dir, path=runtime.paths.models_path
            )
        return resolve_model(
            model_id,
            runtime_state={"manifest": runtime.session.manifest},
            data_dir=runtime.paths.data_dir,
            path=runtime.paths.models_path,
        )

    # ── Extension item builder ──────────────────────────────────
    # Constructed once so every ``build_unified_state`` call and the
    # ``ContextSeams`` injection share one builder instance. Discovers
    # ``lisa.extensions.context_items`` entry points from installed
    # packs and tags each contributed record with its source pack name.
    from kernel.pack_activation import (
        load_disabled_pack_names,
    )

    _disabled_packs = load_disabled_pack_names(runtime.paths.data_dir)
    _item_builder = EntryPointExtensionItemBuilder(
        excluded_packs=_disabled_packs,
    )
    # Runtime context handed to the builder so pack entry-point functions
    # can access host resources (introspection, resolve_model, etc.).
    # ``_introspect_fn`` is defined below and resolves lazily when the
    # builder invokes a pack that reads it.
    _runtime_context: dict[str, Any] = {
        "runtime": runtime,
        "resolve_model": _resolve_model,
    }

    # ── Model router discovery ────────────────────────────────────
    # Discover pack-registered model routers (lisa.extensions.model_routers)
    # and wire the first non-None factory result into ContextSeams. When no
    # router pack is installed, or the manifest has no ``router:`` block, the
    # factories return None and model_router stays None — the turn loop falls
    # back to static resolution.
    def _discover_active_router() -> Any:
        if runtime.session.manifest is None:
            # A missing models.yaml is a valid first-run state: degrade to
            # "no router" instead of aborting the turn. The turn path's
            # resolve_model raises its own actionable error only when a
            # model is actually needed. Non-router approval turns (e.g.
            # /approve building hooks) must not require a manifest at all.
            if runtime.paths.models_path is None or not runtime.paths.models_path.exists():
                logger.debug(
                    "model manifest not found at %s; router discovery skipped",
                    runtime.paths.models_path,
                )
                return None
            runtime.session.manifest = load_manifest(
                {}, data_dir=runtime.paths.data_dir, path=runtime.paths.models_path
            )
        session_plan = getattr(runtime, "session_plan", None)
        for entry in discover_model_router_factories():
            try:
                # Some backends (notably the built-in lisa-router in
                # sdk/router) register a factory-factory that takes 0 args
                # and returns the real ``(manifest, session_plan) ->
                # ModelRouter | None`` factory. Detect the arity via
                # inspect.signature and resolve the double-indirection
                # without relying on TypeError catching (which can silently
                # swallow TypeErrors raised *inside* the factory body — see
                # the slice-provider fix for the same pattern).
                import inspect as _inspect

                try:
                    sig = _inspect.signature(entry.factory)
                    min_params = sum(
                        1
                        for p in sig.parameters.values()
                        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                        and p.default is p.empty
                    )
                except (ValueError, TypeError):
                    min_params = 2  # assume direct factory if signature fails

                if min_params == 0:
                    # Factory-factory: call with 0 args to get the real
                    # ``(manifest, session_plan) -> ModelRouter | None`` factory.
                    factory = entry.factory()  # type: ignore[call-arg]
                    router = factory(runtime.session.manifest, session_plan)  # type: ignore[misc]
                else:
                    # Direct factory: call with 2 args.
                    router = entry.factory(runtime.session.manifest, session_plan)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "model router factory %r failed: %s",
                    entry.pack_name,
                    exc,
                    exc_info=True,
                )
                continue
            if router is not None:
                logger.debug("active model router: %s", entry.pack_name)
                return router
        return None

    # ── Provider gateway discovery ────────────────────────────────
    # Discover pack-registered provider gateway providers
    # (lisa.extensions.provider_gateways) and wire the first non-None
    # result into ContextSeams. The SDK ships the reference backends
    # (OpenAI, Azure, Anthropic, Google, Bedrock) under this group;
    # third-party packs can register their own. When no provider is
    # discovered, provider_gateways stays None.
    def _discover_active_provider_gateways() -> Any:
        for entry in discover_provider_gateway_factories():
            try:
                loaded = entry.provider
                # The entry point may yield a factory-factory (zero-arg
                # callable returning the real ProviderGatewayProvider) or
                # the instance directly. Detect via inspect.signature,
                # same as the model-router pattern.
                import inspect as _inspect

                if callable(loaded) and not hasattr(loaded, "gateway_for"):
                    try:
                        sig = _inspect.signature(loaded)
                        min_params = sum(
                            1
                            for p in sig.parameters.values()
                            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                            and p.default is p.empty
                        )
                    except (ValueError, TypeError):
                        min_params = 0
                    if min_params == 0:
                        loaded = loaded()
                if loaded is not None:
                    logger.debug("active provider gateway provider: %s", entry.pack_name)
                    # The discovered provider returns gateway *classes*; the
                    # kernel calls ``gateway_for(provider).complete(...)`` at
                    # the model-call site. Wrap it in the host provider so that
                    # call receives a credential-bound, runtime-cached gateway
                    # *instance* instead. This is the one place env credentials
                    # meet the gateway seam.
                    return HostGatewayProvider(runtime=runtime, discovered_provider=loaded)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "provider gateway entry %r failed: %s",
                    entry.pack_name,
                    exc,
                    exc_info=True,
                )
                continue
        return None

    # ── Streaming request signal ───────────────────────────────────
    # The kernel calls ``gateway.complete_stream(request)`` directly. This
    # non-None sentinel preserves its existing streaming-request signal.
    streaming_cfg = runtime.config.streaming
    _complete_stream: Callable[..., Any] | None = None
    if streaming_cfg.enabled and streaming_cfg.mode in {"text", "hybrid"}:

        async def _stream_requested() -> None:
            return None

        _complete_stream = _stream_requested

    # ── Content loader (injected into context assembly) ───────────
    loader = content_loader(runtime)

    # ── System prompt assembly ────────────────────────────────────
    def _assemble_system_prompt(mode_instruction: str, stage_instruction: str | None) -> str:
        return assemble_system_prompt(
            mode_instruction,
            stage_instruction,
            loader,
            session_instruction=runtime.session_plan.instructions.primary_body,
            copilot_customizations=runtime.copilot_customizations_text,
        )

    # ── Dynamic user context assembly ─────────────────────────────────
    _budget_values = context_budget_values_from_config(runtime.config)

    def _assemble_user_context(
        mode_id: str,
        query: str,
        pre_turn_briefing: str,
        capability_manifest: Any = None,
    ) -> str:
        # ── Turn number annotation ──────────────────────────────────
        # Prepend the global turn index to the briefing so the model sees
        # its own turn number ("Turn 42") and can use it to address prior
        # turns via ``thread_read_range`` / the markdown render operation.
        # The number is computed at turn start in ``_execute_mode_turn``
        # and threaded through ``run_context`` — the same value stamped on
        # the thread entry, so the model's self-address matches the log.
        turn_index = ctx_dict.get("turn_index")
        if isinstance(turn_index, int):
            pre_turn_briefing = f"Turn {turn_index}\n\n{pre_turn_briefing}"

        # ── Workflow progress summary ────────────────────────────────
        # Derive the workflow view from the thread tail — the thread is
        # the single source of truth. The view is rebuilt every turn from
        # the append-only log; no mutable position store.
        from kernel.workflow_view import (
            derive_workflow_view,
            render_workflow_progress,
        )

        from host.thread import load_recent_thread_entries

        progress = ""
        thread_entries = load_recent_thread_entries(thread_path=runtime.paths.thread_path, limit=30)
        view = derive_workflow_view(thread_entries)
        if view is not None:
            progress = render_workflow_progress(view)
        if progress:
            pre_turn_briefing = f"{progress}\n\n{pre_turn_briefing}"

        # Build the UnifiedState from the runtime snapshot for the broker.
        # The item builder is invoked inside build_unified_state to
        # populate ``context_items`` — the single feed the
        # session-owned context mapper walks.
        state = build_unified_state(
            state_snapshot=runtime.session.state_snapshot,
            item_builder=_item_builder,
            runtime_context=_runtime_context,
        )

        def _session_context_loader() -> str:
            # Primary session instructions are system-level and never become
            # a budgeted dynamic-user-context section.
            return ""

        def _recent_thread_excerpt_builder(limit: int) -> str:
            return build_recent_thread_excerpt(
                thread_path=runtime.paths.thread_path,
                limit=limit,
            )

        def _thread_manifest_builder() -> str:
            from host.thread import build_thread_manifest

            return build_thread_manifest(
                thread_path=runtime.paths.thread_path,
                limit=30,
            )

        # Resolve the model context window for the session-owned context
        # mapper. The mapper needs the model's context window to compute
        # model-aware context budgets.
        context_plan = runtime.session_plan.manifest.context
        resolved_model = _resolve_model("default")
        model_context_window = getattr(resolved_model, "context_window", 0) or 0

        def _snapshot_sink(snapshot: Any) -> None:
            # Stash the mapper's ContextBudgetSnapshot on the runtime so
            # the returned TurnHooks.last_context_snapshot reflects the
            # most-recent context-mapper pass. Utilization-aware slice
            # providers read this via hooks.context.last_context_snapshot.
            runtime.session.last_context_snapshot = snapshot

        return assemble_dynamic_user_context(
            mode_id,
            query,
            pre_turn_briefing,
            content_loader=loader,
            user_slug=runtime.config.user_slug,
            timezone=runtime.config.timezone,
            state=state,
            session_context_loader=_session_context_loader,
            recent_thread_excerpt_builder=_recent_thread_excerpt_builder,
            thread_manifest_builder=_thread_manifest_builder,
            capability_manifest=capability_manifest,
            context_plan=context_plan,
            model_context_window_tokens=model_context_window,
            snapshot_sink=_snapshot_sink,
            context_budget_values=_budget_values,
        )

    # ── Operation execution ───────────────────────────────────────
    async def _execute_operation(
        operation_id: str,
        ctx: OperationExecutionContext,
    ) -> OperationResult:
        ctx.memory_recall_default_limit = 8
        return await execute_operation(operation_id, ctx)

    # ── Tool execution ─────────────────────────────────────────────
    async def _execute_tool(tool_id: str, tool_ctx: Any) -> ToolResult:
        # Delegate to the kernel's MCP tool pool when available. ``tool_ctx``
        # is either a ``ToolExecutionContext`` (from operation-host calls,
        # which carry ``.parameters``) or a plain ``dict`` of arguments
        # (from pack slice providers like the memory-consolidation slice,
        # which call ``execute_tool("memory.list", {…})`` directly). Resolve
        # the argument dict from both shapes so the MCP server receives the
        # caller's parameters instead of an empty dict.
        pool = runtime.resources.mcp_tool_pool
        if pool is not None:
            try:
                if isinstance(tool_ctx, dict):
                    arguments = tool_ctx
                else:
                    arguments = getattr(tool_ctx, "parameters", None) or {}
                result = await pool.call_tool(tool_id, arguments)
                # Adapt the MCP result to ToolResult. ``call_tool`` returns
                # an ``MCPToolResult`` with a string ``output_text`` (the
                # joined text content blocks); a defensive dict branch
                # covers any alternative seam. Do NOT stringify the whole
                # result object — that yields a non-JSON repr that breaks
                # callers (e.g. the memory-consolidation slice) which parse
                # ``output_text`` as JSON.
                if isinstance(result, dict):
                    return ToolResult(
                        tool_id=tool_id,
                        output_text=str(result.get("content", "")),
                        output_json=result,
                    )
                output_text = getattr(result, "output_text", "")
                is_error = bool(getattr(result, "is_error", False))
                if is_error:
                    return ToolResult(tool_id=tool_id, error=output_text or "tool error")
                return ToolResult(tool_id=tool_id, output_text=output_text or "")
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("tool dispatch %s failed: %s", tool_id, exc)
                return ToolResult(tool_id=tool_id, error=str(exc))
        return ToolResult(tool_id=tool_id, error="No MCP tool pool available")

    # ── Turn-complete hook ─────────────────────────────────────────

    # ── Auto-capture trigger detection (extracted to turn_hooks._auto_capture) ─
    # The trigger policy (commitment patterns, write-effect prefixes/suffixes,
    # and the detection function) now lives in host.turn_hooks._auto_capture
    # so it is independently testable without constructing a HostRuntime.
    # ``classify_memory_tier`` maps the trigger salience onto a retention tier
    # so workflow-state captures (write-effect tools, soft commitments) land
    # in ``daily``/``weekly`` rather than polluting ``long_term`` recall.
    from host.turn_hooks._auto_capture import (
        classify_memory_tier as _classify_memory_tier,
    )
    from host.turn_hooks._auto_capture import (
        should_auto_capture as _should_auto_capture,
    )

    # ── Auto-capture provider gateways ────────────────────────────
    # Resolve the provider gateways ONCE at hook-build time (same place the
    # ContextSeams below resolves them) and reuse the instance in the
    # post-turn auto-capture path. Re-discovering per capture is wasteful
    # and — worse — can return a different (possibly empty) result after
    # the turn ends, producing the "no provider gateway wired" distillation
    # failure observed 2026-08-25. The turn's own gateways are the correct
    # completion path for a learning distilled from that turn.
    _auto_capture_provider_gateways = _discover_active_provider_gateways()
    # Same snapshot discipline for the model router: resolve once at hook-build
    # time so the post-turn auto-capture context sees the same operation-scoped
    # routing rules the turn loop itself used. ``_wire_context_seams`` resolves
    # its own instance lazily (seams are assembled after hook construction);
    # reusing that lazy path here would re-run pack discovery per hook build,
    # so resolve eagerly next to the gateway snapshot. Router discovery degrades
    # to ``None`` when no router pack/manifest is present — auto-capture then
    # falls back to the operation's authored model, same as before.
    _auto_capture_model_router = _discover_active_router()

    async def _on_turn_complete(
        mode_id: str,
        query: str,
        response_text: str,
        messages: list[dict[str, Any]],
        effect_receipts: list[Any],
    ) -> None:
        # Update the mutable turn context dict.
        assistant_message_count = sum(
            1 for msg in messages if str(msg.get("role") or "").strip().lower() == "assistant"
        )
        ctx_dict["completion"] = {
            "mode_id": mode_id,
            "query": query,
            "response_preview": response_text[:240],
            "response_chars": len(response_text),
            "message_count": len(messages),
            "assistant_message_count": assistant_message_count,
        }

        # ── Opportunistic post-turn learning capture ──────────────
        # When auto-capture is enabled and a deterministic trigger fires,
        # distill the turn's query+response into durable facts and persist
        # to memory. A down memory backend or failed distillation is logged
        # and swallowed — it never blocks turn completion.
        if runtime.config.memory_auto_capture_enabled:
            should_capture, trigger_salience = _should_auto_capture(
                query, response_text, messages, effect_receipts=effect_receipts
            )
            if (
                should_capture
                and trigger_salience >= runtime.config.memory_auto_capture_min_salience
            ):
                memory_tier = _classify_memory_tier(
                    trigger_salience,
                    daily_threshold=runtime.config.memory_auto_capture_daily_threshold,
                    weekly_threshold=runtime.config.memory_auto_capture_weekly_threshold,
                )
                try:
                    capture_ctx = OperationExecutionContext(
                        workspace_id=runtime.session_plan.manifest.name,
                        user_id=runtime.config.user_slug,
                        user_slug=runtime.config.user_slug,
                        user_name=runtime.config.user_slug,
                        principal_key=runtime.config.user_slug,
                        conversation_id=runtime.config.surface_id,
                        surface_type=runtime.config.surface_type,
                        surface_id=runtime.config.surface_id,
                        memory_scope=runtime.config.memory_scope,
                        mode_id=mode_id,
                        query=query,
                        response=response_text,
                        # Pack config + router: parity with the turn-runner
                        # operation dispatch (``_build_entry_ctx`` /
                        # approval-resume ``common`` context). Without the
                        # resolver, ``synthesize_and_store_learning`` fell
                        # back to the 512-token module default instead of
                        # the session's ``memory_distill_max_tokens``; without
                        # the router, an operation-scoped rule (e.g.
                        # finances-profile routing of learning distills)
                        # never applied on the auto-capture path.
                        pack_configuration_resolver=pack_configuration_resolver,
                        model_router=_auto_capture_model_router,
                        host=build_callable_host(
                            gateway_provider_fn=_auto_capture_provider_gateways,
                            resolve_model_fn=_resolve_model,
                            read_authored_text_fn=getattr(
                                content_registry() if content_registry else None,
                                "read_authored_text",
                                None,
                            ),
                            execute_tool_fn=_execute_tool,
                            execute_operation_fn=_execute_operation,
                        ),
                        params={
                            "content": (f"User asked:\n{query}\n\nLisa answered:\n{response_text}"),
                            "salience": trigger_salience,
                            "memory_tier": memory_tier,
                        },
                    )
                    # Stamp the global turn index into the capture context so
                    # the distilled memory record carries a thread-turn
                    # provenance pointer (memory → thread linkage).
                    raw_turn_index = ctx_dict.get("turn_index")
                    if isinstance(raw_turn_index, int):
                        capture_ctx.extra["thread_turn"] = raw_turn_index
                    capture_result = await _execute_operation(
                        "synthesize_and_store_learning", capture_ctx
                    )
                    # Thread → memory linkage: record the minted memory id(s)
                    # on the turn context so ``append_thread_entry`` stamps
                    # them into the thread entry. The operation reports the
                    # distilled text; the storage id arrives via
                    # ``memory_ids`` when the memory backend echoes it.
                    minted = getattr(capture_result, "extra", {}).get("memory_ids") or []
                    if minted:
                        existing_ids = ctx_dict.get("memory_ids") or []
                        ctx_dict["memory_ids"] = [*existing_ids, *minted]
                    logger.debug(
                        "auto-capture stored learning salience=%.2f tier=%s mode=%s",
                        trigger_salience,
                        memory_tier,
                        mode_id,
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    # Surface structured context for operation-level failures
                    # so the "empty distillation output" class of silent
                    # auto-capture failures is diagnosable. The
                    # ``OperationExecutionError`` hierarchy carries the
                    # operation id and a typed error detail; the generic
                    # ``Exception`` fallback keeps the legacy message shape.
                    exc_type = type(exc).__name__
                    detail = str(exc)
                    op_id = getattr(exc, "operation_id", "") or ""
                    if not op_id:
                        # ``OperationExecutionError`` messages embed the
                        # operation id as ``"Operation '<id>' failed: ..."``;
                        # extract it for the structured log.
                        # lazy: stdlib import avoids top-level re cost on the success path
                        import re

                        m = re.search(r"Operation '([^']+)'", detail)
                        if m:
                            op_id = m.group(1)
                    logger.warning(
                        "auto-capture failed: operation=%s exc_type=%s detail=%s "
                        "mode=%s salience=%.2f memory_tier=%s",
                        op_id or "synthesize_and_store_learning",
                        exc_type,
                        detail[:300],
                        mode_id,
                        trigger_salience,
                        memory_tier,
                    )

    # ── UnifiedState provider callable ─────────────────────────────
    def _state_provider() -> Any:
        return build_unified_state(
            state_snapshot=runtime.session.state_snapshot,
            item_builder=_item_builder,
            runtime_context=_runtime_context,
        )

    # ── Last slice finished at ─────────────────────────────────────
    persisted_last_slice_finished_at: float | None = None
    if context is not None:
        raw = context.get("last_slice_finished_at")
        if isinstance(raw, (int, float)):
            persisted_last_slice_finished_at = float(raw)
    if runtime.session.last_slice_finished_at is not None:
        persisted_last_slice_finished_at = runtime.session.last_slice_finished_at

    # ── Slice callables ────────────────────────────────────────────
    async def _introspect_fn() -> Any:
        """Gather a live introspection snapshot for the self-model operation."""
        return await gather_introspection(
            mcp_pool=runtime.resources.mcp_tool_pool,
            log_path=runtime.paths.host_log_path,
            token_usage_path=runtime.paths.token_usage_path,
        )

    async def _run_self_model_slice() -> tuple[int, str, str]:
        """Run the deterministic self-model and write the vibe to state when degraded.

        Returns ``(work_items, last_item, note)``. The self-model is no-LLM
        and typically completes in <50ms. The vibe is only written to
        ``runtime.session.state_snapshot`` when ``operational_health`` is
        ``degraded`` or ``struggling`` — healthy state leaves the vibe empty
        so the context mapper injects zero self-awareness tokens (ambient
        silence when nothing is wrong).
        """
        try:
            ctx = OperationExecutionContext(
                operation_id="self_model",
                user_slug=runtime.config.user_slug,
                host=build_callable_host(
                    resolve_model_fn=_resolve_model,
                    introspect_fn=_introspect_fn,
                ),
                extra={"timezone": runtime.config.timezone},
            )
            result = await execute_operation("self_model", ctx)
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("self_model slice failed: %s", exc, exc_info=True)
            return 0, "", ""
        artifact_json = result.output_json
        if not isinstance(artifact_json, dict):
            return 0, "", ""
        health = str(artifact_json.get("operational_health") or "")
        if health not in ("degraded", "struggling"):
            return 0, "", ""
        vibe = result.output_text.strip()
        if not vibe:
            return 0, "", ""
        if runtime.session.state_snapshot is None:
            runtime.session.state_snapshot = {}
        # Emit the self-model vibe as a ContextItem record into the
        # single context_items — no arbitrary named bucket. The
        # mapper picks it up via context_candidates_from_items like every
        # other pack-contributed context item.
        from datetime import UTC, datetime

        from kernel.models import ContextItem

        evidence = ContextItem(
            item_id=f"state:self_model:{runtime.config.user_slug}",
            namespace="self_model",
            source="self_model",
            source_type="extension",
            evidence_type="runtime_self_model",
            title=f"Self-model (health={health})",
            body=vibe,
            collected_at=datetime.now(UTC).isoformat(),
            salience_score=0.8,
            confidence=0.9,
            pack="lisa-pack-introspect",
            tags=["self_model", health],
            metadata={"provenance": "system_observed", "operational_health": health},
        )
        catalog = runtime.session.state_snapshot.setdefault("context_items", [])
        if not isinstance(catalog, list):
            catalog = []
            runtime.session.state_snapshot["context_items"] = catalog
        # Replace any prior self-model evidence so only the freshest survives.
        catalog[:] = [
            e
            for e in catalog
            if not (
                (isinstance(e, dict) and e.get("item_id") == evidence.item_id)
                or getattr(e, "item_id", None) == evidence.item_id
            )
        ]
        catalog.append(evidence.model_dump(mode="json"))
        return 1, "self_model", f"self_model health={health}"

    async def _pre_turn_slice(policy: Any) -> Any:
        return await file_state_provider.slice(ctx=None, policy=policy)

    async def _post_turn_slice(policy: Any) -> Any:
        outcome = await file_state_provider.slice(ctx=None, policy=policy)
        runtime.session.last_slice_finished_at = __import__("time").monotonic()
        # Ambient self-awareness: run the deterministic self-model after file
        # compaction. Writes the vibe to state only when health is degraded or
        # struggling — healthy state produces no prompt tokens.
        sm_items, sm_last, sm_note = await _run_self_model_slice()
        if sm_items > 0:
            outcome = SliceOutcome(
                work_items_completed=outcome.work_items_completed + sm_items,
                elapsed_ms=outcome.elapsed_ms,
                last_item=sm_last or outcome.last_item,
                notes=(*outcome.notes, sm_note) if sm_note else outcome.notes,
            )
        return outcome

    async def _explicit_slice(policy: Any) -> Any:
        return await file_state_provider.slice(ctx=None, policy=policy)

    # ── Live stage telemetry ─────────────────────────────────────────
    # Forward the kernel's fine-grained stage events to the transport so the
    # CLI/web stage line can show model_call, tool_dispatch, and
    # fork-child progress as it happens. The kernel's ``_emit_stage`` helper
    # swallows hook failures, so a transport error never breaks the turn.
    async def _on_stage_event(label: str, data: dict[str, Any]) -> None:
        if transport is None:
            return
        await transport.on_stage(label, data)

    # ── Pack-provided slice providers ───────────────────────────────
    # Discover lisa.extensions.slice_providers entry points and chain any
    # non-None factory result ahead of the host's deterministic
    # FileBackedStateProvider slices. The first applicable provider wins;
    # later entries are not consulted for the same slot. A provider
    # returns None when its session-plan tuning is disabled, in which
    # case it is skipped and the host falls back to
    # FileBackedStateProvider alone — no behavior change.
    from kernel.slice_provider_discovery import discover_slice_provider_factories

    _slice_provider_hooks_view = _SliceProviderHooksView(
        runtime=runtime,
    )
    _slice_provider_hooks_view.execute_operation = _execute_operation  # type: ignore[attr-defined]
    _slice_provider_hooks_view.execute_tool = _execute_tool  # type: ignore[attr-defined]

    pack_slice_entries = discover_slice_provider_factories()
    pack_slice_providers: list[Any] = []
    for entry in pack_slice_entries:
        try:
            # The entry point is expected to be the factory callable
            # ``(turn_hooks, session_plan) -> provider | None``. Some packs
            # register a *factory-factory*
            # that takes 0 args and returns the real factory. Detect the
            # arity via ``inspect.signature`` and resolve the double-
            # indirection without relying on ``TypeError`` catching (which
            # can silently swallow TypeErrors raised *inside* the factory
            # body — e.g. an import-time TypeError in a dependency — and
            # re-attempt the 0-arg call, masking the real failure).
            import inspect as _inspect

            try:
                sig = _inspect.signature(entry.factory)
                min_params = sum(
                    1
                    for p in sig.parameters.values()
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                    and p.default is p.empty
                )
            except (ValueError, TypeError):
                min_params = 2  # assume direct factory if signature fails

            if min_params == 0:
                # Factory-factory: call with 0 args to get the real
                # ``(turn_hooks, session_plan) -> provider | None`` factory.
                factory = entry.factory()  # type: ignore[call-arg]
                provider = factory(  # type: ignore[misc]
                    _slice_provider_hooks_view,
                    runtime.session_plan,
                )
            else:
                # Direct factory: call with 2 args.
                provider = entry.factory(
                    # The provider reads last_context_snapshot off the
                    # view and invokes execute_operation to fire the
                    # summarize operation.
                    _slice_provider_hooks_view,
                    runtime.session_plan,
                )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("slice provider %s factory failed: %s", entry.pack_name, exc)
            continue
        if provider is not None:
            pack_slice_providers.append(provider)

    def _chain_pack_slices(
        base_slice: Callable[[Any], Awaitable[Any]],
        trigger_name: str,
    ) -> Callable[[Any], Awaitable[Any]]:
        """Wrap a base slice so pack providers for this trigger run first.

        Each pack provider is asked to run; if it reports work or raises,
        later providers and the base slice still run (pack providers and
        deterministic file compaction are complementary, not exclusive).
        """
        applicable: list[Any] = []
        for provider in pack_slice_providers:
            # The provider's trigger preference is read from its config;
            # if it does not expose one, assume post_turn.
            cfg = getattr(provider, "_config", None)
            trigger = str(getattr(cfg, "trigger", "post_turn") or "post_turn")
            if trigger == trigger_name:
                applicable.append(provider)
        if not applicable:
            return base_slice

        async def _chained(policy: Any) -> Any:

            total_work = 0
            last_item = ""
            notes: list[str] = []
            elapsed_ms = 0.0
            for provider in applicable:
                try:
                    outcome = await provider.slice(ctx=None, policy=policy)
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning(
                        "pack slice provider %s failed: %s",
                        type(provider).__name__,
                        exc,
                    )
                    continue
                if outcome is None:
                    continue
                total_work += int(getattr(outcome, "work_items_completed", 0) or 0)
                if getattr(outcome, "last_item", ""):
                    last_item = str(outcome.last_item)
                outcome_notes = getattr(outcome, "notes", ()) or ()
                notes.extend(str(n) for n in outcome_notes)
                elapsed_ms += float(getattr(outcome, "elapsed_ms", 0.0) or 0.0)
            base_outcome = await base_slice(policy)
            total_work += int(getattr(base_outcome, "work_items_completed", 0) or 0)
            if getattr(base_outcome, "last_item", ""):
                last_item = str(base_outcome.last_item)
            base_notes = getattr(base_outcome, "notes", ()) or ()
            notes.extend(str(n) for n in base_notes)
            elapsed_ms += float(getattr(base_outcome, "elapsed_ms", 0.0) or 0.0)
            return SliceOutcome(
                work_items_completed=total_work,
                elapsed_ms=elapsed_ms,
                last_item=last_item,
                notes=tuple(notes) if notes else (),
            )

        return _chained

    _pre_turn_slice_chained = _chain_pack_slices(_pre_turn_slice, "pre_turn")
    _post_turn_slice_chained = _chain_pack_slices(_post_turn_slice, "post_turn")
    _explicit_slice_chained = _chain_pack_slices(_explicit_slice, "explicit")

    # ── Token-usage telemetry subscription ────────────────────────
    # The kernel publishes one ``TokenUsageEvent`` per model call on the
    # ``lisa.token_usage`` bus topic (per-iteration calls, forced-final,
    # and terminal-stage completions). The runtime's durable
    # ``TokenUsageTelemetryHandler`` appends every event to
    # ``token-usage.jsonl``; introspection (/tokens lifetime view and the
    # 24h summary) reads the resulting file. A fresh ``InProcessEventBus``
    # is created per hook build, so the handler must be re-subscribed
    # every turn. Idempotent: duplicate events dedupe on the handler's
    # idempotency-key... in-memory set (best-effort). A subscription
    # failure is logged and swallowed — telemetry must never break
    # hook construction.
    def _wire_context_seams() -> ContextSeams:
        seams = ContextSeams(
            content_registry=cast("Any", content_registry() or None),
            model_router=_discover_active_router(),
            provider_gateways=_discover_active_provider_gateways(),
            extension_item_builder=_item_builder,
        )
        bus = getattr(seams, "event_bus", None)
        handler = runtime.services.token_usage_handler
        if bus is not None and handler is not None and hasattr(bus, "subscribe"):
            try:
                bus.subscribe(TOKEN_USAGE_TOPIC, handler.publish)
            except Exception:  # pylint: disable=broad-except
                logger.warning(
                    "token-usage telemetry subscribe failed; usage will not be persisted to %s",
                    runtime.paths.token_usage_path,
                    exc_info=True,
                )
        return seams

    return TurnHooks(
        models=ModelHooks(
            resolve_model=_resolve_model,
            complete=None,
            complete_stream=_complete_stream,
        ),
        tools=ToolHooks(
            mcp_tool_pool=runtime.resources.mcp_tool_pool,
            mcp_tool_servers=runtime.resources.mcp_cold_configs,
        ),
        operations=OperationHooks(
            execute_operation=_execute_operation,
        ),
        context=ContextHooks(
            assemble_system_prompt=_assemble_system_prompt,
            assemble_user_context=_assemble_user_context,
            # Stash the latest ContextBudgetSnapshot here each turn via the
            # _snapshot_sink wired into _assemble_user_context. Utilization-
            # aware slice providers read this via hooks.context.last_context_snapshot.
            last_context_snapshot=runtime.session.last_context_snapshot,
        ),
        lifecycle=LifecycleHooks(
            on_turn_complete=_on_turn_complete,
            slice_tuning=slice_tuning,
            pre_turn_slice=_pre_turn_slice_chained,
            post_turn_slice=_post_turn_slice_chained,
            explicit_slice=_explicit_slice_chained,
            last_slice_finished_at=persisted_last_slice_finished_at,
            # Forward the kernel's fine-grained stage events to the transport so
            # the CLI/web stage line can show model_call, tool_dispatch,
            # and fork-child progress as it happens. The kernel's _emit_stage
            # helper swallows hook failures, so a transport error never breaks
            # the turn.
            on_stage=_on_stage_event,
            # ── Operator signal (pause/stop/terminate) ──────────────────────
            # The transport sets ``runtime.session.operator_signal_value`` from a
            # key binding; the kernel polls it once per iteration. ``None`` means
            # the host does not expose operator signals (zero overhead); we
            # always wire it so every transport can participate.
            operator_signal=lambda: runtime.session.operator_signal_value,
        ),
        services=HostServiceHandles(
            state_provider=_state_provider,
            pack_configuration_resolver=pack_configuration_resolver,
            approval_service=approval_service,
            recovery_service=runtime.services.recovery_service,
            code_change_artifact_store=runtime.services.code_change_artifact_store,
            data_dir=str(runtime.paths.data_dir),
        ),
        identity=TurnIdentity(
            workspace_id=runtime.session_plan.manifest.name,
            user_id=runtime.config.user_slug,
            user_slug=runtime.config.user_slug,
            user_name=runtime.config.user_slug,
            principal_key=runtime.config.user_slug,
            conversation_id=runtime.config.surface_id,
            surface_type=runtime.config.surface_type,
            surface_id=runtime.config.surface_id,
            memory_scope=runtime.config.memory_scope,
            session_plan_digest=runtime.session_plan.plan_digest,
        ),
        recall=RecallHooks(
            enabled=runtime.config.memory_recall_enabled,
            limit=runtime.config.memory_recall_pre_turn_limit,
            min_query_chars=runtime.config.memory_recall_min_query_chars,
        ),
        # Wire the content registry into context_seams so the turn runner's
        # _build_minimal_context can resolve read_authored_text_fn for
        # operations that load authored prompts (e.g. synthesize_and_store_learning).
        # The registry is the same module-level singleton that
        # content_loader() reads; setup_framework(runtime) populates it at host
        # startup. When unset, content_registry stays None and operations fail at
        # call time with their authored-text error (preserving graceful failure).
        #
        # Also wire the discovered model router (lisa.extensions.model_routers)
        # so the turn loop can do per-iteration model selection. When no router
        # pack is active (or the manifest has no ``router:`` block),
        # ``model_router`` stays ``None`` and the loop falls back to static
        # resolution — preserving today's behavior.
        context_seams=_wire_context_seams(),
        # ── Workflow view seam ───────────────────────────────────────
        # The kernel reads the workflow view (derived from the thread) via
        # this hook. The view is rebuilt on each call — a pure read from
        # the thread tail. No updater is needed: the thread is append-only
        # and the view is derived, never mutated.
        workflow=WorkflowHooks(
            workflow_position=lambda: _derive_view(runtime),
            workflow_position_updater=None,
        ),
    )

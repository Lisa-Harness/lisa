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

from kernel.introspect import gather_introspection
from kernel.model_router_discovery import discover_model_router_factories
from kernel.models import OperationExecutionContext, OperationResult, ToolResult
from kernel.operations.registry import execute_operation
from kernel.pack_configuration import SessionPackConfigurationResolver
from kernel.seams.protocols import ContentRegistry
from kernel.slice import SliceOutcome, SliceTuning
from kernel.turn_runner import CheckpointTuning, ContextSeams, TurnHooks

from hosts.core.context_assembly import (
    assemble_dynamic_user_context,
    assemble_system_prompt,
)
from hosts.core.model_runtime import StubModelDefinition, load_manifest, resolve_little_lisa_model
from hosts.core.state_provider import FileBackedStateProvider, build_unified_state
from hosts.core.thread import build_recent_thread_excerpt, load_thread_digest

if TYPE_CHECKING:
    from hosts.core.callbacks import TransportCallbacks
    from hosts.core.runtime import HostRuntime

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
    keeps it current each turn), ``write_turn_history_fn`` (the host
    writer closure), and ``execute_operation`` (the kernel operation
    dispatcher). This avoids handing the factory the full ``TurnHooks``
    before it is fully constructed.
    """

    __slots__ = ("_runtime", "_write_turn_history", "execute_operation")

    def __init__(
        self,
        *,
        runtime: HostRuntime,
        write_turn_history: Callable[[list[dict[str, Any]]], None],
    ) -> None:
        self._runtime = runtime
        self._write_turn_history = write_turn_history
        # Lazy: wire execute_operation when the factory asks for it via
        # attribute access; the closure is defined later in build_turn_hooks
        # but factories only invoke it at slice time, not at construction.

    @property
    def last_context_snapshot(self) -> Any:
        return self._runtime.last_context_snapshot

    @property
    def write_turn_history_fn(self) -> Callable[[list[dict[str, Any]]], None]:
        return self._write_turn_history

    def __getattr__(self, name: str) -> Any:
        # Fall back to runtime attributes for seams the factory may need
        # (e.g. execute_operation closure is set on this view explicitly
        # once build_turn_hooks constructs it). Until then, raise.
        raise AttributeError(f"_SliceProviderHooksView has no attribute {name!r}")


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
    from hosts.core.providers import complete_live, complete_live_stream

    ctx_dict = context if context is not None else {}
    pack_configuration_resolver = SessionPackConfigurationResolver(
        runtime.session_plan.manifest.resources.packs.configuration
    )
    approval_service = runtime.approval_service

    # ── Slice tuning ──────────────────────────────────────────────
    slice_config = runtime.config.slice_config
    slice_tuning = SliceTuning.from_mapping(slice_config.to_tuning_dict())

    history_plan = runtime.session_plan.manifest.context.history
    digest_max_lines = history_plan.digest_max_lines
    digest_summary_max_chars = history_plan.digest_summary_max_chars

    # ── State provider (FileBackedStateProvider) ──────────────────
    file_state_provider = FileBackedStateProvider(
        thread_path=runtime.thread_path,
        max_thread_entries=runtime.config.thread_max_entries,
        digest_max_lines=digest_max_lines,
        digest_summary_max_chars=digest_summary_max_chars,
    )

    # ── Model resolution ──────────────────────────────────────────
    def _resolve_model(model_id: str) -> StubModelDefinition:
        if runtime.manifest is None:
            runtime.manifest = load_manifest(
                {}, data_dir=runtime.data_dir, path=runtime.models_path
            )
        return resolve_little_lisa_model(
            model_id,
            runtime_state={"manifest": runtime.manifest},
            data_dir=runtime.data_dir,
            path=runtime.models_path,
        )

    # ── Model router discovery ────────────────────────────────────
    # Discover pack-registered model routers (lisa.extensions.model_routers)
    # and wire the first non-None factory result into ContextSeams. When no
    # router pack is installed, or the manifest has no ``router:`` block, the
    # factories return None and model_router stays None — the turn loop falls
    # back to static resolution.
    def _discover_active_router() -> Any:
        if runtime.manifest is None:
            runtime.manifest = load_manifest(
                {}, data_dir=runtime.data_dir, path=runtime.models_path
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
                    router = factory(runtime.manifest, session_plan)  # type: ignore[misc]
                else:
                    # Direct factory: call with 2 args.
                    router = entry.factory(runtime.manifest, session_plan)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "model router factory %r failed: %s",
                    entry.pack_name,
                    exc,
                    exc_info=True,
                )
                continue
            if router is not None:
                logger.info("active model router: %s", entry.pack_name)
                return router
        return None

    # ── Completion ────────────────────────────────────────────────
    async def _complete(**kwargs: Any) -> Any:
        return await complete_live(
            runtime=runtime,
            transport=transport,
            **kwargs,
        )

    # ── Streaming completion ──────────────────────────────────────
    # Wired only when the host config has text/hybrid streaming enabled.
    # The turn runner checks ``hooks.complete_stream is not None`` to decide
    # whether to take the streaming path; leaving it None when streaming is
    # off preserves the non-streaming behavior.
    streaming_cfg = runtime.config.streaming
    _complete_stream: Callable[..., Any] | None = None
    if streaming_cfg.enabled and streaming_cfg.mode in {"text", "hybrid"}:

        async def _complete_stream_fn(**kwargs: Any) -> Any:
            async for chunk in complete_live_stream(
                runtime=runtime,
                transport=transport,
                **kwargs,
            ):
                yield chunk

        _complete_stream = _complete_stream_fn

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

    # ── Dynamic user context assembly ─────────────────────────────
    def _assemble_user_context(
        mode_id: str,
        query: str,
        pre_turn_briefing: str,
        capability_manifest: Any = None,
    ) -> str:
        # Build the UnifiedState from the runtime snapshot for the broker.
        state = build_unified_state(
            state_snapshot=runtime.state_snapshot,
        )

        def _session_context_loader() -> str:
            # Primary session instructions are system-level and never become
            # a budgeted dynamic-user-context section.
            return ""

        def _thread_digest_loader() -> str:
            return load_thread_digest(thread_path=runtime.thread_path)

        def _recent_thread_excerpt_builder(limit: int) -> str:
            return build_recent_thread_excerpt(
                thread_path=runtime.thread_path,
                limit=limit,
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
            # providers read this via hooks.last_context_snapshot.
            runtime.last_context_snapshot = snapshot

        return assemble_dynamic_user_context(
            mode_id,
            query,
            pre_turn_briefing,
            content_loader=loader,
            user_slug=runtime.config.user_slug,
            timezone=runtime.config.timezone,
            state=state,
            session_context_loader=_session_context_loader,
            thread_digest_loader=_thread_digest_loader,
            recent_thread_excerpt_builder=_recent_thread_excerpt_builder,
            capability_manifest=capability_manifest,
            context_plan=context_plan,
            model_context_window_tokens=model_context_window,
            snapshot_sink=_snapshot_sink,
        )

    # ── Operation execution ───────────────────────────────────────
    async def _execute_operation(
        operation_id: str,
        ctx: OperationExecutionContext,
    ) -> OperationResult:
        ctx.memory_recall_default_limit = 8
        return await execute_operation(operation_id, ctx)

    # ── Tool execution ─────────────────────────────────────────────
    async def _execute_tool(tool_id: str, _tool_ctx: Any) -> ToolResult:
        # Delegate to the kernel's MCP tool pool when available.
        pool = runtime.mcp_tool_pool
        if pool is not None:
            try:
                result = await pool.call_tool(tool_id, {})
                # Adapt the MCP result to ToolResult.
                if isinstance(result, dict):
                    return ToolResult(
                        tool_id=tool_id,
                        output_text=str(result.get("content", "")),
                        output_json=result if isinstance(result, dict) else {},
                    )
                return ToolResult(tool_id=tool_id, output_text=str(result))
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("tool dispatch %s failed: %s", tool_id, exc)
                return ToolResult(tool_id=tool_id, error=str(exc))
        return ToolResult(tool_id=tool_id, error="No MCP tool pool available")

    # ── Turn-complete hook ─────────────────────────────────────────

    # ── Auto-capture trigger detection (extracted to turn_hooks._auto_capture) ─
    # The trigger policy (commitment patterns, write-effect prefixes/suffixes,
    # and the detection function) now lives in hosts.core.turn_hooks._auto_capture
    # so it is independently testable without constructing a HostRuntime.
    from hosts.core.turn_hooks._auto_capture import should_auto_capture as _should_auto_capture

    async def _on_turn_complete(
        mode_id: str,
        query: str,
        response_text: str,
        messages: list[dict[str, Any]],
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
                query, response_text, messages
            )
            if (
                should_capture
                and trigger_salience >= runtime.config.memory_auto_capture_min_salience
            ):
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
                        complete_fn=_complete,
                        resolve_model_fn=_resolve_model,
                        read_authored_text_fn=getattr(
                            content_registry() if content_registry else None,
                            "read_authored_text",
                            None,
                        ),
                        execute_tool_fn=_execute_tool,
                        execute_operation_fn=_execute_operation,
                        params={
                            "content": (
                                f"User asked:\n{query}\n\nLisa answered:\n{response_text}"
                            ),
                            "salience": trigger_salience,
                            "memory_tier": "long_term",
                        },
                    )
                    await _execute_operation(
                        "synthesize_and_store_learning", capture_ctx
                    )
                    logger.debug(
                        "auto-capture stored learning salience=%.1f mode=%s",
                        trigger_salience,
                        mode_id,
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning("auto-capture failed: %s", exc)

    # ── UnifiedState provider callable ─────────────────────────────
    def _state_provider() -> Any:
        return build_unified_state(
            state_snapshot=runtime.state_snapshot,
        )

    # ── Last slice finished at ─────────────────────────────────────
    persisted_last_slice_finished_at: float | None = None
    if context is not None:
        raw = context.get("last_slice_finished_at")
        if isinstance(raw, (int, float)):
            persisted_last_slice_finished_at = float(raw)
    if runtime.last_slice_finished_at is not None:
        persisted_last_slice_finished_at = runtime.last_slice_finished_at

    # ── Slice callables ────────────────────────────────────────────
    async def _introspect_fn() -> Any:
        """Gather a live introspection snapshot for the self-model operation."""
        return await gather_introspection(
            mcp_pool=runtime.mcp_tool_pool,
            log_path=runtime.host_log_path,
            token_usage_path=runtime.token_usage_path,
        )

    async def _run_self_model_slice() -> tuple[int, str, str]:
        """Run the deterministic self-model and write the vibe to state when degraded.

        Returns ``(work_items, last_item, note)``. The self-model is no-LLM
        and typically completes in <50ms. The vibe is only written to
        ``runtime.state_snapshot`` when ``operational_health`` is
        ``degraded`` or ``struggling`` — healthy state leaves the vibe empty
        so the context mapper injects zero self-awareness tokens (ambient
        silence when nothing is wrong).
        """
        try:
            ctx = OperationExecutionContext(
                operation_id="self_model",
                user_slug=runtime.config.user_slug,
                introspect_fn=_introspect_fn,
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
        if runtime.state_snapshot is None:
            runtime.state_snapshot = {}
        runtime.state_snapshot["self_model_vibe"] = vibe
        return 1, "self_model", f"self_model health={health}"

    async def _pre_turn_slice(policy: Any) -> Any:
        return await file_state_provider.slice(ctx=None, policy=policy)

    async def _post_turn_slice(policy: Any) -> Any:
        outcome = await file_state_provider.slice(ctx=None, policy=policy)
        runtime.last_slice_finished_at = __import__("time").monotonic()
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

    # ── Turn-history write-back seam ─────────────────────────────────
    # Used by utilization-aware compaction (lisa-pack-compaction) and the
    # /summarize-history command to persist a compacted turn-history list
    # back to the live thread JSONL. When the host has no thread (e.g.
    # ``--no-thread``), this is None and callers no-op with a warning.
    def _write_turn_history(new_history: list[dict[str, Any]]) -> None:
        thread_path = runtime.thread_path
        if thread_path is None:
            raise RuntimeError("write_turn_history: no thread path configured")
        import json as _json

        thread_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [_json.dumps(entry, ensure_ascii=False) for entry in new_history]
        thread_path.write_text(
            ("\n".join(lines) + "\n") if lines else "",
            encoding="utf-8",
        )

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
    # returns None when its session-plan tuning is disabled (e.g.
    # compaction.enabled=False), in which case it is skipped and the host
    # falls back to FileBackedStateProvider alone — no behavior change.
    from kernel.slice_provider_discovery import discover_slice_provider_factories

    _slice_provider_hooks_view = _SliceProviderHooksView(
        runtime=runtime, write_turn_history=_write_turn_history
    )
    _slice_provider_hooks_view.execute_operation = _execute_operation  # type: ignore[attr-defined]

    pack_slice_entries = discover_slice_provider_factories()
    pack_slice_providers: list[Any] = []
    for entry in pack_slice_entries:
        try:
            # The entry point is expected to be the factory callable
            # ``(turn_hooks, session_plan) -> provider | None``. Some packs
            # (notably lisa-pack-compaction) register a *factory-factory*
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
                    # The provider reads last_context_snapshot and
                    # write_turn_history_fn off the view and invokes
                    # execute_operation to fire the summarize operation.
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
        later providers and the base slice still run (compaction and
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

    return TurnHooks(
        resolve_model=_resolve_model,
        complete=_complete,
        complete_stream=_complete_stream,
        mcp_tool_pool=runtime.mcp_tool_pool,
        mcp_tool_servers=runtime.mcp_cold_configs,
        execute_operation=_execute_operation,
        assemble_system_prompt=_assemble_system_prompt,
        assemble_user_context=_assemble_user_context,
        on_turn_complete=_on_turn_complete,
        slice_tuning=slice_tuning,
        pre_turn_slice=_pre_turn_slice_chained,
        post_turn_slice=_post_turn_slice_chained,
        explicit_slice=_explicit_slice_chained,
        last_slice_finished_at=persisted_last_slice_finished_at,
        state_provider=_state_provider,
        pack_configuration_resolver=pack_configuration_resolver,
        approval_service=approval_service,
        code_change_artifact_store=runtime.code_change_artifact_store,
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
        memory_recall_enabled=runtime.config.memory_recall_enabled,
        memory_recall_pre_turn_limit=runtime.config.memory_recall_pre_turn_limit,
        memory_recall_min_query_chars=runtime.config.memory_recall_min_query_chars,
        # Stash the latest ContextBudgetSnapshot here each turn via the
        # _snapshot_sink wired into _assemble_user_context. Utilization-
        # aware slice providers read this via hooks.last_context_snapshot.
        last_context_snapshot=runtime.last_context_snapshot,
        # Host-owned write-back seam for the live thread JSONL. Used by
        # the compaction pack's slice and by /summarize-history.
        write_turn_history_fn=_write_turn_history,
        checkpoint_tuning=CheckpointTuning(
            enabled=True,
            on_stage_transition=True,
            every_n_turns=10,
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
        context_seams=ContextSeams(
            content_registry=cast("Any", content_registry() or None),
            model_router=_discover_active_router(),
        ),
        # Forward the kernel's fine-grained stage events to the transport so
        # the CLI/web stage line can show model_call, tool_dispatch,
        # and fork-child progress as it happens. The kernel's _emit_stage
        # helper swallows hook failures, so a transport error never breaks
        # the turn.
        on_stage=_on_stage_event,
    )

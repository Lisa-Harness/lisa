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
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from kernel.capability_binding import CapabilityRef
from kernel.decision_telemetry import TOOL_TELEMETRY_TOPIC, TURN_DECISION_TOPIC
from kernel.model_router_discovery import discover_model_router_factories
from kernel.models import (
    OperationExecutionContext,
    OperationResult,
    ToolResult,
    resolve_turn_index,
)
from kernel.operations.registry import execute_operation
from kernel.provider_gateway_discovery import discover_provider_gateway_factories
from kernel.sandbox_floor import (
    SandboxFloorContext,
    collect_pack_floor,
    discover_sandbox_floor_contributions,
    discover_sandbox_implementations,
    merge_sandbox_floor,
    resolve_sandbox_owner,
)
from kernel.seams.host_adapter import build_callable_host
from kernel.seams.protocols import ContentRegistry, ExtensionItemBuilder
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
from sdk.providers.token_counting import count_request_tokens

from runtime.capability_bindings import SessionCapabilityBindings
from runtime.code_artifacts import verify_checked_commit
from runtime.content_roots import load_disabled_pack_names
from runtime.context_assembly import (
    assemble_dynamic_user_context,
    assemble_system_prompt,
    context_budget_values_from_config,
)
from runtime.introspect import gather_introspection
from runtime.model_runtime import StubModelDefinition, load_manifest, resolve_model
from runtime.provider_endpoints import proxy_route_view
from runtime.state_provider import FileBackedStateProvider, build_unified_state
from runtime.thread import (
    build_open_task_messages,
    build_recent_thread_excerpt,
    project_thread_messages,
)
from runtime.turn_continuation import persist_request

if TYPE_CHECKING:
    from runtime.callbacks import TransportCallbacks
    from runtime.runtime import HostRuntime

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
    from kernel.context.content_registry import get_content_registry

    return get_content_registry()


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


_VIEW_CHUNK_TURNS = 256


@dataclass
class _PlanScan:
    """What ``_derive_view`` learned about the thread up to ``next_turn``."""

    next_turn: int
    plan_turn: int | None
    view: Any


# The thread is append-only, so the latest-plan turn found for a prefix stays
# valid; later calls only scan newly appended turns. Keyed by thread path.
_PLAN_SCANS: dict[Path, _PlanScan] = {}
_MAX_PLAN_SCANS = 64


def _latest_plan_turn(thread_path: Path, first_turn: int, last_turn: int) -> int | None:
    """Return the newest global turn in ``[first_turn, last_turn]`` carrying a plan."""
    from kernel.workflow_view import has_workflow_plan

    from runtime.thread import load_thread_range

    end_turn = last_turn
    while end_turn >= first_turn:
        start_turn = max(first_turn, end_turn - _VIEW_CHUNK_TURNS + 1)
        chunk = load_thread_range(thread_path=thread_path, start_turn=start_turn, end_turn=end_turn)
        for offset in range(len(chunk) - 1, -1, -1):
            if has_workflow_plan(str(chunk[offset].get("response_text") or "")):
                return start_turn + offset
        end_turn = start_turn - 1
    return None


def _derive_view(runtime: HostRuntime) -> Any:
    """Derive the workflow view from the latest plan and subsequent thread entries.

    A pure read from the lossless thread log — no mutable position store.
    Returns a ``WorkflowView`` or ``None`` when no active workflow exists.
    The result is reused until the thread grows, and only new turns are
    searched for a newer plan.
    """
    from kernel.workflow_view import derive_workflow_view

    from runtime.thread import load_thread_range, next_turn_index

    thread_path = runtime.paths.thread_path
    next_turn = next_turn_index(thread_path=thread_path)
    cached = _PLAN_SCANS.get(thread_path) if thread_path is not None else None
    if cached is not None and cached.next_turn == next_turn:
        return cached.view
    if cached is None or cached.next_turn > next_turn:
        scan_from, plan_turn = 0, None
    else:
        scan_from, plan_turn = cached.next_turn, cached.plan_turn
    end_turn = next_turn - 1
    if thread_path is not None:
        newer_plan = _latest_plan_turn(thread_path, scan_from, end_turn)
        if newer_plan is not None:
            plan_turn = newer_plan
    entries: list[dict[str, Any]] = []
    if plan_turn is not None:
        # Same window the backwards chunk walk always produced: whole
        # chunks counted back from the newest turn, so mode run-lengths that
        # began before the plan are still visible.
        start_turn = end_turn - _VIEW_CHUNK_TURNS + 1
        while start_turn > plan_turn:
            start_turn -= _VIEW_CHUNK_TURNS
        entries = load_thread_range(
            thread_path=thread_path, start_turn=max(0, start_turn), end_turn=end_turn
        )
    view = derive_workflow_view(entries)
    if thread_path is not None:
        if len(_PLAN_SCANS) >= _MAX_PLAN_SCANS and thread_path not in _PLAN_SCANS:
            _PLAN_SCANS.clear()
        _PLAN_SCANS[thread_path] = _PlanScan(next_turn=next_turn, plan_turn=plan_turn, view=view)
    return view


def _resolve_factory_arity(factory: Any, default_params: int) -> int:
    """Count the factory's required positional params, or ``default_params``.

    Some entry-point factories register a *factory-factory* (zero-arg
    callable returning the real factory); this detects the indirection via
    ``inspect.signature`` without relying on TypeError catching, which can
    silently swallow TypeErrors raised *inside* the factory body (see the
    slice-provider fix for the same pattern).
    """
    import inspect as _inspect

    try:
        sig = _inspect.signature(factory)
        return sum(
            1
            for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty
        )
    except (ValueError, TypeError):
        return default_params  # assume direct factory if signature fails


def _resolve_router_for_runtime(runtime: HostRuntime) -> Any:
    """Discover pack-registered model routers (lisa.extensions.model_routers).

    Wire the first non-None factory result into ``ContextSeams``. When no
    router pack is installed, or the manifest has no ``router:`` block, the
    factories return None and model_router stays None — the turn loop falls
    back to static resolution.
    """
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
            if _resolve_factory_arity(entry.factory, default_params=2) == 0:
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


def _resolve_gateway_provider_for_runtime(runtime: HostRuntime) -> Any:
    """Discover pack-registered provider gateway providers.

    The first non-None result is wrapped in ``HostGatewayProvider`` and wired
    into ``ContextSeams``. The SDK ships the reference backends (OpenAI,
    Azure, Anthropic, Google, Bedrock) under this group; third-party packs
    can register their own. When no provider is discovered,
    provider_gateways stays None.
    """
    from runtime.gateway_provider import HostGatewayProvider

    for entry in discover_provider_gateway_factories():
        try:
            loaded = entry.provider
            # The entry point may yield a factory-factory (zero-arg
            # callable returning the real ProviderGatewayProvider) or the
            # instance directly.
            gateway_indirect = callable(loaded) and not hasattr(loaded, "gateway_for")
            if gateway_indirect and _resolve_factory_arity(loaded, default_params=0) == 0:
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


def _discover_pack_slice_providers(
    *,
    view: Any,
    session_plan: Any,
) -> list[Any]:
    """Discover ``lisa.extensions.slice_providers`` factory results.

    Chain any non-None factory result ahead of the host's deterministic
    FileBackedStateProvider slices. The first applicable provider wins;
    later entries are not consulted for the same slot. A provider
    returns None when its session-plan tuning is disabled, in which
    case it is skipped and the host falls back to
    FileBackedStateProvider alone — no behavior change.
    """
    from kernel.slice_provider_discovery import discover_slice_provider_factories

    providers: list[Any] = []
    for entry in discover_slice_provider_factories():
        try:
            # The entry point is expected to be the factory callable
            # ``(turn_hooks, session_plan) -> provider | None``. Some packs
            # register a *factory-factory* that takes 0 args and returns
            # the real factory. _resolve_factory_arity detects the double-
            # indirection without relying on ``TypeError`` catching (which
            # can silently swallow TypeErrors raised *inside* the factory
            # body — e.g. an import-time TypeError in a dependency — and
            # re-attempt the 0-arg call, masking the real failure).
            if _resolve_factory_arity(entry.factory, default_params=2) == 0:
                # Factory-factory: call with 0 args to get the real
                # ``(turn_hooks, session_plan) -> provider | None`` factory.
                factory = entry.factory()  # type: ignore[call-arg]
                provider = factory(view, session_plan)  # type: ignore[misc]
            else:
                # Direct factory: call with 2 args. The provider reads
                # last_context_snapshot off the view and invokes
                # execute_operation to fire the summarize operation.
                provider = entry.factory(view, session_plan)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("slice provider %s factory failed: %s", entry.pack_name, exc)
            continue
        if provider is not None:
            providers.append(provider)
    return providers


def _chain_pack_slices(
    base_slice: Callable[[Any], Awaitable[Any]],
    trigger_name: str,
    pack_slice_providers: list[Any],
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


async def _maybe_auto_capture(
    *,
    runtime: HostRuntime,
    mode_id: str,
    query: str,
    response_text: str,
    messages: list[dict[str, Any]],
    effect_receipts: list[Any],
    ctx_dict: dict[str, Any],
    capability_binding_provider: Any,
    model_router: Any,
    provider_gateways: Any,
    resolve_model: Callable[[str], Any],
    execute_tool: Any,
    execute_operation: Any,
) -> None:
    """Distill and persist a learning when the auto-capture trigger fires.

    When auto-capture is enabled and a deterministic trigger fires, the
    turn's query+response is distilled into durable facts and persisted to
    memory. A down memory backend or failed distillation is logged and
    swallowed — it never blocks turn completion.
    """
    from runtime.turn_hooks._auto_capture import classify_memory_tier as _classify_memory_tier
    from runtime.turn_hooks._auto_capture import should_auto_capture as _should_auto_capture

    if not runtime.config.memory_auto_capture_enabled:
        return
    should_capture, trigger_salience = _should_auto_capture(
        query, response_text, messages, effect_receipts=effect_receipts
    )
    if not should_capture or trigger_salience < runtime.config.memory_auto_capture_min_salience:
        return

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
            # Pack config + router: parity with the turn-runner operation
            # dispatch (``_build_entry_ctx`` / approval-resume ``common``
            # context). Without the resolver,
            # ``synthesize_and_store_learning`` fell back to the 512-token
            # module default instead of the session's
            # ``memory_distill_max_tokens``; without the router, an
            # operation-scoped rule (e.g. finances-profile routing of
            # learning distills) never applied on the auto-capture path.
            capability_binding_provider=capability_binding_provider,
            model_router=model_router,
            host=build_callable_host(
                gateway_provider_fn=provider_gateways,
                resolve_model_fn=resolve_model,
                read_authored_text_fn=getattr(
                    content_registry() if content_registry else None,
                    "read_authored_text",
                    None,
                ),
                execute_tool_fn=execute_tool,
                execute_operation_fn=execute_operation,
            ),
            params={
                "content": (f"User asked:\n{query}\n\nLisa answered:\n{response_text}"),
                "salience": trigger_salience,
                "memory_tier": memory_tier,
            },
        )
        # Stamp the global turn index into the capture context so the
        # distilled memory record carries a thread-turn provenance
        # pointer (memory → thread linkage).
        raw_turn_index = ctx_dict.get("turn_index")
        if isinstance(raw_turn_index, int):
            capture_ctx.extra["thread_turn"] = raw_turn_index
        capture_result = await execute_operation("synthesize_and_store_learning", capture_ctx)
        # Thread → memory linkage: record the minted memory id(s) on the
        # turn context so ``append_thread_entry`` stamps them into the
        # thread entry. The operation reports the distilled text; the
        # storage id arrives via ``memory_ids`` when the memory backend
        # echoes it.
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
        # Surface structured context for operation-level failures so the
        # "empty distillation output" class of silent auto-capture
        # failures is diagnosable. The ``OperationExecutionError``
        # hierarchy carries the operation id and a typed error detail;
        # the generic ``Exception`` fallback keeps the legacy message
        # shape.
        exc_type = type(exc).__name__
        detail = str(exc)
        op_id = getattr(exc, "operation_id", "") or ""
        if not op_id:
            # ``OperationExecutionError`` messages embed the operation id
            # as ``"Operation '<id>' failed: ..."``; extract it for the
            # structured log.
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


async def _execute_mcp_tool(runtime: HostRuntime, tool_id: str, tool_ctx: Any) -> ToolResult:
    """Dispatch one MCP tool call through the runtime's tool pool.

    ``tool_ctx`` is either a ``ToolExecutionContext`` (from operation-host
    calls, which carry ``.parameters``) or a plain ``dict`` of arguments
    (from pack slice providers like the memory-consolidation slice, which
    call ``execute_tool("memory.list", {…})`` directly). The argument dict is
    resolved from both shapes so the MCP server receives the caller's
    parameters instead of an empty dict.
    """
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
            # Input-required seam: an MCP server that pauses mid-call for
            # user input (the input-required / multi-round-tool-result
            # shape) must surface as a structured pause, not flatten into
            # ``output_text``. The state rides in ``extra`` so downstream
            # consumers (turn runner, pack slices) can distinguish "the
            # tool needs input" from "the tool answered".
            input_required = bool(getattr(result, "input_required", False))
            extra: dict[str, Any] = {}
            if input_required:
                extra["input_required"] = True
                input_request = getattr(result, "input_request", None)
                if isinstance(input_request, dict):
                    extra["input_request"] = input_request
            if is_error:
                return ToolResult(
                    tool_id=tool_id,
                    error=output_text or "tool error",
                    extra=extra,
                )
            return ToolResult(
                tool_id=tool_id,
                output_text=output_text or "",
                extra=extra,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("tool dispatch %s failed: %s", tool_id, exc)
            return ToolResult(tool_id=tool_id, error=str(exc))
    return ToolResult(tool_id=tool_id, error="No MCP tool pool available")


def _wire_runtime_context_seams(
    *,
    runtime: HostRuntime,
    item_builder: Any,
    discover_router: Callable[[], Any],
    discover_gateways: Callable[[], Any],
) -> ContextSeams:
    """Assemble ``ContextSeams`` and subscribe the durable telemetry handlers.

    The kernel publishes one ``TokenUsageEvent`` per model call on the
    ``lisa.token_usage`` bus topic (per-iteration calls, forced-final, and
    terminal-stage completions). The runtime's durable
    ``TokenUsageTelemetryHandler`` appends every event to
    ``token-usage.jsonl``; introspection (/tokens lifetime view and the
    24h summary) reads the resulting file. It also publishes one
    ``turn.decision`` payload per iteration and one ``tool.telemetry``
    payload per dispatch; ``DecisionTelemetryHandler`` appends those to
    ``decisions.jsonl`` so the lifecycle reasoning and the dispatch ledger
    are durable evidence rather than dropped events. A fresh
    ``InProcessEventBus`` is created per hook build, so every handler must be
    re-subscribed every turn. A subscription failure is logged and swallowed —
    telemetry must never break hook construction.
    """
    seams = ContextSeams(
        content_registry=cast("Any", content_registry() or None),
        model_router=discover_router(),
        provider_gateways=discover_gateways(),
        extension_item_builder=item_builder,
    )
    bus = getattr(seams, "event_bus", None)
    if bus is not None and hasattr(bus, "subscribe"):
        _subscribe_telemetry(
            bus=bus,
            topic=TOKEN_USAGE_TOPIC,
            handler=getattr(runtime.services, "token_usage_handler", None),
            method_name="publish",
            destination=runtime.paths.token_usage_path,
            label="token-usage",
        )
        decision_handler = getattr(runtime.services, "decision_telemetry_handler", None)
        _subscribe_telemetry(
            bus=bus,
            topic=TURN_DECISION_TOPIC,
            handler=decision_handler,
            method_name="publish_turn_decision",
            destination=runtime.paths.decisions_path,
            label="turn.decision",
        )
        _subscribe_telemetry(
            bus=bus,
            topic=TOOL_TELEMETRY_TOPIC,
            handler=decision_handler,
            method_name="publish_tool_telemetry",
            destination=runtime.paths.decisions_path,
            label="tool.telemetry",
        )
    return seams


def _subscribe_telemetry(
    *,
    bus: Any,
    topic: str,
    handler: Any,
    method_name: str,
    destination: Path,
    label: str,
) -> None:
    """Subscribe one durable telemetry handler method, best-effort.

    A missing handler or method means the host did not wire that sink; that is
    a configuration choice, not an error. A raising ``subscribe`` is logged and
    swallowed so telemetry can never break hook construction.
    """
    method = getattr(handler, method_name, None) if handler is not None else None
    if method is None:
        return
    try:
        bus.subscribe(topic, method)
    except Exception:  # pylint: disable=broad-except
        logger.warning(
            "%s telemetry subscribe failed; records will not be persisted to %s",
            label,
            destination,
            exc_info=True,
        )


async def _run_self_model_slice_impl(
    runtime: HostRuntime,
    resolve_model: Callable[[str], Any],
    introspect_fn: Any,
) -> tuple[int, str, str]:
    """Run the deterministic self-model and write the vibe to state when degraded.

    Returns ``(work_items, last_item, note)``. The self-model is no-LLM
    and typically completes in <50ms. The vibe is only written to
    ``runtime.session.state_snapshot`` when ``operational_health`` is
    ``degraded`` or ``struggling`` — healthy state leaves the vibe empty
    so the context mapper injects zero self-awareness tokens (ambient
    silence when nothing is wrong). Any other outcome retracts the previous
    turn's item so a recovered runtime stops reporting stale degradation.
    """
    item_id = f"state:self_model:{runtime.config.user_slug}"
    try:
        ctx = OperationExecutionContext(
            operation_id="self_model",
            user_slug=runtime.config.user_slug,
            host=build_callable_host(
                resolve_model_fn=resolve_model,
                introspect_fn=introspect_fn,
            ),
            extra={"timezone": runtime.config.timezone},
        )
        result = await execute_operation("self_model", ctx)
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("self_model slice failed: %s", exc, exc_info=True)
        _drop_context_item(runtime, item_id)
        return 0, "", ""
    artifact_json = result.output_json
    health = (
        str(artifact_json.get("operational_health") or "")
        if isinstance(artifact_json, dict)
        else ""
    )
    vibe = result.output_text.strip() if health in ("degraded", "struggling") else ""
    if not vibe:
        _drop_context_item(runtime, item_id)
        return 0, "", ""
    _emit_context_item(
        runtime,
        item_id=item_id,
        namespace="self_model",
        title=f"Self-model (health={health})",
        body=vibe,
        evidence_type="runtime_self_model",
        tags=["self_model", health],
        metadata={"provenance": "system_observed", "operational_health": health},
    )
    return 1, "self_model", f"self_model health={health}"


def _emit_context_item(
    runtime: HostRuntime,
    *,
    item_id: str,
    namespace: str,
    title: str,
    body: str,
    evidence_type: str,
    tags: list[str],
    metadata: dict[str, Any],
    salience: float = 0.8,
) -> None:
    """Publish one extension context item into the session's evidence lane.

    The single ``context_items`` list is the only lane: the session-owned
    context mapper picks entries up via ``context_candidates_from_items`` like
    any other pack-contributed record, and there is no arbitrary named bucket
    to keep in sync. Replacing an item with the same ``item_id`` means only the
    freshest observation of a given kind survives.
    """
    from datetime import UTC, datetime

    from kernel.models import ContextItem

    if runtime.session.state_snapshot is None:
        runtime.session.state_snapshot = {}
    evidence = ContextItem(
        item_id=item_id,
        namespace=namespace,
        source=namespace,
        source_type="extension",
        evidence_type=evidence_type,
        title=title,
        body=body,
        collected_at=datetime.now(UTC).isoformat(),
        salience_score=salience,
        confidence=0.9,
        pack="lisa-pack-introspect",
        tags=tags,
        metadata=metadata,
    )
    _drop_context_item(runtime, item_id)
    runtime.session.state_snapshot["context_items"].append(evidence.model_dump(mode="json"))


def _drop_context_item(runtime: HostRuntime, item_id: str) -> None:
    """Remove any context item with ``item_id`` from the session evidence lane.

    Guarantees ``state_snapshot["context_items"]`` exists as a list afterwards.
    """
    if runtime.session.state_snapshot is None:
        runtime.session.state_snapshot = {}
    catalog = runtime.session.state_snapshot.setdefault("context_items", [])
    if not isinstance(catalog, list):
        catalog = []
        runtime.session.state_snapshot["context_items"] = catalog
    catalog[:] = [
        entry
        for entry in catalog
        if not (
            (isinstance(entry, dict) and entry.get("item_id") == item_id)
            or getattr(entry, "item_id", None) == item_id
        )
    ]


def _run_behavioral_signals_slice_impl(
    runtime: HostRuntime,
) -> tuple[int, str, str]:
    """Scan recent turns for behavioural patterns worth surfacing.

    Returns ``(work_items, last_item, note)``. Silence is the common case: a
    nominal window emits no ``ContextItem`` and therefore costs no prompt
    tokens, matching the self-model's ambient-silence contract.

    Independently gated from ``operational_health`` on purpose. Health answers
    "is the host degraded?" (memory, tasks, MCP, log errors); a streak of
    blocked turns on a perfectly healthy process is still the thing an
    operator most wants surfaced, and gating it behind health would hide
    exactly that case.

    Budget: the scan is bounded by turns, records, and bytes. Measured on a
    live store (~128 KiB per thread entry), a 12-turn window plus a 256 KiB
    ledger read costs ~34 ms, which fits the slice budget. Widening the window
    is what ``/forensics`` and ``turn_forensics`` are for.
    """
    from sdk.forensics import compute_behavioral_signals, render_signals
    from sdk.thread import ThreadStore

    from runtime.thread import get_thread_redaction_policy

    item_id = f"state:behavioural_signals:{runtime.config.user_slug}"
    try:
        store = ThreadStore(runtime.paths.thread_path, redaction=get_thread_redaction_policy())
        signals = compute_behavioral_signals(
            store,
            decisions_path=runtime.paths.decisions_path,
            token_usage_path=runtime.paths.token_usage_path,
            user_slug=runtime.config.user_slug,
        )
    except Exception as exc:  # pylint: disable=broad-except
        # A slice must never break the turn; a failed scan is simply no signal,
        # and a finding from an earlier scan must not be presented as current.
        logger.debug("behavioural-signals slice failed: %s", exc, exc_info=True)
        _drop_context_item(runtime, item_id)
        return 0, "", ""

    body = render_signals(signals)
    if not body:
        # A recovered window must retract the previous turn's finding.
        _drop_context_item(runtime, item_id)
        return 0, "", ""

    _emit_context_item(
        runtime,
        item_id=item_id,
        namespace="behavioural_signals",
        title=f"Observed patterns ({signals.window_turns} turns)",
        body=body,
        evidence_type="runtime_behavioural_signals",
        tags=["behavioural_signals"],
        metadata={
            "provenance": "system_observed",
            "window_turns": signals.window_turns,
            "counters": signals.counters,
        },
        salience=0.75,
    )
    return 1, "behavioural_signals", f"behavioural_signals count={len(signals.signals)}"


def _assemble_dynamic_user_context(
    *,
    runtime: HostRuntime,
    ctx_dict: dict[str, Any],
    resolve_model: Callable[[str], Any],
    item_builder: Any,
    runtime_context: dict[str, Any],
    loader: Callable[[str], str],
    budget_values: Any,
    mode_id: str,
    query: str,
    pre_turn_briefing: str,
    capability_manifest: Any,
) -> str:
    """Assemble the budgeted dynamic user context for one turn.

    Annotates the briefing with the global turn number and the workflow
    progress (derived from the thread tail), builds the ``UnifiedState``,
    and hands the session-owned context mapper its loaders/sink seam
    callables.
    """
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
    # Derive the workflow view from thread history — the thread is
    # the single source of truth. The view is rebuilt every turn from
    # the append-only log; no mutable position store.
    from kernel.workflow_view import (
        render_workflow_progress,
    )

    progress = ""
    view = _derive_view(runtime)
    if view is not None:
        progress = render_workflow_progress(view)
    if progress:
        pre_turn_briefing = f"{progress}\n\n{pre_turn_briefing}"

    # Build the UnifiedState from the runtime snapshot. The item builder
    # is invoked inside build_unified_state to
    # populate ``context_items`` — the single feed the
    # session-owned context mapper walks. Thread-derived digest items
    # (task state, recent history, open objectives) join the same feed so
    # the retrieval lane renders them at turn start.
    from runtime.thread import get_thread_redaction_policy

    state = build_unified_state(
        state_snapshot=runtime.session.state_snapshot,
        item_builder=item_builder,
        runtime_context=runtime_context,
        thread_path=runtime.paths.thread_path,
        thread_redaction=get_thread_redaction_policy(),
        thread_digest_settings=runtime.session_plan.manifest.context.history.digest,
        live_state=runtime.session.live_workflow_state,
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
        from runtime.thread import build_thread_manifest

        return build_thread_manifest(
            thread_path=runtime.paths.thread_path,
            limit=30,
        )

    # Resolve the model context window for the session-owned context
    # mapper. The mapper needs the model's context window to compute
    # model-aware context budgets.
    context_plan = runtime.session_plan.manifest.context
    resolved_model = resolve_model("default")
    model_context_window = _model_context_window_tokens(resolved_model)

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
        context_budget_values=budget_values,
    )


class _WiredHooks:
    """Bundle of the host-specific callables ``build_turn_hooks`` wires up.

    The nested functions are closures over the runtime; this bundle carries
    them to the module-level :func:`_assemble_turn_hooks` that composes the
    final ``TurnHooks`` dataclass.
    """

    __slots__ = (
        "approval_service",
        "assemble_system_prompt",
        "assemble_user_context",
        "capability_binding_provider",
        "complete_stream",
        "execute_operation",
        "explicit_slice",
        "last_slice_finished_at",
        "on_delta",
        "on_stage",
        "on_turn_complete",
        "post_turn_slice",
        "pre_turn_slice",
        "resolve_model",
        "slice_tuning",
        "state_provider",
        "wire_context_seams",
    )

    def __init__(
        self,
        *,
        resolve_model: Any,
        complete_stream: Any,
        assemble_system_prompt: Any,
        assemble_user_context: Any,
        execute_operation: Any,
        on_turn_complete: Any,
        on_stage: Any,
        on_delta: Any,
        state_provider: Any,
        pre_turn_slice: Any,
        post_turn_slice: Any,
        explicit_slice: Any,
        capability_binding_provider: Any,
        approval_service: Any,
        last_slice_finished_at: Any,
        slice_tuning: Any,
        wire_context_seams: Any,
    ) -> None:
        self.resolve_model = resolve_model
        self.complete_stream = complete_stream
        self.assemble_system_prompt = assemble_system_prompt
        self.assemble_user_context = assemble_user_context
        self.execute_operation = execute_operation
        self.on_turn_complete = on_turn_complete
        self.on_stage = on_stage
        self.on_delta = on_delta
        self.state_provider = state_provider
        self.pre_turn_slice = pre_turn_slice
        self.post_turn_slice = post_turn_slice
        self.explicit_slice = explicit_slice
        self.capability_binding_provider = capability_binding_provider
        self.approval_service = approval_service
        self.last_slice_finished_at = last_slice_finished_at
        self.slice_tuning = slice_tuning
        self.wire_context_seams = wire_context_seams


def _coerce_turn_index(raw: Any) -> int | None:
    """Coerce a turn-context value into an attributable turn index.

    ``turn_index`` arrives from ``run_context``, which hosts populate with the
    global thread ordinal computed at turn start. ``None`` means the work is
    not attributable to a thread turn (off-turn slice work, or a host with no
    thread) and stays ``None`` rather than collapsing to ``0`` — which is the
    thread's first real turn. Anything unusable degrades to ``None``.
    """
    return resolve_turn_index(raw)


def _assemble_turn_hooks(
    *,
    runtime: HostRuntime,
    wired: _WiredHooks,
    task_id: str = "",
    task_goal: str = "",
    turn_index: int | None = None,
) -> TurnHooks:
    """Compose the final ``TurnHooks`` dataclass from the wired callables.

    Pure assembly: every hook callable was built by ``build_turn_hooks``;
    this function only maps them onto the kernel's hook dataclasses.
    """
    turn_identity = TurnIdentity(
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
        task_id=task_id,
        turn_index=turn_index,
        task_goal="diagnostic" if task_goal == "diagnostic" else "implementation",
    )
    artifact_store = runtime.services.code_change_artifact_store
    return TurnHooks(
        models=ModelHooks(
            resolve_model=wired.resolve_model,
            complete=None,
            complete_stream=wired.complete_stream,
            on_delta=wired.on_delta,
            record_request=partial(persist_request, runtime.paths.thread_path),
            count_request_tokens=count_request_tokens,
        ),
        tools=ToolHooks(
            mcp_tool_pool=runtime.resources.mcp_tool_pool,
            mcp_tool_servers=runtime.resources.mcp_cold_configs,
            post_dispatch_result=runtime.decorate_pack_tool_result,
        ),
        operations=OperationHooks(
            execute_operation=wired.execute_operation,
        ),
        context=ContextHooks(
            assemble_system_prompt=wired.assemble_system_prompt,
            assemble_user_context=wired.assemble_user_context,
            # Stash the latest ContextBudgetSnapshot here each turn via the
            # _snapshot_sink wired into _assemble_user_context. Utilization-
            # aware slice providers read this via hooks.context.last_context_snapshot.
            last_context_snapshot=runtime.session.last_context_snapshot,
            prior_messages=partial(
                build_open_task_messages,
                thread_path=runtime.paths.thread_path,
                budget_bytes=_effective_replay_budget_bytes(runtime),
            ),
            project_messages=partial(
                project_thread_messages, thread_path=runtime.paths.thread_path
            ),
        ),
        lifecycle=LifecycleHooks(
            on_turn_complete=wired.on_turn_complete,
            slice_tuning=wired.slice_tuning,
            pre_turn_slice=wired.pre_turn_slice,
            post_turn_slice=wired.post_turn_slice,
            explicit_slice=wired.explicit_slice,
            last_slice_finished_at=wired.last_slice_finished_at,
            # Forward the kernel's fine-grained stage events to the transport so
            # the CLI/web stage line can show model_call, tool_dispatch,
            # and fork-child progress as it happens. The kernel's _emit_stage
            # helper swallows hook failures, so a transport error never breaks
            # the turn.
            on_stage=wired.on_stage,
            # ── Operator signal (pause/stop/terminate) ──────────────────────
            # The transport sets ``runtime.session.operator_signal_value`` from a
            # key binding; the kernel polls it once per iteration. ``None`` means
            # the host does not expose operator signals (zero overhead); we
            # always wire it so every transport can participate.
            operator_signal=lambda: runtime.session.operator_signal_value,
        ),
        services=HostServiceHandles(
            state_provider=wired.state_provider,
            capability_binding_provider=wired.capability_binding_provider,
            approval_service=wired.approval_service,
            recovery_service=runtime.services.recovery_service,
            code_change_artifact_store=artifact_store,
            verify_task_completion=partial(
                verify_checked_commit,
                store=artifact_store,
                identity=turn_identity,
            )
            if artifact_store is not None
            else None,
            data_dir=str(runtime.paths.data_dir),
        ),
        identity=turn_identity,
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
        context_seams=wired.wire_context_seams(),
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


def _bind_state_provider(
    runtime: HostRuntime,
    item_builder: Any,
    runtime_context: dict[str, Any],
) -> Callable[[], Any]:
    """Return a zero-arg UnifiedState provider reading the live session snapshot.

    The provider must resolve ``runtime.session.state_snapshot`` at CALL
    time (not bind time) — the snapshot mutates between turns.
    """

    def _state_provider() -> Any:
        from runtime.thread import get_thread_redaction_policy

        return build_unified_state(
            state_snapshot=runtime.session.state_snapshot,
            item_builder=item_builder,
            runtime_context=runtime_context,
            thread_path=runtime.paths.thread_path,
            thread_redaction=get_thread_redaction_policy(),
            thread_digest_settings=runtime.session_plan.manifest.context.history.digest,
            live_state=runtime.session.live_workflow_state,
        )

    return _state_provider


def _assemble_runtime_system_prompt(
    runtime: HostRuntime,
    loader: Callable[[str], str],
    mode_instruction: str,
    stage_instruction: str | None,
) -> str:
    """Assemble the system prompt from mode/stage instructions and session context."""
    return assemble_system_prompt(
        mode_instruction,
        stage_instruction,
        loader,
        session_instruction=runtime.session_plan.instructions.primary_body,
        copilot_customizations=runtime.copilot_customizations_text,
    )


def _runtime_model_resolver(runtime: HostRuntime, model_id: str) -> StubModelDefinition:
    """Resolve one model id against the runtime's models manifest.

    Loads the manifest lazily on first resolution (a missing models.yaml
    surfaces when a model is actually needed, not at hook-build time).
    """
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


def _model_context_window_tokens(model_def: Any) -> int:
    """Return the model's declared context window, or the shared default.

    Mirrors ``kernel.context.context_budget.model_context_window_tokens``:
    discovery cannot always determine a window (e.g. Ollama ``:cloud``
    models), and a raw ``0`` violates the mapper request's ``ge=1`` contract.
    """
    from kernel.context.context_budget import model_context_window_tokens as _kernel_default

    return _kernel_default(model_def)


def _build_slice_callables(
    runtime: HostRuntime,
    file_state_provider: FileBackedStateProvider,
    resolve_model: Callable[[str], Any],
) -> tuple[
    Callable[[], Any],
    Callable[[], Any],
    Callable[[Any], Any],
    Callable[[Any], Any],
    Callable[[Any], Any],
]:
    """Build the slice hook callables: introspection, self-model, file slices.

    The three file-backed slice entry points delegate to the
    FileBackedStateProvider; the post-turn variant additionally stamps
    ``last_slice_finished_at`` and folds in the deterministic self-model
    (ambient self-awareness — writes the vibe to state only when health
    is degraded or struggling; healthy state produces no prompt tokens).
    """

    async def _introspect_fn() -> Any:
        """Gather a live introspection snapshot for the self-model operation."""
        return await gather_introspection(
            mcp_pool=runtime.resources.mcp_tool_pool,
            proxy_pool=runtime.resources.proxy_pool,
            log_path=runtime.paths.host_log_path,
            token_usage_path=runtime.paths.token_usage_path,
        )

    async def _run_self_model_slice() -> tuple[int, str, str]:
        """Run the deterministic self-model (see _run_self_model_slice_impl)."""
        return await _run_self_model_slice_impl(runtime, resolve_model, _introspect_fn)

    async def _pre_turn_slice(policy: Any) -> Any:
        return await file_state_provider.slice(ctx=None, policy=policy)

    async def _run_behavioral_signals_slice() -> tuple[int, str, str]:
        """Scan recent turns for behavioural patterns (see its impl)."""
        return _run_behavioral_signals_slice_impl(runtime)

    async def _post_turn_slice(policy: Any) -> Any:
        outcome = await file_state_provider.slice(ctx=None, policy=policy)
        runtime.session.last_slice_finished_at = __import__("time").monotonic()
        outcome = _compose_context_items_outcome(outcome, await _run_self_model_slice())
        return _compose_context_items_outcome(outcome, await _run_behavioral_signals_slice())

    async def _explicit_slice(policy: Any) -> Any:
        return await file_state_provider.slice(ctx=None, policy=policy)

    return _introspect_fn, _run_self_model_slice, _pre_turn_slice, _post_turn_slice, _explicit_slice


def _compose_context_items_outcome(outcome: Any, contributor: tuple[int, str, str]) -> Any:
    """Fold one context-lane contributor's slice result into a slice outcome.

    A contributor (the deterministic self-model, the behavioural-signal scan)
    reports ``(work_items, last_item, note)``. Zero work -- the common case,
    since both are silent when nothing is wrong -- returns the base outcome
    unchanged, which is what keeps an ordinary turn free of prompt tokens.
    """
    sm_items, sm_last, sm_note = contributor
    if sm_items > 0:
        outcome = SliceOutcome(
            work_items_completed=outcome.work_items_completed + sm_items,
            elapsed_ms=outcome.elapsed_ms,
            last_item=sm_last or outcome.last_item,
            notes=(*outcome.notes, sm_note) if sm_note else outcome.notes,
        )
    return outcome


def _persisted_last_slice_finished_at(
    runtime: HostRuntime,
    context: dict[str, Any] | None,
) -> float | None:
    """Resolve the last slice-finished-at timestamp (session beats context)."""
    persisted: float | None = None
    if context is not None:
        raw = context.get("last_slice_finished_at")
        if isinstance(raw, (int, float)):
            persisted = float(raw)
    if runtime.session.last_slice_finished_at is not None:
        persisted = runtime.session.last_slice_finished_at
    return persisted


def _stream_requested_signal(runtime: HostRuntime) -> Callable[..., Any] | None:
    """Return the streaming-request sentinel callable, or None.

    The kernel calls ``gateway.complete_stream(request)`` directly. A
    non-None sentinel here preserves its existing streaming-request
    signal: streaming enabled in text/hybrid mode wires a no-op async
    callable; other modes disable streaming entirely.
    """
    streaming_cfg = runtime.config.streaming
    if streaming_cfg.enabled and streaming_cfg.mode in {"text", "hybrid"}:

        async def _stream_requested() -> None:
            return None

        return _stream_requested
    return None


def _build_item_builder_context(
    runtime: HostRuntime,
    resolve_model: Callable[[str], Any],
    extra_item_builders: list[ExtensionItemBuilder] | None = None,
) -> tuple[ExtensionItemBuilder, dict[str, Any]]:
    """Build the shared extension item builder and its runtime context.

    The builder is constructed once so every ``build_unified_state`` call
    and the ``ContextSeams`` injection share one instance. It discovers
    ``lisa.extensions.context_items`` entry points from installed packs
    (minus disabled ones) and tags each contributed record with its
    source pack name. The runtime context hands pack entry-point
    functions access to host resources.

    ``extra_item_builders`` — optional host-supplied builders (the VS Code
    sidecar's editor-context bridge). When present, a
    ``CompositeExtensionItemBuilder`` wraps the entry-point builder first
    (pack items win the ``item_id`` dedup) followed by the host builders,
    matching the ``DelegatingTransport`` composition pattern.
    """
    disabled_packs = load_disabled_pack_names(runtime.paths.data_dir)
    entry_builder = EntryPointExtensionItemBuilder(
        excluded_packs=disabled_packs,
    )
    if extra_item_builders:
        from sdk.context.composite_item_builder import CompositeExtensionItemBuilder

        # lazy: defers the composite import to the host-supplied path
        item_builder: ExtensionItemBuilder = CompositeExtensionItemBuilder(
            [entry_builder, *extra_item_builders]
        )
    else:
        item_builder = entry_builder
    runtime_context: dict[str, Any] = {
        "runtime": runtime,
        "resolve_model": resolve_model,
    }
    return item_builder, runtime_context


def _declared_config_owners(runtime: HostRuntime) -> set[str]:
    return runtime.session_plan.manifest.declared_pack_owners()


def _merged_sandbox_floor(
    runtime: HostRuntime,
) -> tuple[str | None, dict[str, Any] | None, tuple[str, ...]]:
    """Resolve the sandbox owner + the exclusive-merged floor for the session.

    Follows the memory-capability shape: the session's explicit
    ``providers.capabilities.sandbox`` selection wins, a single registered
    implementation auto-owns, ambiguity raises. Contributions are merged
    exclusively onto the harness-authored floor
    (``resources.packs.configuration[owner]["sandbox"]["floor"]``) and the
    final mapping is validated by the implementation (bwrap's
    ``floor_from_config`` semantics) so the merge never produces a floor
    the enforcer cannot parse.

    Returns ``(owner_pack_name, merged_floor_or_None, contributing_packs)``.
    ``owner_pack_name`` is ``None`` when no implementation is registered
    (e.g. a Windows host without bubblewrap): contributions are still
    validated pack-to-pack — a manifest conflict is a load-time authoring
    error regardless of platform — but the merged floor is inert (same
    posture as configuration naming absent packs).
    """
    manifest = runtime.session_plan.manifest
    pack_configurations = manifest.resources.packs.configuration
    selected = str(manifest.providers.capabilities.get("sandbox") or "")
    disabled = load_disabled_pack_names(runtime.paths.data_dir)

    implementations = discover_sandbox_implementations(excluded_packs=disabled)
    owner = resolve_sandbox_owner(implementations, selected)
    contribution_entries = discover_sandbox_floor_contributions(excluded_packs=disabled)
    context = SandboxFloorContext(
        data_dir=str(runtime.paths.data_dir),
        workspace_root=_session_workspace_root(runtime),
    )
    if owner is None:
        if selected:
            logger.warning(
                "Session selects sandbox capability owner %r but no matching "
                "sandbox implementation is active; sandbox policy is inert "
                "this session.",
                selected,
            )
        # No sandbox implementation is active (e.g. a Windows host without
        # bubblewrap). Pack-vs-pack agreement is still validated — a manifest
        # conflict is an authoring bug on every platform — but the merged
        # floor is inert (same posture as configuration naming absent
        # packs), so the authored-floor anchor requirement does not apply:
        # there is no enforced posture to anchor.
        contributions = collect_pack_floor(contribution_entries, pack_configurations, context)
        merge_sandbox_floor({}, contributions)
        return None, None, ()

    entry_config = pack_configurations.get(owner.pack_name)
    authored: dict[str, Any] | None = None
    if isinstance(entry_config, Mapping) and isinstance(entry_config.get("sandbox"), Mapping):
        authored = dict(entry_config["sandbox"].get("floor") or {})

    contributions = collect_pack_floor(contribution_entries, pack_configurations, context)

    def _validate(floor: Mapping[str, Any]) -> None:
        # owner is non-None here; the local name keeps mypy happy about the
        # dataclass field access inside a closure.
        assert owner is not None
        owner.validate_floor(floor)

    merge = merge_sandbox_floor(authored, contributions, validator=_validate)
    if merge.contributing_packs:
        logger.info(
            "Sandbox floor contributions merged from packs: %s (owner %s).",
            ", ".join(merge.contributing_packs),
            owner.pack_name,
        )
    return owner.pack_name, dict(merge.floor), merge.contributing_packs


def _session_workspace_root(runtime: HostRuntime) -> str:
    """Best-effort session workspace root for contribution factories."""
    configurations = runtime.session_plan.manifest.resources.packs.configuration
    for entry in configurations.values():
        if isinstance(entry, Mapping):
            root = entry.get("workspace_root")
            if isinstance(root, str) and root:
                return root
    return ""


def _materialize_sandbox_floor(
    configuration: Mapping[str, Mapping[str, Any]],
    owner_pack_name: str | None,
    merged_floor: Mapping[str, Any] | None,
) -> dict[str, Mapping[str, Any]]:
    """Return pack configurations with the merged floor materialized on the owner.

    The merged floor replaces the owner's authored ``sandbox.floor`` in the
    binding configuration so every reader (the implementation's floor
    reader and the kernel's ``project_execution_floor``) resolves one
    identical final floor. A deep copy keeps the session plan immutable.
    """
    if owner_pack_name is None or merged_floor is None:
        return dict(configuration)
    out: dict[str, Mapping[str, Any]] = {
        pack: dict(values) for pack, values in configuration.items()
    }
    owner_config = dict(out.get(owner_pack_name) or {})
    sandbox_config = dict(owner_config.get("sandbox") or {})
    sandbox_config["floor"] = dict(merged_floor)
    owner_config["sandbox"] = sandbox_config
    out[owner_pack_name] = owner_config
    return out


def _capability_binding_provider(runtime: HostRuntime) -> SessionCapabilityBindings:
    """Build the session capability bindings from the live catalogs.

    Extracted from ``build_turn_hooks`` to keep that assembler under the
    complexity ceiling; owns the operation/tool ref collection and the
    declared-config-owner resolution.
    """
    from kernel.operations.registry import get_operation_catalog
    from kernel.tools.registry import get_tool_catalog

    capability_refs = [
        CapabilityRef("operation", definition.operation_id, definition.owner_id)
        for definition in get_operation_catalog().values()
        if definition.owner_id
    ]
    capability_refs.extend(
        CapabilityRef("tool", definition.tool_id, definition.owner_id)
        for definition in get_tool_catalog().values()
        if definition.owner_id
    )
    owner_pack, merged_floor, _contributing = _merged_sandbox_floor(runtime)
    configuration = _materialize_sandbox_floor(
        runtime.session_plan.manifest.resources.packs.configuration,
        owner_pack,
        merged_floor,
    )
    return SessionCapabilityBindings(
        configuration,
        capabilities=capability_refs or None,
        declared_owners=_declared_config_owners(runtime),
    )


def _transport_forwarder(transport: Any, runtime: HostRuntime | None = None) -> Any:
    """Build guarded transport forwarders (stage + delta events).

    Module-level so ``build_turn_hooks`` stays under the mccabe ceiling:
    the guard branches live here, not in the assembler's nested closures.
    Transport failures never break the turn — same posture as the
    kernel's ``_emit_stage``.
    """

    async def forward(method: str, *args: Any) -> None:
        if transport is None:
            return
        try:
            await getattr(transport, method)(*args)
        except Exception:
            logger.debug("transport %s failed", method, exc_info=True)

    async def on_stage(label: str, data: Any) -> None:
        if label == "model_call" and runtime is not None and isinstance(data, dict):
            data = {
                **data,
                "proxy_route": proxy_route_view(runtime, str(data.get("provider") or "")),
            }
        await forward("on_stage", label, data)

    async def on_delta(text: str, reasoning: str) -> None:
        await forward("on_text_chunk", text, reasoning)

    return on_stage, on_delta


def _effective_replay_budget_bytes(runtime: Any) -> int | None:
    """Resolve explicit replay ceilings; absent ceilings defer to request allocation."""
    ceiling = int(runtime.config.context_budget_replay_max_bytes)
    knob = runtime.session_plan.manifest.context.history.replay_max_bytes
    limits = [value for value in (ceiling, knob) if value is not None and value > 0]
    return min(limits) if limits else None


def build_turn_hooks(
    runtime: HostRuntime,
    context: dict[str, Any] | None = None,
    transport: TransportCallbacks | None = None,
    *,
    extra_item_builders: list[ExtensionItemBuilder] | None = None,
) -> TurnHooks:
    """Build a ``TurnHooks`` dataclass wired to the host runtime.

    Args:
        runtime: The typed host runtime with config, manifest, MCP pool.
        context: Optional mutable turn-context dict. When provided, the
            ``on_turn_complete`` hook updates it with completion metadata.
            When ``None``, a fresh dict is created internally.
        transport: Optional transport callbacks for streaming/debug events.
        extra_item_builders: Optional host-supplied ``ExtensionItemBuilder``
            instances (e.g. the VS Code sidecar's editor-context bridge).
            Composed alongside the installed-pack builder so host-contributed
            context records flow through the same context-mapper feed.
            ``None`` (default) behaves exactly as before.

    Returns:
        A ``TurnHooks`` ready to pass to ``kernel.turn_runner.run_turn``.
    """
    ctx_dict = context if context is not None else {}

    capability_binding_provider = _capability_binding_provider(runtime)
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
    # _runtime_model_resolver keeps the lazy-manifest resolution logic at
    # module level; the thin closure preserves the local name (and any
    # monkeypatch target).
    def _resolve_model(model_id: str) -> StubModelDefinition:
        return _runtime_model_resolver(runtime, model_id)

    # ── Extension item builder ──────────────────────────────────
    # Constructed once so every ``build_unified_state`` call and the
    # ``ContextSeams`` injection share one builder instance. Discovers
    # ``lisa.extensions.context_items`` entry points from installed
    # packs and tags each contributed record with its source pack name.
    # ``_introspect_fn`` is defined below and resolves lazily when the
    # builder invokes a pack that reads it. Host-supplied builders (the
    # editor-context bridge) are composed in after the entry-point builder.
    _item_builder, _runtime_context = _build_item_builder_context(
        runtime, _resolve_model, extra_item_builders=extra_item_builders
    )

    # ── Model router + provider gateway discovery ────────────────
    # Module-level _resolve_router_for_runtime / _resolve_gateway_provider_for_runtime
    # hold the discovery logic; partial binds the runtime so the call sites
    # (wire_context_seams, auto-capture snapshots) keep zero-arg signatures.
    _discover_active_router = partial(_resolve_router_for_runtime, runtime)
    _discover_active_provider_gateways = partial(_resolve_gateway_provider_for_runtime, runtime)

    # ── Streaming request signal ───────────────────────────────────
    # The kernel calls ``gateway.complete_stream(request)`` directly. This
    # non-None sentinel preserves its existing streaming-request signal.
    # _stream_requested_signal returns the sentinel when streaming is
    # enabled in text/hybrid mode, else None.
    _complete_stream: Callable[..., Any] | None = _stream_requested_signal(runtime)

    # ── Content loader (injected into context assembly) ───────────
    loader = content_loader(runtime)

    # ── System prompt assembly ────────────────────────────────────
    def _assemble_system_prompt(mode_instruction: str, stage_instruction: str | None) -> str:
        return _assemble_runtime_system_prompt(runtime, loader, mode_instruction, stage_instruction)

    # ── Dynamic user context assembly ─────────────────────────────────
    _budget_values = context_budget_values_from_config(runtime.config)

    def _assemble_user_context(
        mode_id: str,
        query: str,
        pre_turn_briefing: str,
        capability_manifest: Any = None,
    ) -> str:
        return _assemble_dynamic_user_context(
            runtime=runtime,
            ctx_dict=ctx_dict,
            resolve_model=_resolve_model,
            item_builder=_item_builder,
            runtime_context=_runtime_context,
            loader=loader,
            budget_values=_budget_values,
            mode_id=mode_id,
            query=query,
            pre_turn_briefing=pre_turn_briefing,
            capability_manifest=capability_manifest,
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
        return await _execute_mcp_tool(runtime, tool_id, tool_ctx)

    # ── Turn-complete hook ─────────────────────────────────────────

    # ── Auto-capture trigger detection (extracted to turn_hooks._auto_capture) ─
    # The trigger policy (commitment patterns, write-effect prefixes/suffixes,
    # and the detection function) now lives in runtime.turn_hooks._auto_capture
    # so it is independently testable without constructing a HostRuntime.
    # ``classify_memory_tier`` maps the trigger salience onto a retention tier
    # so workflow-state captures (write-effect tools, soft commitments) land
    # in ``daily``/``weekly`` rather than polluting ``long_term`` recall.
    # The capture body lives in the module-level _maybe_auto_capture helper.

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
        await _maybe_auto_capture(
            runtime=runtime,
            mode_id=mode_id,
            query=query,
            response_text=response_text,
            messages=messages,
            effect_receipts=effect_receipts,
            ctx_dict=ctx_dict,
            capability_binding_provider=capability_binding_provider,
            model_router=_auto_capture_model_router,
            provider_gateways=_auto_capture_provider_gateways,
            resolve_model=_resolve_model,
            execute_tool=_execute_tool,
            execute_operation=_execute_operation,
        )

    # ── UnifiedState provider callable ─────────────────────────────
    # _bind_state_provider returns a zero-arg callable reading the CURRENT
    # session snapshot at call time (the snapshot mutates between turns).
    _state_provider = _bind_state_provider(runtime, _item_builder, _runtime_context)

    # ── Last slice finished at ─────────────────────────────────────
    persisted_last_slice_finished_at = _persisted_last_slice_finished_at(runtime, context)

    # ── Slice callables ────────────────────────────────────────────
    # _build_slice_callables returns the introspection snapshot gatherer,
    # the deterministic self-model slice, and the three file-backed slice
    # entry points as explicit closures over their dependencies.
    (
        _introspect_fn,
        _run_self_model_slice,
        _pre_turn_slice,
        _post_turn_slice,
        _explicit_slice,
    ) = _build_slice_callables(runtime, file_state_provider, _resolve_model)

    # ── Live stage telemetry + token deltas ────────────────────────
    # Forward the kernel's fine-grained stage events and streamed deltas to
    # the transport (on_stage / on_text_chunk) as they happen. The module
    # level _transport_forwarder guards each call — a transport error never
    # breaks the turn, mirroring the kernel's ``_emit_stage`` posture.
    _on_stage_event, _on_delta_event = _transport_forwarder(transport, runtime)

    # ── Pack-provided slice providers ───────────────────────────────
    # _discover_pack_slice_providers (module-level) resolves the
    # factory-factory indirection and collects the non-None providers;
    # _chain_pack_slices (also module-level) wraps the host's base slices.
    _slice_provider_hooks_view = _SliceProviderHooksView(
        runtime=runtime,
    )
    _slice_provider_hooks_view.execute_operation = _execute_operation  # type: ignore[attr-defined]
    _slice_provider_hooks_view.execute_tool = _execute_tool  # type: ignore[attr-defined]

    pack_slice_providers = _discover_pack_slice_providers(
        view=_slice_provider_hooks_view,
        session_plan=runtime.session_plan,
    )

    _pre_turn_slice_chained = _chain_pack_slices(_pre_turn_slice, "pre_turn", pack_slice_providers)
    _post_turn_slice_chained = _chain_pack_slices(
        _post_turn_slice, "post_turn", pack_slice_providers
    )
    _explicit_slice_chained = _chain_pack_slices(_explicit_slice, "explicit", pack_slice_providers)

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
        return _wire_runtime_context_seams(
            runtime=runtime,
            item_builder=_item_builder,
            discover_router=_discover_active_router,
            discover_gateways=_discover_active_provider_gateways,
        )

    return _assemble_turn_hooks(
        runtime=runtime,
        task_id=str(ctx_dict.get("task_id") or ""),
        task_goal=str(ctx_dict.get("task_goal") or ""),
        turn_index=_coerce_turn_index(ctx_dict.get("turn_index")),
        wired=_WiredHooks(
            resolve_model=_resolve_model,
            complete_stream=_complete_stream,
            assemble_system_prompt=_assemble_system_prompt,
            assemble_user_context=_assemble_user_context,
            execute_operation=_execute_operation,
            on_turn_complete=_on_turn_complete,
            on_stage=_on_stage_event,
            on_delta=_on_delta_event,
            state_provider=_state_provider,
            pre_turn_slice=_pre_turn_slice_chained,
            post_turn_slice=_post_turn_slice_chained,
            explicit_slice=_explicit_slice_chained,
            capability_binding_provider=capability_binding_provider,
            approval_service=approval_service,
            last_slice_finished_at=persisted_last_slice_finished_at,
            slice_tuning=slice_tuning,
            wire_context_seams=_wire_context_seams,
        ),
    )

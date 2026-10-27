"""Grouped ``TurnHooks`` — the kernel turn-runner extension points, organized.

``TurnHooks`` formerly packed 40+ fields into one flat dataclass, with
``# ── Comment ──`` banners marking the groupings. This module makes the
groupings real: each cohesive cluster is its own small dataclass with its
field-level docstrings preserved, and ``TurnHooks`` composes them. A reader
now sees *which domain* a hook serves from its attribute path
(``hooks.models.complete``, ``hooks.lifecycle.on_turn_complete``).

The host builds these in ``runtime.turn_hooks_builder.build_turn_hooks``;
``kernel.turn_runner.run_turn`` reads them through the grouped accessors.

DESIGN: hooks are named for *what they do*, not *why the app needs them*.
This keeps the kernel free of app-specific concepts like "passive
intelligence" or "self-model context."
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from kernel.approval import ApprovalService
    from kernel.capability_binding import CapabilityBindingProvider
    from kernel.context.models import ContextBudgetSnapshot
    from kernel.delegation import DelegationResult, DelegationSpec
    from kernel.models import (
        CapabilityManifest,
        ForkResult,
        ForkSpec,
        IntrospectSnapshot,
        OperationResult,
    )
    from kernel.provider_contracts import CompletionRequest, RequestTokenCount
    from kernel.recovery import RecoveryService
    from kernel.seams.protocols import RouteCoordinator, StateProvider
    from kernel.slice import SliceOutcome, SlicePolicy, SliceTuning
    from kernel.tools.mcp_client import MCPClientConfig, MCPToolPool
    from kernel.turn_runner._seams import ContextSeams
    from kernel.workflow_position import WorkflowPosition


# ── ModelHooks ─────────────────────────────────────────────────────────
@dataclass
class ModelHooks:
    """Model resolution and completion callables."""

    #: ``resolve_model(model_id) -> StubModelDefinition``; default fallback
    #: resolves against the statically-declared preferred model only.
    resolve_model: Callable[[str], Any] | None = None

    #: Core completion function. Must accept messages, model_def,
    #: max_tokens, temperature, tools (optional), and response_format
    #: (optional). Returns an object with ``.content`` (str) and
    #: ``.tool_calls`` (list[dict]). A host MUST provide this; if ``None``,
    #: ``run_turn`` raises ``NotImplementedError``.
    complete: Callable[..., Awaitable[Any]] | None = None

    #: Streaming completion function. When set, the turn runner uses this
    #: instead of ``complete`` for the model call. Must be an async generator
    #: that yields ``StreamedChunk`` objects as tokens arrive. v1 streams
    #: tokens only — when the model wants to call tools, the stream's
    #: ``finish_reason`` is ``'tool_calls'`` and the runner falls back to
    #: ``complete`` automatically.
    complete_stream: Callable[..., Any] | None = None

    #: Per-chunk observer for the streaming path. When set, the turn runner
    #: awaits ``on_delta(text, reasoning)`` for every ``StreamedChunk`` as it
    #: arrives (before aggregation), so transports can render live token
    #: streams. Errors raised here are transport decisions, not turn
    #: failures — the runner forwards them raw (the host swallows or
    #: handles, mirroring ``emit_stage``).
    on_delta: Callable[[str, str], Awaitable[None]] | None = None

    #: Persist an exact request synchronously before dispatch; retain only its reference.
    record_request: Callable[[dict[str, Any]], str] | None = None
    count_request_tokens: Callable[[CompletionRequest], RequestTokenCount] | None = None


# ── ContextHooks ────────────────────────────────────────────────────────
class MessageProjector(Protocol):
    """Provider-only projection of one model call's messages.

    ``budget_bytes`` is the resident tool-result capacity left after the
    kernel accounts for the complete request and completion reserve.
    """

    def __call__(
        self, messages: list[dict[str, Any]], task_id: str, *, budget_bytes: int
    ) -> list[dict[str, Any]]: ...


@dataclass
class ContextHooks:
    """Prompt assembly and provider-only message projection.

    project_messages receives raw messages, task_id, and the model-sized
    tool-result budget; it must not mutate inputs. Raw turn recording
    remains separate from the provider request.
    """

    #: ``assemble_system_prompt(mode_instruction, stage_instruction=None)``
    #: returns the full system prompt. Default: ``mode_instruction`` only.
    assemble_system_prompt: Callable[[str, str | None], str] | None = None

    #: ``assemble_user_context(mode_id, query, pre_turn_briefing,
    #: capability_manifest=None)`` returns extra user-context content injected
    #: before the query in the final user message. Default: empty string.
    #: The manifest is compiled from the live tool schemas this turn — the
    #: ground truth for what the model can call — so hosts render a capability
    #: context that matches the injected function schema (the "Tooling
    #: Paradox" fix).
    assemble_user_context: Callable[[str, str, str, CapabilityManifest | None], str] | None = None

    #: ``prior_messages()`` returns earlier turns' messages (user, assistant,
    #: tool) to replay verbatim before this turn's user message. Parent turns
    #: only. Default: none.
    prior_messages: Callable[[], list[dict[str, Any]]] | None = None

    project_messages: MessageProjector | None = None

    #: Most recent ``ContextBudgetSnapshot`` produced by the host's
    #: context-mapper pass. Mutated by the host after each prompt assembly so
    #: slice providers read actual context-window utilization without
    #: re-running the mapper. Read-only from the kernel's perspective.
    last_context_snapshot: ContextBudgetSnapshot | None = None


# ── OperationHooks ──────────────────────────────────────────────────────
@dataclass
class OperationHooks:
    """Internal capability dispatch: operations, delegate, and fork."""

    #: ``execute_operation(name, context)`` — run a named operation with an
    #: ``OperationExecutionContext``. Default: delegates to
    #: ``kernel.operations.registry.execute_operation``. Operations are
    #: advertised to the LLM as tools directly from the in-process catalog.
    execute_operation: Callable[..., Awaitable[OperationResult]] | None = None

    #: ``delegate_fn(spec)`` — run one delegate sub-turn. Default:
    #: ``_default_delegate_fn`` (recurses ``run_turn`` as a ``"delegate"``
    #: child). ``None`` means delegation is not wired; the delegate operation
    #: fails loudly until the kernel fills it in at turn start.
    delegate_fn: Callable[[DelegationSpec], Awaitable[DelegationResult]] | None = None

    #: ``fork_fn(spec)`` — run a parallel fork fan-out. Default:
    #: ``_default_fork_fn`` (``asyncio.TaskGroup``). ``None`` means fork is
    #: not wired; the kernel fills it in at turn start.
    fork_fn: Callable[[ForkSpec], Awaitable[ForkResult]] | None = None


# ── LifecycleHooks ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class TaskCompletionRequest:
    """Opaque task and producer receipts the host must verify against durable state."""

    task_id: str
    origin_tool_id: str
    receipt_artifact_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskCompletionEvidence:
    """Host verdict about whether this task owns a current checked commit."""

    verified: bool
    reason: str = ""
    artifact_id: str = ""


@dataclass
class LifecycleHooks:
    """Turn lifecycle: completion side effects, slices, stage telemetry,
    operator signals, and live introspection."""

    #: Called after every successful turn with ``mode_id, query,
    #: response_text, messages, effect_ledger``. Default: no-op. The app hooks
    #: thread persistence, digest refresh, and provenance here.
    on_turn_complete: (
        Callable[[str, str, str, list[dict[str, Any]], list[Any]], Awaitable[None]] | None
    ) = None

    #: ``SliceTuning`` controlling the slice budget shape. ``None`` means
    #: "use defaults and disable all three triggers."
    slice_tuning: SliceTuning | None = None

    #: Opportunistic slice fired before the model call. Receives a
    #: ``SlicePolicy`` (mode=PRE_TURN) and returns a ``SliceOutcome``.
    pre_turn_slice: Callable[[SlicePolicy], Awaitable[SliceOutcome]] | None = None

    #: Opportunistic slice fired after the response is committed.
    post_turn_slice: Callable[[SlicePolicy], Awaitable[SliceOutcome]] | None = None

    #: Slice fired only when an operation or CLI command invokes it; the
    #: caller's budget is honored verbatim — pressure does not apply.
    explicit_slice: Callable[[SlicePolicy], Awaitable[SliceOutcome]] | None = None

    #: Monotonic timestamp of the last completed slice; the framework's
    #: pressure math reads it. The host can coordinate multiple slices here.
    last_slice_finished_at: float | None = None

    #: Async ``(label, data)`` stage telemetry callback invoked at key turn
    #: points (model_call, tool_dispatch, workflow_stage, fork_*). Failures
    #: are caught and logged; they never break the turn. Default: ``None``.
    on_stage: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None

    #: ``operator_signal()`` polled once per iteration at the top of the turn
    #: loop, before the lifecycle resolver. Returns ``""`` (none),
    #: ``"pause"``, ``"stop"``, or ``"terminate"``. The signal is terminal —
    #: the loop breaks before the model call and the exit is encoded in
    #: ``TurnResult.operator_exit``. The model never sees it.
    operator_signal: Callable[[], str] | None = None

    #: Async callable returning an ``IntrospectSnapshot`` from the host's
    #: live runtime (process memory, asyncio tasks, MCP pool health, token
    #: usage, log tail). Injected into ``OperationExecutionContext.host`` so
    #: the self-model and forensic readers can read live state. Default: ``None``.
    introspect: Callable[[], Awaitable[IntrospectSnapshot]] | None = None


# ── WorkflowHooks ───────────────────────────────────────────────────────
@dataclass
class WorkflowHooks:
    """Durable workflow-position accessors — the inter-mode plan state."""

    #: ``workflow_position()`` returns the current ``WorkflowPosition`` for
    #: the active user/surface (or ``None`` when no workflow is in flight).
    #: The turn loop reads the stage pointer to inject the current stage's
    #: ``instruction_body`` into the system prompt and feeds ``turn_count``
    #: to the lifecycle resolver. Default: ``None`` (no durable state).
    workflow_position: Callable[[], WorkflowPosition | None] | None = None

    #: ``workflow_position_updater(position)`` persists an advanced
    #: ``WorkflowPosition`` after a successful ``transition_stage`` so the
    #: next turn's ``workflow_position`` hook returns it. Default: ``None``
    #: (advances apply in-process but are not persisted across turns).
    workflow_position_updater: Callable[[WorkflowPosition], None] | None = None

    #: Optional ``RouteCoordinator`` backend that persists the next handoff
    #: target across turns (consulted for ``proceed``-style continuations).
    route_coordinator: RouteCoordinator | None = None


# ── HostServiceHandles ──────────────────────────────────────────────────
@dataclass
class HostServiceHandles:
    """Host-owned durable service handles injected into operation/tool
    contexts so pack code can persist and recall without recomputing."""

    #: ``state_provider()`` returns the current session's ``UnifiedState``
    #: (or ``None``). Injected into ``OperationExecutionContext`` so operations
    #: that need cached runtime state can read it.
    state_provider: StateProvider | None = None

    #: Session-owned binding provider for concrete catalog capabilities.
    capability_binding_provider: CapabilityBindingProvider | None = None

    #: Host-owned persistence + receipt service for exact-action approval.
    approval_service: ApprovalService | None = None

    #: Host-owned durable service for interrupted dispatch work items.
    recovery_service: RecoveryService | None = None

    #: Host-owned store for pack-neutral local coding-workflow artifacts.
    #: Structural seam — annotated against the concrete host store so readers
    #: see the real surface; the kernel never imports host types at runtime.
    code_change_artifact_store: Any = None
    verify_task_completion: (
        Callable[[TaskCompletionRequest], Awaitable[TaskCompletionEvidence]] | None
    ) = None

    #: Canonical host data root, injected by the host and threaded into every
    #: operation/tool context. There is deliberately no fallback — a host
    #: wiring tools that need a data dir must inject it.
    data_dir: str = ""


# ── TurnIdentity ────────────────────────────────────────────────────────
@dataclass
class TurnIdentity:
    """Identity scope defaults for the active turn.

    Bare/test hosts that never wire identity get coherent personal-DM
    addresses; production hosts (the CLI host builder) override every field
    from their own config.
    """

    workspace_id: str = "default"
    user_id: str = "default"
    user_slug: str = "default"
    user_name: str = ""
    principal_key: str = "default"
    conversation_id: str = "default"
    surface_type: str = "personal_dm"
    surface_id: str = "default"
    memory_scope: str = "personal"
    session_plan_digest: str = ""
    task_id: str = ""
    #: Global thread turn index this turn will occupy, computed at turn start
    #: (``runtime.thread.next_turn_index``) and stamped on the thread entry.
    #: Threaded into every dispatch/token/decision payload so forensic readers
    #: can join the thread, the dispatch ledger, and the cost log by one key.
    #: ``None`` means the work is not attributable to a thread turn (off-turn
    #: slice work, or a bare/test host with no thread) — distinct from ``0``,
    #: which is the thread's first real turn.
    turn_index: int | None = None
    #: What this task is FOR, declared at task creation by the delegate-spec
    #: author or the operator. ``implementation`` (the default) means the task
    #: promises a committed change, so a checked-commit mode must verify one.
    #: ``diagnostic`` means the task is investigation only, which lets a
    #: checked-commit mode complete without a worktree producer — but only
    #: when the turn also made no writes (the mutation guard). Intent varies
    #: per task, so it lives here rather than on the mode: a mode-level opt-out
    #: would silently false-pass every implementation task in that mode.
    #: Fail-closed default: an undeclared task is an implementation task.
    task_goal: Literal["implementation", "diagnostic"] = "implementation"


# ── RecallHooks ─────────────────────────────────────────────────────────
@dataclass
class RecallHooks:
    """Pre-turn memory-recall configuration."""

    #: When True, the turn loop calls ``recall_memory`` during Phase 1 and
    #: injects the result into the user message as a ``## Memory Recall``
    #: block. Gated independently of ``memory_enabled`` so hosts can opt out
    #: of the per-turn recall cost while keeping the memory backend active.
    enabled: bool = True

    #: Maximum number of recall results to inject into the user message.
    limit: int = 3

    #: Skip pre-turn recall when the query is shorter than this many
    #: characters (e.g. ``"ok"``, ``"thanks"``).
    min_query_chars: int = 3


# ── MCP / tool server wiring ─────────────────────────────────────────────
@dataclass
class ToolHooks:
    """MCP tool-server wiring (warm pool + cold configs)."""

    #: MCP server configurations to spawn for tool discovery and dispatch.
    #: The host declares which servers to run; the kernel manages subprocess
    #: lifecycle via ``MCPToolPool``. Default: empty list.
    mcp_tool_servers: list[MCPClientConfig] = field(default_factory=list)

    #: Optional host-owned long-lived ``MCPToolPool``. When set, ``run_turn``
    #: reuses this across turns instead of spawning a per-turn pool; the host
    #: owns the lifecycle (``await pool.start()`` once at init,
    #: ``await pool.aclose()`` at exit). When ``None``, a per-turn pool is
    #: built from ``mcp_tool_servers``.
    mcp_tool_pool: MCPToolPool | None = None

    #: Optional decorator for the rendered output of an in-process pack tool.
    #:
    #: ``post_dispatch_result(tool_id, arguments, rendered, turn_token)``
    #: returns the output the model should see. The kernel stringifies the
    #: tool's result first and passes that string in, so a host can append
    #: context without re-implementing rendering and without mutating a
    #: tool's own ``output_text`` — callers such as the memory-consolidation
    #: slice parse that field as JSON. Returning the argument unchanged is a
    #: no-op.
    #:
    #: ``turn_token`` is an opaque kernel-assigned string identifying this
    #: turn *and this child*: parent turns and each fork/delegate child get
    #: distinct values within one host message. A host that must inject
    #: something at most once keys its bookkeeping on it. A resumed turn
    #: keeps the token it started with.
    #:
    #: Decorating the *rendered* string keeps this seam free of customization
    #: vocabulary: the kernel invokes a hook and knows nothing about globs,
    #: instructions, or files. A hook that raises is logged and ignored —
    #: decoration is advisory and must never fail a tool call.
    #:
    #: Applies to in-process pack tools only (the ``implementation_module``
    #: branch). MCP-server tools render into ``output_text`` and operations
    #: return typed results; neither is decorated.
    post_dispatch_result: Callable[[str, dict[str, Any], str, str], str] | None = None


# ── TurnHooks — the composed container ──────────────────────────────────
@dataclass
class TurnHooks:
    """Extension points for the kernel turn runner.

    The host (lightweight host or production app) provides concrete
    implementations. Every callable defaults to a no-op or sensible stub, so
    a minimal host participates with zero overrides. The fields are grouped
    into sub-dataclasses so a reader sees the domain a hook serves from its
    attribute path (``hooks.models.complete``).
    """

    models: ModelHooks = field(default_factory=ModelHooks)
    context: ContextHooks = field(default_factory=ContextHooks)
    operations: OperationHooks = field(default_factory=OperationHooks)
    lifecycle: LifecycleHooks = field(default_factory=LifecycleHooks)
    workflow: WorkflowHooks = field(default_factory=WorkflowHooks)
    services: HostServiceHandles = field(default_factory=HostServiceHandles)
    identity: TurnIdentity = field(default_factory=TurnIdentity)
    recall: RecallHooks = field(default_factory=RecallHooks)
    tools: ToolHooks = field(default_factory=ToolHooks)
    #: Context-layer seams (content registry, model policy, provider gateways,
    #: event bus). Kept as a direct field to preserve the
    #: existing ``kernel.context`` default wiring contract.
    context_seams: ContextSeams = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # Lazy import to keep the kernel's default context seams out of the
        # import graph for hosts that never wire context (bare/test hosts).
        if self.context_seams is None:
            from kernel.turn_runner import ContextSeams

            self.context_seams = ContextSeams()


__all__ = [
    "ContextHooks",
    "HostServiceHandles",
    "LifecycleHooks",
    "ModelHooks",
    "OperationHooks",
    "RecallHooks",
    "TaskCompletionEvidence",
    "TaskCompletionRequest",
    "ToolHooks",
    "TurnHooks",
    "TurnIdentity",
    "WorkflowHooks",
]

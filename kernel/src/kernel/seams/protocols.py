"""Kernel seams — host-injection protocols for state and guardrails.

These Protocols define the dependency-inversion surface between the
runtime-neutral kernel and host applications. Hosts implement these
protocols and inject concrete instances into execution contexts; the
kernel never imports host code.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from kernel.models import (
    SurfaceContext,
    UnifiedState,
)


@dataclass(frozen=True, slots=True)
class RouterModelRef:
    """A snapshot of one model's manifest metadata for router conditions.

    Routers receive the *effective* model — the model actually selected for
    the preceding iteration (or the statically-resolved preferred model on
    iteration 1) — so conditions can inspect ``model.cost_tier``,
    ``model.capabilities``, etc. without coupling to the manifest.
    """

    model_id: str = ""
    provider: str = ""
    cost_tier: str = ""
    latency_tier: str = ""
    capabilities: tuple[str, ...] = ()
    reasoning_model: bool = False
    thinking_model: bool = False
    #: Confidence (0.0-1.0) the effective model reliably honors structured
    #: output. Routers can gate JSON-bound work on this, e.g. via
    #: ``model.structured_reliability`` in a ``when`` rule.
    structured_reliability: float = 0.0
    #: User-declared flag: this model needs stronger kernel guidance
    #: backstops (do-the-work self-handoff synth, full enforcer cap).
    #: The turn loop reads this to gate the backstops that strong models
    #: do not need. See ``sdk/router/README.md`` § Model attributes and
    #: ``ManifestModelEntry.needs_guidance``.
    needs_guidance: bool = False


@dataclass(frozen=True, slots=True)
class RouterRequest:
    """Inputs to one model-router ``choose`` call.

    The turn loop constructs one ``RouterRequest`` per iteration and passes
    it to the active ``ModelRouter`` (if wired). The router inspects these
    fields and returns a ``RouterDecision`` — either a model selection or a
    terminal action (``finish`` / ``transition_to``).

    Fields mirror the information already available inside the turn loop:
    the statically-resolved default model, the iteration number, how many
    tool calls have fired, how much tool-call budget remains, the user query,
    the last tool result text (if any), the capabilities the current mode's
    tool surface requires, the active mode id, and the metadata of the
    model effective in the preceding iteration (``effective_model``).
    """

    default_model_id: str
    iteration: int
    tool_calls_so_far: int
    budget_remaining: int | None
    query: str = ""
    last_tool_result: str | None = None
    #: ``True`` when at least one tool has returned a result this turn.
    #: Distinct from ``last_tool_result`` (the *most recent* result text),
    #: which may be empty even when earlier tools returned results.
    has_tool_result: bool = False
    required_capabilities: tuple[str, ...] = ()
    mode_id: str = ""
    effective_model: RouterModelRef | None = None
    #: The operation id when this request originates from an operation
    #: execution (not the turn loop). Empty for turn-loop iterations.
    #: CEL rules can key off ``route.operation_id`` to select a model for
    #: a specific operation without editing the pack's ``.yml``/``.md``.
    operation_id: str = ""


@dataclass(frozen=True, slots=True)
class RouterDecision:
    """The output of one model-router ``choose`` call.

    Exactly one action is set:

    - ``model_id`` — select a model for this iteration. The turn loop feeds
      it back through ``resolve_model()`` and the existing provider-gateway
      path.
    - ``finish`` — stop the tool loop and force one final no-tools
      completion. The value is a short reason slug for telemetry/audit
      (e.g. ``"tool_budget_exhausted"``).
    - ``transition_to`` — move the active workflow to a named terminal stage,
      then produce its closing summary. The value is the target stage id.

    ``reason`` is a human-readable string for telemetry/audit. ``routed_by``
    identifies which router made the decision (e.g. ``"lisa-router"``).
    """

    model_id: str = ""
    finish: str = ""
    transition_to: str = ""
    reason: str = ""
    routed_by: str = ""


@dataclass(frozen=True, slots=True)
class CoordinatorRouteDecision:
    """A persisted next-step route decision for a follow-up turn.

    Hosts can persist a target mode/stage for continuation prompts such as
    ``proceed`` so the kernel can inject a deterministic handoff hint on the
    next turn even when the original query did not carry a fresh routing
    signal.
    """

    next_target: str = ""
    reason: str = ""


@runtime_checkable
class RouteCoordinator(Protocol):
    """Host-owned persistence boundary for route continuation state.

    The kernel calls ``load`` before a turn begins, then ``save`` after a
    successful handoff so the next turn can continue from the same next step.
    """

    def load(self, request: Any) -> CoordinatorRouteDecision | None: ...

    def save(self, decision: CoordinatorRouteDecision, request: Any) -> None: ...


@runtime_checkable
class StateProvider(Protocol):
    """Protocol for in-memory state access.

    Host apps implement this as a zero-argument callable returning the latest
    ``UnifiedState`` (or None).  Injected into execution contexts as
    ``ctx.state_provider``.
    """

    def __call__(self) -> UnifiedState | None: ...


@dataclass(slots=True)
class GuardResult:
    """The outcome of a guardrail check.

    ``allowed=False`` blocks the turn (input guard) or replaces the output
    (output guard). ``sanitized_text`` replaces the original text when the
    guard rewrites it. ``annotations`` carry provider-specific metadata
    (toxicity scores, PII findings, etc.) for audit logging.
    """

    allowed: bool
    blocked_reason: str | None = None
    sanitized_text: str | None = None
    annotations: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class InputGuard(Protocol):
    """Validate user input before it reaches the model.

    Called by the turn runner in Phase 1 (pre-turn), before system
    prompt assembly. If the guard returns ``allowed=False``, the turn
    is aborted and the ``blocked_reason`` is returned to the user.
    """

    async def validate(
        self,
        query: str,
        surface: SurfaceContext,
    ) -> GuardResult: ...


@runtime_checkable
class OutputGuard(Protocol):
    """Validate model output before it reaches the user.

    Called by the turn runner in Phase 3 (post-turn), after the tool
    loop produces final text and before ``store_memory()``. If the guard
    returns ``allowed=False``, the output is replaced with a safe
    fallback message and the original output is logged for audit.
    """

    async def validate(
        self,
        text: str,
        surface: SurfaceContext,
    ) -> GuardResult: ...


@runtime_checkable
class CompleteFn(Protocol):
    """Protocol for LLM completion calls.

    Injected into ``OperationExecutionContext.host.complete``.
    """

    async def __call__(
        self,
        messages: list[dict[str, str]],
        *,
        model_id: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: type[Any] | None = None,
        **kwargs: Any,
    ) -> Any: ...


@runtime_checkable
class ResolveModelFn(Protocol):
    """Protocol for model alias resolution.

    Injected into ``OperationExecutionContext.host.resolve_model``.
    """

    def __call__(self, alias: str) -> str: ...


@runtime_checkable
class ExecuteToolFn(Protocol):
    """Protocol for invoking a tool by ID.

    Injected into ``OperationExecutionContext.host.execute_tool``.
    """

    async def __call__(self, tool_id: str, ctx: Any) -> Any: ...


@runtime_checkable
class ExecuteOperationFn(Protocol):
    """Protocol for invoking an operation by ID.

    Injected into ``OperationExecutionContext.host.execute_operation``.
    """

    async def __call__(self, operation_id: str, ctx: Any) -> Any: ...


# ── Operation host seam ──────────────────────────────────────────────
#
# The typed seam that operations depend on instead of reaching into five
# separate untyped ``Any`` callables on ``OperationExecutionContext``.
# Hosts inject a concrete adapter (``_CallableHost``) that delegates to
# the existing callables; the kernel never imports host dataclasses.
# ``CompletionResult`` and ``ModelRef`` are structural Protocols that the
# host's ``StubCompletion`` / ``StubModelDefinition`` satisfy.


@runtime_checkable
class CompletionResult(Protocol):
    """Structural shape of a completion result (host-owned concrete type).

    Operations read ``content``, ``tool_calls``, ``finish_reason`` (for
    telemetry), and ``truncated`` (the authoritative truncation signal)
    via this Protocol. The host's ``StubCompletion`` structurally satisfies
    it without inheriting.
    """

    content: str
    tool_calls: list[dict[str, Any]]
    finish_reason: str | None
    truncated: bool | None


@runtime_checkable
class ModelRef(Protocol):
    """Structural shape of a resolved model definition (host-owned concrete type).

    The host's ``StubModelDefinition`` structurally satisfies this Protocol.
    """

    model_id: str
    provider: str
    reasoning_model: bool


@runtime_checkable
class OperationHost(Protocol):
    """The single typed seam operations depend on.

    Collapses the untyped ``Any`` callable fields that previously lived on
    ``OperationExecutionContext`` (``complete_fn``, ``resolve_model_fn``,
    ``read_authored_text_fn``, ``execute_tool_fn``, ``execute_operation_fn``,
    ``generate_image_fn``, ``introspect_fn``, ``delegate_fn``) into one
    Protocol. Operations call ``ctx.host.complete(...)`` instead of
    ``ctx.complete_fn(...)``.

    ``complete`` returns a ``CompletionResult`` (structural Protocol) whose
    ``truncated`` field is the provider-owned authoritative truncation
    signal — operations should prefer it over inferring from
    ``finish_reason`` strings.

    ``delegate`` runs one sequential sub-turn via the injected
    ``delegate_fn``. The pack-level ``delegate`` operation calls
    ``ctx.host.delegate(spec)`` with a :class:`~kernel.delegation.DelegationSpec`
    and awaits a :class:`~kernel.delegation.DelegationResult`. The default
    implementation (``_default_delegate_fn``) recurses into ``run_turn``
    as a ``sub_turn_kind="delegate"`` child; a host may override
    ``TurnHooks.delegate_fn`` to inject a custom delegation backend.

    ``fork`` runs a parallel fan-out via the injected ``fork_fn``. The
    pack-level ``fork`` operation calls ``ctx.host.fork(spec)`` with a
    :class:`~kernel.models.ForkSpec` and awaits a
    :class:`~kernel.models.ForkResult`. The default implementation
    (``_default_fork_fn``) wraps ``_execute_fork_fanout``, fanning the
    targets out via an ``asyncio.TaskGroup`` and classifying each child
    via the shared ``classify_child_turn``; a host may override
    ``TurnHooks.fork_fn`` to inject a custom fan-out backend.
    """

    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        model_def: Any,
        max_tokens: int,
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
        response_format: type[Any] | None = None,
    ) -> CompletionResult: ...

    def resolve_model(self, model_id: str) -> ModelRef | None: ...

    def read_authored_text(self, path: str) -> str: ...

    async def execute_tool(self, tool_id: str, ctx: Any) -> Any: ...

    async def execute_operation(self, operation_id: str, ctx: Any) -> Any: ...

    async def generate_image(self, **kwargs: Any) -> Any: ...

    def introspect(self, **kwargs: Any) -> Any: ...

    async def delegate(self, spec: Any) -> Any: ...

    async def fork(self, spec: Any) -> Any: ...


# ── Context seams ───────────────────────────────────────────────────────────
#
# The following Protocols invert the dependencies that ``kernel/context/``
# previously took on private ``app.*`` modules. Hosts inject concrete
# implementations through ``TurnHooks.context_seams``; the kernel ships public
# default implementations so third-party hosts get working context and memory
# without the private app.


@runtime_checkable
class ContentRegistry(Protocol):
    """Load authored text assets from merged content roots.

    Replaces ``app.content_registry.read_authored_text``. The default
    implementation reads from pack and extension content roots discovered
    through ``kernel.content_roots``.
    """

    def read_authored_text(self, relative_path: str) -> str: ...


@runtime_checkable
class ModelPolicy(Protocol):
    """Resolve model aliases and availability without coupling to a provider.

    Replaces ``app.model_policy``. The default implementation resolves
    aliases through ``kernel.model_manifest``.

    ``choose`` is OPTIONAL. When a host wires it, the turn loop consults it
    every iteration instead of resolving one model for the whole turn. This
    lets a host express "cheap model routes, expensive model synthesizes"
    without the operation indirection. The default ``ManifestModelPolicy``
    does not implement ``choose``; the loop falls back to the resolved
    ``preferred_model_id`` when the method is absent.
    """

    def resolve_alias(self, alias: str) -> str: ...
    def is_available(self, model_id: str) -> bool: ...

    # NOTE: ``choose`` is intentionally NOT declared here. Protocol members
    # are mandatory; ``choose`` is an optional capability. The turn loop
    # probes for it with ``getattr(policy, "choose", None)`` and falls back
    # to the resolved default model when absent. Declaring it here would
    # force every ModelPolicy implementation (including the default
    # ``ManifestModelPolicy``) to stub it. Hosts that want per-iteration
    # selection add the method to their own implementation; the runtime
    # duck-types it.


@runtime_checkable
class ModelRouter(Protocol):
    """Per-iteration model selection backed by a pluggable router.

    Distinct from ``ModelPolicy``: ``ModelPolicy`` resolves aliases and
    checks availability statically; ``ModelRouter`` decides which concrete
    model to use for *one iteration* of the turn loop, based on turn state.

    Hosts wire a ``ModelRouter`` into ``ContextSeams.model_router`` at
    assembly time. When ``None``, the loop falls back to the existing
    ``ModelPolicy.choose`` duck-type path, then to the statically-resolved
    ``preferred_model_id`` — preserving today's behavior exactly.

    Routers are discovered via the ``lisa.extensions.model_routers``
    entry-point group (see ``kernel.model_router_discovery``). The
    built-in ``lisa-router`` pack ships a rules-based engine; external packs
    (e.g. ``lisa-pack-aoi-router``) register vendor-backed routers under the
    same group.
    """

    def choose(self, req: RouterRequest) -> RouterDecision | None: ...


@runtime_checkable
class ProviderGatewayProvider(Protocol):
    """Return a ``ProviderGateway`` for a given provider name.

    The kernel ships no default implementation — concrete gateway backends
    live in the SDK (``sdk/providers/``) and are discovered via the
    ``lisa.extensions.provider_gateways`` entry-point group. Hosts wire the
    discovered ``ProviderGatewayProvider`` into ``ContextSeams``; third-party
    packs can register their own under the same entry-point group.
    """

    def gateway_for(self, provider: str) -> Any | None: ...


@runtime_checkable
class EventBus(Protocol):
    """Lightweight pub/sub for salience events and consolidation triggers.

    Replaces ``app.event_bus.BusBackend``. The default implementation is an
    in-process ``asyncio`` queue bus with no external dependencies.
    """

    async def publish(self, topic: str, payload: dict[str, Any]) -> None: ...
    def subscribe(
        self, topic: str, handler: Callable[[dict[str, Any]], Awaitable[None]]
    ) -> None: ...
    def unsubscribe(
        self, topic: str, handler: Callable[[dict[str, Any]], Awaitable[None]]
    ) -> None: ...


@runtime_checkable
class DataSanitizer(Protocol):
    """Apply redaction before memory writes or context assembly.

    Replaces ``app.data_sanitizer.apply_redaction_pipeline``. The default
    implementation is a pass-through that performs no redaction; production
    hosts override with a real sanitizer.
    """

    def redact(self, text: str, *, scope: str) -> str: ...


@runtime_checkable
class ExtensionItemBuilder(Protocol):
    """Build context items from installed extensions.

    Replaces ``app.extension_context_items.build_extension_context_items``.
    The default implementation returns an empty list; production hosts override
    with a builder that walks installed extensions.
    """

    def build_extension_context_items(
        self,
        runtime_context: Any,
        state: Any,
    ) -> list[Any]: ...

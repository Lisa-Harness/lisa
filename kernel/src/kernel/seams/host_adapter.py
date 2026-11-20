"""Concrete ``OperationHost`` adapter bridging legacy callable seams.

The kernel's ``OperationHost`` Protocol (in ``kernel.seams.protocols``) is
the typed seam operations depend on. This module's ``_CallableHost`` is the
wiring-layer adapter that delegates each Protocol method to the
corresponding untyped callable on ``OperationExecutionContext``
(``complete_fn``, ``resolve_model_fn``, ``read_authored_text_fn``,
``execute_tool_fn``, ``execute_operation_fn``, ``generate_image_fn``,
``introspect_fn``).

This is the *wiring layer* (like ``TurnHooks``), not a compat shim: the
host populates the callables, ``_build_minimal_context`` wraps them in a
``_CallableHost``, and operations call ``ctx.host.complete(...)``. The
legacy callable fields are removed once all packs are cut over.

Durable request recording
-------------------------
Off-turn model calls (memory mining, consolidation, self-model distills)
complete through this adapter and previously left no durable record of the
exact request they sent — the thread entry's ``turn_input.calls`` covers
only the turn loop's own calls. When an :class:`CompletionAuditContext` is
attached, every gateway completion through the operation seam archives the
verbatim request content-addressed (``r<sha256>`` result object), appends a
``model.request`` row to the decision ledger, and publishes the call's
token usage — so an off-turn distillation is exactly as auditable as an
in-turn call. Recording and telemetry are strictly best-effort: neither can
fail the completion. The operation registry builds the audit context from
the dispatch's own ``OperationExecutionContext`` (the single choke point
every operation dispatch flows through), so hosts constructing their own
adapters keep the previous no-recording behavior by default.
"""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import Callable
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)


class CompletionAuditContext(NamedTuple):
    """Durable-audit wiring for one operation dispatch's completions.

    Fields:
        record_request: Persist one request snapshot (the dict shape the
            turn loop's ``ModelHooks.record_request`` accepts) and return
            its content-addressed ref. Same contract as
            ``runtime.turn_continuation.persist_request``.
        event_bus: Bus for ``model.request`` rows and ``lisa.token_usage``
            events. ``None`` disables telemetry (recording still happens).
        operation_id: The dispatch's operation id, stamped on every row.
        turn_index: Global thread turn index this call belongs to, or
            ``None`` when the call belongs to no turn (off-turn slice work).
            ``None`` is meaningful and distinct from ``0`` — see
            ``kernel.models.resolve_turn_index``.
        user_slug: Owner of the call, for the shared ledger's scoped reads.
        thread_store_slug: Store stem of the thread the call belongs to.
        call_id: The ``call_id`` minted for this off-turn model call, so its
            ``model.request`` row and its token row join on the same key (the
            off-turn analogue of the turn loop's call identity). Empty means
            the caller did not mint one.
        dispatch_id: The ``dispatch_id`` of the operation dispatch that made
            the call, so an off-turn row joins back to the dispatch that
            caused it (Evidence Envelope §2). Empty means no dispatch behind
            the call.
        record_response: Persist one response snapshot (the dict shape the
            turn loop's ``ModelHooks.record_response`` accepts) and return
            its content-addressed ref (Spec: MODEL-CALL-ARCHIVE §1). Same
            ``persist_request`` shape. ``None`` keeps responses unrecorded;
            the call's ``model.request`` row then carries no
            ``response_ref`` and a reader reports the archive gap.
    """

    record_request: Callable[[dict[str, Any]], str]
    event_bus: Any | None
    operation_id: str
    turn_index: int | None
    user_slug: str
    thread_store_slug: str
    call_id: str = ""
    dispatch_id: str = ""
    record_response: Callable[[dict[str, Any]], str] | None = None


class _CallableHost:
    """Adapt the legacy callable bag into the typed ``OperationHost`` Protocol.

    Each method delegates to the corresponding ``ctx.*`` callable. The
    adapter is intentionally thin — it adds no behavior, only types — with
    one exception: when a :class:`CompletionAuditContext` is attached, the
    gateway completion path records and publishes the audit evidence
    described in the module docstring.
    """

    __slots__ = (
        "_audit",
        "_complete_fn",
        "_delegate_fn",
        "_execute_operation_fn",
        "_execute_tool_fn",
        "_fork_fn",
        "_gateway_provider_fn",
        "_generate_image_fn",
        "_introspect_fn",
        "_read_authored_text_fn",
        "_resolve_model_fn",
    )

    def __init__(
        self,
        *,
        complete_fn: Any = None,
        resolve_model_fn: Any = None,
        read_authored_text_fn: Any = None,
        execute_tool_fn: Any = None,
        execute_operation_fn: Any = None,
        generate_image_fn: Any = None,
        introspect_fn: Any = None,
        delegate_fn: Any = None,
        fork_fn: Any = None,
        gateway_provider_fn: Any = None,
        audit: CompletionAuditContext | None = None,
    ) -> None:
        self._complete_fn = complete_fn
        self._resolve_model_fn = resolve_model_fn
        self._read_authored_text_fn = read_authored_text_fn
        self._execute_tool_fn = execute_tool_fn
        self._execute_operation_fn = execute_operation_fn
        self._generate_image_fn = generate_image_fn
        self._introspect_fn = introspect_fn
        self._delegate_fn = delegate_fn
        self._fork_fn = fork_fn
        self._gateway_provider_fn = gateway_provider_fn
        self._audit = audit

    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        model_def: Any,
        max_tokens: int,
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
        response_format: type[Any] | None = None,
    ) -> Any:
        # The production completion path is the provider gateway: resolve the
        # gateway for the model's provider and call its ``complete`` method.
        # ``complete_fn`` remains for callers that have not yet cut over to a
        # gateway (operations tests until the seam is fully wired). When neither
        # is present this raises the documented no-completion-path error.
        if self._gateway_provider_fn is not None:
            provider_gateways = (
                self._gateway_provider_fn()
                if callable(self._gateway_provider_fn)
                else self._gateway_provider_fn
            )
            if provider_gateways is None:
                raise NotImplementedError(
                    "OperationHost.complete: no provider gateway wired (provider_gateways is None)."
                )
            gateway = provider_gateways.gateway_for(str(getattr(model_def, "provider", "") or ""))
            if gateway is None:
                raise NotImplementedError(
                    f"OperationHost.complete: no provider gateway registered for "
                    f"provider {getattr(model_def, 'provider', '')!r}."
                )
            from kernel.provider_contracts import CompletionRequest

            request = CompletionRequest(
                messages=messages,
                model_def=model_def,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
                response_format=response_format,
                # FIX-2: explicit call origin. OperationHost serves
                # off-turn operations (planner, titles, distillers), so
                # its gateway requests are auxiliary — a consumer that
                # must distinguish call origins (the replay gateway)
                # routes on this, never on the request's shape.
                call_kind="auxiliary",
            )
            request_ref = self._record_request_snapshot(request)
            _complete_started = time.perf_counter()
            try:
                result = await gateway.complete(request)
            except BaseException as exc:  # pylint: disable=broad-except
                # Spec §2: a failing call records ``error`` (or
                # ``cancelled``) and re-raises verbatim.
                from asyncio import CancelledError

                self._record_response_snapshot(
                    self._audit,
                    request,
                    None,
                    request_ref,
                    outcome="cancelled" if isinstance(exc, CancelledError) else "error",
                    elapsed_ms=(time.perf_counter() - _complete_started) * 1000.0,
                    error=exc,
                )
                raise
            response_ref = self._record_response_snapshot(self._audit, request, result, request_ref)
            self._publish_audit(request, result, request_ref, response_ref)
            return result
        if self._complete_fn is None:
            raise RuntimeError("OperationHost.complete is not wired on this host")
        return await self._complete_fn(
            messages=messages,
            model_def=model_def,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            response_format=response_format,
        )

    # ── Durable audit (best-effort; never fails the completion) ────────

    def _record_request_snapshot(self, request: Any) -> str | None:
        """Archive one gateway request and return its ref, or ``None``.

        The snapshot mirrors the turn loop's ``_record_request_snapshot``
        field set (messages, tools, model identity, sampling parameters) plus
        the off-turn provenance (operation id, turn index, user) the turn
        loop's snapshots carry implicitly, so both call sites produce
        interchangeable evidence. A recording failure is logged and
        swallowed: the completion proceeds unrecorded, exactly as the turn
        loop's telemetry does.
        """
        audit = self._audit
        if audit is None:
            return None
        model_def = getattr(request, "model_def", None)
        # ``response_format`` arrives as a Pydantic model *class* (structured
        # -output operations pass ``response_format=MinedFacts``), which is
        # not JSON-serializable. Record its name — the evidence goal is
        # "which schema", and the schema itself is authored pack content.
        response_format = getattr(request, "response_format", None)
        if isinstance(response_format, type):
            response_format = response_format.__name__
        elif response_format is not None:
            response_format = str(response_format)
        snapshot = {
            "messages": copy.deepcopy(getattr(request, "messages", []) or []),
            "tools": copy.deepcopy(getattr(request, "tools", []) or []),
            "model_id": str(getattr(model_def, "model_id", "") or ""),
            "provider": str(getattr(model_def, "provider", "") or ""),
            "reasoning_model": bool(getattr(model_def, "reasoning_model", False)),
            "temperature": getattr(request, "temperature", 0.0),
            "max_tokens": getattr(request, "max_tokens", 0),
            "response_format": response_format,
            "operation_id": audit.operation_id,
            "turn_index": audit.turn_index,
            "user_slug": audit.user_slug,
        }
        try:
            # Same call shape as the turn loop's ``record_request(snapshot)``:
            # the recorder owns the {"content": json.dumps(...)} wrapping
            # (``persist_request``), so a raw snapshot keeps the stored
            # object layout identical to an in-turn snapshot's.
            return audit.record_request(snapshot)
        except Exception:  # pylint: disable=broad-except
            logger.warning(
                "operation completion: request recording failed for %s; proceeding unrecorded",
                audit.operation_id,
                exc_info=True,
            )
            return None

    def _record_response_snapshot(
        self,
        audit: CompletionAuditContext | None,
        request: Any,
        result: Any,
        request_ref: str | None,
        *,
        outcome: str = "ok",
        elapsed_ms: float = 0.0,
        error: BaseException | None = None,
    ) -> str | None:
        """Archive one gateway response and return its ref, or ``None``.

        The response snapshot mirrors the turn loop's field set (Spec:
        MODEL-CALL-ARCHIVE §2) so an off-turn archive is interchangeable
        with an in-turn one and the same reader dereferences both. Off-turn
        calls are not streamed and carry no in-call attempt or iteration, so
        those stay honestly unknown (``None``). ``None`` — no recorder
        wired, or a recording failure — is the countable archive gap a
        reader reports.
        """
        if audit is None:
            return None
        if audit.record_response is None:
            return None
        try:
            from kernel.call_archive import (
                call_error_payload,
                continuation_snapshot,
                response_usage_snapshot,
            )

            model_def = getattr(request, "model_def", None)
            duration = (getattr(result, "duration_ms", 0.0) or 0.0) if result is not None else 0.0
            snapshot = {
                "call_id": audit.call_id or None,
                "attempt": None,
                "iteration": None,
                "request_ref": request_ref,
                "outcome": outcome,
                "content": getattr(result, "content", "") if result is not None else "",
                "tool_calls": copy.deepcopy(getattr(result, "tool_calls", None) or [])
                if result is not None
                else [],
                "reasoning_content": getattr(result, "reasoning_content", "")
                if result is not None
                else "",
                "finish_reason": getattr(result, "finish_reason", None)
                if result is not None
                else None,
                "truncated": getattr(result, "truncated", None) if result is not None else None,
                "continuation": continuation_snapshot(
                    getattr(result, "continuation", None) if result is not None else None
                ),
                "provider": str(getattr(model_def, "provider", "") or ""),
                "model_id": str(getattr(model_def, "model_id", "") or ""),
                "provider_request_id": (getattr(result, "request_id", None) or None)
                if result is not None
                else None,
                "usage": response_usage_snapshot(
                    getattr(result, "usage", None) if result is not None else None
                ),
                "duration_ms": duration or elapsed_ms,
                "streamed": False,
                "time_to_first_token_ms": None,
                "chunk_count": None,
            }
            if outcome != "ok":
                snapshot["error"] = (
                    call_error_payload(error)
                    if error is not None
                    else {"class": outcome, "message": ""}
                )
            return audit.record_response(snapshot)
        except Exception:  # pylint: disable=broad-except
            logger.warning(
                "operation completion: response recording failed for %s; leaving archive gap",
                audit.operation_id,
                exc_info=True,
            )
            return None

    def _publish_audit(
        self,
        request: Any,
        result: Any,
        request_ref: str | None,
        response_ref: str | None = None,
    ) -> None:
        """Publish the ``model.request`` ledger row and usage event for one call.

        Fire-and-forget on the audit context's event bus, mirroring the turn
        loop's telemetry posture: a publish failure is logged and dropped, and
        the completion's result still reaches the caller.
        """
        audit = self._audit
        if audit is None or audit.event_bus is None:
            return
        if request_ref is not None:
            self._publish_model_request_row(audit, request_ref, response_ref)
        self._publish_usage_event(audit, request, result)

    def _publish_model_request_row(
        self, audit: CompletionAuditContext, ref: str, response_ref: str | None = None
    ) -> None:
        from kernel.decision_telemetry import MODEL_REQUEST_TOPIC
        from kernel.telemetry import schedule_telemetry_publish

        schedule_telemetry_publish(
            audit.event_bus,
            MODEL_REQUEST_TOPIC,
            {
                "operation_id": audit.operation_id,
                "request_ref": ref,
                "turn_index": audit.turn_index,
                "user_slug": audit.user_slug,
                "thread_store_slug": audit.thread_store_slug,
                # Off-turn call-level join keys (Evidence Envelope §2): the
                # call this row paid for and the dispatch that made it.
                "call_id": audit.call_id or None,
                "dispatch_id": audit.dispatch_id or None,
                "response_ref": response_ref or None,
            },
            source=f"model_request:{audit.operation_id}",
        )

    def _publish_usage_event(
        self, audit: CompletionAuditContext, request: Any, result: Any
    ) -> None:
        """Publish one ``lisa.token_usage`` event for an off-turn completion.

        Without this, a mining distillation's tokens were published nowhere:
        the durable token ledger only saw the turn loop's own calls, so
        off-turn spend was invisible to ``/tokens`` and to cost forensics.
        """
        from kernel.token_usage import (
            build_token_usage_event as _build_usage,
        )
        from kernel.token_usage import (
            publish_token_usage as _publish_usage,
        )

        model_def = getattr(request, "model_def", None)
        usage = getattr(result, "usage", None)

        def _nullable_counter(obj: Any, name: str) -> int | None:
            """Read one counter without coercing unknown to a fabricated zero.

            The pre-Workstream-1 producer wrote ``int(getattr(x, 0) or 0)``,
            so historical v3 ``operation`` rows carry fabricated zeros; new
            rows preserve ``None`` for an unsurfaced counter (nullable_int
            rules: malformed/negative is also unknown, not zero).
            """
            value = getattr(obj, name, None) if obj is not None else None
            if value is None:
                return None
            try:
                number = int(value)
            except (TypeError, ValueError):
                return None
            return number if number >= 0 else None

        def _nullable_number(obj: Any, name: str) -> float | None:
            """Same unknown-preserving read for a float field."""
            value = getattr(obj, name, None) if obj is not None else None
            if value is None:
                return None
            try:
                number = float(value)
            except (TypeError, ValueError):
                return None
            return number if number >= 0 else None

        try:
            _publish_usage(
                audit.event_bus,
                _build_usage(
                    provider=str(getattr(model_def, "provider", "") or ""),
                    model_id=str(getattr(model_def, "model_id", "") or ""),
                    prompt_tokens=_nullable_counter(usage, "prompt_tokens"),
                    completion_tokens=_nullable_counter(usage, "completion_tokens"),
                    total_tokens=_nullable_counter(usage, "total_tokens"),
                    cached_tokens=_nullable_counter(usage, "cached_tokens"),
                    reasoning_tokens=_nullable_counter(usage, "reasoning_tokens"),
                    cache_creation_tokens=_nullable_counter(usage, "cache_creation_tokens"),
                    duration_ms=_nullable_number(result, "duration_ms"),
                    latency_ms=_nullable_counter(usage, "latency_ms"),
                    request_id=str(getattr(result, "request_id", "") or ""),
                    turn_index=audit.turn_index,
                    call_site="operation",
                    user_slug=audit.user_slug,
                    thread_store_slug=audit.thread_store_slug,
                    effective_model_data_source=(
                        str(x)
                        if (x := getattr(model_def, "effective_model_data_source", None))
                        else None
                    ),
                    effective_model_data_version=(
                        str(x)
                        if (x := getattr(model_def, "effective_model_data_version", None))
                        else None
                    ),
                ),
            )
        except Exception:  # pylint: disable=broad-except
            logger.debug("operation completion: usage publish failed", exc_info=True)

    def resolve_model(self, model_id: str) -> Any:
        if self._resolve_model_fn is None:
            raise RuntimeError("OperationHost.resolve_model is not wired on this host")
        return self._resolve_model_fn(model_id)

    def read_authored_text(self, path: str) -> str:
        if self._read_authored_text_fn is None:
            raise RuntimeError("OperationHost.read_authored_text is not wired on this host")
        loaded = self._read_authored_text_fn(path)
        return str(loaded or "")

    async def execute_tool(self, tool_id: str, ctx: Any) -> Any:
        if self._execute_tool_fn is None:
            raise TypeError("OperationHost.execute_tool is not wired on this host")
        return await self._execute_tool_fn(tool_id, ctx)

    async def execute_operation(self, operation_id: str, ctx: Any) -> Any:
        if self._execute_operation_fn is None:
            raise RuntimeError("OperationHost.execute_operation is not wired on this host")
        return await self._execute_operation_fn(operation_id, ctx)

    async def generate_image(self, **kwargs: Any) -> Any:
        if self._generate_image_fn is None:
            raise RuntimeError("OperationHost.generate_image is not wired on this host")
        return await self._generate_image_fn(**kwargs)

    def introspect(self, **kwargs: Any) -> Any:
        if self._introspect_fn is None:
            raise RuntimeError("OperationHost.introspect is not wired on this host")
        return self._introspect_fn(**kwargs)

    async def delegate(self, spec: Any) -> Any:
        if self._delegate_fn is None:
            raise RuntimeError("OperationHost.delegate is not wired on this host")
        return await self._delegate_fn(spec)

    async def fork(self, spec: Any) -> Any:
        if self._fork_fn is None:
            raise RuntimeError("OperationHost.fork is not wired on this host")
        return await self._fork_fn(spec)


def build_callable_host(
    *,
    complete_fn: Any = None,
    resolve_model_fn: Any = None,
    read_authored_text_fn: Any = None,
    execute_tool_fn: Any = None,
    execute_operation_fn: Any = None,
    generate_image_fn: Any = None,
    introspect_fn: Any = None,
    delegate_fn: Any = None,
    fork_fn: Any = None,
    gateway_provider_fn: Any = None,
    audit: CompletionAuditContext | None = None,
) -> _CallableHost:
    """Construct a ``_CallableHost`` from the callable seams + provider gateway.

    ``gateway_provider_fn`` is either a zero-arg callable returning a
    ``ProviderGatewayProvider`` (resolved lazily so the wiring survives a
    ``TurnHooks`` that is assembled before ``context_seams`` is populated) or
    the provider directly. When set, ``complete()`` dispatches through the
    gateway for the model's provider; ``complete_fn`` is only a fallback for
    hosts that have not cut over yet. The legacy ``complete_fn`` path is on
    the removal track.

    ``audit`` wires the durable request recording described in the module
    docstring. The operation registry derives it from the dispatch's
    context; hosts constructing their own host adapters pass ``None`` (no
    off-turn recording, the previous behavior).
    """
    return _CallableHost(
        complete_fn=complete_fn,
        resolve_model_fn=resolve_model_fn,
        read_authored_text_fn=read_authored_text_fn,
        execute_tool_fn=execute_tool_fn,
        execute_operation_fn=execute_operation_fn,
        generate_image_fn=generate_image_fn,
        introspect_fn=introspect_fn,
        delegate_fn=delegate_fn,
        fork_fn=fork_fn,
        gateway_provider_fn=gateway_provider_fn,
        audit=audit,
    )

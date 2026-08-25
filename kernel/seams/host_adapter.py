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
"""

from __future__ import annotations

from typing import Any


class _CallableHost:
    """Adapt the legacy callable bag into the typed ``OperationHost`` Protocol.

    Each method delegates to the corresponding ``ctx.*`` callable. The
    adapter is intentionally thin — it adds no behavior, only types.
    """

    __slots__ = (
        "_complete_fn",
        "_delegate_fn",
        "_execute_operation_fn",
        "_execute_tool_fn",
        "_fork_fn",
        "_generate_image_fn",
        "_gateway_provider_fn",
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
                self._gateway_provider_fn() if callable(self._gateway_provider_fn) else self._gateway_provider_fn
            )
            if provider_gateways is None:
                raise NotImplementedError(
                    "OperationHost.complete: no provider gateway wired "
                    "(provider_gateways is None)."
                )
            gateway = provider_gateways.gateway_for(
                str(getattr(model_def, "provider", "") or "")
            )
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
            )
            return await gateway.complete(request)
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
) -> _CallableHost:
    """Construct a ``_CallableHost`` from the callable seams + provider gateway.

    ``gateway_provider_fn`` is either a zero-arg callable returning a
    ``ProviderGatewayProvider`` (resolved lazily so the wiring survives a
    ``TurnHooks`` that is assembled before ``context_seams`` is populated) or
    the provider directly. When set, ``complete()`` dispatches through the
    gateway for the model's provider; ``complete_fn`` is only a fallback for
    hosts that have not cut over yet. The legacy ``complete_fn`` path is on
    the removal track.
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
    )

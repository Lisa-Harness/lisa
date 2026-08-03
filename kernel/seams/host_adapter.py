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
        "_resolve_model_fn",
        "_read_authored_text_fn",
        "_execute_tool_fn",
        "_execute_operation_fn",
        "_generate_image_fn",
        "_introspect_fn",
        "_delegate_fn",
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
    ) -> None:
        self._complete_fn = complete_fn
        self._resolve_model_fn = resolve_model_fn
        self._read_authored_text_fn = read_authored_text_fn
        self._execute_tool_fn = execute_tool_fn
        self._execute_operation_fn = execute_operation_fn
        self._generate_image_fn = generate_image_fn
        self._introspect_fn = introspect_fn
        self._delegate_fn = delegate_fn

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
) -> _CallableHost:
    """Construct a ``_CallableHost`` from the legacy callable seams."""
    return _CallableHost(
        complete_fn=complete_fn,
        resolve_model_fn=resolve_model_fn,
        read_authored_text_fn=read_authored_text_fn,
        execute_tool_fn=execute_tool_fn,
        execute_operation_fn=execute_operation_fn,
        generate_image_fn=generate_image_fn,
        introspect_fn=introspect_fn,
        delegate_fn=delegate_fn,
    )
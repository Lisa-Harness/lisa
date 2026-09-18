"""Test-support gateway stubs for the provider-gateway completion path.

The kernel's turn loop dispatches every model call through
``hooks.context_seams.provider_gateways.gateway_for(provider).complete(...)``
— the legacy ``hooks.models.complete`` callable is no longer the production path.
Kernel/unit tests historically faked a ``complete`` function returning a
``_StubCompletion``-shaped object. This module bridges that existing fixture
surface onto the gateway contract so tests exercise the real
``ProviderGateway.complete`` boundary instead of the removed callable path.

The adapter accepts a test's existing ``complete`` callable (which takes
``messages``/``model_def``/``max_tokens``/``temperature``/``tools`` kwargs and
returns an object with ``content``/``tool_calls``/optionally ``usage``), and
presents itself as a ``ProviderGatewayProvider`` whose ``gateway_for`` returns
a ``ProviderGateway``-shaped object. Tests wire it via
``ContextSeams(provider_gateways=StubGatewayProvider(complete_fn))``.

This module is test-only; production code must not import it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Awaitable, Callable

from kernel.provider_contracts import CompletionResult, CompletionUsage, StreamedChunk


def _usage_from_result(result: Any) -> CompletionUsage | None:
    """Best-effort map a fake completion's token counters to ``CompletionUsage``.

    Fakes either carry flat counter attrs (``prompt_tokens`` etc.) or nothing.
    Returns ``None`` when no counters are present so the gateway result omits
    usage rather than fabricating zeros.
    """

    prompt = getattr(result, "prompt_tokens", None)
    if prompt is None:
        return None
    return CompletionUsage(
        prompt_tokens=int(getattr(result, "prompt_tokens", 0)),
        completion_tokens=int(getattr(result, "completion_tokens", 0)),
        total_tokens=int(getattr(result, "total_tokens", 0)),
        cached_tokens=int(getattr(result, "cached_tokens", 0)),
        reasoning_tokens=int(getattr(result, "reasoning_tokens", 0)),
        cache_creation_tokens=int(getattr(result, "cache_creation_tokens", 0)),
        duration_ms=float(getattr(result, "duration_ms", 0.0)),
    )


class StubGateway:
    """A ``ProviderGateway``-shaped fake that delegates to a ``complete`` callable.

    When ``complete_stream_fn`` is provided it is invoked as an async-iterator
    of ``StreamedChunk``s for the kernel's streaming path; it is only bound when
    the owning provider was told the test wanted streaming (``complete_stream``),
    so non-streaming tests keep a gateway with no ``complete_stream`` and the
    kernel correctly takes the non-streaming branch.
    """

    def __init__(
        self,
        complete_fn: Callable[..., Awaitable[Any]],
        complete_stream_fn: Callable[..., Any] | None = None,
    ) -> None:
        self._complete_fn = complete_fn
        self._complete_stream_fn = complete_stream_fn
        # Last-seen request lets tests assert the request the kernel sent
        # (``system_prompt_segments``, ``messages``) without the fake having
        # to add a kwarg to its signature (back-compat).
        self.last_request: Any = None

    async def complete_stream(self, request: Any) -> AsyncIterator[StreamedChunk]:
        """Yield chunks from the test's streaming callable, when one is wired."""

        self.last_request = request
        if self._complete_stream_fn is None:
            raise NotImplementedError("No streaming callable wired on the stub gateway")
        async for chunk in self._complete_stream_fn(
            messages=request.messages,
            model_def=request.model_def,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            tools=request.tools,
        ):
            yield chunk

    async def complete(self, request: Any) -> CompletionResult:
        """Delegate to the test's ``complete`` callable and normalize to a result."""

        self.last_request = request
        result = await self._complete_fn(
            messages=request.messages,
            model_def=request.model_def,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            tools=request.tools,
        )
        provider = str(getattr(request.model_def, "provider", "") or "")
        model_id = str(getattr(request.model_def, "model_id", "") or "")
        return CompletionResult(
            content=str(getattr(result, "content", "") or ""),
            provider=provider,
            model_id=model_id,
            display_name=str(getattr(request.model_def, "display_name", "") or model_id),
            tool_calls=list(getattr(result, "tool_calls", None) or []),
            usage=_usage_from_result(result),
            reasoning_content=str(getattr(result, "reasoning_content", "") or ""),
            # Preserved so completion diagnostics (and the reasoning-tail
            # caveat's truncation clause) can distinguish a budget-starved
            # ``length`` stop from a clean one — real gateways set this.
            finish_reason=getattr(result, "finish_reason", None),
        )


class StubGatewayProvider:
    """A ``ProviderGatewayProvider``-shaped fake backed by one ``complete`` callable.

    ``gateway_for`` returns the same :class:`StubGateway` for any provider the
    test's fake supports (the fake does not discriminate by provider), so every
    provider name resolves. This is what makes a 175-test suite's bare
    ``complete``-callable fakes keep working against the gateway path.
    """

    def __init__(
        self,
        complete_fn: Callable[..., Awaitable[Any]],
        complete_stream_fn: Callable[..., Any] | None = None,
    ) -> None:
        self._complete_fn = complete_fn
        self._complete_stream_fn = complete_stream_fn
        #: Most recent gateway handed out by ``gateway_for``. Lets a test
        #: read the last completion request without re-plumbing its fake.
        self.last_gateway: StubGateway | None = None

    def gateway_for(self, provider: str) -> StubGateway:
        _ = provider  # the fake resolves every provider to the same stub gateway
        gateway = StubGateway(self._complete_fn, self._complete_stream_fn)
        self.last_gateway = gateway
        return gateway


__all__ = ["StubGateway", "StubGatewayProvider"]

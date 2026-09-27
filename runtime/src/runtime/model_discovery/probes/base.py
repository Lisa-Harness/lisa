"""Probe protocol and shared httpx helpers.

A probe is an async callable that probes one provider and returns a
:class:`~runtime.model_discovery.types.ProbeResult`. Probes never
raise: every failure path is converted into a typed ``ProbeResult``
with a :class:`~runtime.model_discovery.types.ProbeError` whose
``safe_message`` has been run through :mod:`~runtime.model_discovery.redact`.

The shared ``_safe_probe`` helper centralises the ``try/except`` shape
so no individual probe can accidentally leak a raw exception string.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from ..classify import is_reasoning_model
from ..redact import classify_status, safe_http_error
from ..types import (
    DiscoveredModel,
    ProbeError,
    ProbeResult,
    ProbeStatus,
)


class ProviderProbe:
    """Base class for provider probes.

    Subclasses set ``provider`` and implement ``_probe`` returning either
    a ``tuple`` of models (success) or raising — in which case the base
    ``probe`` method wraps the exception in a typed result.

    Returning ``None`` from ``_probe`` signals ``ProbeStatus.EMPTY``
    (200 OK but no models). Returning a tuple signals ``OK``.
    """

    provider: str = ""

    async def probe(
        self,
        client: httpx.AsyncClient,
        timeout: float,
    ) -> ProbeResult:
        start = time.perf_counter()
        try:
            models = await self._probe(client, timeout)
        except Exception as exc:  # pylint: disable=broad-except
            # Last-resort safety net: no probe may leak a raw exception.
            status_name = classify_status(exc)
            status = ProbeStatus(status_name)
            safe = safe_http_error(exc, self.provider)
            detail = repr(exc)  # for logs only
            elapsed = int((time.perf_counter() - start) * 1000)
            return ProbeResult(
                provider=self.provider,
                status=status,
                error=ProbeError(
                    provider=self.provider,
                    status=status,
                    safe_message=safe,
                    detail=detail,
                ),
                elapsed_ms=elapsed,
            )

        elapsed = int((time.perf_counter() - start) * 1000)
        if not models:
            return ProbeResult(
                provider=self.provider,
                status=ProbeStatus.EMPTY,
                elapsed_ms=elapsed,
            )
        return ProbeResult(
            provider=self.provider,
            status=ProbeStatus.OK,
            models=tuple(models),
            elapsed_ms=elapsed,
        )

    async def _probe(
        self,
        client: httpx.AsyncClient,
        timeout: float,
    ) -> tuple[DiscoveredModel, ...] | None:
        raise NotImplementedError


def _model(
    model_id: str,
    provider: str,
    *,
    thinking_model: bool = False,
    display_name: str = "",
    capabilities: tuple[str, ...] = (),
    cost_tier: str = "",
    latency_tier: str = "",
    context_window: int = 0,
    prompt_price: float = 0.0,
    completion_price: float = 0.0,
    pricing_currency: str = "USD",
    description: str = "",
    structured_reliability: float = 0.0,
    completion_api: str = "chat_completions",
) -> DiscoveredModel:
    """Construct a :class:`DiscoveredModel` with the shared heuristic.

    The ``reasoning_model`` flag is always derived from the shared
    ``is_reasoning_model`` substring heuristic for consistency. The
    enrichment fields are optional and populated by each probe from the
    metadata the provider's list endpoint returns.

    The ``"reasoning"`` capability is derived from ``reasoning_model`` so a
    model the heuristic flags as reasoning-capable is always routable via
    ``'"reasoning" in route.required_capabilities'`` — even when a probe
    (e.g. ollama) does not explicitly add it. Probes that already include
    ``"reasoning"`` are unaffected; the static registry override in
    ``known_models.yaml`` still wins because each probe applies it before
    calling this constructor.
    """

    reasoning = is_reasoning_model(model_id)
    if reasoning and "reasoning" not in capabilities:
        capabilities = (*capabilities, "reasoning")

    return DiscoveredModel(
        model_id=model_id,
        provider=provider,
        reasoning_model=reasoning,
        thinking_model=thinking_model,
        display_name=display_name,
        capabilities=capabilities,
        cost_tier=cost_tier,
        latency_tier=latency_tier,
        context_window=context_window,
        prompt_price=prompt_price,
        completion_price=completion_price,
        pricing_currency=pricing_currency,
        description=description,
        structured_reliability=structured_reliability,
        completion_api=completion_api,
    )


def _extract_rows(payload: Any, container_key: str) -> list[Any]:
    """Return ``payload[container_key]`` as a list, tolerating bad shapes."""

    if not isinstance(payload, dict):
        return []
    rows = payload.get(container_key)
    if not isinstance(rows, list):
        return []
    return rows

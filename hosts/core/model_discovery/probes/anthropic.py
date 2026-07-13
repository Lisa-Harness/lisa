"""Anthropic probe.

Anthropic now exposes ``GET /v1/models`` (with the ``anthropic-version``
header). This probe calls it for real and falls back to a conservative
hardcoded tuple only on a 404 (environments that don't expose the
endpoint). The fallback is tagged ``EMPTY`` with an explanatory note so
the UI can distinguish "live list" from "conservative defaults".
"""

from __future__ import annotations

import httpx

from ..classify import env_any
from ..types import DiscoveredModel, ProbeError, ProbeResult, ProbeStatus
from .base import ProviderProbe, _model

_LISA_ANTHROPIC_API_KEY_ENV = "LISA_ANTHROPIC_API_KEY"
_ANTHROPIC_VERSION = "2023-06-01"

# Conservative fallback used only when the live list endpoint is absent.
# These IDs are stable aliases that Anthropic maintains across versions.
_FALLBACK_MODELS = (
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-1-20250805",
)


class AnthropicProbe(ProviderProbe):
    provider = "anthropic"

    async def probe(
        self,
        client: httpx.AsyncClient,
        timeout: float,
    ) -> ProbeResult:
        api_key = env_any(_LISA_ANTHROPIC_API_KEY_ENV, "ANTHROPIC_API_KEY")
        if not api_key:
            return ProbeResult(provider=self.provider, status=ProbeStatus.SKIPPED)
        return await super().probe(client, timeout)

    async def _probe(
        self,
        client: httpx.AsyncClient,
        timeout: float,
    ) -> tuple[DiscoveredModel, ...] | None:
        api_key = env_any(_LISA_ANTHROPIC_API_KEY_ENV, "ANTHROPIC_API_KEY")

        response = await client.get(
            "https://api.anthropic.com/v1/models",
            headers={
                "x-api-key": api_key,
                "anthropic-version": _ANTHROPIC_VERSION,
            },
            timeout=timeout,
        )
        if response.status_code == 404:
            # Endpoint absent in this environment — use conservative
            # defaults. Signal EMPTY so the UI notes it.
            return None
        response.raise_for_status()
        payload = response.json()
        rows = (
            payload.get("data", [])
            if isinstance(payload, dict) and isinstance(payload.get("data"), list)
            else []
        )
        models: list[DiscoveredModel] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            model_id = str(row.get("id") or "").strip()
            if not model_id:
                continue
            models.append(_model(model_id, self.provider))
        if models:
            return tuple(models)
        # 200 but empty list — fall back rather than reporting zero.
        return None

    async def probe_with_fallback(
        self,
        client: httpx.AsyncClient,
        timeout: float,
    ) -> ProbeResult:
        """Like ``probe`` but substitutes conservative defaults on EMPTY."""

        result = await self.probe(client, timeout)
        if result.status is ProbeStatus.EMPTY:
            fallback_models = tuple(
                _model(model_id, self.provider) for model_id in _FALLBACK_MODELS
            )
            return ProbeResult(
                provider=self.provider,
                status=ProbeStatus.EMPTY,
                models=fallback_models,
                error=ProbeError(
                    provider=self.provider,
                    status=ProbeStatus.EMPTY,
                    safe_message=(
                        f"{self.provider}: list endpoint unavailable; "
                        "using conservative default models"
                    ),
                ),
                elapsed_ms=result.elapsed_ms,
            )
        return result
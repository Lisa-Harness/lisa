"""Anthropic probe.

Anthropic now exposes ``GET /v1/models`` (with the ``anthropic-version``
header). This probe calls it for real and falls back to a conservative
hardcoded tuple only on a 404 (environments that don't expose the
endpoint). The fallback is tagged ``EMPTY`` with an explanatory note so
the UI can distinguish "live list" from "conservative defaults".
"""

from __future__ import annotations

import httpx

from ..classify import apply_known_override, env_any
from ..types import DiscoveredModel, ProbeError, ProbeResult, ProbeStatus
from .base import ProviderProbe, _model

_LISA_ANTHROPIC_API_KEY_ENV = "LISA_ANTHROPIC_API_KEY"
_ANTHROPIC_VERSION = "2023-06-01"

# Conservative fallback used only when the live list endpoint is absent.
# Stable version-agnostic aliases (verified in the live catalog) — dated
# snapshot ids (``claude-...-20250929``) retire and pin the same broken
# default class this fallback would otherwise re-create.
_FALLBACK_MODELS = (
    "claude-sonnet-5",
    "claude-haiku-5-5",
    "claude-opus-5",
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
        if response.status_code == httpx.codes.NOT_FOUND:
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
            models.append(_enrich_anthropic_model(model_id, row))
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
                _enrich_anthropic_model(model_id, {}) for model_id in _FALLBACK_MODELS
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


def _enrich_anthropic_model(model_id: str, row: dict) -> DiscoveredModel:
    """Build a :class:`DiscoveredModel` enriched from Anthropic's model metadata.

    Anthropic's ``/v1/models`` response carries ``display_name`` and
    ``type`` alongside ``id``. We infer the rest from the model family:

    - Claude Opus: premium tier, extended thinking (4+ models).
    - Claude Sonnet: standard tier, medium latency.
    - Claude Haiku: budget tier, low latency.
    - All Claude models support text, tools, structured_output, and vision.
    """

    display_name = str(row.get("display_name") or "").strip()
    lowered = model_id.lower()

    capabilities: tuple[str, ...] = ("text", "tools", "structured_output", "vision")
    thinking_model = False
    cost_tier = ""
    latency_tier = ""

    if "opus" in lowered:
        cost_tier = "frontier"
        latency_tier = "high"
        thinking_model = True
    elif "sonnet" in lowered:
        cost_tier = "standard"
        latency_tier = "medium"
    elif "haiku" in lowered:
        cost_tier = "budget"
        latency_tier = "low"
    elif "fable" in lowered:
        cost_tier = "premium"
        latency_tier = "medium"

    if not display_name:
        display_name = (
            f"Claude {model_id.split('-', 1)[1].title()}" if "-" in model_id else model_id
        )

    enrichment = apply_known_override(
        "anthropic",
        model_id,
        {
            "thinking_model": thinking_model,
            "display_name": display_name,
            "capabilities": capabilities,
            "cost_tier": cost_tier,
            "latency_tier": latency_tier,
        },
    )
    return _model(
        model_id,
        "anthropic",
        thinking_model=bool(enrichment.get("thinking_model", thinking_model)),
        display_name=str(enrichment.get("display_name", display_name)),
        capabilities=tuple(enrichment.get("capabilities", capabilities)),
        cost_tier=str(enrichment.get("cost_tier", cost_tier)),
        latency_tier=str(enrichment.get("latency_tier", latency_tier)),
        structured_reliability=1.0,
    )

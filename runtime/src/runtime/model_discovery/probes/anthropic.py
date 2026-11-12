"""Anthropic probe.

Anthropic now exposes ``GET /v1/models`` (with the ``anthropic-version``
header). This probe calls it for real and falls back to a conservative
hardcoded tuple only on a 404 (environments that don't expose the
endpoint). The fallback is tagged ``EMPTY`` with an explanatory note so
the UI can distinguish "live list" from "conservative defaults".
"""

from __future__ import annotations

from typing import Any

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
    ) -> tuple[DiscoveredModel, ...] | ProbeResult | None:
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
        dropped: list[str] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            model_id = str(row.get("id") or "").strip()
            if not model_id:
                continue
            if row.get("lifecycle") == "retired":
                dropped.append(model_id)
                continue
            models.append(_enrich_anthropic_model(model_id, row))
        if dropped:
            return ProbeResult(
                provider=self.provider,
                status=ProbeStatus.OK if models else ProbeStatus.EMPTY,
                models=tuple(models),
                dropped_model_ids=tuple(dropped),
            )
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


_LINE_TIERS: dict[str, tuple[str, str]] = {
    "mythos": ("frontier", "high"),
    "opus": ("frontier", "high"),
    "fable": ("premium", "medium"),
    "sonnet": ("standard", "medium"),
    "haiku": ("budget", "low"),
}


def _supported(node: object) -> bool:
    """Read a ``CapabilitySupport`` object (``{"supported": bool}``)."""
    return isinstance(node, dict) and node.get("supported") is True


def _positive_int(value: object) -> int:
    return value if isinstance(value, int) and value > 0 else 0


def _enrich_anthropic_model(model_id: str, row: dict[str, Any]) -> DiscoveredModel:
    """Build a :class:`DiscoveredModel` from Anthropic's ``/v1/models`` row.

    The row is authoritative when it carries the field: ``max_input_tokens``
    (context window), ``max_tokens`` (per-call output ceiling), ``line``
    (cost/latency tier) and ``capabilities`` (vision, structured outputs, and
    manual thinking). ``thinking_model`` tracks
    ``capabilities.thinking.types.enabled`` because the gateway sends
    ``{"type": "enabled"}``; a model that only accepts adaptive thinking
    would reject it. Rows without these fields (older API versions, the
    404 fallback) keep the id-substring heuristic.
    """

    display_name = str(row.get("display_name") or "").strip()
    lowered = model_id.lower()

    line = str(row.get("line") or "").strip().lower()
    if line not in _LINE_TIERS:
        line = next((name for name in _LINE_TIERS if name in lowered), "")
    cost_tier, latency_tier = _LINE_TIERS.get(line, ("", ""))

    raw_caps = row.get("capabilities")
    if isinstance(raw_caps, dict):
        caps: list[str] = ["text", "tools"]
        if _supported(raw_caps.get("structured_outputs")):
            caps.append("structured_output")
        if _supported(raw_caps.get("image_input")):
            caps.append("vision")
        thinking = raw_caps.get("thinking")
        types = thinking.get("types") if isinstance(thinking, dict) else None
        thinking_model = isinstance(types, dict) and _supported(types.get("enabled"))
        if _supported(thinking):
            caps.append("reasoning")
        capabilities: tuple[str, ...] = tuple(caps)
        published: set[str] = {"capabilities", "thinking_model"}
    else:
        capabilities = ("text", "tools", "structured_output", "vision")
        thinking_model = line in {"opus", "mythos"}
        published = set()
    context_window = _positive_int(row.get("max_input_tokens"))
    max_completion_tokens = _positive_int(row.get("max_tokens"))
    if context_window:
        published.add("context_window")
    if max_completion_tokens:
        published.add("max_completion_tokens")

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
            "context_window": context_window,
            "max_completion_tokens": max_completion_tokens,
        },
        published=published,
    )
    return _model(
        model_id,
        "anthropic",
        thinking_model=bool(enrichment.get("thinking_model", thinking_model)),
        display_name=str(enrichment.get("display_name", display_name)),
        capabilities=tuple(enrichment.get("capabilities", capabilities)),
        cost_tier=str(enrichment.get("cost_tier", cost_tier)),
        latency_tier=str(enrichment.get("latency_tier", latency_tier)),
        context_window=int(enrichment.get("context_window") or 0),
        max_completion_tokens=int(enrichment.get("max_completion_tokens") or 0),
        structured_reliability=1.0,
    )

"""OpenRouter probe.

OpenRouter's ``GET /api/v1/models`` returns an unusually rich payload per
model — ``name``, ``context_length``, ``architecture.{input,output}_modalities``,
``supported_parameters``, ``pricing.{prompt,completion,...}``, a ``reasoning``
block, and ``description``. This probe consumes all of it into a
:class:`~runtime.model_discovery.types.DiscoveredModel` so ``lisa models
discover`` surfaces OpenRouter models with real pricing and capabilities.
The static ``known_models.yaml`` registry is still the last-word override for
fields the API under-reports.

Filtered to chat-capable text models (``text`` among the input, not a
media/embedding output modality) to keep the emitted manifest lean; the raw
catalog spans image/audio/embedding models the host cannot drive as chat
completions.

Retirement-aware: the catalog publishes per-model ``expiration_date``;
rows past that date are dropped (recorded as a UI note), upcoming
expiries are carried on the model so pin heuristics can avoid them.
``:batch`` variants (Batch-API-only twins) are excluded — the interactive
turn loop cannot drive them.
"""

from __future__ import annotations

from datetime import date

import httpx

from ..classify import (
    apply_known_override,
    env_any,
    is_batch_only_model,
    is_code_model,
    is_model_expired,
    parse_expiry_date,
)
from ..types import DiscoveredModel, ProbeResult, ProbeStatus
from .base import ProviderProbe, _model

_LISA_OPENROUTER_API_KEY_ENV = "LISA_OPENROUTER_API_KEY"
_OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

#: Cost-tier thresholds in USD per million tokens (prompt and completion
#: prices are each expressed per-token here; the tiers are derived from
#: the scaled per-million rates). Roughly: free/local budget, cheap
#: standard, premium, frontier beyond.
_BUDGET_TIER_MAX_USD_PER_M = 0.5
_STANDARD_TIER_MAX_USD_PER_M = 3.0
_PREMIUM_TIER_MAX_USD_PER_M = 15.0
# Page size is 500 by default; the catalog is a few hundred chat-capable
# models, so one or two pages cover it. We cap pages to bound a runaway
# pagination loop.
_MAX_PAGES = 4


class OpenRouterProbe(ProviderProbe):
    provider = "openrouter"

    async def probe(
        self,
        client: httpx.AsyncClient,
        timeout: float,
    ) -> ProbeResult:
        api_key = env_any(_LISA_OPENROUTER_API_KEY_ENV, "OPENROUTER_API_KEY")
        if not api_key:
            return ProbeResult(provider=self.provider, status=ProbeStatus.SKIPPED)
        return await super().probe(client, timeout)

    async def _probe(
        self,
        client: httpx.AsyncClient,
        timeout: float,
    ) -> tuple[DiscoveredModel, ...] | ProbeResult | None:
        api_key = env_any(_LISA_OPENROUTER_API_KEY_ENV, "OPENROUTER_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"}

        models: list[DiscoveredModel] = []
        dropped: list[str] = []
        url: str | None = _OPENROUTER_MODELS_URL
        seen: set[str] = set()
        for _ in range(_MAX_PAGES):
            if not url:
                break
            response = await client.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            payload = response.json() if response.content else {}
            rows = (
                payload.get("data")
                if isinstance(payload, dict) and isinstance(payload.get("data"), list)
                else []
            )
            for row in rows:
                if not isinstance(row, dict):
                    continue
                model_id = str(row.get("id") or "").strip()
                if not model_id or model_id in seen:
                    continue
                if not _is_chat_capable(row):
                    continue
                seen.add(model_id)
                if is_batch_only_model(model_id):
                    # Batch-API-only twin of an interactive model — the
                    # host cannot drive it on the turn loop.
                    continue
                expiry = parse_expiry_date(row.get("expiration_date"))
                if is_model_expired(expiry):
                    dropped.append(model_id)
                    continue
                models.append(_enrich_openrouter_model(model_id, row, expires_at=expiry))
            # Follow pagination via ``links.next`` when present.
            links = payload.get("links") if isinstance(payload, dict) else None
            next_url = links.get("next") if isinstance(links, dict) else None
            url = next_url if (isinstance(next_url, str) and next_url.strip()) else None
        if dropped:
            return ProbeResult(
                provider=self.provider,
                status=ProbeStatus.OK if models else ProbeStatus.EMPTY,
                models=tuple(models),
                dropped_model_ids=tuple(dropped),
            )
        return tuple(models) if models else None


def _is_chat_capable(row: dict) -> bool:
    """Return True for models the host can drive as chat completions.

    Accepts models whose output modalities include ``text`` and whose input
    includes ``text``. Excludes embedding- / image- / audio-primary models.
    """

    architecture = row.get("architecture") if isinstance(row.get("architecture"), dict) else {}
    input_modalities = architecture.get("input_modalities") or []
    output_modalities = architecture.get("output_modalities") or []
    if "text" not in _as_str_set(input_modalities):
        return False
    out = _as_str_set(output_modalities)
    # Must produce text in its primary modality (the ``modality`` field is
    # e.g. ``"text->text"`` / ``"multimodal->text"``; the output side decides
    # chat vs. embedding/image models).
    modality = str(architecture.get("modality") or "")
    return ("text" in out) or modality.endswith("->text")


def _as_str_set(value: object) -> set[str]:
    """Coerce a list of strings into a set, tolerating non-list input."""

    if isinstance(value, list):
        return {str(v).strip().lower() for v in value if str(v or "").strip()}
    return set()


def _enrich_openrouter_model(
    model_id: str, row: dict, *, expires_at: date | None = None
) -> DiscoveredModel:
    """Build a :class:`DiscoveredModel` from one OpenRouter model row.

    OpenRouter provides capability and pricing metadata directly, so
    capabilities, tiers, and context window come from the API payload, not
    inference — with a known-models override layered on top for the models it
    under-reports. ``expires_at`` carries the catalog's published
    ``expiration_date`` (``None`` = not published).
    """

    display_name = str(row.get("name") or "").strip() or model_id
    context_window = _int_or_zero(row.get("context_length"))
    top_provider = row.get("top_provider") if isinstance(row.get("top_provider"), dict) else {}
    if not context_window:
        context_window = _int_or_zero(top_provider.get("context_length"))

    # ── Pricing ──
    pricing = row.get("pricing") if isinstance(row.get("pricing"), dict) else {}
    prompt_price = _price(pricing.get("prompt"))
    completion_price = _price(pricing.get("completion"))
    cache_read_price = _price(pricing.get("input_cache_read"))
    cache_write_price = _price(pricing.get("input_cache_write"))

    # ── Capabilities ──
    supported_params = _as_str_set(row.get("supported_parameters"))
    architecture = row.get("architecture") if isinstance(row.get("architecture"), dict) else {}
    input_modalities = _as_str_set(architecture.get("input_modalities"))
    capabilities: list[str] = ["text"]
    if "tools" in supported_params:
        capabilities.append("tools")
    if "structured_outputs" in supported_params or "response_format" in supported_params:
        capabilities.append("structured_output")
    if (
        "image" in input_modalities
        or "multimodal" in str(architecture.get("modality") or "").lower()
    ):
        capabilities.append("vision")
    if "reasoning" in supported_params or "include_reasoning" in supported_params:
        capabilities.append("reasoning")
    if is_code_model(model_id):
        capabilities.append("code")

    # ── Reasoning / thinking ──
    # ``row["reasoning"]`` is present for reasoning-capable models (with
    # ``mandatory``/``default_effort``); reasoning_model is additionally
    # derived inside ``_model`` from the shared heuristic. ``thinking_model``
    # = reasoning-capable per the API block OR the heuristic.
    reasoning_block = row.get("reasoning") if isinstance(row.get("reasoning"), dict) else None
    thinking_model = reasoning_block is not None

    # ── Tiers ──
    # Cost tier is derived from the per-token price when available so the
    # router's ``cost_tier`` rules stay driven by real pricing, not a
    # hand-maintained label. Latency tier is derived from the reasoning block
    # (thinking models are slower) and falls back to mid.
    cost_tier = _cost_tier_from_price(prompt_price, completion_price)
    latency_tier = "high" if thinking_model else "medium"

    description = str(row.get("description") or "").strip()

    base_enrichment = {
        "display_name": display_name,
        "capabilities": tuple(capabilities),
        "cost_tier": cost_tier,
        "latency_tier": latency_tier,
        "context_window": context_window,
        "prompt_price": prompt_price,
        "completion_price": completion_price,
        "cache_read_price": cache_read_price,
        "cache_write_price": cache_write_price,
        "description": description,
        "thinking_model": thinking_model,
    }
    merged = apply_known_override("openrouter", model_id, base_enrichment)
    return _model(
        model_id,
        "openrouter",
        thinking_model=bool(merged.get("thinking_model", False)),
        display_name=str(merged.get("display_name") or display_name),
        capabilities=tuple(merged.get("capabilities") or ()),
        cost_tier=str(merged.get("cost_tier") or ""),
        latency_tier=str(merged.get("latency_tier") or ""),
        context_window=_int_or_zero(merged.get("context_window")),
        prompt_price=float(merged.get("prompt_price") or 0.0),
        completion_price=float(merged.get("completion_price") or 0.0),
        cache_read_price=float(merged.get("cache_read_price") or 0.0),
        cache_write_price=float(merged.get("cache_write_price") or 0.0),
        description=str(merged.get("description") or ""),
        expires_at=expires_at,
    )


def _int_or_zero(value: object) -> int:
    """Coerce a numeric payload field (str or int) to a non-negative int."""

    try:
        return max(0, int(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def _price(value: object) -> float:
    """Coerce an OpenRouter price (string, $/token) to a non-negative float.

    OpenRouter's ``pricing.*`` fields are serialized as decimal strings.
    Empty/unparseable → 0.0 (unknown).
    """

    try:
        return max(0.0, float(str(value).strip()))
    except (TypeError, ValueError):
        return 0.0


def _cost_tier_from_price(prompt_price: float, completion_price: float) -> str:
    """Map per-token USD price to a budget/standard/premium/frontier tier.

    Uses completion (output) price as the driver — output dominates a
    thinking/agent workload's cost. Thresholds are per-million-token USD;
    they mirror the tier boundaries the router's rules assume. When neither
    price is known, return "" so the manifest omits the tier and the router
    falls back to a neutral default.
    """
    if completion_price <= 0.0 and prompt_price <= 0.0:
        return ""
    in_m = prompt_price * 1e6
    out_m = completion_price * 1e6
    # Roughly: free/local budget (<0.5), cheap standard (<3), premium (<15),
    # frontier beyond.
    if out_m < _BUDGET_TIER_MAX_USD_PER_M and in_m < _BUDGET_TIER_MAX_USD_PER_M:
        return "budget"
    if out_m < _STANDARD_TIER_MAX_USD_PER_M:
        return "standard"
    if out_m < _PREMIUM_TIER_MAX_USD_PER_M:
        return "premium"
    return "frontier"


__all__ = ["OpenRouterProbe", "_enrich_openrouter_model", "_is_chat_capable"]

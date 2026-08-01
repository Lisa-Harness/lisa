"""Ollama probe — local model tags."""

from __future__ import annotations

from typing import Any

import httpx

from ..classify import apply_known_override, env_any, is_code_model
from ..types import DiscoveredModel
from .base import ProviderProbe, _extract_rows, _model

_LISA_OLLAMA_HOST_ENV = "LISA_OLLAMA_HOST"

#: Maps Ollama ``/api/show`` capability strings onto Lisa capability names.
#: ``completion`` is the baseline ``text`` already assumed for generation
#: models, so it is only consulted to distinguish embedding-only models.
_OLLAMA_CAP_TO_LISA: dict[str, str] = {
    "tools": "tools",
    "vision": "vision",
    "thinking": "reasoning",
}


def _normalize_host(host: str) -> str:
    host = host.strip() or "127.0.0.1:11434"
    if not (host.startswith("http://") or host.startswith("https://")):
        host = f"http://{host}"
    return host.rstrip("/")


async def _fetch_show(
    client: httpx.AsyncClient,
    base: str,
    model_id: str,
    timeout: float,
) -> dict | None:
    """Best-effort per-model ``/api/show`` enrichment, or ``None``.

    A failure for a single model must never fail discovery — we degrade to
    the tag-based heuristic when ``/api/show`` is unavailable or errors.
    """
    try:
        response = await client.post(
            f"{base}/api/show",
            json={"model": model_id},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else None
    except Exception:  # pylint: disable=broad-except
        return None


class OllamaProbe(ProviderProbe):
    provider = "ollama"

    async def _probe(
        self,
        client: httpx.AsyncClient,
        timeout: float,
    ) -> tuple[DiscoveredModel, ...] | None:
        host = env_any(_LISA_OLLAMA_HOST_ENV, "OLLAMA_HOST")
        base = _normalize_host(host)

        response = await client.get(f"{base}/api/tags", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        rows = _extract_rows(payload, "models")
        models: list[DiscoveredModel] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("name") or "").strip()
            if not model_id:
                continue
            show = await _fetch_show(client, base, model_id, timeout)
            models.append(_enrich_ollama_model(model_id, item, show_payload=show))
        return tuple(models) if models else None


def _enrich_ollama_model(
    model_id: str,
    item: dict,
    *,
    show_payload: dict | None = None,
) -> DiscoveredModel:
    """Build a :class:`DiscoveredModel` enriched from Ollama tag metadata.

    Ollama's ``/api/tags`` response carries ``details`` (with
    ``parameter_size``, ``quantization_level``, ``family``) and ``size``
    (bytes on disk). We infer cost/latency tier from parameter size:
    tiny (<3B) = budget/low, small (3-13B) = standard/medium,
    large (>13B) = premium/high. Cloud-suffixed models (``:cloud``) are
    hosted by Ollama and billed as standard tier regardless of size.

    When ``show_payload`` (the per-model ``/api/show`` response) is
    supplied, its authoritative ``capabilities`` array and
    ``model_info.general.context_length`` override the family-based
    ``vision`` guess and the default ``context_window``. The static
    registry in ``known_models.yaml`` is applied last and still wins.
    """

    details = item.get("details") if isinstance(item.get("details"), dict) else {}
    parameter_size = str(details.get("parameter_size") or "").strip()
    family = str(details.get("family") or "").strip()

    # Authoritative per-model capabilities from /api/show (best-effort).
    show_caps: set[str] = set()
    if show_payload:
        raw_caps = show_payload.get("capabilities")
        if isinstance(raw_caps, list):
            for cap in raw_caps:
                lisa_cap = _OLLAMA_CAP_TO_LISA.get(str(cap).strip().lower())
                if lisa_cap:
                    show_caps.add(lisa_cap)
        # Ollama marks embedding-only models by exposing no "completion".
        if "embedding" in {str(c).strip().lower() for c in (raw_caps or [])} and not show_caps:
            show_caps.add("embedding")

    capabilities: list[str] = ["text", "tools", "structured_output"]
    if show_caps:
        # Merge: start from the authoritative set, keep structured_output.
        capabilities = sorted(show_caps | {"structured_output"})
        if "text" not in capabilities and "embedding" not in capabilities:
            capabilities.append("text")
    if family in {"clip", "llava"} or "vision" in model_id.lower():
        if "vision" not in capabilities:
            capabilities.append("vision")
    # Ollama tags carry no capability metadata beyond family/size, so the
    # ``code`` capability is derived from the model-id heuristic — same as
    # ``reasoning`` (which ``_model`` appends). Ollama exposes no ``code``
    # capability in /api/show, so this heuristic cannot be replaced. This
    # lets the router rule ``'"code" in route.required_capabilities'`` and
    # ``_pick_code_model`` select a coding model (e.g. glm-5.2:cloud)
    # instead of silently defaulting. ``known_models.yaml`` can still
    # override per model.
    if is_code_model(model_id):
        capabilities.append("code")

    cost_tier = ""
    latency_tier = ""
    context_window = 0
    # Local Ollama models treat ``response_format`` as a *soft hint*: small
    # models routinely ignore the declared schema and emit prose, while larger
    # models honor it more reliably. Default the confidence to a conservative
    # floor for small models and only raise it as parameter count grows. Cloud
    # (``:cloud``) models are served by Ollama's infrastructure and are more
    # dependable, so they get a higher default. ``known_models.yaml`` can
    # override this per model where measured.
    structured_reliability = 0.0

    lowered = model_id.lower()
    if lowered.endswith(":cloud"):
        # Cloud-hosted models — billing is opaque, assume standard.
        cost_tier = "standard"
        latency_tier = "medium"
        structured_reliability = 0.8
    elif parameter_size:
        # Parse "7B", "13B", "70B" etc.
        size_str = parameter_size.lower().rstrip("b").strip()
        try:
            size_b = float(size_str)
        except ValueError:
            size_b = 0.0
        if size_b <= 0:
            pass
        elif size_b <= 3:
            cost_tier = "budget"
            latency_tier = "low"
            structured_reliability = 0.0
        elif size_b <= 13:
            cost_tier = "standard"
            latency_tier = "medium"
            structured_reliability = 0.4
        elif size_b <= 35:
            cost_tier = "premium"
            latency_tier = "high"
            structured_reliability = 0.7
        else:
            cost_tier = "frontier"
            latency_tier = "high"
            structured_reliability = 0.85

    display_name = model_id.replace("-", " ").title()
    description = ""
    if family:
        description = f"Ollama {family} model ({parameter_size or 'unknown size'})"

    # Authoritative context length from /api/show model_info (best-effort).
    context_window = 0
    if show_payload:
        model_info = (
            show_payload.get("model_info")
            if isinstance(show_payload.get("model_info"), dict)
            else None
        )
        if model_info:
            raw_ctx = model_info.get("general.context_length")
            if isinstance(raw_ctx, (int, float)) and raw_ctx > 0:
                context_window = int(raw_ctx)

    enrichment = apply_known_override(
        "ollama",
        model_id,
        {
            "capabilities": tuple(capabilities),
            "cost_tier": cost_tier,
            "latency_tier": latency_tier,
            "description": description,
            "context_window": context_window,
            "structured_reliability": structured_reliability,
        },
    )
    return _model(
        model_id,
        "ollama",
        display_name=display_name,
        capabilities=tuple(enrichment.get("capabilities", tuple(capabilities))),
        cost_tier=str(enrichment.get("cost_tier", cost_tier)),
        latency_tier=str(enrichment.get("latency_tier", latency_tier)),
        context_window=int(enrichment.get("context_window", context_window)),
        description=str(enrichment.get("description", description)),
        structured_reliability=float(
            enrichment.get("structured_reliability", structured_reliability)
        ),
    )

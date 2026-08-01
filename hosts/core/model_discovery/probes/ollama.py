"""Ollama probe — local model tags."""

from __future__ import annotations

import httpx

from ..classify import apply_known_override, env_any, is_code_model
from ..types import DiscoveredModel
from .base import ProviderProbe, _extract_rows, _model

_LISA_OLLAMA_HOST_ENV = "LISA_OLLAMA_HOST"


def _normalize_host(host: str) -> str:
    host = host.strip() or "127.0.0.1:11434"
    if not (host.startswith("http://") or host.startswith("https://")):
        host = f"http://{host}"
    return host.rstrip("/")


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
            models.append(_enrich_ollama_model(model_id, item))
        return tuple(models) if models else None


def _enrich_ollama_model(model_id: str, item: dict) -> DiscoveredModel:
    """Build a :class:`DiscoveredModel` enriched from Ollama tag metadata.

    Ollama's ``/api/tags`` response carries ``details`` (with
    ``parameter_size``, ``quantization_level``, ``family``) and ``size``
    (bytes on disk). We infer cost/latency tier from parameter size:
    tiny (<3B) = budget/low, small (3-13B) = standard/medium,
    large (>13B) = premium/high. Cloud-suffixed models (``:cloud``) are
    hosted by Ollama and billed as standard tier regardless of size.
    """

    details = item.get("details") if isinstance(item.get("details"), dict) else {}
    parameter_size = str(details.get("parameter_size") or "").strip()
    family = str(details.get("family") or "").strip()

    capabilities: list[str] = ["text", "tools", "structured_output"]
    if family in {"clip", "llava"} or "vision" in model_id.lower():
        capabilities.append("vision")
    # Ollama tags carry no capability metadata beyond family/size, so the
    # ``code`` capability is derived from the model-id heuristic — same as
    # ``reasoning`` (which ``_model`` appends). This lets the router rule
    # ``'"code" in route.required_capabilities'`` and ``_pick_code_model``
    # select a coding model (e.g. glm-5.2:cloud) instead of silently
    # defaulting. ``known_models.yaml`` can still override per model.
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

    enrichment = apply_known_override(
        "ollama",
        model_id,
        {
            "capabilities": tuple(capabilities),
            "cost_tier": cost_tier,
            "latency_tier": latency_tier,
            "description": description,
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
        context_window=context_window,
        description=str(enrichment.get("description", description)),
        structured_reliability=float(
            enrichment.get("structured_reliability", structured_reliability)
        ),
    )

"""Ollama probe — local model tags.

Retirement-aware: Ollama's ``/api/tags`` keeps listing cloud-hosted
models after the remote host retires them (verified live: a retired tag
answers ``/api/show`` and ``/api/chat`` with HTTP 410 Gone). Every tag
is therefore checked against ``/api/show`` — a 410 drops the model from
discovery (recorded as a UI note) instead of letting a dead model win
the default pin.
"""

from __future__ import annotations

from typing import Any

import httpx

from ..classify import apply_known_override, env_any, is_code_model
from ..types import DiscoveredModel, ProbeResult, ProbeStatus
from .base import ProviderProbe, _extract_rows, _model

_LISA_OLLAMA_HOST_ENV = "LISA_OLLAMA_HOST"

#: Parameter-count thresholds (in billions) for cost/latency/reliability
#: tiers derived from Ollama tag sizes ("7B", "13B", "70B", ...).
_BUDGET_MAX_PARAMS_B = 3
_STANDARD_MAX_PARAMS_B = 13
_PREMIUM_MAX_PARAMS_B = 35

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
) -> tuple[dict | None, bool]:
    """Best-effort per-model ``/api/show`` enrichment.

    Returns ``(show_payload, gone)``. ``show_payload`` is ``None`` on
    transient failure (the caller degrades to tag-based enrichment).
    ``gone`` is True only when Ollama explicitly reports the model
    retired (HTTP 410 Gone) — retired cloud tags stay in ``/api/tags``
    but every generation call fails, so the caller must drop them.

    A failure for a single model must never fail discovery — we degrade to
    the tag-based heuristic when ``/api/show`` is unavailable or errors.
    """
    try:
        response = await client.post(
            f"{base}/api/show",
            json={"model": model_id},
            timeout=timeout,
        )
        if response.status_code == httpx.codes.GONE:
            return None, True
        response.raise_for_status()
        payload = response.json()
        return (payload if isinstance(payload, dict) else None), False
    except Exception:  # pylint: disable=broad-except
        return None, False


class OllamaProbe(ProviderProbe):
    provider = "ollama"

    async def _probe(
        self,
        client: httpx.AsyncClient,
        timeout: float,
    ) -> tuple[DiscoveredModel, ...] | ProbeResult | None:
        host = env_any(_LISA_OLLAMA_HOST_ENV, "OLLAMA_HOST")
        base = _normalize_host(host)

        response = await client.get(f"{base}/api/tags", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        rows = _extract_rows(payload, "models")
        models: list[DiscoveredModel] = []
        retired: list[str] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("name") or "").strip()
            if not model_id:
                continue
            show, gone = await _fetch_show(client, base, model_id, timeout)
            if gone:
                retired.append(model_id)
                continue
            models.append(_enrich_ollama_model(model_id, item, show_payload=show))
        if retired:
            # Return a fully-formed result (base ``probe`` passes it
            # through) so OK carries the dropped ids as a UI note.
            return ProbeResult(
                provider=self.provider,
                status=ProbeStatus.OK if models else ProbeStatus.EMPTY,
                models=tuple(models),
                dropped_model_ids=tuple(retired),
            )
        return tuple(models) if models else None


def _ollama_show_capabilities(show_payload: dict[str, Any] | None) -> set[str]:
    """Extract the authoritative per-model capability set from /api/show.

    Maps Ollama's ``capabilities`` array onto Lisa capability vocabulary.
    Embedding-only models (no "completion" capability) get ``embedding``.
    """
    show_caps: set[str] = set()
    if not show_payload:
        return show_caps
    raw_caps = show_payload.get("capabilities")
    if isinstance(raw_caps, list):
        for cap in raw_caps:
            lisa_cap = _OLLAMA_CAP_TO_LISA.get(str(cap).strip().lower())
            if lisa_cap:
                show_caps.add(lisa_cap)
    # Ollama marks embedding-only models by exposing no "completion".
    if "embedding" in {str(c).strip().lower() for c in (raw_caps or [])} and not show_caps:
        show_caps.add("embedding")
    return show_caps


def _derive_ollama_capabilities(
    model_id: str,
    family: str,
    show_caps: set[str],
) -> list[str]:
    """Derive the full Lisa capability list for one Ollama model.

    Starts from the authoritative /api/show set when present (keeping
    ``structured_output`` and ensuring ``text``/``embedding``), falls
    back to the default chat surface. Vision is inferred from the family
    (clip/llava) or the model-id. ``code`` is derived from the model-id
    heuristic — Ollama tags carry no capability metadata beyond
    family/size, so this heuristic cannot be replaced. This lets the
    router rule ``'"code" in route.required_capabilities'`` and
    ``_pick_code_model`` select a coding model (e.g. glm-5.2:cloud)
    instead of silently defaulting. ``known_models.yaml`` can still
    override per model.
    """
    capabilities: list[str] = ["text", "tools", "structured_output"]
    if show_caps:
        # Merge: start from the authoritative set, keep structured_output.
        capabilities = sorted(show_caps | {"structured_output"})
        if "text" not in capabilities and "embedding" not in capabilities:
            capabilities.append("text")
    vision_family = family in {"clip", "llava"} or "vision" in model_id.lower()
    if vision_family and "vision" not in capabilities:
        capabilities.append("vision")
    if is_code_model(model_id):
        capabilities.append("code")
    return capabilities


def _derive_ollama_tiers(
    model_id: str,
    parameter_size: str,
) -> tuple[str, str, float]:
    """Derive ``(cost_tier, latency_tier, structured_reliability)`` from size.

    Ollama's tag ``details.parameter_size`` (e.g. "7B") drives the tier:
    tiny (<3B) = budget/low, small (3-13B) = standard/medium, large
    (>13B) = premium/high, frontier beyond. Cloud-suffixed (``:cloud``)
    models are hosted by Ollama's infrastructure and billed as standard
    tier with higher structured-output reliability regardless of size.

    Local models treat ``response_format`` as a *soft hint*: small models
    routinely ignore the declared schema and emit prose, so the
    structured-output confidence starts at a conservative floor and rises
    with parameter count. ``known_models.yaml`` can override per model
    where measured.
    """
    if model_id.lower().endswith(":cloud"):
        # Cloud-hosted models — billing is opaque, assume standard.
        return "standard", "medium", 0.8
    # Parse "7B", "13B", "70B" etc.; unparseable or absent sizes stay untiered.
    size_str = parameter_size.lower().rstrip("b").strip()
    try:
        size_b = float(size_str)
    except ValueError:
        size_b = 0.0
    if size_b <= 0:
        return "", "", 0.0
    if size_b <= _BUDGET_MAX_PARAMS_B:
        return "budget", "low", 0.0
    if size_b <= _STANDARD_MAX_PARAMS_B:
        return "standard", "medium", 0.4
    if size_b <= _PREMIUM_MAX_PARAMS_B:
        return "premium", "high", 0.7
    return "frontier", "high", 0.85


def _ollama_show_context_window(show_payload: dict[str, Any] | None) -> int:
    """Extract the authoritative context length from /api/show model_info.

    Keys are runtime-defined: real local GGUF builds publish
    ``general.context_length``; the Ollama cloud proxy (and some other
    runtimes) publish family-keyed names such as
    ``glm_dsa_moe.context_length``. We accept any ``*.context_length``
    entry, preferring ``general.context_length`` when both exist.
    """
    if not show_payload:
        return 0
    model_info = (
        show_payload.get("model_info") if isinstance(show_payload.get("model_info"), dict) else None
    )
    if model_info:
        candidates: list[tuple[int, int]] = []
        for key, raw_ctx in model_info.items():
            if not isinstance(raw_ctx, (int, float)) or raw_ctx <= 0:
                continue
            if key == "general.context_length":
                candidates.append((0, int(raw_ctx)))
            elif key.endswith(".context_length"):
                candidates.append((1, int(raw_ctx)))
        if candidates:
            candidates.sort(key=lambda item: item[0])
            return candidates[0][1]
    return 0


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
    show_caps = _ollama_show_capabilities(show_payload)
    capabilities = _derive_ollama_capabilities(model_id, family, show_caps)

    cost_tier, latency_tier, structured_reliability = _derive_ollama_tiers(model_id, parameter_size)
    context_window = 0

    display_name = model_id.replace("-", " ").title()
    description = ""
    if family:
        description = f"Ollama {family} model ({parameter_size or 'unknown size'})"

    # Authoritative context length: /api/show model_info wins any
    # ``*.context_length`` (family-keyed on the Ollama cloud proxy); the
    # tag-derived ``details.context_length`` fills in when /api/show is
    # absent or unusable. ``known_models.yaml`` still overrides both.
    context_window = _ollama_show_context_window(show_payload)
    if context_window <= 0:
        raw_caption_ctx = details.get("context_length")
        if isinstance(raw_caption_ctx, (int, float)) and raw_caption_ctx > 0:
            context_window = int(raw_caption_ctx)

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

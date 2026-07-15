"""Manifest construction and atomic serialization.

Heuristic role assignment (``router``/``default``/``deep``) plus
atomic YAML writes. The heuristics are deliberately stable and
order-independent: the default model is the first low-latency match
*in discovery order*, falling back to the first non-reasoning model,
then the first model overall. The deep model is the first reasoning
model, else the first non-default model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .classify import is_low_latency_model, is_reasoning_model
from .types import DiscoveredModel
from .yaml_emit import compact_list, dump_manifest

_HEADER = "# generated-by: lisa /models discover\n"


def _dedupe(models: tuple[DiscoveredModel, ...]) -> tuple[DiscoveredModel, ...]:
    """Deduplicate by ``model_id`` (case-insensitive) preserving first-seen order.

    The manifest loader rejects proposals where the same ``model_id`` appears
    more than once — even across different providers. The dedup here must
    match that constraint so the generated proposal always loads cleanly.
    """

    seen: dict[str, DiscoveredModel] = {}
    for model in models:
        key = model.model_id.strip().lower()
        if key:
            seen.setdefault(key, model)
    return tuple(seen.values())


def _choose_default(models: tuple[DiscoveredModel, ...]) -> DiscoveredModel:
    """Pick the default model.

    Preference order:
      1. First low-latency model in discovery order.
      2. First non-reasoning model.
      3. First model overall.
    """

    for model in models:
        if is_low_latency_model(model.model_id):
            return model
    for model in models:
        if not model.reasoning_model:
            return model
    return models[0]


def _choose_deep(models: tuple[DiscoveredModel, ...], default: DiscoveredModel) -> DiscoveredModel:
    """Pick the deep model: first reasoning model, else first non-default."""

    for model in models:
        if model.reasoning_model:
            return model
    for model in models:
        if model is not default:
            return model
    return default


def build_manifest_payload(models: tuple[DiscoveredModel, ...]) -> dict[str, Any]:
    """Build a minimal manifest payload from discovered models.

    Non-obvious role assignments are intentionally left for user review by
    writing to ``models.proposed.yaml`` before ``/models apply``.

    Enrichment fields (``display_name``, ``capabilities``, ``cost_tier``,
    ``latency_tier``, ``thinking_model``, ``context_window``,
    ``description``) are written when the probe populated them. Fields
    that remain empty/default are omitted to keep the YAML clean.
    """

    if not models:
        return {"version": 1, "models": []}

    deduped = _dedupe(models)
    default_model = _choose_default(deduped)
    deep_model = _choose_deep(deduped, default_model)

    def key(model: DiscoveredModel) -> tuple[str, str]:
        return (model.provider.strip().lower(), model.model_id.strip())

    default_key = key(default_model)
    deep_key = key(deep_model)

    rows: list[dict[str, Any]] = []
    for model in deduped:
        designations: list[str] = []
        if key(model) == default_key:
            designations.extend(["router", "default"])
        if key(model) == deep_key:
            designations.extend(["deep", "reasoning"])

        entry: dict[str, Any] = {
            "model_id": model.model_id,
            "provider": model.provider,
        }
        if model.reasoning_model:
            entry["reasoning_model"] = True
        if model.thinking_model:
            entry["thinking_model"] = True
        if model.display_name:
            entry["display_name"] = model.display_name
        if model.capabilities:
            entry["capabilities"] = compact_list(list(model.capabilities))
        if model.cost_tier:
            entry["cost_tier"] = model.cost_tier
        if model.latency_tier:
            entry["latency_tier"] = model.latency_tier
        if model.context_window:
            entry["context_window"] = model.context_window
        if model.description:
            entry["description"] = model.description
        if designations:
            entry["designations"] = sorted(set(designations))
        rows.append(entry)

    mode_models = {
        "default": default_model.model_id,
        "discuss": default_model.model_id,
        "learn": default_model.model_id,
        "search": default_model.model_id,
        "deep": deep_model.model_id,
        "workshop": deep_model.model_id,
    }

    return {
        "version": 1,
        "mode_models": mode_models,
        "models": rows,
    }


def render_manifest_yaml(payload: dict[str, Any]) -> str:
    return _HEADER + dump_manifest(payload)


def write_manifest_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write ``payload`` to ``path`` as YAML.

    Uses a temp file + ``os.replace`` so a crash mid-write never leaves a
    corrupt ``models.proposed.yaml`` for ``/models apply`` to choke on.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(render_manifest_yaml(payload), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        # Ensure no stale .tmp lingers on failure.
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
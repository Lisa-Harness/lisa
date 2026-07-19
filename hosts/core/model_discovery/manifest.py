"""Manifest construction and atomic serialization.

Discrete default/role selection plus atomic YAML writes. The algorithm is
deliberately stable and order-independent:

- **default model**: when any local (ollama) models are present, the default
  is the local *mid-range* model — the median by ``context_window`` — because
  local is the cheapest path for the bulk of turns. When only cloud models
  are present, the default is the cheapest gpt-5-family model
  (nano < mini < full, then lowest variant), falling back to the first
  cloud model if no gpt-5+ was probed.
- **deep role**: first reasoning model (any provider), else the chosen
  default. ``deep`` is the cross-cutting escalation role, so
  reasoning-capability is the right signal regardless of provider.

The output emits ``mode_models`` (pinning ``default``) and ``roles``
(pinning ``default`` and ``deep``) at the top level. Per-model
``designations`` tags are gone — the manifest schema resolves modes and
roles through the two top-level maps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .classify import is_cloud_provider, is_gpt_5_family, is_local_provider, is_reasoning_model
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


def _choose_default_model(models: tuple[DiscoveredModel, ...]) -> DiscoveredModel:
    """Pick the default model with a discrete algorithm.

    Precedence:
    1. If any local (ollama) models exist, the default is the local
       *mid-range* model — the median by ``context_window``. Local wins
       over cloud when both are present because it's the cheapest path
       for the bulk of turns.
    2. Otherwise (cloud only): the cheapest gpt-5-family model
       (nano < mini < full, then lowest variant), falling back to the
       first cloud model if no gpt-5+ was probed.
    3. Otherwise: the first model overall.
    """

    local = tuple(m for m in models if is_local_provider(m.provider))
    if local:
        return _mid_range_model(local)

    cloud = tuple(m for m in models if is_cloud_provider(m.provider))
    if cloud:
        gpt5 = tuple(m for m in cloud if is_gpt_5_family(m.model_id))
        if gpt5:
            # Cheapest gpt-5: size tier first (nano < mini < full), then
            # lowest variant, then discovery order via stable min().
            return min(gpt5, key=_gpt5_cost_rank)
        return cloud[0]

    # No models at all — caller guards this, but be defensive.
    return models[0]


def _mid_range_model(models: tuple[DiscoveredModel, ...]) -> DiscoveredModel:
    """Return the median model by ``context_window``.

    Ties: prefer reasoning over non-reasoning (reasoning is the more
    capable end of a tie), then stable discovery order. The median is the
    mid-range pick — cheaper than the largest, more capable than the
    smallest. With an even count, the lower median wins (the cheaper of
    the two middle entries).
    """
    ranked = sorted(
        enumerate(models),
        key=lambda im: (im[1].context_window, not im[1].reasoning_model, im[0]),
    )
    mid = len(ranked) // 2
    if len(ranked) % 2 == 0 and mid > 0:
        mid -= 1
    return ranked[mid][1]


def _gpt5_cost_rank(model: DiscoveredModel) -> tuple[int, float]:
    """Return a sort key for a gpt-5-family model; lower = cheaper.

    The tuple is ``(size_tier, variant)``:
    - ``size_tier``: 0 for nano, 1 for mini, 2 for full. Cheaper sizes win
      first (nano < mini < full), per Adam's "cheapest gpt-5" preference.
    - ``variant``: numeric suffix after ``gpt-5`` (``gpt-5.4`` → 5.4; bare
      ``gpt-5`` → 5.0). Lower variant = cheaper within a size tier.

    ``min()`` is stable, so equal keys preserve discovery order.
    """
    raw = model.model_id.strip().lower()
    variant = 5.0
    if raw.startswith("gpt-5."):
        try:
            suffix = raw[len("gpt-5"):]
            variant = float(f"5{suffix.split('-')[0].split(':')[0]}")
        except ValueError:
            variant = 5.0
    size_tier = 0 if "nano" in raw else (1 if "mini" in raw else 2)
    return (size_tier, variant)


def _choose_deep_model(
    models: tuple[DiscoveredModel, ...],
    default: DiscoveredModel,
) -> DiscoveredModel:
    """Pick the deep model: first reasoning model, else the chosen default."""

    for model in models:
        if model.reasoning_model or is_reasoning_model(model.model_id):
            return model
    return default


def _model_entry(model: DiscoveredModel) -> dict[str, Any]:
    """Build one manifest row for a discovered model, omitting empty fields."""

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
    return entry


def build_manifest_payload(models: tuple[DiscoveredModel, ...]) -> dict[str, Any]:
    """Build a minimal manifest payload from discovered models.

    Emits ``mode_models`` (pinning ``default``) and ``roles`` (pinning
    ``default`` and ``deep``) at the top level. Per-model ``designations``
    tags are not emitted — the manifest schema resolves modes and roles
    through the two top-level maps.

    Enrichment fields (``display_name``, ``capabilities``, ``cost_tier``,
    ``latency_tier``, ``thinking_model``, ``context_window``,
    ``description``) are written when the probe populated them. Fields
    that remain empty/default are omitted to keep the YAML clean.
    """

    if not models:
        return {"version": 1, "models": []}

    deduped = _dedupe(models)
    default_model = _choose_default_model(deduped)
    deep_model = _choose_deep_model(deduped, default_model)
    rows = [_model_entry(model) for model in deduped]

    # Ordering matters for readability: the top-level maps (mode_models,
    # roles) are the user's primary edit surface, so they go first — right
    # after version — and the long models list goes last. yaml.dump
    # preserves dict insertion order.
    return {
        "version": 1,
        "mode_models": {"default": default_model.model_id},
        "roles": {
            "default": default_model.model_id,
            "deep": deep_model.model_id,
        },
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
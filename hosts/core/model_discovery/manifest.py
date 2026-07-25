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
            suffix = raw[len("gpt-5") :]
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


def _resolve_router_ref(
    ref: str,
    *,
    model_ids: set[str],
    mode_models: dict[str, str],
    roles: dict[str, str],
) -> str:
    """Resolve a symbolic ``use``/``fallback`` ref to a concrete model_id.

    The pack's default router config uses *symbolic* references — mode/role
    names like ``"default"`` or ``"deep"`` rather than literal model ids. The
    manifest loader validates that every ``rules[].use`` and ``fallback``
    references a known model *by model_id*, so we must resolve symbolic refs
    to concrete model ids before emitting the ``router:`` block.

    Resolution order mirrors ``ModelManifest.resolve()``:
    1. ``mode_models[ref]`` (mode pins)
    2. ``roles[ref]`` (cross-cutting roles)
    3. Literal model id (already in the discovered set)

    Returns the resolved model_id, or an empty string when the ref cannot be
    resolved (the caller strips the rule / fallback in that case).
    """

    normalized = ref.strip().lower()
    if not normalized:
        return ""
    # mode_models / roles — symbolic refs.
    mapped = mode_models.get(normalized) or roles.get(normalized)
    if mapped:
        return mapped if mapped.lower() in model_ids else ""
    # Literal model id.
    return ref if normalized in model_ids else ""


def _build_router_block(
    models: tuple[DiscoveredModel, ...],
    *,
    mode_models: dict[str, str],
    roles: dict[str, str],
) -> dict[str, Any] | None:
    """Build the ``router:`` block from the pack-declared default config.

    Discovers the first pack-declared router config via
    ``lisa.extensions.model_router_configs`` (parallel to
    ``lisa.extensions.model_roles`` for roles). Resolves symbolic
    ``use``/``fallback`` references (mode/role names) to concrete model ids
    against the discovered set. Strips any rule whose ``use`` cannot be
    resolved and drops ``fallback`` if it cannot be resolved.

    Returns ``None`` when no pack declares a router config or when no rules
    survive resolution (so the emitted YAML never carries an empty router
    block that would fail validation).
    """

    try:
        from kernel.model_router_config_discovery import discover_pack_router_config
    except Exception:  # pylint: disable=broad-except
        return None

    raw_config = discover_pack_router_config()
    if not isinstance(raw_config, dict):
        return None

    backend = str(raw_config.get("backend") or "").strip()
    raw_fallback = str(raw_config.get("fallback") or "").strip()
    raw_rules = raw_config.get("rules")
    if not isinstance(raw_rules, list):
        raw_rules = []

    model_ids = {model.model_id.strip().lower() for model in models}

    resolved_rules: list[dict[str, str]] = []
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, dict):
            continue
        when = str(raw_rule.get("when") or "").strip()
        use = str(raw_rule.get("use") or "").strip()
        if not when or not use:
            continue
        resolved_use = _resolve_router_ref(
            use, model_ids=model_ids, mode_models=mode_models, roles=roles
        )
        if not resolved_use:
            continue  # rule references a model not in the discovered set
        resolved_rules.append({"when": when, "use": resolved_use})

    resolved_fallback = (
        _resolve_router_ref(raw_fallback, model_ids=model_ids, mode_models=mode_models, roles=roles)
        if raw_fallback
        else ""
    )

    if not backend and not resolved_rules and not resolved_fallback:
        return None

    block: dict[str, Any] = {}
    if backend:
        block["backend"] = backend
    if resolved_rules:
        block["rules"] = resolved_rules
    if resolved_fallback:
        block["fallback"] = resolved_fallback
    return block or None


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


def _pick_code_model(
    models: tuple[DiscoveredModel, ...],
    deep_model_id: str,
) -> str:
    """Pick a code-capable model for the coding-focused preset annotation.

    Scans discovered models for one with ``"code"`` in its capabilities.
    Falls back to ``deep_model_id`` when no code-capable model is found.
    """
    for model in models:
        if model.capabilities and "code" in model.capabilities:
            return model.model_id
    return deep_model_id


def _router_preset_annotations(
    mode_models: dict[str, str],
    roles: dict[str, str],
    models: tuple[DiscoveredModel, ...],
) -> str:
    """Build a YAML comment block with alternative router presets.

    Uses resolved concrete model ids from *mode_models* and *roles* so the
    presets are uncomment-and-go. The three presets cover the main
    market-validated routing strategies:

    - **cost-conscious**: cheap default, escalate to deep only on code or
      iteration ≥ 2, downgrade to default at 3+ tool calls.
    - **quality-first**: deep as fallback, downgrade to default only at
      8+ tool calls.
    - **coding-focused**: code model for code turns, deep for synthesis,
      code model as fallback.

    Returns an empty string when the router block would not be emitted
    (no pack-declared config or no rules survive resolution).
    """
    default_id = mode_models.get("default", "")
    deep_id = roles.get("deep", "")
    if not default_id or not deep_id:
        return ""

    code_id = _pick_code_model(models, deep_id)

    lines = [
        "",
        "# ── Alternative router presets ───────────────────────────────────────────",
        "# Uncomment one block below and comment out the active ``router:`` block",
        "# above, then run /models apply.",
        "#",
        "# Preset: cost-conscious",
        "#   Cheap default, escalate to deep only on code or when the turn drags,",
        "#   downgrade to default early on tool-heavy turns.",
        f"# router:",
        f"#   backend: lisa-router",
        f"#   rules:",
        f'#   - {{when: required_capabilities includes "code", use: {code_id}}}',
        f"#   - {{when: iteration >= 2, use: {deep_id}}}",
        f"#   - {{when: tool_calls_so_far >= 3, use: {default_id}}}",
        f"#   fallback: {default_id}",
        "#",
        "# Preset: quality-first",
        "#   Frontier everywhere, downgrade only on very long tool-heavy turns.",
        f"# router:",
        f"#   backend: lisa-router",
        f"#   rules:",
        f'#   - {{when: required_capabilities includes "code", use: {code_id}}}',
        f'#   - {{when: required_capabilities includes "reasoning", use: {deep_id}}}',
        f"#   - {{when: iteration > 1 and last_tool_result is not None, use: {deep_id}}}",
        f"#   - {{when: tool_calls_so_far >= 8, use: {default_id}}}",
        f"#   fallback: {deep_id}",
        "#",
        "# Preset: coding-focused",
        "#   Code model for code turns, deep for synthesis, code model as fallback.",
        f"# router:",
        f"#   backend: lisa-router",
        f"#   rules:",
        f'#   - {{when: required_capabilities includes "code", use: {code_id}}}',
        f"#   - {{when: tool_calls_so_far >= 4 and last_tool_result is not None, use: {code_id}}}",
        f"#   - {{when: iteration > 1 and last_tool_result is not None, use: {deep_id}}}",
        f"#   fallback: {code_id}",
        "",
    ]
    return "\n".join(lines)


def build_manifest_payload(models: tuple[DiscoveredModel, ...]) -> dict[str, Any]:
    """Build a minimal manifest payload from discovered models.

    Emits ``mode_models`` (pinning ``default``) and ``roles`` (pinning
    ``default`` and ``deep``) at the top level. Per-model ``designations``
    tags are not emitted — the manifest schema resolves modes and roles
    through the two top-level maps.

    When a pack declares a default router config via the
    ``lisa.extensions.model_router_configs`` entry-point group (parallel to
    ``lisa.extensions.model_roles`` for roles), the config is resolved,
    symbolic ``use``/``fallback`` refs are mapped to concrete model ids, and
    a ``router:`` block is emitted. Rules referencing models not in the
    discovered set are stripped so the proposal always loads cleanly. When
    no pack declares a router config, no ``router:`` block is emitted.

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

    mode_models = {"default": default_model.model_id}
    roles = {
        "default": default_model.model_id,
        "deep": deep_model.model_id,
    }

    # Ordering matters for readability: the top-level maps (mode_models,
    # roles, router) are the user's primary edit surface, so they go first
    # — right after version — and the long models list goes last. yaml.dump
    # preserves dict insertion order.
    payload: dict[str, Any] = {
        "version": 1,
        "mode_models": mode_models,
        "roles": roles,
    }

    # Seed a ``router:`` block from the first pack-declared router config
    # (lisa.extensions.model_router_configs). Symbolic refs (mode/role
    # names) are resolved to concrete model ids; rules referencing unknown
    # models are stripped. Returns None when no pack declares a config or
    # no rules survive — in which case no router block is emitted.
    router_block = _build_router_block(deduped, mode_models=mode_models, roles=roles)
    if router_block is not None:
        payload["router"] = router_block

    payload["models"] = rows
    return payload


def render_manifest_yaml(payload: dict[str, Any], *, annotations: str | None = None) -> str:
    """Serialize *payload* to YAML, optionally injecting *annotations*.

    When *annotations* is non-empty, it is appended after the ``router:``
    block (or after ``roles:`` when no router block is present). This lets
    ``/models discover`` emit human-readable preset comments that
    ``/models apply`` strips (because ``yaml.safe_load`` discards them).
    """
    rendered = _HEADER + dump_manifest(payload)
    if annotations:
        rendered += annotations
    return rendered


def write_manifest_yaml(
    path: Path,
    payload: dict[str, Any],
    *,
    annotations: str | None = None,
) -> None:
    """Atomically write ``payload`` to ``path`` as YAML.

    Uses a temp file + ``os.replace`` so a crash mid-write never leaves a
    corrupt ``models.proposed.yaml`` for ``/models apply`` to choke on.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(render_manifest_yaml(payload, annotations=annotations), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        # Ensure no stale .tmp lingers on failure.
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise

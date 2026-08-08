"""Manifest construction and atomic serialization.

Discrete default selection plus atomic YAML writes. The algorithm is
deliberately stable and order-independent:

- **default model**: when any local (ollama) models are present, the default
  is the local *mid-range* model — the median by ``context_window`` — because
  local is the cheapest path for the bulk of turns. When only cloud models
  are present, the default is the cheapest gpt-5-family model
  (nano < mini < full, then lowest variant), falling back to the first
  cloud model if no gpt-5+ was probed.

The output emits ``mode_models`` (pinning ``default``) at the top level.
Per-model ``designations`` tags are gone — the manifest schema resolves
modes through the top-level map.
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


def _pick_reasoning_model(
    models: tuple[DiscoveredModel, ...],
    fallback: DiscoveredModel,
) -> DiscoveredModel:
    """Pick a reasoning model for preset annotations, else the fallback."""

    for model in models:
        if model.reasoning_model or is_reasoning_model(model.model_id):
            return model
    return fallback


def _resolve_router_ref(
    ref: str,
    *,
    model_ids: set[str],
    mode_models: dict[str, str],
) -> str:
    """Resolve a symbolic ``use``/``fallback`` ref to a concrete model_id.

    The pack's default router config uses *symbolic* references — mode names
    like ``"default"`` rather than literal model ids. The manifest loader
    validates that every ``rules[].use`` and ``fallback`` references a known
    model *by model_id*, so we must resolve symbolic refs to concrete model
    ids before emitting the ``router:`` block.

    Resolution order mirrors ``ModelManifest.resolve()``:
    1. ``mode_models[ref]`` (mode pins)
    2. Literal model id (already in the discovered set)

    Returns the resolved model_id, or an empty string when the ref cannot be
    resolved (the caller strips the rule / fallback in that case).
    """

    normalized = ref.strip().lower()
    if not normalized:
        return ""
    # mode_models — symbolic ref.
    mapped = mode_models.get(normalized)
    if mapped:
        return mapped if mapped.lower() in model_ids else ""
    # Literal model id.
    return ref if normalized in model_ids else ""


def _resolve_router_rule(
    raw_rule: Any,
    *,
    model_ids: set[str],
    mode_models: dict[str, str],
) -> dict[str, str] | None:
    """Resolve one raw rule dict to a concrete rule dict, or None to drop it.

    Handles all three action types (``use``, ``finish``, ``transition_to``).
    Rules with zero or more than one action are dropped. ``use`` refs are
    resolved to concrete model ids; unresolvable refs drop the rule.
    """

    if not isinstance(raw_rule, dict):
        return None
    when = str(raw_rule.get("when") or "").strip()
    if not when:
        return None
    use = str(raw_rule.get("use") or "").strip()
    finish = str(raw_rule.get("finish") or "").strip()
    transition_to = str(raw_rule.get("transition_to") or "").strip()
    actions = [a for a in (use, finish, transition_to) if a]
    if not actions or len(actions) > 1:
        return None
    if use:
        resolved_use = _resolve_router_ref(use, model_ids=model_ids, mode_models=mode_models)
        if not resolved_use:
            return None
        return {"when": when, "use": resolved_use}
    action_key = "finish" if finish else "transition_to"
    action_val = finish or transition_to
    return {"when": when, action_key: action_val}


def _build_router_block(
    models: tuple[DiscoveredModel, ...],
    *,
    mode_models: dict[str, str],
) -> dict[str, Any] | None:
    """Build the ``router:`` block from the pack-declared default config.

    Discovers the first pack-declared router config via
    ``lisa.extensions.model_router_configs``. Resolves symbolic
    ``use``/``fallback`` references (mode names) to concrete model ids
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
        resolved = _resolve_router_rule(raw_rule, model_ids=model_ids, mode_models=mode_models)
        if resolved is not None:
            resolved_rules.append(resolved)

    resolved_fallback = (
        _resolve_router_ref(raw_fallback, model_ids=model_ids, mode_models=mode_models)
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
    if model.structured_reliability:
        entry["structured_reliability"] = model.structured_reliability
    if model.completion_api and model.completion_api != "chat_completions":
        entry["completion_api"] = model.completion_api
    return entry


def _pick_code_model(
    models: tuple[DiscoveredModel, ...],
    fallback_model_id: str,
) -> str:
    """Pick a code-capable model for the coding-focused preset annotation.

    Scans discovered models for one with ``"code"`` in its capabilities.
    Falls back to ``fallback_model_id`` when no code-capable model is found.
    """
    for model in models:
        if model.capabilities and "code" in model.capabilities:
            return model.model_id
    return fallback_model_id


# Per-predicate human-readable explanations, keyed by a substring of the
# ``when`` CEL expression. The first matching key wins; unknown predicates
# get a generic "the condition is true" fallback.
_PREDICATE_EXPLANATIONS: tuple[tuple[str, str], ...] = (
    ('"code" in route.required_capabilities', "the mode's tool surface needs code"),
    ('"reasoning" in route.required_capabilities', "the mode declares a reasoning requirement"),
    ('"vision" in route.required_capabilities', "the turn needs vision/multimodal input"),
    ("turn.tool_calls_so_far >= 5", "the turn has made 5+ tool calls (budget pressure)"),
    ("turn.tool_calls_so_far >= 8", "the turn is very long (8+ tool calls)"),
    ("turn.tool_calls_so_far >= 3", "the turn has made 3+ tool calls"),
    ("turn.tool_calls_so_far >= 4", "the turn has made 4+ tool calls"),
    (
        "turn.iteration > 1 && turn.has_tool_result",
        "a tool just returned a result (synthesis phase)",
    ),
    ("turn.iteration >= 2", "the turn is on its second iteration or later"),
    ("turn.iteration > 1", "the turn is past the first iteration"),
    ("turn.iteration >= 10", "the turn has run for 10+ iterations"),
    ('model.cost_tier == "standard"', "the effective model is cheap (standard tier)"),
    ("model.reasoning_model", "the effective model is a reasoning model"),
)


def _explain_predicate(when: str) -> str:
    """Return a short human-readable explanation for one ``when`` clause."""
    for needle, explanation in _PREDICATE_EXPLANATIONS:
        if needle in when:
            return explanation
    return "the condition is true"


def _rule_yaml_line(rule: dict[str, str]) -> str:
    """Render one rule as a flow-style YAML ``{when: ..., <action>: ...}`` line.

    CEL ``when`` expressions contain double-quotes (e.g.
    ``"code" in route.required_capabilities``), so the whole ``when`` value
    is single-quoted for YAML safety. ``use`` values containing ``:``
    (e.g. ``glm-5.2:cloud``) are single-quoted so PyYAML parses them as
    strings, not mappings. ``finish`` and ``transition_to`` actions are
    rendered with their own key.
    """
    when = rule["when"]
    # Always single-quote the CEL expression — it routinely contains
    # double-quotes (e.g. '"code" in route.required_capabilities') which
    # YAML flow mappings interpret as string delimiters, breaking parsing.
    # Single-quoting is safe because CEL expressions never contain
    # single-quotes.
    when_repr = f"'{when}'"
    use = rule.get("use", "")
    finish = rule.get("finish", "")
    transition_to = rule.get("transition_to", "")
    if use:
        use_repr = use if ":" not in use else f"'{use}'"
        return f"{{when: {when_repr}, use: {use_repr}}}"
    if finish:
        return f"{{when: {when_repr}, finish: {finish}}}"
    return f"{{when: {when_repr}, transition_to: {transition_to}}}"


def _annotated_rule_lines(
    rules: list[dict[str, str]],
    *,
    indent: str = "# ",
    comment_each: bool = True,
) -> list[str]:
    """Render ``rules`` as YAML lines, optionally with a per-rule comment.

    Each rule is emitted as a flow-style ``{when: ..., use: ...}`` line. When
    *comment_each* is set, a trailing ``# → <model> when <reason>`` is appended
    so the reader sees *why* each rule exists without flipping to the docs.
    """
    out: list[str] = []
    for rule in rules:
        line = f"{indent}- {_rule_yaml_line(rule)}"
        if comment_each:
            action_target = rule.get("use") or rule.get("finish") or rule.get("transition_to") or ""
            line = f"{line}  # → {action_target} when {_explain_predicate(rule['when'])}"
        out.append(line)
    return out


def _preset_block(
    name: str,
    blurb: str,
    best_for: str,
    rules: list[dict[str, str]],
    fallback: str,
) -> list[str]:
    """Render one preset as a fully-annotated commented YAML block."""
    lines = [
        f"# Preset: {name}",
        f"#   {blurb}",
        f"#   Best for: {best_for}",
        "# router:",
        "#   backend: lisa-router",
        "#   rules:",
    ]
    lines.extend(_annotated_rule_lines(rules, indent="#   ", comment_each=True))
    lines.append(f"#   fallback: {fallback}  # if no rule matches")
    return lines


def _annotated_router_block(router_block: dict[str, Any]) -> str:
    """Render the active ``router:`` block with inline per-rule comments.

    Produces the same YAML the dumper would emit, but with a trailing
    ``# → <model> when <reason>`` on every rule line and on the fallback
    line. This is what gets spliced into the proposal in place of the
    plain dumped block so the reader sees *why* each rule exists right
    where they edit.
    """
    rules = router_block.get("rules", [])
    fallback = router_block.get("fallback", "")
    out: list[str] = ["router:"]
    if router_block.get("backend"):
        out.append(f"  backend: {router_block['backend']}")
    if rules:
        out.append("  rules:")
        for rule in rules:
            out.append(f"  - {_rule_yaml_line(rule)}")
    if fallback:
        out.append(f"  fallback: {fallback}")
    return "\n".join(out)


def _router_preset_annotations(
    mode_models: dict[str, str],
    models: tuple[DiscoveredModel, ...],
    *,
    router_block: dict[str, Any] | None = None,
) -> str:
    """Build a YAML comment block with alternative router presets.

    Emits the three market-validated presets (cost-conscious, quality-first,
    coding-focused) as commented, ready-to-uncomment blocks using resolved
    concrete model ids. Each preset carries a blurb, a "Best for" line, and
    per-rule explanations. The active ``router:`` block itself is annotated
    inline by :func:`_annotated_router_block` (spliced in by
    :func:`render_manifest_yaml`), so this function only emits the
    *alternatives*.

    Returns an empty string when the router block would not be emitted
    (no pack-declared config or no rules survive resolution).
    """
    default_id = mode_models.get("default", "")
    if not default_id:
        return ""

    default_model = next((m for m in models if m.model_id == default_id), models[0])
    reasoning_model = _pick_reasoning_model(models, default_model)
    reasoning_id = reasoning_model.model_id
    code_id = _pick_code_model(models, default_id)

    lines: list[str] = [
        "",
        "# ── Alternative router presets ───────────────────────────────────────────",
        "# Uncomment one block below and comment out the active ``router:`` block",
        "# above, then run /models apply. Each preset is ready to use — model ids",
        "# are already resolved to your discovered set.",
        "#",
    ]

    # Cost-conscious preset.
    lines.extend(
        _preset_block(
            "cost-conscious",
            "Cheap default, escalate to reasoning model only on code or when the turn drags, downgrade to default early on tool-heavy turns.",
            "tight token budgets, dev/test, high-volume simple turns.",
            [
                {"when": '"code" in route.required_capabilities', "use": code_id},
                {"when": "turn.iteration >= 2", "use": reasoning_id},
                {"when": "turn.tool_calls_so_far >= 3", "use": default_id},
            ],
            default_id,
        )
    )
    lines.append("#")

    # Quality-first preset.
    lines.extend(
        _preset_block(
            "quality-first",
            "Frontier everywhere, downgrade only on very long tool-heavy turns.",
            "production quality, complex multi-step tasks, cost-insensitive.",
            [
                {"when": '"code" in route.required_capabilities', "use": code_id},
                {"when": '"reasoning" in route.required_capabilities', "use": reasoning_id},
                {"when": "turn.iteration > 1 && turn.has_tool_result", "use": reasoning_id},
                {"when": "turn.tool_calls_so_far >= 8", "use": default_id},
            ],
            reasoning_id,
        )
    )
    lines.append("#")

    # Coding-focused preset.
    lines.extend(
        _preset_block(
            "coding-focused",
            "Code model for code turns, reasoning model for synthesis, code model as fallback.",
            "software engineering work, agentic coding, repo Q&A.",
            [
                {"when": '"code" in route.required_capabilities', "use": code_id},
                {"when": "turn.tool_calls_so_far >= 4 && turn.has_tool_result", "use": code_id},
                {"when": "turn.iteration > 1 && turn.has_tool_result", "use": reasoning_id},
            ],
            code_id,
        )
    )
    lines.append("#")

    # Terminal-actions preset — demonstrates finish and transition_to.
    lines.extend(
        _preset_block(
            "terminal-actions",
            "Model selection plus finish/transition_to to stop runaway turns.",
            "long agentic turns, workflow modes with a summary stage.",
            [
                {"when": '"code" in route.required_capabilities', "use": code_id},
                {"when": "turn.iteration > 1 && turn.has_tool_result", "use": reasoning_id},
                {"when": "turn.tool_calls_so_far >= 8", "finish": "tool_budget_exhausted"},
                {"when": "turn.iteration >= 10", "transition_to": "summary"},
            ],
            default_id,
        )
    )
    lines.append("")
    return "\n".join(lines)


def build_manifest_payload(models: tuple[DiscoveredModel, ...]) -> dict[str, Any]:
    """Build a minimal manifest payload from discovered models.

    Emits ``mode_models`` (pinning ``default``) at the top level. Per-model
    ``designations`` tags are not emitted — the manifest schema resolves
    modes through the top-level map.

    When a pack declares a default router config via the
    ``lisa.extensions.model_router_configs`` entry-point group, the config
    is resolved, symbolic ``use``/``fallback`` refs are mapped to concrete
    model ids, and a ``router:`` block is emitted. Rules referencing models
    not in the discovered set are stripped so the proposal always loads
    cleanly. When no pack declares a router config, no ``router:`` block
    is emitted.

    Enrichment fields (``display_name``, ``capabilities``, ``cost_tier``,
    ``latency_tier``, ``thinking_model``, ``context_window``,
    ``description``) are written when the probe populated them. Fields
    that remain empty/default are omitted to keep the YAML clean.
    """

    if not models:
        return {"version": 1, "models": []}

    deduped = _dedupe(models)
    default_model = _choose_default_model(deduped)
    rows = [_model_entry(model) for model in deduped]

    mode_models = {"default": default_model.model_id}

    # Ordering matters for readability: the top-level maps (mode_models,
    # router) are the user's primary edit surface, so they go first
    # — right after version — and the long models list goes last. yaml.dump
    # preserves dict insertion order.
    payload: dict[str, Any] = {
        "version": 1,
        "mode_models": mode_models,
    }

    # Seed a ``router:`` block from the first pack-declared router config
    # (lisa.extensions.model_router_configs). Symbolic refs (mode names)
    # are resolved to concrete model ids; rules referencing unknown models
    # are stripped. Returns None when no pack declares a config or no rules
    # survive — in which case no router block is emitted.
    router_block = _build_router_block(deduped, mode_models=mode_models)
    if router_block is not None:
        payload["router"] = router_block

    payload["models"] = rows
    return payload


def render_manifest_yaml(
    payload: dict[str, Any],
    *,
    annotations: str | None = None,
    router_block: dict[str, Any] | None = None,
) -> str:
    """Serialize *payload* to YAML with the router block annotated inline.

    The active ``router:`` block is rendered by hand (not by the dumper)
    with a trailing ``# → <model> when <reason>`` on every rule and on the
    fallback line, then spliced in right after ``mode_models:``. The alternative
    preset comments (*annotations*) follow immediately after the active
    block, before the ``models:`` list. Everything sits at the top of the
    file where the user edits, not 400 lines down.

    ``/models apply`` strips all of this (``yaml.safe_load`` discards
    comments and re-emits a clean block).
    """
    # When we have an annotated router block to splice in, drop the plain
    # ``router`` key from the dump so we don't emit two copies. When
    # ``router_block`` is None (e.g. the /models apply re-emit path), keep
    # the payload's ``router`` block as-is so the active manifest retains it.
    if router_block:
        dump_payload = {k: v for k, v in payload.items() if k != "router"}
    else:
        dump_payload = payload
    rendered = _HEADER + dump_manifest(dump_payload)
    if not router_block and not annotations:
        return rendered

    lines = rendered.split("\n")
    insert_block: list[str] = []
    if router_block:
        insert_block.append(_annotated_router_block(router_block))
    if annotations:
        insert_block.append(annotations.rstrip("\n"))

    # Splice right before the ``models:`` top-level key (always the last
    # top-level key). Match the whole line so we don't hit the ``models:``
    # substring inside ``mode_models:``.
    splice = "\n".join(insert_block)
    for idx, line in enumerate(lines):
        if line.rstrip() == "models:":
            lines.insert(idx, splice)
            break
    else:
        lines.append(splice)
    return "\n".join(lines)


def write_manifest_yaml(
    path: Path,
    payload: dict[str, Any],
    *,
    annotations: str | None = None,
    router_block: dict[str, Any] | None = None,
) -> None:
    """Atomically write ``payload`` to ``path`` as YAML.

    Uses a temp file + ``os.replace`` so a crash mid-write never leaves a
    corrupt ``models.proposed.yaml`` for ``/models apply`` to choke on.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(
            render_manifest_yaml(payload, annotations=annotations, router_block=router_block),
            encoding="utf-8",
        )
        tmp.replace(path)
    except Exception:
        # Ensure no stale .tmp lingers on failure.
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise

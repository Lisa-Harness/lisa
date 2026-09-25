from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_ALLOWED_PROVIDERS = frozenset(
    {"aws", "azure_openai", "openai", "anthropic", "google", "ollama", "openrouter"}
)


@dataclass(frozen=True, slots=True)
class ManifestModelEntry:
    """One normalized model record loaded from a shared model manifest."""

    key: str
    model_id: str
    provider: str
    display_name: str
    completion_api: str = "chat_completions"
    reasoning_model: bool = False
    thinking_model: bool = False
    aliases: frozenset[str] = frozenset()
    display_alias: str = ""
    capabilities: tuple[str, ...] = ()
    cost_tier: str = "standard"
    latency_tier: str = "medium"
    context_window: int = 0
    #: Per-token USD price for prompt (input) tokens, when the provider
    #: publishes pricing. 0.0 = unknown/not priced. OpenRouter exposes this
    #: via ``pricing.prompt``; other providers may leave it 0.0. Display
    #: layers multiply by 1e6 for a $/M-token view — the field itself stays
    #: per-token so pricing_units remains implied and forward-compatible.
    prompt_price: float = 0.0
    #: Per-token USD price for completion (output) tokens, when known.
    completion_price: float = 0.0
    #: Currency for the above prices (ISO 4217). Defaults to USD, the only
    #: currency OpenRouter prices in today; kept for forward compatibility.
    pricing_currency: str = "USD"
    operator_only: bool = False
    deployment_only: bool = False
    api_key_env: str = ""
    base_url: str = ""
    base_url_env: str = ""
    model_id_from_config: str = ""
    azure_deploy: dict[str, Any] = field(default_factory=dict)
    #: Confidence (0.0-1.0) that this model reliably honors structured-output
    #: ``response_format`` constraints. 0.0 = unknown/untrusted (e.g. small
    #: local Ollama models), 1.0 = provider-enforced and known to comply.
    #: Routers may read this (via ``model.structured_reliability``) to route
    #: JSON-bound work to a model that returns the declared schema.
    structured_reliability: float = 0.0
    #: User-declared flag that this model needs stronger kernel guidance
    #: backstops. Off by default so strong models stay unhinted; set
    #: ``needs_guidance: true`` per model in the manifest to opt a weak or
    #: finicky model into the backstops:
    #:
    #: - **Do-the-work backstop (self-handoff)**: when the routing hint
    #:   and the plan-driven hint both correctly go silent (the plan's
    #:   next step is the current mode), the kernel arms a mode action
    #:   tool and the empty-completion enforcer synthesizes a call to it
    #:   so the turn does real work instead of ending blocked as
    #:   ``no_tools`` (hallucinated success). One-shot: cleared as soon
    #:   as any tool dispatches.
    #: - **Trust-aware enforcer cap**: the forced-action enforcer's
    #:   ``non_compliant_cap`` is bypassed for frontier, reasoning, or
    #:   high-reliability models. A ``needs_guidance`` model is NOT
    #:   bypassed — it receives the full cap treatment so the kernel
    #:   takes over when it refuses to call the expected tool.
    #:
    #: Exposed to router rules via ``model.needs_guidance``. See
    #: ``sdk/router/README.md`` § Model attributes for the full attribute
    #: reference.
    needs_guidance: bool = False


@dataclass(frozen=True, slots=True)
class RouterRule:
    """One conditional routing rule from the ``router.rules`` manifest block.

    ``when`` is a free-form condition string interpreted by the active router
    backend. For the built-in ``lisa-router`` backend it is a CEL expression
    (see ``sdk/router/router.py`` and ``../sdk/router/README.md``).

    Exactly one action field is set:

    - ``use`` — route to this model_id for the matching iteration.
    - ``finish`` — stop the tool loop after one final no-tools completion.
      The value is a short reason slug for telemetry/audit.
    - ``transition_to`` — move the active workflow to this named terminal
      stage, then produce its closing summary.

    A rule with zero or more than one action set is rejected at load time.
    """

    when: str
    use: str = ""
    finish: str = ""
    transition_to: str = ""

    def action_kind(self) -> str:
        """Return ``"use"``, ``"finish"``, or ``"transition_to"`` for this rule."""
        if self.use:
            return "use"
        if self.finish:
            return "finish"
        return "transition_to"


@dataclass(frozen=True, slots=True)
class RouterConfig:
    """Optional model-router configuration parsed from the manifest's ``router:`` block.

    ``backend`` names the active router backend (e.g. ``"lisa-router"``).
    ``rules`` is an ordered list of ``RouterRule`` entries evaluated
    first-match-wins. There is no ``fallback`` key: when no rule matches the
    router returns ``None`` and the statically-resolved ``mode_models`` pin
    (which is required at load time) is the default. A catch-all
    ``when: 'true'`` rule expresses the old fallback explicitly.
    When ``backend`` is empty, no router is active — the manifest's
    ``mode_models`` static resolution is the only path.
    """

    backend: str = ""
    rules: tuple[RouterRule, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """Validated shared model manifest used by hosts and provider wrappers."""

    models: tuple[ManifestModelEntry, ...]
    mode_models: dict[str, str] = field(default_factory=dict)
    router: RouterConfig | None = None

    def resolve(self, ref: str) -> ManifestModelEntry | None:
        """Resolve one reference to a model entry, or None.

        Precedence: ``mode_models[ref]`` → ``find_model(ref)``.
        Mode pins win; a concrete model id is the last resort. There is no
        silent first-model fallback — if the caller asks for a mode that
        isn't mapped, that's None and the caller decides whether to raise.
        """
        normalized = ref.strip().lower()
        if not normalized:
            return None
        model_ref = self.mode_models.get(normalized)
        if model_ref:
            return self.find_model(model_ref)
        return self.find_model(ref)

    def find_model(self, reference: str) -> ManifestModelEntry | None:
        """Resolve one model by key, model_id, display_alias, or alias."""

        normalized_reference = reference.strip().lower()
        for model in self.models:
            aliases = {alias.lower() for alias in model.aliases}
            if normalized_reference in {
                model.key.lower(),
                model.model_id.lower(),
                model.display_alias.lower(),
                *aliases,
            }:
                return model
        return None


def load_model_manifest(
    path: Path,
    *,
    require_mode_models: bool = False,
    allowed_providers: set[str] | frozenset[str] | None = None,
) -> ModelManifest:
    """Load and validate one shared model manifest from YAML or JSON."""

    if not path.exists():
        raise ValueError(f"Model manifest was not found: {path}")

    raw_payload = _load_payload(path)
    return _load_model_manifest_payload(
        raw_payload,
        require_mode_models=require_mode_models,
        allowed_providers=allowed_providers,
    )


def load_model_manifest_from_text(
    text: str,
    *,
    require_mode_models: bool = False,
    allowed_providers: set[str] | frozenset[str] | None = None,
    source_format: str = "yaml",
) -> ModelManifest:
    """Load and validate one shared model manifest from an in-memory string.

    This mirrors :func:`load_model_manifest` but parses from text instead of a
    file path so callers can validate storage-gateway reads without direct disk
    access.
    """

    raw_text = text.strip()
    if not raw_text:
        raise ValueError("Model manifest text is empty.")

    normalized_format = source_format.strip().lower()
    try:
        if normalized_format == "json":
            raw_payload = json.loads(raw_text)
        elif normalized_format in {"", "yaml", "yml", "auto"}:
            raw_payload = yaml.safe_load(raw_text)
        else:
            raise ValueError(
                f"Model manifest source_format must be 'yaml', 'yml', 'json', or "
                f"'auto'. Got: {source_format}"
            )
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(
            f"Failed to parse model manifest text as {normalized_format or 'yaml'}: {exc}"
        ) from exc

    if not isinstance(raw_payload, dict):
        raise ValueError("Model manifest root must be an object.")

    return _load_model_manifest_payload(
        raw_payload,
        require_mode_models=require_mode_models,
        allowed_providers=allowed_providers,
    )


def _normalize_mode_models(raw_mode_models: Any, *, require_mode_models: bool) -> dict[str, str]:
    """Validate and normalize the optional ``mode_models`` mapping.

    Keys are normalized to lowercase mode ids; values keep their case.
    """
    if require_mode_models and not isinstance(raw_mode_models, dict):
        raise ValueError("Model manifest must include a 'mode_models' mapping.")

    mode_models: dict[str, str] = {}
    if raw_mode_models is None:
        return mode_models
    if not isinstance(raw_mode_models, dict):
        raise ValueError("Model manifest 'mode_models' must be a mapping when provided.")
    for mode_id, model_ref in raw_mode_models.items():
        normalized_mode_id = str(mode_id or "").strip().lower()
        normalized_model_ref = str(model_ref or "").strip()
        if not normalized_mode_id or not normalized_model_ref:
            raise ValueError(f"mode_models entry '{mode_id}' must have a non-empty key and value.")
        mode_models[normalized_mode_id] = normalized_model_ref
    return mode_models


def _parse_manifest_models(
    raw_models: Any,
    *,
    allowed_providers: set[str],
) -> list[ManifestModelEntry]:
    """Parse and duplicate-check the manifest's ``models`` list."""
    if not isinstance(raw_models, list):
        raise ValueError("Model manifest must include a 'models' list.")

    models: list[ManifestModelEntry] = []
    seen_model_refs: set[str] = set()
    for index, raw_model in enumerate(raw_models, start=1):
        model = _parse_model_entry(
            raw_model,
            index=index,
            allowed_providers=allowed_providers,
        )
        for reference in {model.key.lower(), model.model_id.lower()}:
            if reference in seen_model_refs:
                raise ValueError(f"Manifest model reference '{reference}' is duplicated.")
            seen_model_refs.add(reference)
        models.append(model)

    if not models:
        raise ValueError("Model manifest must include at least one model.")
    return models


def _validate_mode_model_refs(
    mode_models: dict[str, str],
    models: list[ManifestModelEntry],
) -> None:
    """Ensure every ``mode_models`` value references a declared model."""
    for mode_id, model_ref in mode_models.items():
        if all(
            model_ref.lower() not in {model.key.lower(), model.model_id.lower()} for model in models
        ):
            available_models = ", ".join(sorted(model.model_id for model in models))
            raise ValueError(
                f"mode_models entry '{mode_id}' references unknown model '{model_ref}'. "
                f"Available models: {available_models}"
            )


def _load_model_manifest_payload(
    raw_payload: dict[str, Any],
    *,
    require_mode_models: bool = False,
    allowed_providers: set[str] | frozenset[str] | None = None,
) -> ModelManifest:
    """Validate one already-parsed manifest payload into ``ModelManifest``."""

    mode_models = _normalize_mode_models(
        raw_payload.get("mode_models"),
        require_mode_models=require_mode_models,
    )

    normalized_allowed_providers = {
        provider.strip().lower() for provider in (allowed_providers or _DEFAULT_ALLOWED_PROVIDERS)
    }

    models = _parse_manifest_models(
        raw_payload.get("models"),
        allowed_providers=normalized_allowed_providers,
    )
    _validate_mode_model_refs(mode_models, models)

    # Parse the optional ``router:`` block. Absence means no router —
    # the manifest's ``mode_models`` static resolution is the only path.
    router_config = _parse_router_config(raw_payload.get("router"), models=models)

    return ModelManifest(
        models=tuple(models),
        mode_models=mode_models,
        router=router_config,
    )


def _parse_router_rule(
    raw_rule: Any,
    index: int,
) -> RouterRule:
    """Validate one ``router.rules`` entry into a :class:`RouterRule`."""
    if not isinstance(raw_rule, dict):
        raise ValueError(f"router.rules entry #{index} must be an object.")
    condition = str(raw_rule.get("when") or "").strip()
    use = str(raw_rule.get("use") or "").strip()
    finish = str(raw_rule.get("finish") or "").strip()
    transition_to = str(raw_rule.get("transition_to") or "").strip()
    if not condition:
        raise ValueError(f"router.rules entry #{index} must include a non-empty 'when'.")
    actions = [a for a in (use, finish, transition_to) if a]
    if not actions:
        raise ValueError(
            f"router.rules entry #{index} must include one of 'use', 'finish', or 'transition_to'."
        )
    if len(actions) > 1:
        raise ValueError(
            f"router.rules entry #{index} must set exactly one of "
            f"'use', 'finish', or 'transition_to' (got {len(actions)})."
        )
    return RouterRule(when=condition, use=use, finish=finish, transition_to=transition_to)


def _validate_router_rule_model_refs(
    rules: list[RouterRule],
    models: list[ManifestModelEntry],
) -> None:
    """Ensure every ``rules[].use`` references a known model."""
    for rule in rules:
        if rule.use and all(
            rule.use.lower() not in {model.key.lower(), model.model_id.lower()} for model in models
        ):
            available = ", ".join(sorted(model.model_id for model in models))
            raise ValueError(
                f"router.rules entry 'when: {rule.when}' references unknown model '{rule.use}'. "
                f"Available models: {available}"
            )


def _parse_router_config(
    raw_router: Any,
    *,
    models: list[ManifestModelEntry],
) -> RouterConfig | None:
    """Parse and validate the optional ``router:`` block from the manifest."""

    if raw_router is None:
        return None
    if not isinstance(raw_router, dict):
        raise ValueError("Model manifest 'router' must be a mapping when provided.")

    backend = str(raw_router.get("backend") or "").strip()
    # ``fallback`` was removed (hard cutover): the statically-resolved
    # ``mode_models`` pin is the only no-rule-match default, and a catch-all
    # ``when: 'true'`` rule expresses the old fallback explicitly. Existing
    # manifests carrying the key fail loudly here with the migration hint.
    if "fallback" in raw_router:
        raise ValueError(
            "Model manifest 'router.fallback' is no longer supported. "
            "Move it into a trailing catch-all rule (e.g. `when: 'true', use: <model_id>`) "
            "or rely on the required mode_models.default pin; "
            "re-run /models discover + /models apply to regenerate."
        )

    raw_rules = raw_router.get("rules")
    if raw_rules is None:
        raw_rules = []
    if not isinstance(raw_rules, list):
        raise ValueError("Model manifest 'router.rules' must be a list when provided.")

    rules = [_parse_router_rule(raw_rule, index) for index, raw_rule in enumerate(raw_rules, 1)]
    _validate_router_rule_model_refs(rules, models)

    return RouterConfig(backend=backend, rules=tuple(rules))


def _parse_model_entry(
    raw_model: Any,
    *,
    index: int,
    allowed_providers: set[str],
) -> ManifestModelEntry:
    if not isinstance(raw_model, dict):
        raise ValueError(f"Manifest model entry #{index} must be an object.")

    model_id = str(raw_model.get("model_id") or "").strip()
    model_id_from_config = str(raw_model.get("model_id_from_config") or "").strip()
    if not model_id and not model_id_from_config:
        raise ValueError(
            f"Manifest model entry #{index} must include 'model_id' or 'model_id_from_config'."
        )

    provider = str(raw_model.get("provider") or "azure_openai").strip().lower()
    if provider not in allowed_providers:
        supported = ", ".join(sorted(allowed_providers))
        raise ValueError(
            f"Manifest provider must be one of {supported}. "
            f"Got '{provider}' for model '{model_id or model_id_from_config}'."
        )

    key = str(raw_model.get("key") or model_id or model_id_from_config).strip()
    display_name = str(raw_model.get("display_name") or key).strip()
    completion_api = str(raw_model.get("completion_api") or "chat_completions").strip()
    if completion_api not in {"chat_completions", "responses"}:
        raise ValueError(
            f"Manifest model entry for '{key}' has unsupported completion_api '{completion_api}'."
        )
    raw_azure_deploy = raw_model.get("azure_deploy") or {}
    if not isinstance(raw_azure_deploy, dict):
        raise ValueError(f"Manifest model entry for '{key}' azure_deploy must be an object.")

    return ManifestModelEntry(
        key=key,
        model_id=model_id or model_id_from_config,
        provider=provider,
        display_name=display_name,
        completion_api=completion_api,
        reasoning_model=bool(raw_model.get("reasoning_model", False)),
        thinking_model=bool(raw_model.get("thinking_model", False)),
        aliases=frozenset(
            _parse_str_list(raw_model.get("aliases"), field_name="aliases", model_id=key)
        ),
        display_alias=str(raw_model.get("display_alias") or "").strip(),
        capabilities=tuple(
            _parse_str_list(raw_model.get("capabilities"), field_name="capabilities", model_id=key)
        ),
        cost_tier=str(raw_model.get("cost_tier") or "standard").strip() or "standard",
        latency_tier=str(raw_model.get("latency_tier") or "medium").strip() or "medium",
        context_window=_parse_nonnegative_int(
            raw_model.get("context_window"),
            field_name="context_window",
            model_id=key,
        ),
        operator_only=bool(raw_model.get("operator_only", False)),
        deployment_only=bool(raw_model.get("deployment_only", False)),
        api_key_env=str(raw_model.get("api_key_env") or "").strip(),
        base_url=str(raw_model.get("base_url") or "").strip(),
        base_url_env=str(raw_model.get("base_url_env") or "").strip(),
        model_id_from_config=model_id_from_config,
        azure_deploy=dict(raw_azure_deploy),
        structured_reliability=_parse_unit_float(
            raw_model.get("structured_reliability"),
            field_name="structured_reliability",
            model_id=key,
        ),
        prompt_price=_parse_price(
            raw_model.get("prompt_price"),
            field_name="prompt_price",
            model_id=key,
        ),
        completion_price=_parse_price(
            raw_model.get("completion_price"),
            field_name="completion_price",
            model_id=key,
        ),
        pricing_currency=str(raw_model.get("pricing_currency") or "USD").strip() or "USD",
        needs_guidance=bool(raw_model.get("needs_guidance", False)),
    )


def _parse_str_list(raw_value: Any, *, field_name: str, model_id: str) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [str(value).strip() for value in raw_value if str(value).strip()]
    if isinstance(raw_value, str):
        return [value.strip() for value in raw_value.split(",") if value.strip()]
    raise ValueError(
        f"Manifest model entry for '{model_id}' {field_name} must be a list "
        "or comma-separated string."
    )


def _parse_nonnegative_int(raw_value: Any, *, field_name: str, model_id: str) -> int:
    """Parse an optional non-negative integer manifest field."""

    if raw_value is None:
        return 0
    if isinstance(raw_value, bool):
        raise ValueError(
            f"Manifest model entry for '{model_id}' {field_name} must be a non-negative integer."
        )
    if isinstance(raw_value, int):
        value = raw_value
    elif isinstance(raw_value, str) and raw_value.strip().isdigit():
        value = int(raw_value.strip())
    else:
        raise ValueError(
            f"Manifest model entry for '{model_id}' {field_name} must be a non-negative integer."
        )
    if value < 0:
        raise ValueError(
            f"Manifest model entry for '{model_id}' {field_name} must be a non-negative integer."
        )
    return value


def _parse_unit_float(raw_value: Any, *, field_name: str, model_id: str) -> float:
    """Parse an optional 0.0-1.0 confidence field, defaulting to 0.0.

    Accepts a number or a numeric string. Values outside ``[0.0, 1.0]`` are
    rejected so a malformed manifest cannot silently claim a model is more
    reliable than it is.
    """

    if raw_value is None:
        return 0.0
    if isinstance(raw_value, bool):
        raise ValueError(
            f"Manifest model entry for '{model_id}' {field_name} must be a float in [0.0, 1.0]."
        )
    if isinstance(raw_value, (int, float)):
        value = float(raw_value)
    elif isinstance(raw_value, str) and raw_value.strip():
        try:
            value = float(raw_value.strip())
        except ValueError as exc:
            raise ValueError(
                f"Manifest model entry for '{model_id}' {field_name} must be a float in [0.0, 1.0]."
            ) from exc
    else:
        raise ValueError(
            f"Manifest model entry for '{model_id}' {field_name} must be a float in [0.0, 1.0]."
        )
    if value < 0.0 or value > 1.0:
        raise ValueError(
            f"Manifest model entry for '{model_id}' {field_name} must be in the range [0.0, 1.0]."
        )
    return value


def _parse_price(raw_value: Any, *, field_name: str, model_id: str) -> float:
    """Parse an optional non-negative per-token price, defaulting to 0.0.

    Accepts a number or a numeric string (providers publish string prices,
    e.g. OpenRouter's ``pricing.prompt``). 0.0 means unknown/not priced.
    """

    if raw_value is None:
        return 0.0
    if isinstance(raw_value, bool):
        raise ValueError(
            f"Manifest model entry for '{model_id}' {field_name} must be a non-negative float."
        )
    if isinstance(raw_value, (int, float)):
        value = float(raw_value)
    elif isinstance(raw_value, str) and raw_value.strip():
        try:
            value = float(raw_value.strip())
        except ValueError as exc:
            raise ValueError(
                f"Manifest model entry for '{model_id}' {field_name} must be a non-negative float."
            ) from exc
    else:
        raise ValueError(
            f"Manifest model entry for '{model_id}' {field_name} must be a non-negative float."
        )
    if value < 0.0:
        raise ValueError(
            f"Manifest model entry for '{model_id}' {field_name} must be a non-negative float."
        )
    return value


def _load_payload(path: Path) -> dict[str, Any]:
    suffix = path.suffix.strip().lower()
    text = path.read_text(encoding="utf-8")

    if suffix == ".json":
        payload = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    else:
        raise ValueError(f"Model manifest must use .json, .yaml, or .yml. Got: {path.name}")

    if not isinstance(payload, dict):
        raise ValueError("Model manifest root must be an object.")
    return payload

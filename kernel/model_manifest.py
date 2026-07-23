from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_ALLOWED_PROVIDERS = frozenset({"azure_openai", "openai", "anthropic", "google", "ollama"})


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
    operator_only: bool = False
    default_for_roles: tuple[str, ...] = ()
    deployment_only: bool = False
    api_key_env: str = ""
    base_url: str = ""
    base_url_env: str = ""
    model_id_from_config: str = ""
    azure_deploy: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RouterRule:
    """One conditional routing rule from the ``router.rules`` manifest block.

    ``when`` is a free-form condition string interpreted by the active router
    backend (e.g. ``required_capabilities includes "code"``).
    ``use`` is the model_id to route to when the condition matches.
    """

    when: str
    use: str


@dataclass(frozen=True, slots=True)
class RouterConfig:
    """Optional model-router configuration parsed from the manifest's ``router:`` block.

    ``backend`` names the active router backend (e.g. ``"lisa-router"``).
    ``rules`` is an ordered list of ``RouterRule`` entries evaluated
    first-match-wins. ``fallback`` is the model_id used when no rule matches.
    When ``backend`` is empty, no router is active — the manifest's
    ``mode_models``/``roles`` static resolution is the only path.
    """

    backend: str = ""
    rules: tuple[RouterRule, ...] = ()
    fallback: str = ""


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """Validated shared model manifest used by hosts and provider wrappers."""

    models: tuple[ManifestModelEntry, ...]
    mode_models: dict[str, str] = field(default_factory=dict)
    roles: dict[str, str] = field(default_factory=dict)
    router: RouterConfig | None = None

    def resolve(self, ref: str) -> ManifestModelEntry | None:
        """Resolve one reference to a model entry, or None.

        Precedence: ``mode_models[ref]`` → ``roles[ref]`` → ``find_model(ref)``.
        Mode pins win over cross-cutting roles; a concrete model id is the
        last resort. There is no silent first-model fallback — if the
        caller asks for a mode/role that isn't mapped, that's None and the
        caller decides whether to raise.
        """
        normalized = ref.strip().lower()
        if not normalized:
            return None
        model_ref = self.mode_models.get(normalized)
        if model_ref:
            return self.find_model(model_ref)
        model_ref = self.roles.get(normalized)
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
    raw_models = raw_payload.get("models")
    if not isinstance(raw_models, list):
        raise ValueError("Model manifest must include a 'models' list.")

    raw_mode_models = raw_payload.get("mode_models")
    if require_mode_models and not isinstance(raw_mode_models, dict):
        raise ValueError("Model manifest must include a 'mode_models' mapping.")

    mode_models: dict[str, str] = {}
    if raw_mode_models is not None:
        if not isinstance(raw_mode_models, dict):
            raise ValueError("Model manifest 'mode_models' must be a mapping when provided.")
        for mode_id, model_ref in raw_mode_models.items():
            normalized_mode_id = str(mode_id or "").strip().lower()
            normalized_model_ref = str(model_ref or "").strip()
            if not normalized_mode_id or not normalized_model_ref:
                raise ValueError(
                    f"mode_models entry '{mode_id}' must have a non-empty key and value."
                )
            mode_models[normalized_mode_id] = normalized_model_ref

    raw_roles = raw_payload.get("roles")
    roles: dict[str, str] = {}
    if raw_roles is not None:
        if not isinstance(raw_roles, dict):
            raise ValueError("Model manifest 'roles' must be a mapping when provided.")
        for role_name, model_ref in raw_roles.items():
            normalized_role = str(role_name or "").strip().lower()
            normalized_model_ref = str(model_ref or "").strip()
            if not normalized_role or not normalized_model_ref:
                raise ValueError(
                    f"roles entry '{role_name}' must have a non-empty key and value."
                )
            roles[normalized_role] = normalized_model_ref

    # Merge pack-declared role defaults: packs declare role names + a default
    # model id. The manifest YAML overrides pack defaults (user wins); roles
    # declared by packs but absent from the YAML are seeded with the pack's
    # default. Empty pack defaults (role name only) are skipped — the user
    # must pin them explicitly.
    try:
        from kernel.model_roles import discover_pack_roles  # noqa: PLC0415
        pack_roles = discover_pack_roles()
    except Exception:  # pylint: disable=broad-except
        pack_roles = {}
    for role_name, model_ref in pack_roles.items():
        if not model_ref:
            continue
        roles.setdefault(role_name, model_ref)

    normalized_allowed_providers = {
        provider.strip().lower() for provider in (allowed_providers or _DEFAULT_ALLOWED_PROVIDERS)
    }

    models: list[ManifestModelEntry] = []
    seen_model_refs: set[str] = set()
    for index, raw_model in enumerate(raw_models, start=1):
        model = _parse_model_entry(
            raw_model,
            index=index,
            allowed_providers=normalized_allowed_providers,
        )
        for reference in {model.key.lower(), model.model_id.lower()}:
            if reference in seen_model_refs:
                raise ValueError(f"Manifest model reference '{reference}' is duplicated.")
            seen_model_refs.add(reference)
        models.append(model)

    if not models:
        raise ValueError("Model manifest must include at least one model.")

    for mode_id, model_ref in mode_models.items():
        if all(
            model_ref.lower() not in {model.key.lower(), model.model_id.lower()} for model in models
        ):
            available_models = ", ".join(sorted(model.model_id for model in models))
            raise ValueError(
                f"mode_models entry '{mode_id}' references unknown model '{model_ref}'. "
                f"Available models: {available_models}"
            )

    for role_name, model_ref in roles.items():
        if all(
            model_ref.lower() not in {model.key.lower(), model.model_id.lower()} for model in models
        ):
            available_models = ", ".join(sorted(model.model_id for model in models))
            raise ValueError(
                f"roles entry '{role_name}' references unknown model '{model_ref}'. "
                f"Available models: {available_models}"
            )

    # Parse the optional ``router:`` block. Absence means no router —
    # the manifest's ``mode_models``/``roles`` static resolution is the
    # only path, preserving today's behavior.
    router_config = _parse_router_config(raw_payload.get("router"), models=models)

    return ModelManifest(
        models=tuple(models),
        mode_models=mode_models,
        roles=roles,
        router=router_config,
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
    fallback = str(raw_router.get("fallback") or "").strip()

    raw_rules = raw_router.get("rules")
    if raw_rules is None:
        raw_rules = []
    if not isinstance(raw_rules, list):
        raise ValueError("Model manifest 'router.rules' must be a list when provided.")

    rules: list[RouterRule] = []
    for index, raw_rule in enumerate(raw_rules, start=1):
        if not isinstance(raw_rule, dict):
            raise ValueError(f"router.rules entry #{index} must be an object.")
        condition = str(raw_rule.get("when") or "").strip()
        use = str(raw_rule.get("use") or "").strip()
        if not condition:
            raise ValueError(f"router.rules entry #{index} must include a non-empty 'when'.")
        if not use:
            raise ValueError(f"router.rules entry #{index} must include a non-empty 'use'.")
        rules.append(RouterRule(when=condition, use=use))

    # Validate that ``fallback`` and every ``rules[].use`` reference a known model.
    for rule in rules:
        if all(
            rule.use.lower() not in {model.key.lower(), model.model_id.lower()}
            for model in models
        ):
            available = ", ".join(sorted(model.model_id for model in models))
            raise ValueError(
                f"router.rules entry 'when: {rule.when}' references unknown model '{rule.use}'. "
                f"Available models: {available}"
            )
    if fallback:
        if all(
            fallback.lower() not in {model.key.lower(), model.model_id.lower()}
            for model in models
        ):
            available = ", ".join(sorted(model.model_id for model in models))
            raise ValueError(
                f"router.fallback references unknown model '{fallback}'. "
                f"Available models: {available}"
            )

    return RouterConfig(backend=backend, rules=tuple(rules), fallback=fallback)


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
        default_for_roles=tuple(
            _parse_str_list(
                raw_model.get("default_for_roles"),
                field_name="default_for_roles",
                model_id=key,
            )
        ),
        deployment_only=bool(raw_model.get("deployment_only", False)),
        api_key_env=str(raw_model.get("api_key_env") or "").strip(),
        base_url=str(raw_model.get("base_url") or "").strip(),
        base_url_env=str(raw_model.get("base_url_env") or "").strip(),
        model_id_from_config=model_id_from_config,
        azure_deploy=dict(raw_azure_deploy),
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

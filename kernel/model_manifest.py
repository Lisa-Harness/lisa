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
    operator_only: bool = False
    default_for_roles: tuple[str, ...] = ()
    deployment_only: bool = False
    designations: frozenset[str] = frozenset()
    api_key_env: str = ""
    base_url: str = ""
    base_url_env: str = ""
    model_id_from_config: str = ""
    azure_deploy: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """Validated shared model manifest used by hosts and provider wrappers."""

    models: tuple[ManifestModelEntry, ...]
    mode_models: dict[str, str] = field(default_factory=dict)

    def designated(self, designation: str) -> ManifestModelEntry | None:
        """Return the model entry for one designation, or None."""

        normalized_designation = designation.strip().lower()
        for model in self.models:
            if normalized_designation in model.designations:
                return model
        return self.model_for_mode(normalized_designation)

    def model_for_mode(self, mode_id: str) -> ManifestModelEntry | None:
        """Return the model entry for one mode, or None if not configured."""

        model_ref = self.mode_models.get(mode_id.strip().lower())
        if not model_ref:
            return None
        return self.find_model(model_ref)

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
    supported_designations: set[str] | None = None,
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

    normalized_allowed_providers = {
        provider.strip().lower() for provider in (allowed_providers or _DEFAULT_ALLOWED_PROVIDERS)
    }
    normalized_supported_designations = (
        {designation.strip().lower() for designation in supported_designations}
        if supported_designations is not None
        else None
    )

    models: list[ManifestModelEntry] = []
    seen_model_refs: set[str] = set()
    seen_designations: dict[str, str] = {}
    for index, raw_model in enumerate(raw_models, start=1):
        model = _parse_model_entry(
            raw_model,
            index=index,
            allowed_providers=normalized_allowed_providers,
            supported_designations=normalized_supported_designations,
        )
        for reference in {model.key.lower(), model.model_id.lower()}:
            if reference in seen_model_refs:
                raise ValueError(f"Manifest model reference '{reference}' is duplicated.")
            seen_model_refs.add(reference)

        for designation in model.designations:
            if designation in seen_designations:
                existing_model = seen_designations[designation]
                raise ValueError(
                    f"Designation '{designation}' is assigned to both '{existing_model}' "
                    f"and '{model.model_id}'."
                )
            seen_designations[designation] = model.model_id

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

    return ModelManifest(models=tuple(models), mode_models=mode_models)


def _parse_model_entry(
    raw_model: Any,
    *,
    index: int,
    allowed_providers: set[str],
    supported_designations: set[str] | None,
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
        operator_only=bool(raw_model.get("operator_only", False)),
        default_for_roles=tuple(
            _parse_str_list(
                raw_model.get("default_for_roles"),
                field_name="default_for_roles",
                model_id=key,
            )
        ),
        deployment_only=bool(raw_model.get("deployment_only", False)),
        designations=frozenset(
            _parse_designations(
                raw_model.get("designations"),
                model_id=key,
                supported_designations=supported_designations,
            )
        ),
        api_key_env=str(raw_model.get("api_key_env") or "").strip(),
        base_url=str(raw_model.get("base_url") or "").strip(),
        base_url_env=str(raw_model.get("base_url_env") or "").strip(),
        model_id_from_config=model_id_from_config,
        azure_deploy=dict(raw_azure_deploy),
    )


def _parse_designations(
    raw_designations: Any,
    *,
    model_id: str,
    supported_designations: set[str] | None,
) -> list[str]:
    parsed_designations = _parse_str_list(
        raw_designations,
        field_name="designations",
        model_id=model_id,
    )
    normalized: list[str] = []
    for designation in parsed_designations:
        normalized_designation = designation.strip().lower()
        if (
            supported_designations is not None
            and normalized_designation not in supported_designations
        ):
            supported = ", ".join(sorted(supported_designations))
            raise ValueError(
                f"Unsupported designation '{normalized_designation}' for model '{model_id}'. "
                f"Allowed: {supported}."
            )
        if normalized_designation:
            normalized.append(normalized_designation)
    return normalized


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

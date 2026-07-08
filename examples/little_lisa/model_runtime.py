from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernel.model_manifest import ManifestModelEntry, ModelManifest

try:
    from .model_manifest import load_model_manifest
except ImportError:  # pragma: no cover - direct script execution fallback
    from model_manifest import load_model_manifest  # type: ignore[no-redef]

MANIFEST_ENV = "LITTLE_LISA_MODELS_FILE"
_DEFAULT_DESIGNATIONS = frozenset({"front-door", "default", "deep", "reasoning"})
_MANIFEST_CACHE_KEY = "manifest"


@dataclass(frozen=True, slots=True)
class StubModelDefinition:
    """Tiny model-definition shape expected by Little Lisa operations."""

    model_id: str
    provider: str = "azure_openai"
    reasoning_model: bool = False


@dataclass(frozen=True, slots=True)
class StubCompletion:
    """Completion result shape consumed by the declarative operation runner.

    Token sub-counters (``cached_tokens``, ``reasoning_tokens``) and
    ``duration_ms`` flow through from the provider response so the
    host's token-usage ledger can bill cache hits, reasoning tokens,
    and time-based providers (e.g. Ollama local GPU time) accurately.
    """

    content: str
    tool_calls: list[dict[str, Any]]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    duration_ms: float = 0.0


def manifest_path_from_env(*, env_var: str = MANIFEST_ENV) -> Path | None:
    """Return the optional manifest path configured through the environment."""

    configured = os.getenv(env_var, "").strip()
    return Path(configured) if configured else None


def load_manifest(
    runtime_state: dict[str, Any],
    *,
    env_var: str = MANIFEST_ENV,
) -> ModelManifest:
    """Return the cached manifest snapshot, loading it from disk on first use."""

    manifest = runtime_state.get(_MANIFEST_CACHE_KEY)
    if manifest is None:
        manifest = load_model_manifest(manifest_path_from_env(env_var=env_var))
        runtime_state[_MANIFEST_CACHE_KEY] = manifest
    return manifest


def reload_manifest(
    runtime_state: dict[str, Any],
    *,
    env_var: str = MANIFEST_ENV,
) -> ModelManifest:
    """Reload the manifest from disk and return the new snapshot."""

    manifest = load_model_manifest(manifest_path_from_env(env_var=env_var))
    runtime_state[_MANIFEST_CACHE_KEY] = manifest
    return manifest


def require_models_present(
    runtime_state: dict[str, Any],
    *,
    env_var: str = MANIFEST_ENV,
) -> None:
    """Assert that the active Little Lisa manifest contains at least one model."""

    manifest = load_manifest(runtime_state, env_var=env_var)
    if not manifest.models:
        raise ValueError("Little Lisa requires at least one model in models.yaml.")


def resolve_little_lisa_model(
    model_id: str,
    *,
    runtime_state: dict[str, Any],
    env_var: str = MANIFEST_ENV,
) -> StubModelDefinition:
    """Resolve one model id or designation against the Little Lisa manifest."""

    manifest = load_manifest(runtime_state, env_var=env_var)
    requested = (model_id or "default").strip()
    normalized = requested.lower()

    if normalized in _DEFAULT_DESIGNATIONS:
        designated = _designated_model(manifest, normalized)
        if designated is not None:
            return _to_stub_model_definition(designated)
        if normalized == "default":
            first_model = manifest.models[0]
            return _to_stub_model_definition(first_model)
        raise ValueError(f"Little Lisa manifest is missing the '{normalized}' designation.")

    resolved = _find_model(manifest, requested)
    if resolved is None:
        known_models = ", ".join(model.model_id for model in manifest.models)
        raise ValueError(
            f"Model '{requested}' is not in the Little Lisa manifest. "
            f"Configured models: {known_models}"
        )
    return _to_stub_model_definition(resolved)


def _find_model(manifest: ModelManifest, model_id: str) -> ManifestModelEntry | None:
    exact_model_id = model_id.strip().lower()
    for model in manifest.models:
        if model.model_id.strip().lower() == exact_model_id:
            return model
    return manifest.find_model(model_id)


def _designated_model(
    manifest: ModelManifest,
    designation: str,
) -> ManifestModelEntry | None:
    if designation not in _DEFAULT_DESIGNATIONS:
        return None
    return manifest.designated(designation)


def _to_stub_model_definition(model: ManifestModelEntry) -> StubModelDefinition:
    return StubModelDefinition(
        model_id=model.model_id,
        provider=model.provider,
        reasoning_model=model.reasoning_model,
    )

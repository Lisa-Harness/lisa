from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernel.model_manifest import ManifestModelEntry

from hosts.core.model_manifest import load_model_manifest

logger = logging.getLogger(__name__)

_MANIFEST_CACHE_KEY = "manifest"


@dataclass(frozen=True, slots=True)
class StubModelDefinition:
    """Tiny model-definition shape expected by the host's operations."""

    model_id: str
    provider: str = "azure_openai"
    reasoning_model: bool = False
    context_window: int = 0


@dataclass(frozen=True, slots=True)
class StubCompletion:
    """Completion result shape consumed by the declarative operation runner.

    Token sub-counters (``cached_tokens``, ``reasoning_tokens``) and
    ``duration_ms`` flow through from the provider response so the
    host's token-usage ledger can bill cache hits, reasoning tokens,
    and time-based providers (e.g. Ollama local GPU time) accurately.

    ``finish_reason`` carries the provider's termination signal
    (``'stop'``, ``'length'``, ``'tool_calls'``, or ``None``/``'unknown'``
    for backends that do not report it). Operations that synthesize
    long-form output (e.g. forge_document) use it to detect mid-document
    truncation and issue a continuation pass rather than persisting a
    partial artifact.
    """

    content: str
    tool_calls: list[dict[str, Any]]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    duration_ms: float = 0.0
    finish_reason: str | None = None


def active_manifest_path(*, data_dir: Path) -> Path:
    """Return the canonical active model-manifest path under one data directory."""

    return data_dir / "models.yaml"


def proposal_manifest_path(*, data_dir: Path) -> Path:
    """Return the canonical proposal model-manifest path under one data directory."""

    return data_dir / "models.proposed.yaml"


def load_manifest(
    runtime_state: dict[str, Any],
    *,
    data_dir: Path,
) -> ModelManifest:
    """Return the cached manifest snapshot, loading it from disk on first use."""

    manifest = runtime_state.get(_MANIFEST_CACHE_KEY)
    if manifest is None:
        logger.debug("manifest cache miss; loading from disk")
        manifest = load_model_manifest(active_manifest_path(data_dir=data_dir))
        runtime_state[_MANIFEST_CACHE_KEY] = manifest
    return manifest


def reload_manifest(
    runtime_state: dict[str, Any],
    *,
    data_dir: Path,
) -> ModelManifest:
    """Reload the manifest from disk and return the new snapshot."""

    logger.debug("reloading model manifest from disk")
    manifest = load_model_manifest(active_manifest_path(data_dir=data_dir))
    runtime_state[_MANIFEST_CACHE_KEY] = manifest
    return manifest


def require_models_present(
    runtime_state: dict[str, Any],
    *,
    data_dir: Path,
) -> None:
    """Assert that the active host manifest contains at least one model."""

    manifest = load_manifest(runtime_state, data_dir=data_dir)
    if not manifest.models:
        logger.error("manifest has no models configured")
        raise ValueError("The host requires at least one model in models.yaml.")


def resolve_little_lisa_model(
    model_id: str,
    *,
    runtime_state: dict[str, Any],
    data_dir: Path,
) -> StubModelDefinition:
    """Resolve one model id, mode, or role against the host's manifest.

    Resolution precedence (single path, no divergence):
    1. ``mode_models[ref]`` — explicit per-mode pin.
    2. ``roles[ref]`` — cross-cutting role (e.g. ``deep``).
    3. concrete ``model_id`` lookup.

    No silent first-model fallback: a missing ``default`` mode is a
    load-time validation error (``require_mode_models=True``), not a
    runtime fallback. Empty input defaults to ``"default"`` so callers
    that pass through a bare alias resolve against the default mode.
    """

    manifest = load_manifest(runtime_state, data_dir=data_dir)
    requested = (model_id or "default").strip()
    resolved = manifest.resolve(requested)
    if resolved is None:
        known_models = ", ".join(model.model_id for model in manifest.models)
        logger.error("model %r not in manifest (configured: %s)", requested, known_models)
        raise ValueError(
            f"Model '{requested}' is not in the host's manifest. "
            f"Configured models: {known_models}"
        )
    logger.debug("resolved %r -> %s", requested, resolved.model_id)
    return _to_stub_model_definition(resolved)


def _to_stub_model_definition(model: ManifestModelEntry) -> StubModelDefinition:
    return StubModelDefinition(
        model_id=model.model_id,
        provider=model.provider,
        reasoning_model=model.reasoning_model,
        context_window=model.context_window,
    )

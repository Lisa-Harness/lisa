from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernel.model_manifest import ModelManifest

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

    completion_api: str = "chat_completions"

    """Which OpenAI-compatible endpoint family this model uses: ``"chat_completions"``
    or ``"responses"``. Forwarded from ``ManifestModelEntry.completion_api`` so the
    host's provider dispatch can route Responses-only models (e.g. Codex on Azure)
    to ``client.responses.create`` instead of the unsupported
    ``chat.completions.create``. Default ``"chat_completions"`` for backward
    compatibility with stubs that do not carry the field."""

    capabilities: tuple[str, ...] = ()

    """``ModelCapability`` strings (``text``, ``tools``, ``code``, ...) the

    resolved model declares in the manifest. Forwarded from

    ``ManifestModelEntry.capabilities`` so the kernel's capability gate can

    validate the resolved model against the mode's ``required_capabilities``

    without the kernel reaching into the host's manifest directly. Default

    empty for backward compatibility with stubs that do not carry caps."""

    structured_reliability: float = 0.0

    """Confidence (0.0-1.0) the resolved model reliably honors structured

    output. Forwarded from ``ManifestModelEntry.structured_reliability`` so

    the router activation (``model.structured_reliability``) can gate

    JSON-bound work. Default 0.0 for backward compatibility."""


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


    ``truncated`` is the provider-owned authoritative truncation signal:

    ``True`` = the model stopped before completing (hit max_tokens or an

    incomplete status); ``False`` = a clean stop; ``None`` = the adapter

    genuinely cannot tell (e.g. a backend that omits the signal and did

    not hit the token budget). Operations should prefer ``truncated`` over

    inferring from ``finish_reason`` strings; ``None`` is treated as a

    single safety-net continuation, not a multi-attempt loop.
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

    truncated: bool | None = None

    reasoning_content: str = ""

    """Reasoning-model "thinking" text (gpt-5.x ``message.reasoning_content``).


    Surfaced as a fallback when ``content`` is empty for reasoning models —

    the turn runner decides whether to fold it into the visible response.

    Empty for non-reasoning backends (back-compat default).
    """


def active_manifest_path(*, data_dir: Path) -> Path:
    """Return the canonical active model-manifest path under one data directory."""

    return data_dir / "models.yaml"


def resolve_session_manifest_path(
    *,
    session_plan: Any,
    session_source_dir: Path | None,
    data_dir: Path,
) -> Path:
    """Resolve the active model-manifest path for one session.


    When the session plan declares ``context.models.path``, it is resolved

    relative to the session-context file's directory (``session_source_dir``)

    or as an absolute path, and returned. When the plan omits

    ``context.models`` — or ``session_plan`` is ``None`` — this falls back to

    ``active_manifest_path(data_dir=data_dir)`` so legacy behavior is

    preserved.


    The host boot path validates that a session-declared path exists and is a

    ``.yaml``/``.yml``/``.json`` file; a missing or wrong-type path raises a

    startup ``ValueError``. The compiler (``kernel.session_plan``) only

    validates non-emptiness — it has no filesystem access by design.
    """

    fallback = active_manifest_path(data_dir=data_dir)

    if session_plan is None:
        return fallback

    try:
        models_ref = session_plan.manifest.context.models

    except AttributeError:
        return fallback

    if models_ref is None:
        return fallback

    raw = models_ref.path.strip()

    if not raw:
        return fallback

    candidate = Path(raw).expanduser()

    if not candidate.is_absolute():
        if session_source_dir is None:
            raise ValueError(
                f"context.models.path {raw!r} is relative but the session "
                f"source directory is unknown; pass an absolute path instead."
            )

        candidate = (session_source_dir / candidate).resolve()

    else:
        candidate = candidate.resolve()

    if not candidate.is_file():
        raise ValueError(
            f"context.models.path {raw!r} resolved to {candidate}, which is not a file."
        )

    if candidate.suffix.lower() not in {".yaml", ".yml", ".json"}:
        raise ValueError(f"context.models.path {raw!r} must be a .yaml, .yml, or .json file.")
    return candidate


def proposal_manifest_path(*, data_dir: Path) -> Path:
    """Return the canonical proposal model-manifest path under one data directory."""

    return data_dir / "models.proposed.yaml"


def load_manifest(
    runtime_state: dict[str, Any],
    *,
    data_dir: Path,
    path: Path | None = None,
) -> ModelManifest:
    """Return the cached manifest snapshot, loading it from disk on first use.


    When ``path`` is provided it overrides ``data_dir``'s default

    ``models.yaml`` location; this is how a session plan's

    ``context.models.path`` reaches the loader. Callers without a session

    override omit ``path`` and get ``active_manifest_path(data_dir)``.
    """

    manifest = runtime_state.get(_MANIFEST_CACHE_KEY)

    if manifest is None:
        logger.debug("manifest cache miss; loading from disk")

        resolved = path if path is not None else active_manifest_path(data_dir=data_dir)

        manifest = load_model_manifest(resolved)

        runtime_state[_MANIFEST_CACHE_KEY] = manifest
    return manifest


def reload_manifest(
    runtime_state: dict[str, Any],
    *,
    data_dir: Path,
    path: Path | None = None,
) -> ModelManifest:
    """Reload the manifest from disk and return the new snapshot."""

    logger.debug("reloading model manifest from disk")

    resolved = path if path is not None else active_manifest_path(data_dir=data_dir)

    manifest = load_model_manifest(resolved)

    runtime_state[_MANIFEST_CACHE_KEY] = manifest
    return manifest


def load_manifest_or_none(
    *,
    data_dir: Path,
    path: Path | None = None,
) -> ModelManifest | None:
    """Load the active manifest, or ``None`` when no manifest file exists yet.

    Unlike :func:`load_manifest`, a missing manifest file is a valid first-run
    state and returns ``None`` instead of raising. A present-but-invalid
    manifest still raises so a broken config fails loudly rather than being
    silently ignored. Callers (e.g. ``setup_framework`` startup) treat
    ``None`` as "unconfigured" and let the graceful-degradation paths handle
    it (``model_alias_for_mode`` → ``"default"``, capability snapshot → empty
    models, ``/models show`` → "No active models manifest is available yet").
    """
    resolved = path if path is not None else active_manifest_path(data_dir=data_dir)
    if not resolved.exists():
        logger.warning("model manifest not found at %s; starting unconfigured", resolved)
        return None
    return load_manifest({}, data_dir=data_dir, path=resolved)


def require_models_present(
    runtime_state: dict[str, Any],
    *,
    data_dir: Path,
    path: Path | None = None,
) -> None:
    """Assert that the active host manifest contains at least one model."""

    manifest = load_manifest(runtime_state, data_dir=data_dir, path=path)

    if not manifest.models:
        logger.error("manifest has no models configured")

        raise ValueError("The host requires at least one model in models.yaml.")


def resolve_model(
    model_id: str,
    *,
    runtime_state: dict[str, Any],
    data_dir: Path,
    path: Path | None = None,
) -> StubModelDefinition:
    """Resolve one model id, mode, or role against the host's manifest.


    Resolution precedence (single path, no divergence):

    1. ``mode_models[ref]`` — explicit per-mode pin.

    2. concrete ``model_id`` lookup.


    No silent first-model fallback: a missing ``default`` mode is a

    load-time validation error (``require_mode_models=True``), not a

    runtime fallback. Empty input defaults to ``"default"`` so callers

    that pass through a bare alias resolve against the default mode.
    """

    manifest = load_manifest(runtime_state, data_dir=data_dir, path=path)

    requested = (model_id or "default").strip()

    resolved = manifest.resolve(requested)

    if resolved is None:
        known_models = ", ".join(model.model_id for model in manifest.models)

        logger.error("model %r not in manifest (configured: %s)", requested, known_models)

        raise ValueError(
            f"Model '{requested}' is not in the host's manifest. Configured models: {known_models}"
        )

    logger.debug("resolved %r -> %s", requested, resolved.model_id)

    return StubModelDefinition(
        model_id=resolved.model_id,
        provider=resolved.provider,
        reasoning_model=resolved.reasoning_model,
        context_window=resolved.context_window,
        completion_api=resolved.completion_api,
        capabilities=tuple(resolved.capabilities),
        structured_reliability=resolved.structured_reliability,
    )

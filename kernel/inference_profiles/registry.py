"""Runtime-neutral registry for named inference profiles."""

from __future__ import annotations

from collections.abc import Iterable

from kernel.models import InferenceProfileDefinition

_PROFILE_REGISTRY: dict[str, InferenceProfileDefinition] = {}


def register_inference_profiles(profiles: Iterable[InferenceProfileDefinition]) -> None:
    """Register or replace named inference-profile definitions."""
    for profile in profiles:
        _PROFILE_REGISTRY[profile.profile_id] = profile


def get_inference_profile(profile_id: str) -> InferenceProfileDefinition | None:
    """Return one registered profile definition, if present."""
    return _PROFILE_REGISTRY.get(profile_id)


def iter_inference_profiles() -> tuple[InferenceProfileDefinition, ...]:
    """Return all registered profiles in deterministic key order."""
    return tuple(_PROFILE_REGISTRY[key] for key in sorted(_PROFILE_REGISTRY))

"""Shared settings source filtering for SDK packs and extensions.

Provides ``OwnedSettingsSource`` and ``build_owned_settings_sources`` so that
pack-owned settings models only consume env keys they actually declare, rather
than picking up unrelated framework env vars.

This module lives in ``sdk`` so that both the SDK's own ``Settings`` and
external packs (``lisa-pack-cloudflare``, ``lisa-pack-github``, etc.) can
import it without depending on ``sdk_packs``.
"""

from __future__ import annotations

from typing import Any

from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource


def _normalized_settings_keys(settings_cls: type[BaseSettings]) -> set[str]:
    """Return the normalized field and alias keys accepted by a settings model."""

    keys: set[str] = set()
    for field_name, field_info in settings_cls.model_fields.items():
        keys.add(field_name.lower())
        alias = field_info.alias
        if isinstance(alias, str) and alias:
            keys.add(alias.lower())
    return keys


class OwnedSettingsSource(PydanticBaseSettingsSource):
    """Drop unrelated env keys before validating pack-owned settings."""

    def __init__(self, source: PydanticBaseSettingsSource) -> None:
        super().__init__(source.settings_cls)
        self._source = source

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return self._source.get_field_value(field, field_name)

    def __call__(self) -> dict[str, Any]:
        data = dict(self._source())
        allowed_keys = _normalized_settings_keys(self.settings_cls)
        return {key: value for key, value in data.items() if key.lower() in allowed_keys}


def build_owned_settings_sources(
    init_settings: PydanticBaseSettingsSource,
    env_settings: PydanticBaseSettingsSource,
    dotenv_settings: PydanticBaseSettingsSource,
    file_secret_settings: PydanticBaseSettingsSource,
) -> tuple[PydanticBaseSettingsSource, ...]:
    """Return settings sources that only pass through keys owned by the model."""

    return (
        init_settings,
        OwnedSettingsSource(env_settings),
        OwnedSettingsSource(dotenv_settings),
        file_secret_settings,
    )
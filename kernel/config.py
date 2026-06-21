"""Kernel-owned runtime configuration.

This module intentionally contains only settings consumed by `kernel/**`.
Public packs and extensions use `lisa_sdk.config`, while the root app owns its
broader host and product settings in `app.config`.
"""

from __future__ import annotations

from typing import Any

from pydantic import SecretStr, field_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

_REMOVED_MODEL_ENV_ALIASES = frozenset(
    {
        "deep_model",
        "DEEP_MODEL",
        "azure_openai_deployment_deep",
        "AZURE_OPENAI_DEPLOYMENT_DEEP",
        "self_model_enabled",
        "SELF_MODEL_ENABLED",
        "self_model_max_entries",
        "SELF_MODEL_MAX_ENTRIES",
        "self_model_max_markdown_chars",
        "SELF_MODEL_MAX_MARKDOWN_CHARS",
        "self_model_include_help_hints",
        "SELF_MODEL_INCLUDE_HELP_HINTS",
        "self_learning_enabled",
        "SELF_LEARNING_ENABLED",
    }
)


def _normalized_settings_keys(settings_cls: type[BaseSettings]) -> set[str]:
    """Return normalized field and alias keys accepted by the settings model."""

    keys: set[str] = set()
    for field_name, field_info in settings_cls.model_fields.items():
        keys.add(field_name.lower())
        alias = field_info.alias
        if isinstance(alias, str) and alias:
            keys.add(alias.lower())
    return keys


class _KnownSettingFilteringSource(PydanticBaseSettingsSource):
    """Drop unrelated env vars before Pydantic validates kernel settings."""

    def __init__(self, source: PydanticBaseSettingsSource) -> None:
        super().__init__(source.settings_cls)
        self._source = source

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return self._source.get_field_value(field, field_name)

    def __call__(self) -> dict[str, Any]:
        data = dict(self._source())
        for alias in _REMOVED_MODEL_ENV_ALIASES:
            data.pop(alias, None)
        allowed_keys = _normalized_settings_keys(self.settings_cls)
        return {key: value for key, value in data.items() if key.lower() in allowed_keys}


class Settings(BaseSettings):
    """Settings needed by `kernel/**` only."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        del settings_cls
        return (
            init_settings,
            _KnownSettingFilteringSource(env_settings),
            _KnownSettingFilteringSource(dotenv_settings),
            file_secret_settings,
        )

    sdk_content_roots: str = ""
    workspace_id_fallback: str = "default"
    surf_enabled: bool = True
    surf_session_ttl_minutes: int = 30
    surf_session_max_pages: int = 12
    surf_session_max_links: int = 12
    surf_session_max_commentary_entries: int = 20
    web_search_api_key: SecretStr = SecretStr("")
    web_search_endpoint: str = "https://api.search.brave.com/res/v1/web/search"
    web_search_count: int = 5
    web_search_market: str = "US"
    workshop_max_context_tokens: int = 2800
    workshop_default_shared_context_tokens: int = 700
    workshop_default_per_panel_context_tokens: int = 500
    workshop_max_panels_hard_cap: int = 6
    workshop_default_max_parallel_panels: int = 4
    workshop_max_parallel_panels_hard_cap: int = 4
    workshop_default_synthesis_max_tokens: int = 700
    workshop_max_synthesis_tokens: int = 1200

    @staticmethod
    def _strip_inline_comment(value: object) -> object:
        if not isinstance(value, str):
            return value
        return value.split("#", 1)[0].strip()

    @field_validator(
        "surf_session_ttl_minutes",
        "surf_session_max_pages",
        "surf_session_max_links",
        "surf_session_max_commentary_entries",
        "workshop_max_context_tokens",
        "workshop_default_shared_context_tokens",
        "workshop_default_per_panel_context_tokens",
        "workshop_max_panels_hard_cap",
        "workshop_default_max_parallel_panels",
        "workshop_max_parallel_panels_hard_cap",
        "workshop_default_synthesis_max_tokens",
        "workshop_max_synthesis_tokens",
        mode="before",
    )
    @classmethod
    def _coerce_numeric_with_inline_comments(cls, value: object) -> object:
        stripped = cls._strip_inline_comment(value)
        if stripped == "":
            return None
        return stripped


settings = Settings()


def reload_settings() -> Settings:
    """Reinitialize the module-level settings singleton from the environment."""

    new_settings = Settings()
    current_settings = globals().get("settings")
    if isinstance(current_settings, Settings):
        for field_name in new_settings.model_dump():
            setattr(current_settings, field_name, getattr(new_settings, field_name))
        return current_settings

    globals()["settings"] = new_settings
    return new_settings

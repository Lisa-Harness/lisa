"""Public runtime configuration for Lisa SDK packs and extensions.

This surface is intentionally narrower than the root app config. It exists for
public packs and extensions that need shared runtime defaults without depending
on the kernel's internal config surface or the private app's broader settings.
"""

from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from sdk_packs._settings import build_owned_settings_sources


class Settings(BaseSettings):
    """Public runtime settings consumed by packs and extensions."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
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
        return build_owned_settings_sources(
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )

    heartbeat_interval_seconds: int = 180
    external_signal_timeout_seconds: float = 10.0
    workspace_id_fallback: str = "default"
    state_container: str = "lisa-state"
    intel_inbox_container: str = "lisa-intel-inbox"
    memory_recall_final_limit: int = 8

    @staticmethod
    def _strip_inline_comment(value: object) -> object:
        if not isinstance(value, str):
            return value
        return value.split("#", 1)[0].strip()

    @field_validator(
        "heartbeat_interval_seconds",
        "external_signal_timeout_seconds",
        "memory_recall_final_limit",
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


__all__ = ["Settings", "reload_settings", "settings"]


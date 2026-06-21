from __future__ import annotations

from dataclasses import dataclass

from pydantic import SecretStr
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from sdk_extensions._settings import build_owned_settings_sources


class ClaritySettings(BaseSettings):
    """Extension-owned environment settings for Microsoft Clarity."""

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

    clarity_api_token: SecretStr = SecretStr("")
    clarity_project_id: str = ""
    clarity_base_url: str = "https://www.clarity.ms/export-data/api/v1"
    clarity_blob_container: str = "clarity-exports"
    clarity_blob_prefix: str = "exports/"
    clarity_processed_prefix: str = "processed/"
    clarity_poll_schedule: str = ""
    clarity_daily_request_budget: int = 8


settings = ClaritySettings()


@dataclass(frozen=True)
class ClarityConnectorSettings:
    """Typed Clarity connector settings owned by the Clarity extension."""

    api_token: str
    project_id: str
    base_url: str


def get_clarity_connector_settings() -> ClarityConnectorSettings:
    """Return the Clarity connector settings from the shared runtime config."""

    raw_token = object.__getattribute__(settings, "clarity_api_token")
    api_token = raw_token.get_secret_value() if isinstance(raw_token, SecretStr) else str(raw_token)
    return ClarityConnectorSettings(
        api_token=api_token,
        project_id=settings.clarity_project_id,
        base_url=settings.clarity_base_url,
    )

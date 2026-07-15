from __future__ import annotations

from dataclasses import dataclass

from pydantic import SecretStr
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from sdk.config import settings as runtime_settings
from sdk.pack_settings import build_owned_settings_sources


class CloudflareSettings(BaseSettings):
    """Extension-owned environment settings for Cloudflare."""

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

    cloudflare_api_token: SecretStr = SecretStr("")
    cloudflare_zone_id: str = ""
    cloudflare_account_id: str = ""
    cloudflare_mcp_api_token: SecretStr = SecretStr("")
    """Read-only Cloudflare API token for the MCP server (cloudflare.execute)."""

    @property
    def external_signal_timeout_seconds(self) -> float:
        return runtime_settings.external_signal_timeout_seconds

    @external_signal_timeout_seconds.setter
    def external_signal_timeout_seconds(self, value: float) -> None:
        runtime_settings.external_signal_timeout_seconds = value


settings = CloudflareSettings()


@dataclass(frozen=True)
class CloudflareConnectorSettings:
    """Typed settings used by the Cloudflare signal connector."""

    api_token: str
    zone_id: str
    account_id: str
    timeout_seconds: float


def get_cloudflare_connector_settings() -> CloudflareConnectorSettings:
    """Return the Cloudflare connector settings from the shared runtime config."""

    raw_token = object.__getattribute__(settings, "cloudflare_api_token")
    api_token = raw_token.get_secret_value() if isinstance(raw_token, SecretStr) else str(raw_token)
    return CloudflareConnectorSettings(
        api_token=api_token.strip(),
        zone_id=settings.cloudflare_zone_id.strip(),
        account_id=settings.cloudflare_account_id.strip(),
        timeout_seconds=settings.external_signal_timeout_seconds,
    )



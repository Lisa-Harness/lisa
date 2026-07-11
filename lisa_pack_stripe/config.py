from __future__ import annotations

from dataclasses import dataclass

from pydantic import SecretStr
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from sdk.config import settings as runtime_settings
from sdk.pack_settings import build_owned_settings_sources


class StripeSettings(BaseSettings):
    """Extension-owned environment settings for Stripe."""

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

    stripe_api_key: SecretStr = SecretStr("")

    @property
    def external_signal_timeout_seconds(self) -> float:
        return runtime_settings.external_signal_timeout_seconds

    @external_signal_timeout_seconds.setter
    def external_signal_timeout_seconds(self, value: float) -> None:
        runtime_settings.external_signal_timeout_seconds = value


settings = StripeSettings()


@dataclass(frozen=True)
class StripeConnectorSettings:
    """Typed settings used by the Stripe signal connector."""

    api_key: str
    timeout_seconds: float


def get_stripe_connector_settings() -> StripeConnectorSettings:
    """Return the Stripe connector settings from the shared runtime config."""

    raw_api_key = object.__getattribute__(settings, "stripe_api_key")
    api_key = (
        raw_api_key.get_secret_value() if isinstance(raw_api_key, SecretStr) else str(raw_api_key)
    )
    return StripeConnectorSettings(
        api_key=api_key,
        timeout_seconds=settings.external_signal_timeout_seconds,
    )



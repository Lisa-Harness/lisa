from __future__ import annotations

from dataclasses import dataclass

from pydantic import SecretStr
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from lisa_sdk.config import settings as runtime_settings
from sdk_extensions._settings import build_owned_settings_sources


class GitHubSettings(BaseSettings):
    """Extension-owned environment settings for GitHub."""

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

    github_token: SecretStr = SecretStr("")
    github_repo: str = ""
    github_branch: str = ""

    @property
    def external_signal_timeout_seconds(self) -> float:
        return runtime_settings.external_signal_timeout_seconds

    @external_signal_timeout_seconds.setter
    def external_signal_timeout_seconds(self, value: float) -> None:
        runtime_settings.external_signal_timeout_seconds = value


settings = GitHubSettings()


@dataclass(frozen=True)
class GitHubExtensionSettings:
    """Typed settings used by the GitHub extension."""

    token: str
    repo: str
    branch: str
    timeout_seconds: float


def get_github_extension_settings() -> GitHubExtensionSettings:
    """Return the GitHub extension settings from the shared runtime config."""

    raw_token = object.__getattribute__(settings, "github_token")
    token = raw_token.get_secret_value() if isinstance(raw_token, SecretStr) else str(raw_token)
    return GitHubExtensionSettings(
        token=token.strip(),
        repo=settings.github_repo.strip(),
        branch=settings.github_branch.strip(),
        timeout_seconds=settings.external_signal_timeout_seconds,
    )

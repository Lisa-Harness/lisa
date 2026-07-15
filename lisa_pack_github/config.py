from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from kernel.models import ToolExecutionContext
from sdk.config import settings as runtime_settings
from sdk.pack_settings import build_owned_settings_sources


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


class GitHubPackConfig(BaseModel):
    """Portable session policy for publishing coding artifacts to GitHub."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_branch: str = Field(default="main", min_length=1, max_length=255)
    allowed_base_branches: tuple[str, ...] = ("main",)
    branch_prefix: str = Field(default="lisa/", min_length=1, max_length=80)
    protected_branches: tuple[str, ...] = ("main", "master", "develop", "prod", "production")

    @model_validator(mode="after")
    def _validate_publication_policy(self) -> GitHubPackConfig:
        if self.base_branch not in self.allowed_base_branches:
            raise ValueError("base_branch must be included in allowed_base_branches")
        if self.branch_prefix in self.protected_branches:
            raise ValueError("branch_prefix must not be a protected branch")
        return self


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


def get_github_pack_config(ctx: ToolExecutionContext) -> GitHubPackConfig:
    """Parse the current target's immutable GitHub publication policy."""

    return GitHubPackConfig.model_validate(ctx.pack_config)



from __future__ import annotations

from pydantic import SecretStr

from sdk_packs.github.config import get_github_extension_settings


def test_get_github_extension_settings_reads_current_runtime_values(monkeypatch) -> None:
    monkeypatch.setattr(
        "sdk_packs.github.config.settings.github_token", SecretStr("github-token")
    )
    monkeypatch.setattr("sdk_packs.github.config.settings.github_repo", "owner/repo")
    monkeypatch.setattr("sdk_packs.github.config.settings.github_branch", "release")
    monkeypatch.setattr(
        "sdk_packs.github.config.settings.external_signal_timeout_seconds", 7.5
    )

    github_settings = get_github_extension_settings()

    assert github_settings.token == "github-token"
    assert github_settings.repo == "owner/repo"
    assert github_settings.branch == "release"
    assert github_settings.timeout_seconds == 7.5


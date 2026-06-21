from __future__ import annotations

from pydantic import SecretStr

from sdk_extensions.clarity.config import get_clarity_connector_settings


def test_get_clarity_connector_settings_reads_current_runtime_values(monkeypatch) -> None:
    monkeypatch.setattr(
        "sdk_extensions.clarity.config.settings.clarity_api_token",
        SecretStr("clarity-token"),
    )
    monkeypatch.setattr(
        "sdk_extensions.clarity.config.settings.clarity_project_id",
        "project-123",
    )
    monkeypatch.setattr(
        "sdk_extensions.clarity.config.settings.clarity_base_url",
        "https://clarity.example/api/v1",
    )

    connector_settings = get_clarity_connector_settings()

    assert connector_settings.api_token == "clarity-token"
    assert connector_settings.project_id == "project-123"
    assert connector_settings.base_url == "https://clarity.example/api/v1"

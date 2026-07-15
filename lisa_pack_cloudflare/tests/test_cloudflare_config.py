from __future__ import annotations

from pydantic import SecretStr

from lisa_pack_cloudflare.config import get_cloudflare_connector_settings


def test_get_cloudflare_connector_settings_reads_current_runtime_values(monkeypatch) -> None:
    monkeypatch.setattr(
        "lisa_pack_cloudflare.config.settings.cloudflare_api_token",
        SecretStr("cf-token"),
    )
    monkeypatch.setattr(
        "lisa_pack_cloudflare.config.settings.cloudflare_zone_id",
        "zone-123",
    )
    monkeypatch.setattr(
        "lisa_pack_cloudflare.config.settings.cloudflare_account_id",
        "acct-123",
    )
    monkeypatch.setattr(
        "lisa_pack_cloudflare.config.settings.external_signal_timeout_seconds",
        8.5,
    )

    connector_settings = get_cloudflare_connector_settings()

    assert connector_settings.api_token == "cf-token"
    assert connector_settings.zone_id == "zone-123"
    assert connector_settings.account_id == "acct-123"
    assert connector_settings.timeout_seconds == 8.5


def test_mcp_api_token_is_separate_from_connector_token(monkeypatch) -> None:
    """CLOUDFLARE_MCP_API_TOKEN is a distinct read-only token for the MCP server."""
    from pydantic import SecretStr

    monkeypatch.setattr(
        "lisa_pack_cloudflare.config.settings.cloudflare_mcp_api_token",
        SecretStr("mcp-readonly-token"),
    )
    raw = object.__getattribute__(
        __import__("lisa_pack_cloudflare.config", fromlist=["settings"]).settings,
        "cloudflare_mcp_api_token",
    )
    assert raw.get_secret_value() == "mcp-readonly-token"

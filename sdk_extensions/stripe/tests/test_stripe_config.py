from __future__ import annotations

from pydantic import SecretStr

from sdk_extensions.stripe.config import get_stripe_connector_settings


def test_get_stripe_connector_settings_reads_current_runtime_values(monkeypatch) -> None:
    monkeypatch.setattr(
        "sdk_extensions.stripe.config.settings.stripe_api_key",
        SecretStr("stripe-test-key"),
    )
    monkeypatch.setattr(
        "sdk_extensions.stripe.config.settings.external_signal_timeout_seconds",
        9.5,
    )

    connector_settings = get_stripe_connector_settings()

    assert connector_settings.api_key == "stripe-test-key"
    assert connector_settings.timeout_seconds == 9.5

from __future__ import annotations

from datetime import UTC, datetime

from sdk.lisa_context.context_broker import build_ambient_facts


def test_build_ambient_facts_renders_expected_fields() -> None:
    now = datetime(2026, 7, 11, 12, 34, tzinfo=UTC)

    rendered = build_ambient_facts(
        timezone="UTC",
        host_os="Windows",
        host_name="little_lisa",
        now=now,
    )

    assert "local_date: 2026-07-11" in rendered
    assert "local_time: 12:34 UTC" in rendered
    assert "local_weekday: Saturday" in rendered
    assert "timezone: UTC" in rendered
    assert "utc_now: 2026-07-11 12:34 UTC" in rendered
    assert "host_os: Windows" in rendered
    assert "host: little_lisa" in rendered


def test_build_ambient_facts_falls_back_to_utc_for_invalid_timezone() -> None:
    now = datetime(2026, 7, 11, 8, 0, tzinfo=UTC)

    rendered = build_ambient_facts(
        timezone="Invalid/Zone",
        now=now,
    )

    assert "timezone: UTC" in rendered
    assert "local_time: 08:00 UTC" in rendered

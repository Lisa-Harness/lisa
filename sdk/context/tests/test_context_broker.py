from __future__ import annotations

from datetime import UTC, datetime, timezone as dt_timezone

import pytest

from sdk.context.context_broker import build_ambient_facts


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


def test_build_ambient_facts_falls_back_to_system_local_for_unresolvable_zone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ZoneInfo can't load the configured zone, fall back to system local.

    Previously this silently fell back to UTC, causing the model to see the
    wrong local time when the IANA tz database was unavailable (e.g. Windows
    without the ``tzdata`` package).  The fallback now uses the system local
    timezone, mirroring the status-line badge in ``hosts/cli/repl.py``.
    """
    fixed_local = dt_timezone(-1 * 3600)  # fixed offset -01:00

    class _FakeDateTime:
        @staticmethod
        def now(tz: object | None = None) -> datetime:
            if tz is UTC:
                return datetime(2026, 7, 11, 8, 0, tzinfo=UTC)
            # astimezone() with no args returns local time
            return datetime(2026, 7, 11, 7, 0, tzinfo=fixed_local)

    monkeypatch.setattr("kernel.context.context_broker.datetime", _FakeDateTime)

    rendered = build_ambient_facts(
        timezone="Invalid/Zone",
        now=datetime(2026, 7, 11, 8, 0, tzinfo=UTC),
    )

    # Should NOT fall back to UTC — should use the system local offset
    assert "timezone: UTC" not in rendered
    assert "local_time: 07:00" in rendered
    assert "utc_now: 2026-07-11 08:00 UTC" in rendered


def test_build_ambient_facts_uses_iana_zone_when_available() -> None:
    """With tzdata installed, an IANA zone produces correct local time."""
    now = datetime(2026, 7, 11, 16, 0, tzinfo=UTC)  # noon EDT

    rendered = build_ambient_facts(
        timezone="America/New_York",
        now=now,
    )

    assert "timezone: America/New_York" in rendered
    assert "local_time: 12:00" in rendered
    assert "utc_now: 2026-07-11 16:00 UTC" in rendered

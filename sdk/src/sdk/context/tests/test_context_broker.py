from __future__ import annotations

import os
import time
from datetime import UTC, datetime

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


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="TZ-driven local zone needs POSIX tzset")
def test_build_ambient_facts_falls_back_to_system_local_for_unresolvable_zone(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """When ZoneInfo can't load the configured zone, fall back to system local.

    Previously this silently fell back to UTC, causing the model to see the
    wrong local time when the IANA tz database was unavailable (e.g. Windows
    without the ``tzdata`` package).  The fallback now uses the system local
    timezone, mirroring the status-line badge in ``hosts/cli/repl.py``.
    """
    # Drive the "system local" zone through the TZ env var instead of faking
    # ``datetime``: the fallback path's ``datetime.now().astimezone().tzinfo``
    # reads the process zone, so setting TZ makes the fallback deterministic.
    prior_tz = os.environ.get("TZ")
    monkeypatch.setenv("TZ", "America/Sao_Paulo")  # UTC-03:00, no DST since 2019
    time.tzset()

    def _restore() -> None:
        if prior_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = prior_tz
        time.tzset()

    request.addfinalizer(_restore)

    rendered = build_ambient_facts(
        timezone="Invalid/Zone",
        now=datetime(2026, 7, 11, 8, 0, tzinfo=UTC),
    )

    # Should NOT fall back to UTC — should use the system local offset (-03:00).
    assert "timezone: UTC" not in rendered
    assert "local_time: 05:00" in rendered
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

"""Tests for disabled-pack exclusion in signal collector discovery."""

from __future__ import annotations

import importlib.metadata

from kernel.intake.connector_discovery import discover_installed_signal_collectors


class _FakeCollector:
    collector_id = "fake"
    display_name = "Fake"


class _FakeEntryPoint:
    def __init__(self, *, name: str, factory: object, dist_name: str | None = None) -> None:
        self.name = name
        self._factory = factory

        class _FakeDist:
            def __init__(self, dname: str) -> None:
                self.metadata = {"Name": dname}

        self.dist = _FakeDist(dist_name) if dist_name else None

    def load(self):
        return self._factory


def test_discover_installed_signal_collectors_excludes_disabled_packs(
    monkeypatch,
) -> None:
    """Entry points whose distribution name is in excluded_packs are skipped."""
    active_collector = type(
        "ActiveCollector",
        (),
        {
            "collector_id": "active",
            "display_name": "Active",
        },
    )
    disabled_collector = type(
        "DisabledCollector",
        (),
        {
            "collector_id": "disabled",
            "display_name": "Disabled",
        },
    )

    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda group: [
            _FakeEntryPoint(name="active", factory=active_collector, dist_name="lisa-pack-active"),
            _FakeEntryPoint(
                name="disabled",
                factory=disabled_collector,
                dist_name="lisa-pack-disabled",
            ),
        ],
    )

    result = discover_installed_signal_collectors(
        excluded_packs={"lisa-pack-disabled"},
    )
    assert "active" in result
    assert "disabled" not in result


def test_discover_installed_signal_collectors_none_returns_all(monkeypatch) -> None:
    """excluded_packs=None preserves backward-compatible behavior."""
    active_collector = type(
        "ActiveCollector",
        (),
        {
            "collector_id": "active",
            "display_name": "Active",
        },
    )
    other_collector = type(
        "OtherCollector",
        (),
        {
            "collector_id": "other",
            "display_name": "Other",
        },
    )

    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda group: [
            _FakeEntryPoint(name="active", factory=active_collector, dist_name="lisa-pack-active"),
            _FakeEntryPoint(name="other", factory=other_collector, dist_name="lisa-pack-other"),
        ],
    )

    result = discover_installed_signal_collectors(excluded_packs=None)
    assert "active" in result
    assert "other" in result


def test_discover_installed_signal_collectors_empty_exclusion_returns_all(
    monkeypatch,
) -> None:
    """An empty exclusion set is equivalent to no filtering."""
    collector = type(
        "TestCollector",
        (),
        {
            "collector_id": "test",
            "display_name": "Test",
        },
    )

    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda group: [
            _FakeEntryPoint(name="test", factory=collector, dist_name="lisa-pack-test"),
        ],
    )

    result = discover_installed_signal_collectors(excluded_packs=set())
    assert "test" in result

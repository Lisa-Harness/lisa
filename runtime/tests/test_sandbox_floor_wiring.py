"""Tests for the runtime's sandbox-floor materialization wiring.

Covers :func:`runtime.turn_hooks_builder._merged_sandbox_floor` (owner
resolution, exclusive merge onto the authored floor, validation via the
implementation validator, and the inert no-implementation path) and
:func:`runtime.turn_hooks_builder._materialize_sandbox_floor` (the merged
floor lands in the owner's binding configuration under ``sandbox.floor``).

Discovery is monkeypatched — the entry-point loader paths are covered by
the kernel discovery tests; these tests assert the runtime join.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import runtime.turn_hooks_builder as builder
from kernel.sandbox_floor import (
    SandboxFloorConflict,
    SandboxFloorContext,
    SandboxFloorContributionEntry,
    SandboxImplementation,
)


def _runtime(
    configuration: dict[str, dict[str, Any]],
    *,
    capabilities: dict[str, str] | None = None,
    data_dir: Any = "/data",
) -> SimpleNamespace:
    manifest = SimpleNamespace(
        resources=SimpleNamespace(packs=SimpleNamespace(configuration=configuration)),
        providers=SimpleNamespace(capabilities=capabilities or {}),
        declared_pack_owners=lambda: set(configuration),
    )
    return SimpleNamespace(
        session_plan=SimpleNamespace(manifest=manifest),
        paths=SimpleNamespace(data_dir=data_dir),
    )


def _bwrap_impl() -> SandboxImplementation:
    def _validate(floor: Any) -> None:
        if not isinstance(floor, dict):
            raise TypeError("floor must be a mapping")
        if floor.get("network") == "sometimes":
            raise ValueError("bwrap cannot enforce 'sometimes'")

    return SandboxImplementation(pack_name="lisa-pack-bwrap", validate_floor=_validate)


def _contribution(factory: Any) -> SandboxFloorContributionEntry:
    return SandboxFloorContributionEntry(pack_name="lisa-pack-coding", factory=factory)


@pytest.fixture(autouse=True)
def _no_discovered_entry_points(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(builder, "discover_sandbox_implementations", lambda **_: [])
    monkeypatch.setattr(builder, "discover_sandbox_floor_contributions", lambda **_: [])
    monkeypatch.setattr(builder, "load_disabled_pack_names", lambda _: set())


def test_no_implementations_yields_inert_floor() -> None:
    runtime = _runtime({})
    owner, floor, contributing = builder._merged_sandbox_floor(runtime)
    assert owner is None and floor is None and contributing == ()


def test_owner_auto_selected_and_contributions_merged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    impl = _bwrap_impl()
    contribution = {"execution": "sandbox", "paths": {"/wt": "rw"}}
    runtime = _runtime(
        {
            "lisa-pack-bwrap": {"sandbox": {"floor": {"persistence": "direct-write"}}},
            "lisa-pack-coding": {"workspace_root": "/ws"},
        }
    )

    def factory(config: Any, context: Any) -> dict[str, Any]:
        assert isinstance(context, SandboxFloorContext)
        assert context.data_dir == "/data"
        return contribution

    monkeypatch.setattr(builder, "discover_sandbox_implementations", lambda **_: [impl])
    monkeypatch.setattr(
        builder,
        "discover_sandbox_floor_contributions",
        lambda **_: [_contribution(factory)],
    )
    owner, floor, contributing = builder._merged_sandbox_floor(runtime)

    assert owner == "lisa-pack-bwrap"
    assert floor is not None
    assert floor["paths"] == {"/wt": "rw"}
    assert floor["persistence"] == "direct-write"
    assert contributing == ("lisa-pack-coding",)


def test_contributions_require_authored_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    impl = _bwrap_impl()

    def factory(config: Any, context: Any) -> dict[str, Any]:
        return {"paths": {"/wt": "rw"}}

    runtime = _runtime({"lisa-pack-coding": {"workspace_root": "/ws"}})
    monkeypatch.setattr(builder, "discover_sandbox_implementations", lambda **_: [impl])
    monkeypatch.setattr(
        builder,
        "discover_sandbox_floor_contributions",
        lambda **_: [_contribution(factory)],
    )
    with pytest.raises(SandboxFloorConflict):
        builder._merged_sandbox_floor(runtime)


def test_implementation_validator_rejects_merged_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    impl = _bwrap_impl()

    def factory(config: Any, context: Any) -> dict[str, Any]:
        # "network" is in the merge vocabulary, so the shape check passes;
        # only the implementation validator can reject the value.
        return {"network": "sometimes"}

    runtime = _runtime(
        {
            "lisa-pack-bwrap": {"sandbox": {"floor": {}}},
            # A pack with no configuration contributes nothing (its factory
            # is never invoked) — the contributing pack must have config.
            "lisa-pack-coding": {},
        }
    )
    monkeypatch.setattr(builder, "discover_sandbox_implementations", lambda **_: [impl])
    monkeypatch.setattr(
        builder,
        "discover_sandbox_floor_contributions",
        lambda **_: [_contribution(factory)],
    )
    with pytest.raises(ValueError, match="cannot enforce"):
        builder._merged_sandbox_floor(runtime)


def test_conflicting_contributions_raise_at_merge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    impl = _bwrap_impl()

    def coding(config: Any, context: Any) -> dict[str, Any]:
        return {"network": "allow"}

    def other(config: Any, context: Any) -> dict[str, Any]:
        return {"network": "deny"}

    runtime = _runtime(
        {
            "lisa-pack-bwrap": {"sandbox": {"floor": {}}},
            "lisa-pack-coding": {"workspace_root": "/ws"},
            "lisa-pack-finances": {"enabled": True},
        }
    )
    entries = [
        SandboxFloorContributionEntry(pack_name="lisa-pack-coding", factory=coding),
        SandboxFloorContributionEntry(pack_name="lisa-pack-finances", factory=other),
    ]
    monkeypatch.setattr(builder, "discover_sandbox_implementations", lambda **_: [impl])
    monkeypatch.setattr(builder, "discover_sandbox_floor_contributions", lambda **_: entries)
    with pytest.raises(SandboxFloorConflict):
        builder._merged_sandbox_floor(runtime)


def test_materialize_places_merged_floor_on_owner_config() -> None:
    configuration = {
        "lisa-pack-bwrap": {"sandbox": {"floor": {"network": "deny"}}, "other": 1},
        "lisa-pack-coding": {"workspace_root": "/ws"},
    }
    merged = builder._materialize_sandbox_floor(
        configuration, "lisa-pack-bwrap", {"network": "deny", "paths": {"/wt": "rw"}}
    )
    assert merged["lisa-pack-bwrap"]["sandbox"]["floor"] == {
        "network": "deny",
        "paths": {"/wt": "rw"},
    }
    assert merged["lisa-pack-bwrap"]["other"] == 1
    assert merged["lisa-pack-coding"] == {"workspace_root": "/ws"}
    # The session plan's configuration is not mutated.
    assert configuration["lisa-pack-bwrap"]["sandbox"]["floor"] == {"network": "deny"}


def test_materialize_without_owner_is_plain_copy() -> None:
    configuration = {"lisa-pack-coding": {"workspace_root": "/ws"}}
    merged = builder._materialize_sandbox_floor(configuration, None, None)
    assert merged == configuration and merged is not configuration

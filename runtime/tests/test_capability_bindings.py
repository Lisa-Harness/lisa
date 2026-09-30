"""Tests for runtime-owned capability binding snapshots."""

from __future__ import annotations

import logging

import pytest
from kernel.capability_binding import CapabilityRef
from runtime.capability_bindings import SessionCapabilityBindings


def test_bindings_freeze_snapshot_and_reject_unknown_capabilities() -> None:
    configuration = {
        "lisa-pack-owner": {"checks": [{"id": "unit"}]},
        "lisa-pack-other": {"secret": "not-visible"},
    }
    capability = CapabilityRef("tool", "owner_tool", "lisa-pack-owner")
    bindings = SessionCapabilityBindings(
        configuration,
        capabilities=[
            capability,
            CapabilityRef("tool", "other_tool", "lisa-pack-other"),
        ],
    )
    configuration["lisa-pack-owner"]["checks"][0]["id"] = "changed"

    owner = bindings.binding_for(capability)
    assert owner is not None
    assert owner.configuration["checks"][0]["id"] == "unit"
    assert "secret" not in owner.configuration
    with pytest.raises(TypeError):
        owner.configuration["checks"][0]["id"] = "changed"
    assert bindings.binding_for(CapabilityRef("operation", "missing", "missing")) is None


def test_bindings_warn_and_ignore_unknown_configured_owner(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="runtime.capability_bindings"):
        bindings = SessionCapabilityBindings(
            {"lisa-pack-unknown": {"enabled": True}},
            capabilities=[CapabilityRef("tool", "known", "lisa-pack-known")],
        )

    assert "lisa-pack-unknown" in caplog.text
    assert bindings.binding_for(CapabilityRef("tool", "known", "lisa-pack-known")) is not None
    assert bindings.binding_for(CapabilityRef("tool", "gone", "lisa-pack-unknown")) is None


def test_bindings_accept_declared_provider_without_catalog_capabilities() -> None:
    bindings = SessionCapabilityBindings(
        {"lisa-pack-memory": {"memory_distill_max_tokens": 5000}},
        capabilities=[CapabilityRef("tool", "known", "lisa-pack-known")],
        declared_owners={"lisa-pack-memory"},
    )

    assert bindings.binding_for(CapabilityRef("tool", "missing", "lisa-pack-memory")) is None

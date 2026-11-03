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
    assert "will fail at dispatch" not in caplog.text
    assert bindings.binding_for(CapabilityRef("tool", "known", "lisa-pack-known")) is not None
    assert bindings.binding_for(CapabilityRef("tool", "gone", "lisa-pack-unknown")) is None


def test_bindings_warn_names_failing_catalog_capabilities(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A configured owner naming a pack whose catalog owner id differs (e.g.
    the bare directory slug before authoritative pack_id authoring) gets the
    dispatch-warning listing the live capabilities that will fail (2026-11-02
    WiDWebsite memory-write loss)."""
    with caplog.at_level(logging.WARNING, logger="runtime.capability_bindings"):
        SessionCapabilityBindings(
            {"lisa-pack-memory": {"memory_distill_max_tokens": 5000}},
            capabilities=[
                CapabilityRef("tool", "known", "lisa-pack-known"),
                CapabilityRef("operation", "recall_memory", "memory"),
                CapabilityRef("operation", "synthesize_and_store_learning", "memory"),
            ],
        )

    assert "lisa-pack-memory" in caplog.text
    assert "operation:recall_memory" in caplog.text
    assert "operation:synthesize_and_store_learning" in caplog.text
    assert "will fail at dispatch" in caplog.text


def test_bindings_accept_declared_provider_without_catalog_capabilities() -> None:
    bindings = SessionCapabilityBindings(
        {"lisa-pack-memory": {"memory_distill_max_tokens": 5000}},
        capabilities=[CapabilityRef("tool", "known", "lisa-pack-known")],
        declared_owners={"lisa-pack-memory"},
    )

    assert bindings.binding_for(CapabilityRef("tool", "missing", "lisa-pack-memory")) is None


def test_bindings_without_capability_catalog_use_owner_configuration() -> None:
    bindings = SessionCapabilityBindings({"lisa-pack-owner": {"enabled": True}})

    configured = bindings.binding_for(CapabilityRef("tool", "owner_tool", "lisa-pack-owner"))
    assert configured is not None
    assert configured.configuration["enabled"] is True
    with pytest.raises(TypeError):
        configured.configuration["enabled"] = False

    unconfigured = bindings.binding_for(CapabilityRef("operation", "other", "lisa-pack-other"))
    assert unconfigured is not None
    assert unconfigured.owner_id == "lisa-pack-other"
    assert not unconfigured.configuration


def test_bindings_with_explicit_empty_capability_catalog_reject_all() -> None:
    bindings = SessionCapabilityBindings(
        {"lisa-pack-owner": {"enabled": True}},
        capabilities=[],
    )

    assert bindings.binding_for(CapabilityRef("tool", "owner_tool", "lisa-pack-owner")) is None


def test_bindings_warn_matches_normalized_owner_alias(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A configured key matching the catalog owner after case/underscore
    normalization (but not via the lisa-pack- prefix) names the failing
    capabilities too (bare-slug alias, second id form)."""
    with caplog.at_level(logging.WARNING, logger="runtime.capability_bindings"):
        SessionCapabilityBindings(
            {"Memory_pack": {"enabled": True}},
            capabilities=[
                CapabilityRef("operation", "recall", "memory-pack"),
                CapabilityRef("tool", "unrelated", "other"),
            ],
        )

    assert "will fail at dispatch" in caplog.text
    assert "operation:recall" in caplog.text

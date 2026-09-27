"""Tests for runtime-owned session configuration snapshots."""

from __future__ import annotations

import pytest
from runtime.pack_configuration import SessionPackConfigurationResolver


def test_resolver_freezes_snapshot_and_returns_only_requested_pack() -> None:
    configuration = {
        "lisa-pack-owner": {"checks": [{"id": "unit"}]},
        "lisa-pack-other": {"secret": "not-visible"},
    }
    resolver = SessionPackConfigurationResolver(configuration)
    configuration["lisa-pack-owner"]["checks"][0]["id"] = "changed"
    owner = resolver.for_pack(" lisa-pack-owner ")
    assert owner["checks"][0]["id"] == "unit"
    assert "secret" not in owner
    with pytest.raises(TypeError):
        owner["checks"][0]["id"] = "changed"
    assert resolver.for_pack("missing") == {}

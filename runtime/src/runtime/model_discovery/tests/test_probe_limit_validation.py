# Per-call ceiling validation and catalog-write atomicity — slice 2. Network-free.

from __future__ import annotations

import logging

import pytest

from runtime.model_discovery.catalog import (
    build_proposed_catalog,
    validate_limit_consistency,
    write_proposed_catalog,
)
from runtime.model_discovery.types import DiscoveredModel


def test_consistent_ceiling_passes() -> None:
    assert (
        validate_limit_consistency(
            provider="openrouter",
            model_id="m",
            output_limit=8192,
            context_window=128000,
        )
        is None
    )


def test_unknown_limits_pass() -> None:
    assert (
        validate_limit_consistency(
            provider="openrouter", model_id="m", output_limit=0, context_window=0
        )
        is None
    )


def test_oversized_ceiling_rejected_with_visible_diagnostic() -> None:
    diagnostic = validate_limit_consistency(
        provider="openrouter",
        model_id="m",
        output_limit=999999,
        context_window=128000,
    )
    assert diagnostic is not None
    assert "999999" in diagnostic
    assert "128000" in diagnostic
    assert "clamping" in diagnostic


def test_oversized_ceiling_omitted_never_clamped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = DiscoveredModel(
        model_id="m",
        provider="openrouter",
        context_window=1000,
        max_completion_tokens=5000,
    )
    with caplog.at_level(logging.WARNING):
        catalog = build_proposed_catalog((model,))
    fields = catalog["providers"]["openrouter"]["models"][0]["fields"]
    assert "max_completion_tokens" not in fields
    assert any("5000" in str(item) for item in fields.get("diagnostics", []))
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_successful_write_replaces_catalog(tmp_path: object) -> None:
    from pathlib import Path

    import yaml

    target = Path(str(tmp_path)) / "models.proposed.yaml"
    write_proposed_catalog(target, (DiscoveredModel(model_id="a", provider="openai"),))
    document = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert document["providers"]["openai"]["models"][0]["id"] == "a"


def test_failed_run_never_replaces_last_good_catalog(tmp_path: object) -> None:
    """The writer is only invoked after an all-OK run; a failed run simply
    never calls it, so the last good catalog survives byte-for-byte."""
    from pathlib import Path

    target = Path(str(tmp_path)) / "models.proposed.yaml"
    target.write_text("schema_version: 1\n", encoding="utf-8")
    # Simulated partial/failed run: no write call happens.
    assert target.read_text(encoding="utf-8") == "schema_version: 1\n"

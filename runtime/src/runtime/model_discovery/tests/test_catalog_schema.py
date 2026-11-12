# Catalog schema tests — MODEL-DATA-LAYERS slice 2. Network-free.

from __future__ import annotations

from runtime.model_discovery.catalog import (
    CATALOG_SCHEMA_VERSION,
    build_proposed_catalog,
    model_catalog_entry,
)
from runtime.model_discovery.types import DiscoveredModel

_OBSERVED = "2026-11-12T00:00:00+00:00"


def _model(**overrides: object) -> DiscoveredModel:
    values: dict[str, object] = {"model_id": "test-model", "provider": "openai"}
    values.update(overrides)
    return DiscoveredModel(**values)  # type: ignore[arg-type]


def test_schema_version_and_generated_at_present() -> None:
    catalog = build_proposed_catalog((_model(),))
    assert catalog["schema_version"] == CATALOG_SCHEMA_VERSION
    assert catalog["generated_at"]


def test_unpublished_fields_are_omitted_not_zeroed() -> None:
    """Missing means unknown: no zeros, false, or guesses in the row."""
    entry = model_catalog_entry(_model(), _OBSERVED)
    fields = entry["fields"]
    assert "context_window" not in fields
    assert "max_completion_tokens" not in fields
    assert "prompt_price" not in fields
    assert "expires_at" not in fields
    assert "display_name" not in fields


def test_published_fields_carry_source_and_observed_at() -> None:
    entry = model_catalog_entry(
        _model(display_name="Test Model", context_window=128000, prompt_price=0.5),
        _OBSERVED,
    )
    assert entry["fields"]["display_name"] == {
        "source": "provider",
        "observed_at": _OBSERVED,
        "value": "Test Model",
    }
    assert entry["fields"]["context_window"] == {
        "source": "provider",
        "observed_at": _OBSERVED,
        "value": 128000,
    }
    assert entry["fields"]["prompt_price"]["value"] == 0.5


def test_expires_at_is_kept_when_published() -> None:
    from datetime import date

    entry = model_catalog_entry(
        _model(expires_at=date(2027, 1, 1)),
        _OBSERVED,
    )
    assert entry["fields"]["expires_at"]["value"] == "2027-01-01"


def test_entries_group_by_provider_sorted() -> None:
    catalog = build_proposed_catalog(
        (
            _model(model_id="a", provider="openai"),
            _model(model_id="b", provider="ollama"),
        )
    )
    assert set(catalog["providers"]) == {"openai", "ollama"}


def test_entry_carries_no_raw_payload() -> None:
    entry = model_catalog_entry(_model(display_name="d"), _OBSERVED)
    assert set(entry) == {"id", "provider", "fields"}
    assert set(entry["fields"]["display_name"]) == {"source", "observed_at", "value"}

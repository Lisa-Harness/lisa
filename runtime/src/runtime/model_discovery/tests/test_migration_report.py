# Migration report: unestablishable ownership + operator pin reporting — slice 3. Network-free.

from __future__ import annotations

import copy

from runtime.model_discovery.compose import compose_effective_manifest


def _catalog() -> dict:
    return {
        "schema_version": 1,
        "providers": {
            "openai": {
                "models": [
                    {
                        "id": "gpt-5.4",
                        "provider": "openai",
                        "fields": {
                            "context_window": {
                                "source": "provider",
                                "observed_at": "2026-11-12T00:00:00+00:00",
                                "value": 256000,
                            }
                        },
                    }
                ]
            }
        },
    }


def test_unestablishable_ownership_is_reported_not_guessed() -> None:
    """An operator field the composition tables do not know is surfaced
    as unestablishable ownership — provenance is never invented."""

    operator = {
        "version": 1,
        "models": [{"model_id": "gpt-5.4", "provider": "openai", "wat_field": 1}],
    }
    _composed, report = compose_effective_manifest(operator, _catalog())
    assert any(
        entry["kind"] == "unestablishable_ownership"
        and entry["model"] == "gpt-5.4"
        and entry["field"] == "wat_field"
        for entry in report
    )


def test_unknown_catalog_field_is_reported_not_merged() -> None:
    operator = {"version": 1, "models": [{"model_id": "gpt-5.4", "provider": "openai"}]}
    catalog = _catalog()
    catalog["providers"]["openai"]["models"][0]["fields"]["quantum_flux"] = {
        "source": "provider",
        "observed_at": "2026-11-12T00:00:00+00:00",
        "value": 42,
    }
    composed, report = compose_effective_manifest(operator, catalog)
    assert "quantum_flux" not in composed["models"][0]
    assert any(
        entry["kind"] == "unknown_catalog_field" and entry["field"] == "quantum_flux"
        for entry in report
    )


def test_operator_pin_to_unavailable_model_is_reported_never_erased() -> None:
    """A pinned model with no catalog row keeps its row and gets a
    pinned_not_measured report — no surprise deletion, rollback stays
    possible."""

    operator = {
        "version": 1,
        "models": [{"model_id": "retired-model", "provider": "openai", "display_name": "Retired"}],
    }
    composed, report = compose_effective_manifest(operator, _catalog())
    rows = composed["models"]
    assert len(rows) == 1
    assert rows[0]["model_id"] == "retired-model"
    assert rows[0]["display_name"] == "Retired"
    assert any(
        entry["kind"] == "pinned_not_measured" and entry["model"] == "retired-model"
        for entry in report
    )


def test_report_never_mutates_either_input() -> None:
    operator = {
        "version": 1,
        "models": [{"model_id": "gpt-5.4", "provider": "openai", "cost_tier": "frontier"}],
    }
    catalog = _catalog()
    operator_snapshot = copy.deepcopy(operator)
    catalog_snapshot = copy.deepcopy(catalog)
    compose_effective_manifest(operator, catalog)
    assert operator == operator_snapshot
    assert catalog == catalog_snapshot

# /models apply promotion gating — slice 3. Network-free.

from __future__ import annotations

import copy

from runtime.model_discovery.catalog import read_catalog
from runtime.model_discovery.compose import compose_effective_manifest


def _operator_document() -> dict:
    return {
        "version": 1,
        "mode_models": {"default": "gpt-5.4"},
        "models": [
            {
                "model_id": "gpt-5.4",
                "provider": "openai",
                "display_name": "GPT-5.4",
                "context_window": 256000,
            }
        ],
    }


def _catalog() -> dict:
    return {
        "schema_version": 1,
        "generated_at": "2026-11-12T00:00:00+00:00",
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
                                "value": 400000,
                            },
                            "prompt_price": {
                                "source": "provider",
                                "observed_at": "2026-11-12T00:00:00+00:00",
                                "value": 1.25,
                            },
                        },
                    }
                ]
            }
        },
    }


def test_operator_document_is_never_mutated_by_composition() -> None:
    """Promotion must leave the operator document byte-stable."""

    operator = _operator_document()
    snapshot = copy.deepcopy(operator)
    composed, _report = compose_effective_manifest(operator, _catalog())
    assert operator == snapshot
    # The composed copy is a different object; operator identity intact.
    assert composed is not operator


def test_promotion_keeps_operator_pins_and_structure() -> None:
    composed, _report = compose_effective_manifest(_operator_document(), _catalog())
    assert composed["version"] == 1
    assert composed["mode_models"] == {"default": "gpt-5.4"}
    rows = composed["models"]
    assert len(rows) == 1
    assert rows[0]["model_id"] == "gpt-5.4"
    assert rows[0]["provider"] == "openai"
    assert rows[0]["display_name"] == "GPT-5.4"


def test_promotion_fills_operator_unset_fields_from_catalog() -> None:
    composed, report = compose_effective_manifest(_operator_document(), _catalog())
    row = composed["models"][0]
    # Operator set context_window (256000); catalog only fills prompt_price.
    assert row["context_window"] == 256000
    assert row["prompt_price"] == 1.25
    assert any(
        entry["kind"] == "composed_from_catalog" and entry["field"] == "prompt_price"
        for entry in report
    )


def test_catalog_models_absent_from_operator_are_reported_not_added() -> None:
    catalog = _catalog()
    catalog["providers"]["anthropic"] = {
        "models": [
            {
                "id": "claude-sonnet-5",
                "provider": "anthropic",
                "fields": {},
            }
        ]
    }
    composed, report = compose_effective_manifest(_operator_document(), catalog)
    assert len(composed["models"]) == 1
    assert any(
        entry["kind"] == "measured_not_pinned" and entry["model"] == "claude-sonnet-5"
        for entry in report
    )


def test_missing_catalog_reads_as_none() -> None:
    """No catalog yet: apply still promotes; composition is skipped."""

    assert read_catalog("/nonexistent/models.catalog.yaml") is None

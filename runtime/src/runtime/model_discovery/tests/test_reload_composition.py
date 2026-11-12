# Field-by-field effective-manifest composition — slice 3. Network-free.

from __future__ import annotations

from runtime.model_discovery.compose import compose_effective_manifest

_OBSERVED = "2026-11-12T00:00:00+00:00"


def _operator_row(**fields: object) -> dict:
    row: dict = {"model_id": "m", "provider": "openai"}
    row.update(fields)
    return row


def _catalog_field(value: object) -> dict:
    return {"source": "provider", "observed_at": _OBSERVED, "value": value}


def _catalog_row(fields: dict) -> dict:
    return {"id": "m", "provider": "openai", "fields": fields}


def _catalog_with(row: dict) -> dict:
    return {
        "schema_version": 1,
        "providers": {"openai": {"models": [row]}},
    }


def test_operator_field_wins_verbatim() -> None:
    operator = {
        "version": 1,
        "models": [_operator_row(prompt_price=2.5)],
    }
    catalog = _catalog_with(_catalog_row({"prompt_price": _catalog_field(1.25)}))
    composed, report = compose_effective_manifest(operator, catalog)
    assert composed["models"][0]["prompt_price"] == 2.5
    assert any(
        entry["kind"] == "operator_field_kept" and entry["field"] == "prompt_price"
        for entry in report
    )


def test_catalog_fills_unset_field_with_observed_at() -> None:
    operator = {"version": 1, "models": [_operator_row()]}
    catalog = _catalog_with(_catalog_row({"context_window": _catalog_field(128000)}))
    composed, report = compose_effective_manifest(operator, catalog)
    assert composed["models"][0]["context_window"] == 128000
    filled = next(
        entry
        for entry in report
        if entry["kind"] == "composed_from_catalog" and entry["field"] == "context_window"
    )
    assert filled["observed_at"] == _OBSERVED


def test_unknown_stays_explicit() -> None:
    """A field published nowhere is absent — never a fabricated default."""

    operator = {"version": 1, "models": [_operator_row()]}
    catalog = _catalog_with(_catalog_row({}))
    composed, _report = compose_effective_manifest(operator, catalog)
    assert "context_window" not in composed["models"][0]
    assert "prompt_price" not in composed["models"][0]


def test_catalog_tightens_operator_ceiling() -> None:
    operator = {"version": 1, "models": [_operator_row(max_completion_tokens=8192)]}
    catalog = _catalog_with(_catalog_row({"max_completion_tokens": _catalog_field(4096)}))
    composed, report = compose_effective_manifest(operator, catalog)
    assert composed["models"][0]["max_completion_tokens"] == 4096
    assert any(entry["kind"] == "ceiling_tightened" for entry in report)


def test_catalog_never_loosens_operator_ceiling() -> None:
    operator = {"version": 1, "models": [_operator_row(max_completion_tokens=4096)]}
    catalog = _catalog_with(_catalog_row({"max_completion_tokens": _catalog_field(999999)}))
    composed, report = compose_effective_manifest(operator, catalog)
    assert composed["models"][0]["max_completion_tokens"] == 4096
    assert any(
        entry["kind"] == "ceiling_not_loosened" and entry["field"] == "max_completion_tokens"
        for entry in report
    )


def test_catalog_context_window_tightens_only() -> None:
    operator = {"version": 1, "models": [_operator_row(context_window=200000)]}
    catalog = _catalog_with(_catalog_row({"context_window": _catalog_field(1000000)}))
    composed, report = compose_effective_manifest(operator, catalog)
    assert composed["models"][0]["context_window"] == 200000
    assert any(
        entry["kind"] == "ceiling_not_loosened" and entry["field"] == "context_window"
        for entry in report
    )


def test_lifecycle_changes_are_reported_not_merged() -> None:
    """Retirement data (expires_at/lifecycle) is surfaced for the operator
    to act on — composition never rewrites the manifest row with it."""

    operator = {"version": 1, "models": [_operator_row()]}
    catalog = _catalog_with(
        _catalog_row(
            {
                "expires_at": _catalog_field("2027-01-01"),
                "lifecycle": _catalog_field("deprecated"),
            }
        )
    )
    composed, report = compose_effective_manifest(operator, catalog)
    assert "expires_at" not in composed["models"][0]
    assert "lifecycle" not in composed["models"][0]
    kinds = {entry["field"] for entry in report if entry["kind"] == "catalog_only_field"}
    assert {"expires_at", "lifecycle"} <= kinds

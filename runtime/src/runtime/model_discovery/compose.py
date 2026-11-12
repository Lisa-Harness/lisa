"""Effective-manifest composition from layered model data (slice 3).

Composes the operator manifest (pins, aliases, tier overrides, mode
routing) with the provider catalog's measured facts into an effective
manifest whose operator-owned content is byte-stable: every field the
operator set is preserved verbatim, and no model the operator pinned is
ever dropped.

Precedence (spec: operator > measured > provider > host_default):

- Operator-set fields win verbatim. An operator ceiling is never
  loosened: a measured/provider ceiling may only tighten it.
- Catalog-published facts fill operator-unset fields (their
  ``observed_at`` rides along as provenance).
- Everything else keeps the manifest schema's host defaults.

Migration report (spec: report, never erase):

- ``measured_not_pinned``: catalog models absent from the operator
  manifest. Reported; never silently added.
- ``pinned_not_measured``: operator-pinned models with no catalog row.
  The operator row is kept as-is.
- ``unestablishable_ownership``: operator fields outside the recognized
  manifest table. Provenance cannot be re-established without guessing,
  and guessing is forbidden.
- ``unknown_catalog_field``: catalog fields outside the recognized
  table. Reported, never merged.
- ``ceiling_not_loosened``: catalog ceiling not lower than the operator
  ceiling; the operator value stays.

Pure function: documents in, composed document + report list out. No
disk I/O; the command layer owns persistence.
"""

from __future__ import annotations

from typing import Any

#: Manifest data fields the catalog may contribute. Operator keys
#: outside this set and the pass-through set are reported, not guessed.
_DATA_FIELDS = frozenset(
    {
        "display_name",
        "context_window",
        "max_completion_tokens",
        "prompt_price",
        "completion_price",
        "cache_read_price",
        "cache_write_price",
        "pricing_currency",
    }
)

#: Operator-owned manifest keys with no catalog counterpart; preserved
#: verbatim and never reported.
_OPERATOR_PASSTHROUGH = frozenset(
    {
        "key",
        "model_id",
        "aliases",
        "display_alias",
        "capabilities",
        "cost_tier",
        "latency_tier",
        "thinking_model",
        "reasoning_model",
        "description",
        "completion_api",
        "structured_reliability",
        "prompt_cache_breakpoints",
        "operator_only",
        "deployment_only",
        "needs_guidance",
        "api_key_env",
        "base_url",
        "base_url_env",
        "model_id_from_config",
        "azure_deploy",
    }
)

#: Catalog fields with no manifest-schema target; reported, never merged.
_CATALOG_ONLY_FIELDS = frozenset({"expires_at", "lifecycle", "supported_parameters", "modalities"})

#: Fields where the catalog may only tighten an operator-set value.
_CEILING_FIELDS = frozenset({"context_window", "max_completion_tokens"})


def _row_fields(row: Any) -> dict[str, Any]:
    """Return a row's ``fields`` mapping (or empty)."""

    fields = row.get("fields") if isinstance(row, dict) else None
    return fields if isinstance(fields, dict) else {}


def _catalog_models_index(catalog: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Index catalog rows by ``(provider, model_id)``."""

    providers = catalog.get("providers")
    if not isinstance(providers, dict):
        return {}
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for provider, block in providers.items():
        models = block.get("models") if isinstance(block, dict) else None
        if not isinstance(models, list):
            continue
        for row in models:
            if isinstance(row, dict) and isinstance(row.get("id"), str) and row["id"]:
                index[(str(provider), row["id"])] = row
    return index


def _numeric(value: Any) -> float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _report_unknown_operator_keys(
    operator_row: dict[str, Any],
    model_id: Any,
    report: list[dict[str, Any]],
) -> None:
    """Surface operator keys outside the composition tables.

    Provenance for such fields cannot be re-established, so they are
    reported (``unestablishable_ownership``) and kept verbatim — never
    dropped, never guessed.
    """

    for key in operator_row:
        if key in ("model_id", "provider"):
            continue
        if key not in _DATA_FIELDS and key not in _OPERATOR_PASSTHROUGH:
            report.append({"kind": "unestablishable_ownership", "model": model_id, "field": key})


def _merge_catalog_field(
    composed: dict[str, Any],
    name: str,
    wrapped: Any,
    model_id: Any,
    report: list[dict[str, Any]],
) -> None:
    """Merge one catalog field into a composed operator row.

    Precedence: the operator value wins verbatim unless the catalog may
    tighten it (ceiling fields only), in which case the lower value
    replaces it and the change is reported. Every merge decision lands
    in the migration report.
    """

    value = _row_value(wrapped)
    if name not in composed:
        observed_at = wrapped.get("observed_at") if isinstance(wrapped, dict) else None
        composed[name] = value
        report.append(
            {
                "kind": "composed_from_catalog",
                "model": model_id,
                "field": name,
                "observed_at": observed_at,
            }
        )
        return
    operator_value = composed[name]
    numeric_operator = _numeric(operator_value)
    numeric_catalog = _numeric(value)
    if name in _CEILING_FIELDS and numeric_operator is not None and numeric_catalog is not None:
        if numeric_catalog < numeric_operator:
            composed[name] = value
            report.append(
                {
                    "kind": "ceiling_tightened",
                    "model": model_id,
                    "field": name,
                    "operator_value": operator_value,
                    "catalog_value": value,
                }
            )
        else:
            report.append(
                {
                    "kind": "ceiling_not_loosened",
                    "model": model_id,
                    "field": name,
                    "operator_value": operator_value,
                    "catalog_value": value,
                }
            )
    else:
        report.append({"kind": "operator_field_kept", "model": model_id, "field": name})


def _compose_model_row(
    operator_row: dict[str, Any],
    catalog_row: dict[str, Any] | None,
    report: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compose one operator row with its catalog row (if any).

    Byte-stability: operator keys are copied verbatim; the catalog only
    fills keys the operator row does not carry, and may only tighten
    operator-set ceilings.
    """

    composed = dict(operator_row)
    model_id = operator_row.get("model_id")
    _report_unknown_operator_keys(operator_row, model_id, report)

    if catalog_row is None:
        report.append(
            {
                "kind": "pinned_not_measured",
                "model": model_id,
                "provider": str(operator_row.get("provider", "")),
            }
        )
        return composed

    for name, wrapped in _row_fields(catalog_row).items():
        if name == "diagnostics":
            continue
        if name in _CATALOG_ONLY_FIELDS:
            report.append({"kind": "catalog_only_field", "model": model_id, "field": name})
            continue
        if name not in _DATA_FIELDS:
            report.append({"kind": "unknown_catalog_field", "model": model_id, "field": name})
            continue
        _merge_catalog_field(composed, name, wrapped, model_id, report)
    return composed


def _row_value(wrapped: Any) -> Any:
    """Unwrap a ``{source, observed_at, value}`` wrapper (or pass through)."""

    if isinstance(wrapped, dict) and "value" in wrapped:
        return wrapped["value"]
    return wrapped


def compose_effective_manifest(
    operator_document: dict[str, Any],
    catalog: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compose the effective manifest from operator + catalog layers.

    Returns ``(composed_document, report)``. The composed document is the
    operator document with each model row enriched inline from the
    catalog under the precedence rules; top-level keys (version,
    mode_models, router) are the operator's and pass through untouched.
    The report is a list of typed entries (see module docstring).
    """

    report: list[dict[str, Any]] = []
    index = _catalog_models_index(catalog)
    rows: list[dict[str, Any]] = []

    operator_rows = operator_document.get("models")
    if isinstance(operator_rows, list):
        for row in operator_rows:
            if not isinstance(row, dict):
                continue
            model_id = row.get("model_id")
            provider = row.get("provider")
            key = (
                (str(provider), str(model_id))
                if model_id is not None and provider is not None
                else None
            )
            catalog_row = index.pop(key, None) if key is not None else None
            rows.append(_compose_model_row(row, catalog_row, report))

    for (provider, model_id), _row in sorted(index.items()):
        report.append({"kind": "measured_not_pinned", "model": model_id, "provider": provider})

    composed = dict(operator_document)
    composed["models"] = rows
    return composed, report

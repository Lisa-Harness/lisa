"""Provider catalog capture, validation, and atomic promotion.

Slice 2 of the MODEL-DATA-LAYERS spec (docs/SPEC/MODEL-DATA-LAYERS.md).

The catalog is the provider-published facts layer, separate from operator
settings (``models.yaml``). Discovery writes a *candidate* catalog to
``models.proposed.yaml``; it never mutates operator settings.

Provenance rules (spec invariants 1 and 4):

- A catalog row contains only fields the provider actually returned.
  Missing means unknown — never zero, false, or a guessed value.
- Every field records its source (``provider`` here; ``operator``,
  ``measured``, and ``host_default`` are composed at reload) and the
  observation time.
- No raw provider JSON, credentials, or request content is retained.

Limit rules (spec probe behavior):

- Provider output limits are per-call ceilings validated against the
  effective context window. Inconsistent data is rejected or omitted
  with a visible diagnostic — never silently clamped.

Atomicity:

- A failed or partial probe run must not replace the last good catalog.
  The candidate is only written after whole-run validation succeeds.
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .types import DiscoveredModel

logger = logging.getLogger(__name__)

#: Schema version of the candidate catalog document.
CATALOG_SCHEMA_VERSION = 1


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def validate_limit_consistency(
    *,
    provider: str,
    model_id: str,
    output_limit: int,
    context_window: int,
) -> str | None:
    """Validate a provider output limit as a per-call ceiling.

    Returns ``None`` when the pair is consistent (or the limit is
    unknown), otherwise a one-line diagnostic naming the provider, the
    model, the published limit, and the effective context window. The
    caller omits the field and logs the diagnostic — it must never clamp
    the published value.
    """

    if output_limit <= 0 or context_window <= 0:
        return None
    if output_limit > context_window:
        return (
            f"{provider}: {model_id} published output ceiling {output_limit} "
            f"exceeds its context window {context_window}; omitting the field "
            f"instead of clamping"
        )
    return None


def _field(source: str, value: Any, observed_at: str) -> dict[str, Any]:
    """Wrap one catalog value with its provenance."""

    return {"source": source, "observed_at": observed_at, "value": value}


#: Provider-published scalars emitted verbatim when truthy — ``0``/``"""
#: means the provider did not publish the field, so it stays omitted
#: (missing = unknown, never a fabricated default).
_SCALAR_FIELDS: tuple[tuple[str, str], ...] = (
    ("display_name", "display_name"),
    ("context_window", "context_window"),
    ("prompt_price", "prompt_price"),
    ("completion_price", "completion_price"),
    ("cache_read_price", "cache_read_price"),
    ("cache_write_price", "cache_write_price"),
    ("lifecycle", "lifecycle"),
)

#: Provider-published sequences emitted as sorted lists when non-empty.
_SEQUENCE_FIELDS: tuple[tuple[str, str], ...] = (
    ("supported_parameters", "supported_parameters"),
    ("modalities", "modalities"),
)

#: Provider-published dates emitted as ISO-8601 strings when present
#: (``None`` = not published — distinct from "never").
_DATE_FIELDS: tuple[tuple[str, str], ...] = (
    ("expires_at", "expires_at"),
    ("deprecated_at", "deprecated_at"),
)


def _capture_output_ceiling(
    model: DiscoveredModel, fields: dict[str, Any], observed_at: str
) -> None:
    """Capture ``max_completion_tokens`` as a validated per-call ceiling.

    Inconsistent published data is omitted with a logged diagnostic —
    never clamped to the context window.
    """

    if not model.max_completion_tokens:
        return
    diagnostic = validate_limit_consistency(
        provider=model.provider,
        model_id=model.model_id,
        output_limit=model.max_completion_tokens,
        context_window=model.context_window,
    )
    if diagnostic is None:
        fields["max_completion_tokens"] = _field(
            "provider", model.max_completion_tokens, observed_at
        )
    else:
        logger.warning("catalog: %s", diagnostic)
        fields.setdefault("diagnostics", []).append(diagnostic)


def _capture_pricing_currency(
    model: DiscoveredModel, fields: dict[str, Any], observed_at: str
) -> None:
    """Record the pricing currency only when prices were published."""

    if model.pricing_currency and (model.prompt_price or model.completion_price):
        fields["pricing_currency"] = _field("provider", model.pricing_currency, observed_at)


def _capture_completion_api(
    model: DiscoveredModel, fields: dict[str, Any], observed_at: str
) -> None:
    """Record the completion API only when it is not the default."""

    if model.completion_api != "chat_completions":
        fields["completion_api"] = _field("provider", model.completion_api, observed_at)


def model_catalog_entry(model: DiscoveredModel, observed_at: str) -> dict[str, Any]:
    """Build one provider catalog row from a discovered model.

    Only provider-published fields appear (``0``/empty sentinels mean the
    provider did not publish and are omitted, keeping unknown explicit).
    Output ceilings are validated against the published context window;
    inconsistent rows are omitted with a logged diagnostic.
    """

    fields: dict[str, Any] = {}
    for field, attr in _SCALAR_FIELDS:
        value = getattr(model, attr)
        if value:
            fields[field] = _field("provider", value, observed_at)
    for field, attr in _SEQUENCE_FIELDS:
        value = getattr(model, attr)
        if value:
            fields[field] = _field("provider", sorted(value), observed_at)
    for field, attr in _DATE_FIELDS:
        value = getattr(model, attr)
        if value is not None:
            fields[field] = _field("provider", value.isoformat(), observed_at)
    _capture_output_ceiling(model, fields, observed_at)
    _capture_pricing_currency(model, fields, observed_at)
    _capture_completion_api(model, fields, observed_at)
    return {"id": model.model_id, "provider": model.provider, "fields": fields}


def build_proposed_catalog(models: tuple[DiscoveredModel, ...]) -> dict[str, Any]:
    """Build the candidate catalog document for a discovery run.

    Pure function: models in, catalog document out. Writing is separate
    so callers can validate the whole run before touching disk.
    """

    observed_at = _utc_now_iso()
    by_provider: dict[str, list[dict[str, Any]]] = {}
    for model in models:
        by_provider.setdefault(model.provider, []).append(model_catalog_entry(model, observed_at))
    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "generated_at": observed_at,
        "providers": {provider: {"models": rows} for provider, rows in sorted(by_provider.items())},
    }


def write_proposed_catalog(path: Any, models: tuple[DiscoveredModel, ...]) -> None:
    """Atomically write the candidate catalog for a successful discovery run.

    Caller contract: only invoke this when the whole run succeeded — a
    failed or partial run must not replace the last good catalog. The
    write is atomic (temp file + ``os.replace``) so a crash mid-write can
    never leave a torn candidate behind.
    """

    document = build_proposed_catalog(models)
    payload = yaml.safe_dump(
        document,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    directory = os.path.dirname(str(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def catalog_path(data_dir: Any) -> Path:
    """Return the provider catalog path inside the host data directory.

    The catalog is the measured/provider layer (``models.catalog.yaml``);
    the operator's edit surface stays ``models.proposed.yaml``.
    """

    return Path(str(data_dir)) / "models.catalog.yaml"


def read_catalog(path: Any) -> dict[str, Any] | None:
    """Read a catalog document, or ``None`` when absent/invalid.

    A missing or malformed catalog is a *reported* condition, never a
    crash: composition callers fall back to the operator document
    unchanged. Structural gate: must be a mapping with a ``providers``
    mapping (the schema's one invariant).
    """

    try:
        text = Path(str(path)).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        logger.warning("catalog: unreadable catalog at %s", path)
        return None
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError:
        logger.warning("catalog: unparseable catalog at %s", path)
        return None
    if not isinstance(document, dict) or not isinstance(document.get("providers"), dict):
        logger.warning("catalog: malformed catalog document at %s", path)
        return None
    return document

"""Pure cache-efficiency metrics over persisted token-usage records.

Workstream 1 of ``docs/SPEC/CACHE-EFFICIENCY.md``: aggregate the durable
usage ledger into honest per-partition cache metrics without ever reading
missing telemetry as a measured zero.

This module is deliberately pure: no filesystem, clock, or ledger access.
Runtime (``runtime.cache_report``) owns reading records, checkpointing,
and rendering; this module owns the *meaning* of a row.

Unknown-versus-zero is the load-bearing rule (the v2/v3 telemetry
contract, ``kernel.token_usage``):

- An absent/``None``/malformed cache counter is *unknown*: it neither
  counts as a hit nor as a zero.
- A zero is an observed zero (a hit-rate denominator) only when the row
  is from the nullable-counter era (``schema_version >= 2``, where the
  writer explicitly preserved provider presence) AND the call site's
  producer chain did not fabricate zeros. ``call_site=operation`` rows
  predate the nullable off-turn producer (``host_adapter`` coerced
  ``int(x or 0)``), so their zeros are untrusted; every other chat call
  site received the gateway's presence-preserving extraction.
- Legacy rows (no ``schema_version``) predate the nullable counters, so
  their zero-valued numerics are ambiguous; such rows never qualify as
  hit-rate denominators. Their known positive sums still ride the token
  share when prompt and cache counters are present and consistent, and
  every ride carries the ``ambiguous_provenance`` flag so the renderer
  can show it beside the totals.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CACHE_HIT_METRIC",
    "MAX_WILSON_HALF_WIDTH",
    "METHOD_VERSION",
    "MIN_ELIGIBLE_SAMPLES",
    "OBSERVATION_WINDOW_DAYS",
    "SNAPSHOT_EXPIRY_DAYS",
    "TOKEN_SHARE_METRIC",
    "CacheMetricsInputError",
    "PartitionKey",
    "PartitionMetrics",
    "UsageRow",
    "aggregate_partition",
    "build_partition_report",
    "classify_usage_record",
    "partition_key_from_record",
    "wilson_half_width",
]

#: Version of the metric method recorded beside every reported figure so a
#: reader can tell which rules produced it. The constants in this module
#: are the proposed initial method (SPEC Workstream 1) — engineering
#: values a caller records, not normative policy facts.
METHOD_VERSION = "cache-metrics/v1"

#: Rolling observation window for a qualified snapshot, in days.
OBSERVATION_WINDOW_DAYS = 14

#: Minimum eligible sample for a qualified snapshot.
MIN_ELIGIBLE_SAMPLES = 30

#: Maximum Wilson 95% half-width for a qualified snapshot.
MAX_WILSON_HALF_WIDTH = 0.15

#: How long a qualified observation stays fresh after its window ends
#: (in days). Exposed so runtime persistence and the renderer quote the
#: same number.
SNAPSHOT_EXPIRY_DAYS = 7

#: Metric names used in measured-layer snapshots and report rows.
CACHE_HIT_METRIC = "cache_request_hit_proxy"
TOKEN_SHARE_METRIC = "cached_token_share"

_Z_95 = 1.959963984540054


class CacheMetricsInputError(ValueError):
    """A record cannot be interpreted under the measurement contract."""


def wilson_half_width(hits: int, total: int) -> float | None:
    """Return the Wilson score 95% half-width for ``hits/total``.

    The half-width is the larger deviation of the interval bounds from
    the point estimate. ``None`` when ``total`` is zero (no eligible
    observations — no interval exists).
    """
    if total <= 0:
        return None
    p = hits / total
    z = _Z_95
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    spread = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    lower, upper = center - spread, center + spread
    return max(p - lower, upper - p)


# ── Row classification ────────────────────────────────────────────────────────


def _record_int(record: Mapping[str, Any], key: str) -> int | None:
    """Read one non-negative integer counter; unknown/malformed is ``None``."""
    value = record.get(key)
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number >= 0 else None


def _record_float(record: Mapping[str, Any], key: str) -> float | None:
    """Read one non-negative float; unknown/malformed is ``None``."""
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        return None
    return number


def _nullable_str(record: Mapping[str, Any], key: str) -> str | None:
    """Read one identifier; empty/absent/non-string is ``None`` (unknown)."""
    value = record.get(key)
    if not isinstance(value, str) or not value:
        return None
    return value


@dataclass(frozen=True, slots=True)
class UsageRow:
    """One usage record reduced to the fields the metrics need.

    ``cache_observation`` is the honest state of the cache-read counter:
    ``"hit"`` (positive value), ``"zero"`` (trusted reported zero),
    ``"untrusted_zero"`` (zero from a producer chain that fabricated
    zeros), or ``"unknown"`` (absent/malformed). ``eligible`` requires a
    chat kind, a trusted cache observation, and a usable prompt.
    """

    provider: str | None
    model_id: str | None
    call_site: str
    kind: str
    schema_version: int | None
    prompt_tokens: int | None
    cached_tokens: int | None
    cache_observation: str
    ambiguous_provenance: bool
    cost_usd: float | None
    provider_cost_usd: float | None
    eligible: bool
    exclusion_reason: str | None = None


_CHAT_KINDS = frozenset({"chat_completion"})


def classify_usage_record(record: Mapping[str, Any]) -> UsageRow:
    """Classify one persisted usage record under the v1 evidence matrix.

    Raises :class:`CacheMetricsInputError` for a non-mapping record (the
    caller counts it as malformed rather than silently skipping it).
    """
    if not isinstance(record, Mapping):
        raise CacheMetricsInputError("usage record is not a mapping")

    kind = str(record.get("kind") or "chat_completion")
    call_site = str(record.get("call_site") or "")
    raw_schema = record.get("schema_version")
    schema_version = (
        raw_schema if isinstance(raw_schema, int) and not isinstance(raw_schema, bool) else None
    )
    prompt = _record_int(record, "prompt_tokens")
    cached = _record_int(record, "cached_tokens")

    # Provenance of the cache counter, decided per row — never assumed.
    ambiguous_provenance = schema_version is None
    if cached is None:
        cache_observation = "unknown"
    elif cached > 0:
        cache_observation = "hit"
    elif schema_version is not None and call_site != "operation":
        # Nullable-counter era, and the producer chain (turn loop,
        # forced-final, and other gateway-fed call sites) preserves
        # presence: a 0 here is a provider-reported zero.
        cache_observation = "zero"
    else:
        # Legacy rows defaulted zeros on write; ``operation`` rows had
        # them fabricated by the pre-nullable off-turn producer.
        cache_observation = "untrusted_zero"

    exclusion: str | None = None
    if kind not in _CHAT_KINDS:
        exclusion = "non_chat"
    elif cache_observation in ("unknown", "untrusted_zero"):
        exclusion = "cache_" + cache_observation
    elif prompt is None:
        exclusion = "prompt_unknown"
    elif prompt <= 0:
        exclusion = "prompt_zero"
    elif cached is not None and cached > prompt:
        exclusion = "inconsistent"

    return UsageRow(
        provider=_nullable_str(record, "provider"),
        model_id=_nullable_str(record, "model_id"),
        call_site=call_site,
        kind=kind,
        schema_version=schema_version,
        prompt_tokens=prompt,
        cached_tokens=cached,
        cache_observation=cache_observation,
        ambiguous_provenance=ambiguous_provenance,
        cost_usd=_record_float(record, "cost_usd"),
        provider_cost_usd=_record_float(record, "provider_cost_usd"),
        eligible=exclusion is None,
        exclusion_reason=exclusion,
    )


# ── Partitioning ─────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PartitionKey:
    """Identity of one measurement partition.

    ``None`` fields are *unknown*, rendered as ``unknown`` and never
    pooled with a known value as if comparable. ``actual_route`` and
    ``prefix_id`` are always ``None`` in this slice: the ledger's
    ``upstream_route`` is a classification rather than observed upstream
    identity, and no prompt-prefix identity exists at all (the cache
    policy fingerprint is not prefix identity).
    """

    provider: str | None
    model_id: str | None
    actual_route: str | None
    cache_policy: str | None
    prefix_id: str | None

    @property
    def identity_known(self) -> bool:
        """True when every qualification dimension is known."""
        return all(
            value is not None
            for value in (self.provider, self.model_id, self.actual_route, self.prefix_id)
        )


def partition_key_from_record(record: Mapping[str, Any]) -> PartitionKey:
    """Build the partition key for one record.

    Only reliably known dimensions are filled; the route and prefix
    dimensions stay unknown because the ledger carries no trustworthy
    value for them in this slice.
    """
    policy = _nullable_str(record, "cache_policy_fingerprint")
    return PartitionKey(
        provider=_nullable_str(record, "provider"),
        model_id=_nullable_str(record, "model_id"),
        # ``upstream_route`` is a classification, not an observed upstream
        # identity — claiming it would fabricate a route.
        actual_route=None,
        cache_policy=policy,
        # No prompt-prefix identity exists anywhere in the ledger yet.
        prefix_id=None,
    )


# ── Aggregation ──────────────────────────────────────────────────────────────


@dataclass(slots=True)
class PartitionMetrics:
    """Accumulated cache metrics for one partition.

    Counts and sums keep their denominators visible; unknown data never
    becomes a zero-valued observation. ``token_share`` is ``None`` when
    the prompt denominator is zero, and ``wilson_half_width`` is ``None``
    with no eligible samples.
    """

    eligible_count: int = 0
    hits: int = 0
    observed_zero_count: int = 0
    excluded_counts: dict[str, int] = field(default_factory=dict)
    prompt_sum: int = 0
    cached_sum: int = 0
    ambiguous_rows: int = 0
    cost_known_count: int = 0
    cost_sum: float = 0.0
    provider_cost_known_count: int = 0
    provider_cost_sum: float = 0.0
    row_count: int = 0

    @property
    def token_share(self) -> float | None:
        """``sum(cached) / sum(prompt)`` over eligible rows, or ``None``."""
        if self.prompt_sum <= 0:
            return None
        return self.cached_sum / self.prompt_sum

    @property
    def request_hit_rate(self) -> float | None:
        """Count of cached>0 rows over eligible rows, or None with no rows."""
        if self.eligible_count <= 0:
            return None
        return self.hits / self.eligible_count

    @property
    def wilson_half_width(self) -> float | None:
        return wilson_half_width(self.hits, self.eligible_count)


def aggregate_partition(rows: list[UsageRow]) -> PartitionMetrics:
    """Fold classified rows into one partition's metrics."""
    metrics = PartitionMetrics()
    for row in rows:
        metrics.row_count += 1
        if row.cost_usd is not None:
            metrics.cost_known_count += 1
            metrics.cost_sum += row.cost_usd
        if row.provider_cost_usd is not None:
            metrics.provider_cost_known_count += 1
            metrics.provider_cost_sum += row.provider_cost_usd
        if row.ambiguous_provenance and row.eligible:
            metrics.ambiguous_rows += 1
        if row.eligible:
            metrics.eligible_count += 1
            if row.cached_tokens and row.cached_tokens > 0:
                metrics.hits += 1
            else:
                metrics.observed_zero_count += 1
            metrics.prompt_sum += row.prompt_tokens or 0
            metrics.cached_sum += row.cached_tokens or 0
        elif row.exclusion_reason:
            metrics.excluded_counts[row.exclusion_reason] = (
                metrics.excluded_counts.get(row.exclusion_reason, 0) + 1
            )
    return metrics


@dataclass(frozen=True, slots=True)
class PartitionReport:
    """One partition's metrics plus its qualification status.

    ``status`` is ``"qualified"`` only when every qualification dimension
    is known AND the sample/half-width gates pass AND the source was
    complete. Callers pass ``source_complete=False`` (rotation, malformed
    lines, unstable capture) which forces ``"incomplete"`` — never a
    qualified observation over unproven data.
    """

    key: PartitionKey
    metrics: PartitionMetrics
    status: str
    gate_failures: tuple[str, ...]
    method_version: str = METHOD_VERSION


def build_partition_report(
    key: PartitionKey,
    metrics: PartitionMetrics,
    *,
    source_complete: bool = True,
) -> PartitionReport:
    """Attach gates and qualification status to one partition's metrics."""
    failures: list[str] = []
    if not key.identity_known:
        failures.append("identity_unknown")
    if metrics.eligible_count < MIN_ELIGIBLE_SAMPLES:
        failures.append("min_samples")
    half_width = metrics.wilson_half_width
    if half_width is None or half_width > MAX_WILSON_HALF_WIDTH:
        failures.append("wilson_half_width")
    if not source_complete:
        failures.append("source_incomplete")
    status = "unqualified" if failures else "qualified"
    return PartitionReport(
        key=key,
        metrics=metrics,
        status=status,
        gate_failures=tuple(failures),
    )

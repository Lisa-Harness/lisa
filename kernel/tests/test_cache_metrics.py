"""Workstream 1 honest cache measurement — pure kernel metrics.

Covers the row-evidence matrix (trusted zero vs fabricated zero vs
unknown), the two separate metrics, Wilson gating, and partitioning
rules where route/prefix stay unknown.
"""

from __future__ import annotations

import pytest
from kernel.cache_metrics import (
    MAX_WILSON_HALF_WIDTH,
    MIN_ELIGIBLE_SAMPLES,
    CacheMetricsInputError,
    PartitionKey,
    aggregate_partition,
    build_partition_report,
    classify_usage_record,
    partition_key_from_record,
)


def _row(**overrides):
    record = {
        "kind": "chat_completion",
        "provider": "openai",
        "model_id": "gpt-test",
        "call_site": "turn_loop",
        "schema_version": 3,
        "prompt_tokens": 100,
        "cached_tokens": 40,
    }
    record.update(overrides)
    return classify_usage_record(record)


# ── Row classification ─────────────────────────────────────────────────────────


def test_hit_positive_counter_is_eligible() -> None:
    row = _row(cached_tokens=40)
    assert row.eligible
    assert row.cache_observation == "hit"


def test_zero_trusted_on_nullable_era_turn_loop() -> None:
    row = _row(cached_tokens=0)
    assert row.eligible
    assert row.cache_observation == "zero"


def test_zero_untrusted_for_legacy_rows() -> None:
    row = _row(cached_tokens=0, schema_version=None)
    assert not row.eligible
    assert row.exclusion_reason == "cache_untrusted_zero"
    assert row.ambiguous_provenance


def test_zero_untrusted_for_operation_call_site() -> None:
    """The pre-Workstream-1 off-turn producer fabricated zeros."""
    row = _row(cached_tokens=0, call_site="operation")
    assert not row.eligible
    assert row.exclusion_reason == "cache_untrusted_zero"


def test_missing_cache_counter_is_unknown_never_zero() -> None:
    row = _row(cached_tokens=None)
    assert not row.eligible
    assert row.cache_observation == "unknown"
    assert row.exclusion_reason == "cache_unknown"


def test_non_chat_rows_excluded() -> None:
    row = _row(kind="embedding")
    assert not row.eligible
    assert row.exclusion_reason == "non_chat"


def test_zero_prompt_is_not_a_hit_denominator() -> None:
    row = _row(prompt_tokens=0, cached_tokens=0)
    assert not row.eligible
    assert row.exclusion_reason == "prompt_zero"


def test_inconsistent_row_excluded() -> None:
    row = _row(prompt_tokens=10, cached_tokens=20)
    assert not row.eligible
    assert row.exclusion_reason == "inconsistent"


def test_malformed_counters_are_unknown() -> None:
    row = _row(cached_tokens="garbage", prompt_tokens=-5)
    assert not row.eligible
    assert row.cache_observation == "unknown"


def test_non_mapping_record_raises() -> None:
    with pytest.raises(CacheMetricsInputError):
        classify_usage_record("not a record")  # type: ignore[arg-type]


# ── Metrics and gating ──────────────────────────────────────────────────────────


def test_metrics_separate_hit_rate_and_token_share() -> None:
    rows = [
        _row(cached_tokens=100, prompt_tokens=100),  # hit, 100%
        _row(cached_tokens=0, prompt_tokens=300),  # zero, 0%
        _row(cached_tokens=25, prompt_tokens=100),  # hit, 25%
    ]
    metrics = aggregate_partition(rows)
    assert metrics.eligible_count == 3
    assert metrics.hits == 2
    assert metrics.observed_zero_count == 1
    assert metrics.request_hit_rate == pytest.approx(2 / 3)
    assert metrics.token_share == pytest.approx(125 / 500)
    assert metrics.wilson_half_width is not None


def test_wilson_boundary_at_thirty_samples() -> None:
    """29 samples fail the min-samples gate; 30 can pass."""
    rows = [_row(cached_tokens=1, prompt_tokens=10) for _ in range(29)]
    key = PartitionKey("p", "m", None, None, None)
    report = build_partition_report(key, aggregate_partition(rows))
    assert report.status == "unqualified"
    assert "min_samples" in report.gate_failures

    rows.append(_row(cached_tokens=1, prompt_tokens=10))
    metrics = aggregate_partition(rows)
    assert metrics.eligible_count == MIN_ELIGIBLE_SAMPLES
    # All-hits keeps the Wilson half-width under the gate.
    assert metrics.wilson_half_width < MAX_WILSON_HALF_WIDTH
    assert build_partition_report(key, metrics).status == "unqualified"  # identity unknown


def test_identity_unknown_blocks_qualification() -> None:
    rows = [_row(cached_tokens=1, prompt_tokens=10) for _ in range(35)]
    key = PartitionKey("p", "m", None, None, None)  # route/prefix unknown
    report = build_partition_report(key, aggregate_partition(rows))
    assert report.status == "unqualified"
    assert "identity_unknown" in report.gate_failures


def test_incomplete_source_blocks_qualification() -> None:
    """A perfectly-gated partition over an unproven source stays unqualified."""
    rows = [_row(cached_tokens=1, prompt_tokens=10) for _ in range(35)]
    key = PartitionKey("p", "m", "direct", "pol", "prefix-1")
    report = build_partition_report(key, aggregate_partition(rows), source_complete=False)
    assert report.status == "unqualified"
    assert "source_incomplete" in report.gate_failures


def test_fully_known_and_gated_partition_qualifies() -> None:
    rows = [_row(cached_tokens=1, prompt_tokens=10) for _ in range(35)]
    key = PartitionKey("p", "m", "direct", "pol", "prefix-1")
    report = build_partition_report(key, aggregate_partition(rows))
    assert report.status == "qualified"
    assert report.gate_failures == ()


def test_unknown_dimensions_render_not_pooled() -> None:
    """Route/prefix stay unknown; the record cannot supply them."""
    key = partition_key_from_record(
        {"provider": "openai", "model_id": "gpt-test", "upstream_route": "kernel-published"}
    )
    assert key.actual_route is None
    assert key.prefix_id is None
    assert key.cache_policy is None
    assert not key.identity_known

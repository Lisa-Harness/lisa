"""Baseline replay tests (SPEC \u00a77 criterion 1).

Replays fixed ledgers \u2014 legacy, incomplete, and malformed rows included \u2014
through the same aggregation readers the live ``/tokens`` section uses,
asserting the reproduced aggregates, the coverage/exclusion counts shown
beside them, and the aggregate-only output contract (no raw rows, no
prompt text).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from runtime.cache_replay import (
    ReplayOptions,
    main,
    render_replay_report,
    replay_baseline,
    replay_json,
)

_NOW = datetime(2026, 11, 13, 12, 0, 0, tzinfo=UTC)
_M1 = ("openai", "m1", None, None, None)
_M2 = ("anthropic", "m2", None, None, None)


def _ts(days_ago: float) -> str:
    return (_NOW - timedelta(days=days_ago)).isoformat().replace("+00:00", "Z")


def _record(
    *,
    cached: int | None = 40,
    prompt: int = 100,
    model: str = "m1",
    provider: str = "openai",
    schema: int | None = 3,
    call_site: str = "turn_loop",
    days_ago: float = 0.0,
    kind: str = "chat_completion",
    key: str = "",
    **extra: object,
) -> dict:
    record = {
        "emitted_at": _ts(days_ago),
        "kind": kind,
        "provider": provider,
        "model_id": model,
        "call_site": call_site,
        "schema_version": schema,
        "prompt_tokens": prompt,
        "cached_tokens": cached,
    }
    if key:
        record["idempotency_key"] = key
    record.update(extra)
    return record


def _write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def test_replay_reproduces_partition_aggregates(tmp_path: Path) -> None:
    ledger = tmp_path / "token-usage.jsonl"
    _write(
        ledger,
        [
            _record(key="a", cached=40, prompt=100, model="m1", days_ago=1),
            _record(key="b", cached=0, prompt=50, model="m1", days_ago=2),
            _record(key="c", cached=10, prompt=200, model="m2", provider="anthropic", days_ago=3),
        ],
    )
    replay = replay_baseline(ledger, options=ReplayOptions(as_of=_NOW))
    assert replay.status == "ok"
    assert replay.source_complete
    assert set(replay.partitions) == {_M1, _M2}
    m1 = replay.partitions[_M1]
    assert (m1.eligible_count, m1.hits) == (2, 1)
    assert (m1.prompt_sum, m1.cached_sum) == (150, 40)
    m2 = replay.partitions[_M2]
    assert (m2.eligible_count, m2.prompt_sum, m2.cached_sum) == (1, 200, 10)
    payload = replay_json(replay)
    assert payload["partitions"]["openai/m1/policy=unknown"]["cached_sum"] == 40
    assert payload["partitions"]["anthropic/m2/policy=unknown"]["prompt_sum"] == 200


def test_replay_full_scan_ignores_rolling_cutoff(tmp_path: Path) -> None:
    """A baseline replays the whole ledger; the live view rolls 14 days."""
    ledger = tmp_path / "token-usage.jsonl"
    _write(ledger, [_record(key="old", days_ago=40), _record(key="new", days_ago=1)])
    full = replay_baseline(ledger, options=ReplayOptions(as_of=_NOW))
    (metrics,) = full.partitions.values()
    assert metrics.eligible_count == 2
    windowed = replay_baseline(ledger, options=ReplayOptions(as_of=_NOW, window_days=14))
    (windowed_metrics,) = windowed.partitions.values()
    assert windowed_metrics.eligible_count == 1


def test_replay_excludes_future_rows_with_diagnostic(tmp_path: Path) -> None:
    ledger = tmp_path / "token-usage.jsonl"
    future = _record(key="f")
    future["emitted_at"] = (_NOW + timedelta(days=2)).isoformat().replace("+00:00", "Z")
    _write(ledger, [future, _record(key="ok")])
    replay = replay_baseline(ledger, options=ReplayOptions(as_of=_NOW))
    (metrics,) = replay.partitions.values()
    assert metrics.eligible_count == 1
    assert replay.diagnostics.get("future_timestamp_rows") == 1


def test_replay_counts_malformed_and_bad_utf8_as_incomplete(tmp_path: Path) -> None:
    """Old/incomplete/malformed rows are counted and shown, never zeroed."""
    ledger = tmp_path / "token-usage.jsonl"
    good = json.dumps(_record(key="a")).encode()
    ledger.write_bytes(good + b"\n{not json}\n\xff\xfe bad\n")
    replay = replay_baseline(ledger, options=ReplayOptions(as_of=_NOW))
    assert not replay.source_complete
    assert replay.diagnostics.get("malformed_lines") == 1
    assert replay.diagnostics.get("invalid_utf8_lines") == 1
    (metrics,) = replay.partitions.values()
    assert metrics.eligible_count == 1
    text = "\n".join(render_replay_report(replay))
    assert "incomplete source" in text
    assert "malformed_lines=1" in text


def test_replay_legacy_and_offturn_zero_rows_are_excluded(tmp_path: Path) -> None:
    """Ambiguous legacy zeros and fabricated off-turn zeros are coverage."""
    ledger = tmp_path / "token-usage.jsonl"
    _write(
        ledger,
        [
            _record(key="l", schema=None, cached=0),
            _record(key="o", call_site="operation", cached=0),
            _record(key="t", cached=0),
        ],
    )
    replay = replay_baseline(ledger, options=ReplayOptions(as_of=_NOW))
    buckets = replay.excluded["openai/m1/policy=unknown"]
    assert buckets.get("cache_untrusted_zero") == 2
    (metrics,) = replay.partitions.values()
    assert (metrics.eligible_count, metrics.observed_zero_count) == (1, 1)
    text = "\n".join(render_replay_report(replay))
    assert "cache_untrusted_zero:2" in text


def test_replay_missing_ledger_is_unavailable(tmp_path: Path) -> None:
    replay = replay_baseline(tmp_path / "absent.jsonl")
    assert replay.status == "unavailable"
    assert "unavailable" in "\n".join(render_replay_report(replay))


def test_replay_report_is_aggregate_only(tmp_path: Path) -> None:
    """The report shows partition labels and aggregates, never raw rows."""
    ledger = tmp_path / "token-usage.jsonl"
    _write(ledger, [_record(key="a", model="secret-model-name", prompt=777)])
    text = "\n".join(render_replay_report(replay_baseline(ledger)))
    assert "secret-model-name" in text  # partition label
    assert "prompt_tokens" not in text  # raw field names never appear
    assert "emitted_at" not in text
    # With one row the aggregate legitimately equals the row value (40/777);
    # the privacy contract is no raw records or per-row decomposition, so
    # use two rows and assert no per-row intermediate appears.
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(_record(key="b", model="secret-model-name", prompt=333, cached=11)) + "\n"
        )
    text = "\n".join(render_replay_report(replay_baseline(ledger)))
    assert "1,110/333,000" not in text  # never decomposed per row
    assert "40/777" not in text  # the single-row breakdown is gone


def test_cli_json_roundtrip(tmp_path: Path, capsys) -> None:
    ledger = tmp_path / "token-usage.jsonl"
    _write(ledger, [_record(key="a")])
    assert main([str(ledger), "--json", "--as-of", _ts(0)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["partitions"]["openai/m1/policy=unknown"]["eligible_count"] == 1


def test_cli_rejects_invalid_as_of(tmp_path: Path) -> None:
    ledger = tmp_path / "token-usage.jsonl"
    _write(ledger, [_record(key="a")])
    assert main([str(ledger), "--as-of", "nonsense"]) == 2

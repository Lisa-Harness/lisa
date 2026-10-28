"""Tests for the durable TokenUsageTelemetryHandler."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from kernel.token_usage import TokenUsageTelemetryHandler, build_token_usage_event


@pytest.mark.asyncio
async def test_handler_appends_event_as_jsonl_line() -> None:
    with TemporaryDirectory() as td:
        path = Path(td) / "lsa-state" / "token-usage.jsonl"
        handler = TokenUsageTelemetryHandler(path)

        event = build_token_usage_event(
            provider="ollama",
            model_id="llama3.2:latest",
            prompt_tokens=120,
            completion_tokens=30,
            cached_tokens=10,
            reasoning_tokens=5,
            duration_ms=450.0,
            latency_ms=450,
            call_site="provider_gateway.openai_compatible",
        )
        await handler.publish(event)

        assert path.exists()
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["event_type"] == "token_used"
        assert record["provider"] == "ollama"
        assert record["model_id"] == "llama3.2:latest"
        assert record["prompt_tokens"] == 120
        assert record["completion_tokens"] == 30
        assert record["total_tokens"] == 150
        assert record["cached_tokens"] == 10
        assert record["reasoning_tokens"] == 5
        assert record["duration_ms"] == 450.0
        assert record["latency_ms"] == 450
        assert record["call_site"] == "provider_gateway.openai_compatible"
        # Idempotency key is stamped when empty.
        assert record["idempotency_key"]


@pytest.mark.asyncio
async def test_handler_appends_multiple_events_without_collapsing() -> None:
    with TemporaryDirectory() as td:
        path = Path(td) / "token-usage.jsonl"
        handler = TokenUsageTelemetryHandler(path)

        for i in range(3):
            event = build_token_usage_event(
                provider="openai",
                model_id="gpt-4o",
                prompt_tokens=i * 10,
                completion_tokens=5,
            )
            await handler.publish(event)

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        # Each line gets a unique idempotency key.
        keys = {json.loads(line)["idempotency_key"] for line in lines}
        assert len(keys) == 3


@pytest.mark.asyncio
async def test_idempotency_keys_survive_a_frozen_wall_clock(monkeypatch) -> None:
    """Uniqueness must not depend on wall-clock tick granularity.

    Regression (2026-10-13 windows CI): Windows ticks ``time.time_ns`` at
    ~15.6ms, so three events published within one tick all received the
    same ``emitted_at:time_ns`` key and collapsed onto one dedupe identity
    (``assert 1 == 3``). A zero-progress clock must still produce strictly
    increasing suffixes.
    """
    from kernel import token_usage

    monkeypatch.setattr(token_usage.time, "time_ns", lambda: 500_000_000)
    monkeypatch.setattr(token_usage, "_time_ns_suffix", token_usage._time_ns_suffix)
    monkeypatch.setattr(token_usage, "_LAST_IDEMPOTENCY_SUFFIX", -1)

    first = token_usage._time_ns_suffix()
    second = token_usage._time_ns_suffix()
    third = token_usage._time_ns_suffix()

    assert len({first, second, third}) == 3


@pytest.mark.asyncio
async def test_handler_ignores_non_token_usage_events() -> None:
    with TemporaryDirectory() as td:
        path = Path(td) / "token-usage.jsonl"
        handler = TokenUsageTelemetryHandler(path)

        # Publish a bare dict (not a TokenUsageEvent).
        await handler.publish({"event_type": "something_else"})

        assert not path.exists()


@pytest.mark.asyncio
async def test_handler_filter_fn_skips_events() -> None:
    with TemporaryDirectory() as td:
        path = Path(td) / "token-usage.jsonl"
        handler = TokenUsageTelemetryHandler(path, filter_fn=lambda e: e.prompt_tokens > 0)

        # Event with zero prompt_tokens is filtered out.
        await handler.publish(
            build_token_usage_event(provider="ollama", model_id="x", prompt_tokens=0)
        )
        # Event with prompt_tokens > 0 is persisted.
        await handler.publish(
            build_token_usage_event(provider="ollama", model_id="y", prompt_tokens=10)
        )

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["model_id"] == "y"


@pytest.mark.asyncio
async def test_handler_swallows_os_errors() -> None:
    # Point the handler at a path whose parent is a file (not a dir),
    # so mkdir fails. The handler must swallow the OSError and return.
    with TemporaryDirectory() as td:
        blocker = Path(td) / "blocker"
        blocker.write_text("i am a file, not a directory", encoding="utf-8")
        path = blocker / "token-usage.jsonl"

        handler = TokenUsageTelemetryHandler(path)
        # Must not raise.
        await handler.publish(
            build_token_usage_event(provider="ollama", model_id="x", prompt_tokens=1)
        )


def test_build_token_usage_event_defaults_total_tokens() -> None:
    event = build_token_usage_event(
        provider="openai",
        model_id="gpt-4o",
        prompt_tokens=100,
        completion_tokens=50,
    )
    assert event.total_tokens == 150
    assert event.kind == "chat_completion"
    assert event.duration_ms == 0.0
    assert event.time_cost_usd == 0.0


def test_build_token_usage_event_supports_time_billed_kind() -> None:
    event = build_token_usage_event(
        provider="ollama",
        model_id="llama3.2:latest",
        kind="time_billed",
        duration_ms=5000.0,
        time_cost_usd=0.001,
        prompt_tokens=200,  # still populated for observability
        completion_tokens=80,
    )
    assert event.kind == "time_billed"
    assert event.duration_ms == 5000.0
    assert event.time_cost_usd == 0.001
    assert event.prompt_tokens == 200


@pytest.mark.asyncio
async def test_reader_skips_invalid_json_lines() -> None:
    """A malformed line in the JSONL file does not break a direct read."""
    with TemporaryDirectory() as td:
        path = Path(td) / "token-usage.jsonl"
        handler = TokenUsageTelemetryHandler(path)

        # Write one valid event
        await handler.publish(
            build_token_usage_event(provider="openai", model_id="valid", prompt_tokens=10)
        )
        # Append one invalid line directly
        with path.open("a", encoding="utf-8") as f:
            f.write("not valid json\n")
        # Write another valid event
        await handler.publish(
            build_token_usage_event(provider="openai", model_id="valid2", prompt_tokens=10)
        )

        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                decoded = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(decoded, dict):
                records.append(decoded)
        # Two valid records, one invalid line skipped
        assert len(records) == 2
        assert records[0]["model_id"] == "valid"
        assert records[1]["model_id"] == "valid2"


# ── Canonical cost model ──────────────────────────────────────────────


def test_estimate_call_cost_prices_uncached_prompt_and_completion() -> None:
    """With no cache counters the cost is plain prompt + completion rates."""
    from kernel.token_usage import estimate_call_cost_usd

    cost = estimate_call_cost_usd(
        prompt_tokens=1000,
        completion_tokens=100,
        prompt_price=0.000002,
        completion_price=0.00001,
    )
    assert cost == pytest.approx(1000 * 0.000002 + 100 * 0.00001)


def test_estimate_call_cost_splits_cache_reads_and_writes() -> None:
    """Cache reads and writes bill at their own rates, not the prompt rate.

    A cache-heavy agentic call is mostly cache reads, which providers bill at
    a large discount. Pricing them at the ordinary prompt rate is what made
    the CLI footer overstate OpenRouter turns by up to 10x.
    """

    from kernel.token_usage import estimate_call_cost_usd

    cost = estimate_call_cost_usd(
        prompt_tokens=100_000,
        completion_tokens=200,
        cached_tokens=90_000,
        cache_creation_tokens=9_000,
        prompt_price=0.000002,
        completion_price=0.00001,
        cache_read_price=0.0000001,
        cache_write_price=0.0000025,
    )
    expected = (
        1_000 * 0.000002  # uncached remainder
        + 90_000 * 0.0000001  # cache reads
        + 9_000 * 0.0000025  # cache writes
        + 200 * 0.00001  # completion
    )
    assert cost == pytest.approx(expected)


def test_estimate_call_cost_falls_back_to_prompt_rate_without_cache_prices() -> None:
    """An unpriced cache rate does not make those tokens free."""
    from kernel.token_usage import estimate_call_cost_usd

    cost = estimate_call_cost_usd(
        prompt_tokens=1000,
        completion_tokens=0,
        cached_tokens=600,
        prompt_price=0.000002,
    )
    assert cost == pytest.approx(1000 * 0.000002)


def test_estimate_call_cost_clamps_overreported_cache_counters() -> None:
    """Cache counters larger than the prompt cannot produce a negative cost."""
    from kernel.token_usage import estimate_call_cost_usd

    cost = estimate_call_cost_usd(
        prompt_tokens=100,
        completion_tokens=0,
        cached_tokens=80,
        cache_creation_tokens=500,
        prompt_price=0.000001,
        cache_write_price=0.000001,
    )
    # 80 cached at 1e-6 (read falls back to prompt rate) + 20 clamped writes.
    assert cost == pytest.approx(100 * 0.000001)
    assert cost >= 0.0


def test_estimate_call_cost_is_zero_when_provider_publishes_no_prices() -> None:
    """Unpriced models (local Ollama) report no cost rather than a fake one."""
    from kernel.token_usage import estimate_call_cost_usd

    assert estimate_call_cost_usd(prompt_tokens=5000, completion_tokens=500) == 0.0


@pytest.mark.asyncio
async def test_event_persists_provider_reported_cost() -> None:
    """The provider's own charge rides beside the canonical estimate."""
    with TemporaryDirectory() as td:
        path = Path(td) / "token-usage.jsonl"
        handler = TokenUsageTelemetryHandler(path)
        event = build_token_usage_event(
            provider="openrouter",
            model_id="openai/gpt-sol",
            prompt_tokens=1000,
            completion_tokens=100,
            cost_usd=0.0031,
            provider_cost_usd=0.0029,
        )
        await handler.publish(event)
        record = json.loads(path.read_text(encoding="utf-8").strip())
        assert record["cost_usd"] == pytest.approx(0.0031)
        assert record["provider_cost_usd"] == pytest.approx(0.0029)


# ── Proxy response telemetry ───────────────────────────────────────────


def _telemetry_spec(**kwargs: object) -> object:
    from kernel.proxy_launch.policy import ProxyTelemetrySpec

    return ProxyTelemetrySpec(**kwargs)  # type: ignore[arg-type]


def test_extract_proxy_telemetry_reads_declared_headers() -> None:
    """Only declared headers are lifted, under the pack's field names."""
    from kernel.provider_contracts import extract_proxy_telemetry

    spec = _telemetry_spec(
        headers=(("x-acme-saved", "tokens_saved"), ("x-acme-before", "tokens_before")),
        savings_field="tokens_saved",
    )
    lifted = extract_proxy_telemetry(
        {"x-acme-saved": "40000", "x-acme-before": "140000", "x-other": "999"}, spec
    )
    assert lifted == {"tokens_saved": 40000, "tokens_before": 140000}


def test_extract_proxy_telemetry_is_case_insensitive_for_plain_dicts() -> None:
    """A proxy writing different casing still matches its declaration."""
    from kernel.provider_contracts import extract_proxy_telemetry

    spec = _telemetry_spec(headers=(("X-Acme-Saved", "tokens_saved"),))
    assert extract_proxy_telemetry({"x-acme-SAVED": "10"}, spec) == {"tokens_saved": 10}


def test_extract_proxy_telemetry_accepts_httpx_headers() -> None:
    """Real ``httpx.Headers`` from a proxied response works unchanged."""
    import httpx
    from kernel.provider_contracts import extract_proxy_telemetry

    request = httpx.Request("POST", "http://127.0.0.1:8787/v1/chat/completions")
    response = httpx.Response(200, headers={"x-acme-saved": "2500"}, request=request)
    spec = _telemetry_spec(headers=(("x-acme-saved", "tokens_saved"),))
    assert extract_proxy_telemetry(response.headers, spec) == {"tokens_saved": 2500}


def test_extract_proxy_telemetry_omits_unreported_counters() -> None:
    """A missing header means "not reported", never a fabricated zero.

    A proxy that skipped compression omits the header, and a caller must be
    able to distinguish that from "saved zero tokens".
    """
    from kernel.provider_contracts import extract_proxy_telemetry

    spec = _telemetry_spec(
        headers=(("x-acme-saved", "tokens_saved"), ("x-acme-before", "tokens_before"))
    )
    assert extract_proxy_telemetry({"x-acme-before": "1000"}, spec) == {"tokens_before": 1000}


def test_extract_proxy_telemetry_drops_malformed_values() -> None:
    """Garbage from a third-party proxy degrades to "no counter"."""
    from kernel.provider_contracts import extract_proxy_telemetry

    spec = _telemetry_spec(headers=(("x-acme-saved", "tokens_saved"),))
    assert extract_proxy_telemetry({"x-acme-saved": "not-a-number"}, spec) == {}
    assert extract_proxy_telemetry({"x-acme-saved": ""}, spec) == {}
    assert extract_proxy_telemetry({"x-acme-saved": "-5"}, spec) == {"tokens_saved": 0}


def test_extract_proxy_telemetry_without_declaration_is_empty() -> None:
    """No declared telemetry means no header reading at all."""
    from kernel.provider_contracts import extract_proxy_telemetry

    assert extract_proxy_telemetry({"x-acme-saved": "10"}, None) == {}
    assert extract_proxy_telemetry(None, _telemetry_spec(headers=(("a", "b"),))) == {}

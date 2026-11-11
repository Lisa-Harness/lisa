"""Slice C boot-stage instrumentation: telemetry helpers and their wiring."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from runtime.startup_telemetry import (
    boot_invocation_id,
    seconds_since_boot,
    stage,
    warn_if_repeat_setup,
)


def _events(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [
        json.loads(record.getMessage().removeprefix("startup "))
        for record in caplog.records
        if record.getMessage().startswith("startup ")
    ]


@pytest.fixture(autouse=True)
def _fresh_telemetry_state(monkeypatch: pytest.MonkeyPatch) -> None:
    import runtime.startup_telemetry as telemetry

    monkeypatch.setattr(telemetry, "_BOOT_INVOCATION_ID", None)
    monkeypatch.setattr(telemetry, "_PREVIOUS_BOOT_INVOCATION_ID", None)
    monkeypatch.setattr(telemetry, "_BOOT_SETUP_COUNT", 0)


def _config(**overrides: Any) -> Any:
    """Build a real ``HostConfig`` via ``from_env`` with overrides (mirrors test_bootstrap)."""
    from runtime.config import HostConfig

    config = HostConfig.from_env()
    return replace(config, **overrides) if overrides else config


def _install_fakes(monkeypatch: pytest.MonkeyPatch, *, config: Any) -> None:
    """Install the eager-wiring fakes ``build_runtime`` needs (mirrors test_bootstrap)."""
    fake_runtime_cls = lambda cfg, in_process_mcp=False: SimpleNamespace(  # noqa: E731
        config=cfg,
        in_process_mcp=in_process_mcp,
        session=SimpleNamespace(state_snapshot=None),
        resources=SimpleNamespace(
            set_mcp_configs=lambda w, c: None,
            set_proxy_configs=lambda p: None,
        ),
    )

    monkeypatch.setattr("runtime.mcp_pool.compile_mcp_configs", lambda r, discovery=None: ([], []))
    monkeypatch.setattr("runtime.mcp_pool.compile_proxy_configs", lambda r, discovery=None: [])
    monkeypatch.setattr("runtime.mcp_pool.configure_runtime", lambda r: None)

    # Production-faithful fakes: mcp_pool instruments these two stages
    # itself (stage() inside discover_framework/setup_framework), so the
    # fakes wrap the same way. A bare lambda here would let an outer
    # stage() wrapper in bootstrap.py reintroduce nested duplicate spans
    # (9 begins instead of 7) without this sequence test noticing.
    def _fake_discover_framework(runtime: Any) -> object:
        with stage("discover_framework"):
            return object()

    def _fake_setup_framework(runtime: Any, discovery: Any = None) -> None:
        with stage("setup_framework"):
            return

    monkeypatch.setattr("runtime.mcp_pool.discover_framework", _fake_discover_framework)
    monkeypatch.setattr("runtime.mcp_pool.setup_framework", _fake_setup_framework)
    monkeypatch.setattr("runtime.runtime.HostRuntime", fake_runtime_cls)
    monkeypatch.setattr(
        "runtime.state_provider.load_state_snapshot",
        lambda *, explicit_path, data_dir: {"source": explicit_path},
    )
    monkeypatch.setattr(
        "runtime.content_roots.discover_pack_redaction_patterns",
        lambda data_dir, always_on_packs: [],
    )
    monkeypatch.setattr("runtime.thread.set_thread_redaction_policy", lambda policy: None)
    monkeypatch.setattr("runtime.host_commands.register_builtin_host_commands", lambda: None)
    monkeypatch.setattr("runtime.config.HostConfig.from_env", classmethod(lambda cls, **kw: config))


def test_stage_logs_begin_end_with_monotonic_duration(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="lisa.startup"), stage("discover_framework"):
        pass

    events = _events(caplog)
    assert [event["event"] for event in events] == ["begin", "end"]
    assert events[0]["stage"] == events[1]["stage"] == "discover_framework"
    assert events[0]["boot_id"] == events[1]["boot_id"] == boot_invocation_id()
    assert events[1]["seconds"] >= 0.0
    assert events[0]["pid"] == events[1]["pid"]


def test_stage_logs_error_with_duration_and_reraises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        caplog.at_level(logging.INFO, logger="lisa.startup"),
        pytest.raises(RuntimeError, match="boom"),
        stage("setup_framework"),
    ):
        raise RuntimeError("boom")

    events = _events(caplog)
    assert [event["event"] for event in events] == ["begin", "error"]
    assert events[1]["seconds"] >= 0.0


def test_stage_payload_carries_no_private_content(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import runtime.startup_telemetry as telemetry

    monkeypatch.setattr(telemetry, "_BOOT_INVOCATION_ID", "fixedbootid")
    with caplog.at_level(logging.INFO, logger="lisa.startup"), stage("pool_readiness"):
        pass

    events = _events(caplog)
    assert events
    # The emitted payload carries only identifiers/counters/durations —
    # never prompt text, credentials or environment values (spec §5).
    for event in events:
        assert set(event) <= {"boot_id", "stage", "event", "pid", "seconds"}
    assert all(event["boot_id"] == "fixedbootid" for event in events)


def test_warn_if_repeat_setup_is_silent_on_first_invocation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="lisa.startup"):
        warn_if_repeat_setup()

    assert caplog.records == []


def test_warn_if_repeat_setup_warns_on_second_distinct_invocation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    warn_if_repeat_setup()
    with caplog.at_level(logging.WARNING, logger="lisa.startup"):
        warn_if_repeat_setup()

    assert len(caplog.records) == 1
    payload = json.loads(
        caplog.records[0].getMessage().removeprefix("startup repeat_setup_detected ")
    )
    assert payload["setup_count"] == 2
    # Same boot, second invocation: the ids match by design — the warning
    # is the count-based evidence that a process called setup twice.
    assert payload["previous_boot_id"] == payload["boot_id"]


def test_seconds_since_boot_is_monotonic() -> None:
    first = seconds_since_boot()
    assert seconds_since_boot() >= first


def test_build_runtime_emits_boot_stage_lines(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from runtime.bootstrap import build_runtime

    _install_fakes(monkeypatch, config=_config(host_kind="cli"))
    with caplog.at_level(logging.INFO, logger="lisa.startup"):
        build_runtime(host_kind="cli", logging_configurator=lambda config: None)

    events = _events(caplog)
    # Begin/end pairs per stage, one boot id, and the boot_complete line
    # (which is a boundary marker, not a stage). Begin events give the
    # ordered stage list. The fakes wrap the two framework calls exactly
    # like mcp_pool does, so exactly one begin/end pair per stage holds;
    # an outer stage() wrapper in bootstrap.py would emit 9 begins and
    # duplicated end pairs — the regression this guard catches.
    assert [event["event"] for event in events] == ["begin", "end"] * 7 + ["boot_complete"]
    begins = [event["stage"] for event in events if event["event"] == "begin"]
    ends = [event["stage"] for event in events if event["event"] == "end"]
    # Single span owner per stage: exactly one begin and one end, no
    # duplicated stage names anywhere in the sequence.
    assert len(begins) == len(ends) == 7
    assert len(set(begins)) == len(set(ends)) == 7
    assert len({event["boot_id"] for event in events}) == 1
    stages = begins
    assert stages == [
        "runtime_construct",
        "state_snapshot_load",
        "discover_framework",
        "setup_framework",
        "configure_runtime",
        "compile_mcp_configs",
        "compile_proxy_configs",
    ]


def test_setup_framework_logs_completion_without_repeat_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from runtime.mcp_pool import setup_framework

    runtime = SimpleNamespace(
        resources=SimpleNamespace(shared_session_lease=None),
        _in_process_mcp=True,
    )
    monkeypatch.setattr(
        "runtime.mcp_pool._setup_framework_impl", lambda rt, *, discovery=None: None
    )

    # The completion line goes through mcp_pool's module logger (its
    # stage() events go through lisa.startup), so INFO must be enabled on
    # every logger for this test's records to contain both.
    with caplog.at_level(logging.INFO, logger="lisa.startup"), caplog.at_level(logging.INFO):
        setup_framework(runtime)  # type: ignore[arg-type]

    messages = [record.getMessage() for record in caplog.records]
    assert not any("repeat_setup_detected" in message for message in messages)
    assert any("setup_framework_complete" in message for message in messages)

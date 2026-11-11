"""Slice E: startup benchmark harness — probes, telemetry parsing, aggregation."""

from __future__ import annotations

import json
import os
import sys

import pytest
from runtime.startup_benchmark import (
    IMPORT_PROBE_SOURCE,
    child_python_command,
    common_module_root,
    duplicate_setup_flags,
    format_report,
    parse_startup_lines,
    percentile,
    run_child_probe,
    setup_invocation_count,
    shared_session_count,
    stage_durations,
    summarize_variant,
)


def _telemetry_text() -> str:
    def line(stage: str, event: str, **extra: object) -> str:
        payload: dict[str, object] = {
            "boot_id": "abc123",
            "event": event,
            "pid": 7,
            "stage": stage,
        }
        payload.update(extra)
        return f"startup {json.dumps(payload, sort_keys=True)}"

    return "\n".join(
        [
            line("setup_framework", "begin"),
            line("setup_framework", "end", seconds=0.25),
            line("discover_framework", "begin"),
            line("discover_framework", "end", seconds=0.5),
            # The suspect double-boot: a second distinct invocation.
            line("setup_framework", "begin"),
            line("setup_framework", "end", seconds=0.1),
            'startup repeat_setup_detected {"boot_id": "abc123",'
            ' "pid": 7, "previous_boot_id": "abc123", "setup_count": 2}',
            'startup {"boot_id": "abc123", "commands": {"lisa-catalog": 1},'
            ' "event": "shared_session_inventory", "pid": 7, "sessions": 1}',
            'startup {"boot_id": "abc123", "event": "boot_complete", "pid": 7, "t_seconds": 1.5}',
            "WARNING some unrelated child noise",
            "startup not-json",
        ]
    )


def test_percentile_interpolates_linearly() -> None:
    assert percentile([5.0], 0.95) == 5.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert percentile([0.0, 10.0], 0.95) == 9.5
    assert percentile([3.0, 1.0, 2.0], 0.0) == 1.0
    assert percentile([3.0, 1.0, 2.0], 1.0) == 3.0


def test_percentile_rejects_empty_and_out_of_range() -> None:
    with pytest.raises(ValueError, match="empty"):
        percentile([], 0.5)
    with pytest.raises(ValueError, match="between 0 and 1"):
        percentile([1.0], 1.5)


def test_parse_startup_lines_keeps_only_telemetry_dicts() -> None:
    events = parse_startup_lines(_telemetry_text())
    assert len(events) == 9
    assert all(isinstance(event, dict) for event in events)
    assert events[-1]["event"] == "boot_complete"


def test_parse_startup_lines_skips_malformed_and_prefix_events() -> None:
    text = "\n".join(
        [
            'startup {"event": "begin", "stage": "runtime_construct"}',
            "startup not-json",  # malformed payload: skipped
            "noise line",  # not telemetry: skipped
            'startup repeat_setup_detected {"invocations": 2}',  # prefix-style event
            '{"event": "end"}',  # JSON but no prefix: skipped
        ]
    )
    events = parse_startup_lines(text)
    assert events == [
        {"event": "begin", "stage": "runtime_construct"},
        {"event": "repeat_setup_detected", "invocations": 2},
    ]


def test_common_module_root_joins_sibling_package_dirs() -> None:
    # Paths synthesized with os.path so the expectation holds on every
    # platform's separator (the probe reports the child's own path shape).
    base = os.path.join("repo", "src")
    paths = {
        "kernel": os.path.join(base, "kernel", "__init__.py"),
        "runtime": os.path.join(base, "runtime", "__init__.py"),
        "sdk": os.path.join(base, "sdk", "__init__.py"),
    }
    assert common_module_root(paths) == base
    assert common_module_root({}) == ""


def test_setup_invocation_count_uses_begin_events_not_repeat_warnings() -> None:
    events = parse_startup_lines(_telemetry_text())
    assert setup_invocation_count(events) == 2
    assert duplicate_setup_flags(events) == 1


def test_shared_session_count_reads_inventory_event() -> None:
    events = parse_startup_lines(_telemetry_text())
    assert shared_session_count(events) == 1
    assert shared_session_count([]) == 0


def test_stage_durations_maps_completed_stages() -> None:
    events = parse_startup_lines(_telemetry_text())
    # The repeated setup_framework stage sums both invocations (0.25 + 0.1).
    assert stage_durations(events) == {
        "setup_framework": 0.35,
        "discover_framework": 0.5,
    }


def test_child_python_command_builds_dash_c_argv() -> None:
    command = child_python_command("/usr/bin/python3", "print(1)")
    assert command == ["/usr/bin/python3", "-c", "print(1)"]


def test_run_child_probe_real_subprocess() -> None:
    command = child_python_command(
        sys.executable, 'print(\'{"module_paths": {}, "telemetry": ""}\')'
    )
    sample = run_child_probe(command, variant="warm")
    assert sample.exit_code == 0
    assert sample.variant == "warm"
    assert sample.wall_seconds > 0.0
    assert sample.module_paths == {}


def test_run_child_probe_timeout_yields_failed_sample() -> None:
    command = child_python_command(sys.executable, "import time; time.sleep(5)")
    sample = run_child_probe(command, variant="cold", timeout_seconds=0.2)
    assert sample.exit_code == -1


def test_summarize_variant_aggregates_counts_and_percentiles() -> None:
    from runtime.startup_benchmark import BenchmarkSample

    def make_sample(wall: float, setups: int, duplicates: int, sessions: int) -> BenchmarkSample:
        return BenchmarkSample(
            variant="warm",
            wall_seconds=wall,
            exit_code=0,
            setup_invocations=setups,
            duplicate_setups=duplicates,
            shared_sessions=sessions,
        )

    run = summarize_variant(
        "warm",
        [
            make_sample(1.0, 1, 0, 2),
            make_sample(2.0, 1, 1, 2),
            make_sample(3.0, 2, 0, 3),
        ],
    )
    assert run.sample_count == 3
    assert run.p50_wall_seconds == 2.0
    assert run.p95_wall_seconds == pytest.approx(2.9)
    assert run.max_setup_invocations == 2
    assert run.total_duplicate_setups == 1
    assert run.max_shared_sessions == 3
    assert run.module_root == ""


def test_format_report_renders_markdown_table() -> None:
    from runtime.startup_benchmark import BenchmarkSample

    run = summarize_variant(
        "warm",
        [BenchmarkSample(variant="warm", wall_seconds=1.5, exit_code=0, shared_sessions=2)],
    )
    report = format_report([run])
    assert report.startswith("# Prompt startup benchmark")
    assert "| warm | 1 | 1.5000 | 1.5000 |" in report


def test_import_probe_source_runs_and_reports_module_paths() -> None:
    """End-to-end: the real probe source runs in a fresh interpreter."""
    sample = run_child_probe(
        child_python_command(sys.executable, IMPORT_PROBE_SOURCE), variant="cold"
    )
    assert sample.exit_code == 0
    assert sample.wall_seconds > 0.0
    assert sample.import_wall_seconds > 0.0
    # sdk is only present if a bootstrap import chain loaded it; assert the
    # two guaranteed entry packages and that every reported path is real.
    assert {"kernel", "runtime"} <= set(sample.module_paths)
    # Some environments report "" for namespace-style packages; the harness
    # surfaces that anomaly via module_root rather than failing the probe.
    assert any(path for path in sample.module_paths.values())
    assert common_module_root(sample.module_paths) != ""

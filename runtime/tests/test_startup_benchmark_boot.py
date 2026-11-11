"""Slice E harness gaps: boot probe coverage."""

from __future__ import annotations

import sys

from runtime.startup_benchmark import (
    BOOT_PROBE_SOURCE,
    BenchmarkSample,
    child_python_command,
    format_report,
    run_child_probe,
    summarize_variant,
)


def test_boot_probe_real_subprocess() -> None:
    sample = run_child_probe(
        child_python_command(sys.executable, BOOT_PROBE_SOURCE),
        variant="boot",
        timeout_seconds=300.0,
    )
    assert sample.exit_code == 0, sample.stage_durations
    assert sample.teardown_wall_seconds > 0.0
    assert sample.setup_invocations >= 1
    assert "runtime_construct" in sample.stage_durations


def test_report_includes_teardown_column() -> None:
    run = summarize_variant(
        "boot",
        [
            BenchmarkSample(
                variant="boot",
                wall_seconds=1.5,
                exit_code=0,
                teardown_wall_seconds=0.25,
                setup_invocations=1,
            )
        ],
    )
    report = format_report([run])
    assert "teardown (s)" in report

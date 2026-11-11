"""Controlled prompt-startup benchmarks (Slice E of ``docs/SPEC/PROMPT_STARTUP.md``).

Implements the measurement protocol the spec mandates: fresh-process
samples under identical interpreter and environment conditions, first-run
("cold") effects reported separately from steady-state ("warm") samples,
p50/p95 wall-time aggregation, and capture of the Slice C
``lisa.startup`` telemetry a boot emits — stage durations, distinct
``setup_framework`` invocation counts, duplicate-setup flags and the
shared-session inventory. Module paths are verified in the child process
so a report states which checkout or installed package actually loaded.

Child probes run via ``python -c`` against the probe sources defined
here, so each sample pays a real interpreter start plus imports, the same
cost model as a ``lisa prompt`` boot's front half. Every payload obeys
the Slice C hygiene rules: durations and counts only — no prompt text,
credentials, environment values or private content.

Measuring a development checkout does not upgrade the installed CLI; the
final release gate still requires probes against built/installed
packages.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "IMPORT_PROBE_SOURCE",
    "BenchmarkRun",
    "BenchmarkSample",
    "child_python_command",
    "common_module_root",
    "duplicate_setup_flags",
    "format_report",
    "parse_startup_lines",
    "percentile",
    "run_child_probe",
    "setup_invocation_count",
    "shared_session_count",
    "stage_durations",
    "summarize_variant",
]

#: Child probe that imports the framework's public entry modules in a fresh
#: interpreter, then prints one JSON line with import wall time and the
#: resolved ``__file__`` of each top-level package (module-path verification,
#: spec §7).
IMPORT_PROBE_SOURCE = """\
import json
import sys
import time

started = time.perf_counter()
import kernel  # noqa: F401
import runtime.bootstrap  # noqa: F401
wall = time.perf_counter() - started
paths = {
    name: getattr(sys.modules[name], "__file__", "")
    for name in ("kernel", "runtime", "sdk")
    if name in sys.modules
}
print(json.dumps({"module_paths": paths, "import_wall_seconds": wall}))
"""


@dataclass(frozen=True)
class BenchmarkSample:
    """One fresh-process probe result (all values durations/counts only)."""

    variant: str
    wall_seconds: float
    exit_code: int
    import_wall_seconds: float = 0.0
    """In-child import wall time for the probe's imports (more precise than
    the parent-observed wall, which includes interpreter start)."""
    module_paths: dict[str, str] = field(default_factory=dict)
    stage_durations: dict[str, float] = field(default_factory=dict)
    setup_invocations: int = 0
    duplicate_setups: int = 0
    shared_sessions: int = 0


@dataclass(frozen=True)
class BenchmarkRun:
    """Aggregated per-variant summary (spec §7: publish p50/p95 + counts)."""

    variant: str
    sample_count: int
    p50_wall_seconds: float
    p95_wall_seconds: float
    min_wall_seconds: float
    max_wall_seconds: float
    max_setup_invocations: int
    total_duplicate_setups: int
    max_shared_sessions: int
    module_root: str

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready mapping for ``--json`` artifacts."""
        return asdict(self)


def percentile(values: list[float], q: float) -> float:
    """Return the ``q`` quantile (0.0-1.0) with linear interpolation."""
    if not values:
        raise ValueError("percentile of an empty sample list")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be between 0 and 1")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def parse_startup_lines(text: str) -> list[dict[str, Any]]:
    """Parse ``startup {json}`` telemetry lines (Slice C) out of any text.

    Non-telemetry lines and malformed payloads are skipped, so a child's
    noisy stderr never breaks aggregation.
    """
    stages: list[dict[str, Any]] = []
    prefix_event = "repeat_setup_detected"
    for line in text.splitlines():
        if not line.startswith("startup "):
            continue
        rest = line.removeprefix("startup ")
        # Prefix-style events ("startup repeat_setup_detected {json}") are
        # normalized into the payload so callers match one ``event`` key.
        event_override: str | None = None
        if rest.startswith(prefix_event + " "):
            event_override = prefix_event
            rest = rest.removeprefix(prefix_event + " ")
        try:
            payload = json.loads(rest)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if event_override is not None:
            payload["event"] = event_override
        stages.append(payload)
    return stages


def setup_invocation_count(stages: list[dict[str, Any]]) -> int:
    """Count distinct ``setup_framework`` invocations in one boot's telemetry.

    Each real invocation emits exactly one ``setup_framework`` begin event;
    cProfile call counts (which also count coroutine resumes) are not used.
    """
    return sum(
        1
        for event in stages
        if event.get("stage") == "setup_framework" and event.get("event") == "begin"
    )


def duplicate_setup_flags(stages: list[dict[str, Any]]) -> int:
    """Count ``repeat_setup_detected`` warnings (invocations beyond the first)."""
    return sum(1 for event in stages if event.get("event") == "repeat_setup_detected")


def shared_session_count(stages: list[dict[str, Any]]) -> int:
    """Return live shared catalog/content sessions from the boot inventory."""
    for event in stages:
        if event.get("event") == "shared_session_inventory":
            return int(event.get("sessions", 0))
    return 0


def stage_durations(stages: list[dict[str, Any]]) -> dict[str, float]:
    """Map stage name to total monotonic duration from ``end`` events.

    A boot with a repeated stage (the suspected double ``setup_framework``)
    reports the SUM of its invocations, so a benchmark row can never
    understate a boot that ran a stage more than once.
    """
    durations: dict[str, float] = {}
    for event in stages:
        if event.get("event") != "end" or not isinstance(event.get("seconds"), (int, float)):
            continue
        name = str(event["stage"])
        durations[name] = durations.get(name, 0.0) + float(event["seconds"])
    return durations


def common_module_root(paths: dict[str, str]) -> str:
    """Return the deepest common directory of resolved module files.

    A single root for kernel/runtime/sdk proves one package tree (editable
    checkout or one site-packages) served the probe; a mixed root would
    mean the installed CLI is loading stale or split implementations.
    """
    files = [os.path.dirname(path) or path for path in paths.values() if path]
    if not files:
        return ""
    if len(files) == 1:
        return files[0]
    return os.path.commonpath(files)


def child_python_command(python: str, source: str) -> list[str]:
    """Build the ``python -c`` argv for one child probe."""
    return [python, "-c", source]


def _parse_child_stdout(text: str) -> dict[str, Any]:
    """Return the last JSON object the child printed (its result payload)."""
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
    return {}


def run_child_probe(
    command: list[str],
    *,
    variant: str,
    timeout_seconds: float = 300.0,
    env: dict[str, str] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> BenchmarkSample:
    """Run one fresh-process probe and fold its output into a sample.

    A timeout or unparsable payload yields a failed sample (exit code
    ``-1``) rather than an exception, so one bad run cannot destroy the
    remaining samples of a sweep.
    """
    started = time.perf_counter()
    try:
        completed = runner(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return BenchmarkSample(
            variant=variant,
            wall_seconds=time.perf_counter() - started,
            exit_code=-1,
        )
    wall = time.perf_counter() - started
    payload = _parse_child_stdout(completed.stdout)
    stages = parse_startup_lines(str(payload.get("telemetry", "")))
    import_wall = payload.get("import_wall_seconds")
    module_paths = payload.get("module_paths")
    return BenchmarkSample(
        variant=variant,
        wall_seconds=wall,
        exit_code=completed.returncode,
        import_wall_seconds=float(import_wall) if isinstance(import_wall, (int, float)) else 0.0,
        module_paths=module_paths if isinstance(module_paths, dict) else {},
        stage_durations=stage_durations(stages),
        setup_invocations=setup_invocation_count(stages),
        duplicate_setups=duplicate_setup_flags(stages),
        shared_sessions=shared_session_count(stages),
    )


def summarize_variant(variant: str, samples: list[BenchmarkSample]) -> BenchmarkRun:
    """Aggregate one variant's samples into the published per-variant row."""
    walls = [sample.wall_seconds for sample in samples]
    if not walls:
        raise ValueError(f"no samples for variant {variant!r}")
    return BenchmarkRun(
        variant=variant,
        sample_count=len(samples),
        p50_wall_seconds=percentile(walls, 0.50),
        p95_wall_seconds=percentile(walls, 0.95),
        min_wall_seconds=min(walls),
        max_wall_seconds=max(walls),
        max_setup_invocations=max(sample.setup_invocations for sample in samples),
        total_duplicate_setups=sum(sample.duplicate_setups for sample in samples),
        max_shared_sessions=max(sample.shared_sessions for sample in samples),
        module_root=common_module_root(
            {key: path for sample in samples for key, path in sample.module_paths.items() if path}
        ),
    )


def format_report(runs: list[BenchmarkRun]) -> str:
    """Render the published markdown report table for a benchmark sweep."""
    header = (
        "| variant | n | p50 (s) | p95 (s) | min (s) | max (s) | setups | dup setups | sessions |"
    )
    rows = [
        f"| {run.variant} | {run.sample_count} | {run.p50_wall_seconds:.4f} | "
        f"{run.p95_wall_seconds:.4f} | {run.min_wall_seconds:.4f} | "
        f"{run.max_wall_seconds:.4f} | {run.max_setup_invocations} | "
        f"{run.total_duplicate_setups} | {run.max_shared_sessions} |"
        for run in runs
    ]
    lines = [
        "# Prompt startup benchmark",
        "",
        header,
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        *rows,
    ]
    return "\n".join(lines) + "\n"

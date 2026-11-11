"""Measure prompt-startup wall time, subprocess and setup-invocation counts.

Runs the Slice E protocol from ``docs/SPEC/PROMPT_STARTUP.md``: fresh-process
probes with identical interpreter/environment conditions, first-run (``cold``)
effects reported separately from steady-state (``warm``) samples, p50/p95
aggregation, ``lisa.startup`` telemetry capture (distinct ``setup_framework``
invocation counts — the dedup-investigation evidence) and module-path
verification of whichever package tree the probes actually loaded.

A checkout report describes that checkout only; the release gate runs this
against built/installed packages.

Usage::

    python scripts/startup_benchmark.py --samples 10 [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from runtime.startup_benchmark import (
    IMPORT_PROBE_SOURCE,
    BenchmarkSample,
    child_python_command,
    format_report,
    run_child_probe,
    summarize_variant,
)


def main(argv: list[str] | None = None) -> int:
    """Run one cold + N warm fresh-process probes and publish the report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--samples", type=int, default=10, help="warm fresh-process samples (spec: 10)"
    )
    parser.add_argument("--python", default=sys.executable, help="interpreter to probe")
    parser.add_argument("--timeout", type=float, default=300.0, help="per-probe timeout")
    parser.add_argument("--json", type=Path, default=None, help="write full JSON results")
    args = parser.parse_args(argv)

    command = child_python_command(args.python, IMPORT_PROBE_SOURCE)
    cold = run_child_probe(command, variant="cold", timeout_seconds=args.timeout)
    warm = [
        run_child_probe(command, variant="warm", timeout_seconds=args.timeout)
        for _ in range(max(args.samples, 1))
    ]
    runs = [summarize_variant("cold", [cold]), summarize_variant("warm", warm)]
    print(format_report(runs))

    samples: list[BenchmarkSample] = [cold, *warm]
    if args.json is not None:
        args.json.write_text(
            json.dumps(
                {
                    "runs": [run.to_dict() for run in runs],
                    "samples": [asdict(sample) for sample in samples],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    failed = [sample for sample in samples if sample.exit_code != 0]
    if failed:
        print(f"{len(failed)} probe(s) failed; treat the run as invalid", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

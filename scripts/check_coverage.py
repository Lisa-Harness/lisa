#!/usr/bin/env python3
"""Per-package coverage gates.

The global ``fail_under`` in ``pyproject.toml`` reflects the suite-wide source
total. A single number would let the small harness packages
(packstest, e2etest) rot below their own floors while the big packages push the
average up, so this script enforces each package at its own measured floor.

Floors are set ~2 points under the baselines measured on 2026-10-31 (branch
coverage) so they catch real regressions without flapping on platform timing
noise. Raise them as coverage improves — never lower them.

Usage:
    uv run python scripts/check_coverage.py --run
    uv run python scripts/check_coverage.py --run --skip-changed-lines
    uv run python scripts/check_coverage.py --run --base origin/main
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# (coverage --include glob, minimum branch-aware %, measured-baseline %)
# Baselines measured with the full offline suite;
# floors sit ~2 pts below their baseline so they catch real regressions without
# flapping on platform timing noise. Raise them as coverage improves — never lower.
FLOORS: list[tuple[str, int, int]] = [
    ("kernel/*", 83, 85),
    ("sdk/src/*", 85, 87),
    ("sdk/packs/*/src/*", 87, 89),
    ("mcp_server/src/*", 77, 79),
    ("runtime/src/*", 82, 84),
    ("packstest/src/*", 68, 70),
    ("e2etest/src/*", 73, 75),
    ("scripts/*", 65, 67),
]


def _coverage(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "coverage", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _check_package_floors() -> int:
    failures = 0
    for include, floor, baseline in FLOORS:
        result = _coverage("report", f"--include={include}", f"--fail-under={floor}")
        # Last data row holds the TOTAL percent for the include window.
        pct = "?"
        for line in result.stdout.splitlines():
            if line.startswith("TOTAL"):
                pct = line.split()[-1]
        ok = result.returncode == 0 and pct != "?"
        status = "ok " if ok else "LOW"
        print(f"  [{status}] {include:22} {pct:>5}  floor={floor}%  baseline={baseline}%")
        if not ok:
            print(result.stderr, file=sys.stderr)
            failures += 1
    if failures:
        print(f"\ncoverage gate: {failures} package(s) below their floor.")
        return 1
    print("\ncoverage gate: all packages at or above their floor.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run full suite and all coverage gates")
    parser.add_argument("--base", help="Branch base for changed-line coverage with --run")
    parser.add_argument(
        "--skip-changed-lines",
        action="store_true",
        help="Skip the changed-line gate when coverage will be combined across CI jobs",
    )
    args = parser.parse_args(argv)
    if args.base is not None and not args.run:
        parser.error("--base requires --run")
    if args.skip_changed_lines and not args.run:
        parser.error("--skip-changed-lines requires --run")
    if args.run:
        tests = subprocess.run(
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "-m",
                "pytest",
                "-m",
                "not live",
                "-q",
                "--tb=short",
            ],
            check=False,
        )
        if tests.returncode:
            return tests.returncode
        global_report = _coverage("report")
        print(global_report.stdout)
        if global_report.returncode:
            print(global_report.stderr, file=sys.stderr)
            return global_report.returncode
    if _check_package_floors():
        return 1
    if args.run and not args.skip_changed_lines:
        command = [sys.executable, str(Path(__file__).with_name("check_changed_coverage.py"))]
        base = args.base if args.base is not None else os.environ.get("COVERAGE_BASE")
        if base:
            command.extend(["--base", base])
        return subprocess.run(command, check=False).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Per-package coverage gates.

The global ``fail_under`` in ``pyproject.toml`` reflects the suite-wide source
total (~78% branch). A single number would let the small harness packages
(packstest, e2etest) rot below their own floors while the big packages push the
average up, so this script enforces each package at its own measured floor.

Floors are set ~2 points under the baselines measured on 2026-10-04 (branch
coverage) so they catch real regressions without flapping on platform timing
noise. Raise them as you add tests — never lower them.

Usage:
    uv run coverage run -m pytest -m "not live"
    uv run python scripts/check_coverage.py
"""

from __future__ import annotations

import subprocess
import sys

# (coverage --include glob, minimum branch-%, measured-baseline-%)
# Baselines measured 2026-10-04 after the test-suite remediation (Rec 5/5b);
# floors sit ~2 pts below their baseline so they catch real regressions without
# flapping on platform timing noise. Raise them as coverage improves — never lower.
FLOORS: list[tuple[str, int, int]] = [
    ("kernel/*", 80, 82),
    ("sdk/src/*", 80, 82),
    ("sdk/packs/*/src/*", 85, 87),
    ("mcp_server/src/*", 73, 75),
    ("runtime/src/*", 79, 81),
    ("packstest/src/*", 55, 57),
    ("e2etest/src/*", 49, 51),
]


def _coverage(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "coverage", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def main() -> int:
    failures = 0
    for include, floor, baseline in FLOORS:
        result = _coverage("report", f"--include={include}", f"--fail-under={floor}")
        # Last data row holds the TOTAL percent for the include window.
        pct = "?"
        for line in result.stdout.splitlines():
            if line.startswith("TOTAL"):
                pct = line.split()[-1]
        ok = result.returncode == 0
        status = "ok " if ok else "LOW"
        print(f"  [{status}] {include:22} {pct:>5}  floor={floor}%  baseline={baseline}%")
        if not ok:
            failures += 1
    if failures:
        print(f"\ncoverage gate: {failures} package(s) below their floor.")
        return 1
    print("\ncoverage gate: all packages at or above their floor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

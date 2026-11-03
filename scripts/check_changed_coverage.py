#!/usr/bin/env python3
"""Require 95% line coverage of changed executable Python source."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from check_tests import resolve_base
from coverage import Coverage
from coverage.exceptions import CoverageException

_ROOT = Path(__file__).resolve().parents[1]
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
_MINIMUM = 95


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "git command failed")
    return result.stdout


def _source(path: str) -> bool:
    candidate = Path(path)
    return (
        candidate.suffix == ".py"
        and "tests" not in candidate.parts
        and not candidate.name.startswith("test_")
        and candidate.name not in {"conftest.py", "vulture_allowlist.py"}
        # Operator tooling and data fixtures that ship nowhere and are never
        # imported by the pytest suites: changing them cannot regress shipped
        # source. (The e2e fixture overlay tool still executes — inside the
        # e2e-smoke host process, whose coverage is not part of this gate.)
        and path != "packstest/run_all_validations.py"
        and "base_data" not in candidate.parts
    )


def changed_lines(cwd: Path, base: str) -> dict[str, set[int]]:
    """Include committed, staged, unstaged, renamed, and untracked sources."""
    tracked = _git(cwd, "diff", "--name-only", "-z", "--diff-filter=AMRT", base, "--")
    untracked = _git(cwd, "ls-files", "--others", "--exclude-standard", "-z")
    new_paths = set(untracked.split("\0"))
    paths = set(tracked.split("\0")) | new_paths
    changes: dict[str, set[int]] = {}
    for path in sorted(paths):
        if not _source(path):
            continue
        source = cwd / path
        if not source.resolve().is_relative_to(cwd.resolve()):
            raise RuntimeError(f"Changed source escapes the repository: {path}")
        if path in new_paths:
            changes[path] = set(range(1, len(source.read_text(encoding="utf-8").splitlines()) + 1))
            continue
        diff = _git(cwd, "diff", "--no-ext-diff", "--no-textconv", "--unified=0", base, "--", path)
        lines: set[int] = set()
        for line in diff.splitlines():
            match = _HUNK.match(line)
            if match:
                start = int(match[1])
                count = int(match[2]) if match[2] is not None else 1
                lines.update(range(start, start + count))
        changes[path] = lines
    return changes


def check_coverage(coverage: Coverage, changes: dict[str, set[int]], cwd: Path) -> int:
    """Analyze current source too, so files absent from coverage cannot pass."""
    data = coverage.get_data()
    if not data.measured_files() or not data.has_arcs():
        raise RuntimeError(
            "Missing branch-coverage data; run the full offline coverage suite first"
        )
    total = 0
    uncovered = 0
    for path, lines in changes.items():
        _, statements, _, missing, _ = coverage.analysis2(str(cwd / path))
        changed = lines.intersection(statements)
        missed = changed.intersection(missing)
        total += len(changed)
        uncovered += len(missed)
        if missed:
            print(f"  {path}: uncovered changed lines {', '.join(map(str, sorted(missed)))}")
    if not total:
        print("changed coverage: no changed executable source lines")
        return 0
    percent = (total - uncovered) * 100 / total
    print(
        f"changed coverage: {total - uncovered}/{total} executable lines "
        f"({percent:.2f}%), required {_MINIMUM}%"
    )
    return int((total - uncovered) * 100 < total * _MINIMUM)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", help="Branch base ref (default: merge-base with origin/main/main)"
    )
    parser.add_argument("--cwd", type=Path, default=_ROOT, help="Repository root")
    parser.add_argument(
        "--data-file", default=".coverage", help="Coverage data file in the repository"
    )
    args = parser.parse_args(argv)
    try:
        base = resolve_base(cwd=args.cwd, explicit=args.base)
        if base is None:
            raise RuntimeError("Cannot resolve branch base; supply --base explicitly")
        base = _git(args.cwd, "merge-base", "HEAD", base).strip()
        config = args.cwd / "pyproject.toml"
        coverage = Coverage(
            data_file=str(args.cwd / args.data_file),
            config_file=str(config) if config.is_file() else False,
        )
        coverage.load()
        return check_coverage(coverage, changed_lines(args.cwd, base), args.cwd)
    except (RuntimeError, CoverageException, OSError, UnicodeError) as exc:
        print(f"changed coverage: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

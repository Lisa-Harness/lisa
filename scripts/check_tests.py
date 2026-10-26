#!/usr/bin/env python3
"""Run the test suites an owned worktree's changed paths actually reach.

DESIGN — The offline suite costs ~1m35s. Most changes touch one package, so
the commit gate should run that package's full testpaths instead of
everything. Selection must never be able to weaken the gate, so it is
computed from Git state rather than declared by whoever is asking:

- **Ownership comes from the branch delta, not the working tree.** A task's
  work is normally committed before the gate runs, so reading only
  ``git status --porcelain`` would miss it entirely. The changed set is the
  union of the working tree (``git status --porcelain``, including untracked
  and deleted files) and the committed delta against a base commit
  (``git diff --name-only <base>...HEAD``). Reading the tree alone let a
  committed change hide behind a doc-only leftover: the doc path was skipped
  as documentation and the selection came back empty with exit code 0, so the
  gate passed having run nothing. Each path maps to its owning package
  through ``PACKAGE_TESTPATHS``, which mirrors
  ``[tool.pytest.ini_options].testpaths`` and is asserted against the real
  configuration by ``packstest/tests/test_check_tests.py`` so the two cannot
  drift.
- **An unresolvable base forces the full suite.** If no base commit can be
  determined, the committed delta is unknown and the selection cannot be
  shown to cover the branch. That escalates rather than guessing.
- **Cross-cutting paths force the full suite.** A test path selector cannot
  reason about a shared configuration file, a toolchain pin, or the test
  harness itself; those escalate rather than guess. Anything outside a
  known package, plus the explicit escalation set below, means "run
  everything".
- **Kernel is the shared contract layer.** The import direction is
  kernel <- sdk <- runtime/packs/mcp_server, so a kernel change can reach
  every consumer. Kernel changes escalate to the full suite; this is the
  documented closure rule, applied mechanically.
- **Colocated tests travel with their package.** A package's tests are
  split between its top-level ``tests/`` and ``src/**/tests/``; both halves
  are always included, because selecting one half silently skips the other.
- **The exit code is pytest's.** The gate sees a real test result: a
  failed, errored, or uncollected run is a failed check.

Usage:
    uv run python scripts/check_tests.py                 # select from Git state
    uv run python scripts/check_tests.py --base origin/main  # explicit base
    uv run python scripts/check_tests.py --paths a.py b.py   # explicit paths
    uv run python scripts/check_tests.py --dry-run        # print the command
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: Owning package testpaths, mirroring ``[tool.pytest.ini_options].testpaths``.
#: ``packstest/tests/test_check_tests.py`` asserts this table covers the real
#: configured testpaths exactly, so adding a suite without updating this map
#: fails a test instead of silently narrowing the commit gate.
PACKAGE_TESTPATHS: dict[str, tuple[str, ...]] = {
    "kernel": (
        "kernel/tests",
        "kernel/src/kernel/context/tests",
    ),
    "sdk": (
        "sdk/tests",
        "sdk/src/sdk/context/tests",
        "sdk/src/sdk/guard/tests",
        "sdk/src/sdk/memory/tests",
        "sdk/src/sdk/storage/tests",
    ),
    "packs": (
        "sdk/packs/core/tests",
        "sdk/packs/memory/tests",
        "sdk/packs/recovery/tests",
    ),
    "mcp_server": (
        "mcp_server/tests",
        "mcp_server/src/mcp_server/fetch/tests",
        "mcp_server/src/mcp_server/memory/tests",
        "mcp_server/src/mcp_server/thread/tests",
    ),
    "runtime": (
        "runtime/tests",
        "runtime/src/runtime/model_discovery/tests",
    ),
    "packstest": ("packstest/tests",),
    "e2etest": ("e2etest/tests",),
}

#: Package testpath roots that a changed path may live under.
_OWNED_PREFIXES: tuple[tuple[str, str], ...] = (
    ("kernel/src/", "kernel"),
    ("kernel/tests/", "kernel"),
    ("sdk/packs/", "packs"),
    ("sdk/src/", "sdk"),
    ("sdk/tests/", "sdk"),
    ("mcp_server/src/", "mcp_server"),
    ("mcp_server/tests/", "mcp_server"),
    ("runtime/src/", "runtime"),
    ("runtime/tests/", "runtime"),
    ("packstest/src/", "packstest"),
    ("packstest/tests/", "packstest"),
    ("e2etest/src/", "e2etest"),
    ("e2etest/tests/", "e2etest"),
)

#: Scripts the packstest suite loads by path and asserts directly. Changing one
#: is covered by ``packstest/tests``, so it selects that package instead of
#: escalating. Any other ``scripts/`` file keeps escalating below.
_COVERED_SCRIPTS: tuple[str, ...] = (
    "scripts/check_docs.py",
    "scripts/check_tests.py",
)

#: Paths whose effect cannot be scoped to one package: the full suite runs.
_ESCALATE_PREFIXES: tuple[str, ...] = (
    "pyproject.toml",
    "uv.lock",
    "conftest.py",
    "typings/",
    "scripts/",
    ".github/",
    "vulture_allowlist.py",
)

#: Documentation and release-note paths that cannot affect any test. They are
#: ignored rather than escalated: nearly every commit updates CHANGELOG.md, and
#: escalating those would run the full suite every time. This mirrors CI, which
#: skips the test matrix for documentation-only changes.
#:
#: These are repository-root paths only. A package-local ``README.md`` or
#: ``AUTHORING.md`` is a runtime asset, not a guide, and it matches no entry
#: here — ``_OWNED_PREFIXES`` holds directory prefixes, so such a path falls
#: through to the escalation branch and runs the full suite. That is the safe
#: direction: the selector cannot show a package-local doc is inert.
_DOC_ONLY_PREFIXES: tuple[str, ...] = (
    "docs/",
    "CHANGELOG.md",
    "README.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "LICENSE",
)

#: Packages whose changes escalate to the full suite. Kernel is the shared
#: contract layer every other package imports, so a kernel change can reach
#: all of them (see docs/TESTING.md "Test-selection tiers").
_ESCALATE_PACKAGES = frozenset({"kernel"})

#: A ``git status --porcelain`` record is ``XY <path>``: two status columns,
#: one space, then the path.
_PORCELAIN_PATH_OFFSET = 3

#: Base refs tried in order when the caller does not name one. The branch's
#: committed work is measured against the merge-base with the first that
#: resolves.
_DEFAULT_BASE_CANDIDATES: tuple[str, ...] = ("origin/main", "main")


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _working_tree_paths(*, cwd: Path = _REPO_ROOT) -> list[str]:
    """Return uncommitted paths from Git, including untracked and deleted files.

    ``--porcelain -z`` is the machine-readable form: entries are NUL
    separated and rename/copy records carry two paths, so a plain line split
    would misread renamed files. Both sides of a rename are returned: the
    destination is what the task owns, and the source is still part of the
    diff's blast radius.
    """

    result = _git("status", "--porcelain", "-z", cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git status failed")
    paths: list[str] = []
    entries = result.stdout.split("\0")
    index = 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if len(entry) <= _PORCELAIN_PATH_OFFSET:
            continue
        status = entry[:2]
        path = entry[_PORCELAIN_PATH_OFFSET:]
        paths.append(path)
        # Rename/copy records are followed by the source path as its own entry.
        if ("R" in status or "C" in status) and index < len(entries):
            source = entries[index]
            index += 1
            if source:
                paths.append(source)
    return list(dict.fromkeys(paths))


def resolve_base(*, cwd: Path = _REPO_ROOT, explicit: str | None = None) -> str | None:
    """Return the commit the branch's committed work is measured against.

    An explicit ref must resolve, or the caller asked for something that does
    not exist and a silent fallback would hide the mistake. Without one, the
    merge-base with the first resolvable default candidate is used. ``None``
    means no base could be determined, which the caller must treat as an
    escalation rather than as "no committed work".
    """

    if explicit is not None:
        result = _git("rev-parse", "--verify", f"{explicit}^{{commit}}", cwd=cwd)
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(f"base ref {explicit!r} does not resolve to a commit")
        return result.stdout.strip()

    for candidate in _DEFAULT_BASE_CANDIDATES:
        merged = _git("merge-base", "HEAD", candidate, cwd=cwd)
        if merged.returncode == 0 and merged.stdout.strip():
            return merged.stdout.strip()
    return None


def committed_paths(*, cwd: Path = _REPO_ROOT, base: str) -> list[str]:
    """Return paths this branch changed relative to ``base``.

    Three-dot form diffs the merge-base of ``base`` and ``HEAD`` against
    ``HEAD``, so a base that has advanced past the branch point does not
    report the base's own new commits as this branch's work.
    """

    result = _git("diff", "--name-only", f"{base}...HEAD", cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git diff failed")
    return [line for line in result.stdout.splitlines() if line.strip()]


def changed_paths(*, cwd: Path = _REPO_ROOT, base: str | None = None) -> list[str]:
    """Return the paths a task changed: working tree plus committed delta.

    The working tree alone is not the task's change set. Work is usually
    committed before the gate runs, so a tree-only read reports whatever
    leftover happens to be uncommitted and can narrow the selection to
    nothing. When ``base`` is ``None`` only the working tree is read, which is
    the pre-existing behavior for callers that have no base to offer.
    """

    paths = _working_tree_paths(cwd=cwd)
    if base is not None:
        paths.extend(committed_paths(cwd=cwd, base=base))
    return list(dict.fromkeys(paths))


def _normalize(raw: str) -> str:
    """Return a repo-relative POSIX path without Git's quoting artifacts.

    Strip exactly one leading ``./`` (never arbitrary leading dots, which
    would turn ``.github/x`` into ``github/x``) and normalize separators so
    Windows checkouts map the same way.
    """

    path = raw.strip().strip('"').replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path


def owning_packages(paths: list[str]) -> tuple[set[str], list[str]]:
    """Map changed paths to owning packages, returning escalations separately.

    An unowned path is an escalation, not a skip: a change outside every
    known package cannot be shown to be covered by any subset.
    """

    packages: set[str] = set()
    escalate: list[str] = []
    for raw in paths:
        path = _normalize(raw)
        if not path:
            continue
        if path in _COVERED_SCRIPTS:
            packages.add("packstest")
            continue
        if any(path == prefix or path.startswith(prefix) for prefix in _ESCALATE_PREFIXES):
            escalate.append(path)
            continue
        # Owner match precedes the doc-only skip so a package-local README or
        # AUTHORING.md (a runtime asset) stays with its package. No
        # ``_OWNED_PREFIXES`` entry is a bare filename, so such a path does not
        # actually match here and escalates below instead.
        owner = next(
            (name for prefix, name in _OWNED_PREFIXES if path.startswith(prefix)),
            None,
        )
        if owner is not None:
            packages.add(owner)
            continue
        if any(path == prefix or path.startswith(prefix) for prefix in _DOC_ONLY_PREFIXES):
            continue
        escalate.append(path)
    return packages, escalate


def select_testpaths(
    paths: list[str], *, base_unresolved: bool = False
) -> tuple[list[str], str]:
    """Return the testpaths to run and the reason for the selection."""

    if base_unresolved:
        # Without a base the committed delta is unknown, so the selection
        # cannot be shown to cover the branch.
        return _all_testpaths(), (
            "no base ref resolved; the committed delta is unknown, running the full suite"
        )

    if not paths:
        # No observed changes means the artifact cannot be shown to be narrow;
        # a clean tree is not evidence that nothing needs testing.
        return _all_testpaths(), "no changed paths detected; running the full suite"

    packages, escalate = owning_packages(paths)
    escalated_packages = sorted(packages & _ESCALATE_PACKAGES)
    if escalate:
        sample = ", ".join(sorted(escalate)[:3])
        return _all_testpaths(), f"cross-cutting path(s) require the full suite: {sample}"
    if escalated_packages:
        return _all_testpaths(), (
            "kernel is the shared contract layer; a kernel change runs the full suite"
        )
    if not packages:
        # Documentation-only changes cannot affect behavior (CI skips the test
        # matrix for them). Report it rather than running the suite for nothing.
        return [], "documentation-only change; no test suites affected"

    selected: list[str] = []
    for package in sorted(packages):
        selected.extend(PACKAGE_TESTPATHS[package])
    return selected, f"affected package(s): {', '.join(sorted(packages))}"


def _all_testpaths() -> list[str]:
    """Return every configured testpath, package order preserved."""

    return [path for paths in PACKAGE_TESTPATHS.values() for path in paths]


def pytest_argv(testpaths: list[str]) -> list[str]:
    """Build the pytest command for the selection.

    ``--tb=short`` matches the session check this replaces: it keeps a real
    failure readable inside the pack's bounded check output instead of
    spending the excerpt on a full traceback.
    """

    return [
        sys.executable,
        "-m",
        "pytest",
        *testpaths,
        "-m",
        "not live",
        "--tb=short",
        "-q",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the test suites the changed paths actually reach.",
    )
    parser.add_argument(
        "--paths",
        nargs="*",
        default=None,
        help="Explicit changed paths (default: read the Git branch delta).",
    )
    parser.add_argument(
        "--base",
        default=None,
        help=(
            "Commit or ref the branch's committed work is measured against "
            "(default: merge-base with origin/main, then main)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the pytest command instead of running it.",
    )
    parser.add_argument(
        "--cwd",
        default=str(_REPO_ROOT),
        help="Repository root used for Git state and test execution.",
    )
    args = parser.parse_args(argv)

    cwd = Path(args.cwd)
    base_unresolved = False
    if args.paths is not None:
        paths = list(args.paths)
    else:
        try:
            base = resolve_base(cwd=cwd, explicit=args.base)
            paths = changed_paths(cwd=cwd, base=base)
        except RuntimeError as exc:
            print(f"check_tests: {exc}", file=sys.stderr)
            return 2
        base_unresolved = base is None

    testpaths, reason = select_testpaths(paths, base_unresolved=base_unresolved)
    print(f"check_tests: {reason}", file=sys.stderr)
    if not testpaths:
        print("check_tests: no suites selected", file=sys.stderr)
        return 0
    command = pytest_argv(testpaths)
    print(f"check_tests: {' '.join(testpaths)}", file=sys.stderr)
    if args.dry_run:
        print(" ".join(command))
        return 0

    completed = subprocess.run(command, cwd=cwd, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

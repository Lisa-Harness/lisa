"""Tests for the changed-path test selector (``scripts/check_tests.py``).

The selector gates commits, so its failure modes matter more than its happy
path: it must never pick a subset that misses a suite the change can reach.
Three invariants carry that guarantee and are asserted here:

- The package table matches the real ``[tool.pytest.ini_options].testpaths``
  exactly, so a suite added to configuration cannot silently fall outside
  selection.
- Every unowned or shared path escalates to the full suite instead of being
  dropped, because a path the selector cannot classify cannot be shown to be
  covered by any subset.
- The changed set is the branch delta, not the working tree. A task's work is
  normally committed before the gate runs, so a tree-only read would miss it
  and a doc-only leftover could narrow the selection to nothing.

The script is loaded by path rather than imported as a package: it lives in
``scripts/``, which is not an importable package in this workspace.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "check_tests.py"


def _load_module() -> ModuleType:
    """Import ``scripts/check_tests.py`` as a module."""
    spec = importlib.util.spec_from_file_location("check_tests_under_test", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check_tests = _load_module()


def _configured_testpaths() -> list[str]:
    """Return the repository's configured pytest testpaths."""
    with (_REPO_ROOT / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)
    return [path.rstrip("/") for path in config["tool"]["pytest"]["ini_options"]["testpaths"]]


def _init_repo(repo: Path) -> list[str]:
    """Create a throwaway repo with one commit and return the git prefix."""

    repo.mkdir()
    git = ["git", "-C", str(repo)]
    # Pin the initial branch: ``init.defaultBranch`` is unset on CI runners, so
    # a bare ``git init`` creates ``master`` there and ``main`` on a developer
    # box. Tests naming ``main`` as a base candidate would then resolve against
    # a branch that does not exist, and silently take a different base.
    subprocess.run([*git, "init", "-q", "--initial-branch=main"], check=True)
    subprocess.run([*git, "config", "user.email", "t@example.com"], check=True)
    subprocess.run([*git, "config", "user.name", "T"], check=True)
    # A throwaway fixture must not inherit the developer's ambient
    # ``commit.gpgsign=true``; signing would launch an interactive pinentry.
    subprocess.run([*git, "config", "commit.gpgsign", "false"], check=True)
    seed = repo / "seed.py"
    seed.write_text("a = 1\n")
    subprocess.run([*git, "add", "seed.py"], check=True)
    subprocess.run([*git, "commit", "-qm", "init"], check=True)
    return git


def _commit(repo: Path, git: list[str], rel_path: str, content: str, message: str) -> None:
    """Write and commit one file."""

    target = repo / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    subprocess.run([*git, "add", rel_path], check=True)
    subprocess.run([*git, "commit", "-qm", message], check=True)


def _head(repo: Path, git: list[str]) -> str:
    return subprocess.run(
        [*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


class TestTableMatchesConfiguration:
    """The selector's table must not drift from the real pytest config."""

    def test_table_covers_every_configured_testpath(self) -> None:
        configured = set(_configured_testpaths())
        mapped = {path for paths in check_tests.PACKAGE_TESTPATHS.values() for path in paths}
        assert configured - mapped == set(), (
            "configured testpaths missing from scripts/check_tests.py "
            f"PACKAGE_TESTPATHS: {sorted(configured - mapped)}"
        )

    def test_table_adds_no_testpath_configuration_lacks(self) -> None:
        configured = set(_configured_testpaths())
        mapped = {path for paths in check_tests.PACKAGE_TESTPATHS.values() for path in paths}
        assert mapped - configured == set(), (
            "scripts/check_tests.py names testpaths the configuration does not: "
            f"{sorted(mapped - configured)}"
        )

    def test_package_tables_are_disjoint_and_nonempty(self) -> None:
        seen: set[str] = set()
        for package, paths in check_tests.PACKAGE_TESTPATHS.items():
            assert paths, f"package '{package}' has no testpaths"
            overlap = seen & set(paths)
            assert overlap == set(), f"package '{package}' shares testpaths: {sorted(overlap)}"
            seen |= set(paths)


class TestSelection:
    """Changed paths select the owning package's complete testpaths."""

    @pytest.mark.parametrize(
        ("path", "package"),
        [
            ("sdk/src/sdk/config.py", "sdk"),
            ("sdk/tests/test_settings.py", "sdk"),
            ("mcp_server/src/mcp_server/fetch/server.py", "mcp_server"),
            ("runtime/src/runtime/bootstrap.py", "runtime"),
            ("packstest/src/packstest/catalog.py", "packstest"),
            ("e2etest/src/e2etest/oracles.py", "e2etest"),
            ("sdk/packs/core/src/sdk/packs/core/tools/x.py", "packs"),
        ],
    )
    def test_single_package_selects_all_of_its_testpaths(self, path: str, package: str) -> None:
        testpaths, reason = check_tests.select_testpaths([path])
        assert testpaths == list(check_tests.PACKAGE_TESTPATHS[package])
        assert package in reason

    def test_colocated_and_top_level_tests_both_covered(self) -> None:
        """A package's split test layout must never be partially selected."""
        testpaths, _ = check_tests.select_testpaths(["sdk/src/sdk/config.py"])
        assert "sdk/tests" in testpaths
        assert "sdk/src/sdk/guard/tests" in testpaths

    def test_multiple_packages_union_their_testpaths(self) -> None:
        testpaths, _ = check_tests.select_testpaths(
            ["runtime/src/runtime/bootstrap.py", "mcp_server/src/mcp_server/x.py"]
        )
        assert set(testpaths) == {
            *check_tests.PACKAGE_TESTPATHS["runtime"],
            *check_tests.PACKAGE_TESTPATHS["mcp_server"],
        }

    def test_kernel_change_escalates_to_full_suite(self) -> None:
        """Kernel is the shared contract layer every package imports."""
        testpaths, reason = check_tests.select_testpaths(["kernel/src/kernel/models.py"])
        assert testpaths == check_tests._all_testpaths()
        assert "kernel" in reason

    @pytest.mark.parametrize(
        "path",
        [
            "pyproject.toml",
            "uv.lock",
            ".github/workflows/ci.yml",
            "typings/something.pyi",
            "conftest.py",
            "vulture_allowlist.py",
        ],
    )
    def test_cross_cutting_paths_escalate(self, path: str) -> None:
        testpaths, reason = check_tests.select_testpaths([path])
        assert testpaths == check_tests._all_testpaths()
        assert "cross-cutting" in reason

    def test_unowned_path_escalates_instead_of_being_dropped(self) -> None:
        """A path outside every known package cannot be shown to be covered."""
        testpaths, reason = check_tests.select_testpaths(["some-new-package/src/x.py"])
        assert testpaths == check_tests._all_testpaths()
        assert "cross-cutting" in reason

    def test_no_changes_runs_full_suite(self) -> None:
        """A clean tree is not evidence that nothing needs testing."""
        testpaths, reason = check_tests.select_testpaths([])
        assert testpaths == check_tests._all_testpaths()
        assert "no changed paths" in reason

    def test_escalation_wins_over_a_narrow_package(self) -> None:
        testpaths, _ = check_tests.select_testpaths(["sdk/src/sdk/config.py", "pyproject.toml"])
        assert testpaths == check_tests._all_testpaths()

    @pytest.mark.parametrize(
        "path",
        ["CHANGELOG.md", "docs/TESTING.md", "README.md", "SECURITY.md", "LICENSE"],
    )
    def test_documentation_only_change_selects_no_suites(self, path: str) -> None:
        """Doc-only changes cannot affect behavior; CI skips the matrix for them.

        Without this, every commit that touches CHANGELOG.md would escalate to
        the full suite and the selection would buy nothing.
        """
        testpaths, reason = check_tests.select_testpaths([path])
        assert testpaths == []
        assert "documentation-only" in reason

    def test_documentation_alongside_code_still_runs_the_code_suites(self) -> None:
        testpaths, reason = check_tests.select_testpaths(["sdk/src/sdk/config.py", "CHANGELOG.md"])
        assert testpaths == list(check_tests.PACKAGE_TESTPATHS["sdk"])
        assert "documentation-only" not in reason

    def test_package_local_readme_escalates_rather_than_being_skipped(self) -> None:
        """A package-local README matches no owner prefix, so it escalates.

        ``_OWNED_PREFIXES`` holds directory prefixes only, so ``sdk/README.md``
        is not owned by ``sdk`` and is not doc-only either. Running the full
        suite is the safe direction: the selector cannot show it is inert.
        """
        testpaths, reason = check_tests.select_testpaths(["sdk/README.md"])
        assert testpaths == check_tests._all_testpaths()
        assert "cross-cutting" in reason

    @pytest.mark.parametrize("path", list(check_tests._COVERED_SCRIPTS))
    def test_scripts_covered_by_packstest_select_packstest(self, path: str) -> None:
        testpaths, reason = check_tests.select_testpaths([path])
        assert testpaths == list(check_tests.PACKAGE_TESTPATHS["packstest"])
        assert "packstest" in reason

    def test_uncovered_script_escalates(self) -> None:
        testpaths, reason = check_tests.select_testpaths(["scripts/cache_report.py"])
        assert testpaths == check_tests._all_testpaths()
        assert "cross-cutting" in reason

    def test_unresolved_base_escalates_to_full_suite(self) -> None:
        """Without a base the committed delta is unknown, so nothing is provable."""
        testpaths, reason = check_tests.select_testpaths([], base_unresolved=True)
        assert testpaths == check_tests._all_testpaths()
        assert "no base ref resolved" in reason

    def test_unresolved_base_escalates_even_with_doc_only_paths(self) -> None:
        """A doc-only leftover must not mask an unknown committed delta."""
        testpaths, reason = check_tests.select_testpaths(["docs/TESTING.md"], base_unresolved=True)
        assert testpaths == check_tests._all_testpaths()
        assert "no base ref resolved" in reason


class TestPathNormalization:
    """Git records paths in forms the selector must read correctly."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("./sdk/src/sdk/config.py", "sdk/src/sdk/config.py"),
            (".github/workflows/ci.yml", ".github/workflows/ci.yml"),
            ("sdk\\src\\sdk\\config.py", "sdk/src/sdk/config.py"),
            ('"sdk/src/sdk/config.py"', "sdk/src/sdk/config.py"),
        ],
    )
    def test_normalize(self, raw: str, expected: str) -> None:
        assert check_tests._normalize(raw) == expected

    def test_leading_dot_directory_is_not_stripped(self) -> None:
        """``.github`` must not become ``github`` (which would look unowned)."""
        packages, escalate = check_tests.owning_packages([".github/x.yml"])
        assert packages == set()
        assert escalate == [".github/x.yml"]


class TestChangedPathsFromGit:
    """Reading real Git state, including the record shapes Git emits."""

    def test_reads_untracked_and_modified_files(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _init_repo(repo)

        (repo / "seed.py").write_text("a = 2\n")
        (repo / "untracked.py").write_text("b = 1\n")

        paths = check_tests.changed_paths(cwd=repo)
        assert "seed.py" in paths
        assert "untracked.py" in paths

    def test_reads_both_sides_of_a_rename(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        git = _init_repo(repo)
        subprocess.run([*git, "mv", "seed.py", "after.py"], check=True)

        paths = check_tests.changed_paths(cwd=repo)
        assert "after.py" in paths
        assert "seed.py" in paths

    def test_missing_git_state_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError):
            check_tests.changed_paths(cwd=tmp_path)


class TestCommittedDelta:
    """Committed work must be visible to the selector, not just the tree."""

    def test_committed_work_is_seen_when_the_tree_is_clean(self, tmp_path: Path) -> None:
        """A clean tree after committing is not an empty change set."""
        repo = tmp_path / "repo"
        git = _init_repo(repo)
        base = _head(repo, git)
        _commit(repo, git, "kernel/src/kernel/models.py", "x = 1\n", "kernel change")

        assert check_tests.changed_paths(cwd=repo) == []
        paths = check_tests.changed_paths(cwd=repo, base=base)
        assert "kernel/src/kernel/models.py" in paths

    def test_doc_only_leftover_cannot_hide_committed_work(self, tmp_path: Path) -> None:
        """The narrowing path: commit code, leave a doc edit uncommitted.

        Reading the tree alone reported only the doc path, which is skipped as
        documentation, so the selection came back empty and the gate passed
        having run nothing. The union must still see the committed kernel
        change and escalate.
        """
        repo = tmp_path / "repo"
        git = _init_repo(repo)
        base = _head(repo, git)
        _commit(repo, git, "kernel/src/kernel/models.py", "x = 1\n", "kernel change")
        (repo / "docs").mkdir()
        (repo / "docs" / "TESTING.md").write_text("# docs\n")

        tree_only = check_tests.changed_paths(cwd=repo)
        # Git reports the untracked directory, not the file inside it.
        assert tree_only == ["docs/"]
        assert "kernel/src/kernel/models.py" not in tree_only
        # The tree-only read narrows to nothing: this is the defect.
        assert check_tests.select_testpaths(tree_only)[0] == []

        paths = check_tests.changed_paths(cwd=repo, base=base)
        testpaths, reason = check_tests.select_testpaths(paths)
        assert testpaths == check_tests._all_testpaths()
        assert "kernel" in reason

    def test_three_dot_diff_ignores_base_advancing_past_the_branch_point(
        self, tmp_path: Path
    ) -> None:
        """A base that moved on must not report its own commits as ours."""
        repo = tmp_path / "repo"
        git = _init_repo(repo)
        base = _head(repo, git)
        _commit(repo, git, "sdk/src/sdk/config.py", "x = 1\n", "branch work")
        # Advance the base ref past the branch point on a separate branch.
        subprocess.run([*git, "checkout", "-q", "-b", "other", base], check=True)
        _commit(repo, git, "runtime/src/runtime/other.py", "y = 1\n", "base work")
        subprocess.run([*git, "checkout", "-q", "-"], check=True)

        paths = check_tests.committed_paths(cwd=repo, base="other")
        assert "sdk/src/sdk/config.py" in paths
        assert "runtime/src/runtime/other.py" not in paths

    def test_explicit_base_must_resolve(self, tmp_path: Path) -> None:
        """A named base that does not exist is an error, not a silent fallback."""
        repo = tmp_path / "repo"
        _init_repo(repo)
        with pytest.raises(RuntimeError):
            check_tests.resolve_base(cwd=repo, explicit="no-such-ref")

    def test_resolve_base_returns_none_when_no_candidate_exists(self, tmp_path: Path) -> None:
        """No origin/main and no main means the base is unknown, not empty."""
        repo = tmp_path / "repo"
        _init_repo(repo)
        subprocess.run(["git", "-C", str(repo), "branch", "-m", "trunk"], check=True)
        assert check_tests.resolve_base(cwd=repo) is None

    def test_stale_candidate_does_not_widen_the_delta(self, tmp_path: Path) -> None:
        """A stale ``origin/main`` must not report other commits as this branch's.

        The branch point is the newest merge-base across candidates. Taking the
        first candidate's merge-base instead would reach back past the branch
        point and pull unrelated commits into the delta, escalating on paths
        this branch never touched.
        """
        repo = tmp_path / "repo"
        git = _init_repo(repo)
        stale = _head(repo, git)
        # ``main`` advances past the stale point with someone else's commit.
        _commit(repo, git, "kernel/pyproject.toml", "[project]\n", "unrelated")
        branch_point = _head(repo, git)
        # This branch's own work, branched from the advanced ``main``.
        subprocess.run([*git, "checkout", "-q", "-b", "feature"], check=True)
        _commit(repo, git, "sdk/src/sdk/config.py", "x = 1\n", "branch work")
        # ``origin/main`` is left behind at the stale commit.
        subprocess.run([*git, "update-ref", "refs/remotes/origin/main", stale], check=True)

        base = check_tests.resolve_base(cwd=repo)
        assert base == branch_point
        paths = check_tests.committed_paths(cwd=repo, base=base)
        assert "sdk/src/sdk/config.py" in paths
        assert "kernel/pyproject.toml" not in paths


class TestCommand:
    """The emitted command is a real, offline pytest invocation."""

    def test_argv_runs_offline_pytest(self) -> None:
        argv = check_tests.pytest_argv(["sdk/tests"])
        assert argv[1:] == [
            "-m",
            "pytest",
            "sdk/tests",
            "-m",
            "not live",
            "--tb=short",
            "-q",
        ]
        assert argv[0] == sys.executable

    def test_dry_run_prints_without_running(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = check_tests.main(["--paths", "sdk/src/sdk/config.py", "--dry-run"])
        assert code == 0
        captured = capsys.readouterr()
        assert "-m pytest" in captured.out
        assert "affected package(s): sdk" in captured.err

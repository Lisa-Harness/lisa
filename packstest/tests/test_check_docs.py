"""Tests for the documentation quality gate (``scripts/check_docs.py``).

The gate must not fire on the corpus as it stands, and it must fire on the
defects it claims to catch. Both halves are asserted here so a change to
the checker cannot silently widen or narrow the corpus.

The script is loaded by path rather than imported as a package: it lives in
``scripts/``, which is not an importable package in this workspace.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "check_docs.py"


def _load_module() -> ModuleType:
    """Import ``scripts/check_docs.py`` as a module."""
    spec = importlib.util.spec_from_file_location("check_docs_under_test", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check_docs = _load_module()


class TestCorpusSelection:
    """The corpus is the maintained documentation, not runtime content."""

    def test_guides_and_readmes_are_included(self) -> None:
        assert check_docs._is_documentation("docs/README.md")
        assert check_docs._is_documentation("docs/SPEC/THREAD_STORE.md")
        assert check_docs._is_documentation("README.md")
        assert check_docs._is_documentation("CHANGELOG.md")
        assert check_docs._is_documentation("sdk/AUTHORING.md")
        assert check_docs._is_documentation("kernel/README.md")

    def test_runtime_and_agent_content_is_excluded(self) -> None:
        # Pack content, test fixtures, and Copilot customization files are
        # inputs to the runtime or the agent host, not documentation.
        assert not check_docs._is_documentation(
            "sdk/packs/core/src/sdk/packs/core/modes/default/_mode.md"
        )
        assert not check_docs._is_documentation(
            "sdk/packs/core/src/sdk/packs/core/content/prompts/system-prompt.md"
        )
        assert not check_docs._is_documentation(
            "kernel/src/kernel/testing/fixtures/prompts/sub_turn/budget_exhausted.md"
        )
        assert not check_docs._is_documentation("sdk/packs/recovery/tests/e2e/session_context.md")
        assert not check_docs._is_documentation(".github/copilot-instructions.md")
        assert not check_docs._is_documentation(
            ".github/instructions/engineering/python.instructions.md"
        )

    def test_readme_inside_a_package_source_tree_is_not_documentation(self) -> None:
        # A README under a package source tree is a runtime asset, not a
        # guide. The CI path filter re-includes these so a change to one
        # still runs the full matrix, and the gate must not treat one as
        # documentation either -- otherwise the two lists drift.
        assert not check_docs._is_documentation("sdk/src/sdk/router/README.md")
        assert not check_docs._is_documentation(
            "sdk/packs/recovery/src/sdk/packs/recovery/README.md"
        )
        assert not check_docs._is_documentation(
            "sdk/packs/core/src/sdk/packs/core/modes/default/README.md"
        )

    def test_ci_path_filter_and_gate_agree_on_the_corpus(self) -> None:
        # The gate's corpus and the CI path filter must not disagree. A
        # file the gate checks but CI skips would never be protected; a
        # file CI runs the matrix for but the gate ignores would go
        # unchecked.
        ci_included = _ci_paths_included()
        for rel_path in check_docs._git_tracked_markdown(_REPO_ROOT):
            if not rel_path.endswith(".md"):
                continue
            is_doc = check_docs._is_documentation(rel_path)
            runs_ci = ci_included(rel_path)
            assert is_doc is not runs_ci, (
                f"{rel_path}: gate treats it as documentation={is_doc}, "
                f"CI runs the matrix for it={runs_ci}"
            )


def _ci_paths_included():
    """Return the predicate the CI path filter implements.

    The patterns are ordered and the last match wins, matching GitHub's
    ``paths`` semantics: a negative pattern after a positive one excludes
    the path, and a positive pattern after a negative one includes it.
    """
    patterns = [
        "**",
        "!docs/**",
        "!*.md",
        "!**/README.md",
        "!**/AUTHORING.md",
        "sdk/src/**",
        "sdk/packs/*/src/**",
        "sdk/packs/*/tests/**",
        "kernel/src/**",
        "mcp_server/src/**",
        "runtime/src/**",
        "packstest/src/**",
        "packstest/tests/**",
        "e2etest/src/**",
        "e2etest/tests/**",
        ".github/**",
    ]

    def _match(path: str, pattern: str) -> bool:
        return _fnmatch(path, pattern)

    def included(path: str) -> bool:
        result = False
        for pattern in patterns:
            negative = pattern.startswith("!")
            candidate = pattern[1:] if negative else pattern
            if _match(path, candidate):
                result = not negative
        return result

    return included


def _fnmatch(path: str, pattern: str) -> bool:
    """Match a repo-relative path against a GitHub-style glob.

    GitHub's filter patterns differ from :mod:`fnmatch`: ``*`` matches
    zero or more characters but never a ``/``, while ``**`` matches zero
    or more of any character.
    """
    parts = ["^"]
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "*":
            if pattern.startswith("**", index):
                parts.append(".*")
                index += 2
                continue
            parts.append("[^/]*")
            index += 1
            continue
        if char == "?":
            parts.append("[^/]")
            index += 1
            continue
        parts.append(re.escape(char))
        index += 1
    parts.append("$")
    return re.match("".join(parts), path) is not None


class TestDetectedDefects:
    """Each check fires on the defect it claims to catch."""

    @pytest.fixture
    def repo(self, tmp_path: Path) -> Path:
        """Return a minimal repository root with one documentation file."""
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "guide.md").write_text("# Guide\n", encoding="utf-8")
        return tmp_path

    def _rules(self, repo: Path) -> set[str]:
        """Return the rules reported for the fixture's documentation file."""
        findings = check_docs.check_file("docs/guide.md", repo, {})
        return {finding.rule for finding in findings}

    def test_broken_link_target_is_reported(self, repo: Path) -> None:
        (repo / "docs" / "guide.md").write_text(
            "# Guide\n\nSee [missing](missing.md).\n", encoding="utf-8"
        )
        assert "link-target" in self._rules(repo)

    def test_broken_anchor_is_reported(self, repo: Path) -> None:
        (repo / "docs" / "guide.md").write_text(
            "# Guide\n\nSee [section](#no-such-heading).\n", encoding="utf-8"
        )
        assert "link-anchor" in self._rules(repo)

    def test_valid_link_and_anchor_are_accepted(self, repo: Path) -> None:
        (repo / "docs" / "guide.md").write_text(
            "# Guide\n\n## Real Section\n\nSee [section](#real-section) and [other](other.md).\n",
            encoding="utf-8",
        )
        (repo / "docs" / "other.md").write_text("# Other\n", encoding="utf-8")
        assert self._rules(repo) == set()

    def test_unlabelled_code_fence_is_reported(self, repo: Path) -> None:
        (repo / "docs" / "guide.md").write_text("# Guide\n\n```\ncode\n```\n", encoding="utf-8")
        assert "fence-language" in self._rules(repo)

    def test_labelled_code_fence_is_accepted(self, repo: Path) -> None:
        (repo / "docs" / "guide.md").write_text(
            "# Guide\n\n```python\ncode\n```\n", encoding="utf-8"
        )
        assert self._rules(repo) == set()

    def test_unclosed_code_fence_is_reported(self, repo: Path) -> None:
        (repo / "docs" / "guide.md").write_text("# Guide\n\n```python\ncode\n", encoding="utf-8")
        assert "fence-unclosed" in self._rules(repo)

    def test_missing_final_newline_is_reported(self, repo: Path) -> None:
        (repo / "docs" / "guide.md").write_text("# Guide", encoding="utf-8")
        assert "final-newline" in self._rules(repo)

    def test_trailing_whitespace_is_reported(self, repo: Path) -> None:
        (repo / "docs" / "guide.md").write_text(
            "# Guide\n\ntext with trailing space  \n", encoding="utf-8"
        )
        assert "trailing-whitespace" in self._rules(repo)

    def test_list_without_a_leading_blank_line_is_reported(self, repo: Path) -> None:
        (repo / "docs" / "guide.md").write_text(
            "# Guide\n\nA paragraph.\n- first item\n", encoding="utf-8"
        )
        assert "list-spacing" in self._rules(repo)

    def test_wrapped_list_items_are_not_reported(self, repo: Path) -> None:
        # A list item's continuation lines are indented and must not be
        # mistaken for a new list that lacks a leading blank line.
        (repo / "docs" / "guide.md").write_text(
            "# Guide\n\n1. first item that wraps\n   onto a second line.\n2. second item\n",
            encoding="utf-8",
        )
        assert "list-spacing" not in self._rules(repo)

    def test_code_span_heading_keeps_its_anchor(self, repo: Path) -> None:
        # GitHub strips the period out of a code-span heading, so the
        # anchor is ``contextlanes``. The gate must agree with that.
        (repo / "docs" / "guide.md").write_text(
            "# Guide\n\n### `context.lanes`\n\nSee [lanes](#contextlanes).\n", encoding="utf-8"
        )
        assert self._rules(repo) == set()

    def test_code_span_heading_rejects_a_wrong_anchor(self, repo: Path) -> None:
        (repo / "docs" / "guide.md").write_text(
            "# Guide\n\n### `context.lanes`\n\nSee [lanes](#context-lanes).\n", encoding="utf-8"
        )
        assert "link-anchor" in self._rules(repo)

    def test_setext_heading_anchor_is_resolved(self, repo: Path) -> None:
        # GitHub renders an anchor for a setext heading too.
        (repo / "docs" / "guide.md").write_text(
            "# Guide\n\nSetext Title\n============\n\nSee [t](#setext-title).\n",
            encoding="utf-8",
        )
        assert self._rules(repo) == set()

    def test_setext_heading_anchor_is_checked(self, repo: Path) -> None:
        (repo / "docs" / "guide.md").write_text(
            "# Guide\n\nSetext Title\n============\n\nSee [t](#wrong).\n", encoding="utf-8"
        )
        assert "link-anchor" in self._rules(repo)

    def test_anchor_into_a_file_without_headings_is_reported(self, repo: Path) -> None:
        # The target has no headings at all, so the fragment cannot match.
        (repo / "docs" / "plain.md").write_text("Just prose.\n", encoding="utf-8")
        (repo / "docs" / "guide.md").write_text(
            "# Guide\n\nSee [plain](plain.md#some-heading).\n", encoding="utf-8"
        )
        assert "link-anchor" in self._rules(repo)

    def test_hard_tab_inside_a_code_block_is_allowed(self, repo: Path) -> None:
        # A tab is normal inside a Go or Makefile example.
        (repo / "docs" / "guide.md").write_text(
            "# Guide\n\n```go\nfunc main() {\n\tprintln(1)\n}\n```\n", encoding="utf-8"
        )
        assert self._rules(repo) == set()

    def test_hard_tab_outside_a_code_block_is_reported(self, repo: Path) -> None:
        (repo / "docs" / "guide.md").write_text(
            "# Guide\n\n\tindented with a tab\n", encoding="utf-8"
        )
        assert "tab-indent" in self._rules(repo)

    def test_trailing_whitespace_outside_a_code_block_is_reported(self, repo: Path) -> None:
        (repo / "docs" / "guide.md").write_text(
            "# Guide\n\ntext with trailing space  \n", encoding="utf-8"
        )
        assert "trailing-whitespace" in self._rules(repo)

    def test_external_links_are_not_resolved(self, repo: Path) -> None:
        (repo / "docs" / "guide.md").write_text(
            "# Guide\n\nSee [site](https://example.com/page) and [mail](mailto:a@b.c).\n",
            encoding="utf-8",
        )
        assert self._rules(repo) == set()

    def test_sibling_repository_link_is_not_resolved(self, repo: Path) -> None:
        # The workspace holds sibling repositories next to this one, and the
        # corpus links to them (``../lisa-cli``). Those links point outside
        # this repository, so the gate cannot say whether they resolve, and
        # the CI checkout does not contain the sibling repositories at all.
        (repo / "docs" / "guide.md").write_text(
            "# Guide\n\nSee [cli](../../lisa-cli).\n", encoding="utf-8"
        )
        assert self._rules(repo) == set()

    def test_missing_in_repository_link_is_still_reported(self, repo: Path) -> None:
        (repo / "docs" / "guide.md").write_text(
            "# Guide\n\nSee [gone](gone.md).\n", encoding="utf-8"
        )
        assert "link-target" in self._rules(repo)

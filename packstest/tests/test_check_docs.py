"""Tests for the documentation quality gate (``scripts/check_docs.py``).

The gate must not fire on the corpus as it stands, and it must fire on the
defects it claims to catch. Both halves are asserted here so a change to
the checker cannot silently widen or narrow the corpus.

The script is loaded by path rather than imported as a package: it lives in
``scripts/``, which is not an importable package in this workspace.
"""

from __future__ import annotations

import importlib.util
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

    def test_repo_corpus_is_clean(self) -> None:
        assert check_docs.check_corpus(_REPO_ROOT) == []

    def test_guides_and_readmes_are_included(self) -> None:
        assert check_docs._is_documentation("docs/README.md")
        assert check_docs._is_documentation("docs/SPEC/THREAD_STORE.md")
        assert check_docs._is_documentation("README.md")
        assert check_docs._is_documentation("CHANGELOG.md")
        assert check_docs._is_documentation("sdk/AUTHORING.md")
        assert check_docs._is_documentation("sdk/src/sdk/router/README.md")

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

    def test_external_links_are_not_resolved(self, repo: Path) -> None:
        (repo / "docs" / "guide.md").write_text(
            "# Guide\n\nSee [site](https://example.com/page) and [mail](mailto:a@b.c).\n",
            encoding="utf-8",
        )
        assert self._rules(repo) == set()

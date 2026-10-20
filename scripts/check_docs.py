#!/usr/bin/env python3
"""Check the maintained documentation corpus for structural defects.

DESIGN — The repository keeps one documentation corpus (the guides under
``docs/``, the package READMEs, and the root-level Markdown). A doc-only
change must not be able to break a link, a heading anchor, or a code fence
without a job noticing, so this script checks the properties that the
corpus already satisfies today:

- **Relative links resolve.** Every relative link target must exist in the
  working tree, and every ``#anchor`` fragment must match a heading in the
  file it points at. Absolute URLs, ``mailto:`` links, and pure fragments
  are resolved against the current file.
- **Code fences are labelled and balanced.** Every fenced block declares a
  language and every fence is closed.
- **Files end with exactly one newline.** No trailing blank lines and no
  missing final newline.
- **No trailing whitespace** on any line, and no tab indentation.
- **Headings and lists are spaced.** A heading needs a blank line above
  it, and a list needs a blank line before it (unless it continues a list).

Checks are deliberately structural rather than stylistic. Line length,
heading duplication, and table spacing vary across the corpus today, and
a gate that flags them would fail on the existing corpus instead of on
regressions. Those remain authoring conventions enforced by review.

Markdown that is *not* documentation is out of scope: pack content
(``_mode.md``, ``*.operation.md``, prompt and session templates), test
fixtures, and Copilot customization files are inputs to the runtime or
the agent host, and are checked by the tests that load them.

Usage:
    uv run python scripts/check_docs.py [--repo-root PATH]
"""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Corpus selection
# ---------------------------------------------------------------------------

#: Directories whose Markdown is documentation.
_DOC_DIRS = ("docs/",)

#: File names that are documentation wherever they live.
_DOC_NAMES = ("README.md", "AUTHORING.md")

#: Root-level Markdown files that are documentation.
_ROOT_DOC_FILES = (
    "CHANGELOG.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "README.md",
    "SECURITY.md",
)


def _git_tracked_markdown(repo_root: Path) -> list[str]:
    """Return the repository's tracked Markdown files as POSIX-relative paths."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "*.md"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line]


def _is_documentation(rel_path: str) -> bool:
    """Report whether a repo-relative Markdown path belongs to the doc corpus.

    Documentation is the guides under ``docs/``, the root-level policy
    files, and any package README or authoring guide. Everything else is
    an input to the runtime or the agent host: pack content (modes,
    operations, prompts, sessions), test fixtures, and Copilot
    customization files. Those are covered by the tests that load them.
    """
    if rel_path.startswith(_DOC_DIRS):
        return True
    if "/" not in rel_path:
        return rel_path in _ROOT_DOC_FILES
    return rel_path.rsplit("/", 1)[1] in _DOC_NAMES


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Finding:
    """One defect in one documentation file."""

    path: str
    line: int
    rule: str
    message: str

    def render(self) -> str:
        """Format the finding as a one-line report."""
        return f"{self.path}:{self.line}: [{self.rule}] {self.message}"


# ---------------------------------------------------------------------------
# Markdown primitives
# ---------------------------------------------------------------------------

#: A fenced code block opener or closer.
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*(.*)$")

#: An ATX heading.
_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")

#: A list item at any indentation.
_LIST_RE = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+")

#: An inline link or image.
_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]*)\)")

#: A scheme-bearing URL that is never resolved against the working tree.
_URL_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")

#: Inline HTML tags.
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][A-Za-z0-9-]*(?:\s[^<>]*)?/?>")


def _iter_code_regions(lines: list[str]) -> list[tuple[int, int]]:
    """Return the ``(start, end)`` line index of each fenced block.

    ``start`` is the opener line and ``end`` is the line after the closer, so
    ``lines[start:end]`` is the block's content. Unbalanced fences extend to
    the end of the file.
    """
    regions: list[tuple[int, int]] = []
    open_at: int | None = None
    open_marker = ""
    for index, line in enumerate(lines):
        match = _FENCE_RE.match(line)
        if match is None:
            continue
        marker, info = match.group(1), match.group(2).strip()
        if open_at is None:
            open_at = index
            open_marker = marker
        elif marker[0] == open_marker[0] and len(marker) >= len(open_marker) and not info:
            regions.append((open_at, index + 1))
            open_at = None
    if open_at is not None:
        regions.append((open_at, len(lines)))
    return regions


def _blank_fences(lines: list[str]) -> list[str]:
    """Blank out fenced code blocks, leaving everything else untouched.

    Inline code spans are preserved: a heading may contain a code span that
    is part of its title (``### `context.lanes```), and stripping it would
    change the anchor the heading produces.
    """
    blanked = list(lines)
    for start, end in _iter_code_regions(lines):
        for index in range(start, end):
            blanked[index] = ""
    return blanked


def _strip_code(lines: list[str]) -> list[str]:
    """Blank out fenced blocks and inline code spans.

    Returns one string per input line, with code content removed so that
    link, heading, and spacing checks do not fire on examples.
    """
    return [re.sub(r"`[^`]*`", "", line) for line in _blank_fences(lines)]


#: Characters GitHub removes from a heading when building its anchor.
#:
#: GitHub strips everything that is not a word character, a hyphen, or a
#: space, then replaces each space with a hyphen. The set is transcribed
#: from ``github-slugger``'s reference regex, which is the implementation
#: GitHub's own Markdown renderer follows.
_SLUG_STRIP_RE = re.compile(
    r"[^\w\- ]"
    r"|[ -⁯⸀-⹿"
    r"\\'‘’‚‛“”„‟"
    r"†‡•‣‧‰"
    r"′″‴‵‶‷‸‹›"
    r"※"
    r"⁇⁈⁉"
    r"₠-₿"
    r"⌈-⌋"
    r"〈-〉"
    r"❨-❵"
    r"⟅-⟯"
    r"⦀-⧿"
    r"Ⳁ-⵿"
    r"⸀-⹿"
    r"、-〿"
    r"぀-ゟ"
    r"゠-ヺー-ヿ"
    r"㄀-ㄯ"
    r"㈀-㋿"
    r"㐀-䶿"
    r"一-鿿"
    r"ꀀ-꓏"
    r"ꥠ-꥿"
    r"가-힯"
    r"豈-﫿"
    r"\U0001d300-\U0001d7ff"
    r"\U0001f000-\U0001ffff]"
)


def _slugify(heading: str) -> str:
    """Return the GitHub-style anchor slug for a heading's text.

    Mirrors ``github-slugger``: lowercase the text, drop every character
    that is not a word character, a hyphen, or a space, then replace each
    space with a hyphen. Underscores survive, and a run of punctuation
    collapses into a single hyphen only because the spaces around it
    remain.
    """
    text = heading.strip().lower()
    return _SLUG_STRIP_RE.sub("", text).replace(" ", "-")


def _heading_anchors(path: Path) -> set[str]:
    """Return the anchor slugs a Markdown file exposes."""
    try:
        lines = path.read_text(encoding="utf-8").split("\n")
    except (OSError, UnicodeDecodeError):
        return set()
    anchors: set[str] = set()
    for line in _blank_fences(lines):
        match = _HEADING_RE.match(line)
        if match is not None:
            anchors.add(_slugify(match.group(2)))
    return anchors


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _normalize_target(raw: str) -> str:
    """Return the link target from a Markdown link's destination text."""
    target = raw.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    return target.split(" ")[0]


def _is_external(target: str) -> bool:
    """Report whether a link target points outside the working tree."""
    return bool(_URL_SCHEME_RE.match(target)) or target.startswith("//")


def _resolve_target(rel_path: str, path_part: str) -> str:
    """Resolve a link's path part to a repo-relative path."""
    if not path_part:
        return rel_path
    return str(Path((Path(rel_path).parent / path_part).as_posix()))


def _check_link(
    rel_path: str,
    index: int,
    target: str,
    repo_root: Path,
    anchors_by_path: dict[str, set[str]],
) -> Finding | None:
    """Check one link target, returning a finding when it does not resolve."""
    if not target or _is_external(target):
        return None
    path_part, _, fragment = target.partition("#")
    resolved = _resolve_target(rel_path, path_part)
    if not (repo_root / resolved).exists():
        return Finding(rel_path, index, "link-target", f"link target does not exist: {target}")
    if not fragment:
        return None
    known = anchors_by_path.get(resolved)
    if known is None:
        known = _heading_anchors(repo_root / resolved)
        anchors_by_path[resolved] = known
    if known and fragment.lower() not in known:
        return Finding(
            rel_path,
            index,
            "link-anchor",
            f"anchor #{fragment} not found in {resolved}",
        )
    return None


def _check_links(
    rel_path: str,
    lines: list[str],
    repo_root: Path,
    anchors_by_path: dict[str, set[str]],
) -> list[Finding]:
    """Check that relative links and anchor fragments resolve."""
    findings: list[Finding] = []
    for index, line in enumerate(_blank_fences(lines), start=1):
        for match in _LINK_RE.finditer(line):
            target = _normalize_target(match.group(1))
            finding = _check_link(rel_path, index, target, repo_root, anchors_by_path)
            if finding is not None:
                findings.append(finding)
    return findings


def _check_fences(rel_path: str, lines: list[str]) -> list[Finding]:
    """Check that fenced code blocks declare a language and are balanced."""
    findings: list[Finding] = []
    open_at: int | None = None
    open_marker = ""
    for index, line in enumerate(lines, start=1):
        match = _FENCE_RE.match(line)
        if match is None:
            continue
        marker, info = match.group(1), match.group(2).strip()
        if open_at is None:
            open_at = index
            open_marker = marker
            if not info:
                findings.append(
                    Finding(rel_path, index, "fence-language", "code fence has no language")
                )
        elif marker[0] == open_marker[0] and len(marker) >= len(open_marker) and not info:
            open_at = None
    if open_at is not None:
        findings.append(Finding(rel_path, open_at, "fence-unclosed", "code fence is never closed"))
    return findings


def _check_spacing(rel_path: str, lines: list[str]) -> list[Finding]:
    """Check blank-line spacing around headings and lists, and trailing whitespace.

    Whitespace is measured on the raw lines so that a trailing space inside a
    code block is still reported. Structural spacing is measured on the
    code-stripped lines so that an example cannot fake a heading or a list.
    """
    findings: list[Finding] = []
    for index, line in enumerate(lines, start=1):
        if line != line.rstrip():
            findings.append(
                Finding(rel_path, index, "trailing-whitespace", "line has trailing whitespace")
            )
        if line.startswith("\t") or re.match(r"^ +\t", line):
            findings.append(Finding(rel_path, index, "tab-indent", "line is indented with a tab"))

    stripped = _strip_code(lines)
    for index, line in enumerate(stripped, start=1):
        if _HEADING_RE.match(line) is None:
            continue
        if index == 1:
            continue
        if stripped[index - 2].strip():
            findings.append(
                Finding(
                    rel_path, index, "heading-spacing", "heading is not preceded by a blank line"
                )
            )

    findings += _check_list_spacing(rel_path, stripped)
    return findings


def _check_list_spacing(rel_path: str, stripped: list[str]) -> list[Finding]:
    """Check that each list is preceded by a blank line.

    A list may follow a blank line, a heading, or another list item. A list
    that directly follows a paragraph is reported. Indented lines continue
    the current list item, so they never start a new list.
    """
    findings: list[Finding] = []
    previous = ""  # The previous non-blank line, or "" at the start of the file.
    separated = True  # Whether a blank line follows ``previous``.
    for index, line in enumerate(stripped):
        if not line.strip():
            separated = True
            continue
        is_item = _LIST_RE.match(line) is not None
        if is_item and previous and not separated:
            follows_prose = (
                _LIST_RE.match(previous) is None
                and _HEADING_RE.match(previous) is None
                and previous[:1] not in (" ", "\t")
            )
            if follows_prose:
                findings.append(
                    Finding(
                        rel_path,
                        index + 1,
                        "list-spacing",
                        "list is not preceded by a blank line",
                    )
                )
        previous = line
        separated = False
    return findings


def _check_file_end(rel_path: str, text: str) -> list[Finding]:
    """Check the file ends with exactly one newline."""
    if not text:
        return []
    if not text.endswith("\n"):
        return [
            Finding(rel_path, text.count("\n") + 1, "final-newline", "file has no final newline")
        ]
    if text.endswith("\n\n"):
        return [Finding(rel_path, text.count("\n"), "final-newline", "file ends with a blank line")]
    return []


def check_file(
    rel_path: str,
    repo_root: Path,
    anchors_by_path: dict[str, set[str]],
) -> list[Finding]:
    """Run every documentation check against one corpus file."""
    path = repo_root / rel_path
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [Finding(rel_path, 1, "unreadable", f"cannot read file: {exc}")]
    lines = text.split("\n")
    findings: list[Finding] = []
    findings += _check_links(rel_path, lines, repo_root, anchors_by_path)
    findings += _check_fences(rel_path, lines)
    findings += _check_spacing(rel_path, lines)
    findings += _check_file_end(rel_path, text)
    return findings


def check_corpus(repo_root: Path) -> list[Finding]:
    """Run every documentation check against the whole doc corpus."""
    corpus = [path for path in _git_tracked_markdown(repo_root) if _is_documentation(path)]
    anchors_by_path: dict[str, set[str]] = {}
    findings: list[Finding] = []
    for rel_path in corpus:
        findings += check_file(rel_path, repo_root, anchors_by_path)
    return findings


def main() -> int:
    """Run the documentation gate and report findings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root to check (default: the checkout containing this script)",
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()

    findings = check_corpus(repo_root)
    if not findings:
        print("docs gate: no findings.")
        return 0
    for finding in findings:
        print(finding.render())
    print(f"\ndocs gate: {len(findings)} finding(s).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

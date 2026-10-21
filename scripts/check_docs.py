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
from collections.abc import Iterator
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

#: Package source trees whose Markdown is runtime or agent input, never
#: documentation. These mirror the CI path filter's re-included paths.
_RUNTIME_SOURCE_PREFIXES = (
    "e2etest/src/",
    "e2etest/tests/",
    "kernel/src/",
    "mcp_server/src/",
    "packstest/src/",
    "packstest/tests/",
    "runtime/src/",
    "sdk/packs/core/src/",
    "sdk/packs/introspect/src/",
    "sdk/packs/memory/src/",
    "sdk/packs/recovery/src/",
    "sdk/packs/recovery/tests/",
    "sdk/src/",
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

    The rule mirrors the CI path filter, so a file is never documentation
    for one and behavior for the other. A README inside a package source
    tree is a runtime asset, not a guide: the catalog scans the
    ``modes/``, ``operations/``, and ``tools/`` directories under each
    content root, and a README there would be read as authored content.
    """
    if rel_path.startswith(_DOC_DIRS):
        return True
    if "/" not in rel_path:
        return rel_path in _ROOT_DOC_FILES
    if rel_path.startswith(_RUNTIME_SOURCE_PREFIXES):
        return False
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

#: A setext heading underline (``Title`` followed by ``===`` or ``---``).
_SETEXT_RE = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")

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
    "[\x00-\x1f\x21-\x2c\x2e-\x2f\x3a-\x40\x5b-\x5e\x60\x7b-\xa9\xab-\xb4\xb6-\xb9\xbb-\xbf\xd7\xf7\u02c2-\u02c5\u02d2-\u02df\u02e5-\u02eb\u02ed\u02ef-\u02ff\u0375\u0378-\u0379\u037e\u0380-\u0385\u0387\u038b\u038d\u03a2\u03f6\u0482\u0530\u0557-\u0558\u055a-\u055f\u0589-\u0590\u05be\u05c0\u05c3\u05c6\u05c8-\u05cf\u05eb-\u05ee\u05f3-\u060f\u061b-\u061f\u066a-\u066d\u06d4\u06dd-\u06de\u06e9\u06fd-\u06fe\u0700-\u070f\u074b-\u074c\u07b2-\u07bf\u07f6-\u07f9\u07fb-\u07fc\u07fe-\u07ff\u082e-\u083f\u085c-\u085f\u086b-\u089f\u08b5\u08c8-\u08d2\u08e2\u0964-\u0965\u0970\u0984\u098d-\u098e\u0991-\u0992\u09a9\u09b1\u09b3-\u09b5\u09ba-\u09bb\u09c5-\u09c6\u09c9-\u09ca\u09cf-\u09d6\u09d8-\u09db\u09de\u09e4-\u09e5\u09f2-\u09fb\u09fd\u09ff-\u0a00\u0a04\u0a0b-\u0a0e\u0a11-\u0a12\u0a29\u0a31\u0a34\u0a37\u0a3a-\u0a3b\u0a3d\u0a43-\u0a46\u0a49-\u0a4a\u0a4e-\u0a50\u0a52-\u0a58\u0a5d\u0a5f-\u0a65\u0a76-\u0a80\u0a84\u0a8e\u0a92\u0aa9\u0ab1\u0ab4\u0aba-\u0abb\u0ac6\u0aca\u0ace-\u0acf\u0ad1-\u0adf\u0ae4-\u0ae5\u0af0-\u0af8\u0b00\u0b04\u0b0d-\u0b0e\u0b11-\u0b12\u0b29\u0b31\u0b34\u0b3a-\u0b3b\u0b45-\u0b46\u0b49-\u0b4a\u0b4e-\u0b54\u0b58-\u0b5b\u0b5e\u0b64-\u0b65\u0b70\u0b72-\u0b81\u0b84\u0b8b-\u0b8d\u0b91\u0b96-\u0b98\u0b9b\u0b9d\u0ba0-\u0ba2\u0ba5-\u0ba7\u0bab-\u0bad\u0bba-\u0bbd\u0bc3-\u0bc5\u0bc9\u0bce-\u0bcf\u0bd1-\u0bd6\u0bd8-\u0be5\u0bf0-\u0bff\u0c0d\u0c11\u0c29\u0c3a-\u0c3c\u0c45\u0c49\u0c4e-\u0c54\u0c57\u0c5b-\u0c5f\u0c64-\u0c65\u0c70-\u0c7f\u0c84\u0c8d\u0c91\u0ca9\u0cb4\u0cba-\u0cbb\u0cc5\u0cc9\u0cce-\u0cd4\u0cd7-\u0cdd\u0cdf\u0ce4-\u0ce5\u0cf0\u0cf3-\u0cff\u0d0d\u0d11\u0d45\u0d49\u0d4f-\u0d53\u0d58-\u0d5e\u0d64-\u0d65\u0d70-\u0d79\u0d80\u0d84\u0d97-\u0d99\u0db2\u0dbc\u0dbe-\u0dbf\u0dc7-\u0dc9\u0dcb-\u0dce\u0dd5\u0dd7\u0de0-\u0de5\u0df0-\u0df1\u0df4-\u0e00\u0e3b-\u0e3f\u0e4f\u0e5a-\u0e80\u0e83\u0e85\u0e8b\u0ea4\u0ea6\u0ebe-\u0ebf\u0ec5\u0ec7\u0ece-\u0ecf\u0eda-\u0edb\u0ee0-\u0eff\u0f01-\u0f17\u0f1a-\u0f1f\u0f2a-\u0f34\u0f36\u0f38\u0f3a-\u0f3d\u0f48\u0f6d-\u0f70\u0f85\u0f98\u0fbd-\u0fc5\u0fc7-\u0fff\u104a-\u104f\u109e-\u109f\u10c6\u10c8-\u10cc\u10ce-\u10cf\u10fb\u1249\u124e-\u124f\u1257\u1259\u125e-\u125f\u1289\u128e-\u128f\u12b1\u12b6-\u12b7\u12bf\u12c1\u12c6-\u12c7\u12d7\u1311\u1316-\u1317\u135b-\u135c\u1360-\u137f\u1390-\u139f\u13f6-\u13f7\u13fe-\u1400\u166d-\u166e\u1680\u169b-\u169f\u16eb-\u16ed\u16f9-\u16ff\u170d\u1715-\u171f\u1735-\u173f\u1754-\u175f\u176d\u1771\u1774-\u177f\u17d4-\u17d6\u17d8-\u17db\u17de-\u17df\u17ea-\u180a\u180e-\u180f\u181a-\u181f\u1879-\u187f\u18ab-\u18af\u18f6-\u18ff\u191f\u192c-\u192f\u193c-\u1945\u196e-\u196f\u1975-\u197f\u19ac-\u19af\u19ca-\u19cf\u19da-\u19ff\u1a1c-\u1a1f\u1a5f\u1a7d-\u1a7e\u1a8a-\u1a8f\u1a9a-\u1aa6\u1aa8-\u1aaf\u1ac1-\u1aff\u1b4c-\u1b4f\u1b5a-\u1b6a\u1b74-\u1b7f\u1bf4-\u1bff\u1c38-\u1c3f\u1c4a-\u1c4c\u1c7e-\u1c7f\u1c89-\u1c8f\u1cbb-\u1cbc\u1cc0-\u1ccf\u1cd3\u1cfb-\u1cff\u1dfa\u1f16-\u1f17\u1f1e-\u1f1f\u1f46-\u1f47\u1f4e-\u1f4f\u1f58\u1f5a\u1f5c\u1f5e\u1f7e-\u1f7f\u1fb5\u1fbd\u1fbf-\u1fc1\u1fc5\u1fcd-\u1fcf\u1fd4-\u1fd5\u1fdc-\u1fdf\u1fed-\u1ff1\u1ff5\u1ffd-\u203e\u2041-\u2053\u2055-\u2070\u2072-\u207e\u2080-\u208f\u209d-\u20cf\u20f1-\u2101\u2103-\u2106\u2108-\u2109\u2114\u2116-\u2118\u211e-\u2123\u2125\u2127\u2129\u212e\u213a-\u213b\u2140-\u2144\u214a-\u214d\u214f-\u215f\u2189-\u24b5\u24ea-\u2bff\u2c2f\u2c5f\u2ce5-\u2cea\u2cf4-\u2cff\u2d26\u2d28-\u2d2c\u2d2e-\u2d2f\u2d68-\u2d6e\u2d70-\u2d7e\u2d97-\u2d9f\u2da7\u2daf\u2db7\u2dbf\u2dc7\u2dcf\u2dd7\u2ddf\u2e00-\u2e2e\u2e30-\u3004\u3008-\u3020\u3030\u3036-\u3037\u303d-\u3040\u3097-\u3098\u309b-\u309c\u30a0\u30fb\u3100-\u3104\u3130\u318f-\u319f\u31c0-\u31ef\u3200-\u33ff\u4dc0-\u4dff\u9ffd-\u9fff\ua48d-\ua4cf\ua4fe-\ua4ff\ua60d-\ua60f\ua62c-\ua63f\ua673\ua67e\ua6f2-\ua716\ua720-\ua721\ua789-\ua78a\ua7c0-\ua7c1\ua7cb-\ua7f4\ua828-\ua82b\ua82d-\ua83f\ua874-\ua87f\ua8c6-\ua8cf\ua8da-\ua8df\ua8f8-\ua8fa\ua8fc\ua92e-\ua92f\ua954-\ua95f\ua97d-\ua97f\ua9c1-\ua9ce\ua9da-\ua9df\ua9ff\uaa37-\uaa3f\uaa4e-\uaa4f\uaa5a-\uaa5f\uaa77-\uaa79\uaac3-\uaada\uaade-\uaadf\uaaf0-\uaaf1\uaaf7-\uab00\uab07-\uab08\uab0f-\uab10\uab17-\uab1f\uab27\uab2f\uab5b\uab6a-\uab6f\uabeb\uabee-\uabef\uabfa-\uabff\ud7a4-\ud7af\ud7c7-\ud7ca\ud7fc-\ud7ff\ue000-\uf8ff\ufa6e-\ufa6f\ufada-\ufaff\ufb07-\ufb12\ufb18-\ufb1c\ufb29\ufb37\ufb3d\ufb3f\ufb42\ufb45\ufbb2-\ufbd2\ufd3e-\ufd4f\ufd90-\ufd91\ufdc8-\ufdef\ufdfc-\ufdff\ufe10-\ufe1f\ufe30-\ufe32\ufe35-\ufe4c\ufe50-\ufe6f\ufe75\ufefd-\uff0f\uff1a-\uff20\uff3b-\uff3e\uff40\uff5b-\uff65\uffbf-\uffc1\uffc8-\uffc9\uffd0-\uffd1\uffd8-\uffd9\uffdd-\uffff\U0001000c\U00010027\U0001003b\U0001003e\U0001004e-\U0001004f\U0001005e-\U0001007f\U000100fb-\U0001013f\U00010175-\U000101fc\U000101fe-\U0001027f\U0001029d-\U0001029f\U000102d1-\U000102df\U000102e1-\U000102ff\U00010320-\U0001032c\U0001034b-\U0001034f\U0001037b-\U0001037f\U0001039e-\U0001039f\U000103c4-\U000103c7\U000103d0\U000103d6-\U000103ff\U0001049e-\U0001049f\U000104aa-\U000104af\U000104d4-\U000104d7\U000104fc-\U000104ff\U00010528-\U0001052f\U00010564-\U000105ff\U00010737-\U0001073f\U00010756-\U0001075f\U00010768-\U000107ff\U00010806-\U00010807\U00010809\U00010836\U00010839-\U0001083b\U0001083d-\U0001083e\U00010856-\U0001085f\U00010877-\U0001087f\U0001089f-\U000108df\U000108f3\U000108f6-\U000108ff\U00010916-\U0001091f\U0001093a-\U0001097f\U000109b8-\U000109bd\U000109c0-\U000109ff\U00010a04\U00010a07-\U00010a0b\U00010a14\U00010a18\U00010a36-\U00010a37\U00010a3b-\U00010a3e\U00010a40-\U00010a5f\U00010a7d-\U00010a7f\U00010a9d-\U00010abf\U00010ac8\U00010ae7-\U00010aff\U00010b36-\U00010b3f\U00010b56-\U00010b5f\U00010b73-\U00010b7f\U00010b92-\U00010bff\U00010c49-\U00010c7f\U00010cb3-\U00010cbf\U00010cf3-\U00010cff\U00010d28-\U00010d2f\U00010d3a-\U00010e7f\U00010eaa\U00010ead-\U00010eaf\U00010eb2-\U00010eff\U00010f1d-\U00010f26\U00010f28-\U00010f2f\U00010f51-\U00010faf\U00010fc5-\U00010fdf\U00010ff7-\U00010fff\U00011047-\U00011065\U00011070-\U0001107e\U000110bb-\U000110cf\U000110e9-\U000110ef\U000110fa-\U000110ff\U00011135\U00011140-\U00011143\U00011148-\U0001114f\U00011174-\U00011175\U00011177-\U0001117f\U000111c5-\U000111c8\U000111cd\U000111db\U000111dd-\U000111ff\U00011212\U00011238-\U0001123d\U0001123f-\U0001127f\U00011287\U00011289\U0001128e\U0001129e\U000112a9-\U000112af\U000112eb-\U000112ef\U000112fa-\U000112ff\U00011304\U0001130d-\U0001130e\U00011311-\U00011312\U00011329\U00011331\U00011334\U0001133a\U00011345-\U00011346\U00011349-\U0001134a\U0001134e-\U0001134f\U00011351-\U00011356\U00011358-\U0001135c\U00011364-\U00011365\U0001136d-\U0001136f\U00011375-\U000113ff\U0001144b-\U0001144f\U0001145a-\U0001145d\U00011462-\U0001147f\U000114c6\U000114c8-\U000114cf\U000114da-\U0001157f\U000115b6-\U000115b7\U000115c1-\U000115d7\U000115de-\U000115ff\U00011641-\U00011643\U00011645-\U0001164f\U0001165a-\U0001167f\U000116b9-\U000116bf\U000116ca-\U000116ff\U0001171b-\U0001171c\U0001172c-\U0001172f\U0001173a-\U000117ff\U0001183b-\U0001189f\U000118ea-\U000118fe\U00011907-\U00011908\U0001190a-\U0001190b\U00011914\U00011917\U00011936\U00011939-\U0001193a\U00011944-\U0001194f\U0001195a-\U0001199f\U000119a8-\U000119a9\U000119d8-\U000119d9\U000119e2\U000119e5-\U000119ff\U00011a3f-\U00011a46\U00011a48-\U00011a4f\U00011a9a-\U00011a9c\U00011a9e-\U00011abf\U00011af9-\U00011bff\U00011c09\U00011c37\U00011c41-\U00011c4f\U00011c5a-\U00011c71\U00011c90-\U00011c91\U00011ca8\U00011cb7-\U00011cff\U00011d07\U00011d0a\U00011d37-\U00011d39\U00011d3b\U00011d3e\U00011d48-\U00011d4f\U00011d5a-\U00011d5f\U00011d66\U00011d69\U00011d8f\U00011d92\U00011d99-\U00011d9f\U00011daa-\U00011edf\U00011ef7-\U00011faf\U00011fb1-\U00011fff\U0001239a-\U000123ff\U0001246f-\U0001247f\U00012544-\U00012fff\U0001342f-\U000143ff\U00014647-\U000167ff\U00016a39-\U00016a3f\U00016a5f\U00016a6a-\U00016acf\U00016aee-\U00016aef\U00016af5-\U00016aff\U00016b37-\U00016b3f\U00016b44-\U00016b4f\U00016b5a-\U00016b62\U00016b78-\U00016b7c\U00016b90-\U00016e3f\U00016e80-\U00016eff\U00016f4b-\U00016f4e\U00016f88-\U00016f8e\U00016fa0-\U00016fdf\U00016fe2\U00016fe5-\U00016fef\U00016ff2-\U00016fff\U000187f8-\U000187ff\U00018cd6-\U00018cff\U00018d09-\U0001afff\U0001b11f-\U0001b14f\U0001b153-\U0001b163\U0001b168-\U0001b16f\U0001b2fc-\U0001bbff\U0001bc6b-\U0001bc6f\U0001bc7d-\U0001bc7f\U0001bc89-\U0001bc8f\U0001bc9a-\U0001bc9c\U0001bc9f-\U0001d164\U0001d16a-\U0001d16c\U0001d173-\U0001d17a\U0001d183-\U0001d184\U0001d18c-\U0001d1a9\U0001d1ae-\U0001d241\U0001d245-\U0001d3ff\U0001d455\U0001d49d\U0001d4a0-\U0001d4a1\U0001d4a3-\U0001d4a4\U0001d4a7-\U0001d4a8\U0001d4ad\U0001d4ba\U0001d4bc\U0001d4c4\U0001d506\U0001d50b-\U0001d50c\U0001d515\U0001d51d\U0001d53a\U0001d53f\U0001d545\U0001d547-\U0001d549\U0001d551\U0001d6a6-\U0001d6a7\U0001d6c1\U0001d6db\U0001d6fb\U0001d715\U0001d735\U0001d74f\U0001d76f\U0001d789\U0001d7a9\U0001d7c3\U0001d7cc-\U0001d7cd\U0001d800-\U0001d9ff\U0001da37-\U0001da3a\U0001da6d-\U0001da74\U0001da76-\U0001da83\U0001da85-\U0001da9a\U0001daa0\U0001dab0-\U0001dfff\U0001e007\U0001e019-\U0001e01a\U0001e022\U0001e025\U0001e02b-\U0001e0ff\U0001e12d-\U0001e12f\U0001e13e-\U0001e13f\U0001e14a-\U0001e14d\U0001e14f-\U0001e2bf\U0001e2fa-\U0001e7ff\U0001e8c5-\U0001e8cf\U0001e8d7-\U0001e8ff\U0001e94c-\U0001e94f\U0001e95a-\U0001edff\U0001ee04\U0001ee20\U0001ee23\U0001ee25-\U0001ee26\U0001ee28\U0001ee33\U0001ee38\U0001ee3a\U0001ee3c-\U0001ee41\U0001ee43-\U0001ee46\U0001ee48\U0001ee4a\U0001ee4c\U0001ee50\U0001ee53\U0001ee55-\U0001ee56\U0001ee58\U0001ee5a\U0001ee5c\U0001ee5e\U0001ee60\U0001ee63\U0001ee65-\U0001ee66\U0001ee6b\U0001ee73\U0001ee78\U0001ee7d\U0001ee7f\U0001ee8a\U0001ee9c-\U0001eea0\U0001eea4\U0001eeaa\U0001eebc-\U0001f12f\U0001f14a-\U0001f14f\U0001f16a-\U0001f16f\U0001f18a-\U0001fbef\U0001fbfa-\U0001ffff\U0002a6de-\U0002a6ff\U0002b735-\U0002b73f\U0002b81e-\U0002b81f\U0002cea2-\U0002ceaf\U0002ebe1-\U0002f7ff\U0002fa1e-\U0002ffff\U0003134b-\U000e00ff\U000e01f0-\U0010ffff]"
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


def _iter_headings(lines: list[str]) -> Iterator[tuple[int, str]]:
    """Yield ``(line number, heading text)`` for every heading in a file.

    Both ATX (``# Title``) and setext (``Title`` underlined with ``===``)
    headings are recognized, because GitHub renders an anchor for either.
    The line number is the line the heading text sits on.
    """
    blanked = _blank_fences(lines)
    for index, line in enumerate(blanked):
        match = _HEADING_RE.match(line)
        if match is not None:
            yield index + 1, match.group(2)
            continue
        if index == 0:
            continue
        underline = _SETEXT_RE.match(line)
        if underline is None:
            continue
        title = blanked[index - 1].strip()
        if not title or _HEADING_RE.match(blanked[index - 1]) is not None:
            continue
        if _LIST_RE.match(blanked[index - 1]) is not None:
            # A ``---`` under a list item is a thematic break, not a heading.
            continue
        yield index, title


def _heading_anchors(path: Path) -> set[str]:
    """Return the anchor slugs a Markdown file exposes."""
    try:
        lines = path.read_text(encoding="utf-8").split("\n")
    except (OSError, UnicodeDecodeError):
        return set()
    return {_slugify(title) for _, title in _iter_headings(lines)}


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


def _is_inside(candidate: Path, root: Path) -> bool:
    """Report whether a path lies inside the repository root."""
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


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
    absolute = (repo_root / resolved).resolve()
    if not _is_inside(absolute, repo_root):
        # The link points at a sibling repository in the developer's
        # workspace (for example ``../lisa-cli``). It is not part of this
        # repository, so this gate cannot say whether it resolves, and the
        # CI checkout does not contain those repositories at all.
        return None
    if not absolute.exists():
        return Finding(rel_path, index, "link-target", f"link target does not exist: {target}")
    if not fragment:
        return None
    known = anchors_by_path.get(resolved)
    if known is None:
        known = _heading_anchors(absolute)
        anchors_by_path[resolved] = known
    if fragment.lower() not in known:
        # An empty anchor set means the target has no headings at all, so
        # the fragment cannot match anything either.
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


def _check_whitespace(rel_path: str, lines: list[str]) -> list[Finding]:
    """Check trailing whitespace and tab indentation outside code blocks.

    A fenced code block may contain a hard tab (Go, Makefile, Python
    examples) and a Markdown hard line break is two trailing spaces, so
    both are exempt inside a fence. Outside a fence, a trailing space is
    almost always an accident and a hard tab is a stray indent.
    """
    findings: list[Finding] = []
    for start, end in _iter_code_regions(lines):
        for index in range(start, end):
            lines[index] = ""
    for index, line in enumerate(lines, start=1):
        if line != line.rstrip():
            findings.append(
                Finding(rel_path, index, "trailing-whitespace", "line has trailing whitespace")
            )
        if line.startswith("\t") or re.match(r"^ +\t", line):
            findings.append(Finding(rel_path, index, "tab-indent", "line is indented with a tab"))
    return findings


def _check_spacing(rel_path: str, lines: list[str]) -> list[Finding]:
    """Check blank-line spacing around headings and lists.

    Structural spacing is measured on the code-stripped lines so that an
    example cannot fake a heading or a list.
    """
    findings: list[Finding] = []
    stripped = _strip_code(lines)
    heading_lines = {index for index, _ in _iter_headings(lines)}
    for index in sorted(heading_lines):
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
    findings += _check_whitespace(rel_path, lines)
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

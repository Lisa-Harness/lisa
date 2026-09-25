"""Tests for the provenance-tagged context envelope.

Verifies:
- XML tag wrapping and joining for all provenance kinds.
- Markdown fence parsing lifts provenance-tagged blocks into structured sections.
- Non-provenance fences (e.g. ```python) are preserved verbatim.
- Unknown provenance kinds raise ValueError.
- Empty content produces no tags.
"""

from __future__ import annotations

import pytest
from kernel.context.provenance import (
    PROVENANCE_KINDS,
    is_provenance_kind,
    normalize_markdown_to_tagged,
    parse_markdown_tagged,
    wrap_section,
    wrap_sections,
)


class TestWrapSection:
    def test_wraps_content_in_xml_tag(self) -> None:
        result = wrap_section("authoritative", "mode_id: default")
        assert result == "<authoritative>\nmode_id: default\n</authoritative>"

    def test_includes_source_attribute(self) -> None:
        result = wrap_section("system_observed", "cpu: 42%", source="ambient_facts")
        assert '<system_observed source="ambient_facts">' in result
        assert "cpu: 42%" in result

    def test_empty_content_returns_empty_string(self) -> None:
        assert wrap_section("authoritative", "") == ""
        assert wrap_section("authoritative", "   \n  ") == ""

    def test_strips_surrounding_whitespace(self) -> None:
        result = wrap_section("authoritative", "\n  hello  \n")
        assert result == "<authoritative>\nhello\n</authoritative>"

    def test_unknown_provenance_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provenance kind"):
            wrap_section("bogus", "content")

    def test_escapes_source_attribute(self) -> None:
        result = wrap_section("authoritative", "x", source='a"b<c>')
        assert "&quot;" in result
        assert "&lt;" in result
        assert "&gt;" in result


class TestWrapSections:
    def test_joins_multiple_sections(self) -> None:
        result = wrap_sections(
            [
                ("authoritative", "runtime", "fact A"),
                ("recalled_memory", "mem0", "fact B"),
            ]
        )
        assert '<authoritative source="runtime">\nfact A\n</authoritative>' in result
        assert '<recalled_memory source="mem0">\nfact B\n</recalled_memory>' in result
        assert result.count("\n\n") == 1  # sections separated by blank line

    def test_skips_empty_bodies(self) -> None:
        result = wrap_sections(
            [
                ("authoritative", "", "fact"),
                ("recalled_memory", "", ""),
            ]
        )
        assert "<authoritative>" in result
        assert "<recalled_memory>" not in result

    def test_empty_list_returns_empty(self) -> None:
        assert wrap_sections([]) == ""


class TestParseMarkdownTagged:
    def test_lifts_provenance_fence(self) -> None:
        text = "```authoritative\nmode_id: default\n```"
        sections = parse_markdown_tagged(text)
        assert sections == [("authoritative", "", "mode_id: default")]

    def test_lifts_multiple_fences(self) -> None:
        text = (
            "```authoritative\nmode: default\n```\n"
            "Some prose.\n"
            "```recalled_memory\nUser likes brevity.\n```"
        )
        sections = parse_markdown_tagged(text)
        assert len(sections) == 3
        assert sections[0] == ("authoritative", "", "mode: default")
        assert sections[1] == ("authoritative", "", "Some prose.")
        assert sections[2] == ("recalled_memory", "", "User likes brevity.")

    def test_preserves_non_provenance_fences(self) -> None:
        text = "```python\nprint('hi')\n```"
        sections = parse_markdown_tagged(text)
        assert len(sections) == 1
        assert sections[0][0] == "authoritative"
        assert "```python" in sections[0][2]

    def test_trailing_prose_becomes_authoritative(self) -> None:
        text = "```authoritative\nfact\n```\nTrailing prose."
        sections = parse_markdown_tagged(text)
        assert sections[-1] == ("authoritative", "", "Trailing prose.")

    def test_plain_text_no_fences(self) -> None:
        sections = parse_markdown_tagged("Just prose, no fences.")
        assert sections == [("authoritative", "", "Just prose, no fences.")]

    def test_all_provenance_kinds_parse(self) -> None:
        for kind in PROVENANCE_KINDS:
            text = f"```{kind}\nbody\n```"
            sections = parse_markdown_tagged(text)
            assert sections == [(kind, "", "body")]


class TestNormalizeMarkdownToTagged:
    def test_converts_fences_to_xml_tags(self) -> None:
        text = "```authoritative\nmode: default\n```"
        result = normalize_markdown_to_tagged(text)
        assert result == "<authoritative>\nmode: default\n</authoritative>"

    def test_prose_and_fences_mix(self) -> None:
        text = "Prose here.\n```recalled_memory\nA memory.\n```"
        result = normalize_markdown_to_tagged(text)
        assert "<authoritative>" in result
        assert "<recalled_memory>" in result

    def test_empty_input(self) -> None:
        assert normalize_markdown_to_tagged("") == ""


class TestIsProvenanceKind:
    def test_known_kinds(self) -> None:
        for kind in PROVENANCE_KINDS:
            assert is_provenance_kind(kind)

    def test_unknown_kind(self) -> None:
        assert not is_provenance_kind("bogus")
        assert not is_provenance_kind("")

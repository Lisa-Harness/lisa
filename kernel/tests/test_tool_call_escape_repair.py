"""Tests for the leaked-JSON-escaped-argument repair in schema_translator.

The 2026-08-28 cloud-init incident: a ``write_file`` tool call arrived
with a ``content`` argument whose every newline was a literal ``\\n``
pair — the model echoed an escaped rendering of the file from its
context instead of the file itself. ``openai_tool_call_to_mcp`` repairs
that signature in top-level string args; these tests pin the repair and
its guards (healthy payloads must pass through byte-for-byte).
"""

from __future__ import annotations

import json

import pytest
from kernel.tools.schema_translator import (
    is_truncated_tool_arguments,
    json_escaped_string_decode,
    looks_like_json_escaped_text,
    openai_tool_call_to_mcp,
)

# ── Signature detector ────────────────────────────────────────────────


class TestLooksLikeJsonEscapedText:
    def test_escaped_document_signature(self) -> None:
        value = (
            "#cloud-config\\npackage_update: true\\npackage_upgrade: true\\n"
            "\\npackages:\\n  - nginx"
        )
        assert looks_like_json_escaped_text(value) is True

    def test_real_newlines_are_never_the_signature(self) -> None:
        value = "#cloud-config\npackage_update: true\npackages:\n  - nginx"
        assert looks_like_json_escaped_text(value) is False

    def test_few_escapes_below_threshold(self) -> None:
        # Prose mentioning "\n" twice must not trip the check; the
        # threshold sits at _MIN_NEWLINE_ESCAPES (3) literal pairs.
        value = r"Use \n for line one, \n for line two only."
        assert looks_like_json_escaped_text(value) is False

    def test_json_object_payload_is_not_the_signature(self) -> None:
        assert looks_like_json_escaped_text('{"a": 1, "b": "\\n\\n\\n\\n\\n"}') is False

    def test_empty_string(self) -> None:
        assert looks_like_json_escaped_text("") is False

    def test_single_lonely_newline_escape(self) -> None:
        assert looks_like_json_escaped_text("a\\nb") is False


# ── Decode function ───────────────────────────────────────────────────


class TestJsonEscapedStringDecode:
    def test_single_layer_decode_restores_newlines(self) -> None:
        source = "alpha\nbeta\ngamma\n\ndelta"
        leaked = json.dumps(source)[1:-1]  # strip outer quotes
        assert json_escaped_string_decode(leaked) == source

    def test_double_layer_decode(self) -> None:
        source = "line one\nline two\nline three\nfour\nfive"
        once = json.dumps(source)[1:-1]
        twice = json.dumps(once)[1:-1]
        assert json_escaped_string_decode(twice) == source

    def test_legit_shell_escapes_survive_decode(self) -> None:
        # The deb-variant cloud-init keeps ``\\$host`` after decode — this
        # matches the healthy on-disk ground truth.
        source = "return 301 https://\\$host\\$request_uri;"
        leaked = json.dumps("a\n" + source + "\nb")[1:-1]
        decoded = json_escaped_string_decode(leaked)
        assert "\\$host" in decoded

    def test_ordinary_text_with_no_escapes_untouched(self) -> None:
        value = "plain text with no escapes at all"
        assert json_escaped_string_decode(value) == value

    def test_mixed_content_body_untouched(self) -> None:
        # A shell script block with real newlines must pass through
        # byte-for-byte, including its own backslash sequences.
        value = "set -euo pipefail\nexec 1> >(tee /var/log/x.log)\nDONE\n"
        assert json_escaped_string_decode(value) == value


# ── Tool-call boundary ────────────────────────────────────────────────


def _tool_call(arguments: dict[str, object]) -> dict[str, object]:
    return {
        "id": "call_test",
        "type": "function",
        "function": {
            "name": "lisa-storage.write_file",
            "arguments": json.dumps(arguments),
        },
    }


class TestOpenAIToolCallToMCP:
    def test_incident_payload_repaired(self) -> None:
        # Exact shape from the 2026-08-28 recovery record: escaped cloud-init
        # document persisted via literal \n pairs.
        corrupted = (
            "#cloud-config\\npackage_update: true\\npackage_upgrade: true\\n"
            "\\npackages:\\n  - nginx\\n  - jq\\n  - curl\\n"
        )
        _, args = openai_tool_call_to_mcp(_tool_call({"path": "a.yaml", "content": corrupted}))
        content = args["content"]
        # 6 `\n` pairs in the fixture + the trailing one = 7 lines, plus the
        # blank line pair, is 8 newlines total (count them in the literal).
        assert content.count("\n") == 8
        assert "\\n" not in content

    def test_unaffected_args_pass_through(self) -> None:
        params = {
            "path": "notes/plain.md",
            "content": "head\nbody with real newlines\nand a \\n mention",
        }
        _, args = openai_tool_call_to_mcp(_tool_call(params))
        assert args == params

    def test_flat_shape_also_repaired(self) -> None:
        # Ollama-style flat tool calls (no ``function`` sub-dict).
        corrupted = "a\\nb\\nc\\nd\\ne\\nf"
        tool_call = {
            "id": "call_test",
            "type": "function",
            "name": "lisa-storage.write_file",
            "arguments": {"path": "x", "content": corrupted},
        }
        _, args = openai_tool_call_to_mcp(tool_call)
        assert args["content"] == "a\nb\nc\nd\ne\nf"

    def test_non_string_args_untouched(self) -> None:
        params = {"count": 4, "listing": ["\\n", "\\n", "\\n", "\\n", "\\n"]}
        _, args = openai_tool_call_to_mcp(_tool_call(params))
        assert args == params

    def test_malformed_arguments_json_marks_truncated(self) -> None:
        # The 2026-09-23 finances incident: a length-cut completion left the
        # arguments unparseable and the tool dispatched with {} — writing a
        # truncated ledger with ok:true. The translator must now mark the
        # payload as truncated so the turn loop refuses the dispatch.
        tool_call = {
            "id": "call_test",
            "type": "function",
            "function": {"name": "x", "arguments": "{not json"},
        }
        name, args = openai_tool_call_to_mcp(tool_call)
        assert name == "x"
        assert is_truncated_tool_arguments(args)

    def test_length_cut_write_payload_marks_truncated(self) -> None:
        # Exact incident shape: write_file arguments cut mid-content-string
        # when the completion hit its token limit.
        cut = '{"path": "ledgers/2024.beancount", "content": "2024-01-02 open Assets:GL1000'
        tool_call = {
            "id": "call_test",
            "type": "function",
            "function": {"name": "lisa-storage.write_file", "arguments": cut},
        }
        name, args = openai_tool_call_to_mcp(tool_call)
        assert name == "lisa-storage.write_file"
        assert is_truncated_tool_arguments(args)
        # And the sentinel must NOT carry any path/content keys a caller
        # could mistake for real arguments.
        assert "path" not in args and "content" not in args

    def test_empty_argument_string_is_legitimate_no_params(self) -> None:
        # An explicitly empty string is a real "no parameters" call, not a
        # truncation; it must stay {} and not carry the sentinel.
        tool_call = {
            "id": "call_test",
            "type": "function",
            "function": {"name": "list_files", "arguments": ""},
        }
        _, args = openai_tool_call_to_mcp(tool_call)
        assert args == {}
        assert not is_truncated_tool_arguments(args)

    def test_valid_arguments_never_carry_sentinel(self) -> None:
        _, args = openai_tool_call_to_mcp(_tool_call({"path": "a.md", "content": "hello"}))
        assert not is_truncated_tool_arguments(args)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # All shapes carry >=4 literal \n pairs so they cross the threshold.
        (
            'x=\\"hi\\"\\ny=\\"there\\"\\nz\\nw=\\"last\\"\\nend',
            'x="hi"\ny="there"\nz\nw="last"\nend',
        ),
        ("a\\tb\\nc\\td\\ne\\nf\\ng\\nh", "a\tb\nc\td\ne\nf\ng\nh"),
        (
            'q1\\"aa\\"\\nq2\\"bb\\"\\nq3\\"cc\\"\\nq4\\"dd\\"\\nend',
            'q1"aa"\nq2"bb"\nq3"cc"\nq4"dd"\nend',
        ),
        # A Windows-style path whose lone \n-ish sequence is below the
        # threshold stays as-is.
        (r"c:\path\to\file\nx", r"c:\path\to\file\nx"),
    ],
)
def test_known_shapes(value: str, expected: str) -> None:
    assert json_escaped_string_decode(value) == expected
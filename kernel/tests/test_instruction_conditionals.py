"""Tests for conditional operation blocks in mode instruction markup.

Covers the pure transform in :mod:`kernel.instruction_conditionals`, both
load-path integrations in :mod:`kernel.ipsar_fsm` (the disk loader and the
MCP catalog path), and the operation-binding warning downgrade.

The real ``coding_design`` / ``coding_review`` / ``researcher`` ``_mode.md``
files live in external pack repositories (``lisa-pack-coding`` and
``lisa_packs``), so the end-to-end test authors a ``coding_design``-shaped
mode in a temp directory and drives the full disk pipeline: ``_mode.md`` →
front-matter parse → conditional strip → validation → prompt intro.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from kernel.instruction_conditionals import (
    conditional_operation_ids,
    has_conditional_markers,
    strip_conditional_operation_blocks,
)
from kernel.ipsar_fsm import (
    _apply_mode_conditionals,
    _evaluate_mode_conditionals,
    load_catalog_from_disk,
    validate_workflow_catalog,
)
from kernel.models import WorkflowModeConfig

from kernel import ipsar_fsm

# ---------------------------------------------------------------------------
# Pure transform: evaluation
# ---------------------------------------------------------------------------


def test_no_directives_returns_input_byte_identical() -> None:
    """A body with no marker is returned untouched, byte for byte."""
    body = "Intro.\n\n\n\nOutro with ``` fence-ish text and <!-- comments -->.\n"

    assert strip_conditional_operation_blocks(body, {"present"}) == body


def test_no_directives_comment_free_body_returns_input_object() -> None:
    """A body with no HTML comment at all skips the scan via the fast path."""
    body = "Intro.\n\n\n\nOutro with ``` fence-ish text.\n"

    assert strip_conditional_operation_blocks(body, {"present"}) is body


def test_true_condition_keeps_content_and_drops_markers() -> None:
    """An installed operation keeps the inner content byte-identical."""
    body = (
        "Intro.\n"
        "\n"
        "<!-- if operation:present_op -->\n"
        "Call `present_op` when done.\n"
        "\n"
        "Second paragraph.\n"
        "<!-- endif -->\n"
        "\n"
        "Outro.\n"
    )

    result = strip_conditional_operation_blocks(body, {"present_op"})

    assert result == ("Intro.\n\nCall `present_op` when done.\n\nSecond paragraph.\n\nOutro.\n")


def test_false_condition_removes_block_and_collapses_whitespace() -> None:
    """A missing operation removes the whole block and collapses the gap."""
    body = (
        "Intro.\n"
        "\n"
        "<!-- if operation:missing_op -->\n"
        "Call `missing_op` when done.\n"
        "<!-- endif -->\n"
        "\n"
        "Outro.\n"
    )

    result = strip_conditional_operation_blocks(body, {"present_op"})

    assert result == "Intro.\n\nOutro.\n"


def test_multiple_blocks_evaluated_independently() -> None:
    """Each block tests its own operation; kept and dropped blocks coexist."""
    body = (
        "Intro.\n"
        "\n"
        "<!-- if operation:present_op -->\n"
        "Kept text.\n"
        "<!-- endif -->\n"
        "\n"
        "<!-- if operation:missing_op -->\n"
        "Dropped text.\n"
        "<!-- endif -->\n"
        "\n"
        "Outro.\n"
    )

    result = strip_conditional_operation_blocks(body, {"present_op"})

    assert result == "Intro.\n\nKept text.\n\nOutro.\n"


def test_block_at_start_and_end_of_body() -> None:
    """Blocks at either boundary of the body are handled without index drift."""
    body = (
        "<!-- if operation:present_op -->\n"
        "First.\n"
        "<!-- endif -->\n"
        "\n"
        "Middle.\n"
        "\n"
        "<!-- if operation:missing_op -->\n"
        "Last.\n"
        "<!-- endif -->\n"
    )

    result = strip_conditional_operation_blocks(body, {"present_op"})

    assert result == "First.\n\nMiddle.\n\n"


def test_consecutive_blocks_back_to_back() -> None:
    """Two blocks with no separating prose both evaluate correctly."""
    body = (
        "<!-- if operation:present_op -->\n"
        "A.\n"
        "<!-- endif -->\n"
        "<!-- if operation:missing_op -->\n"
        "B.\n"
        "<!-- endif -->\n"
        "Tail.\n"
    )

    result = strip_conditional_operation_blocks(body, {"present_op"})

    assert result == "A.\nTail.\n"


def test_kept_block_preserves_preexisting_newline_runs() -> None:
    """No removal means no collapse: unrelated 3+ newline runs survive."""
    body = "Intro.\n\n\n\n<!-- if operation:present_op -->\nKept.\n<!-- endif -->\n\n\n\nOutro.\n"

    result = strip_conditional_operation_blocks(body, {"present_op"})

    assert result == ("Intro.\n\n\n\nKept.\n\n\n\nOutro.\n")


def test_trailing_newline_preserved_through_strip() -> None:
    """The split/join round trip is lossless for trailing newlines."""
    body = "<!-- if operation:missing_op -->\nDropped.\n<!-- endif -->\n\n"

    assert strip_conditional_operation_blocks(body, set()) == "\n"


def test_marker_whitespace_tolerance() -> None:
    """Markers tolerate leading/trailing line whitespace and inner spacing."""
    body = (
        "Intro.\n"
        "\n"
        "   <!--   if   operation:missing_op   -->   \n"
        "Dropped.\n"
        "\t<!--	endif	-->\n"
        "\n"
        "Outro.\n"
    )

    result = strip_conditional_operation_blocks(body, set())

    assert result == "Intro.\n\nOutro.\n"


# ---------------------------------------------------------------------------
# Pure transform: fences
# ---------------------------------------------------------------------------


def test_markers_inside_backtick_fence_are_literal() -> None:
    """Markers inside a fenced code block are documentation, not directives."""
    body = (
        "Intro.\n"
        "\n"
        "```text\n"
        "<!-- if operation:missing_op -->\n"
        "literal doc\n"
        "<!-- endif -->\n"
        "```\n"
        "\n"
        "Outro.\n"
    )

    result = strip_conditional_operation_blocks(body, set())

    assert result == body


def test_markers_inside_tilde_fence_are_literal() -> None:
    """Tilde fences hide markers exactly like backtick fences."""
    body = (
        "Intro.\n"
        "\n"
        "~~~\n"
        "<!-- if operation:missing_op -->\n"
        "literal doc\n"
        "<!-- endif -->\n"
        "~~~\n"
        "\n"
        "Outro.\n"
    )

    assert strip_conditional_operation_blocks(body, set()) == body


def test_mixed_fences_do_not_desync_the_scanner() -> None:
    """A backtick line inside a tilde fence is content, not a closing fence.

    A naive toggle-on-any-fence scanner would close the tilde fence at the
    ``` line, then treat the real top-level conditional marker as fenced
    literal text and leak it into the prompt. Char+length tracking must keep
    the fence open until a same-char, long-enough closer arrives.
    """
    body = (
        "Intro.\n"
        "\n"
        "~~~\n"
        "<!-- if operation:missing_op -->\n"
        "fenced literal\n"
        "```\n"
        "still inside the tilde fence\n"
        "~~~\n"
        "\n"
        "<!-- if operation:missing_op -->\n"
        "conditional text\n"
        "<!-- endif -->\n"
        "\n"
        "Outro.\n"
    )

    result = strip_conditional_operation_blocks(body, set())

    assert "conditional text" not in result
    assert "fenced literal" in result
    assert "still inside the tilde fence" in result


def test_longer_closing_fence_closes_shorter_opener() -> None:
    """CommonMark: a closing fence may be longer than the opening fence."""
    body = (
        "Intro.\n"
        "\n"
        "```\n"
        "<!-- if operation:missing_op -->\n"
        "fenced literal\n"
        "`````\n"
        "\n"
        "<!-- if operation:missing_op -->\n"
        "conditional text\n"
        "<!-- endif -->\n"
        "\n"
        "Outro.\n"
    )

    result = strip_conditional_operation_blocks(body, set())

    assert "conditional text" not in result
    assert "fenced literal" in result


def test_shorter_fence_line_inside_longer_fence_is_content() -> None:
    """A ``` line inside a ```` fence is content, so markers stay fenced."""
    body = (
        "Intro.\n"
        "\n"
        "````\n"
        "<!-- if operation:missing_op -->\n"
        "```\n"
        "still fenced\n"
        "````\n"
        "\n"
        "<!-- if operation:missing_op -->\n"
        "conditional text\n"
        "<!-- endif -->\n"
        "\n"
        "Outro.\n"
    )

    result = strip_conditional_operation_blocks(body, set())

    assert "conditional text" not in result
    assert "still fenced" in result


def test_fenced_marker_inside_conditional_block_is_literal() -> None:
    """A fence opened inside a kept block does not leak past its endif."""
    body = (
        "<!-- if operation:present_op -->\n"
        "Example:\n"
        "\n"
        "```text\n"
        "<!-- if operation:missing_op -->\n"
        "fenced literal\n"
        "<!-- endif -->\n"
        "```\n"
        "<!-- endif -->\n"
        "\n"
        "<!-- if operation:missing_op -->\n"
        "conditional text\n"
        "<!-- endif -->\n"
        "\n"
        "Outro.\n"
    )

    result = strip_conditional_operation_blocks(body, {"present_op"})

    assert "fenced literal" in result
    assert "conditional text" not in result
    assert "<!-- if operation:missing_op -->" not in result.split("```text\n", 1)[0]


# ---------------------------------------------------------------------------
# Pure transform: malformed markup
# ---------------------------------------------------------------------------


def test_unterminated_block_drops_only_the_marker_line() -> None:
    """A missing endif keeps the content and drops the opener marker only.

    The authored content still reaches the prompt, so the missing-operation
    warning stays — the authoring error fails visible, not silent.
    """
    body = (
        "Intro.\n"
        "\n"
        "<!-- if operation:missing_op -->\n"
        "Content that was meant to be conditional.\n"
        "\n"
        "Outro.\n"
    )

    result = strip_conditional_operation_blocks(body, set())

    assert result == ("Intro.\n\nContent that was meant to be conditional.\n\nOutro.\n")


def test_stray_endif_dropped() -> None:
    """A close marker with no open block drops only the marker line."""
    body = "Intro.\n\n<!-- endif -->\n\nOutro.\n"

    assert strip_conditional_operation_blocks(body, set()) == "Intro.\n\nOutro.\n"


def test_nested_opener_makes_outer_block_literal() -> None:
    """A nested opener drops the outer marker line; the inner block evaluates.

    Nesting is unsupported, so the outer block is malformed: its marker line
    is dropped and scanning resumes so the inner block is evaluated on its
    own. No marker survives the pass.
    """
    body = (
        "<!-- if operation:outer_op -->\n"
        "Outer content.\n"
        "<!-- if operation:inner_op -->\n"
        "Inner content.\n"
        "<!-- endif -->\n"
        "Tail content.\n"
        "<!-- endif -->\n"
    )

    result = strip_conditional_operation_blocks(body, {"outer_op"})

    assert result == "Outer content.\nTail content.\n"


def test_unrecognized_marker_stays_literal() -> None:
    """An opener with a non-identifier ID is not a directive and stays put.

    The typo remains visible in the rendered prompt instead of silently
    vanishing; the transform is idempotent because the line is treated the
    same on every pass.
    """
    body = "Intro.\n\n<!-- if operation:bad-id! -->\nContent.\n\nOutro.\n"

    assert strip_conditional_operation_blocks(body, set()) == body


# ---------------------------------------------------------------------------
# Pure transform: idempotency
# ---------------------------------------------------------------------------


_IDEMPOTENCY_BODIES = [
    # Kept block.
    "Intro.\n\n<!-- if operation:present_op -->\nKept.\n<!-- endif -->\n\nOutro.\n",
    # Dropped block.
    "Intro.\n\n<!-- if operation:missing_op -->\nDropped.\n<!-- endif -->\n\nOutro.\n",
    # Mixed blocks and fences.
    (
        "Intro.\n\n```text\n<!-- if operation:missing_op -->\nfenced\n<!-- endif -->\n```\n\n"
        "<!-- if operation:present_op -->\nKept.\n<!-- endif -->\n\n"
        "<!-- if operation:missing_op -->\nDropped.\n<!-- endif -->\n\nOutro.\n"
    ),
    # Unterminated opener.
    "<!-- if operation:missing_op -->\nContent.\n\nOutro.\n",
    # Stray endif.
    "Intro.\n\n<!-- endif -->\n\nOutro.\n",
    # Nested openers.
    (
        "<!-- if operation:outer_op -->\nOuter.\n"
        "<!-- if operation:inner_op -->\nInner.\n<!-- endif -->\nTail.\n<!-- endif -->\n"
    ),
    # Unrecognized marker.
    "<!-- if operation:bad-id! -->\nContent.\n\nOutro.\n",
    # Tilde fence with a backtick line inside.
    "~~~\n<!-- if operation:missing_op -->\n```\nstill fenced\n~~~\n\nOutro.\n",
]


@pytest.mark.parametrize("body", _IDEMPOTENCY_BODIES)
def test_transform_is_idempotent(body: str) -> None:
    """Applying the transform to its own output is a no-op."""
    once = strip_conditional_operation_blocks(body, {"present_op", "outer_op"})

    assert strip_conditional_operation_blocks(once, {"present_op", "outer_op"}) == once


@pytest.mark.parametrize(
    "body",
    [
        # Kept block.
        "Intro.\n\n<!-- if operation:present_op -->\nKept.\n<!-- endif -->\n\nOutro.\n",
        # Dropped block.
        "Intro.\n\n<!-- if operation:missing_op -->\nDropped.\n<!-- endif -->\n\nOutro.\n",
        # Unterminated opener.
        "<!-- if operation:missing_op -->\nContent.\n\nOutro.\n",
        # Stray endif.
        "Intro.\n\n<!-- endif -->\n\nOutro.\n",
        # Nested openers.
        (
            "<!-- if operation:outer_op -->\nOuter.\n"
            "<!-- if operation:inner_op -->\nInner.\n<!-- endif -->\nTail.\n<!-- endif -->\n"
        ),
    ],
)
def test_no_marker_survives_a_pass(body: str) -> None:
    """After one pass no conditional marker line remains outside a fence."""
    result = strip_conditional_operation_blocks(body, {"present_op", "outer_op"})

    assert "<!-- if operation:" not in result
    assert "<!-- endif -->" not in result


# ---------------------------------------------------------------------------
# conditional_operation_ids / has_conditional_markers
# ---------------------------------------------------------------------------


def test_conditional_operation_ids_extracts_openers() -> None:
    """Every opener marker contributes its ID."""
    body = (
        "<!-- if operation:op_a -->\nA\n<!-- endif -->\n\n"
        "<!-- if operation:op_b -->\nB\n<!-- endif -->\n"
    )

    assert conditional_operation_ids(body) == {"op_a", "op_b"}


def test_conditional_operation_ids_is_fence_aware() -> None:
    """Fenced markers are literal text and do not contribute IDs."""
    body = (
        "```text\n"
        "<!-- if operation:fenced_op -->\n"
        "<!-- endif -->\n"
        "```\n"
        "\n"
        "<!-- if operation:real_op -->\n"
        "Content.\n"
        "<!-- endif -->\n"
    )

    assert conditional_operation_ids(body) == {"real_op"}


def test_conditional_operation_ids_includes_malformed_openers() -> None:
    """Unterminated openers still count; the effective body gates the downgrade."""
    body = "<!-- if operation:unterminated_op -->\nContent.\n"

    assert conditional_operation_ids(body) == {"unterminated_op"}


def test_conditional_operation_ids_empty_for_marker_free_body() -> None:
    """No opener substring means no IDs."""
    assert conditional_operation_ids("Intro.\n\nOutro.\n") == set()


def test_has_conditional_markers_substring_check() -> None:
    """The cheap pre-check matches either marker substring."""
    assert has_conditional_markers("<!-- if operation:x -->") is True
    assert has_conditional_markers("<!-- endif -->") is True
    assert has_conditional_markers("plain text") is False


# ---------------------------------------------------------------------------
# Load-path integration: disk loader
# ---------------------------------------------------------------------------


def _write_mode_md(mode_dir: Path, body: str, *, allowed_operation_ids: list[str]) -> None:
    """Write one minimal chat mode with the given body and operation bindings."""
    mode_dir.mkdir(parents=True, exist_ok=True)
    front_matter = "\n".join(
        [
            "---",
            "mode_id: demo",
            "mode_kind: chat",
            "command_pattern: /demo",
            "description: demo mode",
            "allowed_operation_ids: [" + ", ".join(allowed_operation_ids) + "]",
            "---",
            "",
        ]
    )
    (mode_dir / "_mode.md").write_text(front_matter + body, encoding="utf-8")


def _patch_operation_catalog(monkeypatch, available: set[str]) -> None:
    """Point ipsar_fsm's operation-catalog accessor at a fixed ID set."""
    monkeypatch.setattr(ipsar_fsm, "_get_operation_catalog", lambda: dict.fromkeys(available))


def test_disk_loader_strips_blocks_for_missing_operations(tmp_path: Path, monkeypatch) -> None:
    """The disk loader evaluates conditionals against the operation catalog."""
    _patch_operation_catalog(monkeypatch, {"present_op"})
    body = (
        "# Demo\n"
        "\n"
        "<!-- if operation:missing_op -->\n"
        "Call `missing_op`.\n"
        "<!-- endif -->\n"
        "\n"
        "<!-- if operation:present_op -->\n"
        "Call `present_op`.\n"
        "<!-- endif -->\n"
    )
    _write_mode_md(tmp_path / "demo", body, allowed_operation_ids=["present_op"])

    catalog = load_catalog_from_disk(instructions_dir=tmp_path)

    mode = catalog["demo"]
    assert "missing_op" not in mode.instruction_body
    assert "Call `present_op`." in mode.instruction_body
    assert mode.instruction_body_raw == body.strip()


def test_disk_loader_skips_strip_when_catalog_unavailable(tmp_path: Path, monkeypatch) -> None:
    """Without registered operation callbacks the body passes through unchanged."""

    def _unavailable() -> dict[str, object]:
        raise RuntimeError("Operation callbacks not registered.")

    monkeypatch.setattr(ipsar_fsm, "_get_operation_catalog", _unavailable)
    body = "# Demo\n\n<!-- if operation:missing_op -->\nCall `missing_op`.\n<!-- endif -->\n"
    _write_mode_md(tmp_path / "demo", body, allowed_operation_ids=[])

    catalog = load_catalog_from_disk(instructions_dir=tmp_path)

    mode = catalog["demo"]
    assert mode.instruction_body == body.strip()
    assert mode.instruction_body_raw == ""


def test_disk_loader_marker_free_mode_has_empty_raw_body(tmp_path: Path, monkeypatch) -> None:
    """A mode with no conditional blocks keeps an empty raw-body field."""
    _patch_operation_catalog(monkeypatch, {"present_op"})
    body = "# Demo\n\nPlain instructions.\n"
    _write_mode_md(tmp_path / "demo", body, allowed_operation_ids=[])

    catalog = load_catalog_from_disk(instructions_dir=tmp_path)

    mode = catalog["demo"]
    assert mode.instruction_body == body.strip()
    assert mode.instruction_body_raw == ""


def test_evaluate_mode_conditionals_debug_logs_missing_operations(monkeypatch, caplog) -> None:
    """Stripped operation IDs are visible at debug level for authoring."""
    _patch_operation_catalog(monkeypatch, {"present_op"})
    body = "<!-- if operation:missing_op -->\nDropped.\n<!-- endif -->\n"

    with caplog.at_level(logging.DEBUG, logger="kernel.ipsar_fsm"):
        instruction_body, raw = _evaluate_mode_conditionals("demo", body)

    assert instruction_body == ""
    assert raw == body
    assert any("missing_op" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Load-path integration: MCP catalog path
# ---------------------------------------------------------------------------


def _make_mode(mode_id: str, body: str) -> WorkflowModeConfig:
    """Build a minimal mode config as the MCP deserializer would."""
    return WorkflowModeConfig(
        mode_id=mode_id,
        mode_kind="chat",
        description="demo mode",
        instruction_body=body,
    )


def test_apply_mode_conditionals_strips_mcp_catalog(monkeypatch) -> None:
    """The MCP-path helper evaluates conditionals across a loaded catalog."""
    _patch_operation_catalog(monkeypatch, {"present_op"})
    marked_body = (
        "<!-- if operation:missing_op -->\n"
        "Dropped.\n"
        "<!-- endif -->\n"
        "\n"
        "<!-- if operation:present_op -->\n"
        "Kept.\n"
        "<!-- endif -->\n"
    )
    plain_body = "Plain instructions.\n"
    catalog = {
        "marked": _make_mode("marked", marked_body),
        "plain": _make_mode("plain", plain_body),
    }

    _apply_mode_conditionals(catalog)

    assert catalog["marked"].instruction_body == "\nKept.\n"
    assert catalog["marked"].instruction_body_raw == marked_body
    assert catalog["plain"].instruction_body == plain_body
    assert catalog["plain"].instruction_body_raw == ""


def test_apply_mode_conditionals_skips_when_catalog_unavailable(
    monkeypatch,
) -> None:
    """Without operation callbacks the MCP path leaves bodies unevaluated."""

    def _unavailable() -> dict[str, object]:
        raise RuntimeError("Operation callbacks not registered.")

    monkeypatch.setattr(ipsar_fsm, "_get_operation_catalog", _unavailable)
    body = "<!-- if operation:missing_op -->\nDropped.\n<!-- endif -->\n"
    catalog = {"marked": _make_mode("marked", body)}

    _apply_mode_conditionals(catalog)

    assert catalog["marked"].instruction_body == body
    assert catalog["marked"].instruction_body_raw == ""


# ---------------------------------------------------------------------------
# Warning downgrade
# ---------------------------------------------------------------------------


def _make_bound_mode(
    *,
    allowed_operation_ids: list[str],
    instruction_body: str,
    instruction_body_raw: str,
) -> WorkflowModeConfig:
    """Build a mode with operation bindings for the binding validator."""
    return WorkflowModeConfig(
        mode_id="demo",
        mode_kind="chat",
        description="demo mode",
        instruction_body=instruction_body,
        instruction_body_raw=instruction_body_raw,
        allowed_operation_ids=allowed_operation_ids,
    )


def test_conditional_reference_downgrades_warning_to_debug(monkeypatch, caplog) -> None:
    """A missing operation referenced only in a stripped block logs debug."""
    _patch_operation_catalog(monkeypatch, {"present_op"})
    mode = _make_bound_mode(
        allowed_operation_ids=["missing_op"],
        instruction_body="Plain instructions.\n",
        instruction_body_raw=(
            "<!-- if operation:missing_op -->\nCall `missing_op`.\n<!-- endif -->\n"
        ),
    )

    with caplog.at_level(logging.DEBUG, logger="kernel.ipsar_fsm"):
        ipsar_fsm._validate_mode_operation_bindings(mode)

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert not warnings
    assert any("conditional block" in r.message for r in caplog.records)


def test_unconditional_reference_keeps_the_warning(monkeypatch, caplog) -> None:
    """A missing operation referenced outside any block still warns."""
    _patch_operation_catalog(monkeypatch, {"present_op"})
    mode = _make_bound_mode(
        allowed_operation_ids=["missing_op"],
        instruction_body="Call `missing_op` when done.\n",
        instruction_body_raw="",
    )

    with caplog.at_level(logging.DEBUG, logger="kernel.ipsar_fsm"):
        ipsar_fsm._validate_mode_operation_bindings(mode)

    assert any(
        "references operation 'missing_op'" in r.message
        for r in caplog.records
        if r.levelno >= logging.WARNING
    )


def test_conditional_reference_also_named_in_body_keeps_the_warning(monkeypatch, caplog) -> None:
    """A conditional reference plus an unconditional mention still warns.

    The block was stripped, but the prose still names the operation, so the
    model still sees a reference to something that cannot be called.
    """
    _patch_operation_catalog(monkeypatch, {"present_op"})
    mode = _make_bound_mode(
        allowed_operation_ids=["missing_op"],
        instruction_body="Call `missing_op` when done.\n",
        instruction_body_raw=(
            "<!-- if operation:missing_op -->\nCall `missing_op`.\n<!-- endif -->\n"
        ),
    )

    with caplog.at_level(logging.DEBUG, logger="kernel.ipsar_fsm"):
        ipsar_fsm._validate_mode_operation_bindings(mode)

    assert any(
        "references operation 'missing_op'" in r.message
        for r in caplog.records
        if r.levelno >= logging.WARNING
    )


def test_unterminated_conditional_keeps_the_warning(monkeypatch, caplog) -> None:
    """A malformed block keeps its content in the prompt, so it still warns."""
    _patch_operation_catalog(monkeypatch, {"present_op"})
    mode = _make_bound_mode(
        allowed_operation_ids=["missing_op"],
        instruction_body="Call `missing_op` when done.\n",
        instruction_body_raw="<!-- if operation:missing_op -->\nCall `missing_op`.\n",
    )

    with caplog.at_level(logging.DEBUG, logger="kernel.ipsar_fsm"):
        ipsar_fsm._validate_mode_operation_bindings(mode)

    assert any(
        "references operation 'missing_op'" in r.message
        for r in caplog.records
        if r.levelno >= logging.WARNING
    )


# ---------------------------------------------------------------------------
# End-to-end: coding_design-shaped mode through the full disk pipeline
# ---------------------------------------------------------------------------


def _write_coding_design_mode(mode_dir: Path) -> str:
    """Author a coding_design-shaped mode mirroring the external pack's real one.

    The real ``coding_design`` / ``coding_review`` / ``researcher`` modes live
    in external pack repositories, so this fixture reproduces the authoring
    shape that matters here: ``okf_emit_concept`` bound in the front matter
    and its procedural paragraph wrapped in a conditional block. Converting
    the real mode files is follow-up work in those repositories.
    """
    body = (
        "# Coding Design\n"
        "\n"
        "Design the change before any code is written.\n"
        "\n"
        "<!-- if operation:okf_emit_concept -->\n"
        "When the design is settled, call `okf_emit_concept` to emit the\n"
        "structured concept node for the operator.\n"
        "<!-- endif -->\n"
        "\n"
        "Hand off to coding_execute when the design is approved.\n"
    )
    mode_dir.mkdir(parents=True, exist_ok=True)
    front_matter = "\n".join(
        [
            "---",
            "mode_id: coding_design",
            "mode_kind: chat",
            "command_pattern: /design",
            "description: design coding changes",
            "allowed_operation_ids: [okf_emit_concept]",
            "---",
            "",
        ]
    )
    (mode_dir / "_mode.md").write_text(front_matter + body, encoding="utf-8")
    return body


def test_coding_design_mode_end_to_end_okf_absent(tmp_path: Path, monkeypatch, caplog) -> None:
    """With okf uninstalled the prompt has no okf prose and no warning.

    This is the regression the feature exists for: the three real modes warn
    on every load today because their prose names an operation whose pack is
    absent. With the conditional block, the prose disappears and the warning
    is downgraded to debug.
    """
    _patch_operation_catalog(monkeypatch, set())
    raw_body = _write_coding_design_mode(tmp_path / "coding_design")

    with caplog.at_level(logging.DEBUG, logger="kernel.ipsar_fsm"):
        catalog = load_catalog_from_disk(instructions_dir=tmp_path)
        validate_workflow_catalog(catalog)

    mode = catalog["coding_design"]
    assert "okf_emit_concept" not in mode.instruction_body
    assert "Design the change" in mode.instruction_body
    assert "Hand off to coding_execute" in mode.instruction_body
    assert mode.instruction_body_raw == raw_body.strip()
    warnings = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and "okf_emit_concept" in r.message
    ]
    assert not warnings
    assert any(
        "conditional block" in r.message and "okf_emit_concept" in r.message for r in caplog.records
    )


def test_coding_design_mode_end_to_end_okf_present(tmp_path: Path, monkeypatch, caplog) -> None:
    """With okf installed the full procedural paragraph appears verbatim."""
    _patch_operation_catalog(monkeypatch, {"okf_emit_concept"})
    _write_coding_design_mode(tmp_path / "coding_design")

    with caplog.at_level(logging.DEBUG, logger="kernel.ipsar_fsm"):
        catalog = load_catalog_from_disk(instructions_dir=tmp_path)
        validate_workflow_catalog(catalog)

    mode = catalog["coding_design"]
    assert "call `okf_emit_concept` to emit the" in mode.instruction_body
    assert "<!-- if operation:" not in mode.instruction_body
    assert "<!-- endif -->" not in mode.instruction_body
    assert not any(
        r.levelno >= logging.WARNING for r in caplog.records if "okf_emit_concept" in r.message
    )

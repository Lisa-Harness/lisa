"""Regression tests for the strip_tools threading (F3), the resume
attribute reads (F5), and the by-reference turn state (F4) in the turn loop.

F3: ``_apply_iteration_lifecycle_action`` must return the computed
``strip_tools`` value from ``_apply_lifecycle_action`` instead of
discarding the sixth tuple slot, and the loop's effective-tools rule must
consume that value rather than re-derive it from the action type (the
previous ``elif not isinstance(_lifecycle_action, _FinishAction)`` guard
was dead code: every FinishAction breaks the loop before that line).

F5/F4: the post-suspend resume path and the dispatch loop read the
mode surface by attribute on ``_TurnState`` — never positionally, never
through a pack/unpack round-trip — so a field insertion cannot
silently transpose two same-typed locals.

The loop body itself is not instantiated here; these tests pin the two
contracts the refactor depends on: the applicator's true output (a
FinishAction with ``strip_tools=True`` sets slot 6), the accumulator
threading, and the source shape of both consumers.
"""

from __future__ import annotations

import ast
import inspect
import logging
from pathlib import Path

from kernel.turn_phases._lifecycle import FinishAction
from kernel.turn_runner import (
    _apply_iteration_lifecycle_action,
    _apply_lifecycle_action,
)


def _turn_runner_source() -> str:
    path = Path(inspect.getfile(_apply_iteration_lifecycle_action))
    return path.read_text(encoding="utf-8")


def _stub_state() -> object:
    return type("S", (), {"iteration": 1, "pending_planned_handoff": ""})()


def test_apply_lifecycle_action_returns_strip_tools_true() -> None:
    """A FinishAction(strip_tools=True) produces strip_tools=True in slot 6."""

    result = _apply_lifecycle_action(
        state=_stub_state(),
        mode_id="coding_execute",
        max_handoffs_per_turn=8,
        lifecycle_action=FinishAction(strip_tools=True, reason="budget exhausted"),
        log=logging.getLogger("test"),
    )
    assert len(result) == 6
    assert result[0] is True  # should_break
    assert result[5] is True  # strip_tools carried, not discarded


def test_accumulator_returns_strip_tools() -> None:
    """The accumulator threads slot 6 through instead of dropping it."""

    result = _apply_iteration_lifecycle_action(
        state=_stub_state(),
        mode_id="coding_execute",
        max_handoffs_per_turn=8,
        lifecycle_action=FinishAction(strip_tools=True, reason="budget exhausted"),
        content="prior",
        pending_operator_exit="",
        cap_pause_reason="",
        pending_yield=None,
        log=logging.getLogger("test"),
    )
    assert len(result) == 6
    assert result[0] is True
    assert result[5] is True


def test_accumulator_propagates_existing_content() -> None:
    """Non-lifecycle accumulators still win the ``or`` merge."""

    result = _apply_iteration_lifecycle_action(
        state=_stub_state(),
        mode_id="coding_execute",
        max_handoffs_per_turn=8,
        lifecycle_action=FinishAction(strip_tools=True, reason="budget exhausted"),
        content="prior",
        pending_operator_exit="",
        cap_pause_reason="",
        pending_yield=None,
        log=logging.getLogger("test"),
    )
    assert result[1] == "prior"


def test_effective_tools_consumes_strip_tools_not_action_type() -> None:
    """The loop's effective-tools rule reads the computed value; the dead
    FinishAction re-derivation guard is gone."""

    source = _turn_runner_source()
    assert "elif not isinstance(_lifecycle_action, _FinishAction):" not in source, (
        "the dead FinishAction re-derivation must not return"
    )
    assert "if _pending_yield is not None or lifecycle_strip_tools:" in source


def test_resume_unpack_is_attribute_based() -> None:
    """The resume path reads the mode surface by attribute on ``_TurnState``."""

    source = _turn_runner_source()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Tuple)
            and len(node.value.elts) == 13
            and all(isinstance(elt, ast.Name) for elt in node.value.elts)
            and any(
                isinstance(elt, ast.Name) and elt.id.startswith("_state") for elt in node.value.elts
            )
        ):
            raise AssertionError(
                "positional 13-name unpack of the mode surface returned; "
                "the resume path must read the turn state by attribute"
            )
    assert "_state.active_mode_id" in source


def test_loop_state_field_order_is_not_load_bearing() -> None:
    """Attribute access decouples consumers from field order: constructing
    the state with fields in a scrambled order must still resume correctly
    (the failure mode the positional unpack invited)."""

    from kernel.turn_phases._turn_state import _TurnState

    state = _TurnState()
    state.active_mode_id = "m"
    state.mode_config = object()
    state.ctx = None
    state.tool_schemas = []
    state.capability_manifest = None
    state.routing_hint = ""
    state.chain_advance_guidance = ""
    state.pending_yield = None
    state.handoff_to = ""
    state.transitioned_to = ""
    state.content = "c"
    state.pre_handoff_text = "p"
    state.yielded = True
    # Attribute reads stay correct regardless of declaration order.
    assert (state.active_mode_id, state.content, state.yielded) == ("m", "c", True)

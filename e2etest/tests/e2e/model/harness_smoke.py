"""Scripted model for harness_smoke.scenario.yml (see e2etest.scripted).

Exercises the offline path end to end with zero credentials: the model
script answers the planning attempt with "{}" (the planner tolerates a
non-plan) and answers the main chat turn by calling the core pack's
``web_search`` contract stub, then synthesizes a final reply from the
deterministic provider-unavailable error. Every model call resolves
through the runtime's gateway-override seam; routing, tool surface,
dispatch, receipts, and persistence are all production code.
"""

from __future__ import annotations

from e2etest.scripted import Reply, ScriptError, ScriptTurn


def auxiliary(turn: ScriptTurn) -> Reply:
    """Planner/distiller/title calls: empty structured output or empty text."""
    if turn.request.response_format is not None:
        return Reply.text("{}")
    return Reply.text("")


def respond(turn: ScriptTurn) -> Reply:
    if "web_search" not in turn.tool_names:
        raise ScriptError(f"web_search not offered; tools={sorted(turn.tool_names)}")
    if turn.count("web_search") == 0:
        return Reply.call("web_search", query="offline smoke probe")
    # The stub always returns the deterministic provider-unavailable error
    # with NO receipt (errors do not mint effect receipts). Prove the tool
    # error surfaced to the model, then finish the turn.
    last = turn.last("web_search")
    if last is None or "unavailable" not in last.text.lower():
        raise ScriptError(f"web_search failed oddly: {last.text if last else None}")
    return Reply.text(
        "The web_search capability answered with its deterministic "
        "provider-unavailable error, which is the expected offline behavior."
    )

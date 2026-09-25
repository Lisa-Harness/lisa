"""Tests for the ``turn.completed`` audit payload (Phase 8).

``build_turn_completed_payload`` carries mode/handoff/blocked metadata.
The effect-receipt ledger is **not** duplicated in the event payload — the
thread entry is the canonical, turn-addressable source of receipts. The
event links to the entry by turn number.
"""

from __future__ import annotations

from types import SimpleNamespace

from host.turn_logging import build_turn_completed_payload
from kernel.models import EffectReceipt


def _coerce(value: object) -> int:
    try:
        return max(0, int(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _result(**overrides: object) -> SimpleNamespace:
    base: dict[str, object] = {
        "handoff_to": None,
        "tool_call_count": 3,
        "iteration_count": 2,
        "blocked": False,
        "block_origin": "",
        "effect_receipts": [],
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_turn_completed_payload_omits_receipt_ledger() -> None:
    """The receipts field is not in the event payload — the thread entry owns it."""
    result = _result(
        effect_receipts=[
            EffectReceipt(
                effect="write",
                capability_id="markdown_forge_document",
                artifact_id="doc-1",
            ),
            EffectReceipt(effect="external_write", capability_id="memory.add"),
        ]
    )
    payload = build_turn_completed_payload(mode_id="markdown", result=result, coerce_int=_coerce)
    assert payload["mode_id"] == "markdown"
    assert "receipts" not in payload
    assert payload["tool_call_count"] == 3
    assert payload["handoff"] == ""


def test_turn_completed_payload_blocked_turn_metadata() -> None:
    """A blocked turn carries blocked/block_origin — receipts are in the entry."""
    result = _result(blocked=True, block_origin="no_tools")
    payload = build_turn_completed_payload(mode_id="remember", result=result, coerce_int=_coerce)
    assert "receipts" not in payload
    assert payload["blocked"] is True
    assert payload["block_origin"] == "no_tools"
    assert payload["tool_call_count"] == result.tool_call_count
    assert payload["handoff"] == ""


def test_turn_completed_payload_tolerates_sparse_results() -> None:
    """Partial result doubles (missing attrs) still produce a payload."""
    payload = build_turn_completed_payload(
        mode_id="default", result=SimpleNamespace(), coerce_int=_coerce
    )
    assert "receipts" not in payload
    assert payload["blocked"] is False
    assert payload["tool_call_count"] == 0

"""Tests for the ``turn.completed`` audit payload (Phase 8).

``build_turn_completed_payload`` carries the effect-receipt ledger so
hallucination signals — a mode with a completion contract whose ledger
is empty — are queryable from the JSONL stream without reconstructing
individual dispatch entries.
"""

from __future__ import annotations

from types import SimpleNamespace

from kernel.models import EffectReceipt

from host.turn_logging import build_turn_completed_payload


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


def test_turn_completed_payload_carries_receipt_ledger() -> None:
    """The receipts field lists every write receipt with its artifact id."""
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
    payload = build_turn_completed_payload(
        mode_id="markdown", result=result, coerce_int=_coerce
    )
    assert payload["mode_id"] == "markdown"
    assert payload["receipts"] == [
        {
            "capability_id": "markdown_forge_document",
            "effect": "write",
            "artifact_id": "doc-1",
        },
        {"capability_id": "memory.add", "effect": "external_write", "artifact_id": ""},
    ]


def test_turn_completed_payload_empty_ledger_is_explicit() -> None:
    """An empty ledger is the hallucination signal — it must be present."""
    result = _result(blocked=True, block_origin="no_tools")
    payload = build_turn_completed_payload(
        mode_id="remember", result=result, coerce_int=_coerce
    )
    assert payload["receipts"] == []
    assert payload["blocked"] is True
    assert payload["block_origin"] == "no_tools"
    assert payload["tool_call_count"] == result.tool_call_count
    assert payload["handoff"] == ""


def test_turn_completed_payload_tolerates_sparse_results() -> None:
    """Partial result doubles (missing attrs) still produce a payload."""
    payload = build_turn_completed_payload(
        mode_id="default", result=SimpleNamespace(), coerce_int=_coerce
    )
    assert payload["receipts"] == []
    assert payload["blocked"] is False
    assert payload["tool_call_count"] == 0

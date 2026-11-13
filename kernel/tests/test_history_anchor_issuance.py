"""Only kernel-issued history anchors are served to pack resolvers.

``_anchor_issued`` gates whether ``thread_read_content`` runs the owning
pack's ``resolve_history_anchor`` with anchor-derived parameters. The
live-message half must authenticate a stub the kernel rendered — the
whole tool message, as a history fold leaves it — not any message that
merely contains the stub text.
"""

from __future__ import annotations

from types import SimpleNamespace

from kernel.history import HistoryLedger, HistoryRecord, render_stub
from kernel.turn_phases._compaction import LiveArrayEstimator
from kernel.turn_phases._history import _anchor_issued


def _record(call_id: str, anchor: str = "") -> HistoryRecord:
    record = HistoryRecord(
        call_id=call_id,
        tool_id="peek_file",
        owner_id="lisa-pack-x",
        anchor=anchor,
        chars=10,
    )
    record.ref = "r" + "a" * 64
    return record


def test_ledger_record_anchor_is_issued() -> None:
    state = SimpleNamespace(history_ledger=HistoryLedger(), messages=[])
    state.history_ledger.record(_record("c1", anchor="peek_file:a.py"))
    assert _anchor_issued(state, "peek_file:a.py")
    assert not _anchor_issued(state, "peek_file:other.py")


def test_whole_message_rehydrate_stub_is_issued() -> None:
    record = _record("c1", anchor="peek_file:a.py")
    state = SimpleNamespace(
        history_ledger=HistoryLedger(),
        messages=[{"role": "tool", "tool_call_id": "c1", "content": render_stub(record)}],
    )
    assert _anchor_issued(state, "peek_file:a.py")


def test_stub_quoted_inside_tool_output_is_not_issued() -> None:
    record = _record("c1", anchor="peek_file:a.py")
    state = SimpleNamespace(
        history_ledger=HistoryLedger(),
        messages=[
            {
                "role": "tool",
                "tool_call_id": "c9",
                "content": f"an earlier turn rendered: {render_stub(record)}",
            }
        ],
    )
    assert not _anchor_issued(state, "peek_file:a.py")


def test_estimator_release_keeps_estimate_biased_high() -> None:
    estimator = LiveArrayEstimator()
    estimator.observe_request_tokens(1000)
    estimator.observe_append({"content": "x" * 3000})
    estimator.release(bytes_removed=1200)
    # The byte baseline drops in full; the adopted token anchor drops at a
    # quarter weight, so the estimate never dips below the real prompt.
    assert estimator._live_bytes == 1800
    assert estimator._estimate == 700

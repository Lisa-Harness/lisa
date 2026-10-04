"""Tests for the intake tick helpers in ``runtime.intake``."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from kernel.intake.contracts import CollectionResult, ProcessedState
from kernel.models import ContextItem

from runtime import intake

_COLLECTED = "2026-10-04T00:00:00+00:00"


def _alert_obj(**kwargs: Any) -> SimpleNamespace:
    defaults: dict[str, Any] = {
        "severity": "warn",
        "source": "stripe",
        "message": "payout failed",
        "detected_at": "2026-10-03T00:00:00+00:00",
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _sample_item(item_id: str = "state:test:one") -> ContextItem:
    return ContextItem(
        item_id=item_id,
        namespace="test",
        source="unit",
        source_type="intake",
        evidence_type="test",
        title="t",
        body="b",
        collected_at=_COLLECTED,
    )


# ── _evidence_from_alert ─────────────────────────────────────────────


def test_evidence_from_alert_dict_branch() -> None:
    alert = {
        "severity": "CRITICAL",
        "source": "github",
        "message": "ci broken",
        "detected_at": "2026-10-01T00:00:00+00:00",
    }

    item = intake._evidence_from_alert(alert=alert, collected_at=_COLLECTED)

    assert item is not None
    assert item.namespace == "alerts"
    assert item.source == "github"
    assert item.body == "ci broken"
    assert item.collected_at == "2026-10-01T00:00:00+00:00"
    assert item.title == "Critical alert from github"
    assert item.salience_score == 1.0
    assert item.tags == ["alert", "critical", "github"]
    assert item.metadata == {"provenance": "system_observed", "severity": "critical"}


def test_evidence_from_alert_object_branch() -> None:
    item = intake._evidence_from_alert(alert=_alert_obj(), collected_at=_COLLECTED)

    assert item is not None
    assert item.source == "stripe"
    assert item.title == "Warn alert from stripe"
    assert item.salience_score == 0.85
    assert item.collected_at == "2026-10-03T00:00:00+00:00"


def test_evidence_from_alert_defaults_and_fallthrough_salience() -> None:
    item = intake._evidence_from_alert(
        alert=_alert_obj(severity="weird", source="  ", detected_at=""),
        collected_at=_COLLECTED,
    )

    assert item is not None
    assert item.source == "system"
    assert item.collected_at == _COLLECTED
    assert item.salience_score == 0.5  # unknown severity falls through the map
    assert item.metadata["severity"] == "weird"


def test_evidence_from_alert_empty_message_is_none() -> None:
    assert intake._evidence_from_alert(alert={"message": "  "}, collected_at=_COLLECTED) is None
    assert (
        intake._evidence_from_alert(alert=_alert_obj(message=""), collected_at=_COLLECTED) is None
    )


def test_alert_severity_salience_mapping() -> None:
    for severity, expected in (("critical", 1.0), ("warn", 0.85), ("info", 0.65)):
        item = intake._evidence_from_alert(
            alert={"severity": severity, "message": "m"}, collected_at=_COLLECTED
        )
        assert item is not None
        assert item.salience_score == expected


# ── _evidence_from_intel ─────────────────────────────────────────────


def test_evidence_from_intel_dict_branch() -> None:
    intel = {
        "generated_at": "2026-10-02T00:00:00+00:00",
        "summary_bullets": ["one", "  ", "two"],
        "source": "trend",
    }

    item = intake._evidence_from_intel(intel_key="trend", intel=intel, collected_at=_COLLECTED)

    assert item is not None
    assert item.namespace == "intel"
    assert item.body == "- one\n- two"  # blank bullets are skipped
    assert item.source == "trend"
    assert item.collected_at == "2026-10-02T00:00:00+00:00"
    assert item.metadata == {"provenance": "authoritative"}


def test_evidence_from_intel_object_branch_uses_key_source() -> None:
    intel = SimpleNamespace(generated_at="", summary_bullets=["a"])

    item = intake._evidence_from_intel(intel_key="market", intel=intel, collected_at=_COLLECTED)

    assert item is not None
    assert item.source == "market"
    assert item.collected_at == _COLLECTED
    assert item.item_id == "state:intel:market"


def test_evidence_from_intel_no_bullets_is_none() -> None:
    assert (
        intake._evidence_from_intel(
            intel_key="k", intel={"summary_bullets": [" ", ""]}, collected_at=_COLLECTED
        )
        is None
    )
    assert (
        intake._evidence_from_intel(
            intel_key="k", intel=SimpleNamespace(summary_bullets=[]), collected_at=_COLLECTED
        )
        is None
    )


# ── _evidence_from_signal ────────────────────────────────────────────


def test_evidence_from_signal_dict_renders_scalar_pairs() -> None:
    snapshot = {
        "status": "degraded",
        "collected_at": "skip-me",
        "prs": 3,
        "healthy": True,
        "ratio": 0.5,
        "note": "ok",
        "nested": {"ignored": True},
    }

    item = intake._evidence_from_signal(
        collector_id="github", status="ok", snapshot=snapshot, collected_at=_COLLECTED
    )

    assert item is not None
    lines = item.body.splitlines()
    assert lines[0] == "status=degraded"
    assert "prs=3" in lines
    assert "healthy=True" in lines
    assert "ratio=0.5" in lines
    assert "note=ok" in lines
    assert not any(line.startswith("collected_at=") for line in lines)
    assert not any(line.startswith("nested=") for line in lines)
    assert item.title == "Signal: github (ok)"
    assert item.tags == ["signal", "github", "ok"]


def test_evidence_from_signal_non_dict_body() -> None:
    item = intake._evidence_from_signal(
        collector_id="cf", status="ok", snapshot="raw text", collected_at=_COLLECTED
    )

    assert item is not None
    assert item.body == "raw text"


def test_evidence_from_signal_status_default_and_empty_body() -> None:
    item = intake._evidence_from_signal(
        collector_id="c", status="timeout", snapshot=None, collected_at=_COLLECTED
    )
    # None is not a dict → str(None) body still yields a record.
    assert item is not None
    assert item.body == "None"

    empty = intake._evidence_from_signal(
        collector_id="c", status="err", snapshot={}, collected_at=_COLLECTED
    )
    assert empty is not None  # dict branch always emits at least status=<status>
    assert empty.body == "status=err"


# ── _entry_point_pack_name / _discover_processing_loops ──────────────


def test_entry_point_pack_name_edge_cases() -> None:
    assert intake._entry_point_pack_name(SimpleNamespace(dist=None)) == ""
    assert intake._entry_point_pack_name(SimpleNamespace(dist=SimpleNamespace(metadata=None))) == ""
    ep = SimpleNamespace(dist=SimpleNamespace(metadata={"Name": "Lisa Pack Core"}))
    assert intake._entry_point_pack_name(ep) == "lisa-pack-core"


def test_discover_processing_loops_no_group(monkeypatch: Any) -> None:
    import importlib.metadata

    def _boom(group: str) -> Any:
        raise RuntimeError("no group")

    monkeypatch.setattr(importlib.metadata, "entry_points", _boom)
    assert intake._discover_processing_loops() == []


def test_discover_processing_loops_filters_and_validates(monkeypatch: Any) -> None:
    import importlib.metadata

    class _Good:
        loop_id = "good"

        async def process(self, ctx: Any) -> None:  # pragma: no cover - protocol shape only
            return None

        @classmethod
        def dist_name(cls) -> str:
            return "lisa-pack-alpha"

    class _Incomplete:
        pass

    def _ep(name: str, loaded: Any, dist_name: str) -> Any:
        class _Ep:
            dist = SimpleNamespace(metadata={"Name": dist_name})

            def load(self) -> Any:
                return loaded

            def __init__(self, ep_name: str) -> None:
                self.name = ep_name

        return _Ep(name)

    def _boom_loader(name: str) -> Any:
        class _Ep:
            dist = None

            def load(self) -> Any:
                raise RuntimeError("bad import")

            def __init__(self, ep_name: str) -> None:
                self.name = ep_name

        return _Ep(name)

    eps = [
        _ep("beta", _Good, "lisa-pack-beta"),  # kept
        _ep("gamma", _Incomplete, "lisa-pack-gamma"),  # fails protocol check
        _boom_loader("delta"),  # load raises
        _ep("zeta", _Good, "Lisa Pack Zeta"),  # excludable by pack name
    ]
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group=None: eps)

    loops_all = intake._discover_processing_loops()
    assert [loop.loop_id for loop in loops_all] == ["good", "good"]

    loops_filtered = intake._discover_processing_loops(excluded_packs={"lisa-pack-zeta"})
    assert [loop.loop_id for loop in loops_filtered] == ["good"]


# ── _merge_results / _evidence_from_processed_state ──────────────────


def test_merge_results_combines_signals_and_processed() -> None:
    collections = [
        CollectionResult(
            collector_id="github",
            status="ok",
            snapshot={"status": "ok", "prs": 2},
        ),
        CollectionResult(collector_id="broken", status="error", snapshot=None),
    ]
    processed = [
        ProcessedState(
            loop_id="anomaly",
            active_alerts=[{"severity": "warn", "source": "s", "message": "m"}],
            signal_snapshots={"stripe": {"status": "ok", "mrr": 10}},
            intel_collections={"daily": {"summary_bullets": ["note"]}},
        )
    ]

    state = intake._merge_results(collections, processed)

    kinds = {(item.namespace, item.evidence_type) for item in state.context_items}
    assert ("signals", "signal_snapshot") in kinds
    assert ("alerts", "alert") in kinds
    assert ("intel", "intel_summary") in kinds
    sources = {item.source for item in state.context_items}
    assert "github" in sources
    assert "stripe" in sources


def test_evidence_from_processed_state_all_fields() -> None:
    carried = _sample_item()
    ps = ProcessedState(
        loop_id="all",
        context_items=[carried],
        signal_snapshots={"cf": "non-dict"},
        intel_collections={"news": {"summary_bullets": ["x"]}},
        active_alerts=[{"severity": "info", "message": "fyi"}],
    )

    out = intake._evidence_from_processed_state(ps, _COLLECTED)

    assert carried in [item for item in out if item is not None]
    assert len(out) == 4
    assert all(item is not None for item in out)


def test_evidence_from_processed_state_empty() -> None:
    assert intake._evidence_from_processed_state(ProcessedState(), _COLLECTED) == []

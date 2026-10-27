"""Producer-side self-handoff guard for the capability routing hint.

The self-handoff guard used to live only in ``emit_transition_hint``, a
wrapper the primary hint site does not call: the turn runner invokes
``_compute_capability_routing_hint`` directly. That made the wrapper's
"single contract" claim structurally false — the capability producer had
no self-comparison of its own, and only an incidental invariant
(``synthesize_routing_transitions`` excludes self from ``routing_targets``)
kept a self-target unreachable.

These tests pin the guard where it belongs: the producer drops ``mode_id``
from ``routing_targets`` on entry, so no consumer can forget it. They call
``_compute_capability_routing_hint`` directly, per the kernel test-seams
rule (patch ``kernel.turn_phases._routing_hint.get_workflow_catalog``), and
stay self-contained rather than importing fixtures from
``test_turn_runner.py``, which creates a catalog import cycle.
"""

from __future__ import annotations

import logging

import pytest
from kernel.models import CapabilityManifest, WorkflowModeConfig
from kernel.turn_phases._routing_hint import _compute_capability_routing_hint

#: A synthetic intent tag deliberately absent from the representative
#: routing vocabulary's producer (research/gather/learn/read/analyze/
#: review/architect) and consumer (code/write/implement/review) sets, so the
#: producer-first and fork branches cannot fire and the single-handoff
#: branch is what the assertions observe.
_GAP_TAG = "selfguard"


def _manifest() -> CapabilityManifest:
    """A manifest with no entries — the hint does not consult it."""
    return CapabilityManifest(generated_at="2026-01-01T00:00:00Z", entries=[])


def _mode_config(*, mode_id: str, satisfies_intents: list[str] | None = None) -> WorkflowModeConfig:
    """A minimal mode config declaring the synthetic gap tag."""
    return WorkflowModeConfig(
        mode_id=mode_id,
        start_stage_id="none",
        mode_kind="chat",
        routable=True,
        fork_allowed=False,
        satisfies_intents=list(satisfies_intents) if satisfies_intents else [],
        required_capabilities=[],
        routing_intent_tags=[_GAP_TAG],
        routing_intent_keywords={_GAP_TAG: ["selfguard request"]},
    )


def test_producer_drops_self_target_when_another_target_covers_the_gap(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A ``routing_targets`` list that CONTAINS the current ``mode_id`` must
    not yield a hint naming the current mode.

    Both targets cover the whole gap and the current mode is listed first, so
    before the guard moved into the producer the single-handoff branch picked
    ``covering_targets[0]`` — the current mode itself. This is the shape the
    accepted delegate change makes reachable: a target equal to the parent
    mode entering the capability-hint path.
    """
    current_cfg = _mode_config(mode_id="selfcurrent")
    catalog = {
        "selfcurrent": current_cfg,
        "selfother": _mode_config(mode_id="selfother"),
    }
    monkeypatch.setattr(
        "kernel.turn_phases._routing_hint.get_workflow_catalog",
        lambda: catalog,
    )

    with caplog.at_level(logging.INFO, logger="kernel.turn_phases._routing_hint"):
        hint = _compute_capability_routing_hint(
            mode_id="selfcurrent",
            query="please handle this selfguard request",
            capability_manifest=_manifest(),
            routing_targets=["selfcurrent", "selfother"],
            mode_config=current_cfg,
        )

    assert hint, "a real covering target exists, so the hint must not be silent"
    assert "selfcurrent" not in hint, f"hint named the current mode: {hint!r}"
    assert "selfother" in hint, f"hint did not name the covering target: {hint!r}"
    assert "dropping self-target selfcurrent" in caplog.text


def test_producer_returns_empty_when_only_the_self_target_is_routable(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``routing_targets == [mode_id]`` filters to empty, so the producer
    returns ``""`` and logs instead of hinting a handoff to itself."""
    current_cfg = _mode_config(mode_id="selfcurrent")
    catalog = {"selfcurrent": current_cfg}
    monkeypatch.setattr(
        "kernel.turn_phases._routing_hint.get_workflow_catalog",
        lambda: catalog,
    )

    with caplog.at_level(logging.INFO, logger="kernel.turn_phases._routing_hint"):
        hint = _compute_capability_routing_hint(
            mode_id="selfcurrent",
            query="please handle this selfguard request",
            capability_manifest=_manifest(),
            routing_targets=["selfcurrent"],
            mode_config=current_cfg,
        )

    assert hint == ""
    assert "dropping self-target selfcurrent" in caplog.text

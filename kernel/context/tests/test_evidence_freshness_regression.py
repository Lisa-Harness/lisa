"""Regression tests for the kernel config and evidence-freshness fixes.

Guards against two regressions:

1. ``kernel.config.Settings`` must import cleanly. Removing the memory-store
   settings fields left dangling names in the numeric field validator, which
   made the whole ``kernel`` package unimportable under Pydantic v2.
2. ``_new_evidence`` must set ``freshness_expires_at`` to the raw
   ``collected_at`` string (a ``str``), not the ``datetime`` returned by
   ``_parse_iso``, which contradicted the model field type.
"""

from __future__ import annotations

from datetime import UTC, datetime

from kernel.config import Settings
from kernel.context.context_broker import _new_evidence
from kernel.models import ContextEvidence


def test_settings_imports_and_has_active_memory_knobs() -> None:
    """The module-level Settings singleton can be constructed and retains the
    active memory recall/auto-capture knobs after the memory-store removal.
    """
    settings = Settings()
    assert settings.memory_recall_enabled is True
    assert settings.memory_recall_pre_turn_limit == 3
    assert settings.memory_auto_capture_enabled is True


def test_new_evidence_freshness_expires_at_is_collected_at_string() -> None:
    """``_new_evidence`` sets ``freshness_expires_at`` to the raw
    ``collected_at`` string, matching the model field type.
    """
    now = datetime.now(UTC).isoformat()
    evidence = _new_evidence(
        evidence_id="evt:regression",
        source="test",
        evidence_type="test",
        title="Regression evidence",
        body="body",
        collected_at=now,
    )
    assert isinstance(evidence, ContextEvidence)
    assert evidence.freshness_expires_at == now
    assert isinstance(evidence.freshness_expires_at, str)

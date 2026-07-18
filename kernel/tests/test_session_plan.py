"""Focused coverage for strict SessionManifestV2 compilation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from kernel.session_plan import (
    CompactionTuning,
    ContextHistoryPlan,
    SessionPlanError,
    compile_session_plan,
)

_SCHEMA_VERSION = 2
_SHA256_LENGTH = 64


def _manifest(*, extra: str = "", body: str = "# Primary Session\n\nAct precisely.\n") -> str:
    return f"""---
schema_version: 2
name: test-session
surface:
  type: personal_dm
  id: host-console
  memory_scope: testing
resources:
  packs:
    include: [lisa-pack-core, lisa-pack-memory]
    required: [lisa-pack-core]
  servers:
    exclude: [legacy-server]
providers:
  capabilities:
    memory: lisa-pack-mem0
routing:
  start_mode: default
context:
  lanes:
    - lane_id: operational
      min_tokens: 128
      max_tokens: 512
      render_order: 0
    - lane_id: history
      max_tokens: 256
      render_order: 10
  sources:
    - rule_id: alerts
      lane_id: operational
      evidence_types: [alert]
      priority: 100
    - rule_id: fallback
      lane_id: history
      fallback: true
{extra}---
{body}"""


def test_compile_session_plan_preserves_primary_instruction_and_plan_digest() -> None:
    text = _manifest()

    plan = compile_session_plan(text=text, source_uri="workspace:session_context.md")

    assert plan.manifest.schema_version == _SCHEMA_VERSION
    assert plan.manifest.resources.packs.required == ("lisa-pack-core",)
    assert plan.manifest.providers.capabilities == {"memory": "lisa-pack-mem0"}
    assert plan.instructions.primary_body == "# Primary Session\n\nAct precisely."
    assert len(plan.instructions.source_sha256) == _SHA256_LENGTH
    assert len(plan.plan_digest) == _SHA256_LENGTH
    assert plan.plan_digest == compile_session_plan(
        text=text,
        source_uri="workspace:session_context.md",
    ).plan_digest


def test_compile_session_plan_preserves_owner_pack_configuration() -> None:
    text = _manifest(
        extra=(
            "resources:\n"
            "  packs:\n"
            "    include: [lisa-pack-core, lisa-pack-coding]\n"
            "    configuration:\n"
            "      lisa-pack-coding:\n"
            "        workspace_root: .\n"
            "        checks:\n"
            "          - id: unit\n"
            "            argv: [uv, run, pytest, -q]\n"
        )
    )

    plan = compile_session_plan(text=text, source_uri="workspace:session_context.md")

    assert plan.manifest.resources.packs.configuration == {
        "lisa-pack-coding": {
            "workspace_root": ".",
            "checks": [{"id": "unit", "argv": ["uv", "run", "pytest", "-q"]}],
        }
    }


def test_compile_session_plan_rejects_legacy_and_unknown_front_matter() -> None:
    text = _manifest(extra="replay:\n  trigger_mode: manual\n")

    with pytest.raises(SessionPlanError, match="Extra inputs are not permitted"):
        compile_session_plan(text=text, source_uri="workspace:session_context.md")


def test_compile_session_plan_rejects_conflicting_resource_selection() -> None:
    text = _manifest(
        extra="resources:\n  packs:\n    include: [lisa-pack-core]\n    exclude: [lisa-pack-core]\n"
    )

    with pytest.raises(SessionPlanError, match="include and exclude overlap"):
        compile_session_plan(text=text, source_uri="workspace:session_context.md")


def test_compile_session_plan_rejects_context_rules_for_unknown_lanes() -> None:
    text = _manifest(
    extra=(
      "context:\n"
      "  lanes:\n"
      "    - lane_id: operational\n"
      "  sources:\n"
      "    - rule_id: invalid\n"
      "      lane_id: missing\n"
      "      fallback: true\n"
    )
    )

    with pytest.raises(SessionPlanError, match="unknown lanes"):
        compile_session_plan(text=text, source_uri="workspace:session_context.md")


def test_compile_session_plan_requires_nonempty_body() -> None:
    with pytest.raises(SessionPlanError, match="must not be empty"):
        compile_session_plan(
            text=_manifest(body="\n"),
            source_uri="workspace:session_context.md",
        )


def test_bundled_default_session_compiles_with_core_mode_provider() -> None:
    """The bundled default must compile and assign the core pack to the default mode.

    The bundled default intentionally does not declare ``providers.tools`` for
    ``web_search`` — that catalog collision belongs to the host's
    ``session_context.md``, not the core pack's bundled fallback.  The search
    pack ships its own graceful ``web_search`` stub, so the bundled default
    stays pack-encapsulated (no reference to external search provider packs).
    """
    default_session_path = (
        Path(__file__).resolve().parents[2]
        / "sdk"
        / "packs"
        / "core"
        / "sessions"
        / "default.md"
    )
    text = default_session_path.read_text(encoding="utf-8")

    plan = compile_session_plan(text=text, source_uri=str(default_session_path))

    assert plan.manifest.providers.modes == {"default": "lisa-pack-core"}
    assert plan.manifest.providers.tools == {}


# ── CompactionTuning ────────────────────────────────────────────────────


def test_compaction_tuning_defaults_disabled_and_post_turn() -> None:
    """CompactionTuning defaults to disabled, post_turn trigger, safe thresholds."""
    tuning = CompactionTuning()
    assert tuning.enabled is False
    assert tuning.trigger == "post_turn"
    assert tuning.pressure_threshold == 0.75
    assert tuning.hard_threshold == 0.90
    assert tuning.min_interval_turns == 4
    assert tuning.retention_count == 6
    assert tuning.preferred_model_id == "deep"
    assert tuning.summarize_operation == "checkpoint_history_summarize"


def test_compaction_tuning_rejects_unknown_keys() -> None:
    """Strict extra=forbid applies to CompactionTuning."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CompactionTuning.model_validate({"enabled": True, "bogus": 1})


def test_compaction_tuning_rejects_thresholds_out_of_range() -> None:
    """Pressure and hard thresholds must be within [0.0, 1.0]."""
    with pytest.raises(ValidationError):
        CompactionTuning.model_validate({"pressure_threshold": 1.5})
    with pytest.raises(ValidationError):
        CompactionTuning.model_validate({"hard_threshold": -0.1})


def test_compaction_tuning_rejects_hard_below_pressure() -> None:
    """hard_threshold must be >= pressure_threshold (guardrail invariant)."""
    with pytest.raises(ValidationError, match="hard_threshold must be >="):
        CompactionTuning.model_validate(
            {"pressure_threshold": 0.9, "hard_threshold": 0.5}
        )


def test_compaction_tuning_rejects_bad_trigger_literal() -> None:
    """trigger is a closed Literal; unknown values fail validation."""
    with pytest.raises(ValidationError, match="trigger"):
        CompactionTuning.model_validate({"trigger": "mid_turn"})


def test_context_history_plan_defaults_compaction_disabled() -> None:
    """ContextHistoryPlan.compaction defaults to a disabled CompactionTuning."""
    plan = ContextHistoryPlan()
    assert isinstance(plan.compaction, CompactionTuning)
    assert plan.compaction.enabled is False


def test_context_history_plan_accepts_explicit_compaction_block() -> None:
    """A plan can opt in to compaction via the nested compaction block."""
    plan = ContextHistoryPlan.model_validate(
        {
            "compaction": {
                "enabled": True,
                "pressure_threshold": 0.5,
                "hard_threshold": 0.8,
                "min_interval_turns": 2,
            }
        }
    )
    assert plan.compaction.enabled is True
    assert plan.compaction.pressure_threshold == 0.5
    assert plan.compaction.min_interval_turns == 2


def test_compile_session_plan_accepts_compaction_block_in_history() -> None:
    """A full session manifest with context.history.compaction compiles."""
    text = _manifest(
        extra=(
            "context:\n"
            "  history:\n"
            "    recent_message_limit: 4\n"
            "    compaction:\n"
            "      enabled: true\n"
            "      pressure_threshold: 0.6\n"
            "      hard_threshold: 0.85\n"
            "      min_interval_turns: 3\n"
            "      retention_count: 4\n"
        )
    )
    plan = compile_session_plan(text=text, source_uri="workspace:session_context.md")
    comp = plan.manifest.context.history.compaction
    assert comp.enabled is True
    assert comp.pressure_threshold == 0.6
    assert comp.hard_threshold == 0.85
    assert comp.min_interval_turns == 3
    assert comp.retention_count == 4


def test_compile_session_plan_rejects_unknown_compaction_key() -> None:
    """Strict extra=forbid applies inside the compaction block too."""
    text = _manifest(
        extra=(
            "context:\n"
            "  history:\n"
            "    compaction:\n"
            "      enabled: true\n"
            "      bogus: 1\n"
        )
    )
    with pytest.raises(SessionPlanError, match="Extra inputs are not permitted"):
        compile_session_plan(text=text, source_uri="workspace:session_context.md")


def test_compile_session_plan_rejects_inverted_compaction_thresholds() -> None:
    """A compaction block with hard < pressure fails plan compilation."""
    text = _manifest(
        extra=(
            "context:\n"
            "  history:\n"
            "    compaction:\n"
            "      enabled: true\n"
            "      pressure_threshold: 0.9\n"
            "      hard_threshold: 0.5\n"
        )
    )
    with pytest.raises(SessionPlanError, match="hard_threshold must be >="):
        compile_session_plan(text=text, source_uri="workspace:session_context.md")


def test_compile_session_plan_preserves_default_compaction_when_omitted() -> None:
    """Existing plans that omit the compaction block still compile and default to disabled."""
    plan = compile_session_plan(text=_manifest(), source_uri="workspace:session_context.md")
    assert plan.manifest.context.history.compaction.enabled is False
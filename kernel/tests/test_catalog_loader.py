"""Tests for ``kernel.ipsar_fsm`` disk-scanning catalog loader and helpers.

These tests exercise the pure disk path directly, bypassing the MCP
``lisa-catalog`` server. They cover three surfaces:

- ``parse_front_matter``: YAML front-matter extraction from markdown.
- ``_apply_mode_overrides``: ``_overrides.yaml`` deep-merge and the kernel-owned
  key denylist.
- ``load_catalog_from_disk``: end-to-end mode directory scan, focusing on the
  ``extra_front_matter`` opaque remainder that carries app/pack-owned keys
  (e.g. ``lenses``) the kernel never interprets.

The kernel owns a fixed set of front-matter keys (routing, FSM, operation
bindings, workshop config) and passes everything else through untouched as
extension data — so adding a new app-owned key requires no kernel change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from kernel.ipsar_fsm import (
    WorkflowValidationError,
    _apply_mode_overrides,
    load_catalog_from_disk,
    parse_front_matter,
    validate_workflow_catalog,
)
from kernel.models import WorkflowModeConfig

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _write_mode(
    root: Path,
    mode_id: str = "demo",
    *,
    mode_kind: str = "workflow",
    command_pattern: str = "/demo",
    description: str = "demo mode",
    start_stage_id: str = "start",
    extra_front_matter: dict[str, Any] | None = None,
    stage_files: dict[str, dict[str, Any]] | None = None,
) -> Path:
    """Materialise one minimal valid workflow mode under ``root``.

    Returns the mode directory. The ``_mode.md`` carries the required
    front-matter fields plus any ``extra_front_matter`` app-owned keys. One
    terminal stage file is written by default so the loader's stage scan has
    something to consume for workflow modes.
    """
    mode_dir = root / mode_id
    mode_dir.mkdir(parents=True, exist_ok=True)

    front: dict[str, Any] = {
        "mode_id": mode_id,
        "start_stage_id": start_stage_id,
        "mode_kind": mode_kind,
        "command_pattern": command_pattern,
        "description": description,
    }
    if extra_front_matter:
        front.update(extra_front_matter)

    _write_front_matter_file(mode_dir / "_mode.md", front, body=f"# {mode_id}")

    stages = (
        stage_files
        if stage_files is not None
        else {
            start_stage_id: _stage_front_matter(start_stage_id, terminal=True),
        }
    )
    for stage_id, stage_front in stages.items():
        _write_front_matter_file(mode_dir / f"{stage_id}.md", stage_front, body=f"# {stage_id}")

    return mode_dir


def _stage_front_matter(
    stage_id: str,
    *,
    terminal: bool = True,
    allowed_next: list[str] | None = None,
    allow_user_questions: bool = False,
    required_connectors: list[str] | None = None,
    description: str = "stage",
) -> dict[str, Any]:
    return {
        "stage_id": stage_id,
        "terminal": terminal,
        "allowed_next_stages": allowed_next if allowed_next is not None else [],
        "allow_user_questions": allow_user_questions,
        "required_connectors": required_connectors if required_connectors is not None else [],
        "description": description,
    }


def _write_front_matter_file(path: Path, front_matter: dict[str, Any], body: str = "") -> None:
    """Write a markdown file with YAML front matter and an optional body."""
    lines = ["---", yaml.safe_dump(front_matter, sort_keys=False).rstrip(), "---"]
    if body:
        lines.append("")
        lines.append(body)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_overrides(mode_dir: Path, overrides: dict[str, Any]) -> Path:
    """Write a sibling ``_overrides.yaml`` next to ``_mode.md``."""
    overrides_path = mode_dir / "_overrides.yaml"
    overrides_path.write_text(yaml.safe_dump(overrides, sort_keys=False), encoding="utf-8")
    return overrides_path


def _load(root: Path) -> dict[str, WorkflowModeConfig]:
    return load_catalog_from_disk(instructions_dir=root)


# ---------------------------------------------------------------------------
# parse_front_matter
# ---------------------------------------------------------------------------


def test_parse_front_matter_extracts_yaml_and_body() -> None:
    """A valid front-matter block yields its dict and the trailing body text."""
    text = "---\nmode_id: demo\ndescription: hi\n---\n\n# Body line one\nbody line two"
    front, body = parse_front_matter(text)

    assert front == {"mode_id": "demo", "description": "hi"}
    assert body == "# Body line one\nbody line two"


def test_parse_front_matter_returns_empty_dict_when_no_front_matter() -> None:
    """No leading ``---`` means the entire text is returned as the body."""
    text = "# Just a heading\n\nno front matter here"
    front, body = parse_front_matter(text)

    assert front == {}
    assert body == text


# ---------------------------------------------------------------------------
# _apply_mode_overrides
# ---------------------------------------------------------------------------


def test_apply_mode_overrides_no_file_returns_base_unchanged(tmp_path: Path) -> None:
    """With no ``_overrides.yaml`` the base front matter is returned unchanged."""
    mode_dir = tmp_path / "demo"
    mode_dir.mkdir()
    base = {"mode_id": "demo", "description": "authored", "lenses": ["lens.conversation"]}

    assert _apply_mode_overrides(mode_dir, base) is base


def test_apply_mode_overrides_passes_through_app_owned_keys(tmp_path: Path) -> None:
    """App-owned keys in overrides pass through untouched (deep-merged on top)."""
    mode_dir = tmp_path / "demo"
    mode_dir.mkdir()
    base = {"mode_id": "demo", "description": "authored"}
    _write_overrides(mode_dir, {"display_name": "Demo", "lenses": ["lens.reflection"]})

    merged = _apply_mode_overrides(mode_dir, base)
    assert merged["display_name"] == "Demo"
    assert merged["lenses"] == ["lens.reflection"]


def test_apply_mode_overrides_allows_kernel_owned_overridable_keys(tmp_path: Path) -> None:
    """``description`` and ``default_model`` are kernel-owned but overridable."""
    mode_dir = tmp_path / "demo"
    mode_dir.mkdir()
    base = {"mode_id": "demo", "description": "authored", "default_model": "gpt-old"}
    _write_overrides(mode_dir, {"description": "overridden", "default_model": "gpt-new"})

    merged = _apply_mode_overrides(mode_dir, base)
    assert merged["description"] == "overridden"
    assert merged["default_model"] == "gpt-new"


def test_apply_mode_overrides_deep_merges_app_owned_lists(tmp_path: Path) -> None:
    """Dict-valued overrides are deep-merged; list-valued overrides replace."""
    mode_dir = tmp_path / "demo"
    mode_dir.mkdir()
    base = {
        "mode_id": "demo",
        "custom": {"a": 1, "nested": {"x": 1}},
        "custom_list": ["old"],
    }
    _write_overrides(
        mode_dir,
        {"custom": {"b": 2, "nested": {"y": 2}}, "custom_list": ["new"]},
    )

    merged = _apply_mode_overrides(mode_dir, base)
    assert merged["custom"] == {"a": 1, "b": 2, "nested": {"x": 1, "y": 2}}
    # Lists replace (no concatenation).
    assert merged["custom_list"] == ["new"]


def test_apply_mode_overrides_replaces_lenses(tmp_path: Path) -> None:
    """``lenses`` is an app-owned list; overrides replace (not concatenate)."""
    mode_dir = tmp_path / "demo"
    mode_dir.mkdir()
    base = {"mode_id": "demo", "lenses": ["lens.conversation"]}
    _write_overrides(mode_dir, {"lenses": ["lens.reflection"]})

    merged = _apply_mode_overrides(mode_dir, base)
    assert merged["lenses"] == ["lens.reflection"]


def test_apply_mode_overrides_rejects_protected_kernel_key(tmp_path: Path) -> None:
    """Overriding a kernel-owned protected key is rejected at startup."""
    mode_dir = tmp_path / "demo"
    mode_dir.mkdir()
    base = {"mode_id": "demo"}
    _write_overrides(mode_dir, {"command_pattern": "/hijacked"})

    with pytest.raises(WorkflowValidationError, match="kernel-owned keys"):
        _apply_mode_overrides(mode_dir, base)


def test_apply_mode_overrides_rejects_protected_mode_id(tmp_path: Path) -> None:
    """``mode_id`` is the canonical protected kernel key — overriding it fails."""
    mode_dir = tmp_path / "demo"
    mode_dir.mkdir()
    base = {"mode_id": "demo"}
    _write_overrides(mode_dir, {"mode_id": "hijacked"})

    with pytest.raises(WorkflowValidationError, match="kernel-owned keys"):
        _apply_mode_overrides(mode_dir, base)


def test_apply_mode_overrides_rejects_disallowed_key(tmp_path: Path) -> None:
    """Any protected kernel key (here ``routable``) is disallowed."""
    mode_dir = tmp_path / "demo"
    mode_dir.mkdir()
    base = {"mode_id": "demo"}
    _write_overrides(mode_dir, {"routable": True})

    with pytest.raises(WorkflowValidationError, match="kernel-owned keys"):
        _apply_mode_overrides(mode_dir, base)


def test_apply_mode_overrides_rejects_invalid_yaml(tmp_path: Path) -> None:
    """Malformed YAML in ``_overrides.yaml`` raises ``WorkflowValidationError``."""
    mode_dir = tmp_path / "demo"
    mode_dir.mkdir()
    (mode_dir / "_overrides.yaml").write_text(":\n  - [unterminated", encoding="utf-8")

    with pytest.raises(WorkflowValidationError, match="Invalid YAML"):
        _apply_mode_overrides(mode_dir, {"mode_id": "demo"})


def test_apply_mode_overrides_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    """A non-mapping overrides file (e.g. a YAML list) is rejected."""
    mode_dir = tmp_path / "demo"
    mode_dir.mkdir()
    (mode_dir / "_overrides.yaml").write_text("- just\n- a\n- list\n", encoding="utf-8")

    with pytest.raises(WorkflowValidationError, match="must be a YAML mapping"):
        _apply_mode_overrides(mode_dir, {"mode_id": "demo"})


# ---------------------------------------------------------------------------
# load_catalog_from_disk: extra_front_matter collection
# ---------------------------------------------------------------------------


def test_load_catalog_from_disk_reads_lenses_from_front_matter(tmp_path: Path) -> None:
    """A ``lenses`` app-owned key in ``_mode.md`` flows into ``extra_front_matter``."""
    _write_mode(tmp_path, extra_front_matter={"lenses": ["lens.conversation"]})
    catalog = _load(tmp_path)

    assert "demo" in catalog
    assert catalog["demo"].extra_front_matter == {"lenses": ["lens.conversation"]}


def test_load_catalog_from_disk_lenses_default_empty_when_absent(tmp_path: Path) -> None:
    """No ``lenses`` key means ``extra_front_matter`` carries no ``lenses`` entry."""
    _write_mode(tmp_path)
    catalog = _load(tmp_path)

    assert "lenses" not in catalog["demo"].extra_front_matter
    assert catalog["demo"].extra_front_matter == {}


def test_load_catalog_from_disk_collects_app_owned_keys_into_extra_front_matter(
    tmp_path: Path,
) -> None:
    """Multiple app-owned keys are all collected opaquely into ``extra_front_matter``."""
    _write_mode(
        tmp_path,
        extra_front_matter={
            "lenses": ["lens.conversation"],
            "display_name": "Demo",
            "custom": {"nested": True},
        },
    )
    catalog = _load(tmp_path)

    extra = catalog["demo"].extra_front_matter
    assert extra["lenses"] == ["lens.conversation"]
    assert extra["display_name"] == "Demo"
    assert extra["custom"] == {"nested": True}


def test_load_catalog_from_disk_does_not_leak_kernel_keys_into_extra_front_matter(
    tmp_path: Path,
) -> None:
    """Kernel-owned front-matter keys never appear in ``extra_front_matter``."""
    _write_mode(
        tmp_path,
        extra_front_matter={
            "routable": True,
            "priority": 5,
            "default_model": "gpt-demo",
            "description": "should not leak",
        },
    )
    catalog = _load(tmp_path)

    extra = catalog["demo"].extra_front_matter
    assert "routable" not in extra
    assert "priority" not in extra
    assert "default_model" not in extra
    assert "description" not in extra
    # And the typed fields picked them up.
    assert catalog["demo"].routable is True
    assert catalog["demo"].priority == 5
    assert catalog["demo"].default_model == "gpt-demo"
    assert catalog["demo"].description == "should not leak"


def test_load_catalog_from_disk_extra_front_matter_empty_when_no_app_keys(
    tmp_path: Path,
) -> None:
    """A mode with only kernel-owned keys has an empty ``extra_front_matter``."""
    _write_mode(tmp_path)
    catalog = _load(tmp_path)

    assert catalog["demo"].extra_front_matter == {}


# ---------------------------------------------------------------------------
# load_catalog_from_disk: yield_policy
# ---------------------------------------------------------------------------


def test_load_catalog_from_disk_reads_yield_policy(tmp_path: Path) -> None:
    """``yield_policy`` flows from front matter into the typed field.

    The unified turn lifecycle routes ``yield`` actions via this per-mode
    policy. Authored by packs; the kernel never names mode ids in code.
    """
    from kernel.models import YieldPolicy

    _write_mode(
        tmp_path,
        extra_front_matter={
            "yield_policy": {
                "allowed": False,
                "fallback_mode": "discuss",
                "is_default_yield_target": False,
                "default_question": "What should I focus on?",
            }
        },
    )
    catalog = _load(tmp_path)

    policy = catalog["demo"].yield_policy
    assert isinstance(policy, YieldPolicy)
    assert policy.allowed is False
    assert policy.fallback_mode == "discuss"
    assert policy.is_default_yield_target is False
    assert policy.default_question == "What should I focus on?"
    # yield_policy is kernel-owned — it must NOT leak into extra_front_matter.
    assert "yield_policy" not in catalog["demo"].extra_front_matter


def test_load_catalog_from_disk_yield_policy_defaults_when_absent(
    tmp_path: Path,
) -> None:
    """A mode with no ``yield_policy`` front matter gets the default-empty policy."""
    from kernel.models import YieldPolicy

    _write_mode(tmp_path)
    catalog = _load(tmp_path)

    policy = catalog["demo"].yield_policy
    assert isinstance(policy, YieldPolicy)
    assert policy.allowed is True
    assert policy.fallback_mode == ""
    assert policy.is_default_yield_target is False
    assert policy.default_question == ""


def test_load_catalog_from_disk_rejects_non_mapping_yield_policy(
    tmp_path: Path,
) -> None:
    """A non-dict ``yield_policy`` is a catalog validation error."""
    _write_mode(tmp_path, extra_front_matter={"yield_policy": "not-a-mapping"})
    with pytest.raises(WorkflowValidationError, match="yield_policy"):
        _load(tmp_path)


# ---------------------------------------------------------------------------
# load_catalog_from_disk: fork_allowed
# ---------------------------------------------------------------------------


def test_load_catalog_from_disk_reads_fork_allowed_true(tmp_path: Path) -> None:
    """``fork_allowed: true`` flows from front matter into the typed field.

    A mode that opts into being a fork target commits to completing or
    failing without asking the user. The routing fork decision and the fork
    enforcer refuse a fork when any candidate target lacks this.
    """
    _write_mode(
        tmp_path,
        extra_front_matter={
            "fork_allowed": True,
            "satisfies_intents": ["write"],
            "allowed_tool_ids": ["write_file"],
            "yield_policy": {"allowed": False},
        },
    )
    catalog = _load(tmp_path)

    assert catalog["demo"].fork_allowed is True
    # fork_allowed is kernel-owned — it must NOT leak into extra_front_matter.
    assert "fork_allowed" not in catalog["demo"].extra_front_matter


def test_load_catalog_from_disk_fork_allowed_defaults_false_when_absent(
    tmp_path: Path,
) -> None:
    """A mode with no ``fork_allowed`` front matter gets ``False``.

    ``False`` means the routing fork decision and the fork enforcer refuse
    to fork to this mode; it is reachable only via delegate or single-handoff.
    """
    _write_mode(tmp_path)
    catalog = _load(tmp_path)

    assert catalog["demo"].fork_allowed is False


def test_load_catalog_from_disk_rejects_non_bool_fork_allowed(
    tmp_path: Path,
) -> None:
    """A non-bool ``fork_allowed`` is a catalog validation error."""
    _write_mode(tmp_path, extra_front_matter={"fork_allowed": "not-a-bool"})
    with pytest.raises(WorkflowValidationError, match="fork_allowed"):
        _load(tmp_path)


def test_load_catalog_from_disk_rejects_fork_allowed_no_action_tools(
    tmp_path: Path,
) -> None:
    """``fork_allowed: true`` with no authored action tools is rejected."""
    _write_mode(
        tmp_path,
        extra_front_matter={
            "fork_allowed": True,
            "satisfies_intents": ["write"],
            "yield_policy": {"allowed": False},
        },
    )
    # load_catalog_from_disk returns an unvalidated catalog by contract
    # ("Call validate_workflow_catalog() on the result before use"); the
    # fork_allowed invariants live in the validation step.
    with pytest.raises(WorkflowValidationError, match="no authored action tools"):
        validate_workflow_catalog(_load(tmp_path))


def test_load_catalog_from_disk_rejects_fork_allowed_empty_satisfies_intents(
    tmp_path: Path,
) -> None:
    """``fork_allowed: true`` with empty ``satisfies_intents`` is rejected."""
    _write_mode(
        tmp_path,
        extra_front_matter={
            "fork_allowed": True,
            "allowed_tool_ids": ["write_file"],
            "yield_policy": {"allowed": False},
        },
    )
    with pytest.raises(WorkflowValidationError, match="empty satisfies_intents"):
        validate_workflow_catalog(_load(tmp_path))


def test_load_catalog_from_disk_rejects_fork_allowed_yield_policy_contradiction(
    tmp_path: Path,
) -> None:
    """``fork_allowed: true`` with ``yield_policy.allowed: true`` is rejected."""
    _write_mode(
        tmp_path,
        extra_front_matter={
            "fork_allowed": True,
            "satisfies_intents": ["write"],
            "allowed_tool_ids": ["write_file"],
            "yield_policy": {"allowed": True},
        },
    )
    with pytest.raises(WorkflowValidationError, match="yield_policy allows inline asks"):
        validate_workflow_catalog(_load(tmp_path))


# ---------------------------------------------------------------------------
# load_catalog_from_disk: satisfies_intents
# ---------------------------------------------------------------------------


def test_load_catalog_from_disk_reads_satisfies_intents(tmp_path: Path) -> None:
    """``satisfies_intents`` flows from front matter into the typed field.

    Symmetric counterpart to ``routing_intent_tags``: this field declares the
    intents a mode handles in-place as the *current* mode, so the proactive
    routing hint generator knows not to suggest a handoff for them. Authored
    by packs; the kernel never hardcodes mode-specific intent semantics.
    """
    _write_mode(tmp_path, extra_front_matter={"satisfies_intents": ["write", "remember"]})
    catalog = _load(tmp_path)

    assert catalog["demo"].satisfies_intents == ["write", "remember"]


def test_load_catalog_from_disk_satisfies_intents_default_empty(tmp_path: Path) -> None:
    """``satisfies_intents`` defaults to empty when absent from front matter.

    Empty means every detected intent is a gap → a handoff hint is offered.
    This is the safe direction for modes that have not opted in (e.g.
    ``default``), preserving the pre-field behavior for them.
    """
    _write_mode(tmp_path)
    catalog = _load(tmp_path)

    assert catalog["demo"].satisfies_intents == []


def test_load_catalog_from_disk_satisfies_intents_not_in_extra_front_matter(
    tmp_path: Path,
) -> None:
    """``satisfies_intents`` is kernel-owned: it never leaks into ``extra_front_matter``."""
    _write_mode(tmp_path, extra_front_matter={"satisfies_intents": ["write"]})
    catalog = _load(tmp_path)

    assert "satisfies_intents" not in catalog["demo"].extra_front_matter
    assert catalog["demo"].satisfies_intents == ["write"]


def test_load_catalog_from_disk_satisfies_intents_rejects_non_list(
    tmp_path: Path,
) -> None:
    """``satisfies_intents`` must be a list; any other type is a load error."""
    _write_mode(tmp_path, extra_front_matter={"satisfies_intents": "write"})
    with pytest.raises(WorkflowValidationError, match="satisfies_intents must be a list"):
        _load(tmp_path)


def test_load_catalog_from_disk_satisfies_intents_rejects_non_string_element(
    tmp_path: Path,
) -> None:
    """Every ``satisfies_intents`` entry must be a string."""
    _write_mode(tmp_path, extra_front_matter={"satisfies_intents": ["write", 7]})
    with pytest.raises(WorkflowValidationError, match="satisfies_intents\\[1\\] must be a string"):
        _load(tmp_path)


def test_load_catalog_from_disk_satisfies_intents_rejects_empty_element(
    tmp_path: Path,
) -> None:
    """Empty/whitespace ``satisfies_intents`` entries are rejected."""
    _write_mode(tmp_path, extra_front_matter={"satisfies_intents": ["write", "  "]})
    with pytest.raises(WorkflowValidationError, match="satisfies_intents\\[1\\] cannot be empty"):
        _load(tmp_path)


def test_load_catalog_from_disk_overrides_reject_satisfies_intents(
    tmp_path: Path,
) -> None:
    """``satisfies_intents`` is a kernel-owned routing key: overrides may not touch it."""
    _write_mode(tmp_path)
    (tmp_path / "demo" / "_overrides.yaml").write_text(
        "satisfies_intents: [write]\n", encoding="utf-8"
    )
    with pytest.raises(WorkflowValidationError, match="satisfies_intents"):
        _load(tmp_path)


# ---------------------------------------------------------------------------
# load_catalog_from_disk: required_capabilities
# ---------------------------------------------------------------------------


def test_load_catalog_from_disk_reads_required_capabilities(tmp_path: Path) -> None:
    """``required_capabilities`` flows from front matter into the typed field.

    Declares the ``ModelCapability`` strings the resolved model must have for
    the mode to run safely. The turn runner validates the resolved model's
    ``ManifestModelEntry.capabilities`` against this list before the tool
    loop starts. Authored by packs; the kernel never hardcodes capability
    names.
    """
    _write_mode(tmp_path, extra_front_matter={"required_capabilities": ["tools", "code"]})
    catalog = _load(tmp_path)

    assert catalog["demo"].required_capabilities == ["tools", "code"]


def test_load_catalog_from_disk_required_capabilities_default_empty(tmp_path: Path) -> None:
    """``required_capabilities`` defaults to empty when absent from front matter.

    Empty means no capability gate — every model is admitted (backward
    compatible with the pre-field behavior).
    """
    _write_mode(tmp_path)
    catalog = _load(tmp_path)

    assert catalog["demo"].required_capabilities == []


def test_load_catalog_from_disk_required_capabilities_not_in_extra_front_matter(
    tmp_path: Path,
) -> None:
    """``required_capabilities`` is kernel-owned: it never leaks into ``extra_front_matter``."""
    _write_mode(tmp_path, extra_front_matter={"required_capabilities": ["code"]})
    catalog = _load(tmp_path)

    assert "required_capabilities" not in catalog["demo"].extra_front_matter
    assert catalog["demo"].required_capabilities == ["code"]


def test_load_catalog_from_disk_required_capabilities_rejects_non_list(
    tmp_path: Path,
) -> None:
    """``required_capabilities`` must be a list; any other type is a load error."""
    _write_mode(tmp_path, extra_front_matter={"required_capabilities": "code"})
    with pytest.raises(WorkflowValidationError, match="required_capabilities must be a list"):
        _load(tmp_path)


def test_load_catalog_from_disk_required_capabilities_rejects_non_string_element(
    tmp_path: Path,
) -> None:
    """Every ``required_capabilities`` entry must be a string."""
    _write_mode(tmp_path, extra_front_matter={"required_capabilities": ["code", 7]})
    with pytest.raises(
        WorkflowValidationError, match="required_capabilities\\[1\\] must be a string"
    ):
        _load(tmp_path)


def test_load_catalog_from_disk_rejects_routing_intent_keywords_key_not_in_tags(
    tmp_path: Path,
) -> None:
    """A ``routing_intent_keywords`` key not declared in
    ``routing_intent_tags`` is a catalog validation error.

    Regression for the 2026-08-11 executor incident: the executor mode's
    keyword bucket was keyed ``execute:`` while its ``routing_intent_tags``
    were ``[code, write, implement]``. The detector attributed an
    ``execute`` intent from any query containing the bucket's keywords, but
    no routing target declared ``execute`` in its tags, so the gap was
    uncoverable and the hint layer fell through to ambiguity on every
    code-related query. Fail fast at catalog load so a broken pack install
    cannot reach the router.
    """
    _write_mode(
        tmp_path,
        extra_front_matter={
            "routing_intent_tags": ["code", "write", "implement"],
            "routing_intent_keywords": {
                "execute": ["implement", "code", "write code"],
            },
            "satisfies_intents": ["code", "write", "implement"],
        },
    )
    with pytest.raises(
        WorkflowValidationError,
        match="routing_intent_keywords keys must be a subset of routing_intent_tags",
    ):
        _load(tmp_path)


def test_load_catalog_from_disk_accepts_routing_intent_keywords_key_in_tags(
    tmp_path: Path,
) -> None:
    """A ``routing_intent_keywords`` key that IS in ``routing_intent_tags``
    loads cleanly — the guard rejects only orphan keys, not valid ones."""
    _write_mode(
        tmp_path,
        extra_front_matter={
            "routing_intent_tags": ["code", "write", "implement"],
            "routing_intent_keywords": {
                "implement": ["implement", "build"],
                "code": ["write code", "refactor"],
            },
            "satisfies_intents": ["code", "write", "implement"],
        },
    )
    catalog = _load(tmp_path)
    assert set(catalog["demo"].routing_intent_keywords) == {"implement", "code"}


def test_load_catalog_from_disk_rejects_routing_intent_keywords_without_tags(
    tmp_path: Path,
) -> None:
    """``routing_intent_keywords`` without any ``routing_intent_tags`` is
    rejected — every keyword bucket key must have a declared covering tag."""
    _write_mode(
        tmp_path,
        extra_front_matter={
            "routing_intent_keywords": {"research": ["search", "investigate"]},
        },
    )
    with pytest.raises(
        WorkflowValidationError,
        match="routing_intent_keywords cannot be set without routing_intent_tags",
    ):
        _load(tmp_path)


def test_load_catalog_from_disk_required_capabilities_rejects_empty_element(
    tmp_path: Path,
) -> None:
    """Empty/whitespace ``required_capabilities`` entries are rejected."""
    _write_mode(tmp_path, extra_front_matter={"required_capabilities": ["code", "  "]})
    with pytest.raises(
        WorkflowValidationError, match="required_capabilities\\[1\\] cannot be empty"
    ):
        _load(tmp_path)


def test_load_catalog_from_disk_overrides_reject_required_capabilities(
    tmp_path: Path,
) -> None:
    """``required_capabilities`` is a kernel-owned routing key: overrides may not touch it."""
    _write_mode(tmp_path)
    (tmp_path / "demo" / "_overrides.yaml").write_text(
        "required_capabilities: [code]\n", encoding="utf-8"
    )
    with pytest.raises(WorkflowValidationError, match="required_capabilities"):
        _load(tmp_path)


# ---------------------------------------------------------------------------
# load_catalog_from_disk: tools_authoritative / allowed_tool_ids
# ---------------------------------------------------------------------------


def test_load_catalog_from_disk_tools_authoritative_defaults_true(tmp_path: Path) -> None:
    """tools_authoritative defaults to True when absent from front matter."""
    _write_mode(tmp_path)
    catalog = _load(tmp_path)

    assert catalog["demo"].tools_authoritative is True


def test_load_catalog_from_disk_reads_explicit_tools_authoritative_false(
    tmp_path: Path,
) -> None:
    """An explicit tools_authoritative: false is honored."""
    _write_mode(tmp_path, extra_front_matter={"tools_authoritative": False})
    catalog = _load(tmp_path)

    assert catalog["demo"].tools_authoritative is False


def test_load_catalog_from_disk_reads_allowed_tool_ids(tmp_path: Path) -> None:
    """allowed_tool_ids are parsed from front matter into the typed field."""
    _write_mode(
        tmp_path,
        extra_front_matter={"allowed_tool_ids": ["memory.add", "memory.search"]},
    )
    catalog = _load(tmp_path)

    assert catalog["demo"].allowed_tool_ids == ["memory.add", "memory.search"]


def test_load_catalog_from_disk_tools_authoritative_not_in_extra_front_matter(
    tmp_path: Path,
) -> None:
    """tools_authoritative is kernel-owned; it never leaks into extra_front_matter."""
    _write_mode(tmp_path, extra_front_matter={"tools_authoritative": True})
    catalog = _load(tmp_path)

    assert "tools_authoritative" not in catalog["demo"].extra_front_matter
    assert catalog["demo"].tools_authoritative is True


# ---------------------------------------------------------------------------
# load_catalog_from_disk: accept_default_capabilities / excluded_*_ids
# ---------------------------------------------------------------------------


def test_load_catalog_from_disk_accept_default_capabilities_defaults_true(
    tmp_path: Path,
) -> None:
    """accept_default_capabilities defaults to True when absent from front matter."""
    _write_mode(tmp_path)
    catalog = _load(tmp_path)

    assert catalog["demo"].accept_default_capabilities is True


def test_load_catalog_from_disk_reads_explicit_accept_default_capabilities_false(
    tmp_path: Path,
) -> None:
    """An explicit accept_default_capabilities: false is honored."""
    _write_mode(
        tmp_path,
        extra_front_matter={"accept_default_capabilities": False},
    )
    catalog = _load(tmp_path)

    assert catalog["demo"].accept_default_capabilities is False


def test_load_catalog_from_disk_reads_excluded_operation_ids(tmp_path: Path) -> None:
    """excluded_operation_ids are parsed from front matter into the typed field."""
    _write_mode(
        tmp_path,
        extra_front_matter={"excluded_operation_ids": ["recall_memory"]},
    )
    catalog = _load(tmp_path)

    assert catalog["demo"].excluded_operation_ids == ["recall_memory"]


def test_load_catalog_from_disk_reads_excluded_tool_ids(tmp_path: Path) -> None:
    """excluded_tool_ids are parsed from front matter into the typed field."""
    _write_mode(
        tmp_path,
        extra_front_matter={"excluded_tool_ids": ["web_search"]},
    )
    catalog = _load(tmp_path)

    assert catalog["demo"].excluded_tool_ids == ["web_search"]


def test_load_catalog_from_disk_accept_default_capabilities_not_in_extra_front_matter(
    tmp_path: Path,
) -> None:
    """accept_default_capabilities is kernel-owned; it never leaks into extra_front_matter."""
    _write_mode(
        tmp_path,
        extra_front_matter={"accept_default_capabilities": False},
    )
    catalog = _load(tmp_path)

    assert "accept_default_capabilities" not in catalog["demo"].extra_front_matter


# ---------------------------------------------------------------------------
# load_catalog_from_disk: _overrides.yaml application
# ---------------------------------------------------------------------------


def test_load_catalog_from_disk_overrides_description(tmp_path: Path) -> None:
    """``description`` is kernel-owned but NOT protected — overrides may replace it."""
    mode_dir = _write_mode(tmp_path, description="authored")
    _write_overrides(mode_dir, {"description": "overridden"})

    catalog = _load(tmp_path)
    assert catalog["demo"].description == "overridden"
    assert "description" not in catalog["demo"].extra_front_matter


def test_load_catalog_from_disk_applies_overrides_for_lenses(tmp_path: Path) -> None:
    """An override adds a ``lenses`` key that then appears in ``extra_front_matter``."""
    mode_dir = _write_mode(tmp_path)
    _write_overrides(mode_dir, {"lenses": ["lens.reflection"]})

    catalog = _load(tmp_path)
    assert catalog["demo"].extra_front_matter == {"lenses": ["lens.reflection"]}


def test_load_catalog_from_disk_overrides_can_add_lenses_when_absent(tmp_path: Path) -> None:
    """Overrides can introduce a brand-new app-owned ``lenses`` key."""
    mode_dir = _write_mode(tmp_path)
    _write_overrides(mode_dir, {"lenses": ["lens.conversation", "lens.reflection"]})

    catalog = _load(tmp_path)
    assert catalog["demo"].extra_front_matter == {
        "lenses": ["lens.conversation", "lens.reflection"],
    }


def test_load_catalog_from_disk_overrides_can_add_app_owned_key_when_absent(
    tmp_path: Path,
) -> None:
    """Overrides can introduce any app-owned key the kernel does not interpret."""
    mode_dir = _write_mode(tmp_path)
    _write_overrides(mode_dir, {"display_name": "Overridden Display"})

    catalog = _load(tmp_path)
    assert catalog["demo"].extra_front_matter == {"display_name": "Overridden Display"}


def test_load_catalog_from_disk_overrides_flow_into_extra_front_matter(tmp_path: Path) -> None:
    """Override-added app-owned keys flow through to ``extra_front_matter`` opaquely."""
    mode_dir = _write_mode(tmp_path, extra_front_matter={"lenses": ["lens.conversation"]})
    _write_overrides(mode_dir, {"display_name": "Demo", "custom": {"nested": True}})

    catalog = _load(tmp_path)
    extra = catalog["demo"].extra_front_matter
    assert extra["lenses"] == ["lens.conversation"]
    assert extra["display_name"] == "Demo"
    assert extra["custom"] == {"nested": True}


def test_load_catalog_from_disk_overrides_rejects_protected_kernel_key(
    tmp_path: Path,
) -> None:
    """Overriding a kernel-owned protected key (e.g. ``mode_id``) is rejected."""
    mode_dir = _write_mode(tmp_path)
    _write_overrides(mode_dir, {"mode_id": "hijacked"})

    with pytest.raises(WorkflowValidationError, match="kernel-owned keys"):
        _load(tmp_path)


def test_load_catalog_from_disk_overrides_rejects_disallowed_key(tmp_path: Path) -> None:
    """Overriding any protected kernel key — here ``command_pattern`` — is rejected."""
    mode_dir = _write_mode(tmp_path)
    _write_overrides(mode_dir, {"command_pattern": "/hijacked"})

    with pytest.raises(WorkflowValidationError, match="kernel-owned keys"):
        _load(tmp_path)

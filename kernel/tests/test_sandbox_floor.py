"""Exclusive-merge and owner-selection tests for kernel.sandbox_floor.

Pure logic — no entry points, no I/O. The runtime tests cover discovery and
binding materialization; these cover the policy join itself.
"""

from __future__ import annotations

import pytest
from kernel.sandbox_floor import (
    SandboxFloorConflict,
    SandboxFloorContext,
    SandboxFloorContributionEntry,
    SandboxImplementation,
    collect_pack_floor,
    merge_sandbox_floor,
    resolve_sandbox_owner,
)


def _impl(pack_name: str = "lisa-pack-bwrap") -> SandboxImplementation:
    def _validate(floor: object) -> None:
        if not isinstance(floor, dict):
            raise TypeError("not a dict")

    return SandboxImplementation(pack_name=pack_name, validate_floor=_validate)


# --- Exclusive join: scalars -------------------------------------------------


def test_single_contribution_merges_onto_authored_floor() -> None:
    authored = {"execution": "sandbox", "persistence": "direct-write"}
    contributions = {"lisa-pack-coding": {"paths": {"/wt": "rw"}, "network": "allow"}}
    merge = merge_sandbox_floor(authored, contributions)
    assert merge.floor == {
        "execution": "sandbox",
        "persistence": "direct-write",
        "network": "allow",
        "paths": {"/wt": "rw"},
    }
    assert merge.contributing_packs == ("lisa-pack-coding",)


def test_scalar_conflict_between_packs_raises() -> None:
    contributions = {
        "lisa-pack-coding": {"network": "allow"},
        "lisa-pack-finances": {"network": "deny"},
    }
    with pytest.raises(SandboxFloorConflict) as excinfo:
        merge_sandbox_floor({}, contributions)
    assert "network" in str(excinfo.value)
    assert "coding" in str(excinfo.value) and "finances" in str(excinfo.value)


def test_scalar_agreement_between_packs_is_inert() -> None:
    contributions = {
        "lisa-pack-coding": {"network": "allow"},
        "lisa-pack-finances": {"network": "allow"},
    }
    merge = merge_sandbox_floor({}, contributions)
    assert merge.floor["network"] == "allow"


def test_contribution_matching_authored_scalar_is_inert() -> None:
    authored = {"network": "allow"}
    contributions = {"lisa-pack-coding": {"network": "allow"}}
    merge = merge_sandbox_floor(authored, contributions)
    assert merge.floor["network"] == "allow"
    assert merge.contributing_packs == ("lisa-pack-coding",)


def test_contribution_conflicting_with_authored_floor_raises() -> None:
    authored = {"network": "deny"}
    contributions = {"lisa-pack-coding": {"network": "allow"}}
    with pytest.raises(SandboxFloorConflict) as excinfo:
        merge_sandbox_floor(authored, contributions)
    message = str(excinfo.value)
    assert "coding" in message and "harness" in message


def test_contribution_adding_to_authored_scalar_is_permitted() -> None:
    authored = {"execution": "sandbox"}
    contributions = {"lisa-pack-coding": {"persistence": "direct-write", "paths": {"/wt": "rw"}}}
    merge = merge_sandbox_floor(authored, contributions)
    assert merge.floor["persistence"] == "direct-write"
    assert merge.floor["paths"] == {"/wt": "rw"}


def test_budget_conflict_between_packs_raises() -> None:
    contributions = {
        "lisa-pack-coding": {"tmpfs_budget_bytes": 67108864},
        "lisa-pack-finances": {"tmpfs_budget_bytes": 134217728},
    }
    with pytest.raises(SandboxFloorConflict):
        merge_sandbox_floor({}, contributions)


# --- Exclusive join: paths ---------------------------------------------------


def test_disjoint_paths_union() -> None:
    contributions = {
        "lisa-pack-coding": {"paths": {"/wt": "rw"}},
        "lisa-pack-finances": {"paths": {"/fin": "capture"}},
    }
    merge = merge_sandbox_floor({}, contributions)
    assert merge.floor["paths"] == {"/wt": "rw", "/fin": "capture"}


def test_same_path_same_access_is_compatible_overlap() -> None:
    contributions = {
        "lisa-pack-coding": {"paths": {"/wt": "rw"}},
        "lisa-pack-finances": {"paths": {"/wt": "rw"}},
    }
    merge = merge_sandbox_floor({}, contributions)
    assert merge.floor["paths"] == {"/wt": "rw"}


def test_same_path_different_access_raises() -> None:
    contributions = {
        "lisa-pack-coding": {"paths": {"/shared": "rw"}},
        "lisa-pack-finances": {"paths": {"/shared": "capture"}},
    }
    with pytest.raises(SandboxFloorConflict) as excinfo:
        merge_sandbox_floor({}, contributions)
    assert "/shared" in str(excinfo.value)


def test_nested_paths_different_access_raises() -> None:
    contributions = {
        "lisa-pack-coding": {"paths": {"/data": "rw"}},
        "lisa-pack-finances": {"paths": {"/data/fin": "capture"}},
    }
    with pytest.raises(SandboxFloorConflict) as excinfo:
        merge_sandbox_floor({}, contributions)
    assert "nested" in str(excinfo.value)


def test_nested_paths_same_access_is_fine() -> None:
    contributions = {
        "lisa-pack-coding": {"paths": {"/data": "rw"}},
        "lisa-pack-finances": {"paths": {"/data/fin": "rw"}},
    }
    merge = merge_sandbox_floor({}, contributions)
    assert merge.floor["paths"] == {"/data": "rw", "/data/fin": "rw"}


def test_path_agreement_with_authored_floor_is_inert() -> None:
    authored = {"paths": {"/wt": "rw"}}
    contributions = {"lisa-pack-coding": {"paths": {"/wt": "rw", "/uv": "rw"}}}
    merge = merge_sandbox_floor(authored, contributions)
    assert merge.floor["paths"] == {"/wt": "rw", "/uv": "rw"}


def test_path_conflict_with_authored_floor_raises() -> None:
    authored = {"paths": {"/shared": "ro"}}
    contributions = {"lisa-pack-coding": {"paths": {"/shared": "rw"}}}
    with pytest.raises(SandboxFloorConflict) as excinfo:
        merge_sandbox_floor(authored, contributions)
    assert "/shared" in str(excinfo.value) and "harness" in str(excinfo.value)


def test_paths_normalize_tilde_so_agreement_compares_real_paths() -> None:
    contributions = {
        "lisa-pack-coding": {"paths": {"~/.cache/uv": "rw"}},
        "lisa-pack-finances": {"paths": {"/home/adam/.cache/uv": "rw"}},
    }
    merge = merge_sandbox_floor({}, contributions)
    assert len(merge.floor["paths"]) == 1


def test_unknown_floor_key_in_contribution_raises() -> None:
    contributions = {"lisa-pack-coding": {"network": "deny", "cool_factor": 11}}
    with pytest.raises(SandboxFloorConflict) as excinfo:
        merge_sandbox_floor({}, contributions)
    assert "cool_factor" in str(excinfo.value)


def test_paths_scalar_instead_of_mapping_raises() -> None:
    contributions = {"lisa-pack-coding": {"paths": "/some/dir"}}
    with pytest.raises(ValueError, match="mapping"):
        merge_sandbox_floor({}, contributions)


# --- Authored-floor requirement ------------------------------------------------


def test_contributions_without_authored_floor_raise() -> None:
    contributions = {"lisa-pack-coding": {"paths": {"/wt": "rw"}}}
    with pytest.raises(SandboxFloorConflict) as excinfo:
        merge_sandbox_floor(None, contributions)
    assert "no floor authored" in str(excinfo.value)


def test_empty_contributions_with_no_authored_floor_are_fine() -> None:
    merge = merge_sandbox_floor(None, {"lisa-pack-coding": {}})
    assert merge.floor == {}
    assert merge.contributing_packs == ()


def test_contributions_attach_to_empty_authored_floor() -> None:
    contributions = {"lisa-pack-coding": {"paths": {"/wt": "rw"}}}
    merge = merge_sandbox_floor({}, contributions)
    assert merge.floor["paths"] == {"/wt": "rw"}


def test_non_mapping_authored_floor_raises() -> None:
    contributions = {"lisa-pack-coding": {"paths": {"/wt": "rw"}}}
    with pytest.raises(SandboxFloorConflict):
        merge_sandbox_floor("sandbox: direct-write", contributions)  # type: ignore[arg-type]


# --- Validator ----------------------------------------------------------------


def test_validator_receives_merged_floor_and_rejections_propagate() -> None:
    def _reject(floor: object) -> None:
        raise ValueError(f"cannot enforce {sorted(floor)}")

    contributions = {"lisa-pack-coding": {"paths": {"/wt": "rw"}}}
    with pytest.raises(ValueError, match="cannot enforce"):
        merge_sandbox_floor({}, contributions, validator=_reject)


def test_validator_not_invoked_when_no_contributions() -> None:
    calls: list[object] = []

    def _spy(floor: object) -> None:
        calls.append(floor)

    merge = merge_sandbox_floor({"execution": "sandbox"}, None, validator=_spy)
    assert merge.floor == {"execution": "sandbox"}
    assert calls == []


# --- Owner selection -----------------------------------------------------------


def test_no_implementations_yields_none() -> None:
    assert resolve_sandbox_owner([], "lisa-pack-bwrap") is None


def test_single_implementation_auto_owns() -> None:
    assert resolve_sandbox_owner([_impl()], "") is not None


def test_explicit_selection_wins() -> None:
    bwrap = _impl("lisa-pack-bwrap")
    win = _impl("lisa-pack-windows-sandbox")
    assert resolve_sandbox_owner([bwrap, win], "bwrap") is bwrap
    assert resolve_sandbox_owner([bwrap, win], "lisa-pack-windows-sandbox") is win


def test_unknown_selection_yields_none() -> None:
    assert resolve_sandbox_owner([_impl()], "lisa-pack-nope") is None


def test_multiple_implementations_without_selection_raise() -> None:
    with pytest.raises(ValueError, match="explicit"):
        resolve_sandbox_owner([_impl("lisa-pack-bwrap"), _impl("lisa-pack-other")], "")


def test_ambiguous_selection_raises() -> None:
    with pytest.raises(ValueError, match="Multiple"):
        resolve_sandbox_owner([_impl("lisa-pack-bwrap"), _impl("lisa-pack-bwrap")], "bwrap")


# --- Factory collection --------------------------------------------------------


def test_collect_pack_floor_invokes_factory_with_pack_config_and_context() -> None:
    seen: list[tuple[object, object]] = []

    def factory(config: object, context: object) -> dict[str, dict[str, str]]:
        seen.append((config, context))
        return {"paths": {"/wt": "rw"}}

    entries = [SandboxFloorContributionEntry(pack_name="lisa-pack-coding", factory=factory)]
    configs = {"lisa-pack-coding": {"worktree_dir": "/wt"}}
    context = SandboxFloorContext(data_dir="/data", workspace_root="/ws")
    floor = collect_pack_floor(entries, configs, context)
    assert floor == {"lisa-pack-coding": {"paths": {"/wt": "rw"}}}
    assert seen == [({"worktree_dir": "/wt"}, context)]


def test_context_supports_mapping_style_reads_for_pack_factories() -> None:
    context = SandboxFloorContext(data_dir="/data", workspace_root="/ws")
    assert context.get("data_dir") == "/data"
    assert context.get("workspace_root") == "/ws"
    assert context.get("unknown", "fallback") == "fallback"


def test_collect_pack_floor_skips_packs_without_session_configuration() -> None:
    def factory(config: object, context: object) -> dict[str, dict[str, str]]:
        raise AssertionError("must not be called")

    entries = [SandboxFloorContributionEntry(pack_name="lisa-pack-coding", factory=factory)]
    assert collect_pack_floor(entries, {}, SandboxFloorContext()) == {}


def test_collect_pack_floor_opt_out_none_is_skipped() -> None:
    def factory(config: object, context: object) -> None:
        return None

    entries = [SandboxFloorContributionEntry(pack_name="lisa-pack-coding", factory=factory)]
    assert collect_pack_floor(entries, {"lisa-pack-coding": {}}, SandboxFloorContext()) == {}


def test_collect_pack_floor_raises_on_non_mapping_result() -> None:
    def factory(config: object, context: object) -> str:
        return "nope"

    entries = [SandboxFloorContributionEntry(pack_name="lisa-pack-coding", factory=factory)]
    with pytest.raises(ValueError, match="expected a mapping"):
        collect_pack_floor(entries, {"lisa-pack-coding": {}}, SandboxFloorContext())

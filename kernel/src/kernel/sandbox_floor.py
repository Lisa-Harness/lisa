"""Discovery + exclusive merge for pack-contributed sandbox floors.

The sandbox capability follows the memory-capability shape: the kernel owns
the *concept* (a named capability with a declarative floor vocabulary), packs
own the *implementation* (``lisa-pack-bwrap`` today, a Windows sandbox pack
tomorrow) and may contribute *floor needs* — paths and postures their tools
require to run contained.

Two entry-point groups cooperate:

- ``lisa.extensions.sandbox`` — one per sandbox *implementation* pack. The
  entry returns a :class:`SandboxImplementation` describing which floor keys
  the implementation understands (``validate_floor`` must accept any mapping
  the implementation can enforce and raise on what it cannot).
- ``lisa.extensions.sandbox_floors`` — any number of *consumer* packs. Each
  entry returns a :class:`SandboxFloorContribution` factory
  ``(pack_config, context) -> mapping | None`` where the mapping is a subset
  of the floor vocabulary (``paths`` union; scalars must agree — see below).

Merge semantics are **exclusive**, per harness policy: two packs may only
co-author the floor by agreeing. A scalar axis named by two contributions
with different values is a :class:`SandboxFloorConflict` at load, never a
silent lattice join; ``paths`` union with per-path agreement, so overlapping
grants with different access (``rw`` vs ``capture``) also conflict. The
harness-authored floor (session ``resources.packs.configuration`` →
``sandbox.floor``) stays authoritative: a contribution needing *more* than
the authored floor is a conflict naming the pack and the axis; a contribution
already satisfied by the authored floor is inert.

Contributions require an explicitly authored floor to attach to (even an
empty ``floor: {}``): the floor is the session's security posture, so every
effective floor must start from something the harness wrote. Contributions
with no authored floor anywhere raise at load, teaching the boundary loudly.

The merged floor is materialized into the owning implementation pack's
binding configuration *before* any reader resolves it (the implementation's
own ``floor_from_config``-style reader and the kernel's
``project_execution_floor`` both see the same final mapping), so there is
exactly one floor, never two divergent views.
"""

from __future__ import annotations

import importlib.metadata
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from kernel.extension_names import normalize_pack_name

logger = logging.getLogger(__name__)

SANDBOX_ENTRY_POINT_GROUP = "lisa.extensions.sandbox"
SANDBOX_FLOORS_ENTRY_POINT_GROUP = "lisa.extensions.sandbox_floors"

#: Floor keys understood by the merge. The implementation's validator is
#: authoritative for *values*; this set guards the merge itself so unknown
#: keys surface at load, not at first dispatch.
FLOOR_KEYS: frozenset[str] = frozenset(
    {
        "execution",
        "persistence",
        "network",
        "device",
        "ipc",
        "paths",
        "tmpfs_budget_bytes",
        "timeout_ceiling_seconds",
    }
)

#: Scalar (non-path) floor axes; exclusivity means identical values only.
SCALAR_FLOOR_KEYS: frozenset[str] = FLOOR_KEYS - {"paths"}


class SandboxFloorConflict(ValueError):
    """Two floor authors disagree, or a contribution exceeds the authored floor.

    The exclusive-join rule: packs co-author the sandbox floor only by
    agreeing. The error names the two authors (or pack vs. harness), the
    axis, and both values, so the operator can resolve the session policy
    explicitly instead of discovering a silent lattice join mid-turn.
    """

    def __init__(self, axis: str, left: str, left_value: Any, right: str, right_value: Any) -> None:
        self.axis = axis
        self.left = left
        self.left_value = left_value
        self.right = right
        self.right_value = right_value
        super().__init__(
            f"Refused: sandbox floor '{axis}' is contested. {left} requires "
            f"{left_value!r} but {right} requires {right_value!r}. Packs may "
            "only co-author the sandbox floor by agreeing on every axis they "
            "both name; resolve the conflict explicitly (remove one pack, or "
            "align the contributions/floor in the session policy)."
        )


@dataclass(frozen=True, slots=True)
class SandboxFloorContext:
    """Host facts a contribution factory may need to compute dynamic paths.

    ``data_dir`` is the host-resolved canonical data root — the one fact a
    pack needs when its writable locations derive from the data dir (e.g.
    the coding pack's worktree parent). ``workspace_root`` is the session's
    source workspace when one is active. ``get`` mirrors mapping access so
    a factory can read either style (attribute or key) without importing
    kernel internals.
    """

    data_dir: str = ""
    workspace_root: str = ""

    def get(self, key: str, default: Any = None) -> Any:
        """Mapping-style read so factories work without kernel imports."""
        return getattr(self, key, default)


@runtime_checkable
class SandboxFloorContributionFactory(Protocol):
    """``(pack_config, context) -> floor-shaped mapping | None``.

    The context is a :class:`SandboxFloorContext`; factories read it as
    attributes (``context.data_dir``) or mapping-style
    (``context.get("data_dir")``).
    """

    def __call__(
        self,
        pack_config: Mapping[str, Any],
        context: SandboxFloorContext,
    ) -> Mapping[str, Any] | None: ...


@dataclass(frozen=True, slots=True)
class SandboxFloorContributionEntry:
    """One discovered contribution factory attributed to its source pack."""

    pack_name: str
    factory: SandboxFloorContributionFactory


@runtime_checkable
class SandboxFloorValidator(Protocol):
    """Implementation-owned final check on the merged floor mapping.

    The implementation (e.g. bwrap's ``floor_from_config`` semantics) is
    the single loud validator for values: unknown keys, invalid axis
    values, and cross-axis constraints (capture paths require the overlay
    persistence posture). Raise ``ValueError`` with an actionable message.
    """

    def __call__(self, floor: Mapping[str, Any]) -> None: ...


@dataclass(frozen=True, slots=True)
class SandboxImplementation:
    """One sandbox implementation pack's declaration.

    ``pack_name`` is the normalized distribution name (e.g.
    ``lisa-pack-bwrap``); ``validate_floor`` receives the final merged
    mapping and raises ``ValueError`` when the implementation cannot
    enforce it (unknown keys, invalid values, inconsistent combination).
    """

    pack_name: str
    validate_floor: SandboxFloorValidator


@dataclass(frozen=True, slots=True)
class SandboxFloorMerge:
    """Result of merging contributions onto the authored floor.

    ``floor`` is the final mapping to materialize into the implementation
    pack's binding configuration (under ``sandbox.floor``). ``contributing
    _packs`` names the packs whose contribution changed the authored floor
    (for load-time logging and startup surfaces).
    """

    floor: Mapping[str, Any] = field(default_factory=dict)
    contributing_packs: tuple[str, ...] = ()


def _entry_point_pack_name(ep: importlib.metadata.EntryPoint) -> str:
    """Return the distribution package name for an entry point, or empty string."""
    dist = getattr(ep, "dist", None)
    if dist is None:
        return ""
    name = dist.metadata.get("Name", "") if dist.metadata else ""
    return normalize_pack_name(str(name or ""))


def _merge_paths_axis(
    merged: dict[str, Any],
    path_attribution: dict[str, str],
    pack: str,
    paths: Mapping[str, Any],
) -> None:
    """Union one contribution's paths into ``merged``; disagreement raises.

    A path named by an earlier contribution and this one must carry the
    same access value. Nested paths with *different* access also conflict:
    nearest-match resolution would silently break one author's guarantee,
    so the operator resolves it explicitly at load.
    """
    for path, access in paths.items():
        existing = merged.get(path)
        if existing is not None and existing != access:
            raise SandboxFloorConflict(
                f"paths[{path}]", path_attribution[path], existing, pack, access
            )
        merged[path] = access
        path_attribution[path] = pack
    for existing_path, existing_access in merged.items():
        for path, access in paths.items():
            if existing_path == path or existing_access == access:
                continue
            nested = Path(existing_path).is_relative_to(Path(path)) or Path(path).is_relative_to(
                Path(existing_path)
            )
            if nested:
                raise SandboxFloorConflict(
                    f"paths[nested {existing_path} vs {path}]",
                    path_attribution[existing_path],
                    existing_access,
                    pack,
                    access,
                )


def _merge_two_floors(
    authored: Mapping[str, Any],
    contributions: Mapping[str, Any],
) -> dict[str, Any]:
    """Union the authored floor with the joined contributions.

    Conflicts were already rejected per-pack in ``_check_against_authored``,
    so this merge is a plain union: contributions add axes the authored
    floor leaves unset; both stand where they agree.
    """
    out = dict(authored)
    for key, value in contributions.items():
        if key == "paths" and isinstance(out.get("paths"), Mapping) and isinstance(value, Mapping):
            out["paths"] = {**out["paths"], **value}
        else:
            out[key] = value
    return out


def _merge_contribution(
    merged: dict[str, Any],
    attribution: dict[str, str],
    path_attribution: dict[str, str],
    pack: str,
    contribution: Mapping[str, Any],
) -> None:
    """Merge one contribution into the joined needs, tracking attribution.

    Per-key attribution lets a conflict name the two *disagreeing* packs,
    not the current pack twice.
    """
    for key in sorted(contribution):
        value = contribution[key]
        if key == "paths":
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"Contribution from {pack}: sandbox floor 'paths' must be a "
                    "mapping of path -> access, not a scalar. See bwrap floor authoring."
                )
            if "paths" not in merged:
                merged["paths"] = {}
            _merge_paths_axis(merged["paths"], path_attribution, pack, value)
            continue
        existing = merged.get(key)
        if existing is not None and existing != value:
            raise SandboxFloorConflict(key, attribution[key], existing, pack, value)
        merged[key] = value
        attribution[key] = pack


def _validate_floor_shape(pack_name: str, floor: Mapping[str, Any]) -> None:
    """Reject keys the merge vocabulary does not know, naming the pack."""
    unknown = set(floor) - FLOOR_KEYS
    if unknown:
        raise SandboxFloorConflict(
            "sandbox.floor",
            pack_name,
            ",".join(sorted(unknown)),
            "the floor vocabulary",
            ",".join(sorted(FLOOR_KEYS)),
        )


def _normalize_paths(pack_name: str, floor: Mapping[str, Any]) -> dict[str, Any]:
    """Expand users and normalize separators so agreement compares real paths."""
    out = dict(floor)
    raw_paths = floor.get("paths")
    if raw_paths is None:
        return out
    if not isinstance(raw_paths, Mapping):
        raise ValueError(
            f"Contribution from {pack_name}: sandbox floor 'paths' must be a "
            "mapping of path -> access, not a scalar. See bwrap floor authoring."
        )
    out["paths"] = {
        Path(str(raw_path)).expanduser().as_posix(): access
        for raw_path, access in raw_paths.items()
    }
    return out


def merge_sandbox_floor(
    authored: Mapping[str, Any] | None,
    contributions: Mapping[str, Mapping[str, Any]] | None,
    *,
    validator: SandboxFloorValidator | None = None,
) -> SandboxFloorMerge:
    """Merge pack floor contributions onto the authored floor, exclusively.

    Args:
        authored: The harness-authored floor mapping (``sandbox.floor``
            from the session's pack configuration for the implementation
            owner). ``None`` means no floor was authored anywhere.
        contributions: Pack name -> floor-shaped contribution mapping. Only
            packs with a *non-empty* contribution participate.
        validator: Optional implementation-owned validator applied to the
            final merged floor; raise ``ValueError`` to reject.

    Returns:
        :class:`SandboxFloorMerge` with the final floor and the packs whose
        contributions changed it.

    Raises:
        SandboxFloorConflict: Two contributions disagree on an axis, a
            contribution exceeds the authored floor, or contributions
            exist with no authored floor to attach to.
        ValueError: The implementation validator rejects the merged floor.

    Exclusivity rules:

    - Scalar axes named by two contributions must carry identical values.
    - ``paths`` union; same path with different access conflicts, and nested
      paths with different access conflict.
    - A contribution may only *add* to the authored floor. A contribution
      naming a scalar axis with a different value than the authored floor
      conflicts (the harness floor is the security posture; a pack cannot
      overwrite it — but a pack need that the floor already satisfies is
      inert, whether or not the pack restates it).
    - Contributions require an authored floor to attach to, even an empty
      one. No authored floor + non-empty contributions = load error telling
      the operator to author ``sandbox.floor`` for the sandbox owner.
    """
    live = {
        pack: _normalize_paths(pack, contribution)
        for pack, contribution in (contributions or {}).items()
        if contribution
    }
    if not live:
        return SandboxFloorMerge(
            floor=dict(authored) if isinstance(authored, Mapping) and authored else {},
            contributing_packs=(),
        )

    if authored is None:
        raise SandboxFloorConflict(
            "sandbox.floor",
            ",".join(sorted(live)),
            "floor needs (paths/postures)",
            "the harness",
            "no floor authored",
        )
    if not isinstance(authored, Mapping):
        raise SandboxFloorConflict(
            "sandbox.floor",
            ",".join(sorted(live)),
            "floor needs",
            "the harness",
            f"{type(authored).__name__} (not a mapping)",
        )
    authored = _normalize_paths("the harness", authored)

    for pack, contribution in live.items():
        _validate_floor_shape(pack, contribution)

    # Contributions join each other first (pack-to-pack agreement), then the
    # joined needs are checked against the authored floor (pack-vs-harness).
    merged: dict[str, Any] = {}
    attribution: dict[str, str] = {}
    path_attribution: dict[str, str] = {}
    contributing: list[str] = []
    for pack in sorted(live):
        _merge_contribution(merged, attribution, path_attribution, pack, live[pack])
        contributing.append(pack)
    # The authored floor and the joined contributions both stand: neither
    # overwrites the other. A pack naming an axis the harness already
    # authored with a different value conflicts; identical restatement is
    # inert.
    for pack in contributing:
        _check_against_authored(pack, live[pack], authored)

    final = _merge_two_floors(dict(authored), merged)
    if validator is not None:
        validator(final)
    return SandboxFloorMerge(floor=final, contributing_packs=tuple(contributing))


def _check_against_authored(
    pack: str,
    contribution: Mapping[str, Any],
    authored: Mapping[str, Any],
) -> None:
    """A contribution may only add to the authored floor, never rewrite it.

    Same axis + same value is inert restatement. Same axis + different
    value is a conflict naming the pack, the axis, both values, and the
    fix (align the contribution, raise the authored floor, or drop the pack).
    """
    for key in sorted(set(contribution) & set(authored)):
        if key == "paths":
            authored_paths = authored.get("paths") or {}
            contribution_paths = contribution.get("paths") or {}
            if not isinstance(authored_paths, Mapping) or not isinstance(
                contribution_paths, Mapping
            ):
                raise SandboxFloorConflict(
                    f"{key}", pack, contribution.get(key), "the harness floor", authored.get(key)
                )
            for path, access in contribution_paths.items():
                if path in authored_paths and authored_paths[path] != access:
                    raise SandboxFloorConflict(
                        f"paths[{path}]",
                        pack,
                        access,
                        "the harness floor",
                        authored_paths[path],
                    )
            continue
        if contribution[key] != authored[key]:
            raise SandboxFloorConflict(
                key, pack, contribution[key], "the harness floor", authored[key]
            )


def discover_sandbox_implementations(
    excluded_packs: set[str] | None = None,
) -> list[SandboxImplementation]:
    """Return all registered sandbox implementation declarations.

    Loads ``lisa.extensions.sandbox`` entry points in discovery order. Each
    entry must yield a :class:`SandboxImplementation` (or a mapping with
    ``pack_name`` / ``validate_floor`` keys, accepted for test manifests).
    Broken entry points are logged and skipped — a faulty pack must not
    take the sandbox capability down with it.
    """
    implementations: list[SandboxImplementation] = []
    for ep in importlib.metadata.entry_points(group=SANDBOX_ENTRY_POINT_GROUP):
        pack_name = _entry_point_pack_name(ep)
        if excluded_packs and pack_name in excluded_packs:
            continue
        try:
            declared = ep.load()()
        except Exception:  # pylint: disable=broad-except
            logger.warning("Failed to load sandbox entry-point %r", ep.name, exc_info=True)
            continue
        if isinstance(declared, SandboxImplementation):
            implementations.append(declared)
        elif isinstance(declared, Mapping):
            try:
                implementations.append(
                    SandboxImplementation(
                        pack_name=str(declared["pack_name"] or pack_name),
                        validate_floor=declared["validate_floor"],
                    )
                )
            except (KeyError, TypeError) as exc:
                logger.warning(
                    "sandbox entry-point %r yielded an incomplete mapping (%s); skipping",
                    ep.name,
                    exc,
                )
        else:
            logger.warning(
                "sandbox entry-point %r returned %r, expected SandboxImplementation; skipping",
                ep.name,
                type(declared).__name__,
            )
    return implementations


def discover_sandbox_floor_contributions(
    excluded_packs: set[str] | None = None,
) -> list[SandboxFloorContributionEntry]:
    """Return all pack-registered floor-contribution factories, in order.

    Broken entry points are logged and skipped so one faulty pack does not
    poison sandbox-floor discovery for the rest.
    """
    entries: list[SandboxFloorContributionEntry] = []
    for ep in importlib.metadata.entry_points(group=SANDBOX_FLOORS_ENTRY_POINT_GROUP):
        pack_name = _entry_point_pack_name(ep)
        if excluded_packs and pack_name in excluded_packs:
            continue
        try:
            factory = ep.load()
        except Exception:  # pylint: disable=broad-except
            logger.warning("Failed to load sandbox_floors entry-point %r", ep.name, exc_info=True)
            continue
        if not callable(factory):
            logger.warning(
                "sandbox_floors entry-point %r did not yield a callable factory; skipping",
                ep.name,
            )
            continue
        entries.append(SandboxFloorContributionEntry(pack_name=pack_name, factory=factory))
    return entries


def resolve_sandbox_owner(
    implementations: list[SandboxImplementation],
    selected_provider: str,
) -> SandboxImplementation | None:
    """Select the sandbox implementation owner (memory-capability rules).

    An explicit session selection (``providers.capabilities.sandbox``) wins;
    a single registered implementation auto-owns; multiple implementations
    without a selection raise ``ValueError`` — the same explicit-ownership
    contract as :func:`runtime.mcp_pool.select_capability_provider_owner`.
    """
    if not implementations:
        return None
    selected = normalize_pack_name(selected_provider)
    if selected:
        matching = [
            impl for impl in implementations if selected in _provider_identifiers(impl.pack_name)
        ]
        if len(matching) == 1:
            return matching[0]
        if not matching:
            return None
        raise ValueError(
            f"Multiple active sandbox implementations match explicit selection {selected!r}."
        )
    if len(implementations) == 1:
        return implementations[0]
    names = ", ".join(sorted(impl.pack_name for impl in implementations))
    raise ValueError(
        "Multiple active sandbox implementations require an explicit "
        f"providers.capabilities.sandbox selection: {names}"
    )


def _provider_identifiers(pack_name: str) -> set[str]:
    """Normalized identifiers referring to one pack (mirrors mcp_pool)."""
    identifiers = {normalize_pack_name(pack_name)}
    normalized = normalize_pack_name(pack_name)
    if normalized.startswith("lisa-pack-"):
        identifiers.add(normalized[len("lisa-pack-") :])
    return {identifier for identifier in identifiers if identifier}


def collect_pack_floor(
    entries: list[SandboxFloorContributionEntry],
    pack_configurations: Mapping[str, Mapping[str, Any]],
    context: SandboxFloorContext,
) -> dict[str, Mapping[str, Any]]:
    """Invoke each contribution factory against its own pack configuration.

    Only factories whose pack has configuration in this session contribute
    (a pack with no session configuration has no session-scoped needs; the
    factory may still return ``None`` to opt out). A factory that raises is
    a load-time failure: floor needs are security posture, so a broken
    contribution must be loud, not skipped.
    """
    contributions: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        pack_config = pack_configurations.get(entry.pack_name)
        if pack_config is None:
            continue
        declared = entry.factory(pack_config, context)
        if declared is None:
            continue
        if not isinstance(declared, Mapping):
            raise ValueError(
                f"Sandbox floor contribution from {entry.pack_name} returned "
                f"{type(declared).__name__}, expected a mapping or None."
            )
        contributions[entry.pack_name] = dict(declared)
    return contributions


__all__ = [
    "FLOOR_KEYS",
    "SANDBOX_ENTRY_POINT_GROUP",
    "SANDBOX_FLOORS_ENTRY_POINT_GROUP",
    "SCALAR_FLOOR_KEYS",
    "SandboxFloorConflict",
    "SandboxFloorContext",
    "SandboxFloorContributionEntry",
    "SandboxFloorContributionFactory",
    "SandboxFloorMerge",
    "SandboxFloorValidator",
    "SandboxImplementation",
    "collect_pack_floor",
    "discover_sandbox_floor_contributions",
    "discover_sandbox_implementations",
    "merge_sandbox_floor",
    "resolve_sandbox_owner",
]

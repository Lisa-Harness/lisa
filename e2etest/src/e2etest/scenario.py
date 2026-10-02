"""Scenario files: one declarative end-to-end workflow per ``*.scenario.yml``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ScenarioError(ValueError):
    """Raised when a scenario file is malformed."""


@dataclass(frozen=True)
class Approvals:
    """How the driver answers approval requests, keyed by target (tool) id."""

    approve: tuple[str, ...] = ()
    deny: tuple[str, ...] = ()
    max_decisions: int = 20

    def decide(self, target_id: str) -> str:
        """Return ``approve`` or ``deny``; unlisted targets are denied."""
        if target_id in self.deny:
            return "deny"
        if target_id in self.approve or "*" in self.approve:
            return "approve"
        return "deny"


@dataclass(frozen=True)
class Budgets:
    """Hard limits for one scenario run."""

    max_messages: int = 20
    max_continues: int = 0
    max_yield_replies: int = 0
    wall_clock_s: float = 1800.0


@dataclass(frozen=True)
class OracleSpec:
    """One named assertion with its arguments, grouped under a phase."""

    phase: str
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class Scenario:
    """A parsed scenario plus the directory its relative paths resolve against."""

    scenario_id: str
    source: Path
    session_context: Path
    fixture_repo: Path | None
    fixture_data_files: Path | None
    policy: Path | None
    packs: tuple[str, ...]
    turns: tuple[str, ...]
    approvals: Approvals
    budgets: Budgets
    oracles: tuple[OracleSpec, ...]
    continue_message: str = "continue"
    yield_reply: str = "Proceed with your best judgment."
    env: dict[str, str] = field(default_factory=dict)

    @property
    def base_dir(self) -> Path:
        return self.source.parent


def _resolve(base: Path, value: object, *, key: str, required: bool) -> Path | None:
    if value in (None, ""):
        if required:
            raise ScenarioError(f"'{key}' is required")
        return None
    path = (base / str(value)).resolve()
    if not path.exists():
        raise ScenarioError(f"'{key}' does not exist: {path}")
    return path


def _oracles(raw: object) -> tuple[OracleSpec, ...]:
    if not isinstance(raw, dict):
        raise ScenarioError("'oracles' must map phase -> list of {name: args}")
    specs: list[OracleSpec] = []
    for phase, items in raw.items():
        if not isinstance(items, list):
            raise ScenarioError(f"oracles.{phase} must be a list")
        for item in items:
            if not isinstance(item, dict) or len(item) != 1:
                raise ScenarioError(f"oracles.{phase} entries must be single-key maps")
            ((name, args),) = item.items()
            specs.append(OracleSpec(phase=str(phase), name=str(name), args=dict(args or {})))
    return tuple(specs)


def load_scenario(path: Path) -> Scenario:
    """Parse and validate a scenario file."""
    source = path.resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ScenarioError(f"{source}: top level must be a mapping")
    base = source.parent
    turns = raw.get("turns")
    if not isinstance(turns, list) or not turns or not all(isinstance(t, str) for t in turns):
        raise ScenarioError("'turns' must be a non-empty list of strings")
    fixture = raw.get("fixture") or {}
    approvals = raw.get("approvals") or {}
    budgets = raw.get("budgets") or {}
    session_context = _resolve(
        base, raw.get("session_context"), key="session_context", required=True
    )
    assert session_context is not None
    return Scenario(
        scenario_id=str(raw.get("id") or source.stem),
        source=source,
        session_context=session_context,
        fixture_repo=_resolve(base, fixture.get("repo"), key="fixture.repo", required=False),
        fixture_data_files=_resolve(
            base, fixture.get("data_files"), key="fixture.data_files", required=False
        ),
        policy=_resolve(base, raw.get("policy"), key="policy", required=False),
        packs=tuple(str(p) for p in raw.get("packs") or ()),
        turns=tuple(turns),
        approvals=Approvals(
            approve=tuple(approvals.get("approve") or ()),
            deny=tuple(approvals.get("deny") or ()),
            max_decisions=int(approvals.get("max_decisions", 20)),
        ),
        budgets=Budgets(**budgets),
        oracles=_oracles(raw.get("oracles") or {}),
        continue_message=str(raw.get("continue_message") or "continue"),
        yield_reply=str(raw.get("yield_reply") or "Proceed with your best judgment."),
        env={str(k): str(v) for k, v in (raw.get("env") or {}).items()},
    )

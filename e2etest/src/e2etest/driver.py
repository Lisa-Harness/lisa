"""Drive a real ``HostRuntime`` through a scenario's conversation."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import runtime.turn_hooks_builder
from runtime.bootstrap import build_runtime
from runtime.handle_message import handle_message
from runtime.paths import inject_packs_venv_site_packages
from runtime.types import HostCommandResult

from e2etest.environment import RunPaths
from e2etest.scenario import Scenario
from e2etest.scripted import ScriptedGatewayProvider, load_script

_APPROVAL_SCHEMA = "lisa.approval.v1"


@dataclass
class Exchange:
    """One message sent to the host and what came back."""

    index: int
    kind: str
    message: str
    text: str = ""
    is_error: bool = False
    operator_exit: str = ""
    yielded_question: str = ""
    approval: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    duration_s: float = 0.0


@dataclass
class DriverOutcome:
    """Everything the driver observed; ``aborted`` names a blown budget."""

    exchanges: list[Exchange] = field(default_factory=list)
    stages: list[dict[str, Any]] = field(default_factory=list)
    aborted: str = ""
    model_calls: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "aborted": self.aborted,
            "exchanges": [asdict(e) for e in self.exchanges],
            "stages": self.stages,
            "model_calls": self.model_calls,
        }


class RecordingTransport:
    """``TransportCallbacks`` that records stages and errors instead of rendering."""

    def __init__(self, outcome: DriverOutcome) -> None:
        self._outcome = outcome
        self.errors: list[str] = []

    async def on_stage(self, label: str, data: dict[str, Any] | None = None) -> None:
        self._outcome.stages.append(
            {
                "exchange": len(self._outcome.exchanges),
                "t": round(time.monotonic(), 3),
                "label": label,
                "data": _jsonable(data or {}),
            }
        )

    async def on_text_chunk(self, text: str, reasoning: str = "") -> None:
        return None

    async def on_response(self, text: str, data: dict[str, Any] | None = None) -> None:
        return None

    async def on_debug_context(self, context: dict[str, Any]) -> None:
        return None

    async def on_error(self, message: str, exc: Exception | None = None) -> None:
        self.errors.append(message)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def approval_payload(result: HostCommandResult) -> dict[str, Any] | None:
    """Return the structured approval request a turn paused on, if any."""
    data = result.data or {}
    nested = data.get("approval")
    if isinstance(nested, dict) and nested.get("request_id"):
        return nested
    if data.get("schema") == _APPROVAL_SCHEMA and data.get("request_id"):
        return data
    return None


class _Conversation:
    def __init__(self, runtime: Any, scenario: Scenario, outcome: DriverOutcome) -> None:
        self.runtime = runtime
        self.scenario = scenario
        self.outcome = outcome
        self.transport = RecordingTransport(outcome)
        self.decisions = 0
        self.continues = 0
        self.yield_replies = 0

    async def send(self, kind: str, message: str) -> Exchange:
        if len(self.outcome.exchanges) >= self.scenario.budgets.max_messages:
            raise _BudgetExceededError(f"max_messages={self.scenario.budgets.max_messages}")
        message = self._expand_message(message)
        exchange = Exchange(index=len(self.outcome.exchanges), kind=kind, message=message)
        self.outcome.exchanges.append(exchange)
        self.transport.errors = []
        started = time.monotonic()
        result = await handle_message(self.runtime, message, self.transport)
        exchange.duration_s = round(time.monotonic() - started, 2)
        exchange.text = result.text
        exchange.is_error = result.is_error
        exchange.operator_exit = result.operator_exit
        exchange.yielded_question = result.yielded_question
        exchange.approval = approval_payload(result)
        exchange.errors = list(self.transport.errors)
        return exchange

    def _expand_message(self, message: str) -> str:
        """Resolve ``$RECOVERY_OPEN_ID`` from the run's recovery store.

        Substitutes the work-item id of the first open (running/stalled/
        indeterminate) recovery record, read straight from
        ``data/state/recovery-items.json``. Left verbatim when there is no
        open record (the substitution happens at send time, so a stall
        created by an earlier turn resolves here). Lets scenarios drive
        deterministic operator remediation by id without parsing model
        prose.
        """
        if "$RECOVERY_OPEN_ID" not in message:
            return message
        path = self.runtime.paths.data_dir / "state" / "recovery-items.json"
        if not path.is_file():
            return message
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return message
        if not isinstance(records, dict):
            return message
        for record in records.values():
            if record.get("state") in ("running", "stalled", "indeterminate"):
                return message.replace("$RECOVERY_OPEN_ID", str(record.get("work_item_id") or ""))
        return message

    async def follow_up(self, exchange: Exchange) -> Exchange | None:
        """Answer whatever left the turn unfinished, or return ``None`` when it finished."""
        budgets = self.scenario.budgets
        if exchange.approval:
            if self.decisions >= self.scenario.approvals.max_decisions:
                raise _BudgetExceededError("approvals.max_decisions")
            self.decisions += 1
            verdict = self.scenario.approvals.decide(str(exchange.approval.get("target_id", "")))
            return await self.send(verdict, f"/{verdict} {exchange.approval['request_id']}")
        if exchange.yielded_question and self.yield_replies < budgets.max_yield_replies:
            self.yield_replies += 1
            return await self.send("yield_reply", self.scenario.yield_reply)
        if exchange.operator_exit and self.continues < budgets.max_continues:
            self.continues += 1
            return await self.send("continue", self.scenario.continue_message)
        return None

    async def run(self) -> None:
        for turn in self.scenario.turns:
            exchange: Exchange | None = await self.send("turn", turn)
            while exchange is not None:
                exchange = await self.follow_up(exchange)


class _BudgetExceededError(Exception):
    pass


async def _drive(runtime: Any, scenario: Scenario, outcome: DriverOutcome) -> None:
    conversation = _Conversation(runtime, scenario, outcome)
    await runtime.start()
    try:
        async with asyncio.timeout(scenario.budgets.wall_clock_s):
            await conversation.run()
    except _BudgetExceededError as exc:
        outcome.aborted = f"budget exceeded: {exc}"
    except TimeoutError:
        outcome.aborted = f"budget exceeded: wall_clock_s={scenario.budgets.wall_clock_s}"
    finally:
        try:
            await asyncio.wait_for(runtime.aclose(), timeout=30)
        except TimeoutError:
            outcome.aborted = outcome.aborted or "runtime.aclose() timed out"


@contextmanager
def _gateway_override(scripted: ScriptedGatewayProvider | None) -> Iterator[None]:
    """Bind every provider gateway to the model script while the runtime is wired."""
    if scripted is None:
        yield
        return
    hooks = runtime.turn_hooks_builder
    original = hooks._resolve_gateway_provider_for_runtime
    hooks._resolve_gateway_provider_for_runtime = lambda _runtime: scripted
    try:
        yield
    finally:
        hooks._resolve_gateway_provider_for_runtime = original


def run_conversation(scenario: Scenario, paths: RunPaths, packs_venv: Path) -> DriverOutcome:
    """Boot a runtime against ``paths`` and play the scenario's turns."""
    outcome = DriverOutcome()
    if inject_packs_venv_site_packages(packs_venv) is None:
        outcome.aborted = f"packs venv has no site-packages: {packs_venv}"
        return outcome
    scripted: ScriptedGatewayProvider | None = None
    if scenario.model_script is not None:
        scripted = load_script(scenario.model_script)
        outcome.model_calls = scripted.log
    with _gateway_override(scripted):
        runtime = build_runtime(
            host_kind="e2e",
            data_dir=str(paths.data_dir),
            session_context=str(paths.session_context),
            streaming="none",
            surface_type=scenario.surface.type,
            surface_id=scenario.surface.id,
            in_process_mcp=True,
        )
        if runtime.session.restart_required:
            outcome.aborted = "runtime flagged a pending reload during boot (pack auto-install?)"
            return outcome
        asyncio.run(_drive(runtime, scenario, outcome))
    return outcome

"""Exact-mode turn replay engine core (Spec 3 slice 3, TURN-REPLAY.md).

The engine re-runs one recorded turn through production code with two
replacements: the model gateway (the archive-backed provider from slice 2)
and tool *execution* (recorded results, via ``ToolHooks.answer_tool_call``).
Everything beneath the model — routing, policy, packs, sandbox, persistence
— runs as production code. The run happens in a scratch data root so the
source thread and ledgers are never touched.

The report collects every divergence the archive and transcript can see:
model-request divergences (model id, tool list, message count), tool-result
gaps, response mismatch, and effect-receipt mismatch. ``ok`` is True only
for a faithful replay.
"""

from __future__ import annotations

import contextlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kernel.provider_contracts import CompletionRequest, CompletionResult
from sdk.thread import ThreadStore

from runtime.replay._recording import load_auxiliary_calls, load_recorded_turn
from runtime.replay._tools import TranscriptToolAnswerer
from runtime.testing.archive import ArchiveGatewayProvider, ArchiveOutcomeError
from runtime.testing.scripted import ScriptError


@dataclass
class ReplayDivergence:
    """One divergence between a recording and its replay."""

    kind: str  # "model_request" | "tool_result" | "response" | "receipts"
    field: str = ""
    call_id: str = ""
    tool_call_id: str = ""
    recorded: Any = None
    replayed: Any = None
    detail: str = ""


@dataclass
class ReplayGatewayError:
    """One gateway-script failure the replay observed, as structured data.

    FIX-3: the kernel converts a gateway ``ScriptError`` into the turn's
    response text, so a script failure (archive exhaustion, a malformed
    script) previously surfaced only as a response divergence with the
    message buried in the replayed text. The routing provider records
    each failure here — naming the queue that failed and the wrapped
    underlying message — and the report projects them so operators see a
    structured, machine-readable failure instead of inferring one from
    response text.

    ``call_id`` is empty unless the failing call's archived snapshot
    identified itself (``ArchiveOutcomeError`` carries one; exhaustion
    and routing refusals do not).
    """

    kind: str = "gateway_script_error"
    queue: str = ""  # "turn" | "auxiliary"
    message: str = ""
    call_id: str = ""


@dataclass
class ReplayReport:
    """Outcome of replaying one recorded turn in exact mode."""

    turn_index: int
    turn_id: str
    mode: str = "exact"
    divergences: list[ReplayDivergence] = field(default_factory=list)
    gateway_errors: list[ReplayGatewayError] = field(default_factory=list)
    response_match: bool = True
    receipts_match: bool = True
    error: str = ""

    @property
    def ok(self) -> bool:
        """True when the replay was faithful: no error, no divergences, matches."""
        return (
            not self.error
            and not self.gateway_errors
            and not self.divergences
            and self.response_match
            and self.receipts_match
        )


@dataclass
class ReplayGateway:
    """A ``ProviderGateway`` that routes through the bound provider."""

    owner: ReplayGatewayProvider

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        return self.owner.answer(request)


@dataclass
class ReplayGatewayProvider:
    """Route one gateway call to the archive queue that recorded it.

    A replayed turn re-runs two kinds of model calls as production code:
    the turn loop's own calls (recorded in ``turn_input.calls``) and the
    off-turn auxiliary calls the planner, titler, and distillers make
    (recorded in the decisions ledger's ``model.request`` rows). Both
    consume a request, so both need an archived answer — but they are
    recorded in two different places, and interleaving them into one queue
    would desync whichever call arrives first.

    FIX-2: the discriminator is the request's explicit ``call_kind`` —
    ``"turn"`` selects the turn-call queue, ``"auxiliary"`` the
    auxiliary-call queue — stamped at each call's owning seam. An
    unclassified or unknown kind fails closed with a clear
    :class:`ScriptError` instead of guessing: a wrong guess desyncs both
    archives on the first misroute, and the refusal names the field so
    the unclassified seam surfaces at its owner.
    """

    turn_calls: ArchiveGatewayProvider
    auxiliary_calls: ArchiveGatewayProvider
    gateway_errors: list[ReplayGatewayError] = field(default_factory=list)

    def gateway_for(self, provider: str) -> ReplayGateway:
        """Return the routing gateway; ``answer`` routes per request."""
        _ = provider
        return ReplayGateway(self)

    def answer(self, request: CompletionRequest) -> CompletionResult:
        kind = getattr(request, "call_kind", "")
        if kind == "turn":
            return self._answer_queue(self.turn_calls, queue="turn", request=request)
        if kind == "auxiliary":
            return self._answer_queue(self.auxiliary_calls, queue="auxiliary", request=request)
        raise ScriptError(
            f"replay gateway cannot route a request with call_kind={kind!r}: "
            'the owning seam must stamp call_kind ("turn" or "auxiliary") '
            "on every gateway request"
        )

    def _answer_queue(
        self,
        queue_provider: ArchiveGatewayProvider,
        *,
        queue: str,
        request: CompletionRequest,
    ) -> CompletionResult:
        """Answer from one queue, recording script failures as structured data.

        FIX-3: ``ArchiveOutcomeError`` is an intentionally recorded non-ok
        model outcome (the archived call failed and replay must reproduce
        that failure), not a gateway malfunction — it re-raises without
        recording a gateway error. Every other ``ScriptError`` from the
        queue (archive exhaustion, a malformed script, a routing refusal)
        is a genuine gateway failure: exactly one structured
        :class:`ReplayGatewayError` is appended naming the queue and the
        wrapped underlying message, and the same ``ScriptError`` family
        re-raises so kernel error behavior is unchanged.
        """
        try:
            return queue_provider.answer(request)
        except ArchiveOutcomeError:
            raise
        except ScriptError as exc:
            self.gateway_errors.append(
                ReplayGatewayError(
                    kind="gateway_script_error",
                    queue=queue,
                    message=str(exc),
                    call_id="",
                )
            )
            raise ScriptError(f"{queue}-call queue: {exc}") from exc

    @property
    def divergences(self) -> list[dict[str, Any]]:
        """Both queues' divergence notes, turn calls first."""
        return [*self.turn_calls.divergences, *self.auxiliary_calls.divergences]


def _seed_scratch_config(source_data_dir: Path, scratch_data_dir: Path) -> None:
    """Seed the scratch run's config from the source data dir.

    Mirrors ``e2etest.environment._copy_base_data`` (implemented locally —
    the replay package must not import e2etest): copy ``config/models.yaml``
    and ``config/overlays/`` so the scratch runtime resolves the same models
    and overlays; ``.env`` is never copied.
    """
    source_config = source_data_dir / "config"
    if not source_config.is_dir():
        return
    scratch_config = scratch_data_dir / "config"
    scratch_config.mkdir(parents=True, exist_ok=True)
    models_yaml = source_config / "models.yaml"
    if models_yaml.is_file():
        shutil.copy2(models_yaml, scratch_config / "models.yaml")
    overlays = source_config / "overlays"
    if overlays.is_dir():
        shutil.copytree(overlays, scratch_config / "overlays", dirs_exist_ok=True)


def _replay_divergences_from_provider(
    provider: ArchiveGatewayProvider | ReplayGatewayProvider,
) -> list[ReplayDivergence]:
    """Project the archive provider's divergence notes onto the report."""
    return [
        ReplayDivergence(
            kind="model_request",
            field=str(note.get("field", "")),
            call_id=str(note.get("call_id", "")),
            recorded=note.get("recorded"),
            replayed=note.get("replayed"),
        )
        for note in provider.divergences
    ]


def _replay_divergences_from_gaps(answerer: TranscriptToolAnswerer) -> list[ReplayDivergence]:
    """Project recorded tool-result gaps onto the report."""
    return [
        ReplayDivergence(
            kind="tool_result",
            tool_call_id=str(gap.get("tool_call_id", "")),
            detail=str(gap.get("tool", "")),
        )
        for gap in answerer.gaps
    ]


def _project_gateway_errors(report: ReplayReport, provider: ReplayGatewayProvider) -> None:
    """FIX-3: project the gateway's structured script failures onto the report.

    The kernel converts a raised gateway ``ScriptError`` into the turn's
    response text, so the projection must run after the run regardless of
    how the exception surfaced. The first gateway error also wins the
    report's ``error`` summary when nothing louder already claimed it, so
    old operators see a loud failure and new ones read the structured
    entries.
    """
    report.gateway_errors.extend(provider.gateway_errors)
    if not report.error and provider.gateway_errors:
        first = provider.gateway_errors[0]
        report.error = f"gateway script error ({first.queue} queue): {first.message}"


def _compare_receipts(recorded: list[dict[str, Any]], replayed: list[dict[str, Any]]) -> bool:
    """Compare recorded and replayed effect receipts structurally."""

    def _key(receipt: dict[str, Any]) -> tuple[str, str, str]:
        return (
            str(receipt.get("effect", "")),
            str(receipt.get("capability_id", "")),
            str(receipt.get("artifact_id", "")),
        )

    return [_key(r) for r in recorded] == [_key(r) for r in replayed]


def _newest_scratch_entry(scratch_data_dir: Path) -> dict[str, Any] | None:
    """Load the newest thread entry the replayed run wrote, if any.

    The scratch run's thread store lives under ``<data_dir>/threads``; the
    active segment is ``<base>.current.jsonl``. The newest entry is the
    last non-empty line of the newest active segment.
    """
    thread_dir = scratch_data_dir / "threads"
    if not thread_dir.is_dir():
        return None
    candidates: list[tuple[float, Path]] = []
    for active in thread_dir.rglob("*.current.jsonl"):
        try:
            candidates.append((active.stat().st_mtime, active))
        except OSError:
            continue
    if not candidates:
        return None
    newest_path = max(candidates, key=lambda item: item[0])[1]
    with newest_path.open(encoding="utf-8") as handle:
        for line in reversed(handle.readlines()):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except ValueError:
                continue
            if isinstance(entry, dict):
                return entry
    return None


async def replay_turn(
    *,
    source_thread_path: Path,
    turn_index: int,
    scratch_data_dir: Path,
    source_data_dir: Path | None = None,
    session_context: str = "",
) -> ReplayReport:
    """Replay one recorded turn in exact mode and report divergences.

    The source thread is read-only for the engine: it is opened for loading
    only, and the replayed run writes to ``scratch_data_dir``. The engine
    sets and clears the session overrides around the run, mirroring the
    e2e driver's gateway-override binding.
    """
    report = ReplayReport(turn_index=turn_index, turn_id="")

    # 1. Load the recorded turn (read-only).
    from runtime.replay import ReplayError

    try:
        store = ThreadStore(source_thread_path)
        recorded = load_recorded_turn(store, turn_index)
    except ReplayError as exc:
        report.error = str(exc)
        return report
    except Exception as exc:
        report.error = f"failed to load recorded turn {turn_index}: {exc}"
        return report
    report.turn_id = recorded.turn_id

    # 2. Archived calls drive the model; the transcript answers tools. The
    # off-turn auxiliary calls (planner, titles, distillers) come from the
    # decisions ledger — the replay re-runs them as production code, so they
    # need their own archive queue or the first one desyncs the turn calls.
    seed_dir = source_data_dir if source_data_dir is not None else source_thread_path.parent
    auxiliary_calls = load_auxiliary_calls(
        source_data_dir=seed_dir,
        thread_base_path=store.base_path,
        user_slug=str(
            store.load_turns({turn_index}).get(turn_index, {}).get("user_slug", "") or ""
        ),
    )
    provider = ReplayGatewayProvider(
        turn_calls=ArchiveGatewayProvider(calls=recorded.calls),
        auxiliary_calls=ArchiveGatewayProvider(calls=auxiliary_calls),
    )
    answerer = TranscriptToolAnswerer(tool_results=recorded.tool_results)

    # 3. Seed the scratch config from the source data dir.
    _seed_scratch_config(seed_dir, scratch_data_dir)

    # 4. Build the scratch runtime.
    from runtime.bootstrap import build_runtime

    runtime = build_runtime(
        host_kind="replay",
        data_dir=str(scratch_data_dir),
        session_context=session_context,
        streaming="none",
        in_process_mcp=True,
    )

    # 5. Set the session overrides.
    runtime.session.provider_gateways_override = provider
    runtime.session.tool_call_answerer = answerer.answer

    # 6. Run the turn, then close in ``finally``.
    from runtime.callbacks import NullTransport
    from runtime.handle_message import handle_message

    transport = NullTransport()
    result_text = ""
    try:
        await runtime.start()
        result = await handle_message(runtime, recorded.query, transport)
        result_text = result.text
    except Exception as exc:
        report.error = f"replay run failed: {exc}"
    finally:
        with contextlib.suppress(Exception):
            await runtime.aclose()
        runtime.session.provider_gateways_override = None
        runtime.session.tool_call_answerer = None

    # 7. Collect divergences from the archive and the transcript gaps.
    report.divergences.extend(_replay_divergences_from_provider(provider))
    report.divergences.extend(_replay_divergences_from_gaps(answerer))
    _project_gateway_errors(report, provider)

    # 8. Compare outcomes.
    report.response_match = result_text == recorded.response_text
    if not report.response_match:
        report.divergences.append(
            ReplayDivergence(
                kind="response",
                recorded=recorded.response_text,
                replayed=result_text,
            )
        )
    scratch_entry = _newest_scratch_entry(scratch_data_dir)
    replayed_receipts = list(scratch_entry.get("effect_receipts") or []) if scratch_entry else []
    report.receipts_match = _compare_receipts(recorded.effect_receipts, replayed_receipts)
    if not report.receipts_match:
        report.divergences.append(
            ReplayDivergence(
                kind="receipts",
                recorded=recorded.effect_receipts,
                replayed=replayed_receipts,
            )
        )

    # 9. Return the report.
    return report

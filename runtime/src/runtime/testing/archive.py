"""Archive-backed scripted gateway: answer a real runtime from a recording.

Spec 3 (docs/SPEC/TURN-REPLAY.md, "Foundation") requires the archive-backed
replay gateway and the e2e scripted gateway to share one implementation.
Slice 1 moved the scripted core into :mod:`runtime.testing.scripted`; this
module adds the archive-backed provider beside it. Both providers expose the
same ``gateway_for`` seam and raise the same :class:`ScriptError` family, so a
host can bind either at the gateway-override seam.

Two answer bodies exist on purpose. ``ScriptedGatewayProvider.answer``
synthesizes a plausible response from a hand-written script (it regenerates
tool-call ids and hardcodes usage/finish_reason). ``ArchiveGatewayProvider``
reproduces a recorded response verbatim: original tool-call ids, usage,
reasoning content, truncation, and continuation must survive replay exactly,
so it projects the archived snapshot instead of reusing the scripted answer
body.

The kernel mints fresh ``call_id``s per run, so recorded order is the only
sound join: the provider consumes archived calls in ``turn_input.calls``
order, matching ``ScriptTurn``'s index convention.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kernel.provider_contracts import (
    CompletionRequest,
    CompletionResult,
    CompletionUsage,
    ProviderContinuation,
)

from runtime.testing.scripted import ScriptError


class ArchiveOutcomeError(ScriptError):
    """A recorded call ended in a non-ok outcome; replay must reproduce it."""

    call_id: str
    outcome: str

    def __init__(self, call_id: str, outcome: str, message: str = "") -> None:
        self.call_id = call_id
        self.outcome = outcome
        super().__init__(message or f"archived call {call_id} ended with outcome {outcome}")


@dataclass
class ArchivedCall:
    """One recorded model call: its request and response snapshots."""

    call_id: str
    iteration: int
    attempt: str
    request: dict[str, Any]
    response: dict[str, Any]

    @classmethod
    def from_turn_input(
        cls, calls: list[dict[str, Any]], *, load: Callable[[str], dict[str, Any]]
    ) -> list[ArchivedCall]:
        """Dereference each ``turn_input.calls`` entry into an :class:`ArchivedCall`.

        ``load`` dereferences a result-object ref (``r<sha256>``) to its stored
        dict. Tests inject a synthetic loader; production wiring binds
        ``sdk.thread.load_result_object`` to a thread base path.
        """
        archived: list[ArchivedCall] = []
        for entry in calls:
            call_id = str(entry.get("call_id", ""))
            response_ref = entry.get("response_ref")
            if response_ref is None:
                raise ScriptError(
                    f"archived call {call_id} has no response_ref (archive gap); cannot replay"
                )
            request_ref = entry.get("ref")
            if request_ref is None:
                raise ScriptError(f"archived call {call_id} has no request ref; cannot replay")
            try:
                request = load(str(request_ref))
                response = load(str(response_ref))
            except Exception as exc:  # loader errors are archive gaps
                raise ScriptError(f"archived call {call_id} failed to load: {exc}") from exc
            archived.append(
                ArchivedCall(
                    call_id=call_id,
                    iteration=int(entry.get("iteration", 0)),
                    attempt=str(entry.get("attempt", "primary")),
                    request=request,
                    response=response,
                )
            )
        return archived


@dataclass
class ArchiveGateway:
    """A ``ProviderGateway`` that answers from the bound archive provider."""

    owner: ArchiveGatewayProvider

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        return self.owner.answer(request)


@dataclass
class ArchiveGatewayProvider:
    """``ProviderGatewayProvider`` whose every provider resolves to the archive."""

    calls: list[ArchivedCall]
    position: int = 0
    divergences: list[dict[str, Any]] = field(default_factory=list)

    def gateway_for(self, provider: str) -> ArchiveGateway:
        _ = provider
        return ArchiveGateway(self)

    def answer(self, request: CompletionRequest) -> CompletionResult:
        if self.position >= len(self.calls):
            raise ScriptError(
                f"archive exhausted after {len(self.calls)} calls; "
                "the replayed turn made an extra call"
            )
        call = self.calls[self.position]
        self.position += 1
        self._record_divergences(request, call)
        outcome = call.response.get("outcome", "ok")
        if outcome != "ok":
            error = call.response.get("error", {})
            raise ArchiveOutcomeError(
                call_id=call.call_id,
                outcome=str(outcome),
                message=str(error.get("message", "")) if isinstance(error, dict) else "",
            )
        response = call.response
        usage_snapshot = response.get("usage")
        continuation_snapshot = response.get("continuation")
        return CompletionResult(
            content=str(response["content"]),
            provider=str(response["provider"]),
            model_id=str(response["model_id"]),
            display_name=str(response.get("display_name", response["model_id"])),
            request_id=str(response.get("provider_request_id") or ""),
            tool_calls=copy.deepcopy(response.get("tool_calls") or []),
            usage=CompletionUsage(**usage_snapshot) if usage_snapshot else None,
            reasoning_content=str(response.get("reasoning_content", "") or ""),
            duration_ms=float(response.get("duration_ms", 0.0) or 0.0),
            finish_reason=response.get("finish_reason"),
            truncated=response.get("truncated"),
            continuation=(
                ProviderContinuation(**continuation_snapshot) if continuation_snapshot else None
            ),
        )

    def _record_divergences(self, request: CompletionRequest, call: ArchivedCall) -> None:
        recorded_model = call.request.get("model_id")
        replayed_model = getattr(request.model_def, "model_id", None)
        if recorded_model != replayed_model:
            self.divergences.append(
                {
                    "call_id": call.call_id,
                    "field": "model_id",
                    "recorded": recorded_model,
                    "replayed": replayed_model,
                }
            )
        recorded_tools = _tool_names(call.request.get("tools"))
        replayed_tools = _tool_names(request.tools)
        if recorded_tools != replayed_tools:
            self.divergences.append(
                {
                    "call_id": call.call_id,
                    "field": "tools",
                    "recorded": sorted(recorded_tools),
                    "replayed": sorted(replayed_tools),
                }
            )
        recorded_messages = len(call.request.get("messages", []))
        replayed_messages = len(request.messages)
        if recorded_messages != replayed_messages:
            self.divergences.append(
                {
                    "call_id": call.call_id,
                    "field": "message_count",
                    "recorded": recorded_messages,
                    "replayed": replayed_messages,
                }
            )


def _tool_names(tools: Any) -> set[str]:
    names: set[str] = set()
    for tool in tools or ():
        function = tool.get("function", tool) if isinstance(tool, dict) else None
        if isinstance(function, dict) and function.get("name"):
            names.add(str(function["name"]))
    return names


def archive_loader(base: Path) -> Callable[[str], dict[str, Any]]:
    """Bind ``sdk.thread.load_result_object`` to a thread base path."""

    def _load(ref: str) -> dict[str, Any]:
        from sdk.thread import load_result_object  # lazy: defers sdk startup cost

        return load_result_object(base, ref)

    return _load

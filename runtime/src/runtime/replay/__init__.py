"""Exact-mode turn replay (Spec 3, docs/SPEC/TURN-REPLAY.md).

Slice 3 provides the engine core: load a recorded turn from the thread
store, bind the archive-backed gateway (``runtime.testing.archive``) at the
runtime's gateway-override seam, answer every tool dispatch from the
recorded transcript (tools are never re-executed), run in a scratch data
root so the source thread and ledgers are never touched, and report the
divergences found.

The engine drives a real runtime — routing, policy, packs, sandbox, and
persistence all run as production code. Only two things are replaced: the
model gateway (the archive) and tool *execution* (recorded results, via
``ToolHooks.answer_tool_call``).

Later slices add counterfactual mode, scenario export, the ``lisa replay``
CLI and ``/replay`` in-session surfaces, full divergence detection
(request-content hashing, ``turn.decision`` comparison, run-manifest
comparison), child-turn replay, and approval-record replay.
"""

from runtime.replay._engine import (
    ReplayDivergence,
    ReplayGatewayError,
    ReplayGatewayProvider,
    ReplayReport,
    replay_turn,
)
from runtime.replay._recording import RecordedTurn, load_auxiliary_calls, load_recorded_turn
from runtime.replay._tools import ReplayToolGapError, TranscriptToolAnswerer


class ReplayError(RuntimeError):
    """A recorded turn cannot be replayed (missing turn, archive gap)."""


__all__ = [
    "RecordedTurn",
    "ReplayDivergence",
    "ReplayError",
    "ReplayGatewayError",
    "ReplayGatewayProvider",
    "ReplayReport",
    "ReplayToolGapError",
    "TranscriptToolAnswerer",
    "load_auxiliary_calls",
    "load_recorded_turn",
    "replay_turn",
]

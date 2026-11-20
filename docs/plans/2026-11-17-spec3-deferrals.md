# Spec 3 Deferred Work — Recorded 2026-11-17

**Status:** consciously deferred, with operator approval (2026-11-17): build
the Spec 3 e2e determinism gate now; record everything else as deferred.
**Companion plan:** `docs/plans/2026-11-17-spec3-determinism-gate-plan.md`.

## Deferred items

| Item | Rationale | Resume pointer |
| --- | --- | --- |
| Spec 3 counterfactual mode | Developer ergonomics; no near-term commercial buyer | `docs/SPEC/TURN-REPLAY.md` §1 (counterfactual row) |
| Spec 3 export mode | Developer ergonomics; defer with counterfactual | `docs/SPEC/TURN-REPLAY.md` §5 |
| `lisa replay` / `/replay` surfaces | Blocked on the modes above; nothing to surface yet | `docs/SPEC/TURN-REPLAY.md` §6 |
| Spec 4 Trace Export & Timeline (OTel) | Infra plumbing; only matters when a customer demands OTel/Datadog | `docs/SPEC/TRACE-EXPORT.md` (unstarted) |

## Why these are safe to defer

The determinism gate (the one slice the operator approved) closes the
spec's core acceptance: a recorded turn replays with no divergence, and a
replay never writes the source data root. Everything listed above is
surface area on top of that proven core — modes, exports, and CLI surfaces
that a future slice can add without re-deriving the engine.

## How to resume

Each row's resume pointer names the spec section that specifies the work.
Start from the pointer, re-read the engine core (`runtime/src/runtime/replay/`)
and the gate (`e2etest/tests/test_replay_gate.py`), and keep the gate green
while adding the surface. The engine is the substrate; the gate is the
insurance that it stays faithful.

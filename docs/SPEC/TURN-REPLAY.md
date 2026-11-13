# Turn Replay Specification

> **Status:** Proposed.
> **Order:** 3 of 4 in the observability and replay program.
> **Depends on:** [evidence envelope](./EVIDENCE-ENVELOPE.md) (call identity,
> run manifest) and [model call archive](./MODEL-CALL-ARCHIVE.md) (response
> pairs).

## Goal

Re-run any recorded turn through production code with the model answered from
the archive. Detect where a re-run diverges from the recording, branch a
recorded turn into a live counterfactual, and export any turn as a permanent
regression scenario.

Reserve **replay** for re-executing recorded work. Rebuilding the open task as
model history is **rehydration**
([`_rehydrate.py`](../../runtime/src/runtime/thread/_rehydrate.py)); it does
not re-execute tools or model calls. Showing a previous response is a
**reprint** (`/last`), and continuing a paused task is a **resume**.

## Foundation

[`ScriptedGatewayProvider`](../../e2etest/src/e2etest/scripted.py) already
replaces every provider gateway with a deterministic responder, and runs
everything beneath the model as production code: routing, policy, approvals,
packs, sandbox, and persistence. Replay is a scripted gateway whose script is
the archive, not hand-written Python.

## 1. Replay modes

| Mode | Model | Tools | Purpose |
| --- | --- | --- | --- |
| `exact` | Archived responses, in `call_id` order | Recorded results | Prove the loop reproduces the recording; find behavior drift |
| `counterfactual` | Archived up to a branch point, then live | Recorded results until the branch, then policy-governed live | "What would a different model, prompt, or pack have done here?" |
| `export` | Archived | Recorded | Write a regression scenario (§5) |

### Tool results

- By default, tools are never re-executed. Each dispatch is answered from the
  recorded transcript result for its `tool_call_id`. Write and execute effects
  are never repeated against the real workspace.
- `--live-tools=read` re-executes read-effect tools inside a scratch copy of
  the workspace, for checking whether a read would return the same thing
  today.
- In counterfactual mode, tools after the branch point run live under normal
  policy and approvals, in a scratch data root and workspace.

### Approvals

Recorded approval decisions from `approvals.jsonl` answer approval requests by
`request_id` and `tool_call_id`. An approval request with no recorded decision
is a divergence, not a prompt to the operator.

## 2. Context pinning

Request assembly reads live state: memory, the thread store, and context
routing. Without pinning, the first call of every old turn would diverge.

| Setting | Behavior |
| --- | --- |
| `pinned` (default for `exact`) | Each call's request is served from its archived snapshot. Only the loop's decisions are under test |
| `rebuilt` | Requests are re-assembled from current state and compared with the archive. The first difference is the context-drift report |

## 3. Divergence detection

At each model call, the engine compares the request the loop built with the
archived request for the same `call_id`:

- the model ID and sampling parameters;
- the tool list (names and schema hashes);
- the messages, by index and by content hash.

At each loop decision, it compares the action and reason with the archived
`turn.decision` row. The first mismatch stops the replay (or, with
`--continue`, is reported and skipped). The report names the call or decision,
the field, and both values' hashes. Content diffs are available behind the
same redaction rules as forensics.

Before starting, the engine compares the recorded `run_ref` with the current
run manifest and lists the packs, versions, and config hashes that differ. A
divergence with a manifest change is expected; a divergence without one is a
determinism bug.

## 4. Isolation

- Replay runs in a scratch data root. It never appends to the source thread or
  the source ledgers.
- Its own thread and ledger rows carry `replay_of: <turn_id>` and the replay
  mode, so they are never mistaken for production evidence.
- Wall-clock budgets are disabled in `exact` mode. Recorded call durations are
  reported, not re-enacted.
- Network access for model calls is disabled in `exact` and `export` modes.
  Any call the archive cannot answer is a divergence.

## 5. Scenario export

`export` writes a scenario under a pack's `tests/e2e/`:

- a `.scenario.yml` that declares `model: {script: ...}`;
- a script module that serves the archived responses by call order, generated
  rather than hand-written;
- assertions derived from the recording: effect receipts, `block_origin`,
  handoffs and transitions, yields, and the final answer's hash.

The exported scenario then runs under `lisa-e2e run <path>` like any other.
Any production incident becomes a permanent regression test in one command.
Export applies the redaction policy to the archived content it writes; content
the policy cannot clear blocks the export, and the error names it.

## 6. Surfaces and ownership

| Surface | Use |
| --- | --- |
| `lisa replay <turn> [--mode exact\|counterfactual\|export] [...]` | Operator CLI |
| `/replay <turn>` | In-session operator command; `exact` only, report rendered like `/forensics` |

The replay engine lives in `runtime` (`runtime.replay`), because it drives
a real runtime. The archive-backed gateway and the scripted gateway share one
implementation. Move the gateway core out of `e2etest` into a runtime test
seam, and have `e2etest` import it, rather than maintaining two copies.

There is no model-facing operation in this spec. A model that re-runs its own
turns is a separate design question.

## Acceptance

- Every existing e2e scenario, recorded once and then replayed with `exact`,
  has no divergence. This is the determinism gate, and it runs in CI.
- A recorded production turn replays with identical effect receipts,
  `block_origin`, and lifecycle decisions.
- Changing a mode prompt and replaying in `rebuilt` mode reports the first
  differing message and names the changed `config` hash.
- An exported scenario passes under `lisa-e2e run`.
- A replay never writes to the source data root. A test asserts the source
  files are byte-identical before and after.

## Open questions

- Sources of non-determinism inside the loop (UUIDs, timestamps in prompts,
  set ordering). The determinism gate will find them; each fix is either
  injection or recorded input.
- Child turns: replay the child from its own archived calls via
  `parent_dispatch_id`, or treat the delegate dispatch as a recorded tool
  result? The current lean is a full child replay in `exact`, and a recorded
  result in `export`.
- Turns recorded before the [model call archive](./MODEL-CALL-ARCHIVE.md)
  existed have no responses. They can be replayed only by reconstructing
  responses from the transcript, which is lossy: no reasoning, no
  intermediate attempts. Supporting that is a separate decision, and the
  default is to refuse.

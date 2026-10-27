# Turn forensics

How to read the evidence for a past turn — and what that evidence does and
does not prove.

Lisa records four kinds of durable evidence about a turn. They are written by
different components, for different reasons, and any single one of them
answers only part of the question. Forensics is the act of joining them.

## The four sources

| Source | Written by | Answers |
| --- | --- | --- |
| `data/threads/<user>.*.jsonl` | [runtime.thread](../runtime/src/runtime/thread/) | What was asked, what was said, how it routed, whether it was blocked, what receipts it produced |
| `logs/decisions.jsonl` | [DecisionTelemetryHandler](../kernel/src/kernel/decision_telemetry.py) | Why the loop continued, finished, or yielded at each iteration; and what ran, for how long, with what status |
| `logs/token-usage.jsonl` | [TokenUsageTelemetryHandler](../kernel/src/kernel/token_usage.py) | Which model was called, with how many tokens, at what cost |
| `logs/runtime.jsonl` | [sdk.logging_setup](../sdk/src/sdk/logging_setup.py) | Process-level warnings and errors in the same window |

The thread is the primary record: it is lossless, append-only, and segmented.
The other three are telemetry around it.

## The join key

Everything joins on `turn_index` — the global thread ordinal computed once at
turn start, stamped on the thread entry, and carried into every decision,
dispatch, and token record.

Two rules matter more than the mechanics:

**Exact match only.** There is no nearest-turn fallback. A dispatch that
happened 200 ms before turn N+1 started is not evidence about turn N, and
guessing would fabricate a timeline that looks authoritative.

**`null` is not `0`.** `turn_index: null` means the record belongs to *no*
thread turn — off-turn slice work, or a host with no thread. Turn `0` is the
thread's first real turn. Treating a null index as turn 0 attributes
background processing to the first turn, which is the most damaging failure a
forensic reader can have: it manufactures plausible-looking evidence.
[`resolve_turn_index`](../kernel/src/kernel/models.py) normalizes raw values
once, at every publish site.

## Two entry points, one renderer

| Caller | Surface |
| --- | --- |
| The model | `turn_forensics` operation (ambient, read-effect) |
| The operator | `/forensics [last \| turn \| start-end]` |

Both go through [`ForensicReader`](../sdk/src/sdk/forensics/reader.py) and
[`render_bundle`](../sdk/src/sdk/forensics/render.py). That is deliberate: two
formatters describing one turn is how the introspection path once ended up
reading a field the writer never emitted. One reader, one formatter, two entry
points.

## Reading the output

```text
### Sources                    Which evidence actually existed for this read
### Turn N · mode              Identity, routing, blocked state, receipts
  - decisions (N iteration(s)) Why the loop did what it did
  - dispatches (N, M failed)   What executed, how long, what failed
  - cost                       Tokens and spend for the turn
### Totals                     Aggregates over the requested turns
### Unattributed records       Activity that attached to no requested turn
### Log warnings and errors    Process problems in the same window
### Read notes                 Every cap or bound that fired
```

### Distinguish the three failure shapes

| Shape | Meaning |
| --- | --- |
| **Blocked** turn | Ended without satisfying its completion contract. `block_origin` names the contract (`task_completion`, `output_guard`, …) |
| **Failed dispatch** | One call errored or timed out. The turn may still have completed |
| **Degraded operation** | Completed with warnings; the status is `degraded`, not `error` |

They have different causes and different fixes. A blocked turn is a routing or
contract problem; a failed dispatch is a capability problem.

## What the reader will not do

- **Present a missing file as an empty one.** An unreadable source appears in
  `sources.missing`. "No errors" and "I could not read the error log" are
  different answers, and only one is safe to act on.
- **Drop unattributable records.** A decision or dispatch naming a turn outside
  the requested range is reported under *Unattributed records*.
- **Silently truncate.** Turns, records, log lines, and bytes are all capped,
  and every cap that fires appends a note.
- **Exceed its bounds.** A bundle is a summary with pointers, not a second copy
  of the thread. Expand anything with `thread_read_range`.

## Privacy

The thread store persists raw prompts, responses, and model reasoning at rest.
[docs/LOGGING.md](LOGGING.md) is explicit that persisted threads are **not**
sanitized.

Read-time redaction therefore applies to every forensic read, resolved through
[`sdk.thread.resolve_redaction_policy`](../sdk/src/sdk/thread/redaction.py) —
the same contract the `lisa-thread` server uses. The store's own redaction
covers only `build_recent_excerpt` / `build_manifest` / `search`;
`iter_range` returns raw entries by design, so any surface that projects them
into a prompt must apply the policy itself.

The `turn_forensics` operation exposes `include_raw`, which is **off by
default**. Enabling it bypasses read-time redaction for that one call, logs a
warning, and states the bypass in the bundle. Use it only when the redacted
view genuinely cannot answer the question, and treat the result as private.

Scope — the data root and `user_slug` — comes from the execution context, never
from a tool parameter. A caller cannot name another user's evidence.

## Intentionally unavailable

- **Full `reasoning_text` and raw `turn_input`** through the thread readout.
  Reasoning is capped to an 800-character excerpt; the raw model request is not
  exposed by `thread_read_content`. This is a deliberate read-surface limit, not
  a bug, and forensics does not bypass it.
- **Live state.** Everything here was read from durable files. A mode that was
  blocked in turn 1057 is not blocked now.
- **Anything before the join key existed.** Records written before
  `turn_index` was stamped carry no index and appear as unattributed. That is
  accurate: they cannot be attributed retroactively.

## Verifying a claim

Every line in a bundle names its turn. Before acting on a conclusion —
especially "this keeps happening" — confirm it against the primary record:

```text
thread_read_range(start_turn=N, end_turn=N, include_transcript=true)
```

The bundle is a summary. The thread is the evidence.

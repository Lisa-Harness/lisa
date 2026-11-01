# Turn forensics

How to read the evidence for a past turn — and what that evidence does and
does not prove.

Lisa records four kinds of durable evidence about a turn. They are written by
different components, for different reasons, and any single one of them
answers only part of the question. Forensics is the act of joining them.

## The five sources

| Source | Written by | Answers |
| --- | --- | --- |
| `data/threads/<store>.*.jsonl` | [runtime.thread](../runtime/src/runtime/thread/) | What was asked, what was said, how it routed, whether it was blocked, what receipts it produced. `<store>` is the user slug, or `<user>--<conversation>` when the host isolates one store per conversation |
| `logs/decisions.jsonl` | [DecisionTelemetryHandler](../kernel/src/kernel/decision_telemetry.py) | Why the loop continued, finished, or yielded at each iteration; and what ran, for how long, with what status. The full segment set (`decisions.jsonl` plus every sealed `decisions.jsonl.<n>`) is read, not only the active file |
| `logs/token-usage.jsonl` | [TokenUsageTelemetryHandler](../kernel/src/kernel/token_usage.py) | Which model was called, with how many tokens, at what cost |
| `logs/approvals.jsonl` | [ApprovalLedger](../runtime/src/runtime/approval_ledger.py) | Who authorized each write/execute effect, and how: one append-only row per approval state transition (requested, approved, denied, consumed, expired), with the operator identity, the policy-vs-operator source, and the turn attribution |
| `logs/runtime.jsonl` | [sdk.logging_setup](../sdk/src/sdk/logging_setup.py) | Process-level warnings and errors in the same window |

The thread is the primary record: it is lossless, append-only, and segmented.
The other four are telemetry around it. The approval ledger is one row per
state *transition*, not per request: a single approval can contribute
requested → approved → consumed, and the bundle groups those by
`request_id` into the decision's lifecycle. A policy auto-allow records
`decision_source: "policy"` so an automatic authorization is never
misattributed to an operator. Off-turn model calls an operation
dispatch makes (memory mining, consolidation, distills) never reach the
thread entry's `turn_input.calls`; their `model.request` rows in the
decisions ledger — carrying the content-addressed `request_ref` for the
archived request — are the record for exactly those calls, and they join
on the same `turn_index` (`null` when the dispatch belongs to no turn).

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
  - approvals (N request(s))  Who authorized each effect, and how
### Totals                     Aggregates over the requested turns
### Unattributed records       Activity that attached to no requested turn
### Log warnings and errors    Process problems in the same window
### Read notes                 Every cap or bound that fired
```

A *dropped row* note is special: both ledger handlers stamp each row with a
per-instance `sequence` and `ledger_instance`, so a gap within one writer's
numbering proves rows were lost (failed publish, lock timeout, crash) and
counts them. The note is a lower bound and rows predating the stamps are
skipped, not reported — the same honest posture as pre-attribution ledger
rows. See [loss observability](LOGGING.md#loss-observability).

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

Scope — the data root, `user_slug`, and the thread store's own name — comes
from the execution context, never from a tool parameter. A caller cannot name
another user's evidence.

**The store and the ledgers are scoped by different values.** The thread
store is named by its *store stem*, which for a host that isolates one store
per conversation is `<user_slug>--<conversation_id>`. The shared ledgers are
scoped by *owner*, which is the plain `user_slug` stamped on every ledger
row, and by *thread*, which is the store stem also stamped on every row. A
reader therefore takes both: `ForensicReader(thread_store_slug=...)` opens the
store the host wrote, while `user_slug` selects the owner's ledger rows and
the stem keeps one thread's rows off another's turns. Conflating them fails
quietly — reading the store by owner finds nothing, and scoping the ledger
by owner alone joins another thread's rows onto this one's turns.

The logs under `<data-dir>/logs` are shared by every user of a data root, and
`turn_index` is per-thread, so two users' turn 5 collide. Decision, dispatch,
and token records therefore carry `user_slug`, and a scoped read (forensics and
behavioural signals) keeps only the requesting user's records. Ledger records
with no `user_slug` predate attribution and cannot be assigned to one user, so
they are excluded and the bundle states how many. `runtime.jsonl` carries no
user and is read as host-wide process diagnostics.

Two conversations of one user share the same ledgers, and their turn indices
collide the same way two users' do — so rows also carry the thread's store
stem, and a scoped read keeps each thread's rows off the others' turns. Rows
written before that stamp existed carry no stem and are excluded from a
scoped read; the bundle states how many.

Both views read the same store. The host resolves the thread directory once
and injects it, so the operator's `/forensics` and the model's
`turn_forensics` open the same directory even when the host moved it with a
command-line argument the environment does not carry.

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
- **Ledger records before per-user attribution.** Decision and dispatch rows
  written before `user_slug` was stamped are excluded from scoped reads, so
  turns from before the upgrade show no dispatches or decisions.

## Ambient behavioural signals

`/forensics` and `turn_forensics` answer questions you ask. The post-turn slice
also answers a question nobody asks explicitly: *is this pattern repeating?*
After each turn, `compute_behavioral_signals` scans a short window and, **only
when something crosses its threshold**, injects a context item:

| Signal | Fires when |
| --- | --- |
| Blocked streak | N consecutive turns ended blocked (`blocked_streak`, default 3) |
| Dispatch failure rate | ≥25% of ≥10 windowed dispatches failed |
| Concentrated failures | One turn recorded ≥3 failed dispatches |
| Missing receipts | A turn dispatched work but recorded no effect receipts |
| Mode dwell | N consecutive turns in one mode (default 8) |
| Cost spike | A turn used ≥3× the window median tokens |

**Gated independently of `operational_health`.** Health answers "is the host
degraded?" — memory, asyncio tasks, MCP pool, log errors. A streak of blocked
turns on a perfectly healthy process is still the thing an operator most wants
surfaced, so reusing the health gate would hide exactly the case this exists
for.

**Silent by default.** An ordinary window emits nothing and costs no prompt
tokens, matching the self-model's ambient-silence contract. Thresholds live in
[`SignalThresholds`](../sdk/src/sdk/forensics/signals.py); the defaults are
deliberately conservative, because a signal that fires on ordinary turns makes
the whole channel ignorable.

**Bounded for the slice budget.** The window is 12 turns and the telemetry read
is capped at 256 KiB per file. Measured on a live store — where a thread entry
averages ~128 KiB because it carries a full transcript — the scan costs
~21–34 ms. That is why the window is small: `load_recent` JSON-decodes every
entry, and a 40-turn window cost ~50 ms, which would consume the budget on its
own. A wider behavioural view belongs in `/forensics`, which is on-demand and
not budget-bound.

## Verifying a claim

Every line in a bundle names its turn. Before acting on a conclusion —
especially "this keeps happening" — confirm it against the primary record:

```text
thread_read_range(start_turn=N, end_turn=N, include_transcript=true)
```

The bundle is a summary. The thread is the evidence.

## Verifying the evidence itself

The thread is append-only, but "append-only" is a property of the writer,
not a proof about the files. Every entry since the chain feature commits
to its predecessor's raw-line hash, and every seal's index line records
its terminal hash (see the [thread spec](SPEC/THREAD_STORE.md#hash-chain)).
`/forensics <selection> --verify-chain` (and
`turn_forensics(verify_chain=true)` for the model) walks the whole store
and appends one read note: verified (with chained/pre-chain counts), or
**THREAD CHAIN BROKEN** naming the first divergence. It decompresses every
sealed segment, so it is an explicit audit — never the default read.

Honest limits: it detects after-the-fact edits and deletions (an attacker
must coherently rewrite everything after the edit point, including the
index); it cannot detect a full-store rewrite by an attacker who controls
the files, or tampering with the *newest* entry — nothing links to it yet,
which is why the note also reports the chain head, the value to notarize
externally. History predating the chain is reported as pre-chain, never
pretended to be checked.

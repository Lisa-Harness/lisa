# Model Call Archive Specification

> **Status:** Proposed.
> **Order:** 2 of 4 in the observability and replay program.
> **Depends on:** [evidence envelope](./EVIDENCE-ENVELOPE.md) (`call_id`,
> `attempt`).
> **Unblocks:** [turn replay](./TURN-REPLAY.md).

## Goal

Every model call, whether in-turn or off-turn, has an archived request and
response pair, addressed by `call_id`. The pair must be sufficient to
reproduce what the call did to the loop without contacting the provider.

## Current state

- **Requests are archived.** The turn loop records a request snapshot through
  `TurnHooks.models.record_request` ([turn_hooks.py](../../kernel/src/kernel/turn_hooks.py))
  and stores `{"iteration", "ref"}` in the thread entry's `turn_input.calls`.
  Off-turn calls archive the request in
  [`OperationHost._record_request_snapshot`](../../kernel/src/kernel/seams/host_adapter.py)
  and publish a `model.request` row.
- **Responses are not archived.** What survives is a by-product:
  - the assistant message and tool calls the loop appended to `transcript`;
  - a token usage row;
  - the final `finish_reason` in the entry's provenance.

  The read surface caps reasoning at 800 characters. Intermediate calls'
  finish reasons, truncation flags, continuations, provider request IDs,
  streaming facts, and failed or cancelled calls leave no durable record.
  Off-turn calls keep only token usage.

This means half of every model exchange cannot be inspected or replayed.

## Non-goals

- No change to what the model sees, or to read-surface caps. This spec adds
  recording, not exposure.
- Raw provider wire payloads are not the primary record. Replay consumes the
  normalized [`CompletionResult`](../../kernel/src/kernel/provider_contracts.py).

## 1. Recording hook

Add `record_response` next to `record_request` on `TurnHooks.models`, with the
same contract:

- It accepts a snapshot and returns an `r<sha256>` ref.
- It persists into the same result-object store, in the same record shape, so
  thread tools and the forensic reader can dereference it.
- A recording failure logs a WARNING and the call proceeds unrecorded, which
  matches the existing request-recording posture.

`OperationHost` records off-turn responses through the same hook on its audit
context.

## 2. Response snapshot

| Field | Source |
| --- | --- |
| `call_id`, `attempt`, `iteration` | [Evidence envelope](./EVIDENCE-ENVELOPE.md#2-call-level-join-keys) |
| `request_ref` | The paired request snapshot |
| `outcome` | `ok`, `error`, `cancelled`, `wall_budget_expired`, `stream_idle_timeout` |
| `content`, `tool_calls`, `reasoning_content` | `CompletionResult`, verbatim and uncapped |
| `finish_reason`, `truncated`, `continuation` | `CompletionResult` |
| `provider`, `model_id`, `provider_request_id` | `CompletionResult` (`request_id`) |
| `usage` | `CompletionResult.usage`, using the same "unknown is not zero" rules as the token ledger |
| `duration_ms` | Wall time of the call |
| `streamed`, `time_to_first_token_ms`, `chunk_count` | Streaming path only; `null` when not streamed |
| `error` | Exception class and a bounded message for non-`ok` outcomes; never a traceback or headers |

Non-`ok` calls are recorded too. A call that hit the wall budget or a silent
stream is exactly the call an operator needs to see.

Raw provider payloads can be archived alongside the normalized snapshot only
with an explicit operator opt-in. This is for debugging gateway normalization,
and is off by default because raw payloads can carry provider metadata the
normalized form deliberately drops.

## 3. Where refs land

| Record | Change |
| --- | --- |
| Thread entry `turn_input.calls[]` | Adds `response_ref`, so each entry becomes `{call_id, iteration, attempt, ref, response_ref}` |
| `model.request` ledger row | Adds `response_ref` |
| Token usage row | Unchanged; joins to the call by `call_id` |

Update the [thread store spec](./THREAD_STORE.md) `turn_input` row when this
lands.

## 4. Storage and retention

- Content addressing deduplicates identical responses, including scripted test
  runs.
- Responses are small relative to requests, which carry the full message
  array, so the expected growth is a small fraction of current request
  archive size. Measure this on a live store before landing, and record the
  figure here.
- Retention follows the thread store. Ledger and object retention policy is
  owned by [trace export and timeline](./TRACE-EXPORT-AND-TIMELINE.md#7-export-and-retention).

## 5. Read surfaces

- Read-time redaction ([`resolve_redaction_policy`](../../sdk/src/sdk/thread/redaction.py))
  applies to response objects exactly as it does to request objects.
- `/forensics` and `turn_forensics` show one line per call: model, attempt,
  outcome, finish reason, time to first token, duration, tokens, and cost.
  Content stays behind the ref.
- Full `reasoning_content` is readable only through the existing `include_raw`
  bypass. The 800-character read cap stays the default.
- The reader reports a call with a `ref` but no `response_ref` as an archive
  gap in read notes, so loss stays countable.

## Acceptance

- A streamed turn with a tool-call fallback archives two pairs, and the
  fallback's `attempt` is `stream_tool_fallback`.
- A truncation rescue archives the truncated response (`truncated: true`) and
  the continuation as separate pairs.
- A wall-budget expiry archives a pair with `outcome: wall_budget_expired` and
  no content.
- A memory-mining operation archives a pair, and its `model.request` row
  carries `response_ref`.
- A failing `record_response` logs a WARNING, the turn completes, and the
  forensic reader reports the archive gap.
- Redaction tests cover response objects.

## Open questions

- Should streaming record per-chunk timing (a time series) or only
  time-to-first-token and chunk count? Chunk timing helps diagnose stalls but
  multiplies storage.
- Should tool *results* gain a parallel snapshot with duration and outcome?
  The transcript already holds results verbatim, and `tool.telemetry` holds
  timing, so the current lean is no.

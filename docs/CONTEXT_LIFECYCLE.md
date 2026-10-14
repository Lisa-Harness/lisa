# Context lifecycle

Context is a bounded, provider-visible view over durable evidence, not the
record of truth itself. This guide describes shipped residency and continuation
behavior; the original residency proposal and its phased implementation log
have been consolidated here.

## Three different mechanisms

| Mechanism | Owns | Reference |
| --- | --- | --- |
| Admission and lanes | Which evidence earns prompt space, ordering, ranking, required/pinned rules | [Context routing](CONTEXT_ROUTING_GUIDE.md) |
| Request residency | How much tool evidence can remain in the full model request | [Request allocator](../kernel/src/kernel/context/request_budget.py) |
| Durable storage | Exact conversation, tool results, model requests, stable references | [Thread specification](SPEC/THREAD_STORE.md) |

The model's context window limits one request, not the size of the thread.
Provider caching changes billing/reuse, not storage durability or the
physical input window.

## Whole-request allocation

Before a primary turn-loop request is sent, the allocator accounts for
serialized messages, system instructions, tool-call arguments, schemas,
structured-output instructions, and the requested completion ceiling.
`CONTEXT_REQUEST_MARGIN_SHARE` reserves a fraction of the model window
(default 5%) for counting and framing uncertainty.

Full permitted tool results stay resident while the complete request fits.
Under pressure the allocator projects results into smaller representations
and recounts the request, including the reference/inventory overhead. It
allocates result space newest-first, retaining recoverable references for
older results. Previously referenced results do not automatically re-expand.
This is not a general access-tracked LRU.

`CONTEXT_REDUCTION_TARGET_SHARE` (default 0.75) leaves append headroom when
reducing results, avoiding a rewrite on every subsequent call. Required
input is not silently dropped: the allocator can reduce the output
reservation and records the effective ceiling; if required input itself
cannot fit, it raises an explicit budget error.

Recognized model IDs use local tokenizer counts; unknown tokenizers use a
labelled UTF-8-byte/3 estimate. Provider input usage calibrates underestimates
for subsequent requests. These counts and the safety margin are estimates,
not a guarantee for every provider framing or multimodal payload.

## References and bounded retrieval

Large tool results are persisted before provider projection. A result ref
is `r<sha256>`; thread refs address a turn/message/call or field. A preview
must still refer to the original content, not to another stored preview.

Thread range, search, content-window, and find tools let the model retrieve
addressed evidence on demand. Pagination and byte/character limits still
apply; expanding a ref does not grant unlimited request space. The read
path validates object integrity and rejects unsafe references.

The thread readout also expands exact model-request snapshots recorded
under `turn_input.calls`. Snapshots include `context_budget`: window,
requested/effective completion ceiling, safety margin, append headroom,
required/input tokens, result-byte allowance, counting method, and whether
projection reduced the request.

## Replay across turns

The host rebuilds the open task from the lossless transcript. Replay and
live tool results use the request's remaining model window rather than
an unconditional fixed 64 KiB working set.

The runtime replay ceiling defaults to **0** (no independent cap). A session
can request `context.history.replay_max_bytes`; the tighter finite host/session
ceiling wins. This is separate from the token-capped recent-conversation
and context lanes. Neither a reference nor a lane exclusion deletes evidence.

## Approval continuation

For model-loop approvals, the kernel captures a
[TurnContinuation](../kernel/src/kernel/turn_continuation.py). The runtime
persists it by content address before returning the approval panel.
Resuming an exact approval continues the saved dispatch cursor, including
remaining calls in the model's batch, without replaying entry operations
or synthesizing a new user query.

Resume revalidates ownership, expiry, mode configuration, and registry
authorization. Consumed or expired approvals cannot replay an action.
Policy-approved actions execute in the same loop; reaching
`MAX_AUTO_APPROVED_ACTIONS` (default 10) saves the same kind of continuation.
Another explicit allowance preserves cumulative counters, receipts, and
turn budgets; it does not reset them.

This mechanism does not replace entry-operation approval recovery or all
iteration/step-cap continuation paths. Recovery records and `/resume` of a
yielded turn are different contracts.

## Cache-stable requests

Stable instructions and sorted tool schemas form a reusable prefix.
Actionable lifecycle guidance is appended initially and when it changes,
at complete tool-batch boundaries; changing counters do not rewrite the
system prefix. An actual mode/stage change can establish a new cache boundary.

- Anthropic gateways use explicit stable-prefix and request-level cache
  controls where supported.
- OpenAI/Azure paths use prefix-cache keys; capable Azure models can also
  use explicit breakpoints. Rejected optional cache parameters are retried
  without them according to the adapter contract.
- Gemini uses implicit caching; explicit Cached Content lifecycle management
  is not supplied by this mechanism.
- Provider-specific opaque reasoning continuation is replayed only for its
  owning provider/model, not converted into visible content.

`LISA_PROMPT_CACHE_ENABLED=false` disables the host's explicit cache controls.
It does not promise to disable caching performed independently by a provider.
See [token accounting](TOKEN_BUDGET.md#accounting-and-cost).

## Limits and follow-up work

Do not infer full cache residency restoration across ordinary new turns,
general LRU behavior, reduction of every kind of non-tool history, or exact
provider-native token counts from the current tool-result allocator.
Host integration and provider translation remain part of the boundary.

For investigations, use the request snapshot, usage event, and addressed
result objects together. Do not reconstruct model input from the final
answer alone, and do not upload those artifacts without sanitizing them.

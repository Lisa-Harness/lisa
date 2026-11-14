# Evidence Envelope Specification

> **Status:** Proposed.
> **Order:** 1 of 4 in the observability and replay program.
> **Depends on:** nothing.
> **Unblocks:** [model call archive](./MODEL-CALL-ARCHIVE.md),
> [turn replay](./TURN-REPLAY.md), and
> [trace export and timeline](./TRACE-EXPORT-AND-TIMELINE.md).

## Goal

Make every durable evidence row self-describing, versioned, and addressable to
the exact model call, dispatch, run configuration, and parent turn that
produced it.

Today the ledgers join only at turn granularity (`turn_index`, `user_slug`,
`thread_store_slug`). A turn with several iterations cannot attribute a cost
to the request that incurred it, a failed dispatch cannot be matched to the
tool call in the transcript, an old turn cannot say which packs and modes
produced it, and delegated child work has no recorded parent.

## Non-goals

- No new ledger files. The existing ledgers (`decisions.jsonl`,
  `token-usage.jsonl`, `approvals.jsonl`) and the thread entry gain fields.
- No model response content. That is the
  [model call archive](./MODEL-CALL-ARCHIVE.md).
- No OpenTelemetry work. This spec reserves the `trace_id` and `span_id` slots;
  [trace export](./TRACE-EXPORT-AND-TIMELINE.md) fills them.

## Current state

| Ledger | Versioned | Write-side allowlist | Join keys |
| --- | --- | --- | --- |
| `token-usage.jsonl` | `schema_version` (v3) | `_LEDGER_FIELDS` in [token_usage.py](../../kernel/src/kernel/token_usage.py) | turn, user, pseudonymous thread |
| `decisions.jsonl` | No | No: [`DecisionTelemetryHandler`](../../kernel/src/kernel/decision_telemetry.py) persists any mapping a publisher sends | turn, user, thread stem |
| `approvals.jsonl` | Per the [approval ledger](../../runtime/src/runtime/approval_ledger.py) | Ledger-owned | turn, `request_id` |

The token ledger's posture — typed fields, an allowlist that drops unknown
keys at write, and "missing is unknown, never zero" — is the model this spec
extends to every topic.

## 1. Common envelope

Every row on every ledger carries the envelope below. Handler-stamped fields
are written by the ledger handler, never by publishers.

| Field | Stamped by | Meaning |
| --- | --- | --- |
| `topic` | handler | Existing topic name (`turn.decision`, `tool.telemetry`, …) |
| `schema_version` | publisher model | Integer version of this topic's payload schema |
| `emitted_at` | publisher | ISO 8601 UTC time the fact occurred |
| `logged_at` | handler | Existing append time |
| `sequence`, `ledger_instance` | handler | Existing loss-detection stamps, unchanged |
| `user_slug`, `thread_store_slug` | publisher | Existing scope keys, unchanged |
| `turn_index` | publisher | Existing join key; `null` still means "no turn" |
| `turn_id` | publisher | Stable per-execution identity (§4); distinct for child turns |
| `run_ref` | publisher | Content-addressed run manifest (§3) |
| `trace_id`, `span_id` | publisher | Reserved; `null` until trace export lands |

### Typed topic payloads

- Each topic has one typed payload model in the kernel and an allowlist
  derived from it, exactly like `_LEDGER_FIELDS`. The decision handler drops
  unknown keys at write instead of persisting arbitrary mappings.
- Payload schemas are exported as JSON Schema and committed so non-Python
  hosts (the Go host) and readers can validate rows.
- Adding a field bumps the topic's `schema_version`. Removing or retyping a
  field is a hard cutover with a new version; readers never guess.

### Reader compatibility

- Readers accept every version up to the current one. A row with no
  `schema_version` is version 0 (legacy): read on a best-effort basis, never
  rejected, and counted in a read note, as pre-attribution rows already are.
- A row with a version newer than the reader understands is reported as
  unreadable, not silently skipped.

## 2. Call-level join keys

Attribution goes down to the individual model call and dispatch.

### Model call identity

Each model call gets a `call_id`, minted when the request is built, which is
unique within its turn. A call is identified by `(turn_id, iteration, attempt)`,
where `attempt` names why the call happened:

| `attempt` | When |
| --- | --- |
| `primary` | The iteration's first call |
| `stream_tool_fallback` | The streaming path fell back to `complete()` for the tool-call payload |
| `truncation_continuation` | A truncation rescue continued a cut-off completion |
| `retry` | A provider error retry |

Today the streaming fallback and truncation rescue make extra calls inside one
iteration, which `turn_input.calls` (`{"iteration", "ref"}`) cannot tell
apart.

### Where the keys land

| Record | New fields |
| --- | --- |
| Thread entry `turn_input.calls[]` | `call_id`, `attempt` (alongside `iteration`, `ref`) |
| Token usage row | `call_id`, `iteration`, `request_ref` |
| `turn.decision` row | `call_id` of the call whose output the decision resolved |
| `tool.telemetry` row | `dispatch_id`, `tool_call_id`, `call_id` of the requesting model call, `iteration` |
| `model.request` row (off-turn) | `call_id`, `dispatch_id` of the operation that made the call |
| Approval ledger row | `tool_call_id`, `dispatch_id` |

`dispatch_id` is minted per tool or operation dispatch, including rejected
dispatches, so a rejection row and its transcript tool result address each
other.

## 3. Run provenance

A turn's behavior is a function of code and content, not just inputs.
Replaying turn 900 under today's packs must be able to say what changed.

### Run manifest

The host builds one run manifest at startup and rebuilds it whenever packs,
modes, policy, or router configuration reload. It is stored once in the
result-object store and addressed as `run_ref` (`r<sha256>`).

| Field | Content |
| --- | --- |
| `lisa` | Installed versions of `kernel`, `runtime`, `sdk`, and `mcp_server` |
| `packs[]` | Name, version, and a content hash of each installed pack's tree |
| `config` | Content hashes of the resolved mode, policy, and router configuration |
| `model_data_version` | Effective model data version (see [model data layers](./implemented/MODEL-DATA-LAYERS.md)) |
| `host` | Host type and host protocol version |
| `python` | Interpreter version |

The manifest never contains filesystem paths, environment variables, or
credentials. Pack identity is name, version, and hash, not install location.

### Per-turn stamps

- Thread entries gain `run_ref` and `mode_digest`: the content hash of the
  resolved mode definition the turn actually ran (after overrides).
- Every ledger row carries `run_ref` via the envelope.

## 4. Lineage

### Turn identity

Every execution gets a `turn_id`. The kernel already mints one
(`provenance.kernel_turn_id`); this spec promotes it to a top-level thread
entry field and an envelope field.

### Child turns

Fork and delegate child turns ([`_child_turn.py`](../../kernel/src/kernel/turn_phases/_child_turn.py))
currently send tokens and status to the parent but record no identity of
their own. They will:

- mint their own `turn_id`;
- carry `parent_turn_id` and `parent_dispatch_id` (the fork or delegate
  dispatch that spawned them);
- keep the parent's `turn_index`, because they belong to the parent's thread
  turn.

This gives a tree under each `turn_index` that both the forensic reader and
trace export can render.

## 5. Reader changes

[`ForensicReader`](../../sdk/src/sdk/forensics/reader.py) and
[`render_bundle`](../../sdk/src/sdk/forensics/render.py):

- join token rows to calls by `call_id` and render per-call cost under each
  iteration;
- join dispatch rows to transcript tool calls by `tool_call_id`;
- render the delegation tree from `parent_turn_id`;
- name the run manifest per turn, and flag when consecutive turns ran under
  different manifests;
- report legacy (version 0) and unreadable rows in read notes.

## 6. Privacy

- Envelope and payload fields are identifiers, counts, hashes, statuses, and
  bounded reasons. No prompt, response, or tool-argument content.
- Allowlists are enforced at write, so a publisher bug cannot leak content to
  disk.
- `run_ref` objects are host-wide and contain no user data.

## Acceptance

- A schema test asserts every publisher's payload validates against its
  committed JSON Schema, and that the decision handler drops unknown keys.
- A turn with a streaming tool-call fallback produces two `turn_input.calls`
  entries with distinct `call_id`s and `attempt` values, and two token rows
  that join to them.
- A failed dispatch row joins to exactly one transcript tool result by
  `tool_call_id`.
- A delegate child's rows carry its own `turn_id`, the parent's `turn_index`,
  and `parent_dispatch_id` equal to the delegate dispatch's `dispatch_id`.
- Reloading a pack changes `run_ref`; the next thread entry records the new
  one.
- Legacy rows still render, with a read note counting them.

## Open questions

- Should `run_ref` objects live in the per-thread result store or a host-wide
  store under `<data-dir>`? Host-wide avoids duplicating them across
  conversations.
- How to version the approval ledger's existing rows without rewriting the
  file. The default is to treat them as version 0.

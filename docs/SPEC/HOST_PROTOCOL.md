# SPEC: Lisa Host Protocol v1

> **Status:** Initial specification; implementation has not started. No API,
> schema, command, or compatibility promise in this document is shipped yet.
> Architecture and ownership: [Host SDK architecture](../HOST_SDK_ARCHITECTURE.md).

## 1. Goal and non-goals

Define the smallest language-neutral protocol that lets an application host a
real Lisa runtime without embedding CPython or porting the framework. The
Python runtime remains the only orchestration implementation. A host-language
SDK owns a child process and exposes idiomatic sessions, turns, events,
approvals, and signals.

Non-goals for v1:

- Converting or modifying `lisa-cli`.
- Native reimplementation of the kernel, runtime, packs, or policy.
- MCP server development, MCP-pool brokering, or a new provider protocol.
- Network transports, remote daemon discovery, FFI, or CPython embedding.
- Arbitrary administration, catalog, runtime-attribute, or tool RPC.
- Concurrent turns in one runtime, process pooling, or multi-session children.
- General-purpose attachment streaming or large binary payloads.

The initial proof is a Python Host SDK client and a Go Host SDK client driving
the same real Python runtime through the same conformance scenarios.

## 2. Terminology and ownership

- **Host:** application that owns UI, authentication, deployment, secrets,
  scheduling, and the child process.
- **Client:** Host SDK implementation in the application's language.
- **Bridge:** Python `runtime.host_bridge` process implementing the server side.
- **Session:** one client connection to one child and one `HostRuntime`.
- **Operation:** one turn request or approval decision; operations do not
  overlap within a session.
- **Event:** bridge-to-client notification associated with an operation.
- **Outcome:** the single terminal response to an operation.

The SDK owns public DTOs and the client. The runtime owns the bridge and all
runtime behavior. JSON Schemas under `sdk/src/sdk/host/schemas/v1/` are the
canonical wire shapes once landed. This specification owns their semantics;
the two must change together.

## 3. Process model and launch

The client launches an explicitly configured Python interpreter with:

```text
<python> -m runtime.host_bridge --protocol-version 1
```

The command is an argv vector, never a shell command. The client supplies the
installation/environment selection and any optional Python environment path.
The bridge does not search for or install Python, packs, or models on behalf
of an SDK. Missing or incompatible installations fail initialization.

The child inherits no ambient environment by default in SDK process builders.
The caller supplies an explicit environment map or deliberately opts into
inheriting the parent environment. Provider credentials travel through that
environment, not JSON messages. Clients must offer a secret-safe way to
construct this map and must never log its values.

One client session owns one child. The child holds at most one `HostRuntime`.
The same runtime-owner async task performs runtime start, all turns and
approval decisions, and close. The protocol reader may run separately but may
not execute runtime operations on another task.

The child's stdout is exclusively protocol bytes. Diagnostics go to the
configured log destination or stderr, never stdout. The bridge does not read
interactive input from stdin or emit terminal control sequences.

The client retains ownership of descendants launched by its child. Normal
runtime close must stop them. After an ungraceful bridge death, the client's
bounded cleanup uses the platform's child/job/process-group mechanism without
a shell or name-based process killing; conformance verifies that MCP/proxy
children do not survive. If the platform cannot provide that guarantee, the
client reports incomplete cleanup rather than claiming a clean shutdown.

## 4. Framing and JSON-RPC envelope

### 4.1 Framing

Each stdio message is one UTF-8 JSON object followed by LF (`0x0a`). JSON
strings escape embedded newlines. Writers must not add blank lines, prefixes,
or other output. Readers accept CRLF and remove the trailing CR before JSON
decoding. There is no `Content-Length` header in v1.

The maximum frame size is **16 MiB**, including the terminating LF. A reader
enforces the limit incrementally while reading rather than first creating an
unbounded line buffer. Oversize, invalid UTF-8, and malformed JSON are protocol
failures. V1 has no resumable event stream or binary side channel.

### 4.2 Envelope

Requests, responses, and notifications use JSON-RPC 2.0:

```json
{"jsonrpc":"2.0","id":"req-1","method":"session.start","params":{}}
```

```json
{"jsonrpc":"2.0","id":"req-1","result":{"session_id":"s-1"}}
```

```json
{"jsonrpc":"2.0","method":"turn.event","params":{}}
```

Rules:

- Request IDs are client-generated strings, unique among outstanding
  requests. They correlate responses only; they are not operation IDs.
- Notifications have no `id` and receive no JSON-RPC response.
- The bridge sends no requests to the client in v1; interaction uses client
  requests and bridge notifications.
- Batch requests are not supported.
- Params are objects. Omitted optional fields and explicit `null` are distinct
  only where a schema says so; otherwise both mean the documented default.
- Unknown methods and invalid params produce JSON-RPC errors. Unknown fields
  in otherwise valid DTOs are ignored for forward-compatible additive changes.
- JSON numbers used as IDs, counts, or sequence numbers must be safe integers
  across languages; generated IDs are strings rather than integers.

A transport failure, protocol-version failure, or closed stdin can prevent a
normal JSON-RPC error. Clients must then fail outstanding work explicitly.

## 5. Version and capability negotiation

`host.initialize` is mandatory and is the first request on a new process:

```json
{
  "jsonrpc": "2.0",
  "id": "req-1",
  "method": "host.initialize",
  "params": {
    "protocol_version": 1,
    "client": {"name": "example-host", "version": "0.1.0"},
    "capabilities": []
  }
}
```

The bridge returns:

```json
{
  "protocol_version": 1,
  "server": {"name": "lisa-runtime", "version": "<runtime-version>"},
  "session_id": "s-1",
  "capabilities": ["typed_events", "approval_decision", "operator_signal"]
}
```

The requested major version must equal the supported version. There is no
silent downgrade to another major version or automatic fallback to an older
protocol. Unknown capabilities are ignored, not treated as approval to use a
feature. The returned session ID is opaque and unique within the process.

`host.initialize` performs no runtime construction or environment discovery
beyond protocol/server identification. Initialization is valid only in the
`new` lifecycle state. After a fatal framing or negotiation error, the bridge
returns a typed JSON-RPC error when the request ID and transport state permit,
then closes the connection.

## 6. Public session and operation DTOs

### 6.1 Session options

`session.start` receives a curated options object corresponding to the public
runtime composition inputs needed by an external host:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `host_kind` | string | yes | Host identity for diagnostics; nonempty |
| `data_dir` | string | yes | Resolved data root; no implicit child cwd dependence |
| `session_context` | string | no | Resolved session-plan path |
| `user_slug` | string | no | Runtime conversation owner override |
| `conversation_id` | string | no | Conversation-scoped persistence identity |
| `overlays_dir` | string | no | Resolved authored overlay root |
| `thread_dir` | string | no | Resolved thread root |
| `no_thread` | boolean | no | Disable thread persistence; default false |
| `timezone` | string | no | IANA timezone; runtime resolves default |
| `state_snapshot_path` | string | no | Resolved local snapshot path |
| `streaming` | string | no | `none`, `stage`, `text`, or `hybrid` |
| `surface_type` | string | no | Authenticated surface category |
| `surface_id` | string | no | Authenticated surface identity |
| `in_process_mcp` | boolean | no | Runtime MCP implementation choice; default false |
| `identity` | object | no | Explicit authenticated identity fields below |

Paths are local to the machine running the bridge, are resolved by the trusted
client/host, and are passed as values rather than shell-expanded strings. V1
does not provide a remote filesystem abstraction. `user_slug` defaults through
the existing runtime composition; the bridge does not invent another default.

`identity` is an optional object with `workspace_id`, `user_id`, `user_slug`,
`user_name`, `principal_key`, `conversation_id`, `surface_type`, `surface_id`,
and `memory_scope` string fields. `workspace_id` is an authenticated host
scope, not a replacement for the session plan's manifest name. Supplying a
value in both `identity` and its top-level counterpart requires exact equality;
a conflict is `invalid_params`. The bridge must map these to one explicit
runtime configuration seam before start. If that seam cannot represent a
supplied value, start fails rather than dropping it. No identity field may be
taken from model text or a later turn request.

The current runtime derives several turn-identity values from user and surface
configuration. Slice A therefore includes a narrow public runtime identity
configuration seam and tests that its values reach turn and approval identity.
Until that mapping exists and is tested, the bridge must not claim support for
a separate principal, workspace, or memory scope.

Configuration resolution remains runtime-owned. The bridge uses public
runtime composition, not its own YAML loader, environment expansion, manifest
selection, or pack discovery.

### 6.2 Operation envelope

`turn.start` and `approval.decide` params include a client-generated
`operation_id` string, unique for the life of the session. Event notifications
and the terminal result carry this ID. Reusing an active ID is invalid;
reusing a completed ID returns `duplicate_operation` without replaying work.

Turn params:

```json
{
  "operation_id": "op-1",
  "message": "Explain this configuration"
}
```

The message is passed verbatim to `handle_message`. It may be conversational
text or a runtime host command, including `/help`, `/approve`, and `/deny`.
There is no client-side slash grammar. An empty or whitespace-only message is
invalid. The raw-output prompt decoration and exit-code policy used by
`lisa-cli` remain CLI concerns and are not bridge behavior.

Approval decision params:

```json
{
  "operation_id": "op-2",
  "request_id": "approval-id",
  "approve": true,
  "remember": null
}
```

`request_id` is required and nonempty; v1 does not implement the CLI's bare
`pending` lookup convention. `approve=false` means deny. `remember` is null or
an explicit policy prefix; no implicit default prefix is applied. The bridge
uses the typed approval-decision API and preserves the saved invocation's
identity and dispatch state.

### 6.3 Events

Every operation event is a `turn.event` notification with an envelope:

```json
{
  "session_id": "s-1",
  "operation_id": "op-1",
  "sequence": 3,
  "event": {
    "type": "text_chunk",
    "text": "partial answer",
    "reasoning": ""
  }
}
```

`sequence` starts at 1 for each operation and increases by exactly one for each
event. The client detects gaps or duplicate delivery as a protocol failure.
Events for an operation precede its JSON-RPC response; writes are serialized
so an event cannot overtake the terminal result.

| `event.type` | Additional fields | Source and meaning |
| --- | --- | --- |
| `stage` | `label` string, `data` object | `on_stage`; `data` is projected using the v1 allowlist below |
| `text_chunk` | `text`, `reasoning` strings | `on_text_chunk`; visible and reasoning deltas stay separate |
| `response` | `text` string, `data` object | `on_response`; does not replace the terminal outcome |
| `debug_context` | `context` object | `on_debug_context`; emitted only when runtime debug behavior requests it |
| `error` | `message` string, `category` string | `on_error`; nonfatal unless the terminal outcome says otherwise |

The event `data` object is an explicitly versioned projection, not arbitrary
Python callback serialization. V1 preserves fields consumed by rich hosts:
mode/stage/model identifiers, provider, iteration and turn indexes, usage and
budget counters, tool/operation identity and status, durations, proxy routing,
handoff/fork identifiers, and bounded tool-output text. The schema defines
their concrete names and scalar/list/object types. A field whose value cannot
be represented without a Python object is omitted and recorded as a mapping
defect in tests; it is not stringified wholesale.

Approval presentation is a `response.data.approval` object using the existing
runtime's structured approval projection. The v1 schema includes the existing
`lisa.approval.v1` fields: request/target/pack IDs, effect, safe preview,
redacted command/parameters, containment and reasons, expiry, and policy
verdict/reason/tier. The terminal `approval` object additionally includes an
`invocation_identity` object with `workspace_id`, `user_id`, `user_slug`,
`principal_key`, `conversation_id`, `surface_type`, `surface_id`,
`memory_scope`, `mode_id`, and `session_plan_digest` string fields, using the
values recorded with the pending invocation. These values are evidence for the
host; a decision cannot replace them. Approval decisions must use the returned
`request_id` exactly.

### 6.4 Terminal outcome

Every accepted operation receives exactly one response whose result is:

```json
{
  "session_id": "s-1",
  "operation_id": "op-1",
  "status": "completed",
  "text": "final response",
  "data": {},
  "is_error": false,
  "operator_exit": "",
  "yielded_question": "",
  "yield": null,
  "approval": null
}
```

`status` is one of `completed`, `yielded`, `awaiting_approval`, `paused`,
`stopped`, `terminated`, `interrupted`, or `failed`. The mapping is explicit:

| Runtime evidence | Status |
| --- | --- |
| Nonempty `yielded_question` | `yielded` |
| Undecided structured approval in the terminal result | `awaiting_approval` |
| `operator_exit=pause` | `paused` |
| `operator_exit=stop` | `stopped` |
| `operator_exit=terminate` | `terminated` |
| `operator_exit=interrupt` | `interrupted` |
| `is_error=true`, or an operation exception | `failed` |
| Otherwise | `completed` |

The fields `text`, `data`, `is_error`, `operator_exit`, and `yielded_question`
preserve the public `HostCommandResult` values; they do not replace `status`.
For a yield, `yield` is an object with `question`, `context`, `mode_id`, and
`remediation` (an array of command strings). For an approval, `approval` is the
structured request. Other outcomes use null for both fields.

The bridge does not infer success from prose or convert a yield/approval into
a normal answer. Slice B adds a narrow runtime-owned terminal-result
classifier that normalizes the existing approval projection, yield payload,
operator exit, and error flag into the contract above; the bridge consumes it
rather than duplicating runtime dispatch rules. An exception after an operation
is admitted becomes a `failed` outcome with a safe error category, not a
transport error that makes the client retry the operation. Errors before
admission use the JSON-RPC codes below.

One-shot exit codes, whether to print a question to stderr, and whether to
resume a turn are host policy.

## 7. Methods and lifecycle

| Method | Valid state | Behavior |
| --- | --- | --- |
| `host.initialize` | `new` | Negotiate version and allocate session ID |
| `session.start` | `initialized` | Build and start the runtime; return effective public settings |
| `turn.start` | `ready` | Run one message or host command |
| `approval.decide` | `ready` | Apply one typed decision, including continuation behavior |
| `turn.signal` | `running` | Set a cooperative operator signal; no turn result of its own |
| `session.close` | any post-initialize state | Stop admission and close the runtime if constructed |
| `host.shutdown` | `closed` | Exit the process successfully after replying |

Lifecycle states are `new`, `initialized`, `starting`, `ready`, `running`,
`closing`, and `closed`. Invalid transitions return `invalid_state`. A second
start or concurrent operation is rejected. `session.close` is idempotent only
as a client-facing cleanup operation: after the first close completes it
returns the same closed result without touching runtime resources again.
A close racing the final phase of an operation joins the existing close rather
than starting another one.

`session.start` returns the resolved settings a host legitimately needs for
rendering: effective streaming mode, user/conversation/surface identifiers,
and data/thread paths. It does not return provider credentials, raw process
environment, runtime objects, model prices, or internal service handles. A
host requiring additional diagnostics needs a later curated read API.

`turn.signal` params are `operation_id` and `signal`. V1 accepts `pause`,
`stop`, `terminate`, and `interrupt`; `eof` and `suspend` are process/host UI
concerns and are not sent to the runtime. The target must be the active
operation. The signal request is admitted while that operation is running,
but its response acknowledges only signal recording, never turn completion.

Closing while an operation runs first records `interrupt`, then allows the
runtime to finish cleanup on its owner task. The operation ends in
`interrupted` or `failed`, never `completed` merely because the pipe is
closing. If the turn already committed a successful terminal outcome before
the interrupt was observed, that outcome remains truthful and the close result
reports that no active operation was interrupted; close must not rewrite a
committed outcome.

Graceful close has a configurable client timeout. On timeout or child death the
client terminates its owned child and reaps it; termination is not a
runtime-level clean close. A child death cannot retract an already delivered
terminal outcome, but the client reports the broken session and does not admit
new work.

`host.shutdown` is rejected while a runtime is running or closing. The client
also treats child EOF as implicit shutdown: fail all outstanding requests,
stop event delivery, and release process resources.

## 8. Errors and failure containment

JSON-RPC errors use standard codes where applicable and these Lisa-specific
codes:

| Code | `data.category` | Meaning |
| --- | --- | --- |
| `-32700` | `parse_error` | Invalid JSON or envelope |
| `-32600` | `invalid_request` | Invalid envelope or unsupported batch |
| `-32601` | `method_not_found` | Unknown method |
| `-32602` | `invalid_params` | Schema-valid envelope with invalid params |
| `-32001` | `version_mismatch` | Unsupported protocol major |
| `-32002` | `invalid_state` | Method not valid in current lifecycle |
| `-32003` | `operation_in_progress` | A second operation was admitted |
| `-32004` | `duplicate_operation` | Operation ID would replay completed work |
| `-32005` | `startup_failed` | Runtime build/start failed |
| `-32006` | `runtime_failed` | Runtime operation failed outside a normal turn outcome |
| `-32007` | `frame_too_large` | Frame exceeded 16 MiB |
| `-32008` | `protocol_violation` | Ordering, sequence, or other protocol invariant failed |

Error `data` may contain `category`, `method`, `operation_id`, and `retryable`.
It contains a safe human-readable `message` in the standard error field. It
must not contain raw exceptions, Python tracebacks, environment maps, API
keys, or private runtime object representations. `retryable=true` is limited
to failures where the contract proves no operation started; runtime operations
are never automatically replayed.

A malformed individual request receives an error when its ID can be trusted.
Malformed notifications, oversized frames, and invalid event sequencing are
fatal protocol violations because no safe stream resynchronization exists.
The bridge closes the transport after recording such a failure; the client
fails outstanding work and performs bounded child cleanup.

A callback/event serialization failure fails the operation visibly rather than
silently dropping required terminal state. Optional diagnostic fields may be
omitted under the projection rules. A crashed child is not translated into a
model response.

## 9. Async, persistence, and security invariants

- Runtime start, operation execution, and close share one event loop and owner
  task. No `asyncio.run` per request and no cross-task pool close.
- The bridge serializes runtime operations. It does not add a queue that hides
  admission delays or allow concurrent mutation of one `HostRuntime`.
- Awaiting an event write applies backpressure to the runtime callback. A slow
  client can slow a turn, matching `TransportCallbacks` semantics.
- Interrupt is cooperative and leaves the MCP pool alive until normal runtime
  close. Abrupt process termination is a last-resort host action.
- Conversation sharing and isolation use existing runtime locks, thread
  stores, and conversation IDs. The protocol adds no second persistence layer.
- The host authenticates users and supplies identity. The protocol does not
  authenticate a remote caller or make a local child multi-tenant.
- Approval policy is evaluated in the runtime. The client cannot submit a
  preauthorized tool call or substitute a different saved invocation identity.
- Provider secrets are absent from messages and logs. Frame/error logging is
  metadata-only by default; message bodies require an explicit redacted debug
  facility outside the initial implementation.
- Stdout corruption, unsupported versions, malformed input, and invalid state
  fail closed. There is no direct-Python fallback hidden inside a client.

## 10. Schema and SDK requirements

Before implementation, add canonical schemas for the envelope, initialize,
session options, turn request, approval decision, signal, event, outcome, and
error data. Package them with `lisa-harness-sdk` and test that Python DTO
serialization validates against them.

`sdk.host` exposes the equivalent of:

```python
async with LisaHost(process_config) as host:
    await host.initialize(client_info)
    session = await host.start(options)
    result = await host.run_turn(message, on_event=...)
    await host.signal(result_operation_id, "interrupt")
    await host.close()
```

This is illustrative API shape, not authorization to expose implementation
objects. Event handlers are async or bridged by the client without blocking
its protocol reader. Process close is idempotent at the SDK resource boundary.

`lisa-go/host` provides equivalent Go types and an event callback/channel API.
It uses argv-based process launch, context cancellation tied to cooperative
signals and bounded cleanup, and no shell. Both SDKs generate or mechanically
validate DTOs against the same schemas; hand-maintained duplicate shapes are
not sufficient evidence of parity.

## 11. Conformance and acceptance gates

### Slice A: contract and lifecycle vertical slice

Deliver schemas, framing tests, Python bridge, and Python client. Verify:

- Version negotiation, unknown capabilities, and incompatible major versions.
- Initialize/start/`/help`/close/shutdown with a real runtime.
- Empty input, invalid state, duplicate operation, malformed JSON, CRLF,
  oversize frames, and stdout purity.
- Start failure and close after partial construction.
- No surviving child or runtime-owned async work after normal close.

### Slice B: turns, control, and durable outcomes

Use the existing scripted-model/test facilities; no live model dependency.
Verify:

- Successful turn with ordered stage/text/response events and terminal result.
- Host command, error, operator exit, yield, and approval outcomes.
- Typed approval decision with explicit ID and optional remember prefix.
- Pause/stop/terminate/interrupt signals and signal acknowledgment semantics.
- Slow event consumer, concurrent-turn rejection, disconnect during a turn,
  child death, hung close, and bounded termination.
- Identity and conversation mapping, no secret values in frames or logs.

### Slice C: real Go host

Deliver `lisa-go/host`, a minimal example host, and shared language-neutral
fixtures. Run every applicable A/B scenario through Python and Go clients.
Compare event ordering and typed outcomes, not terminal styling. Include at
least one packaged runtime launch, Unicode and multiline messages, a yield,
an approval continuation, and an interrupt.

The feature is not ready to declare v1 stable until Slice C passes. A Python
client talking to a Python bridge proves mechanics, not language neutrality.

## 12. Change control

Additive optional fields and capabilities may be introduced within v1 after
schema and conformance updates. Removing/renaming fields, changing outcome or
lifecycle semantics, weakening identity/approval handling, or changing frame
interpretation requires a new protocol major version.

The [Go Connector SDK contract](../GO_CONNECTOR_SDK.md) has an independent
version. A Host Protocol change does not implicitly change MCP server or MCP
broker contracts. Implementation PRs update the specification, schemas,
client/server mappings, and conformance fixtures together.

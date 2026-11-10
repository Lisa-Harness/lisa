# Host SDK architecture

> **Status:** Accepted architecture; implementation has not started. The
> [Host Protocol specification](SPEC/HOST_PROTOCOL.md) defines the initial
> delivery contract. The [design primer](HOST_SDK_PRIMER.md) explains the
> decisions and rejected alternatives.

## Purpose

Make it possible to build a Lisa host in any programming language while
keeping Lisa's turn loop, packs, policy, persistence, and runtime composition
in one implementation. A host owns its user interface, authentication,
deployment, secrets, and scheduling. It does not reimplement orchestration.

This document is the architectural guide for contributors and coding agents.
It defines ownership and dependency rules that remain true as implementation
proceeds. Use the specification for exact v1 wire behavior and acceptance
gates; do not infer a public API from Python implementation objects.

## Decision summary

1. Provide a language-facing **Host SDK**, not a native port of Lisa.
2. Run the existing Python runtime in a **long-lived child process** owned by
   the host application.
3. Connect the host and child with a **versioned, bidirectional protocol over
   stdio**. Streaming events travel in the opposite direction from turn
   requests.
4. Put the public contracts and client-side SDK in `sdk.host`; put the Python
   protocol server and runtime lifecycle adapter in `runtime.host_bridge`.
5. Start with **one child process per host session** and one active turn per
   session. Multiple processes may share a data directory under the existing
   runtime persistence and locking rules.
6. Prove the design with a real Go Host SDK and example host in `lisa-go`.
7. Keep `lisa-cli` in-process. It is a behavioral reference and conformance
   oracle, not a migration target.

## Existing boundaries that this design preserves

The current dependency direction is intentional:

```text
host application ──► runtime ──► sdk ──► kernel
                          │
                          └────► mcp_server
```

The detailed package rules are in [architecture and terminology](BOUNDARIES.md).
In particular:

- The kernel owns contracts and the model/tool loop, not process or UI wiring.
- The SDK owns reusable, dependency-light contracts and primitives. It may
  import the kernel and must never import the runtime.
- The runtime owns composition, process boundaries, environment-specific
  wiring, persistence writes, and async resource lifetime.
- Hosts own transport, rendering, authentication, and deployment.
- Packs remain trusted Python extensions discovered through entry points.

The Host SDK adds a wire boundary across the existing host/runtime seam. It
does not move that seam into the kernel or reverse the SDK/runtime dependency.

## Target shape

```text
┌─────────────────────────────────────────────────────────────┐
│ Application written in Python, Go, Rust, Java, .NET, etc.  │
│ UI · authentication · delivery · secrets · scheduling       │
└──────────────────────────┬──────────────────────────────────┘
                           │ idiomatic Host SDK API
┌──────────────────────────▼──────────────────────────────────┐
│ Host-language client                                        │
│ session API · typed events · process ownership · JSON-RPC   │
└──────────────────────────┬──────────────────────────────────┘
                           │ stdio, one child per session
┌──────────────────────────▼──────────────────────────────────┐
│ Python runtime.host_bridge                                  │
│ framing · dispatch · event adapter · lifecycle owner        │
└──────────────────────────┬──────────────────────────────────┘
                           │ public runtime composition calls
┌──────────────────────────▼──────────────────────────────────┐
│ Existing Python runtime                                     │
│ configuration · turns · approvals · state · MCP lifecycle   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    kernel / SDK / packs
```

The protocol is not MCP. MCP remains Lisa's tool/server interoperability
boundary. The Host Protocol carries runtime lifecycle, user turns, operator
signals, approvals, and transport callbacks. The optional Go MCP broker is a
separate implementation choice inside the runtime and is not a prerequisite.

## Module ownership

### `sdk.host`: public host contracts and Python client

Initial planned source layout:

```text
sdk/src/sdk/host/
    __init__.py
    protocol.py
    client.py
    stdio.py
    schemas/v1/
```

Responsibilities:

- Expose a small, typed Python API for starting a session, running a turn,
  receiving events, deciding approvals, sending operator signals, and closing.
- Define immutable protocol DTOs and typed error categories.
- Publish the canonical language-neutral JSON Schemas as package resources.
- Implement the client side of stdio framing, request correlation, ordered
  event delivery, and child-process ownership.
- Offer explicit process configuration for the Python executable, arguments,
  environment, and installation location.

The SDK client may depend on generic protocol libraries when justified. It
must not import `runtime`, compose a `HostRuntime`, read runtime globals, or
implement turn, routing, approval-policy, or pack behavior. Importing
`sdk.host` must not read the environment, spawn a process, register an entry
point, or perform I/O.

The Python DTOs are adapters to the wire schema, not the source of truth for
other languages. Schema/client parity tests are required.

### `runtime.host_bridge`: server-side runtime adapter

Initial planned source layout:

```text
runtime/src/runtime/host_bridge/
    __main__.py
    server.py
    transport.py
    mapping.py
```

Responsibilities:

- Provide the executable entered as `python -m runtime.host_bridge`.
- Negotiate the protocol, validate requests, and reject incompatible versions
  or invalid lifecycle transitions.
- Translate session options into the runtime's public composition API.
- Call `build_runtime`, `runtime.start`, `handle_message`, the typed approval
  API, and `runtime.aclose`.
- Adapt `TransportCallbacks` to typed protocol events.
- Consume a narrow runtime-owned terminal-result classifier that normalizes
  `HostCommandResult`, approval state, and yield data into an explicit outcome
  without inferring success from response prose.
- Keep startup, turn execution, and close in one runtime-owner async task and
  one event loop, as required by [host integration](HOST_INTEGRATION.md).
- Serialize protocol writes and preserve callback backpressure.

The bridge is a thin adapter. It must not grow a second dispatcher, turn loop,
policy engine, manifest compiler, or catalog implementation. A capability
that cannot be expressed through public runtime entry points needs a narrow
runtime API promotion, not remote access to private attributes.

### `kernel`: unchanged process and protocol boundary

The kernel must not learn about stdio, JSON-RPC, child processes, host-session
IDs, or protocol DTOs. Existing hooks, typed turn results, and policy behavior
remain the implementation underneath the runtime. Do not place Host Protocol
schemas in the kernel merely because both the SDK and runtime can depend on
it.

### `mcp_server` and packs: unchanged host boundary

First-party MCP servers continue to wrap SDK domain primitives. Packs continue
to contribute Python extensions and host commands. Neither layer imports the
host bridge or speaks the Host Protocol. A host command may be submitted as a
message through `handle_message`; that does not create a generic remote-admin
API.

### `lisa-go/host`: first foreign-language SDK

The Go repository adds a package separate from `connector/`:

```text
host/
    client.go
    process.go
    protocol.go
    events.go
```

It owns idiomatic Go session/event APIs, schema-compatible DTOs, child-process
startup, request correlation, cancellation, and shutdown. It does not load
Python manifests, execute packs, choose policy, or reimplement the runtime.
The existing Go Connector SDK continues to implement MCP servers; the planned
MCP broker remains independent.

Later language SDKs follow the same contract. They should be small clients
with generated or schema-tested types, not framework ports.

### `lisa-cli`: no conversion

`lisa-cli` continues to use `lisa-harness-runtime` directly in Python. It is a
one-shot process per command, has mature Rich rendering and lifecycle
handling, and gains no language interoperability from spawning another Python
process. Converting only its prompt path would also leave administrative and
inspection commands on a different boundary.

Do not add a production direct/wire fallback switch or make the CLI a partial
wire client. Use its observable behavior to construct parity fixtures and
expected results. A future conversion requires a separate product decision
about process isolation or remote runtime deployment, plus a complete protocol
surface for every feature being moved.

## Stable contracts versus implementation details

The public wire surface is deliberately curated. It includes:

- Session configuration and authenticated identity values.
- Turn input, typed events, and terminal outcomes.
- Exact-action approval requests and decisions.
- Cooperative operator signals.
- Protocol errors, lifecycle state, and version/capability negotiation.

It does not expose:

- `HostRuntime`, `HostConfig`, `HostCommandResult`, or kernel classes as wire
  types.
- Arbitrary Python callback dictionaries as an undocumented stable API.
- Python exception objects, tracebacks, module paths, or entry-point names.
- Generic method invocation, remote attribute access, or tool execution.
- Runtime resource collections, MCP pool internals, or persistence internals.

Runtime and SDK types may inform a mapping, but schema changes are reviewed as
public API changes. Unknown additive fields are ignored by clients; they are
not an excuse to serialize an entire internal object graph.

## Lifecycle and concurrency

Each session has one child process, one event loop, one runtime-owner task,
and one mutable `HostRuntime`. Session start and close are balanced on that
owner task. Only one turn or approval continuation is active at a time.
Concurrent turn admission fails explicitly rather than queueing implicitly.

Turn requests remain outstanding while event notifications stream. Every
operation has a client-supplied ID and every event identifies its operation.
Monotonic sequence numbers establish order. A terminal result ends an
operation exactly once, including error and cancellation paths.

Operator signals are cooperative. Process termination is the host's final
cleanup mechanism after graceful interruption and close fail; it is not the
normal meaning of cancel. EOF, child death, malformed input, and half-closed
pipes must not leave a success-shaped turn or orphaned grandchildren.

Cross-process conversation behavior remains the runtime's responsibility.
Distinct sessions can share `data_dir`; distinct `conversation_id` values
isolate conversation-scoped state where supported. The bridge adds no global
scheduler, background timer, or cross-session pool.

## Identity, secrets, and trust

The parent host is trusted to derive identity from its authenticated context.
It supplies workspace, user, principal, conversation, surface, and memory
scope as explicit values. Model text cannot set those fields. Missing optional
identity fields use documented runtime defaults only for trusted single-user
hosts; they are not a multi-tenant authorization scheme.

The child runs with the host user's OS privileges. Stdio provides no sandbox,
and installed packs remain trusted code. The initial transport is local
pipes, not a listening socket or remote daemon. A future network transport
needs a separate authentication and authorization design.

Provider credentials and other secrets stay in the child environment or
host-owned secret configuration. They never appear in protocol messages,
request logs, error details, or event payloads. Protocol diagnostics contain
messages, not credentials, raw environments, prompt contents, or tracebacks.

## Versioning and documentation authority

The Host Protocol has its own version, independent of the MCP Connector
Contract and any MCP-broker protocol. Version negotiation is mandatory;
incompatible peers fail rather than silently degrading.

Authority is divided by concern:

| Concern | Canonical source |
| --- | --- |
| Package ownership and invariants | This document and [BOUNDARIES.md](BOUNDARIES.md) |
| Exact v1 messages and behavior | [SPEC/HOST_PROTOCOL.md](SPEC/HOST_PROTOCOL.md) and its schemas |
| Python host composition | [HOST_INTEGRATION.md](HOST_INTEGRATION.md) |
| Design rationale and alternatives | [HOST_SDK_PRIMER.md](HOST_SDK_PRIMER.md) |
| MCP server contract | [GO_CONNECTOR_SDK.md](GO_CONNECTOR_SDK.md) |

Schemas and the specification land before client/server implementation.
Behavioral guides describe only shipped functionality; until then, links to
this design must retain its implementation status.

## Delivery order and gates

1. **Contract:** land the specification, canonical schemas, and conformance
   fixtures. Resolve ambiguous lifecycle and error mappings before coding.
2. **Python vertical slice:** implement `sdk.host` and `runtime.host_bridge`
   together. Prove initialize, start, `/help`, close, and failure behavior.
3. **Turns and control:** add scripted model turns, ordered streaming,
   structured outcomes, approval decisions, signals, and disconnect cleanup.
4. **Go proof:** implement `lisa-go/host` and an example host using the real
   Python sidecar. No Python-specific test double counts as this gate.
5. **Cross-language conformance:** run common fixtures against Python and Go
   clients, including malformed frames, version mismatch, interrupts, yields,
   approvals, and process death.
6. **Release readiness:** document installation and environment ownership,
   test packaged rather than only source-tree launches, and mark a protocol
   version stable only after both implementations pass.

No milestone requires changing `lisa-cli`, porting the kernel, implementing
the Go MCP broker, or supporting multiple wire transports.

## Review checklist for implementation changes

- Does the change preserve `sdk` independence from `runtime`?
- Is new behavior implemented in the existing runtime rather than the bridge?
- Are wire DTOs curated and schema-tested instead of Python-object dumps?
- Are start, turn, and close still owned by one async task and event loop?
- Can every turn end in exactly one observable terminal outcome?
- Do approval identity, yield state, operator exits, and errors survive the
  mapping without relying on rendered prose?
- Are callbacks ordered and backpressured, with no stdout logging?
- Are secrets and authenticated identity handled at the host boundary?
- Does the same conformance scenario pass through a non-Python client?
- Have documentation and schemas changed in the same commit as the contract?

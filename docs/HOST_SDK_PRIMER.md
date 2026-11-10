# Host SDK design primer

> **Status:** Design rationale for a planned feature. For binding implementation
> rules, read the [architecture](HOST_SDK_ARCHITECTURE.md) and
> [Host Protocol v1 specification](SPEC/HOST_PROTOCOL.md).

## The question

How can someone write a Lisa host in Go, Rust, Java, .NET, TypeScript, or
another language without porting Lisa's runtime to that language?

The practical answer is to leave orchestration in Python and give the host a
small client SDK. The host launches a Python runtime child, sends it turns and
control operations, and receives structured events. The interesting design
work is choosing the boundary and keeping it narrow.

## The insight from `lisa-go`

Lisa already has language interoperability, but in the opposite direction.
The [Go Connector SDK](GO_CONNECTOR_SDK.md) lets Python launch a Go program as
an MCP server. Go supplies tools and storage; Python still runs Lisa.

The proposed Go MCP broker changes which implementation manages MCP child
processes and connections. It does not let a Go application host a Lisa turn.
Configuration, policy, packs, models, and approvals would still be Python.

That distinction led to three separate contracts:

| Contract | Question it answers |
| --- | --- |
| MCP Connector Contract | How does a program provide an MCP server to Lisa? |
| MCP broker protocol | Which process manages MCP pool mechanics? |
| **Host Protocol** | **How does an application run and observe Lisa?** |

Only the third solves the host problem. Reusing the name or version of the
first two would hide important differences.

## Why not embed CPython directly?

Many languages can embed CPython, but that convenience is mostly at the call
boundary. A production host would also inherit the GIL, Python threading and
event-loop rules, native-extension compatibility, virtual-environment setup,
and process-wide lifecycle constraints.

Lisa's runtime makes in-process embedding especially unattractive as the
common API. MCP async contexts must start, run, and close on the same task and
event loop. Pack installation and discovery have their own environment. A
native extension failure could terminate the host.

A child process makes those constraints explicit and contained. Every language
already knows how to launch a process and exchange bytes. The cost is protocol
design and process management; the benefit is that each language SDK stays
small and failures have a clear owner.

This is not a claim that FFI can never work. It is a decision not to make FFI
the portability layer or maintain direct and embedded behavior as two public
runtime modes.

## Why stdio and JSON-RPC?

The interaction is local and bidirectional:

```text
host:    start session ──► run turn ──► signal ──► close
runtime: ready          ──► events  ──► outcome ──► closed
```

Stdio gives the child no listening port and no ambient network identity. The
parent owns its lifetime. JSON-RPC supplies request IDs, responses, typed
errors, and notifications without inventing an RPC envelope.

Newline-delimited JSON keeps the first implementation inspectable and easy to
conformance-test. A real schema matters more than the serialization choice:
without one, each SDK gradually accumulates guesses about field names, nulls,
numbers, and missing values.

The protocol therefore uses JSON-RPC over framed stdio, with explicit limits
and failure behavior. It is not a generic RPC service. It exposes only the
operations a host needs.

## Why does part of this belong in `sdk`?

The word "SDK" describes the developer-facing abstraction: start a session,
run a turn, observe events, decide an approval, interrupt, and close. Those
contracts and the client-side stdio implementation fit the SDK's role as the
reusable, dependency-light layer.

The server-side adapter does not. Existing architecture permits
`runtime -> sdk`, not `sdk -> runtime`. Only the runtime can compose
`HostRuntime`, resolve its configuration, and own MCP lifecycle.

The split is therefore:

```text
sdk.host             what a host sees; how a client speaks the protocol
runtime.host_bridge  how Python fulfills that protocol using the runtime
```

`sdk.host` never imports the runtime. `runtime.host_bridge` may import SDK
contracts. The kernel sees neither. This gives the SDK a real host-facing role
without moving process ownership into the wrong package.

## What is actually being standardized?

Not Python objects. The protocol standardizes host-visible concepts:

- A session configuration.
- An authenticated identity supplied by the host.
- A message or runtime host command.
- Structured activity and streaming events.
- A terminal outcome such as completed, yielded, awaiting approval, stopped,
  interrupted, or failed.
- An exact approval decision.
- Cooperative operator signals.

This is why `HostConfig` and `HostCommandResult` cannot simply become JSON
schemas. They are useful Python APIs, but they contain implementation choices
and evolving fields. Arbitrary callback dictionaries have the same problem.
If every internal field is serialized, an internal refactor becomes a breaking
wire change.

The bridge deliberately maps public runtime values into versioned DTOs. When
a host needs information that is unavailable through public runtime APIs, we
promote a narrow capability or add a curated projection. We do not provide
remote attribute access or a generic `call` method.

## The hard parts are semantics, not transport

A request/response turn looks simple until streaming, cancellation, and
approval are included. The design makes several choices to avoid ambiguity.

### One child, one runtime, one active operation

That matches the existing mutable runtime and async lifecycle. A second turn
is rejected rather than silently queued. Parallel work uses separate runtime
processes under the existing persistence and locking rules.

This is intentionally less flexible than a daemon. It avoids building session
routing, authentication, resource isolation, and multi-tenancy before a single
host works correctly.

### Events precede one terminal result

The turn request remains outstanding while notifications arrive. Every event
has an operation ID and sequence number. Exactly one terminal response ends
the operation, including failures and interruption.

A streamed response event is not itself proof of success. A host must inspect
the outcome because a turn can yield a question, pause for approval, stop,
terminate, or fail.

### Cancellation is cooperative

`interrupt` records the existing operator signal. It is not equivalent to
killing the process. The runtime gets an opportunity to preserve durable state
and clean up its resources. The parent still has a bounded last-resort cleanup
path if the child will not exit.

### Approvals stay exact and runtime-owned

A client sends the returned request ID and a decision. It does not submit an
approved tool invocation or ask the model to approve something in prose. The
runtime rebuilds and validates the saved invocation. This preserves policy
rather than moving trust to the SDK.

### Identity is configuration, not prompt text

The host derives user, principal, workspace, conversation, and surface scope
from authentication. The turn message cannot change them. This matters more
than framing: a perfectly valid protocol can still cross a privacy boundary if
identity is treated as model-supplied metadata.

Local stdio is not a security sandbox. Packs are trusted Python code and the
child has the user's OS permissions. A future network transport would need a
separate security design.

## Why the first real host should be Go

A Python client and Python bridge can prove framing and lifecycle, but they
can accidentally share assumptions about JSON, exception shapes, dictionaries,
and null handling. A Go client forces the contract to stand on its own.

The desired evidence is a shared scenario suite: Python and Go clients drive
the same runtime, receive equivalent ordered events, and report equivalent
outcomes. The comparison is semantic, not a requirement that both hosts render
the same terminal UI.

Only after that suite passes should v1 be described as language-neutral.

## Why `lisa-cli` stays as it is

The CLI already uses the intended Python host seam directly. It is a one-shot
process per command and has mature behavior for rendering, approvals, exit
codes, bounded teardown, model information, and administration.

Moving only `lisa prompt` to the new protocol would require a large enough
surface to preserve those details while leaving other commands in-process. A
hybrid would test neither boundary cleanly. Adding a runtime/admin RPC broad
enough for every CLI command would undermine the goal of a small host API.

There is also little user benefit: the CLI would launch Python so that Python
could launch the same runtime. That adds process and serialization failure
modes without enabling a new language.

Keeping the CLI in-process is therefore not a temporary lack of commitment.
It is the default architecture. Its behavior remains valuable as a reference
for parity scenarios. A later conversion would need a separate reason, such
as remote runtime deployment or deliberate process isolation, and a complete
migration design.

## Trade-offs accepted in v1

| Choice | Benefit | Cost accepted |
| --- | --- | --- |
| Python child process | No framework ports; failure isolation | Python installation and packaging remain required |
| Stdio JSON-RPC | Local, portable, debuggable | Manual framing and backpressure implementation |
| One active operation | Matches runtime lifecycle; simple correctness | No concurrent turns per session |
| Curated DTOs | Stable language-neutral contract | Mapping work and schema maintenance |
| Rich projected events | Useful native host rendering | Event projection needs explicit parity tests |
| No CLI migration | Avoids admin protocol and duplicate path | CLI does not exercise the SDK in production |
| Go as first external SDK | Real proof of language neutrality | Cross-repository release coordination |

## Decisions deliberately left to implementation

The specification fixes v1's external behavior, but it does not pretend every
runtime detail already exists. Two small runtime additions are prerequisites,
not reasons to weaken the boundary:

- A public configuration seam for authenticated identity that does not match
  today's single-user derivation. The bridge must not silently discard a
  requested principal, workspace, or memory scope.
- A runtime-owned classifier for the existing approval and yield projections.
  The bridge should not duplicate outcome rules or assume a nested approval
  shape that the current result does not have.

The canonical JSON Schemas will make the event and approval field lists
mechanical before the server and client are implemented. Performance targets,
installation UX, and additional administration APIs remain separate decisions;
there is no measured basis for promising them in v1.

## What would make us revisit the design?

Revisit the process model if measurements show that per-session startup or
memory use prevents intended deployments and lifecycle isolation cannot solve
it. Revisit stdio if an authenticated remote-runtime product becomes a real
requirement. Revisit the operation surface when multiple external hosts need
the same curated diagnostics or administration capability.

Do not revisit it merely to avoid writing a mapper, expose a convenient Python
internal, or make the CLI serve as the first demo. The design succeeds when a
new host is mostly UI and lifecycle code, while Lisa's behavior remains in one
place.

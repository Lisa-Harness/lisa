# Architecture and boundaries

Lisa separates orchestration from capability content, reusable domain logic,
and application transport. The [project README](../README.md#architecture)
shows the dependency diagram.

## Package ownership

| Layer | Owns | Must not own |
| --- | --- | --- |
| [Kernel](../kernel/README.md) | Typed contracts, model/tool loop, mode/workflow FSMs, authorization and completion mechanisms, session compilation | Runtime/SDK imports, UI, application workflows, deployment |
| [SDK](../sdk/README.md) | Catalog/content scanners, reusable backends, provider gateways, CEL model router, memory/storage/thread-read primitives | Host transport and lifecycle, runtime imports, private application policy |
| [Packs](../sdk/packs/README.md) | Modes, operations, tools, prompts, sessions, extension contributions | Kernel state ownership, transport, cross-pack coupling |
| [Runtime](../runtime/README.md) | Composition, resolved config, commands, MCP/proxy lifecycle, context assembly, thread writes, durable services | Rich/Typer/chat SDKs, product-specific workflow logic |
| [MCP servers](../mcp_server/README.md) | Protocol adapters, tool decorators, CLI entry points, server-local state | Domain logic duplicated from SDK, host imports, cross-server imports |
| Host application | Authentication, transport callbacks, delivery, secrets, deployment, scheduling | Duplicated orchestration or a hidden policy bypass |
| [packstest](../packstest/README.md) | Source-based deterministic pack checks and isolated test doubles | Runtime production wiring |
| [e2etest](../e2etest/README.md) | Whole-runtime scenarios and evidence-based oracles | Product behavior or a substitute for unit tests |

Packs should import kernel/SDK and package-local code, not runtime internals
or another pack. Shared contracts belong on neutral ground. Some existing
bundled integrations contain legacy lazy runtime coupling; that is not an
extension API to copy.

Runtime wires callbacks across layers; callback/MCP communication does not
reverse import ownership. First-party servers can run in-process or as
subprocesses without moving their domain logic out of the SDK.

## Terminology

| Concept | Meaning |
| --- | --- |
| **Host** | Concrete application: terminal, chat, editor, or headless integration |
| **Runtime** | Transport-neutral composition used by a host |
| **Pack** | Installable capability/content contribution discovered through entry points |
| **Mode** | Instructions, capability allowlists, routing metadata, and lifecycle policy for a kind of work |
| **Stage** | One state of a workflow mode's FSM |
| **Operation** | Typed Python behavior or a prompt-authored task with host/tool access |
| **Tool** | A callable capability, supplied by a pack, kernel interception, or MCP server |
| **Turn** | Processing one message, potentially making multiple model/tool calls and handoffs |
| **Session plan** | Immutable, compiled control plane from strict Markdown/YAML authoring |
| **Model router** | Selects a model per iteration; distinct from routing work between modes |
| **Effect receipt** | Dispatch evidence used by completion contracts, not the model's claim of success |
| **WorkProduct** | Owned, durable unit of approvable work; checked-commit tasks verify its revision/tree |
| **Thread** | Lossless conversation and evidence store |
| **Memory** | Scoped, derived learning available for recall |
| **Slice** | Opportunistic, budgeted housekeeping at turn boundaries or explicit invocation |

## One message through the system

1. The transport supplies authenticated context and a message to the runtime.
2. Host control commands are dispatched without a model. Other input resolves
   to an active mode, preserving the current mode across conversational turns.
3. The runtime assembles instructions, scoped context candidates, history,
   and capability catalogs. The kernel fits the complete model request.
4. The kernel iterates model completions and authorized dispatch. Operations
   and pack tools execute in-process; external tools use the MCP pool.
5. `transition_stage` advances workflow state, changes mode, or forks
   independent child work. Delegation handles sequential parent/child work.
6. Effectful actions pass exact-action approvals/policy. Pauses, yields,
   blocks, and budget exits are distinct from successful completion.
7. Completion gates evaluate receipts and, when configured, the owned
   checked-commit artifact. Runtime hooks persist evidence/usage and perform
   bounded slices.

For detail, read [mode routing](ROUTER_DESIGN.md), [context lifecycle](CONTEXT_LIFECYCLE.md),
[token budgets](TOKEN_BUDGET.md), [approval policy](POLICY_GUIDE.md), and
[memory architecture](MEMORY_ARCHITECTURE.md).

## Host responsibilities and privacy

Use the [integration checklist](HOST_INTEGRATION.md#integration-checklist).
A host owns provider credentials, async resource lifetime, transport,
authentication and identity, authorized scope, proactive delivery targets,
and deployment/scheduling.

`SurfaceContext` carries `surface_type`, `surface_id`, and `memory_scope`.
These fields must be enforced in storage keys, retrieval filters, and delivery
routing. Shared surfaces must not receive personal context without explicit,
audited sharing semantics. Prompt wording is not a privacy control.

Default guards are pass-through. Installed Python packs are trusted code.
Approval policy is not OS containment, and profile/data-root isolation is
not a multi-tenant guarantee. See [SECURITY.md](../SECURITY.md).

## External ecosystem

The terminal host lives in [lisa-cli](https://github.com/Lisa-Harness/lisa-cli);
public integrations are maintained in
[lisa-packs](https://github.com/Lisa-Harness/lisa-packs).
Their versions, dependencies, support promises, and release workflows are
owned by those repositories. Sibling checkout paths are development
conveniences, not prerequisites for consuming the framework documentation.

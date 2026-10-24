# Memory architecture

Memory is a pack-owned capability composed by the runtime, not an implicit
kernel database or the same thing as conversation history.

## Layers

| Layer | Responsibility |
| --- | --- |
| Kernel contracts | Scoped contexts, typed memory results, execution/slice seams |
| [SDK memory](../sdk/src/sdk/memory/) | SQLite store, retrieval, embeddings integration, backup primitives |
| [Memory MCP adapter](../mcp_server/src/mcp_server/memory/) | Transport, resolved configuration, backend access |
| [Memory pack](../sdk/packs/memory/) | Learning/recall operations, capability manifest, consolidation slice |
| Runtime | Capability selection, MCP lifecycle, ambient identity, pre/post-turn wiring |
| Host | Authentication, credentials, authorized scope, retention and delivery |

The bundled memory pack declares its MCP server in
[mcp_servers.yaml](../sdk/packs/memory/src/sdk/packs/memory/mcp_servers.yaml).
The runtime selects one `memory` capability implementation and publishes the
required `memory.add`, `memory.search`, and `memory.list` tools under stable
IDs. Provider-specific argument and result adaptation belongs in the selected
pack's MCP server. Optional tools such as `memory.get` and `memory.retire` may
be declared by an implementation, but are not part of the swappable core
contract and may be absent from other providers.

Installation does not imply activation. Enable/select the memory pack in the
host/session and resolve duplicate owners with `providers.capabilities.memory`
when needed. Core and introspect are host-mandatory; memory is optional.

## Thread versus memory

- **Thread:** the lossless record of turns, transcripts, receipts, and addressed
  results. It is evidence of what happened.
- **Memory:** derived, scoped learning used for later recall. It may be
  classified, consolidated, disputed, expired, or replaced.
- **Context:** the subset admitted into a particular request, including recalled
  memory and thread-derived task state.
- **Recovery:** write-ahead dispatch state, not learned memory. It answers
  whether an action completed or may need operator resolution.

Do not overwrite the thread with a summary to save prompt space. Bounded
context and compressed thread segments are separate mechanisms.

## Learning and recall

Runtime hooks can recall scoped memories before a turn and capture learning
after it when the backend and operations are available. The memory pack
provides operations for recall, synthesis, dispute, semantic consolidation,
and mining sealed thread segments.

The memory model distinguishes short-lived tiers (`daily`, `weekly`), longer
lived learning (`long_term`, `permanent`), and salience. A salient fact is
not automatically globally visible or guaranteed admission into every prompt.
Retrieval remains subject to scope and context budgets.

Embedding failure is logged at the store's best-effort boundary; lexical
memory access remains available. Configure a supported embedding endpoint
and dimensions when vector retrieval is required. Do not equate lexical
fallback with a healthy embedding service.

## Consolidation

The memory pack's slice provider uses episodic pressure and the configured
slice slot/interval to decide when to consolidate. It can mine newly sealed
thread segments into facts with thread provenance, tracking progress in
sidecars. It does not mutate the thread.

Tune `context.memory.consolidation` in the
[session schema](SESSION_CONTEXT_AUTHORING_GUIDE.md#contextmemoryconsolidation):
enablement, trigger, pressure/hard thresholds, interval, batch size, record
pressure denominator, preferred model, and operation. Mining and synthesis
can make auxiliary model calls even when the user's main request looks simple.

Housekeeping is opportunistic and budgeted, not a continuously running
scheduler. Disabling consolidation does not disable all recall, capture,
or explicit memory operations.

## Scope, privacy, and backup

Carry the authenticated workspace, user, surface, and memory scope through
storage and retrieval. Shared surfaces must not receive personal learning
without explicit host-enforced sharing semantics. Prompt instructions alone
cannot enforce this.

Use SQLite-aware backups and include the associated configuration and thread
provenance when needed. A thread backup is not a memory backup, and deleting
a thread entry does not erase derived memories or provider copies.
See [storage and data handling](STORAGE.md) for retention and restore guidance.

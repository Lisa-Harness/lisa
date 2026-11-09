# Memory architecture

Memory is a pack-owned capability composed by the runtime, not an implicit
kernel database or the same thing as conversation history.

## Layers

| Layer | Responsibility |
| --- | --- |
| Kernel contracts | Scoped contexts, typed memory results, execution/slice seams |
| [SDK memory](../sdk/src/sdk/memory/) | SQLite store, retrieval, embeddings integration, read-only mode |
| [Memory MCP adapter](../mcp_server/src/mcp_server/memory/) | Transport, resolved configuration, backend access |
| [Memory pack](../sdk/packs/memory/) | Capture/recall/dispute operations, capability manifest, mining slice |
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
provides operations for capture, recall, and dispute, plus a mining slice for
sealed thread segments.

The memory model distinguishes short-lived tiers (`daily`, `weekly`), longer
lived learning (`long_term`, `permanent`), and salience. A salient fact is
not automatically globally visible or guaranteed admission into every prompt.
Retrieval remains subject to scope and context budgets.

Embedding failure is logged at the store's best-effort boundary; lexical
memory access remains available. Configure a supported embedding endpoint
and dimensions when vector retrieval is required. Do not equate lexical
fallback with a healthy embedding service.

### Write-time dedup (create / update / skip)

Capture is an LLM-guided create / update / skip decision, not an insert.
`synthesize_and_store_learning` and
`mine_thread_segment` search the scoped store for similar existing records
before distilling, hand them to the distiller alongside the new content, and
persist a per-fact decision:

- **create** — new durable knowledge; a new semantic record.
- **update** — an existing record covers the knowledge but the new content
  refines or corrects it; a superseding record is written with
  `supersedes_id`, and the backend retires the scoped original in the same
  transaction — supersession is atomic, so a successful add means the
  correction is the only active claim. The retired original stays
  queryable with `status=None`, out of the active recall pool.
- **skip** — already stored; nothing is written.

The store's `retire` (not `delete`) preserves the audit trail, so
supersession is temporal: the thread and the retired records remain the
evidence, the active pool carries the current claim. `dispute_memory` uses
the same atomic supersession seam for corrections. The search-then-write

Captures are stored as `memory_kind="semantic"` — they are distilled,
self-contained facts by construction. The store has no remaining episodic
writer.

### Recall

Recall is a two-stage rank with one tokenizer (`sdk.memory.lexical`): the
backend surfaces a candidate pool (semantic KNN when embeddings are
available, otherwise recent-rows lexical overlap), then the pack re-ranks
candidates relevance-first — a record must clear a lexical relevance floor to
lead; salience breaks ties among relevant records only. Below-floor
candidates still surface (a paraphrase can share no tokens with the query)
but never lead over a relevant one. The operation asks the backend for
candidate headroom above the final evidence limit so the re-ranker sees the
full pool before truncating.

## Mining

The memory pack's slice provider mines each newly sealed thread segment into
durable semantic records during the configured slice slot (`post_turn` by
default). Progress is tracked in a `mine.marker` sidecar beside the durable
data root; the marker is positional, so a rebuilt or re-sealed store cannot
strand it mid-scan, and the write-time dedup makes even a full re-mine
non-destructive. Mining is bounded (`max_segments_per_slice`) and never
mutates the thread — the sealed segment is the archive.

Tune `context.memory.mining` in the
[session schema](SESSION_CONTEXT_AUTHORING_GUIDE.md#contextmemorymining):
enablement, trigger slot, preferred model, operation, and the per-slice
segment cap. Mining can make auxiliary model calls even when the user's main
request looks simple.

Housekeeping is opportunistic and budgeted, not a continuously running
scheduler. Disabling mining does not disable recall, capture, dispute, or
explicit memory operations.

## Scope, privacy, and backup

Carry the authenticated workspace, user, surface, and memory scope through
storage and retrieval. Shared surfaces must not receive personal learning
without explicit host-enforced sharing semantics. Prompt instructions alone
cannot enforce this.

The framework does not dump memory backups: neither the server nor the SDK
writes point-in-time copies on boot. Use SQLite-aware backups and include the
associated configuration and thread provenance when needed. A thread backup is
not a memory backup, and deleting a thread entry does not erase derived
memories or provider copies. See [storage and data handling](STORAGE.md) for
retention and restore guidance.

## Read-only stores

`SqliteMemoryStore(readonly=True)` opens the database with SQLite `mode=ro`,
refuses to create one when absent, and raises `ReadOnlyMemoryStoreError` from
`record_memory`, `delete_memory`, `retire_memory`, and `update_salience`.
Recall still works; its access-tracking update is skipped rather than turning a
read into a failure. `lisa-memory --readonly` wires this through, so a host can
hand scoped recall to a client that must not mutate the store. Read-only is a
store-level guarantee, not a substitute for identity enforcement: the caller
still supplies the scope fields.

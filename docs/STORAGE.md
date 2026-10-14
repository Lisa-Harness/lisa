# Storage and data handling

Lisa separates persisted evidence, learned memory, execution state, and
provider-visible context. They have different lifetimes and must not be
treated as interchangeable caches.

## What is persisted

Paths below are relative to the [resolved data root](CONFIGURATION.md#data-directory-and-profiles).
Hosts and sessions can override some locations.

| Category | Data | Handling |
| --- | --- | --- |
| `config/` | Session instructions, model roster, approval policy, overlays | Back up; review changes as executable configuration |
| `state/` | Approvals, recovery records, work products, pack state, model proposals | Back up with durable data; preserve identity and revision relationships |
| `data/threads/` | Turn entries, compressed segments, index/dictionary, result objects | Retain the complete store, not only the active JSONL |
| `data/memory.db` | Scoped learning and recall data | Use SQLite-aware backup methods; see [memory](MEMORY_ARCHITECTURE.md) |
| `data/files/` | Files exposed through the storage sandbox | Treat as user artifacts, not disposable scratch |
| `cache/` | Regenerable fetch/cache data | Confirm ownership and stop dependent work before cleanup |
| `logs/` | Runtime, MCP, and token-usage records | Apply access controls and a deployment-specific retention policy |
| `run/` | Session scratch, sandbox facts, jobs, execution files | Check for active jobs and recoverable work before cleanup |

The [layout constants](../sdk/src/sdk/layout.py) define categories, not a
universal retention or garbage-collection service. There is no claim that
all persisted objects are automatically expired or encrypted.

## Thread evidence versus context

The thread is the lossless conversation/evidence record. Active segments seal
to zstd; sealing is not summarization or deletion. Tool results and exact
model requests can live in content-addressed result objects referenced from
entries. Copying only `.current.jsonl` loses referenced evidence and older turns.

The model sees a bounded projection: selected history, full resident tool
results, and references/previews under pressure. Memory consolidation learns
from evidence without rewriting the thread. See [context lifecycle](CONTEXT_LIFECYCLE.md)
and the [thread specification](SPEC/THREAD_STORE.md).

## Storage sandbox and direct I/O

Model-requested file access should use the storage tool seam. The
[SDK storage backend](../sdk/src/sdk/storage/_backend.py) owns sandbox-relative
paths and deny-glob checks; [lisa-storage](../mcp_server/README.md) exposes
that backend. This file sandbox is not an OS sandbox for arbitrary Python
pack code or shell processes.

Host-owned persistence and backend implementations sometimes need direct I/O.
The repository's [direct-I/O check](../scripts/check_direct_io_policy.py) is
an AST-based contributor gate:

```sh
uv run python scripts/check_direct_io_policy.py
```

- In-tree runtime code is scanned for write/mutation primitives. A detected
  call must use the storage seam or have an explicit file-and-snippet
  allowlist entry.
- Tests are excluded according to the scanner's path rules. Shipped testing
  helpers are not automatically excluded just because their name is `testing`.
- When present, sibling external-pack checkouts are scanned for reads as well
  as writes. These findings are currently advisory, not a passing security
  certification.
- The scanner is not a complete information-flow/type analysis, an OS
  containment mechanism, or proof that every I/O path is covered.

When moving an allowlisted writer, review the actual trust boundary, update
the new call site, and remove obsolete entries. Snippets match the stripped
source line at the AST call's start; a renamed variable or multiline rewrite
can invalidate the match. Do not widen the allowlist simply to silence CI.

**Maintenance limitation:** unused allowlist snippets are not rejected by the
current scanner. A green result alone does not establish that every carve-out
is still live or necessary. Review dead keys/snippets when changing the
allowlist. This preserves the actionable finding from the former I/O audit
without retaining its obsolete patch or CI-status snapshot.

## Privacy and retention

Threads and request snapshots may contain exact prompts, tool arguments,
private retrieved content, and provider continuation material. Memory contains
derived facts; logs and scenario bundles may expose additional details.

- Keep data directories and `.env` files out of Git and public bug reports.
- Restrict filesystem access; use encryption and secure backups according
  to the deployment's requirements.
- Thread read-time redaction changes what a reader sees. It is **not**
  at-rest sanitization, encryption, or secure deletion of original bytes.
- Scope identifiers are not authentication. Hosts must enforce authorized
  workspace/user/surface/memory scope and delivery routing.
- Establish a retention/deletion policy covering threads, result objects,
  memory, logs, external providers, and backups. Removing one visible file
  does not erase copies or derived memories.

## Back up and restore

1. Resolve the actual data root, thread overrides, session-selected roster,
   and packs environment. Record the framework/host/pack versions.
2. Stop new work and close the runtime through its owner task. For an online
   memory backup, use the backend's SQLite-aware backup path rather than
   copying a live database file without its journal/WAL.
3. Copy configuration, durable data, and related state together. Include
   all thread segments, indexes, dictionaries, and `.results` directories.
   Do not publish the archive.
4. Restore into a separate location with the matching software and reviewed
   pack sources. Validate roster/session loading, thread reference expansion,
   memory access, and pending approvals/recovery before returning to service.

Restoring old approvals or work products does not authorize replaying an
effect. Verify the current external state and invocation identity first.
Do not delete active state or backups as a troubleshooting shortcut.

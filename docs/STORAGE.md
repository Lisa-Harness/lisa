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
| `logs/` | Runtime, MCP, token-usage, decision, and approval-ledger records | Apply access controls and a deployment-specific retention policy; the approval and decision ledgers are audit evidence — prefer export-and-archive over deletion |
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

The storage server is the model-facing implementation. Its tools publish as
the implementation-independent `storage.*` capability. Routine read/write
tools, including `storage.move_file`, are ambient for modes that accept the
default capabilities. Destructive directory, delete, trash-purge, and batch
tools remain opt-in: a mode or host must name their `storage.*` ids before
the model can call them.

| Operation | Availability | Success receipt |
| --- | --- | --- |
| `move_file(source, destination, expected_sha256="", create_parents=false)` | Ambient | `ok`, `source`, `destination`, `bytes`, `sha256`, `durable` |
| `move_directory(source, destination, create_parents=false)` | Opt-in | `ok`, `source`, `destination`, `durable` |
| `delete_directory(path, recursive=false, trash=true, expected_empty=false)` | Opt-in | `ok`, `path`, `removed_files`, `removed_dirs`, optional `trashed_to` |
| `purge_trash(older_than=null, limit=100)` | Opt-in | `ok`, `purged`, `has_more` |
| `delete_files(paths)` | Opt-in | `ok`, `count`, `succeeded`, `failed`, `results` |
| `make_dirs(paths)` | Opt-in | `ok`, `count`, `succeeded`, `failed`, `results` |
| `move_files(pairs, create_parents=false)` | Opt-in | `ok`, `count`, `succeeded`, `failed`, `results` |

### Write contract

`write_file` stores **non-empty text** content newline-terminated. A POSIX
text file ends with a newline, and a file whose final byte is missing reads
back as if its last line were cut off — `cat` glued the shell prompt onto
the last line of a session plan on 2026-10-19. The receipt reports
`trailing_newline_added` so the caller knows the stored bytes differ from
the bytes it sent, and `bytes` reports the size of the bytes actually
stored. Empty text is the exception: an empty write stays an empty file
and gains no newline.

The newline rule applies to text only, and the input is text — there is no
byte-oriented API. A path with a binary suffix (`.png`, `.db`, `.pdf`, …)
only suppresses newline normalization for an otherwise valid text string;
the payload is still UTF-8 encoded. This is not a byte-for-byte
arbitrary-binary round trip. Content that is not valid UTF-8 — a lone
surrogate, for example — is rejected with an error rather than written.

`append_file` appends exactly the bytes it is given. Its receipt reports
`bytes` (the resulting file size) **and** `appended_bytes` (what this
call added). The size alone cannot tell an append that moved bytes from
an empty append that moved none, and a receipt that claims a write
happened when no bytes moved is not proof of an effect.

### Exact edits

`edit_file` changes one region of an existing file and leaves every other
byte as it was on disk. Rewriting a whole document to change one line
means reconstructing the rest from the caller's own memory, which is the
wrong basis for a document that exists to record evidence. The primitive
removes that reconstruction entirely.

- **The match is exact.** Indentation, line endings, and surrounding text
  must be byte-identical. Exactness is the safety property: a fuzzy match
  would rewrite a region the caller never saw.
- **Ambiguity is refused.** `old_string` must occur once, or the caller
  passes `replace_all`. A multi-site match returns the candidate line
  numbers and writes nothing, so the caller adds context instead of the
  tool guessing a site.
- **The revision guard is optional and checked at read time.** `stat_file`
  reports `sha256`; passing it as `expected_sha256` makes the edit fail if
  the file already changed when it is read. The check and the replacement
  are separate steps with no lock or compare-and-swap between them, so a
  concurrent writer can still modify the file in that window. Omitting the
  hash is supported: the edit then applies without a revision check.
- **The write is atomic; cleanup is best-effort.** New bytes are staged
  beside the target and swapped in with `os.replace`, so an interrupted
  edit leaves the original intact instead of truncated. The staged
  temporary file is removed on failure paths, but a cleanup error is
  suppressed so it cannot mask the primary result — a failed cleanup can
  therefore leave a temporary file behind.

Passing `new_string` as empty deletes the matched region. `edit_file` adds
no newline normalization: the fragment's edges are part of the match, so
the stored bytes are exactly `new_string` in place of `old_string`.

### File moves

`move_file` renames one regular file inside `data/files` without replacing
an existing destination. It refuses directories, symlinks, hard-linked
files, denied paths, unsafe paths, and read-only storage. A missing
destination directory chain is created only when `create_parents=true`.
The optional `expected_sha256` must match the hash reported by
`stat_file`; a mismatch leaves the source in place.

On Linux, the operation opens every parent relative to the sandbox root
without following symlinks and uses the kernel's atomic no-overwrite
rename. If the platform, kernel, or filesystem cannot provide that
primitive, the move fails without falling back to copy-and-delete or an
overwriting rename. The first release therefore supports only Linux.
The receipt reports `durable`; a directory `fsync` failure is surfaced as
a warning on a completed move because the namespace change has already
landed.

`storage.move_file` is the direct model-facing tool. It is not wrapped by the
core pack; the MCP capability is the single owner of its schema, validation,
filesystem behavior, and receipt.

`expected_sha256` is a revision check, not an inter-process lock. Another
process can still modify an open inode concurrently; the backend rejects
metadata changes observed while hashing, checks inode identity after the
rename, and reports or rolls back an observed swap. Independent processes
that bypass the storage seam remain outside those cooperative guarantees.

### Directory operations and trash

`move_directory` renames a whole subtree inside `data/files` as one atomic
no-overwrite operation. It uses the same Linux-only, descriptor-relative
rename machinery as `move_file`. It refuses a destination inside the source
and symlinked path components; a missing destination chain is created only
when `create_parents=true`. Its receipt has no content hash because a
directory has no single content revision to pin.

`delete_directory` is rmdir-only by default: a non-empty directory is
refused unless `recursive=true` names the blast radius. The recursive
default trashes first — the subtree is renamed into a timestamped
`.trash/<ts>/` session and the receipt carries `trashed_to`, so the delete
is reversible. `expected_empty` binds the call to the caller's belief that
the directory is empty (the directory analogue of `expected_sha256`); the
refusal lands before the trash rename. `trash=false` is the hard delete,
and deleting a `.trash/...` path is always a hard delete. Symlinks are
never followed anywhere in the delete surface: `delete_file` unlinks a
leaf link itself, and hard deletes inside a tree unlink links rather
than their targets. Recursive hard delete fails closed on platforms that do
not support descriptor-relative unlink/rmdir operations.

The `.trash/**` subtree — and the bare `.trash` root itself — is
deny-glob hidden from every routine operation. Restore is the
sanctioned exception — `move_directory` accepts a `.trash/...`
source, so `move_directory(trashed_to, original_path)` undoes a
trashed delete. `purge_trash` is the drain: each call hard-deletes at
most `limit` (default 100) of the oldest sessions and returns
`{purged, has_more}`, so a large trash empties across bounded calls;
`delete_directory` refuses the bare `.trash` root so one unbounded
call cannot drain every session at once (deleting a single
`.trash/<ts>/...` session subtree remains the sanctioned per-session
purge). `older_than` (RFC 3339; a naive value reads as UTC) skips
newer sessions.
Foreign entries and symlinked timestamp-shaped entries are ignored rather
than followed or removed. A failed purge reports `error`, the completed
`purged` count, and whether eligible sessions remain. Nothing auto-purges —
the trash window exists to be drained explicitly.

To restore, pass the exact `trashed_to` receipt value back to
`move_directory`; the original destination must not already exist. To delete
permanently, call `purge_trash` explicitly (or delete one trashed subtree
with `delete_directory(..., recursive=true)`).

### Batch operations

`delete_files`, `make_dirs`, and `move_files` take explicit path or pair
lists — no glob expansion, ever. Each item is validated and executed in
list order; a failing item records its own error and the batch never
aborts mid-way. Batches cap at 100 items per call. `delete_files` is a
per-file hard-delete batch; it does not use the directory trash substrate.

A batch is a queue of independent atomic operations with one receipt,
not a transaction: `{ok, count, succeeded, failed, results}` with `ok`
true only when every item succeeded. All-or-nothing deletes are not
offered; trash-first (`delete_directory` with the default `trash=true`)
is the reversible substrate when that shape is needed. Batch-level
refusals (read-only storage, empty list, over-cap) fire before the first
item executes and surface as tool errors; per-item failures are summary
data. `move_files` pairs carry an optional `expected_sha256` that binds
that pair's move to a revision reported by `stat_file`.

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
   memory backup, use SQLite's own backup mechanism (`sqlite3 memory.db
   ".backup copy.db"`) or a filesystem snapshot rather than copying a live
   database file without its journal/WAL. Memory retention dumps are not
   produced by the framework; if you need point-in-time copies, create them
   in the deployment's own backup pipeline.
3. Copy configuration, durable data, and related state together. Include
   all thread segments, indexes, dictionaries, and `.results` directories.
   Do not publish the archive.
4. Restore into a separate location with the matching software and reviewed
   pack sources. Validate roster/session loading, thread reference expansion,
   memory access, and pending approvals/recovery before returning to service.

Restoring old approvals or work products does not authorize replaying an
effect. Verify the current external state and invocation identity first.
Do not delete active state or backups as a troubleshooting shortcut.

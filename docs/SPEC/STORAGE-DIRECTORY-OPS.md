# Storage Directory Operations — Completion Digest

> **Status:** Implemented (2026-11-18).
> **Scope:** Directory moves and deletes, reversible trash, bounded purge,
> explicit-list batches, and delete symlink normalization.
> **Canonical guide:** [storage and data handling](../STORAGE.md).

## What shipped

The `lisa-storage` MCP server now exposes one direct `storage.*` contract for
file and directory operations. `storage.move_file` is a normal storage
capability tool; it has no duplicate core-pack wrapper. The storage SDK owns
the filesystem behavior and the MCP server remains a thin transport facade.

| Operation | Shipped contract |
| --- | --- |
| `storage.move_directory` | Atomic no-overwrite subtree rename on supported Linux filesystems; refuses a destination inside the source. |
| `storage.delete_directory` | Empty-directory `rmdir` by default; explicit recursion trashes first unless `trash=false`. |
| `storage.purge_trash` | Explicit, bounded hard deletion of the oldest eligible trash sessions with `purged` and `has_more`. |
| `storage.delete_files` | Ordered, explicit-list file hard deletes with per-item receipts. |
| `storage.make_dirs` | Ordered directory creation with per-item receipts. |
| `storage.move_files` | Ordered regular-file moves with an optional hash guard per pair. |

`storage.move_file`, directory moves, and their shared security machinery use
descriptor-relative parent traversal and `renameat2(RENAME_NOREPLACE)`.
Unsupported platforms or unavailable atomic primitives fail closed; there is
no copy-and-delete fallback. File moves additionally verify a content hash and
single-link inode state. Directory moves verify directory identity and refuse
a move into the source subtree, but have no content hash.

## Trash and restore

Recursive directory deletion uses rename-first trash by default. The backend
creates `.trash/<utc-timestamp>/<original-relative-path>` and returns that
exact location as `trashed_to`. Repeated deletes of the same original path get
separate timestamped sessions.

`.trash/**` is always added to the deny-glob policy, including when the host
provides an empty or custom deny list. It is hidden from routine listing,
reading, writing, editing, and deletion. The sanctioned exceptions are:

- restore with `storage.move_directory(trashed_to, original_path)`;
- direct hard deletion of a trashed subtree with recursive
  `storage.delete_directory`; and
- bounded `storage.purge_trash`.

Purge processes at most 100 sessions by default and may be called with a
positive smaller or larger `limit`. `older_than` is an RFC 3339 cutoff;
timestamps without an offset are interpreted as UTC. Foreign `.trash`
entries and symlinked timestamp-shaped sessions are left alone. Purge never
runs automatically.

## Delete safety

`delete_directory` validates path components without following symlinks. The
default call is rmdir-only, and `expected_empty=true` refuses a non-empty
directory even when recursion was requested. Recursive hard deletion uses
descriptor-relative unlink/rmdir and fails closed when the platform lacks
that support.

`delete_file` uses nofollow semantics: a leaf symlink is unlinked itself, not
its target. Symlinks found inside a recursively hard-deleted tree are handled
the same way.

Trash and hard-delete receipts count files and directories removed. A trash
receipt includes `trashed_to`. Error text is path-sanitized and does not
embed absolute host paths.

## Batch contract

Batches accept explicit lists only; there is no glob expansion. They cap at
100 items per call, execute in list order, and return
`{ok, count, succeeded, failed, results}`. Once execution starts, an item
failure is recorded without aborting later items. Read-only mode, an empty
list, or an over-cap list is refused before the first item runs.

Batches are not transactions. Move rollback is per move, not per batch.
`delete_files` permanently deletes files and does not route them through
directory trash.

## Tool exposure

`storage.move_file` is ambient for modes accepting default MCP capabilities.
`move_directory`, `delete_directory`, `purge_trash`, `delete_files`,
`make_dirs`, and `move_files` are part of the selected storage capability but
are not ambient. Modes or hosts must opt into those exact `storage.*` IDs.

## Verification

The backend and MCP test suites cover atomic moves, no-overwrite refusal,
symlink refusal, trash/restore, policy hiding, bounded and age-filtered purge,
partial purge failure accounting, hard-delete platform requirements, direct
directory-error sanitization, and batch mid-flight failure behavior.

The canonical operator contract remains [storage and data handling](../STORAGE.md);
this completion digest records the implementation boundary and lifecycle.

# STORAGE-DIRECTORY-OPS

> **Status:** Proposed, Rev 2 (spec-first, before implementation).
> **Date:** 2026-11-18.
> **Decision authority:** Adam Migus.
> **Workflow provenance:** coding_review audit of the storage operation
> surface (verified at source HEAD `035cab943c20`) → thinker decision-lock
> → design. All design questions are closed; this document is the decision
> record.

## Goal

Give the Lisa storage MCP surface directory operations and batch operations
with the same safety discipline the file operations already have: separate
`delete_directory` and `move_directory` operations, rename-first trash,
budgetable purge, and non-transactional batches with per-item receipts.

## §1 Problem statement

The Lisa storage MCP surface (`mcp_server/src/mcp_server/storage/server.py`
wrapping `sdk/src/sdk/storage/_backend.py`) is regular-files-only. Verified
audit findings:

- `delete_file` refuses directories: `sdk/src/sdk/storage/_backend.py:683-684`
  returns "delete_file targets a directory; use a file path".
- `move_file` cannot move directories: `_hash_open_file` at
  `sdk/src/sdk/storage/_move.py:161` requires `stat.S_ISREG` on a
  `follow_symlinks=False` stat; backend and MCP docstrings both say
  "directories are refused"; and the mover hashes file content for its
  sha256 receipt, which a directory has none of.
- Evidence from the 2026-11-17/18 storage reorg (organizer mode, ~230
  files): 49 moves + 13 deletions were 62 sequential tool calls; the
  round-trip overhead, not the filesystem, was the cost center; and four
  emptied directories (`.probes`, `__pycache__`, `summaries`, `salvage`) had
  to be left for the user to `rmdir` manually because no tool could remove
  or relocate them. The user confirmed manual cleanup — this spec is the
  follow-up work.

## §2 Locked decisions

User-confirmed, 2026-11-18:

1. Both `delete_directory` AND `move_directory` (separate operations, not a
   dual-mode `delete_file`).
2. Plain opt-in recursive flag on `delete_directory`, with trash-first
   default behavior.
3. All three batch operations: `delete_files`, `move_files`, `make_dirs`.
4. Trash via rename-first into a timestamped directory, deny-glob hidden,
   explicit purge only.
5. Spec-first landing (this document), then worktree + checked-commit
   implementation.
6. Batches are non-transactional (queue of independent atomic ops),
   documented explicitly.

*Rulings on the §11 open items (user-confirmed, 2026-11-18, second revision):*

7. **Symlinks (R7):** delete does not follow links — `delete_file` is
   normalized to nofollow (POSIX `rm` semantics: the link is unlinked itself,
   never the target); directory ops refuse symlink path components.
8. **`purge_trash` day one (R8):** budgetable slices — bounded per-call item
   limit with `has_more` pagination; purge only ever runs when explicitly
   called.
9. **Batch cap (R9):** 100 items per batch (provisional 64 raised).

## §3 New operation: delete_directory

Signature: `delete_directory(path, recursive=False, trash=True, expected_empty=False)`

- Default `recursive=False`: rmdir-only; fails with a clear error if the
  directory has any content. Covers the exact reorg residue case.
- `recursive=True`: the explicitly-named blast-radius mode — the only
  operation in the surface that can destroy unbounded data in one call,
  hence its own name rather than overloading `delete_file`.
- `expected_empty`: staleness-binding flag — the rmdir only proceeds if the
  directory is verifiably empty at execution, mirroring `stat_file`'s
  `expected_sha256` proof pattern.
- Receipt: `{ok, path, removed_files, removed_dirs, trashed_to?, durable?}`.
  Counts of removed entries; `trashed_to` present when `trash=True`.
- Symlink discipline (R7, locked): `delete_file` is normalized to
  **nofollow** — a leaf symlink is unlinked itself, never its target (POSIX
  `rm` semantics), superseding its current `resolve()` follow behavior. The
  directory ops refuse symlink path components: directories are never
  resolved through links, and a link at a move destination is an
  `EEXIST`-style refusal, not a target. The prior asymmetry (deleting a link
  deleted its target) is removed by this change and gets its own test per
  §8.

## §4 New operation: move_directory

Signature: `move_directory(source, destination, create_parents=False)`

- Reuses the existing secure-mover machinery in `_move.py`:
  descriptor-relative parent walk with `O_NOFOLLOW` on every component,
  `renameat2(RENAME_NOREPLACE)` at `_move.py:70-99` — the same primitive,
  same `EEXIST`/`EXDEV` failure modes, atomic on one filesystem.
- Source check: `nofollow` stat requiring `S_ISDIR` (mirror of
  `_hash_open_file`'s `S_ISREG` check at `_move.py:161`).
- No hashing, no recursive copy — the whole subtree moves as one metadata
  operation. Receipt omits sha256 (nothing content-pinnable):
  `{ok, source, destination, durable}`.
- New guard file moves do not have: destination must not be inside the source
  subtree (`dest_parts[:len(source_parts)] == source_parts` → refuse).
  Also refuse `source == destination` (existing `_prepare_move` check
  applies unchanged).
- `EXDEV` (cross-filesystem) → fail closed, no copy-tree fallback. The
  sandbox is one filesystem (`data_dir/files`) so this should never fire in
  practice.

## §5 Trash mechanics

User-questioned, answered:

- **Mechanism:** rename-first, ALWAYS. No hard-link fallback — hard links are
  files-only (cannot trash directories) and reintroduce `st_nlink > 1`, the
  exact state the secure mover refuses. If the rename fails `EXDEV`, fail
  closed.
- **Layout:** `.trash/<UTC-timestamp-filesystem-safe (colons → dashes)>/<original relative path>`. One timestamped directory per trash operation, so
  repeated trashes of the same path never collide; original relative layout
  preserved inside, so restore is simply
  `move_directory('.trash/<ts>/<relpath>', '<relpath>')`.
- **Deny-glob policy:** add `.trash/**` to the default deny-glob list. Effect
  is free: `list_files` already filters denied paths (at `_backend.py:665`)
  and `read_file`/`stat_file`/`edit_file`/recursive ops refuse denied paths.
  Trash is invisible to routine listing but not silently swept into any
  future recursive walk.
- **The sanctioned exception, named explicitly in the design:** purge.
  Purge = `delete_directory` on a timestamped subtree with
  `recursive=True, trash=False` (hard delete), or
  `purge_trash(older_than=None, limit=100)` — **day one per R8**.
  Budgetable-slice contract: each call removes at most `limit` oldest
  timestamped subtrees and returns `{purged, has_more}`, so a large trash is
  drained across bounded calls rather than one unbounded sweep;
  `older_than` (RFC 3339) is an optional age cutoff. Never auto-purge on
  delete — that would destroy the safety window the feature exists to
  provide.
- **Default:** `trash=True` is the default for `delete_directory` (and for
  batch `delete_files` when its own trash flag is added), so the plain
  recursive opt-in flag rides on a reversible substrate.
- **Receipt** carries `trashed_to` so restore is actionable.

## §6 Batch operations

- `delete_files(paths[])`: validate all items (existence, policy/deny-glob,
  type) → execute sequentially in list order → per-item receipts → batch
  summary `{ok, count, succeeded, failed, results: [{path, ok, error? | trashed_to?}]}`.
- `move_files(pairs[])`: same shape; per-pair `expected_sha256` optional for
  staleness binding; per-item receipts.
- `make_dirs(paths[])`: same shape (validate → create → per-item receipt).
- **Contract:** validate-then-execute, never mid-batch abort, no glob
  expansion in deletes — explicit path lists only (the reorg's own history:
  the user approving a named list of 13 files is the correct
  human-in-the-loop shape; globs turn one typo into a wipe).
- **Cap:** 100 items per batch (R9 — the provisional 64 was raised; a
  documented request-byte ceiling may still apply), recorded in
  `docs/STORAGE.md` beside the existing manifest-pair/suffix-list limits.
- **Non-transactional by design** (locked decision 6): a batch is a queue of
  independent atomic operations with one receipt, not an all-or-nothing
  transaction. All-or-nothing deletes require trash-first (§5 covers it);
  move rollback machinery is not worth the complexity. Document this in
  `docs/STORAGE.md`.

## §7 Ride-along fixes to existing operations

1. `make_dir` is mislabeled `destructive_hint=True`
   (`mcp_server/src/mcp_server/storage/server.py:384`; backend `make_dir` at
   `_backend.py:798`). mkdir is not destructive — flip to
   `destructive_hint=False` in the same worktree.
2. `delete_file` symlink normalization (R7): replace the trailing
   `resolve()` with an nofollow unlink (`os.unlink`/`lstat` path) so a leaf
   symlink is unlinked itself, never its target. This is the one
   intentional behavior change to an existing op in this change set —
   directory contents reached only through a link are no longer deletable
   via the link, which is the fix, and it gets its own receipt/error surface
   so callers see the distinction.

## §8 Touched surfaces & testing plan

- **Backend:** `sdk/src/sdk/storage/_backend.py` (new `delete_directory`/
  `move_directory`/methods for the three batch ops; the trash helper).
- **Mover:** `sdk/src/sdk/storage/_move.py` (directory-move function beside
  `move_regular_file`, reusing `_rename_noreplace`, `_open_parent`,
  `_sync_directories`).
- **Tool surface:** `mcp_server/src/mcp_server/storage/server.py` (six new
  `@mcp.tool` facades — `delete_directory`, `move_directory`,
  `purge_trash`, `delete_files`, `move_files`, `make_dirs` — plus the
  `make_dir` hint fix and the `delete_file` nofollow normalization from
  §7).
- **Policy:** default `deny_globs` gains `.trash/**` (threaded through
  `server.py`'s `_cfg.deny_globs` composition at `server.py:117-170/460`,
  including the env-var path).
- **Tests:** `sdk/src/sdk/storage/tests/test_backend.py` and
  `mcp_server/tests/test_storage_server.py` — mirror structure: happy path,
  no-overwrite refusal, symlink refusal, delete_file nofollow
  normalization (leaf link removed itself; target and a link-to-directory
  each get clear refusals), EXDEV fail-closed, trash round-trip (trash →
  restore via `move_directory` → purge), `purge_trash` slice contract
  (`limit` respected, `{purged, has_more}`, `older_than` honored),
  deny-glob hiding, batch validate-then-execute with a mid-batch item that
  fails policy (expect per-item failure, no abort, subsequent items still
  executed), `expected_empty` staleness refusal, destination-inside-source
  refusal.
- **Docs:** `docs/STORAGE.md` — new ops, trash semantics, the
  non-transactional batch statement, the batch cap.

## §9 Explicitly out of scope

- Glob/pattern expansion in any delete path.
- Cross-filesystem copy fallbacks.
- Hard-link trash fallback.
- Auto-purge/TTL scheduling (trash is only drained by explicit
  `purge_trash` calls; no background sweeper). The delete-on-a-timestamped-
  subtree path remains available as the direct alternative.
- Extending `delete_file`/`move_file` to directories via dual behavior (the
  separate-op decision is locked).
- Symlink-following anywhere in the surface (R7 removes the last instance
  in `delete_file`; no new op ever follows).

## §10 Landing plan

1. This spec committed (spec-first, per locked decision 5).
2. Worktree branch `lisa/storage-directory-ops` with per-op commits in the
   order: ride-alongs (`make_dir` hint fix + `delete_file` nofollow
   normalization + deny-glob addition) → `move_directory` (with
   destination-inside-source guard) → `delete_directory` (with trash) →
   `purge_trash` (bounded slices) → `delete_files`/`make_dirs` →
   `move_files` → `docs/STORAGE.md`.
3. Each commit: red/green tests per the §8 plan; the only intentional
   behavior change to an existing op is `delete_file` nofollow (R7) —
   everything else must leave existing-op receipts byte-compatible, and
   that change's test must prove old-target-following is gone.
4. Adam files/pushes the PR per the usual flow.

## §11 Open items / next steps for the implementing mode

The three Rev-1 open items are closed by the Rev-2 rulings (§2, items
7–9):

- R7: `delete_file` nofollow normalization — in scope, §7.2; implementing
  mode adds the test and checks the codebase for any caller relying on the
  old follow behavior before landing.
- R8: `purge_trash` is day one with the bounded-slice contract (§5) — no
  deferral.
- R9: batch cap is 100 (§6); if a request-byte ceiling binds below 100
  items in practice, surface it as a follow-up observation rather than
  silently lowering the cap.

Remaining implementer checklist:

- Verify no existing test or caller depends on `delete_file` following
  symlinks.
- Keep the 100-cap enforced server-side per batch call (a single batch
  request, not cumulative).
- Update `docs/STORAGE.md` op table receipts to include `trashed_to`,
  `{purged, has_more}`, and batch summaries.

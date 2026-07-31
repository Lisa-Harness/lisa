# File I/O Contract Verification - 2026-07-31

Follow-up verification of `file-io-audit-2026-07-28.md` across `lisa`,
`../lisa-packs`, and `../lisa-packs-wip`.

## Contract under test

From the originating audit:

- `lisa-storage` is the only allowed file I/O path.
- Exception: a pack may do its own I/O only if it has explicit I/O
  sandboxing guarantees (like coding mode).

`docs/BOUNDARIES.md` frames storage as a host-injected seam for
blob/object storage (`lisa-storage` is the built-in). It does **not**
state a strict "lisa-storage only" rule for host-internal control-plane
files. The audit's policy is therefore stricter than the documented
boundary. This verification reports against the audit's strict policy
and flags where the policy itself needs a decision.

## Verification method

Repository-wide `rg` scans for direct read/write/mutation callsites
(`.write_text`, `.read_text`, `.write_bytes`, `.read_bytes`, `.open`,
`sqlite3.connect`, `os.replace`, `os.unlink`, `os.mkdir`, `shutil.*`)
across runtime code, excluding `**/tests/**`, `**/build/**`,
`**/__pycache__/**`, and `*.egg-info`. Each surviving callsite was
read in context and classified.

## Status of original findings

### Critical

1. **Checkpoint pack direct write fallback** — **FIXED.**
   `sdk/packs/checkpoint/commands.py` no longer writes the thread file
   directly. `_persist_compacted_history` now calls the host
   `write_turn_history_fn` seam and, when the seam is absent, logs a
   warning and returns `False` without writing. No `thread_path.write_text`
   remains in the checkpoint pack.

### High

1. **`/models apply` direct write** — **FIXED.**
   `hosts/core/commands/_models.py` now routes through
   `lisa-storage.write_model_manifest` via
   `_write_manifest_text_to_storage`. No `active_path.write_text`
   remains; the only `write_model_manifest` references are the storage
   tool call and its response handling.

2. **Thread and digest persistence are host-direct writes** — **STILL
   OPEN.** Direct writes remain at:
   - `hosts/core/thread.py:110` (thread append, `thread_path.open("a")`)
   - `hosts/core/state_provider.py:310` (digest compaction,
     `digest_path.write_text`)
   - `hosts/core/state_provider.py:313` (thread write-back,
     `self._thread_path.write_text`)
   - `hosts/core/turn_hooks_builder.py:549` (write-back seam
     implementation, `thread_path.write_text`)

   These are the write-back seam that the checkpoint pack now depends
   on. They are host-internal, not pack-level. Still a policy violation
   under the audit's strict reading; needs the same host-internal
   exemption decision as Medium #1.

### Medium

1. **Host control-plane state writes** — **STILL OPEN, plus two new
   callsites since the audit.** Direct writes remain at all originally
   flagged locations:
   - `hosts/core/approvals.py:175` (temp write + `os.replace`)
   - `hosts/core/code_artifacts.py:80` (temp write + `os.replace`)
   - `hosts/core/intake.py:244` (`state_path.write_text`)
   - `hosts/core/content_roots.py:181` (`path.write_text`)
   - `hosts/core/model_discovery/manifest.py:655` (`tmp.write_text`)
   - `kernel/token_usage.py:134` (`self._path.open("a")`)
   - `hosts/core/session_context_fetch.py:135` (`cache_md.write_bytes`)
   - `hosts/core/session_context_fetch.py:142` (`cache_meta.write_text`)

   **New since the audit** (same class — host-internal atomic-write
   control-plane state):
   - `hosts/core/workflow_instance_store.py:196` (temp write +
     `os.replace`)
   - `hosts/core/recovery.py:134` (temp write + `os.replace`)

   All are host-owned internal JSON state files with atomic temp+replace
   patterns and no path-injection surface (paths are constructed from
   `data_dir`, not user input). They violate the strict "storage only"
   policy but match the audit's "host-internal control-plane" carve-out
   candidate. No decision has been recorded yet.

2. **Memory server local SQLite** — **STILL OPEN.**
   `sdk/memory/store.py:106` still opens a local DB file directly via
   `sqlite3.connect`. First-party server-owned persistence; the audit's
   "treat `lisa-memory` as an explicit first-party storage exception"
   decision has not been recorded.

### Low

1. **Read-only filesystem discovery** — **UNCHANGED, still read-only.**
   Direct reads remain in `sdk/content/_catalog.py`, `_reader.py`,
   `_copilot_discovery.py`, `kernel/ipsar_fsm.py`,
   `kernel/content_roots.py`, `hosts/core/session_context.py`. No write
   callsites appeared in these paths. Still needs the documented
   read-only exception allowlist called for in the audit.

## Sibling pack repos

### `../lisa-packs` (public)

Runtime (non-test, non-build) direct file I/O:

| Pack | Callsites | Status |
| --- | --- | --- |
| `brave_search` | none | Compliant |
| `mem0` | none | Compliant |
| `search` | none | Compliant |
| `markdown` | none | Compliant |
| `okf` | `okf_helpers.py:167/171/173` (`append_log_entry`, deprecated) | Compliant in practice — operations call `append_log_entry_storage` (the `lisa-storage` route) via `okf_reindex.py:129` and `okf_log.py:86`. The direct `append_log_entry` is explicitly marked deprecated and is not called by any operation. |
| `compaction` | `slice_provider.py:149` (sidecar) | **Fixed (commands.py).** `_write_turn_history` now routes through the host `write_turn_history_fn` seam with no direct pack write (mirrors the checkpoint pack fix). The slice sidecar (`slice_provider.py:149`) remains a direct write — same class as the letta slice sidecar; policy-undecided. |
| `letta` | `slice_provider.py:102` (sidecar) | **Fixed (identity_map.py).** `IdentityMap` now routes all reads/writes through `sdk.storage.StorageBackend` (the `lisa-storage` sandbox backend). No direct `os.replace`/`unlink`/`tempfile` remains. The slice sidecar (`slice_provider.py:102`) remains a direct write — pack-internal slice state, not wired in production (factory passes `thread_dir=None`); policy-undecided. |

### `../lisa-packs-wip` (private)

Runtime (non-test, non-build) direct file I/O:

| Pack | Callsites | Status |
| --- | --- | --- |
| `cloudflare` | none | Compliant |
| `github` | none | Compliant |
| `gmail` | none | Compliant |
| `stripe` | none | Compliant |
| `coding` | `local_files.py:120` (`path.open("rb")`) | **Compliant — qualifies as the audit's named exception.** `resolve_relative_path_at_root` enforces a workspace root, rejects absolute paths, `..` traversal, symlink escape, and excluded directories. Reads are bounded (`max_bytes`). This is the explicit I/O sandboxing guarantee the audit carves out. |

All other `*.py` matches in the pack repos are under `**/tests/**` or
`**/build/**` and are out of scope per the audit's method.

## New issues since the audit

1. **`compaction` pack replicated the removed checkpoint anti-pattern**
   — **FIXED.** `lisa-packs/compaction/commands.py:_write_turn_history`
   now routes through the host `write_turn_history_fn` seam with no direct
   pack write, mirroring the checkpoint pack fix.

2. **`letta` pack identity-map direct filesystem writes** — **FIXED.**
   `lisa-packs/letta/utils/identity_map.py:IdentityMap` now routes all
   reads/writes through `sdk.storage.StorageBackend` (the `lisa-storage`
   sandbox backend). The identity-map file moved from
   `<data_dir>/letta/identity-map.json` to
   `<data_dir>/files/letta/identity-map.json` (inside the storage files
   sandbox). Hard cutover, no migration (no existing data at the old
   path). `lisa-sdk` added to the letta pack's `pyproject.toml`
   dependencies.

3. **Two new host-internal write callsites not in the original ledger:**
   `hosts/core/workflow_instance_store.py:196` and
   `hosts/core/recovery.py:134`. Same class as Medium #1; include them
   in the host-internal exemption decision.

## Contract preservation summary

| Boundary | Preserved? |
| --- | --- |
| Pack-level I/O without a sandbox contract | **Yes** — `compaction` and `letta` pack-owned writes are now fixed. `compaction` routes through the host `write_turn_history_fn` seam; `letta` `IdentityMap` routes through `sdk.storage.StorageBackend`. The in-repo `checkpoint` pack was already fixed. Remaining: the two slice sidecars (`compaction`, `letta`) are pack-internal slice state, not wired in production, and policy-undecided. |
| `lisa-storage` as the manifest write gateway | **Yes** — `/models apply` now routes through `write_model_manifest`. |
| Coding pack sandboxed I/O exception | **Yes** — `coding` enforces root + traversal/symlink/exclusion guards. |
| Host-internal control-plane writes | **Undecided** — 10 callsites across `hosts/core` and `kernel` write internal state directly. They match the audit's proposed "host-internal exemption" but no policy decision has been recorded. |
| `lisa-memory` first-party SQLite exception | **Undecided** — still direct; no exception recorded. |
| Read-only discovery exception set | **Undecided** — still direct reads, no allowlist recorded. |

## Required decisions to close the contract

1. **Record the host-internal control-plane policy.** Either (a) exempt
   host-owned internal state files explicitly and add them to a CI
   allowlist, or (b) migrate them behind a storage facade. The audit's
   Medium #1 and the two new callsites all wait on this.
2. **Record the `lisa-memory` first-party exception** or route DB
   lifecycle through a unified storage layer.
3. **Record the read-only discovery allowlist** and enforce it in CI.
4. ~~**Fix `lisa-packs/compaction` `_write_turn_history`**~~ — **DONE.**
5. ~~**Decide on `lisa-packs/letta` identity-map writes**~~ — **DONE** (routed through `sdk.storage.StorageBackend`; slice sidecar remains policy-undecided).
6. **Add the CI guard** from the audit's enforcement plan so future
   pack direct-writes are caught automatically.
7. **Decide on the two slice sidecars** (`compaction`, `letta`) — they
   are pack-internal slice state, not wired in production (factories pass
   `thread_dir=None`), but still direct writes. Either grant a
   documented pack-internal slice-state exception or route through
   `lisa-storage`.

## Conclusion

Four of the audit's findings are fixed (Critical #1, High #1, plus the
`compaction` and `letta` pack-owned writes identified in this
verification). The remaining findings are all "policy-undecided"
rather than "policy-broken": the code has not drifted, but the policy
decisions needed to classify the host-internal control-plane writes,
`lisa-memory` SQLite, read-only discovery, and the two slice sidecars as
compliant or non-compliant have not been recorded. No genuine contract
breaks remain in the pack layer.

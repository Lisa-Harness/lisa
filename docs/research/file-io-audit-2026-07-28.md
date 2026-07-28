# File I/O Audit - 2026-07-28

## Objective

Audit runtime file I/O across the repository against this policy:

- `lisa-storage` is the only allowed file I/O path.
- Exception: a pack may do its own I/O only if it has explicit I/O sandboxing guarantees (like coding mode).

## Scope

- Included: runtime Python code in `kernel/`, `hosts/`, `sdk/`, `mcp_server/`.
- Excluded: tests (`**/tests/**`, `test_*.py`).

## Method

Repository-wide pattern scans for:

- Direct reads/writes: `read_text`, `write_text`, `read_bytes`, `write_bytes`, `open`.
- Filesystem mutation: `os.replace`, `unlink`, `mkdir`, etc.
- Local DB files: `sqlite3.connect`.
- Path traversal/inspection APIs for discovery and catalog loading.

## Inventory Summary

- Runtime file-I/O/traversal callsites: 272
- Runtime files containing those callsites: 45
- Callsite distribution:
  - `hosts`: 123
  - `sdk`: 72
  - `kernel`: 43
  - `mcp_server`: 34

## Baseline Compliant Surface

The canonical compliant surface is the storage MCP server implementation in:

- `mcp_server/storage/server.py`

It enforces path safety and sandboxing for file tools and owns checkpoint/manifest persistence.

## Findings

## Critical

1. Pack-level direct write bypass in checkpoint pack (no pack sandbox boundary)

- Evidence:
  - `sdk/packs/checkpoint/commands.py:282`
- Detail:
  - `/summarize-history` falls back to direct `thread_path.write_text(...)` when `write_turn_history_fn` is absent.
- Why this is critical:
  - This is a pack-level write path outside `lisa-storage`.
  - The checkpoint pack does not define an independent storage sandbox contract equivalent to coding.
- Required action:
  - Remove the direct file-write fallback.
  - Require the host seam (`write_turn_history_fn`) or route through `lisa-storage`.

## High

1. Model manifest apply writes directly to disk instead of `lisa-storage` manifest tools

- Evidence:
  - `hosts/core/commands/_models.py:220`
- Detail:
  - `/models apply` writes `models.yaml` directly with `active_path.write_text(...)`.
- Why this is high:
  - `lisa-storage` already exposes `read_model_manifest` and `write_model_manifest` and should be the single write gateway.
- Required action:
  - Route apply through `lisa-storage.write_model_manifest`.

1. Thread and digest persistence are host-direct writes, not storage-routed

- Evidence:
  - `hosts/core/thread.py:111`
  - `hosts/core/state_provider.py:310`
  - `hosts/core/state_provider.py:313`
  - `hosts/core/turn_hooks_builder.py:534`
- Detail:
  - Thread append, digest compaction, and write-back seam all write directly.
- Why this is high:
  - These are central conversation artifacts and bypass storage policy.
- Required action:
  - Consolidate thread/digest persistence behind one storage abstraction and route through `lisa-storage` (or a documented host-internal storage exception contract).

## Medium

1. Host control-plane state writes bypass `lisa-storage`

- Evidence:
  - `hosts/core/approvals.py:175`
  - `hosts/core/approvals.py:179`
  - `hosts/core/code_artifacts.py:80`
  - `hosts/core/code_artifacts.py:84`
  - `hosts/core/intake.py:244`
  - `hosts/core/content_roots.py:181`
  - `hosts/core/model_discovery/manifest.py:671`
  - `kernel/token_usage.py:134`
  - `hosts/core/session_context_fetch.py:150`
  - `hosts/core/session_context_fetch.py:157`
  - `hosts/core/session_context_fetch.py:166`
- Detail:
  - Approvals/artifacts/state/caches/telemetry/provenance locks are all directly persisted.
- Why this is medium:
  - Most are host-owned internal files with low path-injection risk, but still violate strict "storage only" policy.
- Required action:
  - Decide policy boundary explicitly:
    - either exempt host-internal control-plane files, or
    - migrate these writes behind `lisa-storage`/host storage facade.

1. Memory server uses direct local SQLite file backend

- Evidence:
  - `sdk/memory/store.py:106`
- Detail:
  - `SqliteMemoryStore` opens a local DB file directly.
- Why this is medium:
  - This is first-party server-owned persistence, separate from `lisa-storage`.
- Required action:
  - Either treat `lisa-memory` as an explicit first-party storage exception, or route DB file lifecycle through a unified storage policy layer.

## Low

1. Broad read-only file discovery is direct filesystem access

- Evidence includes:
  - `sdk/content/_catalog.py`
  - `sdk/content/_reader.py`
  - `sdk/content/_copilot_discovery.py`
  - `kernel/ipsar_fsm.py`
  - `kernel/content_roots.py`
  - `hosts/core/session_context.py`
- Detail:
  - These paths are predominantly read-only and needed for local catalog/session loading.
- Why this is low:
  - Operationally expected, but still direct filesystem access outside storage MCP.
- Required action:
  - Keep as read-only exception set, documented and enforced by lint/CI allowlist.

## Runtime Write Callsite Ledger

- Compliant (storage server internals):
  - `mcp_server/storage/server.py:181`
  - `mcp_server/storage/server.py:302`
  - `mcp_server/storage/server.py:415`

- Non-compliant or policy-unclear write paths:
  - `hosts/core/content_roots.py:181`
  - `hosts/core/turn_hooks_builder.py:534`
  - `hosts/core/code_artifacts.py:80`
  - `sdk/packs/checkpoint/commands.py:282`
  - `hosts/core/approvals.py:175`
  - `hosts/core/intake.py:244`
  - `hosts/core/state_provider.py:310`
  - `hosts/core/state_provider.py:313`
  - `hosts/core/commands/_models.py:220`
  - `hosts/core/model_discovery/manifest.py:671`
  - `hosts/core/session_context_fetch.py:150`
  - `hosts/core/session_context_fetch.py:157`
  - `hosts/core/session_context_fetch.py:166`
  - `kernel/token_usage.py:134`
  - `hosts/core/thread.py:111`
  - `sdk/memory/store.py:106`

## Recommended Enforcement Plan

1. Immediate (policy safety):
   - Remove checkpoint pack direct write fallback at `sdk/packs/checkpoint/commands.py:282`.

2. Short-term (single manifest path):
   - Route `/models apply` writes through `lisa-storage` manifest tools.

3. Medium-term (thread/state consolidation):
   - Introduce one host storage facade that is backed by `lisa-storage` for thread, digest, token usage, and control-plane files.

4. Continuous enforcement:
   - Add a CI guard that fails on new runtime direct write APIs outside an explicit allowlist.

## Conclusion

Current runtime architecture is not yet aligned with strict "`lisa-storage` only" policy. The most actionable policy break is the checkpoint pack direct write fallback. Host internals currently rely on many direct writes and need either explicit policy exemption or staged migration to a single storage path.

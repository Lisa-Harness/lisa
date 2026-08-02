# Instructions for the lisa-packs coding agent — markdown pack deferred work

Work in the pack repo: `C:\Code\lisa-packs\markdown` (this is the source of
truth; the main `C:\Code\lisa` repo installs it editable). Run all tests
from the pack repo root using the main-repo venv:

```powershell
Set-Location C:\Code\lisa-packs\markdown
.\..\..\lisa\.venv\Scripts\python.exe -m pytest lisa_packs\markdown\tests\ -q
```

Phase 1 (storage granular I/O) and Phase 2 (map-reduce consolidation) are
already shipped and committed:

- `1931588 feat(forge_document): add map-reduce consolidation path`
- main repo `126e7b5 feat(storage): add read_file_range and append_file`
- main repo `2695234 fix(openai_tools): disable strict mode for union item schemas`

`read_file_range` and `append_file` are now available on the
`lisa-storage` MCP server (13 tools). The map-reduce path in `forge_document`
already works but cut two corners that this work closes, plus Phase 3.

Do NOT touch the single-pass path or the single-doc/memory-only path —
those are covered by existing tests and must stay green. Only extend the
map-reduce branch and add the gated Phase 3 path.

## Task 1: Per-source density floor + honest failure (Phase 2 completion)

**File:** `lisa_packs/markdown/operations/forge_document/forge_document.py`

The current `_forge_one_source` (~L1020) returns `density_passed=True`
unconditionally because a single source has no consolidation floor. This
is correct for the floor itself, but it skips the honest-failure contract
the plan requires: if a per-source section comes back implausibly tiny
relative to ITS source (e.g. 100B for a 20KB source), that is a lossy
section and must be surfaced, not silently merged.

1. Add a per-source density heuristic: if `len(section_bytes) <
   _PER_SOURCE_MIN_RATIO * spec.bytes` (new constant, default `0.10` — a
   section under 10% of its single source is presumed abridged), retry the
   per-source forge once with a sharpened instruction (reuse the
   `_DENSITY_RETRY_PATH` prompt shape, or add a
   `forge_document_per_source_density_retry.md` authored prompt). If the
   retry still trips, keep the section but prepend an inline HTML comment:
   `<!-- section density floor failed: NNB vs NB source -->` and record
   the path in a `density_failed_sources: list[str]` returned alongside the
   section.
2. Change `_forge_one_source` to return `(section, finish_reason,
   density_passed, density_reason, failed_paths)` (extend the current
   4-tuple — update the call site in `execute()`).
3. In `execute()` map-reduce branch, aggregate `density_failed_sources`
   across all per-source calls and add it to `OperationResult.output_json`
   as `density_failed_sources`. The global `density_passed` becomes
   `density_failed_sources == []` for the map-reduce path. Keep the
   existing `density_ratio` aggregate for telemetry.
4. Tests in `test_forge_document_sources.py`:
   - Add `test_map_reduce_per_source_density_failure_surfaces_failed_paths`:
     one source returns a tiny section, retry stays tiny →
     `density_failed_sources` contains that path AND the persisted doc
     carries the inline `<!-- section density floor failed -->` marker.
   - Add `test_map_reduce_per_source_density_retry_recovers`: first call
     tiny, retry full → `density_failed_sources` is empty.
   - Do NOT modify the existing single-pass density-floor tests (they pin
     `_single_pass_pack_config()` and must stay green).

## Task 2: OKF provenance per-source columns (Phase 2 completion)

**File:** `lisa_packs/markdown/operations/forge_document/forge_document.py`

`_build_okf_provenance_body` (~L1272) renders a Sources table with columns
`# | Path | Priority | Bytes | Treatment`. Add two columns for the
map-reduce path: `Passes` (1 + per-source density retries) and `Density`
(pass / FAILED / n/a for single-source). The single-pass path passes `0`
retry counts and `n/a` density so the table shape is stable.

1. Extend the Sources table header to
   `| # | Path | Priority | Bytes | Passes | Density | Treatment |` and
   the separator row to 7 columns. Update every row emit (readable + failed).
2. Thread per-source pass/density data into `_build_okf_provenance_body`
   via a new optional param (e.g. `per_source_meta: list[dict] | None =
   None` where each dict has `{passes, density}` indexed by source order).
   When `None` (single-pass path), emit `0` passes and `n/a` density so
   the single-pass OKF concept is unchanged.
3. Wire it from `execute()`: build `per_source_meta` from the
   `density_failed_sources` + per-source retry counts collected in Task 1.
4. Test: extend `test_okf_provenance_concept_written_and_logged` (or add
   a sibling) to assert the map-reduce OKF concept body contains the
   `Passes` and `Density` columns and per-source values. Pin
   `_map_reduce_pack_config()`.

## Task 3: forge_document.yml tool_ids + timeout

**File:** `lisa_packs/markdown/operations/forge_document/forge_document.yml`

1. Add `read_file_range` and `append_file` to `tool_ids` (Phase 1 shipped
   them; Phase 3 uses `read_file_range`). The tool_ids list becomes:
   `[markdown_generate, write_file, read_file, read_file_range, append_file,
   list_files, memory.search, okf_emit_concept, okf_log]`
2. Raise `timeout_seconds` from `300` to `600`. The map-reduce path issues
   N per-source calls + 1 merge; 300s was sized for one pass. 600s covers
   a 7-source consolidation (8 calls × ~60s Ollama) with headroom. Document
   the reason in a comment above the line. (If you prefer per-call
   sub-deadlines, that's a larger kernel change — defer; just raise the
   wall for now.)

## Task 4: Phase 3 — rolling accumulation (gated, default off)

**Files:** `_source_paths.py`, `forge_document.py`, new prompt files.

Only build this if Tasks 1–3 are green. This is the upgrade path for when
per-source map-reduce boundaries demonstrably lose cross-source coherence
on a SINGLE large source (>32KB). It uses `read_file_range` (Phase 1) to
chunk one big source into bounded windows and forge a running document.

1. **`_source_paths.py`** — add `read_source_chunks(spec, *, ctx,
   chunk_chars=8000) -> list[SourceSpec]`:
   - Call `stat_file` to get `total` bytes; plan chunk count from
     `chunk_chars`. Produce an ordered list of `SourceSpec`-like fragments
     (reuse `SourceSpec` with a `chunk_index` extra field) by calling
     `read_file_range(path, start=offset, length=chunk_chars)` in a loop
     until `eof`. Each fragment carries `spec.path`, the chunk text, and
     its byte count. Respect the existing `_MAX_PER_SOURCE_CHARS` cap by
     default (don't read beyond it unless `chunk_chars` is raised).
   - This goes through `ctx.execute_tool_fn` routed to `lisa-storage`
     (same pattern as `read_sources`). No direct filesystem access.
2. **`forge_document.py`** — add `_forge_rolling(ctx, chunks, *,
   user_input, memory_context, model_def, max_tokens, ...)`:
   - Maintain a running `document` string. For each chunk, send the
     running tail (capped to the last N headings — new helper
     `_tail_headings(document, max_headings=3)`) + the next chunk to a
     per-chunk forge prompt, append the extension. Run
     `_continue_until_complete` per chunk. Persist incrementally via
     `append_file` (Phase 1) after each chunk so progress survives a
     mid-forge crash — but only when a `forge_rolling_incremental_persist`
     flag is set (default off; incremental persistence is opt-in since it
     leaves partial docs on disk).
   - Return the final `document` + finish reasons.
3. **New prompts** (authored Markdown via `ctx.read_authored_text_fn`):
   - `forge_document_system_rolling.md` — "you are extending a running
     document with the next chunk; preserve everything already written;
     append only new content from this chunk."
   - `forge_document_user_rolling.md` — template with `{user_input}`,
     `{running_tail}`, `{chunk_text}`, `{chunk_index}`, `{chunk_total}`.
4. **Gate**: add `forge_rolling_accumulation: bool = False` to
   `_ResolvedSettings` + `_resolve_settings` (follow the `_cfg_int` pattern
   but read a bool — add `_cfg_bool` helper). In `execute()` map-reduce
   branch, when `forge_rolling_accumulation` is True AND a single source
   exceeds `_MAX_PER_SOURCE_CHARS`, use `_forge_rolling` instead of
   `_forge_one_source` for THAT source only. Multi-source consolidations
   still use map-reduce per source; only an oversize single source takes
   the rolling path. Default off → no behavior change unless opted in.
5. **Tests**:
   - `test_read_source_chunks_plans_chunks_from_stat`: fake `stat_file`
     returns 50000, `read_file_range` returns windows → assert chunk
     count and ordering, last chunk `eof=True`.
   - `test_rolling_accumulation_extends_running_doc`: single oversize
     source, `forge_rolling_accumulation=True` → multiple completion calls,
     final document contains content from all chunks.
   - `test_rolling_accumulation_default_off_uses_map_reduce`: same
     source, default config → map-reduce path (no `read_file_range`
     calls).

## Verification (run after each task, all must pass)

```powershell
Set-Location C:\Code\lisa-packs\markdown
.\..\..\lisa\.venv\Scripts\python.exe -m pytest lisa_packs\markdown\tests\ -q
```

Baseline before starting: 51 passed. After each task, the count goes up
and nothing goes red. The single-doc/memory-only tests
(`test_forge_document.py`) must stay 100% green — they are the regression
guard for the untouched paths.

## Boundaries (from repo architecture rules — do not violate)

- All chunking policy stays in the pack. `read_file_range` gives
  bytes/chars; the pack decides how to slice. Do NOT add chunking logic
  to the MCP storage server.
- Prompts, templates, and user-facing content stay in authored Markdown
  under `prompts/`, loaded via `ctx.read_authored_text_fn`. No fallback
  prose in Python (pack-content rule). Missing/empty loader raises.
- Pack writes go through `lisa-storage` tools via `ctx.execute_tool_fn`
  with sandbox-relative paths. No direct filesystem access.
- No kernel or host changes. No new MCP tools. Everything is inside the
  pack + the already-shipped storage tools.

## Commit

One commit per task (4 commits), conventional-commit format, e.g.:
`feat(forge_document): per-source density floor with honest failure`
Do not push — leave local for review.

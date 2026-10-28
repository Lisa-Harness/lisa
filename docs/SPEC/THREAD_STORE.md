# Thread Store Format Specification

> **Status:** Maintained storage contract.
> **Implementation:** [runtime write path](../../runtime/src/runtime/thread/),
> [SDK read path](../../sdk/src/sdk/thread/), and
> [MCP facade](../../mcp_server/src/mcp_server/thread/server.py).

See [storage and backup](../STORAGE.md) for operational handling and
[context lifecycle](../CONTEXT_LIFECYCLE.md) for provider-visible projections.

## Overview

The Lisa thread store is a **lossless**, **append-only** conversation
history backed by a three-tier file layout:

1. **Active segment** — plain JSONL, append-only. Recent-N reads start here
   and spill into sealed history when needed. Bounded by `segment_turns` entries.
2. **Sealed segments** — zstd-compressed JSONL, one per sealed chunk.
   Sealing happens at append time when the active segment crosses
   `segment_turns` — no background scheduler, no cron.
3. **Sparse index** — one JSON line per sealed segment recording its
   turn/timestamp range. Size grows with the number of segments.

Content-addressed result objects live alongside these files and are part
of the durable evidence store.

The store is keyed by a **base path** (`<thread_dir>/<user_slug>`, no
file extension). All derived file paths are computed from this base.

## On-disk layout

| File | Purpose |
| --- | --- |
| `<base>.current.jsonl` | Active segment — plain JSONL, append-only |
| `<base>.<seg_id>.jsonl.zst` | Sealed segment — zstd-compressed JSONL |
| `<base>.index.jsonl` | Sparse segment index — one JSON line per sealed segment |
| `<base>.dict.zstd` | Trained zstd dictionary (optional, immutable once created) |
| `<base>.lock` | Cross-process advisory lock coordinating reads, appends, and segment seals |
| `<base>.results/r<sha256>.json` | Tool results, model request snapshots, continuation objects |

Where `<base> = <thread_dir>/<user_slug>` (or
`<thread_dir>/<user_slug>--<conversation_id>` when `LISA_CONVERSATION_ID` is
set) and `<seg_id>` is a zero-padded
4-digit counter (`0001`, `0002`, ...).

The legacy lossy digest sidecar (`<base>.digest.md` +
`<base>.digest.marker`) was **removed** in the memory × thread
realignment. The thread manifest addresses history for the model, and the
memory pack's mining slice distills it; no lossy digest exists. Stale
digest files on disk are simply ignored.

### Legacy migration

A one-time rename guard in `append_thread_entry` moves a legacy
`<base>.jsonl` to `<base>.current.jsonl` on first write. The old path is
never read again after the rename (hard cutover, no compat shims, no
dual-path read code).

## Record schemas

### Thread entry

One JSON object per line in the active and sealed segments:

```json
{
  "timestamp_utc": "2026-08-10T12:34:56.789000+00:00",
  "turn": 0,
  "user_slug": "local-user",
  "mode_id": "default",
  "query": "tell me about your context",
  "response_text": "here is my reflection",
  "provenance": {"source": "observed", "finish_reason": "stop", "model_id": "gpt-5.4-mini"}
}
```

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `timestamp_utc` | string (ISO 8601) | yes | UTC timestamp of the turn |
| `user_slug` | string | yes | User identity |
| `mode_id` | string | yes | Mode that handled the turn |
| `query` | string | yes | User's input |
| `response_text` | string | yes | Model/system response |
| `provenance` | object | no | Optional metadata (`source`, `finish_reason`, `model_id`) |
| `turn` | int | yes | 0-based global turn index |
| `effect_receipts` | list[object] | no | Each: `{capability_id, effect, artifact_id}` |
| `handoff_to` | string | no | Mode handoff target |
| `transitioned_to` | string | no | Stage transition target |
| `yielded` | bool | no | True when the turn paused to ask the user (YieldAction) |
| `yield_question` | string | no | The question surfaced when `yielded` is true |
| `turn_input` | object | no | Model request archive: runtime calls use `{"calls": [{"iteration": 1, "ref": "r<sha256>"}]}`; expand the result object to recover the exact request and `context_budget`. A kernel host without a recording hook can retain inline snapshots. Absent when no model call occurred. |
| `transcript` | list[object] | no | Verbatim non-system messages the turn added: its user message, every assistant tool call (with arguments), and every tool result. Replayed messages from earlier turns are excluded. The latest turn and its unfinished task (paused, awaiting approval, host-approved) are replayed at full fidelity; `thread_read_range(include_transcript=true)` returns older ones. Redaction applies on read. |
| `memory_ids` | list[int] | no | Memory records minted from this turn (auto-capture or `synthesize_and_store_learning`). The thread → memory provenance link. |

### Segment index record

One JSON line per sealed segment in `<base>.index.jsonl`:

```json
{
  "seg_id": "0001",
  "turn_lo": 0,
  "turn_hi": 255,
  "ts_lo": "2026-08-10T12:00:00+00:00",
  "ts_hi": "2026-08-10T14:30:00+00:00",
  "bytes_raw": 1048576,
  "bytes_zst": 218000
}
```

| Field | Type | Notes |
| --- | --- | --- |
| `seg_id` | string | Zero-padded 4-digit segment id |
| `turn_lo` | int | Global turn index of the first entry in this segment (0-based) |
| `turn_hi` | int | Global turn index of the last entry |
| `ts_lo` | string | Timestamp of the first entry |
| `ts_hi` | string | Timestamp of the last entry |
| `bytes_raw` | int | Uncompressed segment size |
| `bytes_zst` | int | Compressed segment size |

### Digest marker

Removed. The `mine_thread_segment` operation tracks its own per-segment
completion via a `mine.marker` sidecar written by the memory pack's
consolidation slice; the thread store no longer reads or writes a lossy
digest marker. Session `context.history.digest` configures a prompt projection,
not an on-disk digest sidecar.

## Write path

### Append

All appends and segment seals execute under the cross-process advisory lock
`<base>.lock` ([filelock.py](../../sdk/src/sdk/filelock.py)). Readers also
acquire `<base>.lock` when present so reads never observe a half-sealed
segment or unindexed rotation.

1. Acquire `<base>.lock` and resolve the authoritative monotonic `turn`
   index from `<base>.index.jsonl` + `<base>.current.jsonl` (advancing any
   stale pre-computed turn number from a concurrent turn).
2. Compute `current_path = <base>.current.jsonl`.
3. One-time rename guard: if `current_path` does not exist but the
   legacy `<base>.jsonl` does, rename it to `current_path`.
4. Count lines in `current_path`. If the count is >= `segment_turns`,
   seal the active segment first.
5. Append the new entry as JSON (`ensure_ascii=False`) to the (possibly
   fresh) active segment and return the committed `turn` index.

### Sealing

Sealing happens **synchronously at append time** when the active segment
reaches `segment_turns` entries. There is no background scheduler and no
cron — this matches the house rule "no cron in lightweight hosts". Sealing cost depends on payload size, compression, and filesystem; there
is no latency guarantee based only on the entry count.

1. Read the active segment bytes.
2. Compress with zstd (level 5, optional trained dictionary).
3. Write `<base>.<seg_id>.jsonl.zst`.
4. Compute the global turn range from the index (sum of prior sealed
   segment turn counts).
5. Append a `SegmentRecord` to `<base>.index.jsonl`.
6. Delete the active segment file (a fresh one is created by the next
   append).
7. Opportunistic dictionary retraining (every 10 seals).

### Dictionary training

- The **first seal** uses raw zstd (no dictionary).
- After the first seal, a zstd dictionary is trained from the sealed
  segment and stored atomically as `<base>.dict.zstd`.
- Once `<base>.dict.zstd` exists, it is **never overwritten in place** so
  earlier sealed segments compressed with that dictionary remain
  decompressable across the lifetime of the thread.

## Read paths

### Recent-N (`load_recent_thread_entries`)

1. Read `current.jsonl` from its end in 64 KiB blocks, parsing lines
   newest-first until `limit` valid entries are found. Bytes parsed are
   O(`limit`), not O(segment).
2. If it has fewer, take the tail of the most recent sealed segment: stream
   it through a bounded zstd decoder keeping only the newest raw lines
   resident, then parse backwards (the keep window grows if malformed lines
   eat the budget).
3. `limit <= 0` returns every active entry.

### Range query (`load_thread_range`) and single turn

1. Consult the index to find segments overlapping `[start_turn, end_turn]`.
2. Overlapping sealed segments are stream-decoded one line at a time and
   decoding stops at the window end. Sealed ordinals count valid entries, so
   lines before the window in the same segment are parsed transiently and
   discarded; nothing but the requested records is retained.
3. Active ordinals count **valid** entries (malformed lines never consume a
   turn number), exactly as before. A per-process, incrementally extended map
   of malformed line slots (keyed by path and a file-head fingerprint, reset
   if the file shrinks or is replaced) is built by one streaming pass that
   parses each line transiently; afterwards only newly appended lines are
   parsed, `next_turn_index` is O(appended lines), and reads parse only the
   lines they return. The first read of a process parses the active segment
   once, with bounded memory.
4. `ThreadStore.load_turns` fetches an arbitrary set of turns in one pass
   (used to build search pages).

### Search (`thread_search`)

Entries are scored as they stream past and only the top `limit` (at most 20)
are retained (`heapq.nlargest`); no candidate list is built.

### Result objects (`r<sha256>` refs)

Objects stay `{prefix}.results/<ref>.json` with the ref being the SHA-256 of
the stored bytes. `read_result_window` streams the canonical
`{"content":"…",…}` layout in 64 KiB blocks, hashes every byte, and verifies
the digest before returning, while retaining only the requested character
window (memory O(page), integrity unchanged). It still reads the whole file
because the ref hashes the whole payload. Objects in another key order, and
readers with a non-empty redaction policy (offsets count characters after
redaction), take the full-load path; `find_content` also full-loads.

### Lossless round-trip guarantee

zstd is a lossless compressor. Sealed segments decompress to the exact
bytes of the original active segment. The `load_thread_range` API gives
the harness a way to pull old turns back in by range on demand.

## Compaction model (post realignment)

**There is no compaction.** The thread store is the lossless record of
what happened; history is never truncated or summarized in place.

- **Sealing** is storage management (bounding the active segment for
  O(limit) reads), not history loss — sealed segments decompress to exact
  original bytes.
- **Context injection** is bounded by the manifest window: recent turns
  verbatim, older turns as a manifest table the model addresses on demand
  via `thread_read_range`.
- **Distillation** is owned by the memory pack's consolidation slice
  (`sdk/packs/memory/slice_provider.py`), which mines sealed segments
  opportunistically inside the post-turn slice budget and writes durable
  facts to `memory.db` with thread-turn provenance. It never mutates the
  thread.

The legacy lossy digest pipeline (`<base>.digest.md` sidecar + marker +
`FileBackedStateProvider._compact_thread`) was deleted in the memory ×
thread realignment (hard cutover, no compat shims).

## Configuration

| Env var | Default | Purpose |
| --- | --- | --- |
| `LISA_THREAD_DIR` | `<data_dir>/data/threads` | Thread directory |
| `LISA_CONVERSATION_ID` | `""` | Optional conversation identifier; when set, isolates the thread base path as `<thread_dir>/<user_slug>--<conversation_id>` |
| `LISA_THREAD_SEGMENT_TURNS` | `256` | Seal threshold — active segment is sealed when it reaches this many entries |

`LISA_THREAD_SEGMENT_TURNS` (256) controls how often the active segment
is sealed. Session `context.history` configures recent-message selection,
digest projection, and optional replay ceilings. Context windows and
retrieval pages stay bounded even as durable history grows.

## Scaling notes

### Segment size rationale

256 entries is the default seal threshold, not a performance promise.
Record sizes can differ substantially; test archive latency and memory
with representative data before changing the threshold.

### Seekable frames deferral

Whole-segment decompress is fine for 256-turn segments. For larger
segments, zstd's seekable frame format (`frame_dlog`) enables
byte-offset range reads within a compressed file without full
decompress. This is deferred until segment size grows beyond ~1000 turns
or decompress latency exceeds ~50ms. The deferral is marked with inline
`# TODO(seekable-frames):` comments at the compress/decompress call sites
in the [write implementation](../../runtime/src/runtime/thread/_store.py).

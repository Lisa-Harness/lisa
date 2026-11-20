# Evidence Envelope — Reader Changes Design (Spec 1 §5)

## Problem Statement and Constraints

The Evidence Envelope join-key work wired forensic identifiers (turn_id, call_id/attempt, dispatch_id/tool_call_id on tool.telemetry, parent_turn_id/parent_dispatch_id on child turns, run_ref/mode_digest on thread entries and ledger rows). But the forensic READER still joins everything by turn_index alone and does not read any of those keys. §5 is the consumer that makes the whole detour safe: it teaches ForensicReader and render_bundle to join on the new keys, render the delegation tree, name run manifests per turn and flag manifest changes, and report legacy/unreadable rows.

### Constraints

- **Reader compatibility**: accept every schema_version up to the current understood maximum; a row with no schema_version is version 0 (legacy) — read best-effort, **never rejected**, and counted in a read note; a row with a NEWER version than the reader understands is reported as unreadable, **not silently skipped**.
- **Join keys are never backfilled**: absent means unknown. Legacy rows still render, with a read note counting them.
- **SDK architecture**: `sdk` may import from `kernel`, never from `runtime`. The reader currently imports only from `sdk.forensics._join`, `sdk.forensics._logrec`, `sdk.forensics._models`, `sdk.layout`, `sdk.thread` — zero kernel imports. Keep it that way.
- **Hard-cutover repo**: no compatibility shims or stale aliases.
- **Engineering workflow**: smallest change that solves the root cause; preserve existing module boundaries and naming; row-level tests (drive real writers, read published rows back off disk — builders prove nothing).
- **Concurrency**: another agent is actively working on COMPACTION logic in `kernel/src/kernel/turn_phases/_compaction.py`, `kernel/src/kernel/config.py` intra-turn compaction settings, and `kernel/src/kernel/turn_runner/` wiring. The forensic reader lives in `sdk/src/sdk/forensics/` — a different package with zero kernel imports — so the collision surface is minimal, but the design must explicitly avoid touching those kernel files. Work happens in a fresh worktree stacked on the current dev tip.

## Concrete Design

### Schema version and constants

Introduce a new module `sdk/src/sdk/forensics/_schema.py` containing:

```python
SCHEMA_VERSION_CURRENT = 3  # TokenUsageEvent (v3); evidence topics v2
```

This constant is the single source of truth for the reader's understood maximum. A `schema_version` value of `None` or `0` is treated as legacy; any integer `> SCHEMA_VERSION_CURRENT` is unreadable. The reader never emits `schema_version` itself; it only consumes rows written by kernel writers.

### _models.py changes

Keep `ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=False)` on every model. All new fields are optional with defaults that mean "unknown": strings default to `""`, integers/numbers default to `None`, and sequences default to `()`.

**`DecisionRecord`** (currently line 40) add:

```python
call_id: str = ""
attempt: str | None = None
request_ref: str = ""
```

**`DispatchRecord`** (currently line 60) add:

```python
dispatch_id: str = ""
tool_call_id: str = ""
```

**`CostRecord`** (currently line 77) add:

```python
call_id: str = ""
attempt: str | None = None
iteration: int | None = None
request_ref: str = ""
```

**`TurnEvidence`** (currently line 117) add:

```python
turn_id: str = ""
parent_turn_id: str = ""
parent_dispatch_id: str = ""
run_ref: str = ""
mode_digest: str = ""
```

**`ApprovalEvidence`** — no new join keys are required for approval rows; they continue to join by turn_index. Keep unchanged.

**`ForensicBundle`** (currently line 195) add:

```python
delegation_tree: tuple[DelegationNode, ...] = ()
run_manifest_changes: tuple[RunManifestChange, ...] = ()
```

**`BundleSources`** (currently line 172) add:

```python
legacy_row_count: int = 0
unreadable_row_count: int = 0
```

New grouping shapes (place in `_models.py` after the existing record classes):

```python
class CallCost(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=False)
    call_id: str
    attempt: str | None = None
    iteration: int | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    request_ref: str = ""


class DelegationNode(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=False)
    turn_index: int
    turn_id: str = ""
    parent_turn_id: str = ""
    parent_dispatch_id: str = ""
    children: tuple[DelegationNode, ...] = ()


class RunManifestChange(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=False)
    turn_index: int
    previous_run_ref: str = ""
    previous_mode_digest: str = ""
    current_run_ref: str = ""
    current_mode_digest: str = ""
```

### _join.py changes

Extend each parser to read the new keys via `.get()` with the same default meaning: absent → `""` for strings, `None` for ints, and `None` treated as `""` (stamp_envelope fills unknown lineage keys with an explicit null). No backfilling.

- `decision_from_record`: add call_id/attempt/request_ref (attempt via str coercion: absent or None → None; str or int → str).
- `dispatch_from_record`: add dispatch_id, tool_call_id.
- `cost_from_record`: add call_id, attempt (str coercion), iteration, request_ref.
- `approval_from_record`: unchanged.
- `turn_from_entry`: add turn_id, parent_turn_id, parent_dispatch_id, run_ref, mode_digest — all absent → ""; None → "".

Extend `build_bundle` to:

- **(a) Group token rows by call_id under iteration** — after joining cost records to turns, build per-call groupings for records with a non-empty call_id: a list `call_costs` grouped by (turn_index, call_id) and rendered under their iteration; legacy rows (empty call_id) keep the existing aggregate rendering.
- **(b) Join dispatch rows to transcript tool calls by tool_call_id** — the dispatch's `tool_call_id` is carried on the DispatchRecord so a bundle consumer can correlate a dispatch row to the transcript's tool result; the renderer names the tool_call_id on the dispatch line. A dispatch with no tool_call_id renders without the join reference.
- **(c) Build delegation tree from parent_turn_id/parent_dispatch_id** — after turns are assembled, construct DelegationNode objects: every turn gets its own node; turns with a non-empty parent_turn_id are appended to their parent's children when the parent turn exists in the bundle; roots are the remaining nodes (a top-level turn has turn_id but no parent keys). A child whose parent is outside the bundle window stays a root but keeps its parent keys recorded.
- **(d) Collect run_ref/mode_digest per turn and detect consecutive-turn manifest changes** — iterate turns in order; for consecutive turns where either run_ref or mode_digest differs (comparing turns that carry the keys), append a RunManifestChange naming the boundary turn.
- **(e) Count legacy and unreadable rows** — the reader counts them while reading; it passes the counts to build_bundle, which stores them in BundleSources and adds the read notes.

### _logrec.py changes

Add one helper in `_logrec.py`:

```python
def check_schema_version(record: dict[str, Any]) -> tuple[int, bool]:
    """Return (version, is_unreadable).

    version 0 for missing/0/None/unparseable; version > SCHEMA_VERSION_CURRENT
    is unreadable. String digits are accepted (int coercion).
    """
    raw = record.get("schema_version", 0)
    try:
        version = int(raw or 0)
    except (TypeError, ValueError):
        return 0, False
    if version > SCHEMA_VERSION_CURRENT:
        return version, True
    return version, False
```

### reader.py changes

Apply the check in each ledger read (`_read_decisions`, `_read_costs`, `_read_approvals`) plus the log reads (`_read_log_errors`) — after ownership filtering and before the turn_index filtering, so the counts cover the read window:

- unreadable rows (version > current): increment `unreadable_row_count`, remove them from the result set, and append a note naming the highest version seen; never silently skipped.
- version-0 rows: increment `legacy_row_count`, keep them, and append a note counting them.
- The counts ride BundleSources so the totals survive to the bundle.

### render.py changes

- **`_render_cost`**: when a turn's joined costs carry call ids (new writer rows), render one line per call, grouped by iteration: `- cost (per call): ...` with `call {call_id}`, attempt when known, iterations in ascending order. Legacy rows (no call_id) keep the existing aggregate line. The aggregate line stays when any legacy row exists.
- **`_render_dispatches`**: name `tool_call_id` on the dispatch line when present (`-> tool_call {id}`); absent renders as before.
- **Delegation tree**: new `_render_delegation_tree` — one line per root node plus indented children: `- turn {index} (turn_id=...)` and `  - turn ...` lines; nodes without children are leaves. Rendered when the tree has any nodes or when any turn carries a turn_id or parent key.
- **Run-manifest lines**: in `_render_turn_header`, when run_ref or mode_digest is present, render `- run: {run_ref} · modes: {mode_digest}`.
- **Delegation/manifest sections**: after the turns, `### Delegation tree` and `### Run manifest changes` sections when non-empty; a change line reads `- at turn N: {...} -> {...}` rendering only the differing keys.
- **Legacy/unreadable counts**: in `_render_sources`, when either count is positive, add `- schema reads: {legacy_row_count} legacy rows read best-effort, {unreadable_row_count} unreadable rows skipped and counted`.

### Data Shapes

All new fields are optional with defaults meaning "unknown". The grouping dataclasses live in `_models.py` and are used by `_join.py` and `render.py`.

### Error Handling

- **Legacy (version 0)**: rows parse best-effort; new join keys default ""/None; rendering falls back to existing behavior; count recorded and surfaced.
- **Newer version**: rows are counted in BundleSources.unreadable_row_count, reported in a note, and not parsed.
- **Missing keys**: absent means unknown; never backfilled.
- **Absent ledger**: existing behavior preserved (missing sources and notes).

### Sequence of Changes

1. Create `sdk/src/sdk/forensics/_schema.py` with `SCHEMA_VERSION_CURRENT = 3`.
2. Extend `_models.py`: new fields on existing records; add `CallCost`, `DelegationNode`, `RunManifestChange`; add `delegation_tree`/`run_manifest_changes` on ForensicBundle and counts on BundleSources.
3. Extend `_join.py` parsers and `build_bundle` (grouping, delegation tree, manifest changes, counts).
4. Add `check_schema_version` to `_logrec.py`.
5. `reader.py`: apply the version check in each read, count, note, and pass counts through.
6. `render.py`: per-call cost, tool_call joins, delegation tree, run-manifest lines and change sections, counts.
7. New tests: `sdk/tests/test_forensics_reader_joins.py` plus additions to the existing forensics tests.
8. Run the full SDK test suite and the existing forensic tests.

### Alternatives Considered

- **Joining by turn_index only** — rejected; that is the status quo the spec replaces.
- **Putting version logic in each parser vs one helper** — one helper in `_logrec.py` avoids duplication and keeps behavior consistent.
- **Rendering per-call cost as a flat list vs grouped under iteration** — grouped under iteration chosen because a flat list would lose the iteration context.
- **A separate lineage module vs extending `_join.py`** — extending `_join.py` chosen because lineage data is already assembled there and a new module would be overkill for this scope.

### Acceptance Checks

New test file `sdk/tests/test_forensics_reader_joins.py`:

- **Streaming-fallback turn produces two turn_input.calls with distinct call_id/attempt and two token rows that join** — drive a real writer producing a streaming turn with fallback; read rows back off a disk via ForensicReader; assert the per-call cost groupings carry two distinct call ids and attempts, and aggregate totals match.
- **Failed dispatch joins to exactly one transcript tool result by tool_call_id** — write a dispatch and its `tool.telemetry` row sharing one `tool_call_id`; assert the rendered dispatch line names the join and the dispatch record carries the id.
- **Delegate/fork child rows carry own turn_id + parent's turn_index + parent_dispatch_id** — seed parent and child thread entries; assert the child's TurnEvidence carries `parent_turn_id`/`parent_dispatch_id`, and the rendered tree shows the child under the parent.
- **Top-level turn has turn_id but no parent keys** — assert top-level TurnEvidence has a non-empty turn_id with empty parent keys; the tree treats it as a root.
- **Resumed continuation keeps original turn_id** — entries sharing one turn_id across a resumed continuation render one node/turn identity.
- **Pack reload changes run_ref and the next entry records it** — two entries with different run_ref produce a RunManifestChange with correct before/after values.
- **Legacy rows still render with a read note counting them** — rows without schema_version produce `bundle.sources.legacy_row_count > 0` and notes mentioning them.
- **Unreadable rows are reported, not dropped** — rows with schema_version > SCHEMA_VERSION_CURRENT produce `bundle.sources.unreadable_row_count > 0` and a note; they are excluded from joins.

Existing tests `sdk/tests/test_forensics_approvals.py`, `test_forensics_chain_note.py`, `test_forensics_sequence_gaps.py` must remain green; run the full SDK suite.

### Concurrency Guardrails

- **Do NOT touch**: `kernel/src/kernel/turn_phases/_compaction.py`, `kernel/src/kernel/config.py` compaction settings, `kernel/src/kernel/turn_runner/`.
- Work only in `sdk/src/sdk/forensics/` and `sdk/tests/`.
- Worktree stacked on the dev tip at creation (d250a6fa); rebase if dev moves.
- No kernel import changes; zero kernel imports in the reader.

<!-- generated by lisa · provider: ollama · model: deepseek-v4-pro:cloud · 2026-11-15 -->

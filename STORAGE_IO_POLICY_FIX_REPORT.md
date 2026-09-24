# Direct I/O policy — framework CI is red

**Status:** `uv run python scripts/check_direct_io_policy.py` exits **1** on
`main` (20 in-tree violations). Committed, not local WIP — the scanner and
all referenced host files are unmodified.

**Cause:** `scripts/check_direct_io_policy.py` was last updated 2026-09-14,
but the host package was refactored after that (2026-09-13 → 2026-09-19).
The `_ALLOWLIST` map and the code have drifted apart in three ways: deleted
files, changed call sites, and never-allowlisted files.

**Blast radius:** framework-only. External packs cannot fix this and should
not be blocked on it. The external-pack scan is advisory (exit 0) and
unaffected by these changes.

> Unrelated note: the working tree already contains 4 modified `kernel/`
> files (`mcp_launch/compiler.py`, `proxy_launch/compiler.py`, two tests)
> plus untracked `docs/blog/`. Those are **not** part of this report and were
> not touched by it.

## Verified patch

The diff below was **applied and confirmed** to take the scanner to exit 0
**and** leave zero unused snippets, then reverted — adding allowlist entries
to a security gate is a policy decision, so it is left for review. The
working tree is untouched: only this report and `docs/blog/` are new, plus
four pre-existing `kernel/` modifications that are unrelated to this work.

Net effect: 25 insertions, 28 deletions, `_ALLOWLIST` 21 keys → 20 keys,
53 snippets → 52, all live.

```diff
--- a/scripts/check_direct_io_policy.py
+++ b/scripts/check_direct_io_policy.py
@@ -75,18 +75,10 @@ _WRITE_OPEN_MODES: frozenset[str] = frozenset({"w", "a", "x", "wb", "ab", "xb"})
 # Allowed direct-write callsites, keyed by repo-relative path.
 # Each value is a tuple of line substrings that are permitted in that file.
 _ALLOWLIST: dict[str, tuple[str, ...]] = {
-    "mcp_server/src/mcp_server/storage/server.py": (
-        "resolved.write_text(",
-        "path.write_text(",
-    ),
     "host/src/host/content_roots.py": (
         "path.write_text(",
         "path.parent.mkdir(parents=True, exist_ok=True)",
     ),
-    "host/src/host/turn_hooks_builder.py": (
-        "thread_path.write_text(",
-        "thread_path.parent.mkdir(parents=True, exist_ok=True)",
-    ),
     "host/src/host/code_artifacts.py": (
         "temporary_path.write_text(",
         "os.replace(temporary_path, self._path)",
@@ -99,35 +91,42 @@ _ALLOWLIST: dict[str, tuple[str, ...]] = {
     ),
     "host/src/host/intake.py": (
         "state_path.write_text(",
-        "data_dir.mkdir(parents=True, exist_ok=True)",
+        "state_path.parent.mkdir(parents=True, exist_ok=True)",
     ),
     "host/src/host/recovery.py": (
         "self._path.parent.mkdir(parents=True, exist_ok=True)",
         "temp_path.write_text(",
         "os.replace(temp_path, self._path)",
     ),
-    "host/src/host/state_provider.py": (
-        "digest_path.write_text(",
-        "self._thread_path.write_text(",
-    ),
     "host/src/host/model_discovery/manifest.py": (
         "tmp.write_text(",
         "path.parent.mkdir(parents=True, exist_ok=True)",
         "tmp.unlink(missing_ok=True)",
     ),
-    "host/src/host/session_context_fetch.py": (
-        "cache_meta.write_text(",
-        "cache_md.write_bytes(",
-        "cache_md.parent.mkdir(parents=True, exist_ok=True)",
-    ),
-    "host/src/host/thread.py": (
-        'thread_path.open("a"',
+    "host/src/host/thread/__init__.py": (
         "thread_path.parent.mkdir(parents=True, exist_ok=True)",
+        "paths.current.parent.mkdir(parents=True, exist_ok=True)",
     ),
-    "host/src/host/workflow_instance_store.py": (
-        "self._path.parent.mkdir(parents=True, exist_ok=True)",
-        "temp_path.write_text(",
-        "os.replace(temp_path, self._path)",
+    "host/src/host/thread/_store.py": (
+        "index_path.parent.mkdir(parents=True, exist_ok=True)",
+        "dict_path.parent.mkdir(parents=True, exist_ok=True)",
+        "dict_path.write_bytes(dict_data.as_bytes())",
+        "sealed_path.parent.mkdir(parents=True, exist_ok=True)",
+        "sealed_path.write_bytes(compressed)",
+        "paths.current.unlink()",
+    ),
+    "host/src/host/policy.py": (
+        "path.parent.mkdir(parents=True, exist_ok=True)",
+        "path.write_text(header + body, encoding=\"utf-8\")",
+    ),
+    "host/src/host/testing/runtime_factory.py": (
+        "manifest_dir.mkdir(parents=True, exist_ok=True)",
+        "(manifest_dir / \"models.yaml\").write_text(",
+    ),
+    "sdk/packs/memory/src/sdk/packs/memory/slice_provider.py": (
+        "path.parent.mkdir(parents=True, exist_ok=True)",
+        "path.write_text(",
+        "path.write_text(seg_id, encoding=\"utf-8\")",
     ),
     "host/src/host/commands/_packs.py": (
         "packs_venv.parent.mkdir(parents=True, exist_ok=True)",
@@ -159,14 +158,12 @@ _ALLOWLIST: dict[str, tuple[str, ...]] = {
         "target_dir.mkdir(parents=True, exist_ok=True)",
     ),
     "sdk/src/sdk/storage/_backend.py": (
-        "(data_dir / _FILES_SUBDIR).mkdir(parents=True, exist_ok=True)",
-        "(data_dir / _CHECKPOINTS_SUBDIR).mkdir(parents=True, exist_ok=True)",
+        "(data_dir / DATA_DIR / _FILES_SUBDIR).mkdir(parents=True, exist_ok=True)",
+        "shutil.rmtree(legacy_checkpoints)",
         "resolved.parent.mkdir(parents=True, exist_ok=True)",
         "resolved.write_text(content, encoding=\"utf-8\")",
         "resolved.unlink()",
         "resolved.mkdir(parents=True, exist_ok=True)",
-        "checkpoints_root.mkdir(parents=True, exist_ok=True)",
-        "path.write_text(json.dumps(bundle, default=str, indent=2), encoding=\"utf-8\")",
         "path.parent.mkdir(parents=True, exist_ok=True)",
         "path.write_text(content, encoding=\"utf-8\")",
     ),
```

Note the six deleted keys: three were **fully dead** before this patch
(`storage/server.py`, `turn_hooks_builder.py`, `state_provider.py`) — those
files still exist but contain no I/O. See the follow-on finding below for
why that matters.

**Deletions (dead keys and dead snippets — also in the patch above):**

| Entry | Action |
| --- | --- |
| `mcp_server/src/mcp_server/storage/server.py` (both snippets) | Delete key — implementation moved to `sdk/` |
| `host/src/host/turn_hooks_builder.py` (both) | Delete key — file has no I/O left |
| `host/src/host/state_provider.py` (both) | Delete key — file has no I/O left |
| `host/src/host/session_context_fetch.py` | Delete key — file deleted |
| `host/src/host/thread.py` | Delete key — moved to `thread/` package |
| `host/src/host/workflow_instance_store.py` | Delete key — file deleted |
| `sdk/.../_backend.py`: `_CHECKPOINTS_SUBDIR` line, `checkpoints_root.mkdir(...)`, `path.write_text(json.dumps(bundle, ...))` | Delete — checkpoint store removed |
| `host/src/host/intake.py`: `data_dir.mkdir(...)` | Replace with `state_path.parent.mkdir(...)` |

Result after applying: `uv run python scripts/check_direct_io_policy.py`
→ exit **0**, no in-tree violations, and the unused-snippet diagnostic
reports **0** unused snippets.

> **Review point:** this patch is the minimum to unbreak CI, but the three
> new host-internal entries (`thread/`, `policy.py`, `runtime_factory.py`)
> are carve-outs by convention only — nobody has reviewed whether they
> *should* be carved out. The reasoning for each is in section 3.

---

## Summary

| Cause | Files | Violations |
| --- | --- | --- |
| Allowlist snippet drift (file allowlisted, snippet no longer matches) | 2 | 3 |
| Never allowlisted | 5 | 17 |
| **Total** | | **20** |

Separately, **6 of 21 allowlist keys are entirely dead** and 19 of 53
snippets were unused — a silent-rot problem, detailed under
[Follow-on finding](#follow-on-finding-the-allowlist-has-rotted-silently).

---

## 1. Stale `_ALLOWLIST` keys (dead entries)

These keys point at files that no longer exist. They are harmless at runtime
but they hide the *current* carve-out targets — the same code now lives
elsewhere and is unallowlisted.

| Dead key | Deleted by |
| --- | --- |
| `host/src/host/thread.py` | `d257af7` (2026-09-13) "refactor(host): split host/thread.py into a thread package" |
| `host/src/host/session_context_fetch.py` | `d839887` (2026-09-19) "refactor!(host): local-only session context; drop resume + refresh knobs" |
| `host/src/host/workflow_instance_store.py` | `d839887` (2026-09-19) |

### Fix

Move each entry's snippets to the file that inherited the code:

- `thread.py` → **`host/src/host/thread/__init__.py`** and
  **`host/src/host/thread/_store.py`**
- `session_context_fetch.py` / `workflow_instance_store.py` → confirm whether
  the writers were deleted outright (then drop the keys) or relocated (then
  re-point them). `d839887` dropped a lot of persistence, so deletion is
  likely for at least one.

Note the current key for `host/src/host/thread.py` only allowlists
`thread_path.open("a"` and `thread_path.parent.mkdir(...)`, but the refactor
split the writer across two files with 8 new callsites. A path-keyed
allowlist does not survive file→package refactors; see the recommendation at
the end.

---

## 2. Allowlist snippet drift (file allowlisted, snippet no longer matches)

### 2.1 `host/src/host/intake.py:409` — the mkdir target was renamed

**Allowlist says:**

```python
"host/src/host/intake.py": (
    "state_path.write_text(",
    "data_dir.mkdir(parents=True, exist_ok=True)",
),
```

**Code says** (line 408-409):

```python
state_path = data_dir / STATE_DIR / _DEFAULT_STATE_FILE
state_path.parent.mkdir(parents=True, exist_ok=True)
```

The mkdir target changed from `data_dir` to `state_path.parent` when the
state file moved under the `state/` lifecycle category. The snippet never
matched `state_path.parent.mkdir(...)`.

**Fix:** replace the snippet with
`"state_path.parent.mkdir(parents=True, exist_ok=True)"`.

### 2.2 `sdk/src/sdk/storage/_backend.py` — 2 stale snippets, and this is the sandbox itself

The allowlist entry has 10 snippets; 7 of the 9 flagged lines match, and the
2 that do not are both drift:

**Allowlist contains:**

```python
"(data_dir / _FILES_SUBDIR).mkdir(parents=True, exist_ok=True)",
"(data_dir / _CHECKPOINTS_SUBDIR).mkdir(parents=True, exist_ok=True)",
```

**Code says:**

```python
# line 85 — DATA_DIR prefix added since the snippet was written
(data_dir / DATA_DIR / _FILES_SUBDIR).mkdir(parents=True, exist_ok=True)

# line 92 — legacy checkpoints teardown; no allowlist snippet covers it
shutil.rmtree(legacy_checkpoints)
```

`_CHECKPOINTS_SUBDIR` no longer exists anywhere in the codebase — it is
defined only inside the allowlist string that names it. The
`shutil.rmtree(legacy_checkpoints)` call was added alongside the
"One-time migration" comment and never allowlisted.

**This file is the sandbox implementation.** It is the one carve-out that
must always pass. If it fails, the check is meaningless.

**Fix — update the two stale snippets:**

```python
"sdk/src/sdk/storage/_backend.py": (
    "(data_dir / DATA_DIR / _FILES_SUBDIR).mkdir(parents=True, exist_ok=True)",
    "shutil.rmtree(legacy_checkpoints)",
    "resolved.parent.mkdir(parents=True, exist_ok=True)",
    "resolved.write_text(content, encoding=\"utf-8\")",
    "resolved.unlink()",
    "resolved.mkdir(parents=True, exist_ok=True)",
    "checkpoints_root.mkdir(parents=True, exist_ok=True)",
    "path.write_text(json.dumps(bundle, default=str, indent=2), encoding=\"utf-8\")",
    "path.parent.mkdir(parents=True, exist_ok=True)",
    "path.write_text(content, encoding=\"utf-8\")",
),
```

Delete the `_CHECKPOINTS_SUBDIR` line and fix the `_FILES_SUBDIR` one. The
other 8 snippets are still live — leave them.

> **How matching works:** the scanner records `node.lineno` and compares the
> snippet against `lines[line_no - 1].strip()` — i.e. the **whole stripped
> source line** where the call starts. This is a substring test
> (`snippet in line_text`), so a snippet must appear verbatim within that one
> line. A call split across lines exposes only its opening fragment, and a
> snippet with a different prefix (like the missing `DATA_DIR /`) never
> matches even though the call is obviously the same.

---

## 3. Never allowlisted (5 files, 17 violations)

| File | Lines | Nature | Recommendation |
| --- | --- | --- | --- |
| `host/src/host/thread/_store.py` | 122, 197, 198, 286, 287, 311 | Thread store: index mkdir, compaction dict write, sealed-segment write, unlink on rotate | **Allowlist** — inherits the old `thread.py` carve-out |
| `host/src/host/thread/__init__.py` | 163, 207 | Thread path mkdir, current-path mkdir | **Allowlist** — same inheritance |
| `host/src/host/policy.py` | 471, 490, 516 | Policy rule store: mkdir ×2, `path.write_text(header + body, ...)` | **Allowlist** — host-internal config writer, added in `845cb39` (2026-09-05) |
| `host/src/host/testing/runtime_factory.py` | 75, 76 | Test-runtime fixture writing `models.yaml` | **Allowlist or exclude** — test-support module inside a shipped package |
| `sdk/packs/memory/src/sdk/packs/memory/slice_provider.py` | 159, 160, 328, 329 | Turn-counter sidecar read/write | **Allowlist** — see note below |

### Note on `runtime_factory.py`

`_is_runtime_python` excludes paths containing `/tests/` and filenames
starting with `test_`, but `host/src/host/testing/runtime_factory.py` is
neither. It is test-support code that ships inside the host package.

Either add a `host/testing/` exclusion to `_is_runtime_python`, or allowlist
the two snippets. Excluding is cleaner — nothing in that module is a
production writer — but it widens the exclusion surface, so allowlisting is
the lower-risk change.

### Note on the memory slice provider

`sdk/packs/memory/src/sdk/packs/memory/slice_provider.py` is an **in-tree
pack** that the scanner treats as in-tree code, so it needs a carve-out too.
It persists a turn-counter sidecar via direct `pathlib`.

The external `letta` pack does the identical thing
(`lisa-packs/letta/lisa_packs/letta/slice_provider.py:88,101,102`) and is
flagged advisory. Whichever way this is decided should be applied to both —
see the `docs/STORAGE_IO_POLICY_SPEC.md` section on slice providers in
`../lisa-packs`.

---

## Verification

After patching:

```bash
uv run python scripts/check_direct_io_policy.py   # expect exit 0
```

---

## Follow-on finding: the allowlist has rotted silently

Applying the unused-snippet check below to the **current, unpatched**
scanner produces a result that is worse than the CI failure:

```
allowlist keys: 21   snippets: 53   unused: 19

FULLY DEAD KEYS (6):
  mcp_server/src/mcp_server/storage/server.py     [2/2]
  host/src/host/turn_hooks_builder.py             [2/2]
  host/src/host/state_provider.py                 [2/2]
  host/src/host/session_context_fetch.py          [3/3]   <- FILE DELETED
  host/src/host/thread.py                         [2/2]   <- FILE DELETED
  host/src/host/workflow_instance_store.py        [3/3]   <- FILE DELETED

PARTIALLY DEAD (2):
  host/src/host/intake.py                         [1/2]
  sdk/src/sdk/storage/_backend.py                 [4/10]
```

**6 of 21 keys are entirely dead and 19 of 53 snippets are unused.**

`_is_allowlisted` passes if *any* snippet matches, so dead snippets are
completely inert — they neither break the check nor protect anything. The
allowlist can rot indefinitely without a single signal, and it has been
rotting across at least four refactors.

Most of this is benign cleanup, but three entries are informative:

- **`mcp_server/src/mcp_server/storage/server.py`** — both snippets
  (`resolved.write_text(`, `path.write_text(`) are dead because the sandbox
  implementation moved to `sdk/src/sdk/storage/_backend.py`. This was the
  *intended* carve-out for the storage server; the entry now carves out
  nothing, and the live code needs new entries instead (section 2.2).
- **`host/src/host/turn_hooks_builder.py`, `state_provider.py`** — the files
  still exist but contain **zero** I/O primitives; their writes moved into
  `host/thread/` (verified: `index_path` / `digest_path` writers now live in
  `thread/_store.py`). These are dead keys pointing at files that merely
  stopped doing I/O.
- **`sdk/src/sdk/storage/_backend.py`** — 4/10 unused. Two are the drift
  bugs causing the CI failure; two (`checkpoints_root.mkdir(...)`,
  `path.write_text(json.dumps(bundle, ...))`) reference a checkpoint store
  that no longer exists in the file.

### Diagnostic

Drop this in as a second gate, or run it ad hoc:

```bash
uv run python - <<'PY'
import sys
sys.path.insert(0, "scripts")
import check_direct_io_policy as c

root = c._repo_root()
used = set()
for path in c._iter_in_tree_files(root):
    rel = path.relative_to(root).as_posix()
    snippets = c._ALLOWLIST.get(rel)
    if not snippets:
        continue
    src = path.read_text(encoding="utf-8")
    for snippet in snippets:
        if snippet in src:
            used.add((rel, snippet))

dead = 0
for rel, snippets in c._ALLOWLIST.items():
    unused = [s for s in snippets if (rel, s) not in used]
    dead += len(unused)
    if unused and len(unused) == len(snippets):
        print(f"DEAD KEY  {rel}")
    for snippet in unused:
        print(f"  unused  {snippet!r}")
print(f"\n{dead} unused snippets")
PY
```

A dead key is not automatically wrong — a file may legitimately retain
authorization while having no current callsites — but six wholly-dead keys
against a 21-key map means the map is tracking history rather than
enforcement.

---

## Recommendation: stop keying the allowlist by path + snippet

Four refactors (`845cb39`, `d257af7`, `d839887`, and the storage extraction)
each silently broke or hollowed out this check. The failure mode is
inverted: **a dead snippet makes CI pass silently, while a renamed target
makes CI fail loudly.** Nothing verifies that a snippet is still used, so
the allowlist rots toward uselessness while the only feedback loop is the
occasional false CI failure.

Two options, in increasing order of durability:

1. **Add the unused-snippet check above to the scanner itself** and fail
   when an allowlist entry is dead or unreferenced. Cheap, and it catches
   the `intake.py` / `_backend.py` drift class immediately. It also makes
   the 6 dead keys visible so they can be deleted deliberately.

2. **Key carve-outs by directory, not by file + line text.** Every current
   violation is a host-internal lifecycle-category writer (`state/`, `run/`,
   `cache/`, `config/`). A small set of declared roots — e.g. "everything
   under `host/src/host/thread/`", "`host/policy.py`",
   "`sdk/src/sdk/storage/_backend.py`" — survives file moves and snippet
   rewrites and would have produced zero false positives across all four
   refactors.

Option 1 is the minimum to make this check trustworthy again.

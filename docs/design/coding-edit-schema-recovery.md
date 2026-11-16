# Coding edit schema recovery

> **Status:** Implemented. The recovery hook lives in the framework kernel
> (`kernel.tools.validation_recovery`, wired into the tool registry's
> parameter validation); the catalog declaration is translated by the SDK
> catalog loader; the classifier, digest case normalization, authored
> recovery content, and scenarios live in the coding-pack repository.
> Verified by deterministic pack tests, packstest evals driving the real
> kernel dispatch (including the approval round-trip), the offline scripted
> e2e scenario `coding/edit-recovery-scripted`, and the opt-in live scenario
> `coding/edit-recovery-live`. Telemetry distinguishes rejection
> (`tool.telemetry` error rows carry the recovery text) from successful
> execution; per-key counters remain a follow-up.

## Problem

`coding_edit_files` rejects malformed model arguments before executing a
write. Current failures expose Pydantic paths such as
`edits.0.expected_sha256`, but they do not consistently tell the model which
valid record shape to send next. Under long-context pressure, the model can
repeat the same rejected batch and consume its tool/iteration allowance.

The edit tool already has bounded normalization for a JSON-string `edits`
array and for an exact legacy flat-record signature. This proposal adds the
remaining low-risk normalization and authored, actionable recovery text. It
must not relax exact-text matching, worktree ownership, complete-file digest
checks, or approval policy.

## Current contracts to preserve

- Arguments are validated by `FileEdit` and `Parameters` in the coding pack's
  `tools/edit_files/edit_files.py` before the edit executes.
- Kernel tool dispatch canonicalizes parameters, validates against the tool's
  Pydantic schema, and raises `ToolExecutionError` on failure. Pre-execution
  rejections are published to decision telemetry.
- An `update` requires a complete current-file SHA-256 digest and non-empty
  exact replacements; a `create` requires content; a `delete` requires only
  the digest. `old_text` must match exactly once.
- The existing JSON-string decoder accepts only an array of objects, with
  bounded tail recovery. The legacy lift applies only to an exact recognized
  flat shape with a valid lift source.
- The `.tool.yml` descriptor is part of the model-facing contract. Authored
  prompts, hints, and user-facing error text live in versioned Markdown loaded
  through `ctx.host.read_authored_text`; missing content fails loudly.

## Scope

### In scope

1. A compact, structured tool description with an explicit example.
2. Recovery text for the common validation families, preserving the original
   field-level error for diagnostics.
3. Case normalization for hexadecimal complete-file digests.
4. Focused deterministic tests, packstest evals, and an opt-in live scenario
   that demonstrates recovery without a write on the rejected attempt.
5. Telemetry that distinguishes validation, normalization, and execution
   failures without logging edit contents.

### Out of scope

- Guessing, recomputing, truncating, or accepting a placeholder digest.
- General repair of malformed JSON, invalid UTF-8, unknown record kinds, or
  ambiguous replacement spans.
- Accepting arguments-level `path`/`expected_sha256` for current kind-shaped
  records, beyond the existing exact legacy lift.
- Automatic repeated dispatch of a rejected edit.
- Changing kernel-wide validation errors or weakening `extra="forbid"`
  semantics elsewhere.
- Relaxing `coding_record_fact` length limits or unrelated Git/read tools.

## Authored model-facing contract

Replace the single long descriptor paragraph with a concise description plus
a small shape-first example. Keep the full safety contract in the descriptor
or its linked versioned content:

```yaml
edits:
  - kind: update
    path: relative/path.py
    expected_sha256: 64-lowercase-hex-characters-from-current-read
    replacements:
      - old_text: exact text occurring once
        new_text: replacement text
```

The text must state, in separate short rules:

- `edits` is an array, not an object and not a string-encoded array;
- every record contains its own `kind` and `path`;
- `update` uses the complete current file's digest plus `replacements`;
- `create` uses `content` and no digest/replacements;
- `delete` uses the digest and no content/replacements;
- obtain the digest from the current `coding_read_file` result, not memory;
- after a successful edit or external file change, re-read before another
  update rather than reusing a stale digest.

The JSON-string decoder remains a normalization convenience; authored content
must still teach the native array shape rather than presenting string
encoding as preferred.

## Validation and normalization design

### Hexadecimal digest normalization

Add a `field_validator("expected_sha256", mode="before")` on `FileEdit` that
applies only when the supplied value is a string:

1. Do not trim, repair, or infer the value.
2. If it matches exactly 64 hexadecimal characters in either case, convert
   `[A-F]` to lowercase.
3. Pass every other value through unchanged so the existing pattern and shape
   validators reject it.

Record a `normalized_fields` entry containing only
`edits.<index>.expected_sha256_case` for changed values. The field name and
normalization kind are telemetry, not the digest itself.

This is safe because hexadecimal uppercase/lowercase spellings denote the
same digest. The normalized value must still match the independently computed
snapshot digest before a write.

### Preserve existing bounded normalizers

Retain `_accept_string_edits` and `_maybe_lift_flat_string_records` as narrow
normalization layers before structural validation. Do not add a fallback that
turns arbitrary scalars into lists, drops unknown fields, converts malformed
records into deletes, or substitutes a newly computed digest.

The shared kernel already tracks normalized fields from its generic payload
normalization. Extend the pack-to-kernel result path only if necessary to
preserve pack-declared normalization metadata; do not bypass the kernel's
validation or dispatch sequence to report it.

### Structured recovery details

Introduce a small pack-owned function that maps a Pydantic
`ValidationError` to one authored recovery key. Match by structured error
`loc`, `type`, and the existing kind-shape validator context, not by parsing
localized prose. Return a typed result:

```text
EditValidationRecovery(
    recovery_key,
    field_paths,
    original_validation_summary,
    normalized_fields,
)
```

Recovery keys:

| Key | Trigger | Recovery instruction |
| --- | --- | --- |
| `edits_not_array` | `edits` is not a list after existing decoding | Send one JSON array under `edits`; include one record per file action |
| `record_shape` | Wrong fields or kind-specific shape violation | Show the minimal update/create/delete shape and require the record's own `kind`/`path` |
| `digest_format` | Digest fails the 64-hex pattern | Re-read the target and copy its complete current digest; never invent one |
| `replacement_shape` | `replacements` is not an array or an item lacks `old_text`/`new_text` | Send an array of exact replacement objects |
| `missing_required` | Another required argument is absent | Name only the missing field and its parent object |
| `unknown_or_other` | Any unclassified validation failure | Return the original concise validation summary and a link/reference to the authored shape |

Keep the original field path in the rendered error. For example:

```text
Edit rejected before any write: edits.0.expected_sha256 must be the current
complete-file SHA-256 (64 hexadecimal characters). Re-read the target with
coding_read_file and copy its digest. Do not invent or truncate it.
Validation detail: <bounded original message>.
```

The error must state that no edit was applied by this call. It must not claim
that earlier calls in the same turn had no effect.

## Authored content loading

Place recovery text in a versioned Markdown resource shipped with the edit
tool, for example `tools/edit_files/recovery.md`. Give each recovery key a
stable heading or keyed section and support only named substitutions such as
`{field_path}` and `{validation_detail}`.

The tool validation boundary loads the required authored resource using
`ctx.host.read_authored_text`. Use missing-key-safe `str.format_map`; do not
embed fallback prose in Python. A missing/unreadable/empty resource is a pack
installation error and must fail loudly with the resource path.

Use one generic **kernel validation-recovery hook**, not a second validation
pass inside the pack. `execute_tool` validates parameters before it calls the
pack's `execute`, so pack-level recovery alone would never see the malformed
arguments that this proposal addresses.

Add an optional catalog-authored `validation_recovery` declaration to
`ToolDefinition`, populated by the SDK catalog from `.tool.yml`:

```yaml
validation_recovery:
  content_path: tools/edit_files/recovery.md
```

The declaration names content only; it does not select a Python callable or
allow arbitrary imports. The SDK maps it to the tool's packaged content root
and validates that the resource exists when loading the catalog. The pure
classifier is a fixed, named convention on the already-loaded implementation
module, not a catalog-configurable import path.

Extend `ToolExecutionContext` with the existing host authored-content reader
surface, using the same `read_authored_text` binding as other pack content.
When `_validate_parameters` catches `ValidationError`:

1. Keep the original schema, normalized payload, and `ToolExecutionError`
   control flow.
2. Classify the structured errors through a small pack-owned pure classifier
   on the implementation module, `validation_recovery_key(exc) -> str`.
   The classifier reads only error `loc`, `type`, and kind context; it has no
   execution side effects.
3. Load the declared Markdown through `ctx.host.read_authored_text`, select
   the returned recovery-key section, and format named fields with
   missing-key-safe `str.format_map`.
4. Append the rendered recovery to the bounded original validation summary,
   then publish and raise through the existing `_publish_rejection_telemetry`
   and `ToolExecutionError` path.

A tool without the declaration or classifier keeps its current error. For a
tool that declares recovery, missing/unreadable/empty content or an unknown
recovery key is a pack installation error: fail loudly with the content path
and key, never substitute Python fallback prose. Do not catch execution
exceptions with this hook, do not turn the failure into a successful
`ToolResult`, and do not invoke it again during approval resumption unless
validation actually fails again.

Rejected calls retain the existing pre-execution rejection identity and
status vocabulary. No write effect receipt is emitted. The SDK owns catalog
translation, the kernel owns validation and dispatch, and the pack owns only
the pure classifier and authored recovery content.

## Retry behavior

The model may correct its arguments on a later iteration, but Lisa adds no
automatic replay loop for schema rejection. The next call must contain the
complete corrected arguments; recovery text is not server-side state.

After any successful write, the previous complete-file digest is stale. A
subsequent update must read the current snapshot. Normalization and recovery
must never silently rebase an edit onto changed content.

## Telemetry and privacy

Record these counters per tool call/turn:

- `validation_rejected` by recovery key;
- `normalized` by normalization kind;
- `validation_recovered` when a later call succeeds after a rejection in the
  same task;
- `execution_failed` separately from validation rejection.

Telemetry includes tool id, turn/call identity, bounded field paths, and
counts. It excludes `old_text`, `new_text`, file contents, digest values,
reasoning, and full raw arguments. Existing effect-receipt semantics remain:
only an actually dispatched write can satisfy a write contract.

## Test plan

### Deterministic pack tests

Add cases that fail without the change:

- Uppercase 64-hex digest normalizes to lowercase and succeeds against the
  same snapshot digest.
- Mixed-case digest behaves identically; normalization metadata names the
  field but not the digest.
- Short, non-hex, empty, null, placeholder, and whitespace-padded digests
  remain rejected with `digest_format` recovery.
- Native array, existing valid JSON-string array, and existing exact legacy
  flat shape retain their current behavior.
- Scalar/object `edits`, malformed JSON strings, wrong replacement shapes,
  missing record fields, and kind-specific extra fields receive the correct
  bounded recovery text.
- Recovery includes the exact field path and no-write statement, and does not
  leak edit contents or full raw arguments.
- Missing authored recovery content fails loudly with its resource path.
- A stale digest still fails after normalization; a changed file is never
  silently overwritten.
- Repeated valid edits require and use a newly read digest.

### Packstest

Follow the pack-testing contract before any installed-package verification:

1. `validate_pack(<coding-pack-path>)` verifies descriptors, packaged
   Markdown, schema, and declared `effect: write`.
2. `run_tool` with malformed arguments verifies pre-execution rejection,
   recovery text, and no filesystem mutation.
3. `run_tool` with uppercase digest verifies the real intended edit and
   resulting Git diff.
4. `run_scenario` covers rejection → corrected arguments → fresh read →
   successful edit, asserting outcomes from evidence and final file bytes.
5. Register eval markers in the pack project; tests use temporary directories
   and no model calls or private runtime imports.

### Live scenario

Add an opt-in coding-pack e2e scenario with a tiny source fixture and a
scripted/tool-plan oracle that exercises the malformed-array rejection,
uppercase-digest normalization, and corrected edit. Assert:

- the rejected call creates no diff;
- recovery text identifies the corrective action;
- the successful call produces exactly the expected diff;
- the worktree and Git evidence confirm the final state independently of the
  model's prose;
- telemetry distinguishes rejection, normalization, and successful execution.

Run the in-process eval first, then the live scenario on at least one model
selected for coding execution. A model-quality observation is not a reason to
loosen the deterministic schema assertions.

## Implementation slices

1. Add authored recovery resources, the structured error classifier, digest
   case normalization, and focused pack tests.
2. Implement the generic kernel recovery hook and SDK catalog declaration,
   preserving kernel telemetry and no-write semantics.
3. Shorten the tool description and add the packstest trajectory/eval.
4. Add and run the live scenario, inspect its evidence bundle, and document
   any provider-specific remaining failure without adding arbitrary repair
   fallbacks.

## Acceptance criteria

- Common malformed calls return actionable field-specific recovery instead
  of only raw Pydantic text.
- Uppercase/mixed-case valid hex digests work without changing digest safety;
  every invalid or unknown digest remains rejected.
- No schema recovery writes files, recomputes a requested digest, invents a
  match, or silently retries a write.
- All user-facing recovery text is packaged, versioned, loaded through the
  authored-content seam, and covered by missing-resource tests.
- Existing valid edit shapes and safety checks remain unchanged.
- Deterministic packstest coverage and the live scenario prove both recovery
  and exact final repository state.

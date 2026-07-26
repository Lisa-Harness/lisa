---
name: Pack Content Authoring
description: "Use when writing or reviewing pack operation/mode Python handlers. Enforces that prompts and user-facing text live in authored Markdown loaded via the SDK content registry — never embedded in .py and never silently fall back to missing content. Applies to all pack Python handlers (sdk/packs/** and ../lisa-packs/**/lisa_packs/** and ../lisa-packs-wip/**/lisa_packs/**)."
applyTo: "**/operations/**/*.py,**/modes/**/*.py,**/tools/**/*.py"
---

# Pack Content Authoring

Inherits all rules in `.github/instructions/python-development.md` and
`.github/instructions/code-quality.md`. The rules below are additive and
apply to every pack — both bundled `sdk/packs/*` and external `lisa_packs.*`
directories.

## Two absolute rules

1. **No embedding Markdown/text in `.py`.** System prompts, user-prompt
   templates, usage hints, and user-facing error messages are **authored
   content**. They live as `.md` files under the operation's `prompts/`
   directory (or the pack's `content/` directory) and are loaded at runtime
   through the SDK content-registry loader (`ctx.read_authored_text_fn`). A
   Python handler may contain path constants and template-substitution glue;
   it must not contain prompt prose, multi-line instruction strings, or
   user-facing message text. The only string literals allowed are:
   - Path constants (`"operations/<name>/prompts/<file>.md"`).
   - Field names and short structural tokens (`"role"`, `"system"`, log
     messages, exception type names).
   - Pydantic `description=` fields on parameter models.

2. **No fallbacks to missing content.** If `ctx.read_authored_text_fn` is
   not wired, or returns empty/`None`, or raises — the operation **fails
   loudly** with a clear `RuntimeError` naming the missing path. It must
   **not** silently substitute a hardcoded fallback string, an empty
   result, or a "good-enough" default. Silent fallbacks cause the pack to
   diverge from its authored content in production with no signal.

## Authoring layout

```plaintext
operations/<op_name>/
    <op_name>.yml              # metadata
    <op_name>.py               # handler (glue only)
    prompts/
        <op_name>_system.md    # system prompt
        <op_name>_user.md      # user-prompt template ({placeholder} tokens)
        <op_name>_usage.md     # usage hint
        <op_name>_<error>.md   # one file per distinct error message
```

Prompt files are Markdown assets shipped with the pack. Declare them in
`pyproject.toml` under `[tool.setuptools.package-data]` so they install
alongside the Python.

## Loading contract

```python
def _load_text(ctx: OperationExecutionContext, path: str) -> str:
    if not callable(ctx.read_authored_text_fn):
        raise RuntimeError(f"read_authored_text_fn not wired; cannot load {path}")
    text = str(ctx.read_authored_text_fn(path) or "").strip()
    if not text:
        raise RuntimeError(f"Authored text at {path} is empty or missing")
    return text
```

No `fallback=` parameter. No `try/except` that swallows the failure. The
host's turn hooks builder wires `read_authored_text_fn` from
`hooks.context_seams.content_registry.read_authored_text`; production hosts
always have it. Unit tests that need to exercise the handler inject a
fake `read_authored_text_fn` that returns the authored text (or the real
file via the pack's content root).

### Required inputs are required (no silent downgrade)

The "no fallback to missing content" rule extends to the *inputs* the
authored content needs: a completion function (`ctx.complete_fn`), a
resolvable model (`ctx.resolve_model_fn`), and non-empty turn history.
When any required input is missing, the operation **raises `RuntimeError`**
naming the missing dependency. It must **not**:

- silently persist a bare/raw/unenriched record instead of the modeled one,
- catch a distillation/enrichment failure with a broad `except` and store
  the raw input,
- substitute a hardcoded `_fallback_summary` / `_fallback_*` helper for the
  authored model output.

The only allowed "deterministic rendering" path is honest status reporting
when the *upstream content* genuinely produced nothing (e.g. all workshop
panels failed/timed out) — that renders the failure, not a substitute for
missing content.

### Mode-relative prompt files

Operations that load prompts relative to the active mode's content root
(via `mode.mode_markdown_path`) must fail loudly when:

- the mode has no `mode_markdown_path` (no content root to load from),
- the authored prompt file (`planner.md`, `synthesis.md`, or a configured
  `*_prompt_file`) is missing on disk.

Raise `RuntimeError` naming the mode and the missing file. Do **not** keep
embedded `_PLANNER_PROMPT_FALLBACK` / `_SYNTHESIS_PROMPT_FALLBACK` string
constants — author the prompt as `.md` and ship it with the mode.

### No deterministic default panels/config

A workshop-style operation must not embed `_deterministic_default_panels()`
(or any equivalent) to substitute for a mode that forgot to author
`workshop_panels`. Panels are required authored content: the mode's
`_mode.md` must declare `workshop_panels`, or the caller must pass them via
`params["workshop_panels"]`. If neither is present, raise `RuntimeError` —
do not synthesize two default panels in Python.

## Template substitution

Authored templates may use `{placeholder}` tokens. Substitution is the
handler's job and uses `str.format_map` with a missing-key-safe dict so
unfilled tokens survive visibly rather than raising `KeyError`. The handler
never re-embeds the prose — it only supplies the values.

## Why these rules exist

- **Editability**: non-Python authors (PMs, domain experts) can edit prompts
  without touching code or breaking imports.
- **Reviewability**: prompt diffs land in code review as Markdown, not as
  Python string-literal noise.
- **No silent drift**: a missing prompt file must break loudly at runtime,
  not produce a subtly-wrong fallback that ships to production unnoticed.
- **Pack portability**: the same authored content loads identically across
  every host that wires the SDK content registry.

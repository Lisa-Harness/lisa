---
name: Pack Content Authoring
applyTo: "**/operations/**/*.py,**/modes/**/*.py,**/tools/**/*.py"
description: "Use when authoring pack handlers, prompts, modes, or tools; keeps user-facing content in versioned Markdown and prevents silent fallbacks."
---

# Pack content authoring

- Keep prompts, templates, usage hints, and user-facing errors in authored Markdown shipped as package data. Python handlers contain path constants, structured tokens, and substitution glue only.
- Load authored content through `ctx.host.read_authored_text` and fail loudly with the missing path when the loader is unavailable, raises, or returns empty content. Do not embed fallback prose.
- Required operation inputs, including completion, model resolution, and turn history, also fail loudly. Do not silently persist raw or degraded substitutes after enrichment fails.
- Mode-relative prompts and required mode configuration must exist. Do not synthesize default panels, prompts, summaries, or other authored content in Python.
- Use missing-key-safe `str.format_map` substitution so unfilled placeholders remain visible instead of raising `KeyError`.
- A deterministic status rendering is allowed only when upstream work genuinely produced no content; it reports the failure rather than replacing missing authored content.
- For structured-output handlers, follow the recovery, token-diagnostic, and scalar-list normalization rules in the Packs Architecture instruction.
- Every operation and tool definition MUST declare an explicit `effect:` field (`read`, `write`, `execute`, or `external_write`). The default is `write` (the conservative assumption), so a read-only capability that omits `effect` is wrongly treated as a completion requirement and blocks legitimate recall/research-only turns as "hallucinated success". Declare `effect: read` on search, recall, distillation, briefing, and self-model capabilities; `effect: write` on anything that persists to disk or memory; `effect: external_write` on capabilities that mutate state outside the workspace (email drafts, provider APIs); `effect: execute` on capabilities that run commands.
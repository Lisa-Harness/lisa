---
name: Markdown Quality
description: "Use when writing, editing, or reviewing Markdown docs in the lisa_framework workspace. Enforces lint-clean structure, explicit formatting, and scope-narrowed suppressions. Applies to *.md."
applyTo: "**/*.md"
---

# Markdown

Follow the repo's `markdown-preferences.md` user memory: write lint-clean Markdown by default.

- Surround headings, lists, tables, and fenced code blocks with blank lines unless the file already uses a required exception.
- Use fenced code blocks with an explicit language. If the block is not source code, use `plaintext`.
- Prefer native Markdown over inline HTML. Only use inline HTML or markdownlint suppression comments when Markdown cannot express the required structure.
- Scope suppressions narrowly. If a markdownlint disable comment is unavoidable, limit it to the smallest possible block and preserve or restore the prior rule state.
- Tables must use one style per table. Do not mix aligned, compact, or tight pipe spacing inside the same table. For new tables, prefer compact pipe tables.
- Do not auto-fix repository-wide Markdown in normal workflows. Use non-fixing lint checks by default so unrelated files are not rewritten. Use auto-fix only when explicitly requested and scope it to files intended for cleanup.

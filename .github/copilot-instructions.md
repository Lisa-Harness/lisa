---
name: Project Contract
---

# Lisa Copilot Project Contract

Use the repository customization taxonomy in this order:

1. Always-on project contract: this file and the package-scoped guidance below.
2. Engineering standards: workflow, Python, content-authoring, and file-format rules.
3. Architecture boundaries: package ownership and dependency rules for kernel, SDK, packs, MCP servers, and hosts.
4. Customization governance: how to author instructions, agents, prompts, and skills without bloating the prompt.

## Operating defaults

- Prefer the smallest rule file that can explain the intent. Keep instructions concise and task-driven.
- Resolve scope explicitly. A change should bind to the narrowest `applyTo` that still covers the relevant files.
- Treat the repository as a hard-cutover codebase. Do not add compatibility shims or stale aliases for old patterns.
- Follow the shared engineering loop: verify, review the diff, then commit. Land changes on `main` by rebase or squash — never a merge commit. If a change is already shared or pushed, prefer a `fixup!` commit over rewriting history.
- Keep troubleshooting notes, incident logs, and one-off test workarounds out of instructions. Put them in docs or tests instead.

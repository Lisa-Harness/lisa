---
name: Code Quality
description: "Use for the required dev-loop verification order in the lisa_framework workspace. The local dev loop and CI use the same commands in the same order; any failure blocks. Type-specific quality rules live in markdown-quality.md, yaml-quality.md, json-quality.md, shell-bash.md, and shell-powershell.md."
applyTo: "**/*"
---

# Code Quality

## Dev loop — required order

The local dev loop and CI use the same commands in the same order. There is no warn-only stage; any failure blocks.

```powershell
# 1. Format check (deterministic; idempotent)
uv run ruff format --check kernel sdk mcp_server hosts

# 2. Lint (complexity + style + import ordering)
uv run ruff check kernel sdk mcp_server hosts

# 3. Type check (strict, all public packages)
uv run mypy kernel sdk mcp_server

# 4. Dead code check
uv run vulture kernel sdk mcp_server hosts/little_lisa vulture_allowlist.py

# 5. Tests (no live API calls)
uv run pytest -m "not live" --tb=short -v

# 6. Smoke test
uv run little-lisa verify
```

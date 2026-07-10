# Contributing to Lisa Framework

Thank you for your interest in contributing.

## Getting Started

1. Fork the repo and clone it locally.
2. Install dependencies: `uv sync --extra dev`
3. Run the verification: `uv run little-lisa verify`
4. Make your changes on a feature branch.

## Development Loop

The local loop and CI use the same commands in the same order. There is no
warn-only stage — any failure blocks. Run all steps from the repo root.

```powershell
# 1. Format check (deterministic; idempotent)
uv run ruff format --check kernel sdk sdk_packs hosts

# 2. Lint (complexity + style + import ordering)
uv run ruff check kernel sdk sdk_packs hosts

# 3. Type check (strict, all public packages)
uv run mypy kernel sdk sdk_packs

# 4. Dead code check
uv run vulture kernel sdk sdk_packs hosts/little_lisa vulture_allowlist.py

# 5. Tests (no live API calls)
uv run pytest -m "not live" --tb=short -v

# 6. Smoke test
uv run little-lisa verify
```

## Pull Request Checklist

- [ ] `ruff format --check` passes for touched files.
- [ ] `ruff check` passes with zero new violations for touched files.
- [ ] `mypy` passes with zero new errors for touched packages.
- [ ] `vulture` passes with zero new reports for touched files.
- [ ] All non-live tests pass (`pytest -m "not live"`).
- [ ] `little-lisa verify` exits 0.
- [ ] New public APIs have docstrings (module, class, function/method).
- [ ] New or materially changed public callables have co-maintained tests.
- [ ] No new `# noqa` without a specific rule code and one-line justification.
- [ ] No `dict[str, Any]` state bags introduced — use Pydantic model or dataclass.
- [ ] No broad `except Exception:` outside top-level process boundaries.
- [ ] No nested closures capturing mutable parent state — flatten to module level.
- [ ] Breaking changes are noted in the PR description and `CHANGELOG.md`.

## Package Layout

| Package | Path | Description |
| --- | --- | --- |
| `lisa-kernel` | `kernel/` | Runtime-neutral contracts, FSM engine, operation registry |
| `lisa-sdk` | `sdk/` | Author-facing helpers for packs and integration contracts |
| `lisa-pack-*` | `sdk_packs/` | Public mode/operation packs |
| `little-lisa` | `hosts/little_lisa/` | Local terminal host CLI |

## Commit Conventions

- `feat:` — new behavior
- `fix:` — bug fix
- `imp:` — non-breaking improvement
- `docs:` — documentation only
- `infra:` — CI/CD, build, or tooling
- `copilot:` — agent-related changes

## Questions?

Open a discussion or issue on GitHub.


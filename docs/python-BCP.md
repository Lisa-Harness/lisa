# Python development in Lisa

This is the project's Python reference, not a general survey of Python tools.
Use the configuration in [pyproject.toml](../pyproject.toml) and the
[testing guide](TESTING.md) as the executable source of truth.

## Environment and style

- Python 3.11 is the minimum. Do not use syntax or standard-library APIs
  unavailable there without an explicit support-range change.
- Use the uv workspace and lockfile. Do not install dependencies globally
  or replace the project's dependency manager.
- Ruff formats and lints the source: line length 100, target `py311`,
  LF newlines, configured import/style/complexity rules.
- Run strict mypy on the configured public-source selection. Add typed
  dataclasses, Pydantic models, or protocols for shared state rather than
  untyped dictionary bags.
- Follow existing naming, boundaries, and helpers. Do not add generic
  abstractions or compatibility aliases without a concrete need.

## Runtime correctness

- Keep imports/discovery side-effect-free and accept resolved configuration
  explicitly at integration seams.
- Keep domain behavior out of transport hosts and UI dependencies out of
  the runtime. Packs depend on kernel/SDK, not runtime or other packs.
- Use the storage seam for model-requested file I/O. Review explicit backend
  carve-outs under the [direct-I/O policy](STORAGE.md#storage-sandbox-and-direct-io).
- Handle errors at the correct boundary, with observable diagnostics. Catch
  specific exceptions; never turn an unexpected failure into silent success.
- Preserve async resource/task ownership. See [host integration](HOST_INTEGRATION.md).
- Keep user-facing authored instructions in versioned pack content, not
  long Python string literals.

## Tests and review

Start with a narrow reproduction, add a regression test, and verify the
relevant behavior before broader gates. Do not rely on private data directories,
live credentials, or sibling packs in unit tests.
See [CONTRIBUTING.md](../CONTRIBUTING.md), [logging](LOGGING.md), and the
[coding workflow](CODING_WORKFLOW.md) before making core changes.

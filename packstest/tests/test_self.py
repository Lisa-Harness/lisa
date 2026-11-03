"""Harness self-tests: packstest's own contracts stay regression-locked.

The harness builds on the SDK's production catalog scanner and the kernel's
testing stubs; these tests verify the harness wiring itself with synthetic
packs so no real pack is required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest
from packstest.catalog import (
    _probe_entry_callable,
    activate_tool,
    build_tool_catalog,
    check_tool_yaml_schema,
    load_pack_content_root,
    validate_pack,
)
from packstest.context import ToolContextFactory, make_context
from packstest.git import git_repo
from packstest.store import MemoryWorkProductStore


def test_entry_point_runtime_factory_is_not_called_with_fake_inputs() -> None:
    def factory(pack_config: dict, *, context: object) -> object:
        raise AssertionError("runtime inputs are required")

    assert _probe_entry_callable(factory) is factory


def test_entry_point_zero_argument_factory_is_invoked() -> None:
    assert _probe_entry_callable(lambda: "content-root") == "content-root"


def test_entry_point_internal_type_error_is_not_retried() -> None:
    calls = []

    def factory(config: object = None) -> object:
        calls.append(config)
        raise TypeError("broken factory")

    with pytest.raises(TypeError, match="broken factory"):
        _probe_entry_callable(factory)
    assert calls == [None]


def _write_tool(root: Path, tool_id: str, provider_id: str) -> None:
    tools_dir = root / "tools" / tool_id
    tools_dir.mkdir(parents=True, exist_ok=True)
    (tools_dir / f"{tool_id}.py").write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "from kernel.models import ToolExecutionContext, ToolResult",
                "",
                "",
                "async def execute(ctx: ToolExecutionContext) -> ToolResult:",
                "    name = str(ctx.parameters.get('name', 'world'))",
                "    return ToolResult(tool_id=ctx.tool_id, output_text=f'hello {name}')",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (tools_dir / f"{tool_id}.tool.yml").write_text(
        "\n".join(
            [
                f"tool_id: {tool_id}",
                f"tool_provider_id: {provider_id}",
                "description: synthetic harness tool",
                "effect: read",
                "timeout_seconds: 30",
                "",
            ]
        ),
        encoding="utf-8",
    )


@pytest.fixture()
def synthetic_pack(tmp_path: Path) -> Path:
    """A minimal pack checkout: pyproject + lisa_packs/<slug>/tools."""

    pack = tmp_path / "packs-wip" / "synthetic"
    pkg = pack / "lisa_packs" / "synthetic"
    pkg.mkdir(parents=True)
    (pack / "pyproject.toml").write_text(
        '[project]\nname = "lisa-pack-synthetic"\n',
        encoding="utf-8",
    )
    _write_tool(pkg, "synthetic_hello", "lisa-pack-synthetic")
    return pack


def test_load_pack_content_root_resolves_package_dir(synthetic_pack: Path) -> None:
    """The content root is the lisa_packs/<slug> dir, not the checkout."""

    root = load_pack_content_root(synthetic_pack)
    assert root.name == "synthetic"
    assert (root / "tools").is_dir()


def test_build_tool_catalog_uses_sdk_scanner(synthetic_pack: Path) -> None:
    """The real SDK catalog scan resolves tool id, module path, provider id."""

    catalog = build_tool_catalog(synthetic_pack)
    assert set(catalog) == {"synthetic_hello"}
    tool = catalog["synthetic_hello"]
    assert tool["tool_provider_id"] == "lisa-pack-synthetic"
    assert tool["effect"] == "read"
    assert tool["module_path"].startswith("lisa_packs.synthetic.tools.")


@pytest.mark.parametrize("path_kind", ["list", "tuple"])
def test_activate_additional_pack_after_namespace_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path_kind: str
) -> None:
    """A second source checkout is visible to an already-imported pack namespace."""

    import sys

    import lisa_packs

    monkeypatch.setattr(sys, "path", list(sys.path))
    namespace_path = list(lisa_packs.__path__)
    monkeypatch.setattr(
        lisa_packs, "__path__", tuple(namespace_path) if path_kind == "tuple" else namespace_path
    )
    pack = tmp_path / "late-loaded"
    package = pack / "lisa_packs" / "late_loaded"
    package.mkdir(parents=True)
    (pack / "pyproject.toml").write_text(
        '[project]\nname = "lisa-pack-late_loaded"\n', encoding="utf-8"
    )
    _write_tool(package, "late_loaded_hello", "lisa-pack-late_loaded")

    catalog = build_tool_catalog(pack)

    assert callable(activate_tool(catalog["late_loaded_hello"]))


def test_validate_pack_passes_synthetic_pack(synthetic_pack: Path) -> None:
    """Validation passes when entry points, imports, and ids are healthy."""

    report = validate_pack(synthetic_pack)
    assert report.ok, report.summary()
    assert "synthetic_hello" in report.tools


def test_validate_pack_accepts_self_bounding_tool_timeout_zero(tmp_path: Path) -> None:
    """``timeout_seconds: 0`` (wrapper disabled, tool self-bounds) is valid.

    The harness previously rejected 0 with "must be > 0", contradicting the
    kernel contract (kernel/tools/registry.py treats 0 as wrapper-disabled;
    sdk/AUTHORING.md documents it). That mis-model forced packs to declare
    positive lifetime bounds no number could satisfy for self-bounding tools.
    """
    pack = tmp_path / "lisa-pack-selfbound"
    package = pack / "lisa_packs" / "selfbound"
    package.mkdir(parents=True)
    (pack / "pyproject.toml").write_text(
        '[project]\nname = "lisa-pack-selfbound"\n', encoding="utf-8"
    )
    _write_tool(package, "selfbound_hello", "lisa-pack-selfbound")
    (package / "tools" / "selfbound_hello" / "selfbound_hello.tool.yml").write_text(
        "\n".join(
            [
                "tool_id: selfbound_hello",
                "tool_provider_id: lisa-pack-selfbound",
                "description: synthetic self-bounding tool",
                "effect: read",
                "timeout_seconds: 0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    report = validate_pack(pack)
    assert report.ok, report.summary()
    assert report.tools["selfbound_hello"]["timeout_seconds"] == 0


def test_validate_pack_rejects_negative_tool_timeout(tmp_path: Path) -> None:
    """Negative timeout_seconds stays a validation error (kernel rejects < 0)."""

    issues = check_tool_yaml_schema(
        "bad_tool",
        {"tool_id": "bad_tool", "description": "x", "timeout_seconds": -5},
    )
    assert any("negative timeout_seconds" in issue.detail for issue in issues)


def test_run_tool_executes_in_process(synthetic_pack: Path, tmp_path: Path) -> None:
    """run_tool dispatches through the real ToolExecutionContext contract."""

    import asyncio  # stdlib, no cycle risk

    factory = ToolContextFactory(
        root=tmp_path,
        pack_config={"workspace_root": str(tmp_path)},
    )
    outcome = asyncio.run(_invoke(synthetic_pack, "synthetic_hello", factory, {"name": "harness"}))
    assert outcome.result.output_text == "hello harness"
    assert outcome.result.error == ""


async def _invoke(
    pack_path: Path, tool_id: str, factory: ToolContextFactory, parameters: dict
) -> object:
    from packstest.runner import run_tool as _run_tool  # lazy: avoid import cycle

    return await _run_tool(pack_path, tool_id, factory, parameters)


def test_factory_wires_mcp_tool_call_seam(tmp_path: Path) -> None:
    """Tools that delegate to an MCP capability get the test double."""

    async def _capability(_tool_id: str, _arguments: object) -> object:
        raise AssertionError("not called")

    factory = ToolContextFactory(root=tmp_path, pack_config={}, mcp_tool_call=_capability)
    assert factory.context().mcp_tool_call is _capability
    assert ToolContextFactory(root=tmp_path, pack_config={}).context().mcp_tool_call is None


def test_make_context_stamps_identity_and_store(tmp_path: Path) -> None:
    """One-shot contexts carry ownership identity and an artifact store."""

    ctx = make_context(tmp_path, {"path": "x"})
    assert ctx.workspace_id == "default"
    assert ctx.session_plan_digest == "packstest-digest"
    assert ctx.code_change_artifact_store is not None
    assert ctx.pack_config == {"workspace_root": str(tmp_path)}


@pytest.mark.asyncio
async def test_memory_store_roundtrip(tmp_path: Path) -> None:
    """The store double satisfies save/get/delete round-trips."""

    store = MemoryWorkProductStore()
    artifact = store.make_artifact(source_root=str(tmp_path))
    await store.create(artifact)
    fetched = await store.get(artifact.artifact_id)
    assert fetched is not None
    assert fetched.artifact_id == artifact.artifact_id
    fetched.revision = 3
    await store.save(fetched)
    saved = await store.get(artifact.artifact_id)
    assert saved is not None and saved.revision == 3
    assert await store.delete(artifact.artifact_id) is True
    assert await store.get(artifact.artifact_id) is None


@pytest.mark.asyncio
async def test_git_repo_fixture_offline(tmp_path: Path) -> None:
    """The git fixture creates a committed repo plus bare remote."""

    fixture = await git_repo(tmp_path)
    assert (fixture.root / ".git").is_dir()
    assert fixture.remote is not None and fixture.remote.is_dir()
    sha = await fixture.head()
    assert len(sha) == 40
    fixture.write("next.txt", "more\n")
    assert (fixture.root / "next.txt").read_text(encoding="utf-8") == "more\n"


# ── content-root locator: src/ layouts resolve, not warn ──────────────


def _write_src_layout_pack(root: Path, *, package: str, slug: str) -> Path:
    """A pack checkout whose package lives under ``src/`` (bundled-style).

    The bundled packages (``sdk.*``) share one namespace across the
    workspace; writing a stub ``sdk`` package whose ``src/`` dir later lands
    on ``sys.path`` would shadow the real ``sdk`` for the whole process. The
    helper therefore rewrites the top-level segment to a never-installed
    probe name (``e2e_probe_sdk``) — same layout shape, no shadowing.
    """
    parts = package.split(".")
    if parts and (parts[0] in {"sdk", "kernel", "runtime", "mcp_server"}):
        parts[0] = f"e2e_probe_{parts[0]}"
        package = ".".join(parts)
    pkg = root / "src" / package.replace(".", "/")
    pkg.mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                f'name = "lisa-pack-{slug}"',
                "",
                '[project.entry-points."lisa.extensions.content_roots"]',
                f'{slug} = "{package}:get_content_root"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    (pkg / "__init__.py").write_text(
        "from pathlib import Path\n"
        "def get_content_root() -> Path:\n"
        "    return Path(__file__).parent\n",
        encoding="utf-8",
    )
    tools = pkg / "tools" / f"{slug}_hello"
    tools.mkdir(parents=True)
    (tools / f"{slug}_hello.py").write_text(
        "from kernel.models import ToolExecutionContext, ToolResult\n"
        "async def execute(ctx: ToolExecutionContext) -> ToolResult:\n"
        "    return ToolResult(tool_id=ctx.tool_id)\n",
        encoding="utf-8",
    )
    (tools / f"{slug}_hello.tool.yml").write_text(
        f"tool_id: {slug}_hello\ntool_provider_id: lisa-pack-{slug}\n"
        "description: src-layout probe\neffect: read\n",
        encoding="utf-8",
    )
    return root


def test_content_root_resolves_bundled_src_layout(tmp_path: Path) -> None:
    """``src/sdk/packs/<slug>`` resolves from the declared entry point.

    The helper renames the top-level segment (``e2e_probe_sdk``) so the test
    tree cannot shadow the real ``sdk`` namespace on ``sys.path``; the layout
    shape the locator must handle is unchanged.
    """

    pack = _write_src_layout_pack(tmp_path / "core-like", package="sdk.packs.core", slug="core")
    from packstest.catalog import load_pack_content_root

    root = load_pack_content_root(pack)
    assert root == pack / "src" / "e2e_probe_sdk" / "packs" / "core"
    assert (root / "tools" / "core_hello").is_dir()


def test_content_root_resolves_nested_lisa_packs_src_layout(tmp_path: Path) -> None:
    """``src/lisa_packs/<slug>`` (external src-layout packs) resolves."""

    pkg = tmp_path / "ext" / "src" / "lisa_packs" / "widgets"
    pkg.mkdir(parents=True)
    (tmp_path / "ext" / "src" / "__pycache__").mkdir()
    (tmp_path / "ext" / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "lisa-pack-widgets"',
                "",
                '[project.entry-points."lisa.extensions.content_roots"]',
                'widgets = "lisa_packs.widgets:get_content_root"',
            ]
        ),
        encoding="utf-8",
    )
    (pkg / "__init__.py").write_text(
        "from pathlib import Path\n"
        "def get_content_root() -> Path:\n"
        "    return Path(__file__).parent\n",
        encoding="utf-8",
    )
    (pkg / "tools").mkdir()
    from packstest.catalog import _ensure_source_importable, load_pack_content_root

    _ensure_source_importable(tmp_path / "ext")
    root = load_pack_content_root(tmp_path / "ext")
    assert root == pkg


def test_stale_lisa_packs_modules_classifies_outside_and_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Eviction candidates: outside-checkout modules are stale; inside ones stay."""

    from packstest.catalog import _stale_lisa_packs_modules

    inside = ModuleType("lisa_packs.inside")
    inside.__file__ = str(tmp_path / "checkout" / "lisa_packs" / "inside" / "__init__.py")
    outside_tool = ModuleType("lisa_packs.tool")
    outside_tool.__file__ = str(tmp_path / "elsewhere" / "tool.py")
    monkeypatch.setitem(sys.modules, "lisa_packs", None)
    monkeypatch.setitem(sys.modules, "lisa_packs.inside", inside)
    monkeypatch.setitem(sys.modules, "lisa_packs.tool", outside_tool)

    stale = _stale_lisa_packs_modules((tmp_path / "checkout").resolve())
    assert "lisa_packs.tool" in stale
    assert "lisa_packs.inside" not in stale


def test_stale_lisa_packs_modules_returns_all_on_unusable_namespace_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A namespace module whose __path__ was already invalidated is unusable:
    the walk gives up and returns every candidate for unconditional eviction."""

    from packstest.catalog import _stale_lisa_packs_modules

    class BrokenPath:
        def __iter__(self):
            raise KeyError("parent evicted")

        def __len__(self):
            raise KeyError("parent evicted")

    broken = ModuleType("lisa_packs.broken")
    broken.__path__ = BrokenPath()  # type: ignore[assignment]
    monkeypatch.setitem(sys.modules, "lisa_packs.broken", broken)

    assert _stale_lisa_packs_modules((tmp_path / "checkout").resolve()) == ["lisa_packs.broken"]


def test_stale_lisa_packs_modules_skips_module_without_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A module with neither __file__ nor __path__ has nothing to classify."""

    from packstest.catalog import _stale_lisa_packs_modules

    bare = ModuleType("lisa_packs.bare")
    monkeypatch.setitem(sys.modules, "lisa_packs.bare", bare)

    assert _stale_lisa_packs_modules((tmp_path / "checkout").resolve()) == []


def test_validate_pack_src_layout_scans_tools(tmp_path: Path) -> None:
    """A src-layout pack validates with a real tool scan (no empty warning)."""

    pack = _write_src_layout_pack(
        tmp_path / "widgets-like", package="lisa_packs.widgets", slug="widgets"
    )
    report = validate_pack(pack)
    assert report.ok, report.summary()
    assert "widgets_hello" in report.tools
    assert not report.warnings


def test_pack_slug_handles_missing_or_unprefixed_names(tmp_path: Path) -> None:
    """The slug helper degrades to '' without pyproject/unprefixed names."""

    from packstest.catalog import _pack_slug

    empty = tmp_path / "empty"
    empty.mkdir()
    assert _pack_slug(empty) == ""
    (empty / "pyproject.toml").write_text('[project]\nname = "bare-toolkit"\n', encoding="utf-8")
    assert _pack_slug(empty) == "bare-toolkit"
    (empty / "pyproject.toml").write_text("not = valid [[", encoding="utf-8")
    assert _pack_slug(empty) == ""


def test_content_root_src_fallback_without_entry_point(tmp_path: Path) -> None:
    """A src/ checkout with no entry-point group falls back to the bounded walk."""

    from packstest.catalog import load_pack_content_root

    src_pkg = tmp_path / "src" / "widgets"
    (src_pkg / "tools").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "lisa-pack-widgets"\n')
    assert load_pack_content_root(tmp_path) == src_pkg


def test_content_root_falls_back_to_checkout_for_flat_layout(tmp_path: Path) -> None:
    """Flat content-root checkouts and non-pack dirs resolve to themselves."""

    from packstest.catalog import load_pack_content_root

    flat = tmp_path / "flat"
    (flat / "tools").mkdir(parents=True)
    assert load_pack_content_root(flat) == flat

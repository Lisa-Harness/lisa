"""Harness self-tests: packstest's own contracts stay regression-locked.

The harness builds on the SDK's production catalog scanner and the kernel's
testing stubs; these tests verify the harness wiring itself with synthetic
packs so no real pack is required.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from packstest.catalog import (
    build_tool_catalog,
    load_pack_content_root,
    validate_pack,
)
from packstest.context import ToolContextFactory, make_context
from packstest.git import git_repo
from packstest.store import MemoryWorkProductStore


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


def test_validate_pack_passes_synthetic_pack(synthetic_pack: Path) -> None:
    """Validation passes when entry points, imports, and ids are healthy."""

    report = validate_pack(synthetic_pack)
    assert report.ok, report.summary()
    assert "synthetic_hello" in report.tools


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

"""Unit tests for the host-layer ``from:`` and ``body:`` directive resolver."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from textwrap import dedent

import pytest
from kernel.session_plan import SessionPlanError, compile_session_plan
from runtime.content_roots import resolve_pack_session_path
from runtime.session_from_resolver import resolve_session_from_directives

# ── Fixtures ─────────────────────────────────────────────────────────


def _write_pack_session(pack_root: Path, session: str, body: str) -> Path:
    """Write a minimal v2 session file under ``<pack_root>/sessions/<session>.md``."""

    sessions_dir = pack_root / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    path = sessions_dir / (session if session.endswith(".md") else f"{session}.md")
    path.write_text(body, encoding="utf-8")
    return path


def _core_pack_session_text(*, with_routing: bool = True, with_context: bool = True) -> str:
    """A representative pack session manifest used across the tests."""

    routing = (
        (
            "routing:\n"
            "  start_mode: default\n"
            "  sequential_connectors:\n"
            "    - then\n"
            "    - and then\n"
            "    - after that\n"
            "    - next\n"
            "    - followed by\n"
            "    - based on\n"
            "    - once\n"
        )
        if with_routing
        else ""
    )
    context_block = (
        (
            "context:\n"
            "  sources:\n"
            "    - rule_id: default\n"
            "      lane_id: operational\n"
            "      fallback: true\n"
        )
        if with_context
        else ""
    )
    template = dedent(
        """\
        ---
        schema_version: 2
        name: core-default
        surface:
          type: personal_dm
          id: host-console
          memory_scope: personal
        resources:
          packs:
            required: [lisa-pack-core]
        providers:
          modes:
            default: lisa-pack-core
        {optional_blocks}---
        # Core session body

        Use the available tools and supplied context carefully.
        """
    )
    return template.format(optional_blocks=routing + context_block)


@pytest.fixture
def core_pack_root(tmp_path: Path) -> Path:
    """A fake ``lisa-pack-core`` install dir with a pyproject + default session."""

    root = tmp_path / "lisa_pack_core"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nname = "lisa-pack-core"\nversion = "0.0.0"\n', encoding="utf-8"
    )
    _write_pack_session(root, "default", _core_pack_session_text())
    return root


@pytest.fixture
def bare_pack_root(tmp_path: Path) -> Path:
    """A pack that ships NO ``routing`` block, to exercise the missing-block error."""

    root = tmp_path / "lisa_pack_minimal"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nname = "lisa-pack-minimal"\nversion = "0.0.0"\n', encoding="utf-8"
    )
    _write_pack_session(
        root,
        "default",
        dedent(
            """\
            ---
            schema_version: 2
            name: minimal
            surface:
              type: personal_dm
              id: host-console
              memory_scope: personal
            ---
            # Minimal body
            """
        ),
    )
    return root


def _lookup_for(*roots: Path) -> Callable[..., Path]:
    """Build the injected lookup over a fixed list of discovered pack roots."""

    def _lookup(*, pack: str, session: str) -> Path:
        return resolve_pack_session_path(pack=pack, session=session, pack_roots=list(roots))

    return _lookup


# ── Passthrough (no directives) ──────────────────────────────────────


def test_passthrough_when_no_directives(core_pack_root: Path) -> None:
    """A session with no ``from:``/``body:`` is returned byte-for-byte equivalent."""

    user_text = dedent(
        """\
        ---
        schema_version: 2
        name: mine
        surface: { type: personal_dm, id: host-console, memory_scope: personal }
        ---
        # My session
        """
    )
    resolved = resolve_session_from_directives(
        text=user_text,
        source_uri="workspace:mine.md",
        pack_session_lookup=_lookup_for(core_pack_root),
    )
    # Re-emit normalizes whitespace; assert semantic equality via the compiler.
    plan = compile_session_plan(text=resolved, source_uri="workspace:mine.md")
    assert plan.manifest.name == "mine"
    # No routing directive pulled — kernel default applies.
    assert plan.manifest.routing.start_mode == "default"
    assert plan.manifest.routing.sequential_connectors == ()


# ── Block ``from:`` ──────────────────────────────────────────────────


def test_from_pulls_routing_block(core_pack_root: Path) -> None:
    """``routing: from:`` substitutes the pack's routing block wholesale."""

    user_text = dedent(
        """\
        ---
        schema_version: 2
        name: mine
        surface: { type: personal_dm, id: host-console, memory_scope: personal }
        routing:
          from:
            pack: lisa-pack-core
            session: default
        ---
        # My session
        """
    )
    resolved = resolve_session_from_directives(
        text=user_text,
        source_uri="workspace:mine.md",
        pack_session_lookup=_lookup_for(core_pack_root),
    )
    plan = compile_session_plan(text=resolved, source_uri="workspace:mine.md")
    assert plan.manifest.routing.start_mode == "default"
    assert plan.manifest.routing.sequential_connectors == (
        "then",
        "and then",
        "after that",
        "next",
        "followed by",
        "based on",
        "once",
    )


def test_from_pulls_execution_block(tmp_path: Path) -> None:
    """``execution: from:`` substitutes the pack's execution block wholesale."""

    pack_root = tmp_path / "lisa_pack_core"
    pack_root.mkdir()
    (pack_root / "pyproject.toml").write_text(
        '[project]\nname = "lisa-pack-core"\nversion = "0.0.0"\n', encoding="utf-8"
    )
    _write_pack_session(
        pack_root,
        "default",
        dedent(
            """\
            ---
            schema_version: 2
            name: core-default
            surface:
              type: personal_dm
              id: host-console
              memory_scope: personal
            execution:
              max_session_tokens: 15000000
              max_completion_tokens: 500000
            ---
            # Core session body
            """
        ),
    )

    user_text = dedent(
        """\
        ---
        schema_version: 2
        name: mine
        surface: { type: personal_dm, id: host-console, memory_scope: personal }
        execution:
          from:
            pack: lisa-pack-core
            session: default
        ---
        # My session
        """
    )
    resolved = resolve_session_from_directives(
        text=user_text,
        source_uri="workspace:mine.md",
        pack_session_lookup=_lookup_for(pack_root),
    )
    plan = compile_session_plan(text=resolved, source_uri="workspace:mine.md")
    assert plan.manifest.execution.max_session_tokens == 15_000_000
    assert plan.manifest.execution.max_completion_tokens == 500_000


def test_from_accepts_bare_pack_slug(core_pack_root: Path) -> None:
    """Both ``lisa-pack-core`` and the bare slug ``core`` resolve to the same pack."""

    user_text = dedent(
        """\
        ---
        schema_version: 2
        name: mine
        surface: { type: personal_dm, id: host-console, memory_scope: personal }
        routing:
          from:
            pack: core
            session: default
        ---
        # My session
        """
    )
    resolved = resolve_session_from_directives(
        text=user_text,
        source_uri="workspace:mine.md",
        pack_session_lookup=_lookup_for(core_pack_root),
    )
    plan = compile_session_plan(text=resolved, source_uri="workspace:mine.md")
    assert plan.manifest.routing.sequential_connectors[0] == "then"


def test_from_cherry_picks_three_leaves_with_local_context(
    core_pack_root: Path,
) -> None:
    """resources + providers + routing pulled, context authored locally."""

    user_text = dedent(
        """\
        ---
        schema_version: 2
        name: mine
        surface: { type: personal_dm, id: host-console, memory_scope: personal }
        resources:
          from: { pack: lisa-pack-core, session: default }
        providers:
          from: { pack: lisa-pack-core, session: default }
        routing:
          from: { pack: lisa-pack-core, session: default }
        context:
          sources:
            - rule_id: mine
              lane_id: operational
              fallback: true
        ---
        # My session
        """
    )
    resolved = resolve_session_from_directives(
        text=user_text,
        source_uri="workspace:mine.md",
        pack_session_lookup=_lookup_for(core_pack_root),
    )
    plan = compile_session_plan(text=resolved, source_uri="workspace:mine.md")
    # Pulled from pack.
    assert plan.manifest.resources.packs.required == ("lisa-pack-core",)
    assert plan.manifest.providers.modes["default"] == "lisa-pack-core"
    assert plan.manifest.routing.sequential_connectors[0] == "then"
    # Local context wins (no merge).
    assert tuple(r.rule_id for r in plan.manifest.context.sources) == ("mine",)


def test_from_missing_block_errors(core_pack_root: Path, bare_pack_root: Path) -> None:
    """Pulling a block the pack session omits raises a clear SessionPlanError."""

    user_text = dedent(
        """\
        ---
        schema_version: 2
        name: mine
        surface: { type: personal_dm, id: host-console, memory_scope: personal }
        routing:
          from:
            pack: lisa-pack-minimal
            session: default
        ---
        # My session
        """
    )
    with pytest.raises(SessionPlanError) as exc:
        resolve_session_from_directives(
            text=user_text,
            source_uri="workspace:mine.md",
            pack_session_lookup=_lookup_for(core_pack_root, bare_pack_root),
        )
    assert "routing" in str(exc.value)
    assert "lisa-pack-minimal" in str(exc.value)


def test_from_mutual_exclusivity_errors(core_pack_root: Path) -> None:
    """A ``from:`` block with sibling keys is rejected."""

    user_text = dedent(
        """\
        ---
        schema_version: 2
        name: mine
        surface: { type: personal_dm, id: host-console, memory_scope: personal }
        routing:
          from: { pack: lisa-pack-core, session: default }
          start_mode: other
        ---
        # My session
        """
    )
    with pytest.raises(SessionPlanError) as exc:
        resolve_session_from_directives(
            text=user_text,
            source_uri="workspace:mine.md",
            pack_session_lookup=_lookup_for(core_pack_root),
        )
    assert "may not declare any other keys" in str(exc.value)


def test_from_pack_not_installed_errors(core_pack_root: Path) -> None:
    """An unknown pack name surfaces a clear error."""

    user_text = dedent(
        """\
        ---
        schema_version: 2
        name: mine
        surface: { type: personal_dm, id: host-console, memory_scope: personal }
        routing:
          from: { pack: lisa-pack-ghost, session: default }
        ---
        # My session
        """
    )
    with pytest.raises(SessionPlanError) as exc:
        resolve_session_from_directives(
            text=user_text,
            source_uri="workspace:mine.md",
            pack_session_lookup=_lookup_for(core_pack_root),
        )
    assert "lisa-pack-ghost" in str(exc.value)


def test_from_pack_session_file_missing_errors(core_pack_root: Path) -> None:
    """Installed pack but missing ``sessions/<name>.md`` errors clearly."""

    user_text = dedent(
        """\
        ---
        schema_version: 2
        name: mine
        surface: { type: personal_dm, id: host-console, memory_scope: personal }
        routing:
          from: { pack: lisa-pack-core, session: absent }
        ---
        # My session
        """
    )
    with pytest.raises(SessionPlanError) as exc:
        resolve_session_from_directives(
            text=user_text,
            source_uri="workspace:mine.md",
            pack_session_lookup=_lookup_for(core_pack_root),
        )
    assert "absent" in str(exc.value)
    assert "lisa-pack-core" in str(exc.value)


# ── ``body:`` directive ───────────────────────────────────────────────


def test_body_pull_uses_pack_body_and_allows_empty_user_body(
    core_pack_root: Path,
) -> None:
    """``body: pull`` substitutes the pack's body; user body may be empty."""

    user_text = dedent(
        """\
        ---
        schema_version: 2
        name: mine
        surface: { type: personal_dm, id: host-console, memory_scope: personal }
        body:
          from: { pack: lisa-pack-core, session: default }
          mode: pull
        ---

        """
    )
    resolved = resolve_session_from_directives(
        text=user_text,
        source_uri="workspace:mine.md",
        pack_session_lookup=_lookup_for(core_pack_root),
    )
    plan = compile_session_plan(text=resolved, source_uri="workspace:mine.md")
    assert "Use the available tools" in plan.instructions.primary_body
    # The directive is stripped before the kernel sees the front matter.
    assert "body" not in resolved.split("---")[1]


def test_body_context_prepends_pack_body_wrapped(core_pack_root: Path) -> None:
    """``body: context`` prepends the pack body in a wrapper tag."""

    user_text = dedent(
        """\
        ---
        schema_version: 2
        name: mine
        surface: { type: personal_dm, id: host-console, memory_scope: personal }
        body:
          from: { pack: lisa-pack-core, session: default }
          mode: context
        ---
        # My user body
        """
    )
    resolved = resolve_session_from_directives(
        text=user_text,
        source_uri="workspace:mine.md",
        pack_session_lookup=_lookup_for(core_pack_root),
    )
    plan = compile_session_plan(text=resolved, source_uri="workspace:mine.md")
    body = plan.instructions.primary_body
    assert body.startswith("<pack_session_body>")
    assert "Use the available tools" in body
    assert "My user body" in body
    # Wrapper closes before the user body.
    assert "</pack_session_body>" in body
    assert body.index("</pack_session_body>") < body.index("My user body")


def test_body_context_requires_nonempty_user_body(core_pack_root: Path) -> None:
    """``body: context`` with an empty user body is an error."""

    user_text = dedent(
        """\
        ---
        schema_version: 2
        name: mine
        surface: { type: personal_dm, id: host-console, memory_scope: personal }
        body:
          from: { pack: lisa-pack-core, session: default }
          mode: context
        ---

        """
    )
    with pytest.raises(SessionPlanError) as exc:
        resolve_session_from_directives(
            text=user_text,
            source_uri="workspace:mine.md",
            pack_session_lookup=_lookup_for(core_pack_root),
        )
    assert "context" in str(exc.value)


def test_body_write_is_default_when_directive_omitted(core_pack_root: Path) -> None:
    """No ``body:`` directive = the user body is primary (current behavior)."""

    user_text = dedent(
        """\
        ---
        schema_version: 2
        name: mine
        surface: { type: personal_dm, id: host-console, memory_scope: personal }
        ---
        # My session
        """
    )
    resolved = resolve_session_from_directives(
        text=user_text,
        source_uri="workspace:mine.md",
        pack_session_lookup=_lookup_for(core_pack_root),
    )
    plan = compile_session_plan(text=resolved, source_uri="workspace:mine.md")
    assert plan.instructions.primary_body.strip() == "# My session"


def test_body_invalid_mode_errors(core_pack_root: Path) -> None:
    """An unknown ``body: mode`` value is rejected."""

    user_text = dedent(
        """\
        ---
        schema_version: 2
        name: mine
        surface: { type: personal_dm, id: host-console, memory_scope: personal }
        body:
          from: { pack: lisa-pack-core, session: default }
          mode: merge
        ---
        # My session
        """
    )
    with pytest.raises(SessionPlanError) as exc:
        resolve_session_from_directives(
            text=user_text,
            source_uri="workspace:mine.md",
            pack_session_lookup=_lookup_for(core_pack_root),
        )
    assert "mode" in str(exc.value)


# ── Packs-are-leaves enforcement ──────────────────────────────────────


def test_pack_session_with_from_directive_is_rejected(
    core_pack_root: Path,
) -> None:
    """A pack session containing ``from:`` cannot be pulled (packs are leaves).

    The pack's own ``from:`` would survive the resolver's initial parse only
    if it reached the kernel; but because we compile the pack session through
    the strict ``SessionManifestV2`` schema (extra=\"forbid\"), the unknown
    ``from`` key is rejected. This test asserts the kernel-side rejection
    surfaces as a ``SessionPlanError``.
    """

    nested_pack = core_pack_root.parent / "lisa_pack_nested"
    nested_pack.mkdir()
    (nested_pack / "pyproject.toml").write_text(
        '[project]\nname = "lisa-pack-nested"\nversion = "0.0.0"\n', encoding="utf-8"
    )
    _write_pack_session(
        nested_pack,
        "default",
        dedent(
            """\
            ---
            schema_version: 2
            name: nested
            surface: { type: personal_dm, id: host-console, memory_scope: personal }
            routing:
              from: { pack: lisa-pack-core, session: default }
            ---
            # Nested body
            """
        ),
    )
    user_text = dedent(
        """\
        ---
        schema_version: 2
        name: mine
        surface: { type: personal_dm, id: host-console, memory_scope: personal }
        routing:
          from: { pack: lisa-pack-nested, session: default }
        ---
        # My session
        """
    )
    with pytest.raises(SessionPlanError):
        resolve_session_from_directives(
            text=user_text,
            source_uri="workspace:mine.md",
            pack_session_lookup=_lookup_for(core_pack_root, nested_pack),
        )


# ── Lookup primitive ─────────────────────────────────────────────────


def test_resolve_pack_session_path_alias_match(core_pack_root: Path) -> None:
    """The primitive accepts both ``lisa-pack-core`` and ``core``."""

    full = resolve_pack_session_path(
        pack="lisa-pack-core", session="default", pack_roots=[core_pack_root]
    )
    bare = resolve_pack_session_path(pack="core", session="default", pack_roots=[core_pack_root])
    assert full == bare
    assert full.is_file()


def test_resolve_pack_session_path_pack_missing() -> None:
    with pytest.raises(SessionPlanError):
        resolve_pack_session_path(pack="lisa-pack-ghost", session="default", pack_roots=[])


def test_resolve_pack_session_path_session_missing(core_pack_root: Path) -> None:
    with pytest.raises(SessionPlanError) as exc:
        resolve_pack_session_path(
            pack="lisa-pack-core", session="absent", pack_roots=[core_pack_root]
        )
    assert "absent" in str(exc.value)

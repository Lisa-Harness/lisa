"""Test fixtures for host tests.

Registers the lisa-catalog MCP server so ``get_workflow_catalog()`` works
in tests that exercise command dispatch (``/help``, ``/debug``, etc.).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from kernel.ipsar_fsm import (
    get_workflow_catalog,
    register_content_mcp,
    register_operation_callbacks,
)
from kernel.operations.registry import (
    execute_operation,
    get_operation_catalog,
    register_operation_mcp,
)
from kernel.tools.mcp_client import MCPClientConfig

_CORE_ROOT: Path = (Path(__file__).resolve().parents[2] / "sdk" / "packs" / "core").resolve()
_LEARNING_ROOT: Path = (Path(__file__).resolve().parents[2] / "sdk" / "packs" / "memory").resolve()


@pytest.fixture(autouse=True)
def _isolate_default_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ``HostConfig.from_env()`` defaults off the developer's real ``~/.lisa``.

    The live host process exports runtime state at startup
    (``runtime.mcp_pool``: ``LISA_THREAD_DIR``, ``FRAMEWORK_ROOT``,
    ``LISA_TIMEZONE``, ``LISA_USER_SLUG``); a test run launched from that
    host (the sandboxed coding `test` check) inherits them, and config
    resolution would then resolve the REAL thread dir despite an explicit
    ``data_dir`` (2026-10-15: 78 ``Errno 30`` failures under the sandbox's
    read-only data-dir mount). Scrub every ambient runtime-state var here;
    tests that need one set it themselves afterwards.
    """
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    for name in (
        "LISA_CONTEXT_BUDGET_REPLAY_MAX_BYTES",
        "LISA_DATA_DIR",
        "LISA_PROFILE",
        "LISA_PACKS_VENV",
        "LISA_THREAD_DIR",
        "FRAMEWORK_ROOT",
        "LISA_USER_SLUG",
        "LISA_USER_ID",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _environ_hermeticity_guard() -> Any:
    """Restore ``os.environ`` after every test and fail on a leak.

    The runtime's own autouse guards cover ``kernel.config.settings`` and
    ``sdk.config.settings``; the process environment had no equivalent,
    so a test that wrote to ``os.environ`` directly (bypassing
    ``monkeypatch``) leaked its values into every later test. The
    runtime's MCP export was one such writer (Lisa-Harness/lisa#3): the
    exported values outlived the compile call, and the next test's
    ``HostConfig.from_env()`` resolved its thread dir, user slug,
    timezone and thread-redaction from the previous runtime's values
    instead of its own.

    This guard snapshots the environment before the test and restores it
    afterwards, so a leak cannot reach the next test. It fails the test
    that caused the leak rather than letting the damage surface later
    as an unrelated order-dependent failure.

    Only the runtime-state variables are compared: the suite legitimately
    mutates a few others (``LISA_RESTART_RESUME_MESSAGE``), and
    ``PYTEST_CURRENT_TEST`` changes on its own as pytest advances
    through the test's own setup/call/teardown phases.
    """
    watched = (
        "FRAMEWORK_ROOT",
        "LISA_CONTEXT_BUDGET_REPLAY_MAX_BYTES",
        "LISA_DATA_DIR",
        "LISA_PACKS_VENV",
        "LISA_PROFILE",
        "LISA_SESSION_CONTEXT_PATH",
        "LISA_THREAD_DIR",
        "LISA_THREAD_REDACTION_ENABLED",
        "LISA_THREAD_REDACTION_PATTERNS",
        "LISA_TIMEZONE",
        "LISA_USER_ID",
        "LISA_USER_SLUG",
    )
    before = {name: os.environ.get(name) for name in watched}
    yield

    after = {name: os.environ.get(name) for name in watched}
    leaked = {name: (before[name], after[name]) for name in watched if before[name] != after[name]}
    for name, prior in before.items():
        if prior is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = prior
    if leaked:
        diff = "\n".join(
            f"  {name}: before={old!r} after={new!r}" for name, (old, new) in sorted(leaked.items())
        )
        raise AssertionError(
            "Test mutated the process environment without restoring it.\n"
            "The suite requires hermeticity; use monkeypatch for env changes "
            "so they are undone at teardown.\n"
            f"Leaked variables:\n{diff}"
        )


# search, markdown, and letta migrated to ../../lisa_packs/<name>
# (external submodules). Only include a content root when the submodule is
# checked out alongside the framework; tests that need a migrated mode skip
# when absent.
_PACKS_ROOT: Path = (Path(__file__).resolve().parents[2].parent / "lisa_packs").resolve()


def _external_pack_roots() -> list[Path]:
    """Return the content roots of migrated packs that are checked out."""
    roots: list[Path] = []
    for name in ("search", "markdown", "letta"):
        root = (_PACKS_ROOT / name / "lisa_packs" / name).resolve()
        if root.exists():
            roots.append(root)
    return roots


def _make_catalog_mcp_config(*, log_path: Path) -> MCPClientConfig:
    """Build an MCPClientConfig that spawns lisa-catalog with the core pack.

    ``log_path`` routes the child's JSONL log out of the repo root (the
    ``configure_logging`` fallback writes ``<cwd>/lisa-catalog.jsonl`` when
    ``LISA_MCP_LOG_PATH`` is unset, and these configs leave cwd=None so the
    subprocess inherits the pytest cwd). Mirrors the env injection that
    ``runtime.mcp_pool.build_catalog_mcp_config`` does in production.
    """
    content_roots = [_CORE_ROOT, _LEARNING_ROOT, *_external_pack_roots()]
    return MCPClientConfig(
        server_id="lisa-catalog-test",
        command=sys.executable,
        args=[
            "-m",
            "mcp_server.catalog.server",
            "--data-dir",
            str(_CORE_ROOT),
        ],
        env={
            "SDK_CONTENT_ROOTS": os.pathsep.join(str(r) for r in content_roots),
            "LISA_SESSION_PROVIDER_SELECTIONS": json.dumps(
                {
                    "modes": {"default": "lisa-pack-core"},
                    "tools": {"web_search": "lisa-pack-brave"},
                }
            ),
            "LISA_MCP_LOG_PATH": str(log_path),
        },
    )


@pytest.fixture(scope="session", autouse=True)
def _mcp_catalog_setup(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Wire the catalog MCP, operation callbacks, and content registry.

    This mirrors what ``runtime.mcp_pool.setup_framework(runtime)`` does,
    but without requiring a HostRuntime: it builds the catalog and content
    MCP configs directly and registers them so the kernel can load catalogs
    and read authored text via the content registry.
    """
    log_dir = tmp_path_factory.mktemp("lisa-catalog")
    config = _make_catalog_mcp_config(log_path=log_dir / "lisa-catalog.jsonl")
    register_operation_mcp(config)
    register_operation_callbacks(get_operation_catalog, execute_operation)
    register_content_mcp(config)

    # Wire the MCPContentRegistry as the content registry for authored context
    # assets and the sub-turn contract assets (fork/delegate briefings + midpoint
    # write-nudge). Both modules share the same registry instance.
    from kernel.context.content_registry import set_content_registry
    from kernel.context.mcp_content_registry import MCPContentRegistry
    from kernel.context.sub_turn_contract import (
        set_content_registry as _set_sub_turn_content_registry,
    )

    _registry = MCPContentRegistry(config)
    set_content_registry(_registry)
    _set_sub_turn_content_registry(_registry)

    # Eagerly load the workflow catalog now (sync context, before any async test runs).
    get_workflow_catalog()


@pytest.fixture(autouse=True)
def _routing_vocabulary() -> None:
    """Register a representative routing vocabulary for each test.

    The kernel has no hardcoded fallbacks (2026-08-11 cutover): a reader
    raises ``RuntimeError`` when no session vocabulary is wired. Host tests
    that call ``handle_message`` read resume phrases via
    ``get_resume_phrases()``; register a representative set so the common
    case works without each test opting in.
    """
    from kernel.routing_vocabulary import (
        register_routing_vocabulary,
        reset_routing_vocabulary,
    )

    reset_routing_vocabulary()
    register_routing_vocabulary(
        sequential_connectors=["then", "and then", "after that", "next", "followed by"],
        producer_intents=["research", "gather", "learn", "read", "analyze", "review", "architect"],
        consumer_intents=["code", "write", "implement", "review"],
        resume_phrases=[
            "proceed",
            "continue",
            "go ahead",
            "go for it",
            "do it",
            "carry on",
            "pick up",
            "resume",
            "next step",
            "keep going",
        ],
        topic_abandonment_markers=[
            "side quest",
            "rewind",
            "never mind",
            "forget",
            "different topic",
            "not about",
        ],
    )
    yield
    reset_routing_vocabulary()

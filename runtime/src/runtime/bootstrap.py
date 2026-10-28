"""Shared composition root for host runtimes.

The CLI host (``hosts/cli/main.py``)
builds a :class:`runtime.runtime.HostRuntime` from CLI args + env vars
using the same ordered sequence:

    HostConfig.from_env  ->  HostRuntime  ->  load_state_snapshot
        ->  discover_framework  ->  setup_framework
        ->  configure_runtime  ->  compile_mcp_configs  ->  set_mcp_configs

Previously each host duplicated this ~30-line sequence in its own
``_initialize_runtime``. This module owns the single
:func:`build_runtime` factory so the two hosts cannot drift. The caller
still owns the async lifecycle (``runtime.start()`` / ``runtime.aclose()``)
— :func:`build_runtime` does **not** enter async resources, honoring the
hosts-architecture invariant that MCP resources start and close in the
same task/event loop that runs turns.

Heavy imports (``HostRuntime``, ``runtime.mcp_pool``, the state
provider) are deferred inside :func:`build_runtime` so that importing this
module does not drag in ``kernel.ipsar_fsm`` / ``openai`` — the CLI
``--version`` / ``modes`` / ``verify`` fast paths must stay cheap.

Dependency direction: ``runtime.bootstrap`` ->
``runtime.config``, ``runtime.runtime``, ``runtime.mcp_pool``,
``runtime.state_provider``, ``sdk.logging_setup``. No imports from
transport hosts.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from runtime.config import HostConfig

if TYPE_CHECKING:
    from runtime.runtime import HostRuntime

logger = logging.getLogger(__name__)

# A logging configurator receives the resolved ``HostConfig`` and configures
# stdlib root logging. The default is the shared ``sdk.logging_setup``
# configurator; tests (and hosts that need a no-op) inject their own.
LoggingConfigurator = Callable[[HostConfig], None]


def build_runtime(
    *,
    host_kind: str,
    debug: bool = False,
    log_level: str = "",
    log_path: str = "",
    no_log: bool = False,
    data_dir: str = "",
    overlays_dir: str = "",
    thread_dir: str = "",
    no_thread: bool = False,
    user_slug: str = "",
    conversation_id: str = "",
    timezone: str = "",
    session_context: str = "",
    state_snapshot_path: str = "",
    streaming: str = "hybrid",
    host_log_path: str = "",
    surface_type: str = "",
    surface_id: str = "",
    logging_configurator: LoggingConfigurator | None = None,
    in_process_mcp: bool = False,
) -> HostRuntime:
    """Build and wire a ``HostRuntime`` from CLI args + env vars.

    Runs the canonical composition sequence (config -> runtime -> state
    snapshot -> framework wiring -> MCP config compilation) and returns
    the runtime. The MCP pool is **not** started here — the caller must
    invoke ``runtime.start()`` inside its own event loop and
    ``runtime.aclose()`` on shutdown.

    ``host_kind`` is stamped onto the config via ``dataclasses.replace``
    so the transport-neutral config carries its host identity for logging
    and telemetry without a host-specific subclass.

    ``logging_configurator`` configures stdlib root logging from the
    resolved config. When ``None`` (default), the shared
    ``sdk.logging_setup.configure_logging`` is used; callers that need a
    no-op (tests) or a host-specific setup pass their own.
    """
    # Lazy imports keep --version / modes / verify fast: ``HostRuntime`` +
    # the mcp_pool symbols pull in ``kernel.ipsar_fsm`` + ``openai``.
    from runtime.mcp_pool import (
        compile_mcp_configs,
        compile_proxy_configs,
        configure_runtime,
        discover_framework,
        setup_framework,
    )
    from runtime.runtime import HostRuntime
    from runtime.state_provider import load_state_snapshot

    _ = no_log  # honored via HostConfig.log_path resolution below

    config = HostConfig.from_env(
        debug=debug,
        log_level=log_level,
        log_path=log_path,
        host_log_path=host_log_path,
        data_dir=data_dir,
        overlays_dir=overlays_dir,
        thread_dir=thread_dir,
        no_thread=no_thread,
        timezone=timezone,
        session_context=session_context,
        user_slug=user_slug,
        conversation_id=conversation_id,
        state_snapshot=state_snapshot_path,
        streaming=streaming,
        surface_type=surface_type,
        surface_id=surface_id,
    )
    config = dataclasses.replace(config, host_kind=host_kind)

    _configure_logging(config, logging_configurator)

    # ── Thread redaction policy ───────────────────────────────────────
    # Wire the redaction policy into the module-level ``runtime.thread``
    # factory so every ThreadStore created by the host applies the
    # configured strikethrough transform on read. The policy is compiled
    # once from ``HostConfig`` and shared for the lifetime of the host.
    #
    # Pattern sources (in priority order):
    # 1. ``LISA_THREAD_REDACTION_PATTERNS`` env override — when non-empty,
    #    it replaces all other sources (hard cutover, no merge).
    # 2. Pack-contributed patterns — packs ship
    #    ``content/redaction_patterns.yml``; the host scans active pack
    #    content roots and merges all discovered patterns.
    # 3. Framework defaults — ``DEFAULT_REDACTION_PATTERNS`` in
    #    ``sdk.thread.redaction``.
    #
    # When no env override is set, pack patterns are merged with the
    # framework defaults (union, deduplicated, order-stable) so a pack's
    # best-practice redactions augment the baseline rather than replacing
    # it.
    from sdk.thread import DEFAULT_REDACTION_PATTERNS, ThreadRedactionPolicy

    from runtime.content_roots import discover_pack_redaction_patterns
    from runtime.thread import set_thread_redaction_policy as _set_thread_redaction

    if config.thread_redaction_enabled:
        if config.thread_redaction_patterns:
            # Env override replaces everything (hard cutover).
            merged_patterns = list(config.thread_redaction_patterns)
        else:
            # Merge framework defaults + pack-contributed patterns.
            pack_patterns = discover_pack_redaction_patterns(
                config.data_dir,
                always_on_packs=config.always_on_packs,
            )
            merged_patterns = list(DEFAULT_REDACTION_PATTERNS)
            seen = set(merged_patterns)
            for p in pack_patterns:
                if p not in seen:
                    seen.add(p)
                    merged_patterns.append(p)
        _set_thread_redaction(ThreadRedactionPolicy.from_patterns(merged_patterns))
    else:
        _set_thread_redaction(None)

    # Register the built-in host-command vocabulary before any turn can be
    # routed. Done here — once, in the composition root — rather than at
    # ``handle_message`` import time so the kernel rule "registration must
    # be side-effect-free on import" holds and tests do not trigger global
    # registry mutation just by importing the dispatcher.
    from runtime.host_commands import register_builtin_host_commands

    register_builtin_host_commands()

    runtime = HostRuntime(config, in_process_mcp=in_process_mcp)
    runtime.session.state_snapshot = load_state_snapshot(
        explicit_path=state_snapshot_path,
        data_dir=config.data_dir,
    )
    discovery = discover_framework(runtime)
    setup_framework(runtime, discovery=discovery)
    configure_runtime(runtime)
    warm, cold = compile_mcp_configs(runtime, discovery=discovery)
    runtime.resources.set_mcp_configs(warm, cold)
    runtime.resources.set_proxy_configs(compile_proxy_configs(runtime, discovery=discovery))
    return runtime


def _configure_logging(
    config: HostConfig,
    configurator: LoggingConfigurator | None,
) -> None:
    """Run the logging configurator, defaulting to the shared setup."""
    if configurator is not None:
        configurator(config)
        return
    from sdk.logging_setup import configure_logging

    configure_logging(
        log_path=config.resolved_host_log_path,
        level=config.effective_log_level,
        debug_modules=list(config.debug_log_modules),
        host_kind=config.host_kind,
    )

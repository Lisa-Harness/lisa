"""Shared composition root for host runtimes.

The CLI host (``hosts/cli/main.py``)
builds a :class:`host.runtime.HostRuntime` from CLI args + env vars
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

Heavy imports (``HostRuntime``, ``host.mcp_pool``, the state
provider) are deferred inside :func:`build_runtime` so that importing this
module does not drag in ``kernel.ipsar_fsm`` / ``openai`` — the CLI
``--version`` / ``modes`` / ``verify`` fast paths must stay cheap.

Dependency direction: ``host.bootstrap`` ->
``host.config``, ``host.runtime``, ``host.mcp_pool``,
``host.state_provider``, ``sdk.logging_setup``. No imports from
transport hosts.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from host.config import HostConfig

if TYPE_CHECKING:
    from host.runtime import HostRuntime

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
    timezone: str = "",
    session_context: str = "",
    session_context_refresh: bool = False,
    state_snapshot_path: str = "",
    streaming: str = "hybrid",
    host_log_path: str = "",
    logging_configurator: LoggingConfigurator | None = None,
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
    from host.mcp_pool import (
        compile_mcp_configs,
        configure_runtime,
        discover_framework,
        setup_framework,
    )
    from host.runtime import HostRuntime
    from host.state_provider import load_state_snapshot

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
        session_context_refresh=session_context_refresh,
        user_slug=user_slug,
        state_snapshot=state_snapshot_path,
        streaming=streaming,
    )
    config = dataclasses.replace(config, host_kind=host_kind)

    _configure_logging(config, logging_configurator)

    # ── Thread redaction policy ───────────────────────────────────────
    # Wire the redaction policy into the module-level ``host.thread``
    # factory so every ThreadStore created by the host applies the
    # configured strikethrough transform on read. The policy is compiled
    # once from ``HostConfig`` and shared for the lifetime of the host.
    from host.thread import set_thread_redaction_policy as _set_thread_redaction
    from sdk.thread import DEFAULT_REDACTION_PATTERNS, ThreadRedactionPolicy

    if config.thread_redaction_enabled:
        patterns: list[str] = list(DEFAULT_REDACTION_PATTERNS)
        if config.thread_redaction_patterns:
            # Non-empty override replaces defaults entirely (hard cutover,
            # no merge — per the project convention).
            patterns = list(config.thread_redaction_patterns)
        _set_thread_redaction(ThreadRedactionPolicy.from_patterns(patterns))
    else:
        _set_thread_redaction(None)

    runtime = HostRuntime(config)
    runtime.state_snapshot = load_state_snapshot(
        explicit_path=state_snapshot_path,
        data_dir=config.data_dir,
    )
    discovery = discover_framework(runtime)
    setup_framework(runtime, discovery=discovery)
    configure_runtime(runtime)
    warm, cold = compile_mcp_configs(runtime, discovery=discovery)
    runtime.set_mcp_configs(warm, cold)
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

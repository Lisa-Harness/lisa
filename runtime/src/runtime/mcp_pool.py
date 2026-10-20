"""Transport-neutral MCP server config compilation.

This module consolidates the logic for compiling MCP server configurations
from the host manifest (``mcp_servers.yaml``) and any pack-contributed
manifests discovered under the active SDK content roots. It produces
``MCPClientConfig`` entries split into warm/cold lists for
:class:`kernel.tools.mcp_client.MCPToolPool`, plus the content/catalog
server configs that wire the kernel's content registry and operation/tool
catalogs.

All functions are transport-neutral: they take an explicit
:class:`runtime.runtime.HostRuntime` (or its derived paths) rather than
reading a global ``_RUNTIME_STATE`` dict. Transport hosts (CLI, web, VS
Code) construct a runtime and call :func:`setup_framework` /
:func:`compile_mcp_configs` during initialization.

The compiled configs are stored on the runtime via
``runtime.resources.set_mcp_configs(warm, cold)``; pool creation/start/close lives
in the host's async lifecycle (``runtime.start()`` / ``runtime.aclose()``),
never in this module.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernel.extension_names import normalize_pack_name
from kernel.mcp_launch.compiler import compile_registry
from kernel.mcp_launch.policy import MCPServerRegistry
from kernel.mcp_launch.server_manifest import (
    MCPServerManifestError,
    MemoryBackendConfig,
    load_mcp_server_manifest,
)
from kernel.proxy_launch.compiler import ProxyClientConfig, compile_proxy_registry
from kernel.proxy_launch.manifest import ProxyServerManifestError, load_proxy_server_manifest
from kernel.proxy_launch.policy import ProxyServerRegistry
from kernel.session_plan import ProviderSelections
from kernel.tools.mcp_client import MCPClientConfig
from sdk.layout import LOGS_DIR, MCP_LOGS_SUBDIR

from runtime.config import HostConfig
from runtime.content_roots import (
    always_on_pack_names,
    host_mcp_manifest_path,
    infer_pack_package_name,
    load_disabled_pack_names,
    load_installed_packs_lock,
    load_pack_policy_dict,
    sdk_content_roots,
    seed_bundled_packs_lock,
)
from runtime.provider_endpoints import direct_provider_base_urls
from runtime.runtime import HostRuntime

logger = logging.getLogger(__name__)


class PackPrerequisiteError(RuntimeError):
    """A registered pack cannot satisfy a host-level prerequisite."""


def _validate_pack_prerequisites() -> None:
    """Fail startup loudly when a pack's host prerequisites are missing.

    Packs expose a zero-arg callable via the
    ``lisa.extensions.prerequisites`` entry point that returns a list of miss
    reasons (empty = can run). Any non-empty list aborts startup, so a""
    present-but-cannot-run sandbox (or any other capability pack) never
    silently degrades to a less-safe posture without the operator knowing.
    A broken callable (raises) is logged and skipped - a faulty probe must not
    itself take the host down; only a *reported* miss is fatal.
    """
    from importlib import metadata

    try:
        eps = metadata.entry_points(group="lisa.extensions.prerequisites")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("pack prerequisites: entry-point discovery failed (%s)", exc)
        return
    problems: list[str] = []
    for ep in eps:
        try:
            checker = ep.load()
            reasons = checker()
        except Exception as exc:
            logger.warning("pack prerequisites: %s probe failed (%s); skipping", ep.name, exc)
            continue
        if reasons:
            problems.extend(f"[{ep.name}] {reason}" for reason in reasons)
    if problems:
        raise PackPrerequisiteError(
            "Pack prerequisites unmet; refusing to start with a silently degraded "
            "capability:\n  - " + "\n  - ".join(problems)
        )


# Venv root containing the console-script entry points (lisa-context,
# lisa-catalog, etc.). In repo mode this is <repo>/.venv; in tool-install
# mode this is sys.prefix. Exported as the FRAMEWORK_ROOT env var so that
# mcp_servers.yaml's ``venv_dir: "${FRAMEWORK_ROOT}"`` resolves to the
# directory whose Scripts/ (Windows) or bin/ (POSIX) holds the executables.
from runtime.paths import packs_venv_for as _packs_venv_for  # noqa: E402
from runtime.paths import packs_venv_site_packages as _packs_venv_site_packages  # noqa: E402
from runtime.paths import venv_root as _venv_root  # noqa: E402

FRAMEWORK_ROOT = _venv_root()


def _resolve_packs_venv_site_packages(data_dir: Path) -> Path | None:
    """Resolve the packs venv site-packages directory for a host data_dir.

    Returns the site-packages path if it exists, otherwise ``None``. The
    result is forwarded to :func:`sdk.mcp_host.build_content_mcp_config`
    so the child ``lisa-content`` / ``lisa-catalog`` MCP servers inherit
    the packs venv site-packages via ``LISA_PACKS_VENV_SITE_PACKAGES`` and
    can import editable-installed packs (``lisa_packs.<name>``).

    The host process calls ``inject_packs_venv_site_packages()`` at startup
    to put the packs venv on ``sys.path``, but child processes are fresh
    interpreters that only inherit env vars — without this, operation
    modules like ``lisa_packs.markdown.operations.forge.forge_document``
    raise ``ModuleNotFoundError: No module named 'lisa_packs'`` in the child.
    """
    packs_venv = _packs_venv_for(data_dir)
    site_packages = _packs_venv_site_packages(packs_venv)
    if not site_packages.is_dir():
        return None
    return site_packages


# ── Shared discovery cache ─────────────────────────────────────────


@dataclasses.dataclass(slots=True)
class _FrameworkDiscovery:
    """Shared discovery result for setup_framework + compile_mcp_configs.

    Both functions need the same sdk_content_roots walk, disabled-pack
    set, and installed-packs lock. Computing it once in
    :func:`discover_framework` and passing it to both avoids a redundant
    filesystem walk + lock read on every startup.

    ``framework_manifest_path`` is always the in-repo ``host/
    mcp_servers.yaml`` — the fixed base layer of first-party servers.
    ``user_manifest_path`` is the optional user-editable manifest at
    ``<data_dir>/mcp_servers.yaml`` (or an explicit override). It may be
    ``None`` when the file does not exist; callers treat absence as "no
    user servers" (same graceful first-run convention as ``models.yaml``).
    """

    roots: list[Path]
    disabled_packs: set[str]
    lock: dict[str, Any]
    host_manifest_path: Path
    user_manifest_path: Path | None = None


def discover_framework(
    runtime: HostRuntime,
    *,
    seed_lock: bool = True,
) -> _FrameworkDiscovery:
    """Compute the shared discovery data once for a runtime.

    Call this in the host's ``_initialize_runtime`` after constructing the
    runtime, then pass the result to :func:`setup_framework` and
    :func:`compile_mcp_configs` via their optional ``discovery`` parameter.
    When the caller does not pass ``discovery``, each function falls back
    to calling this internally (preserving standalone-test back-compat).

    When ``seed_lock`` is True (default), the installed-packs lock is
    seeded with any discovered bundled packs before reading. This matches
    the original :func:`setup_framework` behavior. Callers that must not
    seed (e.g. :func:`compile_mcp_configs` falling back standalone)
    pass ``seed_lock=False``.
    """
    config = runtime.config
    manifest_path = _resolve_framework_manifest_path(config)
    user_manifest_path = _resolve_user_manifest_path(config)
    # Seed before reading the lock so discovery sees a consistent state.
    # This closes the lifecycle hole where an unlisted bundled pack was
    # treated as enabled by intake/evidence (fail-open) but disabled by
    # content roots (default-off). compile_mcp_configs skips seeding to
    # preserve its original standalone behavior (no side effects).
    if seed_lock:
        seed_bundled_packs_lock(runtime.paths.data_dir)
    # Auto-install any session-required packs that are not yet importable,
    # before content-root discovery runs. When at least one pack is installed,
    # request a restart: importlib.metadata entry points are cached per-
    # process and may not see the newly installed pack without a relaunch.
    # On the second boot pass this is a no-op and discovery proceeds.
    from runtime.pack_autoinstall import auto_install_required_packs

    installed_packs = auto_install_required_packs(runtime)
    if installed_packs:
        runtime.session.request_restart(reason=f"auto-install: {', '.join(installed_packs)}")
    plan_packs = runtime.session_plan.manifest.resources.packs
    # Provider owners count as included: selecting a pack as the memory/mode
    # owner while leaving it out of `required` must not drop its MCP manifest.
    session_pack_include = tuple(sorted(runtime.session_plan.manifest.declared_pack_owners()))
    session_pack_exclude = plan_packs.exclude
    roots = sdk_content_roots(
        runtime.paths.data_dir,
        host_manifest_path=manifest_path,
        always_on_packs=config.always_on_packs,
        session_pack_include=session_pack_include,
        session_pack_exclude=session_pack_exclude,
    )
    host_mandatory_packs = always_on_pack_names(
        manifest_path,
        always_on_packs=config.always_on_packs,
    )
    disabled_packs = load_disabled_pack_names(runtime.paths.data_dir)
    disabled_packs.difference_update(
        normalize_pack_name(pack_name) for pack_name in session_pack_include
    )
    disabled_packs.difference_update(host_mandatory_packs)
    disabled_packs.update(
        normalize_pack_name(pack_name)
        for pack_name in session_pack_exclude
        if normalize_pack_name(pack_name) not in host_mandatory_packs
    )
    lock = load_installed_packs_lock(runtime.paths.data_dir)
    return _FrameworkDiscovery(
        roots=roots,
        disabled_packs=disabled_packs,
        lock=lock,
        host_manifest_path=manifest_path,
        user_manifest_path=user_manifest_path,
    )


# ── Content / catalog server configs ─────────────────────────────────


def build_content_mcp_config(
    *,
    data_dir: Path,
    content_roots: list[Path],
    disabled_packs: set[str],
    provider_selections: ProviderSelections,
    overlays_dir: Path | None = None,
) -> MCPClientConfig:
    """Build an MCPClientConfig that spawns lisa-content for discovered pack roots.

    Thin host-side wrapper over :func:`sdk.mcp_host.build_content_mcp_config`
    that resolves the packs venv site-packages via this host's path machinery
    (``runtime.paths``) and forwards it. The pure projection logic lives
    in the SDK so external hosts can reuse it without depending on
    ``runtime``.

    See :func:`sdk.mcp_host.build_content_mcp_config` for the full
    parameter contract; only the host-owned ``packs_venv_site_packages``
    resolution is added here.
    """
    from sdk.mcp_host import build_content_mcp_config as _build

    return _build(
        data_dir=data_dir,
        content_roots=content_roots,
        disabled_packs=disabled_packs,
        provider_selections=provider_selections,
        packs_venv_site_packages=_resolve_packs_venv_site_packages(data_dir),
        overlays_dir=overlays_dir,
    )


def session_provider_selection_payload(
    selections: ProviderSelections,
) -> dict[str, dict[str, str]]:
    """Convert immutable plan ownership choices into child-process payload data.

    Re-exported from :mod:`sdk.mcp_host.content_servers` for back-compat of
    existing ``runtime.mcp_pool`` call sites. New code should import from
    ``sdk.mcp_host`` directly.
    """
    from sdk.mcp_host import (
        session_provider_selection_payload as _payload,
    )

    return _payload(selections)


def build_catalog_mcp_config(*, content_config: MCPClientConfig, data_dir: Path) -> MCPClientConfig:
    """Build an MCPClientConfig that spawns lisa-catalog for catalog tools.

    Thin host-side wrapper over :func:`sdk.mcp_host.build_catalog_mcp_config`.
    The pure projection logic lives in the SDK so external hosts can reuse
    it without depending on ``runtime``.
    """
    from sdk.mcp_host import build_catalog_mcp_config as _build

    return _build(content_config=content_config, data_dir=data_dir)


def _collect_default_tool_ids(
    specs: list[Any],
    *,
    server_id_prefix: str = "",
) -> set[str]:
    """Collect namespaced ambient tool ids from one manifest's specs.

    Args:
        specs: Parsed ``MCPServerSpec`` entries from one manifest.
        server_id_prefix: Optional pack slug applied at compile time
            (``<pack_slug>.<server_id>``); empty for first-party/user
            manifests which keep bare server ids.

    Returns:
        The set of namespaced ``server_id.tool`` ambient tool ids.
    """
    from kernel.tools.mcp_client import namespaced_tool_name

    ids: set[str] = set()
    for spec in specs:
        if spec.disabled or not spec.default_available_tools:
            continue
        namespaced_server_id = (
            f"{server_id_prefix}.{spec.server_id}" if server_id_prefix else spec.server_id
        )
        for tool_name in spec.default_available_tools:
            ids.add(namespaced_tool_name(namespaced_server_id, tool_name))
    return ids


def resolve_default_mcp_tool_ids(
    *,
    host_manifest_path: Path,
    pack_roots: list[Path],
    user_manifest_path: Path | None = None,
) -> set[str]:
    """Resolve the namespaced ambient MCP-native tool id set.

    Walks the host manifest plus every pack-contributed ``mcp_servers.yaml``
    and collects ``default_available_tools`` entries from each spec, namespacing
    them as ``server_id.tool``. Pack-contributed servers are registered under
    namespaced ids (``<pack_slug>.<server_id>``) at compile time, so the
    ambient id the model sees is the same namespaced form the catalog merge
    and the turn-runner filter use; this helper emits that compile-time
    namespacing here too (using the pack slug derived the same way as
    :func:`compile_mcp_configs`). Malformed pack manifests are skipped
    defensively, mirroring :func:`compile_mcp_configs`.

    When ``user_manifest_path`` is supplied and exists, its
    ``default_available_tools`` entries are collected with the same bare-id
    namespacing as the framework manifest — user-added servers are not
    pack-prefixed. Malformed user manifests are skipped defensively.
    """
    ids: set[str] = set()
    # Host manifest first (first-party servers keep their bare server_id).
    try:
        host_specs, _ = load_mcp_server_manifest(host_manifest_path)
    except MCPServerManifestError:
        host_specs = []
    ids |= _collect_default_tool_ids(host_specs)
    # User manifest (bare server_id, same as first-party). User-added
    # servers are not pack-prefixed; their default_available_tools become
    # ambient ids the same way lisa-storage.read_file does.
    if user_manifest_path is not None and user_manifest_path.exists():
        try:
            user_specs, _ = load_mcp_server_manifest(user_manifest_path)
        except MCPServerManifestError:
            user_specs = []
        ids |= _collect_default_tool_ids(user_specs)
    # Pack manifests (namespaced as <pack_slug>.<server_id> at compile time).
    used_pack_slugs: set[str] = set()
    for root in pack_roots:
        pack_manifest_path = root / "mcp_servers.yaml"
        if not pack_manifest_path.exists():
            continue
        try:
            pack_specs, _ = load_mcp_server_manifest(pack_manifest_path)
        except MCPServerManifestError:
            continue
        pack_slug = unique_pack_slug_for_root(root, used_pack_slugs)
        ids |= _collect_default_tool_ids(pack_specs, server_id_prefix=pack_slug)
    return ids


# ── Framework wiring ─────────────────────────────────────────────────


def setup_framework(
    runtime: HostRuntime,
    *,
    discovery: _FrameworkDiscovery | None = None,
) -> None:
    """Wire the lisa-content and lisa-catalog MCP servers onto the runtime.

    File-reading (``read_authored_text``) goes through ``lisa-content``.
    Catalog queries (``list_modes``, ``list_operations``, ``list_tools``)
    go through ``lisa-catalog``. The kernel never touches the filesystem
    directly — it calls these tools on the appropriate server.

    Order matters: operation MCP first, then operation callbacks, then
    content MCP. ``validate_workflow_catalog`` calls
    ``_validate_mode_operation_bindings`` which needs the operation
    callbacks registered before it can validate.

    When ``discovery`` is provided (computed once by
    :func:`discover_framework`), the shared roots / disabled-packs / lock
    are reused instead of recomputed. When ``None``, this function calls
    :func:`discover_framework` internally (back-compat for standalone
    callers like tests).

    This function is also the home for kernel-state side effects that were
    previously in ``HostRuntime.__init__``: the session routing-vocabulary
    registration (a kernel module-global mutation) and the durable
    workflow-instance boot-load (file I/O + ``current_mode_id`` override).
    Keeping construction side-effect-free means ``HostRuntime(config)`` is
    pure and ``setup_framework`` owns all kernel/world mutation.
    """
    data_dir = runtime.paths.data_dir

    # ── Session routing vocabulary ────────────────────────────────────
    # Inject the session-authored sequential-connector list into the
    # kernel registry so the routing-hint generator and the plan-mode
    # planner share one vocabulary source. Empty list (the default)
    # resets to the historical built-in set; sessions opt into richer
    # coverage via ``routing.sequential_connectors`` front matter.
    from kernel.routing_vocabulary import register_routing_vocabulary

    register_routing_vocabulary(
        sequential_connectors=runtime.session_plan.manifest.routing.sequential_connectors,
        resume_phrases=runtime.session_plan.manifest.routing.resume_phrases,
        producer_intents=runtime.session_plan.manifest.routing.producer_intents,
        consumer_intents=runtime.session_plan.manifest.routing.consumer_intents,
        topic_abandonment_markers=runtime.session_plan.manifest.routing.topic_abandonment_markers,
    )

    if discovery is None:
        discovery = discover_framework(runtime, seed_lock=True)
    roots = discovery.roots
    disabled_packs = discovery.disabled_packs

    content_config = build_content_mcp_config(
        data_dir=data_dir,
        content_roots=roots,
        disabled_packs=disabled_packs,
        provider_selections=runtime.session_plan.manifest.providers,
        overlays_dir=runtime.paths.overlays_dir,
    )
    default_mcp_tool_ids = resolve_default_mcp_tool_ids(
        host_manifest_path=discovery.host_manifest_path,
        pack_roots=roots,
        user_manifest_path=discovery.user_manifest_path,
    )
    catalog_config = build_catalog_mcp_config(content_config=content_config, data_dir=data_dir)
    if default_mcp_tool_ids:
        # build_catalog_mcp_config always returns a populated env dict, but
        # MCPClientConfig.env is typed dict|None, so normalize defensively
        # before mutating (the empty-default case is skipped above).
        if catalog_config.env is None:
            catalog_config = dataclasses.replace(catalog_config, env={})
        catalog_config.env["LISA_DEFAULT_MCP_TOOLS"] = ",".join(sorted(default_mcp_tool_ids))

    from kernel.ipsar_fsm import register_content_mcp, register_operation_callbacks
    from kernel.operations.registry import (
        execute_operation,
        get_operation_catalog,
        register_operation_mcp,
    )
    from kernel.tools.registry import register_tool_mcp

    register_operation_mcp(catalog_config)
    register_tool_mcp(catalog_config)

    # ── Pack prerequisite validation (bare-vs-none elimination) ────────
    # A pack that cannot satisfy a host-level prerequisite (e.g. bwrap needs
    # the bubblewrap binary) must fail startup LOUDLY rather than be silently
    # skipped at runtime - a "present but cannot run" sandbox would otherwise
    # degrade to unsandboxed host execution without the operator knowing. Packs
    # expose a zero-arg callable via the ``lisa.extensions.prerequisites``
    # entry point returning a list of miss reasons; any miss aborts startup.
    _validate_pack_prerequisites()

    register_operation_callbacks(get_operation_catalog, execute_operation)
    register_content_mcp(catalog_config)

    # Wire the MCPContentRegistry as the content registry for authored context
    # assets and the sub-turn contract assets (fork/delegate briefings + midpoint
    # write-nudge). Both modules share the same registry instance.
    from kernel.context.content_registry import set_content_registry
    from kernel.context.mcp_content_registry import MCPContentRegistry
    from kernel.context.sub_turn_contract import (
        set_content_registry as _set_sub_turn_content_registry,
    )

    registry = MCPContentRegistry(content_config)
    set_content_registry(registry)
    _set_sub_turn_content_registry(registry)

    # Eagerly validate required authored content at startup so a broken
    # content install fails at boot rather than producing a silently-
    # degraded first turn with an empty system prompt.
    from sdk.content import validate_required_content

    validate_required_content(
        registry.read_authored_text,
        ["prompts/system-prompt.md"],
        context="setup_framework startup validation",
    )

    # ── Model manifest pre-load ──────────────────────────────────────
    # Load the model manifest once at startup so mode_models overrides
    # are honored from the first turn and the turn path never does file
    # I/O to fetch it. Moved here from four scattered lazy-init blocks in
    # handle_message.py / _log.py / _render.py that each did
    # ``if runtime.session.manifest is None: runtime.session.manifest = load_manifest(...)``.
    # The ``/models reload`` command still reassigns ``runtime.session.manifest``
    # directly after re-reading from disk.
    #
    # A missing manifest is a valid first-run state: the user must be able
    # to launch the REPL and run ``/models discover`` to generate one, so we
    # tolerate an absent file (``runtime.session.manifest`` stays ``None``) and let
    # the graceful-degradation paths handle it (``model_alias_for_mode`` →
    # ``"default"``, ``build_capability_snapshot`` → empty models, ``/models
    # show`` → "No active models manifest is available yet"). A
    # present-but-invalid manifest still raises so a broken config fails
    # loudly. ``models_path`` is pre-validated in
    # ``HostRuntime._resolve_models_path``, so a missing session-pinned path
    # already raises before we reach this point — only the default
    # ``data_dir/models.yaml`` first-run case lands here.
    from runtime.model_runtime import (
        load_manifest_or_none as _load_manifest_or_none,
    )

    runtime.session.manifest = _load_manifest_or_none(
        data_dir=runtime.paths.data_dir, path=runtime.paths.models_path
    )


def configure_runtime(runtime: HostRuntime) -> None:
    """Reset cached provider clients on the runtime."""
    runtime.providers.azure_client = None
    runtime.providers.openai_client = None
    runtime.providers.anthropic_http_client = None
    runtime.providers.ollama_client = None
    runtime.providers.openrouter_client = None
    runtime.providers.google_client = None


# ── MCP tool server compilation ──────────────────────────────────────


def build_mcp_tool_servers(runtime: HostRuntime) -> list[Any]:
    """Build the approved MCP server registry and compile it to configs.

    Registers all MCP servers (first-party and external) from
    ``mcp_servers.yaml`` with the launch policy registry, then compiles
    them into ``MCPClientConfig`` entries for ``MCPToolPool``. The
    ``warm`` / ``required`` / ``disabled`` launch policy is declared in
    YAML — no hardcoded server list or warm_ids set remains.

    Memory is an MCP feature: ``lisa-memory`` serves the three core
    memory calls, Mem0 (or any other HTTP/stdio memory server) serves the
    same three plus its full surface, and the generic ``memory.*``
    synthetic aliases route to the default backend via the pool. No
    in-process memory seam remains.

    All first-party servers use ``VENV_BIN`` runtime — they run as native
    console_scripts entry points (``.venv/Scripts/lisa-*.exe``) with zero
    subprocess overhead. No ``uv`` resolution needed.

    Warm vs cold: servers marked ``warm: true`` in the manifest are
    spawned eagerly at pool ``start()`` and kept alive across turns (with
    bounded retry and event-driven restart). Servers with ``warm: false``
    (default) are cold — they are **not** connected at start; they are
    lazily connected on first ``list_tools`` or ``call_tool`` that needs
    them.
    """
    warm_configs, cold_configs = compile_mcp_configs(runtime)
    return [*warm_configs, *cold_configs]


def _mcp_environment(runtime: HostRuntime, workspace_root: str) -> dict[str, str | None]:
    """Build the MCP manifest-expansion environment for subprocess servers.

    Make FRAMEWORK_ROOT available for ${FRAMEWORK_ROOT} expansion in
    the manifest's path-like fields (venv_dir, command_path, etc.).
    Use unconditional assignment, not ``setdefault``: a parent shell
    may export these as empty strings (e.g. ``LISA_USER_SLUG=""``),
    and ``setdefault`` treats a present-but-empty value as already
    set — silently dropping the host's resolved value. The empty
    value then expands to ``""`` in the manifest, sending subprocesses
    to the wrong path (the 2026-08-25 lisa-thread empty-read incident:
    ``LISA_USER_SLUG=""`` made the thread server read
    ``<data_dir>/threads`` instead of ``<data_dir>/threads/<slug>``
    and return ``[]``). The host is the authoritative source for these
    values once it has resolved them from CLI flags and config.

    Returns the exported values as a mapping rather than writing them to
    ``os.environ``. The caller scopes them to the manifest-load and
    compile window (see :func:`_exported_mcp_environment`) so the
    process environment is unchanged afterwards.

    A ``None`` value means the variable must be ABSENT inside the scope
    — the host resolved no value for it, so a stale parent-shell value
    must not be inherited by the child. ``LISA_THREAD_REDACTION_PATTERNS``
    is the one such case: with no configured patterns the server falls
    back to the framework defaults, and a stale shell value would
    silently replace them.
    """
    config = runtime.config
    exported: dict[str, str | None] = {
        "FRAMEWORK_ROOT": workspace_root,
        "LISA_TIMEZONE": config.timezone,
    }
    if not config.no_thread:
        exported["LISA_THREAD_DIR"] = str(config.resolved_thread_dir)
    if config.user_slug:
        exported["LISA_USER_SLUG"] = config.user_slug
    # Thread-read redaction: the same env contract ``runtime.config`` resolves
    # for the host's in-process excerpt/manifest/search reads. The read-only
    # lisa-thread subprocess applies the same policy to ``thread_read_range``
    # / ``thread_read_recent`` / ``thread_search`` so the model cannot pull
    # prior harness-authored budget/cap prose back into context verbatim
    # through deep reads (the 2026-08-30 budget-claim relapse: the model
    # called ``thread_read_range`` and re-read its own unredacted cap-fire
    # claims). ``enabled=0`` or ``patterns=none`` tells the server to build
    # an unredacted store; unset patterns fall back to the framework
    # defaults inside the server. Unconditional assignment (never
    # ``setdefault``) per the parent-shell pollution note above.
    exported["LISA_THREAD_REDACTION_ENABLED"] = "1" if config.thread_redaction_enabled else "0"
    if config.thread_redaction_enabled and config.thread_redaction_patterns:
        exported["LISA_THREAD_REDACTION_PATTERNS"] = "\n".join(config.thread_redaction_patterns)
    else:
        exported["LISA_THREAD_REDACTION_PATTERNS"] = None
    return exported


@contextmanager
def _exported_mcp_environment(runtime: HostRuntime, workspace_root: str) -> Iterator[None]:
    """Scope the MCP manifest-expansion environment to the compile window.

    The exported values are needed in two places, and only inside this
    window:

    1. Manifest load — ``load_mcp_server_manifest`` expands
       ``${FRAMEWORK_ROOT}`` / ``${LISA_THREAD_DIR}`` /
       ``${LISA_USER_SLUG}`` into ``cli_args`` and ``venv_dir``.
    2. Compile — ``compile_env`` snapshots ``os.environ`` into the
       spawned server's environment, and the stdio transport replaces the
       child's environment with that snapshot.

    Writing them to the ambient environment and leaving them there made
    the host process permanently carry the values it resolved at startup
    (Lisa-Harness/lisa#3). Any later ``HostConfig.from_env()`` in the
    same process then resolved ``thread_dir``, ``user_slug``,
    ``timezone`` and thread-redaction from the stale startup values
    instead of its own inputs — an order-dependent leak across the
    test suite and across any long-lived host that rebuilds a runtime.

    This context manager writes the values for the duration of the
    window and restores the previous state on exit: a var that was
    absent before is removed again, and a var that was present is
    restored to its prior value. The export therefore stays invisible
    to everything outside the window.
    """
    exported = _mcp_environment(runtime, workspace_root)
    previous: dict[str, str | None] = {name: os.environ.get(name) for name in exported}
    for name, value in exported.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    try:
        yield
    finally:
        for name, prior in previous.items():
            if prior is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = prior


def _resolve_session_capability_preferences(
    runtime: HostRuntime,
) -> tuple[set[str], dict[str, str]]:
    """Resolve session-plan server exclusions and capability preferences.

    The immutable plan can narrow installed resources and select an
    explicit owner, but cannot alter manifest launch/security policy.

    Returns:
        A ``(session_exclude_servers, selected_capability_providers)``
        tuple: the lowercased server ids to exclude, and the explicit
        capability → provider-owner mapping (empty when the session plan
        declares no capability preferences).
    """
    plan_resources = runtime.session_plan.manifest.resources
    plan_providers = runtime.session_plan.manifest.providers
    session_exclude_servers: set[str] = set()
    for entry in plan_resources.servers.exclude:
        normalized = entry.strip().lower()
        if normalized:
            session_exclude_servers.add(normalized)
    session_cap_preferences: dict[str, list[str]] = {}
    for cap, pack in plan_providers.capabilities.items():
        session_cap_preferences.setdefault(cap, []).append(pack)
    # Capability ownership is explicit session policy, not host preference.
    selected_capability_providers: dict[str, str] = {}
    if session_cap_preferences:
        selected_capability_providers = {
            capability: providers[0]
            for capability, providers in session_cap_preferences.items()
            if providers
        }
    return session_exclude_servers, selected_capability_providers


def _register_host_and_user_servers(
    registry: MCPServerRegistry,
    host_manifest_path: Path,
    user_manifest_path: Path | None,
) -> Any:
    """Register first-party and user-owned servers onto the registry.

    Framework-owned (first-party) servers from the in-repo mcp_servers.yaml
    keep bare server_ids and cannot be redefined by the user manifest.
    The manifest loader honors ``disabled: true`` by skipping entries, and
    parses ``warm`` / ``required`` onto the spec. The compiler passes
    them through to MCPClientConfig. The framework manifest no longer
    carries a ``memory:`` section or memory server entries; memory
    backends + memory tool servers are contributed by the memory pack
    (see sdk/packs/memory/mcp_servers.yaml) and merged by the pack pass.

    User-owned servers from <data_dir>/mcp_servers.yaml (optional) keep
    bare server_ids (same as first-party), but collide loudly with any
    framework server_id — a user cannot override lisa-storage, only add
    new servers. Malformed user manifests are logged and skipped (fail
    soft on user config, same as pack manifests).

    Returns:
        The (possibly None-bearing) memory config from the host manifest.
    """
    specs, memory_config = load_mcp_server_manifest(host_manifest_path)
    for spec in specs:
        registry.register(spec)

    if user_manifest_path is not None and user_manifest_path.exists():
        try:
            user_specs, user_memory = load_mcp_server_manifest(user_manifest_path)
        except MCPServerManifestError:
            logger.warning(
                "Skipping invalid user MCP manifest at %s",
                user_manifest_path,
                exc_info=True,
            )
            user_specs = []
            user_memory = None
        for spec in user_specs:
            registry.register(spec)
        if user_memory is not None and (user_memory.default_backend or user_memory.backends):
            logger.warning(
                "User MCP manifest at %s declares a memory section; "
                "memory backends are pack-owned and ignored from the user manifest.",
                user_manifest_path,
            )
    elif user_manifest_path is not None and not user_manifest_path.exists():
        logger.debug(
            "User MCP manifest path %s does not exist; skipping user servers.",
            user_manifest_path,
        )
    return memory_config


def _register_pack_servers(
    registry: MCPServerRegistry,
    roots: list[Path],
    excluded_pack_servers: set[str],
    memory_candidates: list[MemoryProviderCandidate],
) -> None:
    """Register pack-contributed servers (namespaced) onto the registry.

    Compile-time namespacing (``<pack_slug>.<server_id>``) allows multiple
    packs to declare the same server_id without collision in the global
    registry. A pack may also contribute memory backends + memory tool
    servers; ownership is collected into ``memory_candidates`` and selected
    deterministically by the caller. Malformed pack manifests are logged
    and skipped defensively.
    """
    used_pack_slugs: set[str] = set()
    for root in roots:
        pack_manifest_path = root / "mcp_servers.yaml"
        if not pack_manifest_path.exists():
            continue
        try:
            pack_specs, pack_memory = load_mcp_server_manifest(pack_manifest_path)
        except MCPServerManifestError:
            logger.warning(
                "Skipping invalid pack MCP manifest at %s",
                pack_manifest_path,
                exc_info=True,
            )
            continue

        pack_slug = unique_pack_slug_for_root(root, used_pack_slugs)
        pack_package_name = infer_pack_package_name(root)

        # A pack may contribute memory backends + memory tool servers. Ownership
        # is selected deterministically from host policy + lock state.
        if pack_memory.default_backend or pack_memory.backends:
            memory_candidates.append(
                MemoryProviderCandidate(
                    pack_slug=pack_slug,
                    package_name=pack_package_name,
                    manifest_path=pack_manifest_path,
                    memory_config=pack_memory,
                )
            )

        for spec in pack_specs:
            if pack_server_excluded(pack_slug, spec.server_id, excluded_pack_servers):
                logger.info(
                    "Pack MCP server excluded by policy: %s.%s",
                    pack_slug,
                    spec.server_id,
                )
                continue

            namespaced = dataclasses.replace(spec, server_id=f"{pack_slug}.{spec.server_id}")
            registry.register(namespaced)


def _select_memory_owner(
    memory_candidates: list[MemoryProviderCandidate],
    selected_capability_providers: dict[str, str],
) -> tuple[Any, str | None]:
    """Select the memory-backend owner pack from host policy + candidates.

    Returns:
        A ``(memory_config, memory_owner_slug)`` tuple. When a candidate
        owns the memory backend, its memory config replaces the host
        manifest's (None-bearing) config and its pack slug is stashed for
        ``_build_turn_hooks``; non-selected candidates are logged and
        ignored. Otherwise the host manifest's config stands and the
        owner slug is ``None``.
    """
    selected_memory_provider = normalize_pack_name(
        str(selected_capability_providers.get("memory") or "")
    )
    selected_memory_owner = select_capability_provider_owner(
        candidates=memory_candidates,
        selected_provider=selected_memory_provider,
    )
    if selected_memory_owner is None:
        return None, None

    memory_config = selected_memory_owner.memory_config
    memory_owner_slug = selected_memory_owner.pack_slug or None
    selected_manifest = str(selected_memory_owner.manifest_path)
    selected_package = selected_memory_owner.package_name
    logger.info(
        "Selected memory provider owner: package=%s slug=%s manifest=%s",
        selected_package,
        memory_owner_slug or "(none)",
        selected_manifest,
    )
    for candidate in memory_candidates:
        if candidate is selected_memory_owner:
            continue
        logger.info(
            "Ignoring memory manifest from package=%s slug=%s at %s; selected owner is %s.",
            candidate.package_name,
            candidate.pack_slug,
            candidate.manifest_path,
            selected_package,
        )
    return memory_config, memory_owner_slug


def _with_default_mcp_log_paths(
    compiled: list[MCPClientConfig], registry: MCPServerRegistry, data_dir: str
) -> list[MCPClientConfig]:
    """Set host-owned log paths for spec-driven servers without overriding explicit values."""
    specs = {spec.server_id: spec for spec in registry.all_specs()}
    configured: list[MCPClientConfig] = []
    for config in compiled:
        spec = specs[config.server_id]
        env = dict(config.env or {})
        if config.runtime_kind != "in_process" and not env.get("LISA_MCP_LOG_PATH"):
            server_data_dir = Path(data_dir)
            if spec.data_dir_subdir:
                server_data_dir /= spec.data_dir_subdir
            env["LISA_MCP_LOG_PATH"] = str(
                server_data_dir / LOGS_DIR / MCP_LOGS_SUBDIR / f"{config.server_id}.jsonl"
            )
        configured.append(dataclasses.replace(config, env=env))
    return configured


def _rewrite_in_process_configs(
    runtime: HostRuntime,
    compiled: list[MCPClientConfig],
) -> list[MCPClientConfig]:
    """Rewrite first-party VENV_BIN configs to IN_PROCESS when enabled.

    When in-process mode is active, rewrite first-party VENV_BIN configs
    to IN_PROCESS so the pool connects via memory transport instead of
    spawning subprocesses. Only first-party servers (registered in
    ``mcp_server.registry``) are rewritten; pack-contributed and
    third-party servers keep their original runtime kind.

    ``args`` MUST be preserved: the pool's in-process arg reconstruction
    helpers (``_in_process_data_dir`` et al.) parse the standard CLI
    contract flags out of ``args``. Wiping them drops ``--data-dir`` /
    ``--deny-glob`` / ``--readonly``, so the server's ``_init`` receives
    ``data_dir=""`` and builds a cwd-relative sandbox (the 2026-09-04
    $HOME/files sandbox-escape: lisa-storage wrote the research doc the
    model asked for into the host process's cwd instead of the data dir).
    """
    if not getattr(runtime, "_in_process_mcp", False):
        return compiled
    from mcp_server.registry import in_process_server_ids

    first_party_ids = set(in_process_server_ids())
    rewritten: list[MCPClientConfig] = []
    for cfg in compiled:
        if cfg.server_id in first_party_ids and cfg.runtime_kind == "venv_bin":
            rewritten.append(dataclasses.replace(cfg, runtime_kind="in_process"))
        else:
            rewritten.append(cfg)
    return rewritten


def compile_mcp_configs(
    runtime: HostRuntime,
    *,
    discovery: _FrameworkDiscovery | None = None,
) -> tuple[list[Any], list[Any]]:
    """Compile all MCP servers from mcp_servers.yaml and split into (warm, cold).

    All servers (first-party and external) are declared in
    ``mcp_servers.yaml`` with their ``warm`` / ``required`` / ``disabled``
    launch policy. The manifest loader produces ``MCPServerSpec`` entries
    that carry these fields; the compiler passes them through to
    ``MCPClientConfig``. No hardcoded server list or ``warm_ids`` set
    remains — adding, removing, disabling, or changing a server's launch
    policy is a YAML edit.

    The manifest's memory-backend designation (which server_id backs the
    generic ``memory.*`` aliases) is stashed on the runtime for
    ``_build_turn_hooks`` to consume.

    When ``discovery`` is provided (computed once by
    :func:`discover_framework`), the shared roots / lock / manifest path
    are reused instead of recomputed. When ``None``, this function calls
    :func:`discover_framework` internally (back-compat for standalone
    callers like tests).
    """
    data_dir = str(runtime.paths.data_dir)
    workspace_root = str(FRAMEWORK_ROOT)
    if discovery is None:
        discovery = discover_framework(runtime, seed_lock=False)
    host_manifest_path = discovery.host_manifest_path
    roots = discovery.roots
    user_manifest_path = discovery.user_manifest_path

    # ── Session-plan resource preferences ──────────────────────────
    # The immutable plan can narrow installed resources and select an
    # explicit owner, but cannot alter manifest launch/security policy.
    session_exclude_servers, session_cap_preferences = _resolve_session_capability_preferences(
        runtime
    )
    selected_capability_providers = dict(runtime.session_plan.manifest.providers.capabilities)
    if session_cap_preferences:
        selected_capability_providers = session_cap_preferences

    registry = MCPServerRegistry()

    # The exported values are needed for manifest expansion and for the
    # compiled child environment; both happen inside this scope, and the
    # ambient environment is restored on exit.
    with _exported_mcp_environment(runtime, workspace_root):
        # Framework-owned (first-party) + user-owned servers.
        memory_config = _register_host_and_user_servers(
            registry, host_manifest_path, user_manifest_path
        )

        excluded_pack_servers = load_pack_server_exclusions(host_manifest_path)
        if user_manifest_path is not None and user_manifest_path.exists():
            excluded_pack_servers |= load_pack_server_exclusions(user_manifest_path)
        excluded_pack_servers |= session_exclude_servers

        memory_candidates: list[MemoryProviderCandidate] = []
        _register_pack_servers(registry, roots, excluded_pack_servers, memory_candidates)

        selected_config, memory_owner_slug = _select_memory_owner(
            memory_candidates, selected_capability_providers
        )
        if selected_config is not None:
            memory_config = selected_config

        runtime.services.memory_config = memory_config
        runtime.services.memory_config_pack_slug = memory_owner_slug

        compiled = compile_registry(registry, data_dir=data_dir, workspace_root=workspace_root)
        compiled = _with_default_mcp_log_paths(compiled, registry, data_dir)
        compiled = _rewrite_in_process_configs(runtime, compiled)

    warm = [c for c in compiled if c.warm]
    cold = [c for c in compiled if not c.warm]
    return warm, cold


def compile_proxy_configs(  # noqa: C901
    runtime: HostRuntime,
    *,
    discovery: _FrameworkDiscovery | None = None,
) -> list[ProxyClientConfig]:
    """Compile provider proxies declared by the host, user, and active packs.

    Proxy declarations share ``mcp_servers.yaml`` with MCP declarations, but
    they are a separate resource kind: they have no model-facing tools and are
    consumed only by provider client construction after readiness succeeds.
    """
    if discovery is None:
        discovery = discover_framework(runtime, seed_lock=False)
    workspace_root = str(FRAMEWORK_ROOT)
    registry = ProxyServerRegistry()
    plan_proxies = runtime.session_plan.manifest.resources.proxies
    included = {entry.strip().lower() for entry in plan_proxies.include if entry.strip()}
    excluded = {entry.strip().lower() for entry in plan_proxies.exclude if entry.strip()}
    required = {entry.strip().lower() for entry in plan_proxies.required if entry.strip()}

    def add(spec: Any, *, proxy_id: str) -> None:
        normalized = proxy_id.lower()
        declared = spec.proxy_id.lower()
        selected = not included or normalized in included or declared in included
        if not selected and normalized not in required and declared not in required:
            logger.info("Proxy not included by session policy: %s", proxy_id)
            return
        if normalized in excluded or declared in excluded:
            logger.info("Proxy excluded by session policy: %s", proxy_id)
            return
        registry.register(
            dataclasses.replace(
                spec,
                proxy_id=proxy_id,
                required=(spec.required or normalized in required or declared in required),
            )
        )

    def load(path: Path, *, source: str) -> list[Any]:
        try:
            return load_proxy_server_manifest(path)
        except ProxyServerManifestError:
            logger.warning("Skipping invalid %s proxy manifest at %s", source, path, exc_info=True)
            return []

    with _exported_mcp_environment(runtime, workspace_root):
        # FRAMEWORK_ROOT is needed for ${FRAMEWORK_ROOT} expansion in the
        # proxy manifests and for the compiled proxy environment; scope it
        # to the load/compile window so the ambient environment is left
        # unchanged.
        for spec in load(discovery.host_manifest_path, source="host"):
            add(spec, proxy_id=spec.proxy_id)
        if discovery.user_manifest_path is not None and discovery.user_manifest_path.exists():
            for spec in load(discovery.user_manifest_path, source="user"):
                add(spec, proxy_id=spec.proxy_id)

        used_pack_slugs: set[str] = set()
        for root in discovery.roots:
            manifest_path = root / "mcp_servers.yaml"
            if not manifest_path.exists():
                continue
            pack_slug = unique_pack_slug_for_root(root, used_pack_slugs)
            for spec in load(manifest_path, source="pack"):
                add(spec, proxy_id=f"{pack_slug}.{spec.proxy_id}")

        providers = sorted(
            {provider for spec in registry.all_specs() for provider in spec.providers}
        )
        provider_base_urls = direct_provider_base_urls(runtime, providers)

        return compile_proxy_registry(
            registry,
            workspace_root=workspace_root,
            provider_base_urls=provider_base_urls,
        )


# ── Pack slug + policy helpers ───────────────────────────────────────


def unique_pack_slug_for_root(root: Path, used_slugs: set[str]) -> str:
    """Return a stable, filesystem-safe slug for one content-pack root."""
    raw = root.name.strip().lower()
    slug = re.sub(r"[^a-z0-9_-]+", "-", raw).strip("-") or "pack"
    if slug not in used_slugs:
        used_slugs.add(slug)
        return slug

    suffix = 2
    while f"{slug}-{suffix}" in used_slugs:
        suffix += 1
    unique = f"{slug}-{suffix}"
    used_slugs.add(unique)
    return unique


def load_pack_server_exclusions(host_manifest_path: Path) -> set[str]:
    """Load host-side pack server exclusions from ``pack_policy.exclude_servers``.

    Exclusion entries are normalized lowercase strings and may be expressed as:
    - ``pack_slug.server_id`` (preferred)
    - ``pack_slug:server_id``
    - ``pack_slug/server_id``
    - ``pack_slug.*`` (exclude all servers from one pack)
    """
    policy = load_pack_policy_dict(host_manifest_path, warning_subject="pack exclusion policy")
    entries = policy.get("exclude_servers") or []
    if not isinstance(entries, list):
        logger.warning("pack_policy.exclude_servers must be a list in %s", host_manifest_path)
        return set()

    exclusions: set[str] = set()
    for item in entries:
        if not isinstance(item, str):
            continue
        normalized = item.strip().lower()
        if normalized:
            exclusions.add(normalized)
    return exclusions


@dataclass(frozen=True, slots=True)
class MemoryProviderCandidate:
    """One pack's memory-backend contribution + ownership metadata.

    Frozen-slots dataclass so the four keys (``pack_slug``, ``package_name``,
    ``manifest_path``, ``memory_config``) are accessed as attributes, not
    string-keyed dict lookups. A typo in a field name is an
    ``AttributeError`` at the call site, not a silent ``None`` from
    ``dict.get("…")``. Mirrors the ``PackProviderGatewayEntry`` /
    ``PackSliceProviderEntry`` pattern in the kernel.
    """

    pack_slug: str
    package_name: str
    manifest_path: Path
    memory_config: MemoryBackendConfig


def provider_identifiers_for_pack(*, package_name: str, pack_slug: str) -> set[str]:
    """Return the set of normalized identifiers that refer to one pack."""
    identifiers = {
        normalize_pack_name(package_name),
        normalize_pack_name(pack_slug),
    }
    normalized_package = normalize_pack_name(package_name)
    if normalized_package.startswith("lisa-pack-"):
        identifiers.add(normalized_package[len("lisa-pack-") :])
    return {identifier for identifier in identifiers if identifier}


def select_capability_provider_owner(
    *,
    candidates: list[MemoryProviderCandidate],
    selected_provider: str,
) -> MemoryProviderCandidate | None:
    """Select a capability owner without any host or lock preference fallback."""
    if not candidates:
        return None

    selected = normalize_pack_name(selected_provider)
    if selected:
        matching = [
            candidate
            for candidate in candidates
            if selected
            in provider_identifiers_for_pack(
                package_name=candidate.package_name,
                pack_slug=candidate.pack_slug,
            )
        ]
        if len(matching) == 1:
            return matching[0]
        if not matching:
            return None
        raise ValueError(
            f"Multiple active capability providers match explicit selection '{selected}'."
        )

    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        "Multiple active capability providers require an explicit providers.capabilities selection."
    )


def pack_server_excluded(pack_slug: str, declared_server_id: str, exclusions: set[str]) -> bool:
    """Return True when host policy excludes a pack-contributed server."""
    slug = pack_slug.strip().lower()
    server = declared_server_id.strip().lower()
    if not slug or not server or not exclusions:
        return False

    dotted = f"{slug}.{server}"
    return (
        dotted in exclusions
        or f"{slug}:{server}" in exclusions
        or f"{slug}/{server}" in exclusions
        or f"{slug}.*" in exclusions
    )


# ── Internal helpers ─────────────────────────────────────────────────


def _resolve_framework_manifest_path(config: HostConfig) -> Path:
    """Return the framework (in-repo) MCP manifest path.

    This is always ``host/mcp_servers.yaml`` — the fixed base layer of
    first-party servers (lisa-catalog, lisa-storage, etc.) that users
    cannot redefine. The ``config.mcp_manifest_path`` override does **not**
    apply here; it controls the user manifest (see
    :func:`_resolve_user_manifest_path`).
    """
    return host_mcp_manifest_path()


def _resolve_user_manifest_path(config: HostConfig) -> Path | None:
    """Return the user-editable MCP manifest path, or ``None`` if absent.

    Resolution order:

    1. ``config.mcp_manifest_path`` — explicit CLI/env override. When set,
       that path is used even if it does not exist (the user asked for it).
    2. ``\u003cdata_dir\u003e/mcp_servers.yaml`` — the default user manifest
       location. Returns ``None`` when the file does not exist (graceful
       first-run, same convention as ``models.yaml``).

    The framework manifest is always loaded separately by
    :func:`_resolve_framework_manifest_path`; this function only resolves
    the user layer.
    """
    if config.mcp_manifest_path is not None:
        return config.mcp_manifest_path
    from sdk.layout import CONFIG_DIR

    default_user_path = config.data_dir / CONFIG_DIR / "mcp_servers.yaml"
    if default_user_path.exists():
        return default_user_path
    return None


def register_mcp_aliases(pool: Any) -> None:
    """Register generic aliases from compiled MCP configs as synthetic tools.

    After ``pool.start()``, walk every config's ``aliases`` tuple and register
    each as a synthetic tool via ``mcp_alias_tool``. The alias gives the model
    a stable name (e.g. ``cloudflare.search``) instead of the namespaced
    server id (e.g. ``lisa_packs.cloudflare.cloudflare-api.search``).

    Must be called after ``pool.start()`` so the alias handlers can route to
    live server sessions.
    """
    from kernel.tools.synthetic import mcp_alias_tool

    configs = list(getattr(pool, "configs", []))
    for config in configs:
        aliases = getattr(config, "aliases", ()) or ()
        if not aliases:
            continue
        server_id = config.server_id
        for alias in aliases:
            alias_name = alias.alias_name
            native_tool = alias.native_tool
            description = alias.description or f"Alias for {server_id}.{native_tool}"
            spec = mcp_alias_tool(
                name=alias_name,
                description=description,
                pool=pool,
                target_tool=native_tool,
                target_server_id=server_id,
            )
            pool.add_synthetic_tools([spec])


__all__ = [
    "MemoryProviderCandidate",
    "build_catalog_mcp_config",
    "build_content_mcp_config",
    "build_mcp_tool_servers",
    "compile_mcp_configs",
    "configure_runtime",
    "load_pack_server_exclusions",
    "pack_server_excluded",
    "provider_identifiers_for_pack",
    "register_mcp_aliases",
    "select_capability_provider_owner",
    "setup_framework",
    "unique_pack_slug_for_root",
]

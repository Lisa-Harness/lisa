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
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kernel.extension_names import normalize_pack_name
from kernel.mcp_launch.compiler import compile_registry
from kernel.mcp_launch.policy import MCPServerRegistry, MCPServerSpec
from kernel.mcp_launch.server_manifest import (
    MCPServerManifestError,
    load_mcp_server_manifest,
)
from kernel.proxy_launch.compiler import ProxyClientConfig, compile_proxy_registry
from kernel.proxy_launch.manifest import ProxyServerManifestError, load_proxy_server_manifest
from kernel.proxy_launch.policy import ALL_PROVIDERS, ProxyServerRegistry, ProxyServerSpec
from kernel.session_plan import ProviderSelections
from kernel.tools.mcp_client import MCPClientConfig
from sdk.layout import CACHE_DIR, LOGS_DIR, MCP_LOGS_SUBDIR

from runtime.capability_servers import (
    CapabilityCandidate,
    CapabilitySelectionError,
    candidate_identifiers,
    log_capability_selection,
    register_capability_server,
    reject_unclaimed_selections,
    select_capability_owner,
    valid_capability_name,
)
from runtime.config import HostConfig
from runtime.content_roots import (
    always_on_pack_names,
    host_mcp_manifest_path,
    infer_pack_package_name,
    installed_pack_requested_revision,
    installed_pack_version,
    load_disabled_pack_names,
    load_installed_packs_lock,
    load_pack_policy_dict,
    sdk_content_roots,
    seed_bundled_packs_lock,
)
from runtime.provider_endpoints import direct_provider_base_urls
from runtime.runtime import HostRuntime

_MAX_TCP_PORT = 65_535

logger = logging.getLogger(__name__)

#: Pack versions that map onto a ``vX.Y.Z`` binary release tag.
_PACK_RELEASE_VERSION_RE = re.compile(r"\d+\.\d+\.\d+")
#: VCS refs that name a ``vX.Y.Z`` binary release tag.
_PACK_RELEASE_TAG_RE = re.compile(r"v?(\d+\.\d+\.\d+)")


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
class _PackManifestSpecs:
    """One pack root's parsed ``mcp_servers.yaml`` server specs.

    Parsed once by :func:`discover_framework` so the ambient-tool-id
    resolver and the server registrar share a single parse per manifest.
    ``slug`` is the compile-time namespacing prefix, derived once in
    roots order so every consumer namespaces identically.
    """

    root: Path
    slug: str
    manifest_path: Path
    specs: list[MCPServerSpec]


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

    The parsed server specs for every layer are computed here once and
    carried on ``host_specs`` / ``user_specs`` / ``pack_specs`` so
    :func:`resolve_default_mcp_tool_ids` and the compile-time server
    registration consume one parse per manifest instead of each
    re-reading every layer. A malformed host manifest raises here (the
    same startup failure the registrar previously produced); malformed
    user or pack manifests are warned and skipped, matching the
    registrar's fail-soft behavior.
    """

    roots: list[Path]
    disabled_packs: set[str]
    lock: dict[str, Any]
    host_manifest_path: Path
    user_manifest_path: Path | None = None
    host_specs: list[MCPServerSpec] = dataclasses.field(default_factory=list)
    user_specs: list[MCPServerSpec] = dataclasses.field(default_factory=list)
    pack_specs: list[_PackManifestSpecs] = dataclasses.field(default_factory=list)


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

    # Parse every manifest layer once here; both consumers (ambient
    # tool-id resolution in setup_framework, server registration in
    # compile_mcp_configs) reuse these specs instead of re-reading files.
    # ``${FRAMEWORK_ROOT}`` / ``${LISA_USER_SLUG}`` / ``${LISA_THREAD_DIR}``
    # expand against the explicitly resolved MCP environment, never the
    # ambient shell — a stale or empty parent value must not win (the
    # 2026-08-25 empty-read incident shape).
    resolved_env = _resolved_mcp_env(runtime, str(FRAMEWORK_ROOT))
    host_specs = load_mcp_server_manifest(manifest_path, env=resolved_env)
    user_specs: list[MCPServerSpec] = []
    if user_manifest_path is not None and user_manifest_path.exists():
        try:
            user_specs = load_mcp_server_manifest(user_manifest_path, env=resolved_env)
        except MCPServerManifestError:
            logger.warning(
                "Skipping invalid user MCP manifest at %s",
                user_manifest_path,
                exc_info=True,
            )
    elif user_manifest_path is not None:
        logger.debug(
            "User MCP manifest path %s does not exist; skipping user servers.",
            user_manifest_path,
        )
    pack_specs: list[_PackManifestSpecs] = []
    used_pack_slugs: set[str] = set()
    for root in roots:
        pack_manifest_path = root / "mcp_servers.yaml"
        if not pack_manifest_path.exists():
            continue
        try:
            pack_entry_specs = load_mcp_server_manifest(pack_manifest_path, env=resolved_env)
        except MCPServerManifestError:
            logger.warning(
                "Skipping invalid pack MCP manifest at %s",
                pack_manifest_path,
                exc_info=True,
            )
            continue
        pack_specs.append(
            _PackManifestSpecs(
                root=root,
                slug=unique_pack_slug_for_root(root, used_pack_slugs),
                manifest_path=pack_manifest_path,
                specs=pack_entry_specs,
            )
        )
    return _FrameworkDiscovery(
        roots=roots,
        disabled_packs=disabled_packs,
        lock=lock,
        host_manifest_path=manifest_path,
        user_manifest_path=user_manifest_path,
        host_specs=host_specs,
        user_specs=user_specs,
        pack_specs=pack_specs,
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
    """Collect published ambient tool ids from one manifest's specs.

    Args:
        specs: Parsed ``MCPServerSpec`` entries from one manifest.
        server_id_prefix: Optional pack slug applied at compile time
            (``<pack_slug>.<server_id>``); empty for first-party/user
            manifests which keep bare server ids.

    Returns:
        The set of published ambient tool ids. For capability-claiming
        servers the prefix is the capability (``<capability>.<tool>``) —
        the id the live catalog publishes regardless of which
        implementation won selection — and the listed bare names are
        interpreted as *published* names (post-tool_map). Ordinary
        servers keep ``server_id.tool`` (pack servers:
        ``<pack_slug>.<server_id>.<tool>``).
    """
    from kernel.tools.mcp_client import namespaced_tool_name

    ids: set[str] = set()
    for spec in specs:
        if spec.disabled or not spec.default_available_tools:
            continue
        capability = getattr(spec, "capability", None)
        if capability:
            # Capability servers publish under the capability prefix — the
            # same id regardless of which implementation is selected, so
            # collecting from losing declarations is harmless (identical ids).
            for tool_name in spec.default_available_tools:
                ids.add(namespaced_tool_name(capability, tool_name))
            continue
        namespaced_server_id = (
            f"{server_id_prefix}.{spec.server_id}" if server_id_prefix else spec.server_id
        )
        for tool_name in spec.default_available_tools:
            ids.add(namespaced_tool_name(namespaced_server_id, tool_name))
    return ids


def resolve_default_mcp_tool_ids(
    *,
    host_specs: list[MCPServerSpec],
    pack_specs: list[_PackManifestSpecs],
    user_specs: list[MCPServerSpec],
) -> set[str]:
    """Resolve the namespaced ambient MCP-native tool id set.

    Consumes the manifest specs parsed once by :func:`discover_framework`
    and collects ``default_available_tools`` entries from each spec,
    namespacing them as ``server_id.tool``. Pack-contributed servers are
    registered under namespaced ids (``<pack_slug>.<server_id>``) at
    compile time, so the ambient id the model sees is the same namespaced
    form the catalog merge and the turn-runner filter use; the slug on
    each parsed pack entry is derived once at discovery, identically for
    every consumer.

    Host and user servers keep bare server_ids — user-added servers are
    not pack-prefixed; their ``default_available_tools`` become ambient
    ids the same way lisa-storage.read_file does.
    """
    ids: set[str] = set()
    # Host manifest first (first-party servers keep their bare server_id).
    ids |= _collect_default_tool_ids(host_specs)
    # User manifest (bare server_id, same as first-party).
    ids |= _collect_default_tool_ids(user_specs)
    # Pack manifests (namespaced as <pack_slug>.<server_id> at compile time).
    for entry in pack_specs:
        ids |= _collect_default_tool_ids(entry.specs, server_id_prefix=entry.slug)
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
    / parsed manifest specs are reused instead of recomputed. When
    ``None``, this function calls :func:`discover_framework` internally
    (back-compat for standalone callers like tests).

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
        host_specs=discovery.host_specs,
        pack_specs=discovery.pack_specs,
        user_specs=discovery.user_specs,
    )
    catalog_config = build_catalog_mcp_config(content_config=content_config, data_dir=data_dir)
    if default_mcp_tool_ids:
        # build_catalog_mcp_config always returns a populated env dict, but
        # MCPClientConfig.env is typed dict|None, so normalize defensively
        # before mutating (the empty-default case is skipped above).
        if catalog_config.env is None:
            catalog_config = dataclasses.replace(catalog_config, env={})
        catalog_config.env["LISA_DEFAULT_MCP_TOOLS"] = ",".join(sorted(default_mcp_tool_ids))

    # Start the shared lisa-catalog and lisa-content subprocesses together so
    # their interpreter startups overlap; every catalog/content adapter below
    # reuses these two long-lived sessions instead of spawning per call.
    from kernel.context.mcp_catalog import MCPOperationCatalog
    from kernel.context.mcp_content_registry import MCPContentRegistry

    registry = MCPContentRegistry(content_config)
    registry.prewarm()
    MCPOperationCatalog(catalog_config).prewarm()

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
    from kernel.context.sub_turn_contract import (
        set_content_registry as _set_sub_turn_content_registry,
    )

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

    Returns the resolved mapping rather than writing it to
    ``os.environ``; consumers pass it explicitly to the manifest loaders
    and compilers (see :func:`_resolved_mcp_env`), so the process
    environment is never mutated.

    A ``None`` value means the variable must be ABSENT from the resolved
    environment — the host resolved no value for it, so a stale
    parent-shell value must not be inherited by the child.
    ``LISA_THREAD_REDACTION_PATTERNS`` is the one such case: with no
    configured patterns the server falls back to the framework defaults,
    and a stale shell value would silently replace them.
    """
    config = runtime.config
    exported: dict[str, str | None] = {
        "FRAMEWORK_ROOT": workspace_root,
        "LISA_TIMEZONE": config.timezone,
    }
    if not config.no_thread:
        exported["LISA_THREAD_DIR"] = str(config.resolved_thread_dir)
    if config.thread_slug:
        exported["LISA_USER_SLUG"] = config.thread_slug
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


def _resolved_mcp_env(runtime: HostRuntime, workspace_root: str) -> dict[str, str]:
    """Build the resolved MCP environment for manifest expansion and child envs.

    The manifest loaders expand ``${FRAMEWORK_ROOT}`` /
    ``${LISA_THREAD_DIR}`` / ``${LISA_USER_SLUG}`` against this mapping at
    parse time, and the compilers snapshot it into each spawned server's
    environment. Both consumers receive it explicitly, so the ambient
    ``os.environ`` is never mutated and never consulted for these values.

    The base is the ambient environment (the child inherits the parent
    shell like any subprocess) with the host's resolved overrides
    applied: unconditional assignment, never ``setdefault`` — a parent
    shell may export these as empty strings (e.g. ``LISA_USER_SLUG=""``),
    and ``setdefault`` treats a present-but-empty value as already set,
    silently dropping the host's resolved value (the 2026-08-25
    empty-read incident). A ``None`` override pops the key so a stale
    parent value cannot leak into the child.
    """
    resolved = dict(os.environ)
    for name, value in _mcp_environment(runtime, workspace_root).items():
        if value is None:
            resolved.pop(name, None)
        else:
            resolved[name] = value
    return resolved


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
    *,
    host_manifest_path: Path,
    host_specs: list[MCPServerSpec],
    user_manifest_path: Path | None,
    user_specs: list[MCPServerSpec],
    capability_candidates: list[CapabilityCandidate],
) -> None:
    """Register first-party and user-owned servers onto the registry.

    Specs are pre-parsed by :func:`discover_framework`; malformed user
    manifests were already warned and skipped there. The manifest paths
    are retained for capability-candidate metadata.

    Framework-owned (first-party) servers from the in-repo mcp_servers.yaml
    keep bare server_ids and cannot be redefined by the user manifest.
    The manifest loader honors ``disabled: true`` by skipping entries, and
    parses ``warm`` / ``required`` onto the spec. The compiler passes
    them through to MCPClientConfig.

    Capability-claiming servers compete through the same selection policy,
    regardless of whether they are declared in the host, user, or pack
    manifest.

    Servers that declare ``capability:`` are NOT registered directly —
    they are collected into ``capability_candidates`` (``source="builtin"``
    for the host manifest, ``source="user"`` for the user manifest) and
    resolved to one owner per capability after every layer has declared
    its candidates (:func:`_resolve_capability_owners`).

    User-owned servers from <data_dir>/mcp_servers.yaml (optional) keep
    bare server_ids (same as first-party), but collide loudly with any
    framework server_id — a user cannot override lisa-storage, only add
    new servers.
    """
    for spec in host_specs:
        _declare_server_or_candidate(
            registry,
            spec,
            capability_candidates,
            source="builtin",
            manifest_path=str(host_manifest_path),
        )

    user_manifest_label = str(user_manifest_path) if user_manifest_path is not None else ""
    for spec in user_specs:
        _declare_server_or_candidate(
            registry,
            spec,
            capability_candidates,
            source="user",
            manifest_path=user_manifest_label,
        )


def _declare_server_or_candidate(
    registry: MCPServerRegistry,
    spec: Any,
    capability_candidates: list[CapabilityCandidate],
    *,
    source: str,
    manifest_path: str,
    pack_slug: str | None = None,
    package_name: str | None = None,
) -> None:
    """Register one spec, or collect it as a capability candidate.

    A spec with ``capability`` set competes for that capability; it is
    collected instead of registered, and only the resolved winner is
    registered later under its real server id (see
    :func:`runtime.capability_servers.winning_spec_as_capability`).
    """
    capability = getattr(spec, "capability", None)
    if capability and valid_capability_name(capability):
        capability_candidates.append(
            CapabilityCandidate(
                capability=capability,
                spec=spec,
                source=source,
                pack_slug=pack_slug,
                package_name=package_name,
                manifest_path=manifest_path,
            )
        )
        return
    registry.register(spec)


def _resolve_capability_owners(
    registry: MCPServerRegistry,
    capability_candidates: list[CapabilityCandidate],
    selected_capability_providers: dict[str, str],
    *,
    skipped_candidates: list[_SkippedCandidate] | None = None,
) -> None:
    """Resolve and register one owner per claimed capability.

    Called after every manifest layer (host, user, packs) has declared its
    candidates. For each capability:

    - resolve the winner via
      :func:`runtime.capability_servers.select_capability_owner` (explicit
      session selection > built-in > unique candidate; contested
      capabilities raise);
    - register the winner's spec under its real server id; the MCP pool
      publishes ``<capability>.<tool>`` ids from its declared contract —
      identical model-facing ids regardless of implementation;
    - log (not register) the losers.

    Raises :class:`CapabilitySelectionError` when a capability cannot be
    resolved deterministically — a contested or misconfigured capability,
    or a selection naming a capability nothing claims, is a startup error,
    not a silent fallback. When the selection names a server that was
    skipped before selection, the error says why.
    """
    try:
        _select_capability_owners(registry, capability_candidates, selected_capability_providers)
    except CapabilitySelectionError as exc:
        reasons = _skipped_selection_reasons(
            selected_capability_providers, skipped_candidates or []
        )
        if not reasons:
            raise
        raise CapabilitySelectionError(f"{exc}. {' '.join(reasons)}") from exc


def _select_capability_owners(
    registry: MCPServerRegistry,
    capability_candidates: list[CapabilityCandidate],
    selected_capability_providers: dict[str, str],
) -> None:
    by_capability: dict[str, list[CapabilityCandidate]] = {}
    for candidate in capability_candidates:
        by_capability.setdefault(candidate.capability, []).append(candidate)
    reject_unclaimed_selections(selected_capability_providers, set(by_capability))
    for capability in sorted(by_capability):
        candidates = by_capability[capability]
        contracts = {frozenset(candidate.spec.capability_tools) for candidate in candidates}
        if len(contracts) > 1:
            details = "; ".join(
                f"{candidate.spec.server_id}={sorted(candidate.spec.capability_tools)}"
                for candidate in sorted(candidates, key=lambda item: item.spec.server_id)
            )
            raise CapabilitySelectionError(
                f"Servers claiming capability {capability!r} declare different "
                f"capability_tools contracts: {details}"
            )
        ambient_contracts = {
            frozenset(candidate.spec.default_available_tools) for candidate in candidates
        }
        if len(ambient_contracts) > 1:
            details = "; ".join(
                f"{candidate.spec.server_id}={sorted(candidate.spec.default_available_tools)}"
                for candidate in sorted(candidates, key=lambda item: item.spec.server_id)
            )
            raise CapabilitySelectionError(
                f"Servers claiming capability {capability!r} declare different "
                f"default_available_tools: {details}"
            )
        selection = select_capability_owner(
            capability,
            candidates,
            selected_provider=selected_capability_providers.get(capability, ""),
        )
        log_capability_selection(capability, selection)
        if selection.winner is not None:
            register_capability_server(registry, selection.winner)


@dataclasses.dataclass(frozen=True, slots=True)
class _SkippedCandidate:
    """A capability server dropped before selection, with the reason."""

    candidate: CapabilityCandidate
    reason: str


@dataclasses.dataclass(frozen=True, slots=True)
class _PackBinaryResolution:
    """Pack specs with binary releases resolved, plus the servers dropped."""

    specs: list[Any]
    skipped: list[Any]
    skip_reason: str = ""


def _skipped_selection_reasons(
    selected_capability_providers: dict[str, str],
    skipped_candidates: list[_SkippedCandidate],
) -> list[str]:
    """Explain explicit selections that name a server skipped before selection."""
    reasons: list[str] = []
    for capability, provider in sorted(selected_capability_providers.items()):
        selected = provider.strip().lower()
        if not selected:
            continue
        for skipped in skipped_candidates:
            if skipped.candidate.capability == capability and selected in candidate_identifiers(
                skipped.candidate
            ):
                reasons.append(f"{skipped.candidate.spec.server_id} was skipped: {skipped.reason}.")
    return reasons


def _pack_binary_release(root: Path, package_name: str) -> tuple[str | None, str | None]:
    """Return ``(release_tag, observed_version)`` for a pack's binary servers.

    The tag the pack was installed from (``direct_url.json``) wins: a
    ``git+...@vX.Y.Z`` install names its release directly, independent of
    the version a VCS build backend derives from possibly stale local tags.
    Otherwise a plain ``X.Y.Z`` distribution version maps to ``vX.Y.Z``.
    """
    requested = installed_pack_requested_revision(root, package_name)
    if requested is not None:
        match = _PACK_RELEASE_TAG_RE.fullmatch(requested)
        if match:
            return f"v{match.group(1)}", requested
    pack_version = installed_pack_version(root, package_name)
    if pack_version is not None and _PACK_RELEASE_VERSION_RE.fullmatch(pack_version):
        return f"v{pack_version}", pack_version
    return None, pack_version


def _with_pack_binary_versions(
    specs: list[Any], root: Path, package_name: str
) -> _PackBinaryResolution:
    """Resolve unversioned pack ``binary_download`` sources to the pack's release.

    A pack that omits ``binary_download.version`` downloads the release tagged
    with the release the pack was installed from (see
    :func:`_pack_binary_release`), so that tag is the single version
    selector. Installs that do not name a release cannot resolve that
    selector, so those binary servers are skipped.
    """
    unresolved = [
        spec
        for spec in specs
        if spec.binary_download is not None and spec.binary_download.version is None
    ]
    if not unresolved:
        return _PackBinaryResolution(specs=specs, skipped=[])
    release, observed_version = _pack_binary_release(root, package_name)
    if release is None:
        reason = (
            f"pack {package_name} version {observed_version!r} is not a release; "
            "install it from a vX.Y.Z tag or set binary_download.version"
        )
        logger.warning(
            "Pack %s has version %r, which is not a release; skipping binary MCP servers "
            "without an explicit version: %s",
            package_name,
            observed_version,
            ", ".join(spec.server_id for spec in unresolved),
        )
        return _PackBinaryResolution(
            specs=[spec for spec in specs if spec not in unresolved],
            skipped=unresolved,
            skip_reason=reason,
        )
    return _PackBinaryResolution(
        specs=[
            dataclasses.replace(
                spec,
                binary_download=dataclasses.replace(spec.binary_download, version=release),
            )
            if spec in unresolved
            else spec
            for spec in specs
        ],
        skipped=[],
    )


def _register_pack_servers(
    registry: MCPServerRegistry,
    pack_specs: list[_PackManifestSpecs],
    excluded_pack_servers: set[str],
    capability_candidates: list[CapabilityCandidate],
    skipped_candidates: list[_SkippedCandidate] | None = None,
) -> None:
    """Register pack-contributed servers (namespaced) onto the registry.

    Specs are pre-parsed by :func:`discover_framework` with their slugs
    derived once in roots order; malformed pack manifests were already
    logged and skipped there.

    Compile-time namespacing (``<pack_slug>.<server_id>``) allows multiple
    packs to declare the same server_id without collision in the global
    registry.

    Capability-claiming pack servers are collected into
    ``capability_candidates`` (``source="pack"``) instead of registered;
    the resolved winner retains its real server id for launch and dispatch.
    Capability servers skipped for an unresolvable binary release are
    collected into ``skipped_candidates`` so selection errors can say why.
    """
    for entry in pack_specs:
        pack_slug = entry.slug
        pack_package_name = infer_pack_package_name(entry.root)
        resolution = _with_pack_binary_versions(entry.specs, entry.root, pack_package_name)

        if skipped_candidates is not None:
            for spec in resolution.skipped:
                capability = getattr(spec, "capability", None)
                if (
                    capability
                    and valid_capability_name(capability)
                    and not pack_server_excluded(pack_slug, spec.server_id, excluded_pack_servers)
                ):
                    skipped_candidates.append(
                        _SkippedCandidate(
                            candidate=CapabilityCandidate(
                                capability=capability,
                                spec=spec,
                                source="pack",
                                pack_slug=pack_slug,
                                package_name=pack_package_name,
                                manifest_path=str(entry.manifest_path),
                            ),
                            reason=resolution.skip_reason,
                        )
                    )

        for spec in resolution.specs:
            if pack_server_excluded(pack_slug, spec.server_id, excluded_pack_servers):
                logger.info(
                    "Pack MCP server excluded by policy: %s.%s",
                    pack_slug,
                    spec.server_id,
                )
                continue

            capability = getattr(spec, "capability", None)
            if capability and valid_capability_name(capability):
                capability_candidates.append(
                    CapabilityCandidate(
                        capability=capability,
                        spec=spec,
                        source="pack",
                        pack_slug=pack_slug,
                        package_name=pack_package_name,
                        manifest_path=str(entry.manifest_path),
                    )
                )
                continue

            namespaced = dataclasses.replace(spec, server_id=f"{pack_slug}.{spec.server_id}")
            registry.register(namespaced)


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
        binary_cache_dir = (
            str(Path(data_dir) / CACHE_DIR / "mcp-binaries")
            if config.binary_download is not None
            else None
        )
        configured.append(dataclasses.replace(config, env=env, binary_cache_dir=binary_cache_dir))
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
    :func:`discover_framework`), the shared roots / lock / parsed manifest
    specs are reused instead of recomputed. When ``None``, this function
    calls :func:`discover_framework` internally (back-compat for
    standalone callers like tests).
    """
    data_dir = str(runtime.paths.data_dir)
    workspace_root = str(FRAMEWORK_ROOT)
    if discovery is None:
        discovery = discover_framework(runtime, seed_lock=False)
    host_manifest_path = discovery.host_manifest_path
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

    # The resolved MCP environment is passed explicitly to the compiler;
    # the ambient environment is never mutated (see
    # :func:`_resolved_mcp_env`).
    resolved_env = _resolved_mcp_env(runtime, workspace_root)
    # Framework-owned (first-party) + user-owned servers. Capability
    # candidates are collected per layer and resolved to one owner
    # per capability after every layer has declared.
    capability_candidates: list[CapabilityCandidate] = []
    _register_host_and_user_servers(
        registry,
        host_manifest_path=host_manifest_path,
        host_specs=discovery.host_specs,
        user_manifest_path=user_manifest_path,
        user_specs=discovery.user_specs,
        capability_candidates=capability_candidates,
    )

    excluded_pack_servers = load_pack_server_exclusions(host_manifest_path)
    if user_manifest_path is not None and user_manifest_path.exists():
        excluded_pack_servers |= load_pack_server_exclusions(user_manifest_path)
    excluded_pack_servers |= session_exclude_servers

    skipped_candidates: list[_SkippedCandidate] = []
    _register_pack_servers(
        registry,
        discovery.pack_specs,
        excluded_pack_servers,
        capability_candidates,
        skipped_candidates,
    )

    # One owner per claimed capability; the winner registers under the
    # bare capability id (model-facing tool ids are implementation-
    # independent). Contested or misconfigured capabilities fail
    # startup loudly.
    _resolve_capability_owners(
        registry,
        capability_candidates,
        selected_capability_providers,
        skipped_candidates=skipped_candidates,
    )

    compiled = compile_registry(
        registry, data_dir=data_dir, workspace_root=workspace_root, base_env=resolved_env
    )
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
            return load_proxy_server_manifest(path, env=resolved_env)
        except ProxyServerManifestError:
            logger.warning("Skipping invalid %s proxy manifest at %s", source, path, exc_info=True)
            return []

    # The resolved MCP environment is passed explicitly to the proxy
    # manifest loader and compiler; the ambient environment is never
    # mutated (see :func:`_resolved_mcp_env`).
    resolved_env = _resolved_mcp_env(runtime, workspace_root)
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

    _expand_wildcard_providers(registry, runtime)

    providers = sorted({provider for spec in registry.all_specs() for provider in spec.providers})
    provider_base_urls = direct_provider_base_urls(runtime, providers)

    return compile_proxy_registry(
        registry,
        workspace_root=workspace_root,
        provider_base_urls=provider_base_urls,
        base_env=resolved_env,
    )


def _expand_wildcard_providers(registry: ProxyServerRegistry, runtime: HostRuntime) -> None:
    """Expand ``providers: [all]`` specs to the session's manifest providers.

    The kernel keeps packs host-agnostic: a pack declares the wildcard and the
    runtime expands it against the providers the host can actually call
    (providers named in the session model manifest). Each expansion produces
    a sibling spec ``<proxy_id>.<provider>`` — one process, one upstream —
    with its own sequential port (base port + N), validated to the TCP max.

    Which providers a proxy *can* serve is gateway knowledge, so it is asked
    of the ``ProviderGatewayProvider`` seam rather than answered from a host
    table: a gateway that builds its client from credentials (Google's native
    REST path, Bedrock's boto3 client, the Azure endpoint) reports no proxy
    protocol, and its providers are skipped. A wildcard is a pack's request,
    not an author error, so every skip is logged and none aborts startup.

    Three further skips keep the expansion honest: providers speaking a
    different wire protocol than the proxy declares (an ``openai`` proxy has
    nothing to do with an ``anthropic`` model), sibling ids an author declared
    explicitly (the explicit spec wins over the generated one), and ports that
    would overflow the TCP range.

    An empty surviving set removes the wildcard entirely; the host logs why.

    Args:
        registry: The session's proxy registry (mutated in place).
        runtime: The host runtime supplying the manifest and the gateway seam.
    """
    manifest = getattr(runtime.session, "manifest", None)
    manifest_providers = (
        tuple(sorted({str(model.provider).strip().lower() for model in (manifest.models or ())}))
        if manifest is not None
        else ()
    )
    proxy_protocol_for = _proxy_protocol_resolver(runtime)
    for spec in registry.all_specs():
        if spec.providers != (ALL_PROVIDERS,):
            continue
        if not manifest_providers:
            logger.info(
                "Proxy %r: 'providers: all' expands to nothing — no manifest providers",
                spec.proxy_id,
            )
            registry.replace_spec(spec.proxy_id, [])
            continue
        declared_ids = {other.proxy_id for other in registry.all_specs()}
        wildcard_overlay = spec.provider_options.get(ALL_PROVIDERS)
        expanded: list[ProxyServerSpec] = []
        unroutable: list[str] = []
        other_protocol: list[str] = []
        overflows: list[str] = []
        collisions: list[str] = []
        for provider in manifest_providers:
            wire = proxy_protocol_for(provider)
            if wire is None:
                unroutable.append(provider)
                continue
            if wire != spec.protocol.value:
                other_protocol.append(provider)
                continue
            sibling_id = f"{spec.proxy_id}.{provider}"
            if sibling_id in declared_ids:
                collisions.append(provider)
                continue
            port = spec.port + len(expanded)
            if port > _MAX_TCP_PORT:
                overflows.append(provider)
                continue
            expanded.append(
                dataclasses.replace(
                    spec,
                    proxy_id=sibling_id,
                    providers=(provider,),
                    port=port,
                    provider_options=(
                        {**spec.provider_options, provider: wildcard_overlay}
                        if wildcard_overlay is not None
                        else spec.provider_options
                    ),
                )
            )
        _log_wildcard_skips(
            proxy_id=spec.proxy_id,
            unroutable=unroutable,
            other_protocol=other_protocol,
            overflows=overflows,
            collisions=collisions,
            protocol=spec.protocol.value,
        )
        registry.replace_spec(spec.proxy_id, expanded)


def _log_wildcard_skips(
    *,
    proxy_id: str,
    unroutable: list[str],
    other_protocol: list[str],
    overflows: list[str],
    collisions: list[str],
    protocol: str,
) -> None:
    """Report each reason a wildcard provider did not become a proxy process."""
    if unroutable:
        logger.info(
            "Proxy %r: 'all' skips providers no proxy can serve: %s",
            proxy_id,
            ", ".join(sorted(unroutable)),
        )
    if other_protocol:
        logger.info(
            "Proxy %r: 'all' skips providers speaking a protocol other than %r: %s",
            proxy_id,
            protocol,
            ", ".join(sorted(other_protocol)),
        )
    if collisions:
        logger.warning(
            "Proxy %r: 'all' skips providers an explicit proxy already serves: %s",
            proxy_id,
            ", ".join(sorted(collisions)),
        )
    if overflows:
        logger.warning(
            "Proxy %r: skipping 'all'-expanded providers whose port overflows the valid range: %s",
            proxy_id,
            ", ".join(sorted(overflows)),
        )


def _proxy_protocol_resolver(runtime: HostRuntime) -> Callable[[str], str | None]:
    """Return a ``provider -> proxy wire protocol`` resolver for this host.

    Provider identity and wire shape are gateway knowledge, so the answer
    comes from the ``ProviderGatewayProvider`` seam; the host keeps no
    provider table of its own. A discovered provider that predates
    ``proxy_protocol_for`` reports nothing, which reads as "no proxy can
    serve it" — the same fail-closed posture as an unknown provider, so an
    old third-party gateway cannot have a proxy forced onto it.

    Args:
        runtime: The host runtime to resolve the active gateway seam against.

    Returns:
        A callable mapping a provider id to ``"openai"``/``"anthropic"``,
        or ``None`` when no proxy can serve that provider.
    """
    provider = _active_provider_gateway_provider(runtime)
    protocol_for = getattr(provider, "proxy_protocol_for", None)
    if not callable(protocol_for):
        return lambda _provider: None

    def _resolver(name: str) -> str | None:
        try:
            return protocol_for(name)
        except Exception:  # pylint: disable=broad-except
            logger.warning(
                "provider gateway seam failed to report a proxy protocol for %r; "
                "treating it as not proxy-routable",
                name,
                exc_info=True,
            )
            return None

    return _resolver


def _active_provider_gateway_provider(runtime: HostRuntime) -> Any:
    """Return the host's active ``ProviderGatewayProvider``, or ``None``.

    Discovered from the ``lisa.extensions.provider_gateways`` entry-point
    group — the same group ``TurnHooks`` assembly consults — because proxy
    compilation runs at startup, before any turn has wired a seam. Resolved
    once per compile, not once per provider.

    Args:
        runtime: The host runtime (unused today; kept so the signature can
            consult host state if a future host pins its provider).

    Returns:
        The discovered provider, or ``None`` when no gateway pack is active.
    """
    del runtime  # discovery is host-independent; see the docstring
    from kernel.provider_gateway_discovery import discover_provider_gateway_factories

    for entry in discover_provider_gateway_factories():
        loaded = entry.provider
        try:
            if callable(loaded) and not hasattr(loaded, "gateway_for"):
                loaded = loaded()
        except Exception:  # pylint: disable=broad-except
            logger.warning(
                "provider gateway entry %r failed to construct; skipping",
                entry.pack_name,
                exc_info=True,
            )
            continue
        if loaded is not None:
            return loaded
    return None


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
    "build_catalog_mcp_config",
    "build_content_mcp_config",
    "compile_mcp_configs",
    "configure_runtime",
    "load_pack_server_exclusions",
    "pack_server_excluded",
    "register_mcp_aliases",
    "setup_framework",
    "unique_pack_slug_for_root",
]

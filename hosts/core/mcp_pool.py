"""Transport-neutral MCP server config compilation.

This module consolidates the logic for compiling MCP server configurations
from the host manifest (``mcp_servers.yaml``) and any pack-contributed
manifests discovered under the active SDK content roots. It produces
``MCPClientConfig`` entries split into warm/cold lists for
:class:`kernel.tools.mcp_client.MCPToolPool`, plus the content/catalog
server configs that wire the kernel's content registry and operation/tool
catalogs.

All functions are transport-neutral: they take an explicit
:class:`hosts.core.runtime.HostRuntime` (or its derived paths) rather than
reading a global ``_RUNTIME_STATE`` dict. Transport hosts (CLI, web, VS
Code) construct a runtime and call :func:`setup_framework` /
:func:`compile_mcp_configs` during initialization.

The compiled configs are stored on the runtime via
``runtime.set_mcp_configs(warm, cold)``; pool creation/start/close lives
in the host's async lifecycle (``runtime.start()`` / ``runtime.aclose()``),
never in this module.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

from kernel.mcp_launch.compiler import compile_registry
from kernel.mcp_launch.policy import MCPServerRegistry
from kernel.mcp_launch.server_manifest import (
    MCPServerManifestError,
    load_mcp_server_manifest,
)
from kernel.pack_activation import load_disabled_pack_names, normalize_pack_name
from kernel.session_plan import ProviderSelections
from kernel.tools.mcp_client import MCPClientConfig

from hosts.core.config import HostConfig
from hosts.core.content_roots import (
    always_on_pack_names,
    host_mcp_manifest_path,
    infer_pack_package_name,
    load_installed_packs_lock,
    load_pack_policy_dict,
    sdk_content_roots,
    seed_bundled_packs_lock,
    unique_paths,
)
from hosts.core.runtime import HostRuntime

logger = logging.getLogger(__name__)

# ── Environment-variable name constants ──────────────────────────────
# Shared with the child ``lisa-content`` / ``lisa-catalog`` MCP servers
# so they receive the same provider-preference and disabled-pack signals.

_LISA_SESSION_PROVIDER_SELECTIONS_ENV = "LISA_SESSION_PROVIDER_SELECTIONS"
_LISA_DISABLED_PACKS_ENV = "LISA_DISABLED_PACKS"

# Venv root containing the console-script entry points (lisa-context,
# lisa-catalog, etc.). In repo mode this is <repo>/.venv; in tool-install
# mode this is sys.prefix. Exported as the FRAMEWORK_ROOT env var so that
# mcp_servers.yaml's ``venv_dir: "${FRAMEWORK_ROOT}"`` resolves to the
# directory whose Scripts/ (Windows) or bin/ (POSIX) holds the executables.
from hosts.core.paths import venv_root as _venv_root  # noqa: E402

FRAMEWORK_ROOT = _venv_root()


# ── Shared discovery cache ─────────────────────────────────────────


@dataclasses.dataclass(slots=True)
class _FrameworkDiscovery:
    """Shared discovery result for setup_framework + compile_mcp_configs.

    Both functions need the same sdk_content_roots walk, disabled-pack
    set, and installed-packs lock. Computing it once in
    :func:`discover_framework` and passing it to both avoids a redundant
    filesystem walk + lock read on every startup.
    """

    roots: list[Path]
    disabled_packs: set[str]
    lock: dict[str, Any]
    host_manifest_path: Path


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
    manifest_path = _resolve_host_manifest_path(config)
    # Seed before reading the lock so discovery sees a consistent state.
    # This closes the lifecycle hole where an unlisted bundled pack was
    # treated as enabled by intake/evidence (fail-open) but disabled by
    # content roots (default-off). compile_mcp_configs skips seeding to
    # preserve its original standalone behavior (no side effects).
    if seed_lock:
        seed_bundled_packs_lock(runtime.data_dir)
    plan_packs = runtime.session_plan.manifest.resources.packs
    session_pack_include = (*plan_packs.include, *plan_packs.required)
    session_pack_exclude = plan_packs.exclude
    roots = sdk_content_roots(
        runtime.data_dir,
        host_manifest_path=manifest_path,
        always_on_packs=config.always_on_packs,
        session_pack_include=session_pack_include,
        session_pack_exclude=session_pack_exclude,
    )
    host_mandatory_packs = always_on_pack_names(
        manifest_path,
        always_on_packs=config.always_on_packs,
    )
    disabled_packs = load_disabled_pack_names(runtime.data_dir)
    disabled_packs.difference_update(
        normalize_pack_name(pack_name) for pack_name in session_pack_include
    )
    disabled_packs.difference_update(host_mandatory_packs)
    disabled_packs.update(
        normalize_pack_name(pack_name)
        for pack_name in session_pack_exclude
        if normalize_pack_name(pack_name) not in host_mandatory_packs
    )
    lock = load_installed_packs_lock(runtime.data_dir)
    return _FrameworkDiscovery(
        roots=roots,
        disabled_packs=disabled_packs,
        lock=lock,
        host_manifest_path=manifest_path,
    )


# ── Content / catalog server configs ─────────────────────────────────


def build_content_mcp_config(
    *,
    data_dir: Path,
    content_roots: list[Path],
    disabled_packs: set[str],
    provider_selections: ProviderSelections,
) -> MCPClientConfig:
    """Build an MCPClientConfig that spawns lisa-content for discovered pack roots.

    The kernel calls ``list_modes``, ``list_operations``, and
    ``read_authored_text`` on this server to discover and load all content
    packs. The server does all filesystem access; the kernel never touches
    the filesystem directly.

    ``SDK_CONTENT_ROOTS`` includes both the pack root (for modes/operations
    discovery) and the ``content/`` subdirectory (so ``read_authored_text``
    can resolve authored text assets like ``prompts/system-prompt.md`` without a
    ``content/`` prefix).

    Args:
        data_dir: Host data directory (passed to the child as ``--data-dir``).
        content_roots: Active SDK content roots (pack roots + configured roots).
        disabled_packs: Normalized pack names to exclude in the child process.
        provider_selections: Explicit SessionPlan ownership decisions.
    """
    roots_for_child = content_roots
    subdir_roots = [root / "content" for root in roots_for_child if (root / "content").is_dir()]
    configured_roots = unique_paths([*roots_for_child, *subdir_roots])
    env: dict[str, str] = {
        "SDK_CONTENT_ROOTS": os.pathsep.join(str(root) for root in configured_roots)
    }

    # Pass disabled pack names to the lisa-content child process so that
    # entry-point discovery in the child also excludes disabled packs.
    if disabled_packs:
        env[_LISA_DISABLED_PACKS_ENV] = os.pathsep.join(sorted(disabled_packs))

    selection_payload = session_provider_selection_payload(provider_selections)
    if any(selection_payload.values()):
        env[_LISA_SESSION_PROVIDER_SELECTIONS_ENV] = json.dumps(
            selection_payload,
            ensure_ascii=True,
            sort_keys=True,
        )
    return MCPClientConfig(
        server_id="lisa-content",
        command=sys.executable,
        args=[
            "-m",
            "mcp_server.content.server",
            "--data-dir",
            str(data_dir),
        ],
        env=env,
    )


def session_provider_selection_payload(
    selections: ProviderSelections,
) -> dict[str, dict[str, str]]:
    """Convert immutable plan ownership choices into child-process payload data."""

    return {
        "modes": dict(selections.modes),
        "operations": dict(selections.operations),
        "tools": dict(selections.tools),
    }


def build_catalog_mcp_config(*, content_config: MCPClientConfig, data_dir: Path) -> MCPClientConfig:
    """Build an MCPClientConfig that spawns lisa-catalog for catalog tools.

    The kernel calls ``list_modes``, ``list_operations``, and ``list_tools``
    on this server to discover and load all content-pack catalogs. The
    server does all filesystem access; the kernel never touches the
    filesystem directly.

    Shares the same env/roots setup as the content server — the only
    difference is the module (``mcp_server.catalog.server`` vs
    ``mcp_server.content.server``).
    """
    return MCPClientConfig(
        server_id="lisa-catalog",
        command=content_config.command,
        args=[
            "-m",
            "mcp_server.catalog.server",
            "--data-dir",
            str(data_dir),
        ],
        env=content_config.env,
    )


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
    """
    data_dir = runtime.data_dir

    if discovery is None:
        discovery = discover_framework(runtime, seed_lock=True)
    roots = discovery.roots
    disabled_packs = discovery.disabled_packs

    content_config = build_content_mcp_config(
        data_dir=data_dir,
        content_roots=roots,
        disabled_packs=disabled_packs,
        provider_selections=runtime.session_plan.manifest.providers,
    )
    catalog_config = build_catalog_mcp_config(content_config=content_config, data_dir=data_dir)

    from kernel.ipsar_fsm import register_content_mcp, register_operation_callbacks  # noqa: PLC0415
    from kernel.operations.registry import (  # noqa: PLC0415
        execute_operation,
        get_operation_catalog,
        register_operation_mcp,
    )
    from kernel.tools.registry import register_tool_mcp  # noqa: PLC0415

    register_operation_mcp(catalog_config)
    register_tool_mcp(catalog_config)
    register_operation_callbacks(get_operation_catalog, execute_operation)
    register_content_mcp(catalog_config)

    # Wire the MCPContentRegistry as the content registry for context frames.
    from kernel.context.context_frames import set_content_registry  # noqa: PLC0415
    from kernel.context.mcp_content_registry import MCPContentRegistry  # noqa: PLC0415

    set_content_registry(MCPContentRegistry(content_config))

    # Wire the production extension-evidence builder so the context broker's
    # catalog builder picks up evidence from installed SDK extensions
    # (Cloudflare, GitHub, Stripe) via entry-point discovery.
    from sdk.context.extension_evidence_builder import (  # noqa: PLC0415
        set_production_extension_evidence_builder,
    )

    set_production_extension_evidence_builder(
        excluded_packs=disabled_packs,
    )


def configure_runtime(runtime: HostRuntime) -> None:
    """Reset cached provider clients on the runtime."""
    runtime.azure_client = None
    runtime.openai_client = None
    runtime.anthropic_http_client = None
    runtime.ollama_client = None
    runtime.google_client = None


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
    config = runtime.config
    data_dir = str(runtime.data_dir)
    workspace_root = str(FRAMEWORK_ROOT)
    if discovery is None:
        discovery = discover_framework(runtime, seed_lock=False)
    host_manifest_path = discovery.host_manifest_path
    roots = discovery.roots
    selected_capability_providers = dict(
        runtime.session_plan.manifest.providers.capabilities
    )

    # ── Session-plan resource preferences ──────────────────────────
    # The immutable plan can narrow installed resources and select an
    # explicit owner, but cannot alter manifest launch/security policy.
    plan_resources = runtime.session_plan.manifest.resources
    plan_providers = runtime.session_plan.manifest.providers
    session_exclude_servers: set[str] = set()
    session_cap_preferences: dict[str, list[str]] = {}
    for entry in plan_resources.servers.exclude:
        normalized = entry.strip().lower()
        if normalized:
            session_exclude_servers.add(normalized)
    for cap, pack in plan_providers.capabilities.items():
        session_cap_preferences.setdefault(cap, []).append(pack)

    # Capability ownership is explicit session policy, not host preference.
    if session_cap_preferences:
        selected_capability_providers = {
            capability: providers[0]
            for capability, providers in session_cap_preferences.items()
            if providers
        }

    # Make FRAMEWORK_ROOT available for ${FRAMEWORK_ROOT} expansion in
    # the manifest's path-like fields (venv_dir, command_path, etc.).
    os.environ.setdefault("FRAMEWORK_ROOT", workspace_root)

    # Propagate the resolved timezone so ${LISA_TIMEZONE} expansion in
    # mcp_servers.yaml (e.g. lisa-context --timezone) works even when the
    # user hasn't set the env var explicitly — the host resolved it from
    # CLI flags or the system-local tzinfo.
    os.environ.setdefault("LISA_TIMEZONE", config.timezone)

    registry = MCPServerRegistry()

    # All host-owned servers (first-party) from mcp_servers.yaml. The
    # manifest loader honors ``disabled: true`` by skipping entries, and
    # parses ``warm`` / ``required`` onto the spec. The compiler passes
    # them through to MCPClientConfig. The host manifest no longer carries
    # a ``memory:`` section or memory server entries; memory backends +
    # memory tool servers are contributed by the learning pack (see
    # sdk/packs/learning/mcp_servers.yaml) and merged below.
    specs, memory_config = load_mcp_server_manifest(host_manifest_path)
    for spec in specs:
        registry.register(spec)

    excluded_pack_servers = load_pack_server_exclusions(host_manifest_path)
    excluded_pack_servers |= session_exclude_servers
    used_pack_slugs: set[str] = set()
    memory_owner_slug: str | None = None
    memory_candidates: list[dict[str, Any]] = []
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
                {
                    "pack_slug": pack_slug,
                    "package_name": pack_package_name,
                    "manifest_path": pack_manifest_path,
                    "memory_config": pack_memory,
                }
            )

        for spec in pack_specs:
            if pack_server_excluded(pack_slug, spec.server_id, excluded_pack_servers):
                logger.info(
                    "Pack MCP server excluded by policy: %s.%s",
                    pack_slug,
                    spec.server_id,
                )
                continue

            # Compile-time namespacing allows multiple packs to declare the same
            # server_id without collision in the global registry.
            namespaced = dataclasses.replace(spec, server_id=f"{pack_slug}.{spec.server_id}")
            registry.register(namespaced)

    selected_memory_provider = normalize_pack_name(
        str(selected_capability_providers.get("memory") or "")
    )
    selected_memory_owner = select_capability_provider_owner(
        candidates=memory_candidates,
        selected_provider=selected_memory_provider,
    )

    if selected_memory_owner is not None:
        memory_config = selected_memory_owner["memory_config"]
        memory_owner_slug = str(selected_memory_owner.get("pack_slug") or "") or None
        selected_manifest = str(selected_memory_owner.get("manifest_path") or "")
        selected_package = str(selected_memory_owner.get("package_name") or "")
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
                candidate.get("package_name"),
                candidate.get("pack_slug"),
                candidate.get("manifest_path"),
                selected_package,
            )

    runtime.memory_config = memory_config
    runtime.memory_config_pack_slug = memory_owner_slug

    compiled = compile_registry(registry, data_dir=data_dir, workspace_root=workspace_root)
    warm = [c for c in compiled if c.warm]
    cold = [c for c in compiled if not c.warm]
    return warm, cold


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


def provider_identifiers_for_pack(*, package_name: str, pack_slug: str) -> set[str]:
    """Return the set of normalized identifiers that refer to one pack."""
    identifiers = {
        normalize_pack_name(package_name),
        normalize_pack_name(pack_slug),
    }
    normalized_package = normalize_pack_name(package_name)
    if normalized_package.startswith("lisa-pack-"):
        identifiers.add(normalized_package[len("lisa-pack-"):])
    return {identifier for identifier in identifiers if identifier}


def select_capability_provider_owner(
    *,
    candidates: list[dict[str, Any]],
    selected_provider: str,
) -> dict[str, Any] | None:
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
                package_name=str(candidate.get("package_name") or ""),
                pack_slug=str(candidate.get("pack_slug") or ""),
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
        "Multiple active capability providers require an explicit "
        "providers.capabilities selection."
    )


def pack_server_excluded(
    pack_slug: str, declared_server_id: str, exclusions: set[str]
) -> bool:
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


def _resolve_host_manifest_path(config: HostConfig) -> Path:
    """Return the host MCP manifest path from config or the core default."""
    if config.mcp_manifest_path is not None:
        return config.mcp_manifest_path
    return host_mcp_manifest_path()


def register_mcp_aliases(pool: Any) -> None:
    """Register generic aliases from compiled MCP configs as synthetic tools.

    After ``pool.start()``, walk every config's ``aliases`` tuple and register
    each as a synthetic tool via ``mcp_alias_tool``. The alias gives the model
    a stable name (e.g. ``cloudflare.search``) instead of the namespaced
    server id (e.g. ``lisa_packs.cloudflare.cloudflare-api.search``).

    Must be called after ``pool.start()`` so the alias handlers can route to
    live server sessions.
    """
    from kernel.tools.synthetic import mcp_alias_tool  # noqa: PLC0415

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
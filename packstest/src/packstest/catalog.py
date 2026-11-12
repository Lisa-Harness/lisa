"""In-process pack activation and validation.

The framework's real pack route is entry-point discovery → content-root
discovery → MCP catalog subprocess. That route is correct for production but
far too slow for the edit-test-fix loop. This module activates a pack
directly from its source checkout:

- :func:`load_pack_content_root` — resolve a pack's content root from source.
- :func:`build_tool_catalog` — run the SDK's real tool catalog scanner
  (``sdk.content.build_tool_catalog_json``) over the pack's ``tools/`` dir.
- :func:`validate_pack` — the pack-validation eval: entry points resolve,
  every tool YAML parses, implementation modules import, tool IDs are unique,
  provider ids resolve the way session configuration expects.
- :func:`activate_tool` — import a tool's implementation module in-process.

Validation mirrors the production contract in
``sdk.content._catalog`` (module-path inference, provider-id backfill) and
``kernel.content_roots`` (entry-point groups) without duplicating that logic —
it calls the real loaders wherever possible so the harness cannot drift.
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from pkgutil import extend_path
from typing import Any

import yaml
from sdk.content import build_tool_catalog_json, resolve_tool_roots

__all__ = [
    "PackIssue",
    "PackValidationReport",
    "activate_tool",
    "build_tool_catalog",
    "load_pack_content_root",
    "load_pack_entry_points",
    "validate_pack",
]

logger = logging.getLogger(__name__)


class PackIssue:
    """One validation finding; ``severity`` gates pass/fail."""

    __slots__ = ("check", "detail", "severity")

    def __init__(self, severity: str, check: str, detail: str) -> None:
        self.severity = severity
        self.check = check
        self.detail = detail

    def __repr__(self) -> str:
        return f"[{self.severity}] {self.check}: {self.detail}"


@dataclass(slots=True)
class PackValidationReport:
    """Structured validation result for one content pack."""

    pack_path: Path
    provider_id: str
    tools: dict[str, dict[str, Any]] = field(default_factory=dict)
    entry_points: dict[str, Any] = field(default_factory=dict)
    issues: list[PackIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[PackIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[PackIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        """One-line human summary for test failure messages."""

        lines = [f"pack={self.pack_path} provider_id={self.provider_id!r} ok={self.ok}"]
        lines.extend(repr(i) for i in self.issues)
        return "\n".join(lines)


def _pack_slug(pack_path: Path) -> str:
    """Return the distribution slug (``lisa-pack-markdown`` → ``markdown``)."""
    import tomllib  # stdlib, no cycle risk

    pyproject = pack_path / "pyproject.toml"
    if not pyproject.is_file():
        return ""
    try:
        payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    name = str(payload.get("project", {}).get("name", "")).strip()
    if name.startswith("lisa-pack-"):
        return name.removeprefix("lisa-pack-")
    return name


def _namespace_children(pack_path: Path) -> list[Path]:
    """Pack packages under a shared ``lisa_packs/`` namespace at the root."""
    namespace = pack_path / "lisa_packs"
    if not namespace.is_dir():
        return []
    return [d for d in namespace.iterdir() if d.is_dir() and d.name != "__pycache__"]


def _resolve_namespace_child(children: list[Path], slug: str) -> Path:
    """Pick the namespace child matching the distribution slug."""
    if len(children) == 1:
        return children[0]
    for child in children:
        if child.name == slug:
            return child
    return children[0]


def _src_fallback_root(pack_path: Path, slug: str) -> Path | None:
    """Bounded ``src/*/`` walk for checkouts with no readable entry point."""
    src = pack_path / "src"
    if not src.is_dir():
        return None
    for parent in sorted(src.glob("*")):
        if not parent.is_dir() or parent.name in ("__pycache__", "tests"):
            continue
        nested = parent / "lisa_packs" / slug
        if nested.is_dir():
            return nested
        if _has_content_subdirs(parent):
            return parent
    return None


def load_pack_content_root(pack_path: Path) -> Path:
    """Return the pack's content root (the directory holding content subdirs).

    The pack path is the distribution checkout (``pyproject.toml`` at top).
    Resolution order:

    1. A shared ``lisa_packs/`` namespace at the checkout root (standard
       external-pack layout): the child matching the distribution slug.
    2. The ``lisa.extensions.content_roots`` entry point's module: converted
       from a dotted module path to its package directory. This handles every
       ``src/`` layout (``src/lisa_packs/<slug>``, bundled
       ``src/sdk/packs/<slug>``, nested ``lisa_packs/<slug>/lisa_packs/<slug>``)
       using the same declaration the production router consumes, so the
       locator cannot drift from what the runtime resolves.
    3. A bounded ``src/*/`` walk for checkouts without a readable entry point.
    4. The checkout itself (flat content-root checkouts, synthetic packs).
    """

    children = _namespace_children(pack_path)
    if children:
        return _resolve_namespace_child(children, _pack_slug(pack_path))
    entry_root = _content_root_from_entry_point(pack_path)
    if entry_root is not None:
        return entry_root
    return _src_fallback_root(pack_path, _pack_slug(pack_path)) or pack_path


def _read_content_root_entries(pack_path: Path) -> list[tuple[str, str]]:
    """Return ``(name, module-path)`` content-root entry-point targets."""
    import tomllib  # stdlib, no cycle risk

    pyproject = pack_path / "pyproject.toml"
    try:
        payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        groups = payload.get("project", {}).get("entry-points", {})
        group = groups.get("lisa.extensions.content_roots", {})
    except (OSError, ValueError):
        return []
    if not isinstance(group, dict):
        return []
    return [(str(k), str(v)) for k, v in group.items()]


_CONTENT_SUBDIRS = ("tools", "modes", "operations", "content")


def _has_content_subdirs(candidate: Path) -> bool:
    return any((candidate / d).is_dir() for d in _CONTENT_SUBDIRS)


def _content_root_from_entry_point(pack_path: Path) -> Path | None:
    """Resolve the content root from the pack's declared entry point, or None.

    Reads ``[project.entry-points."lisa.extensions.content_roots"]`` from
    ``pyproject.toml`` and converts the declared dotted module path to its
    package directory *relative to the pack checkout* — never through
    ``sys.modules``, because a same-named package from another checkout may
    already be cached in this process (two external packs in one interpreter)
    and importlib would return the wrong one. Returns the first root that
    exists and holds a content subdir; ``None`` when nothing resolves, leaving
    ``load_pack_entry_points`` to report entry-point load issues.
    """
    # Directories a dotted module path could resolve against, in order: the
    # checkout, its ``src/`` tree, and the namespace leaf (bundled packs host
    # ``sdk.packs.core`` under ``src/sdk``, so the leaf name applies too).
    search_roots = [pack_path]
    src = pack_path / "src"
    if src.is_dir():
        search_roots.insert(1, src)
    for _, target in _read_content_root_entries(pack_path):
        module_name = target.partition(":")[0]
        if not module_name:
            continue
        parts = module_name.split(".")
        candidates = [base.joinpath(*parts) for base in search_roots]
        if len(parts) > 1:
            candidates.extend(base / parts[-1] for base in search_roots)
        resolved = next((c for c in candidates if c.is_dir() and _has_content_subdirs(c)), None)
        if resolved is not None:
            return resolved
    return None


def build_tool_catalog(
    pack_path: Path,
    *,
    invalidate: bool = True,
    ensure_importable: bool = True,
) -> dict[str, dict[str, Any]]:
    """Run the SDK's real tool-catalog scan over the pack's ``tools/`` dir.

    Uses ``sdk.content.build_tool_catalog_json`` (the production code path)
    so validation exercises the same module-path inference and provider-id
    backfill the host's catalog server performs. Module-path inference walks
    up from the content root while directories resolve as importable
    packages, so the pack checkout must be on ``sys.path`` first.
    """

    if ensure_importable:
        _ensure_source_importable(pack_path)
    if invalidate:
        from sdk.content import invalidate_catalog_cache  # lazy: cache reset

        invalidate_catalog_cache()
    root = load_pack_content_root(pack_path)
    tool_roots = resolve_tool_roots([root])
    if not tool_roots:
        return {}
    catalog: object = json.loads(build_tool_catalog_json(tool_roots))
    if not isinstance(catalog, dict) or any(
        not isinstance(key, str) or not isinstance(value, dict) for key, value in catalog.items()
    ):
        raise ValueError("Tool catalog must be an object mapping tool IDs to objects")
    return catalog


def _importable(module_path: str) -> bool:
    """Return whether a dotted module path imports without error."""

    if not module_path:
        return False
    try:
        importlib.import_module(module_path)
    except Exception:  # pylint: disable=broad-except
        logger.debug("import failed for %s", module_path, exc_info=True)
        return False
    return True


def activate_tool(tool_dict: dict[str, Any]) -> Any:
    """Import a tool's implementation module and return its ``execute``.

    Mirrors the kernel's in-process dispatch: import ``module_path`` and
    resolve ``execute`` (falling back to module-level ``execute`` attribute
    lookup identical to the production dispatch).
    """

    module_path = str(tool_dict.get("module_path", ""))
    if not module_path:
        raise ImportError(f"Tool {tool_dict.get('tool_id')!r} has no module_path")
    module = importlib.import_module(module_path)
    return module.execute


def load_pack_entry_points(pack_path: Path) -> dict[str, Any]:
    """Load the pack's declared extension entry points from source.

    Reads ``pyproject.toml`` ``[project.entry-points."lisa.extensions.*"]``
    and invokes each callable directly (no installed metadata required).
    Returns ``{group: {name: result}}`` for groups whose callables load and
    return without raising.
    """

    import tomllib  # stdlib, no cycle risk

    pyproject = pack_path / "pyproject.toml"
    if not pyproject.is_file():
        return {}
    payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    groups = payload.get("project", {}).get("entry-points", {})
    results: dict[str, Any] = {}
    for group, entries in groups.items():
        if not group.startswith("lisa.extensions."):
            continue
        for name, target in entries.items():
            try:
                module_name, _, attr = target.partition(":")
                module = importlib.import_module(module_name)
                callable_obj = getattr(module, attr)
                results.setdefault(group, {})[name] = _probe_entry_callable(callable_obj)
            except Exception:  # pylint: disable=broad-except
                logger.warning("entry point %s:%s failed to load", group, name, exc_info=True)
                results.setdefault(group, {})[name] = None
    return results


def _probe_entry_callable(callable_obj: Any) -> Any:
    """Invoke an entry-point callable in a load-time-safe way.

    Entry points come in several calling conventions:

    - Zero-arg factories (``get_content_root()``) and classes with
      all-default ``__init__`` — callable with no arguments.
    - Runtime factories requiring arguments are loaded without invocation.
      Their inputs belong to the runtime, not to a load-time probe.
    """

    signature = inspect.signature(callable_obj)
    try:
        signature.bind()
    except TypeError:
        return callable_obj
    return callable_obj()


def _checkout_lisa_packs_names(checkout: Path) -> set[str]:
    """``lisa_packs.<slug>`` names the checkout itself provides."""

    names: set[str] = set()
    for base in (checkout / "lisa_packs", checkout / "src" / "lisa_packs"):
        if not base.is_dir():
            continue
        for child in base.iterdir():
            if child.is_dir() and child.name != "__pycache__":
                names.add(f"lisa_packs.{child.name}")
            elif child.suffix == ".py" and child.stem != "__init__":
                names.add(f"lisa_packs.{child.stem}")
    return names


def _stale_lisa_packs_modules(checkout: Path) -> list[str]:
    """Cached ``lisa_packs.<slug>*`` modules that collide with ``checkout``.

    Two source checkouts in one interpreter collide on the same module name:
    once ``lisa_packs.widgets`` is imported from pack A, importlib returns
    pack A's module for pack B. A module the checkout also provides whose own
    file (or namespace search path) resolves outside the checkout is stale and
    must be evicted before the pack's modules import from the right source.

    Modules the checkout does not provide never collide and stay cached, and
    the ``lisa_packs`` namespace itself is widened rather than evicted.
    Evicting either would orphan live references held by callers that
    imported another pack first (their functions would keep the old module's
    globals while fresh imports see a new module object).
    """
    stale: list[str] = []
    provided = _checkout_lisa_packs_names(checkout)
    names_lisa = [
        m for m in list(sys.modules) if any(m == p or m.startswith(f"{p}.") for p in provided)
    ]
    for name in names_lisa:
        module = sys.modules.get(name)
        if module is None:
            continue
        files: list[Path] = []
        module_file = getattr(module, "__file__", None)
        if module_file:
            files.append(Path(module_file))
        try:
            module_path = getattr(module, "__path__", None)
            if module_path:
                files.extend(Path(p) for p in list(module_path))
        except KeyError:
            # A namespace path whose parent was already evicted: unusable.
            return names_lisa
        if not files:
            continue
        if all(checkout not in f.resolve().parents for f in files):
            stale.append(name)
    return stale


def _ensure_source_importable(pack_path: Path) -> list[Path]:
    """Make the pack's importable packages reachable from ``sys.path``.

    Standard src-less layout (``lisa_packs/`` at the checkout root) adds the
    checkout; a ``src/`` layout adds ``src/``. Returns the paths added.

    Same-named modules from another checkout (a previous ``validate_pack``
    call in this process) are evicted first: a regular package's ``__path__``
    is pinned at first import and ``extend_path`` cannot widen a stale child.
    The shared ``lisa_packs`` namespace is widened with this checkout's
    portions placed first so its slugs win over same-named older portions.
    """

    added: list[Path] = []
    for candidate in (pack_path, pack_path / "src"):
        resolved = str(candidate.resolve())
        if candidate.is_dir() and resolved not in sys.path:
            sys.path.insert(0, resolved)
            added.append(candidate)
    importlib.invalidate_caches()
    checkout = pack_path.resolve()
    for name in _stale_lisa_packs_modules(checkout):
        sys.modules.pop(name, None)
    importlib.invalidate_caches()
    namespace = sys.modules.get("lisa_packs")
    if namespace is not None and hasattr(namespace, "__path__"):
        widened = extend_path(list(namespace.__path__), namespace.__name__)
        own = [p for p in widened if checkout in Path(p).resolve().parents]
        namespace.__path__ = own + [p for p in widened if p not in own]
    return added


def _validate_tool_entry(
    tool_id: str,
    tool_dict: dict[str, Any],
    seen: set[str],
) -> list[PackIssue]:
    """Validate one cataloged tool against the production contracts."""

    issues: list[PackIssue] = []
    if tool_id in seen:
        issues.append(PackIssue("error", "tool-id", f"duplicate tool_id {tool_id!r}"))
        return issues
    seen.add(tool_id)

    issues.extend(check_tool_yaml_schema(tool_id, tool_dict))

    module_path = str(tool_dict.get("module_path", ""))
    if not _importable(module_path):
        issues.append(
            PackIssue("error", "tool-import", f"{tool_id}: module {module_path!r} not importable")
        )
        return issues
    if not hasattr(importlib.import_module(module_path), "execute"):
        issues.append(PackIssue("error", "tool-execute", f"{tool_id}: module lacks execute()"))
    if not tool_dict.get("tool_provider_id"):
        issues.append(PackIssue("error", "tool-provider", f"{tool_id}: provider id unresolved"))

    # Validate against the real EffectKind contract, not a hardcoded set.
    from typing import get_args  # stdlib, no cycle risk

    from kernel.approval import EffectKind  # lazy: optional import

    effect = str(tool_dict.get("effect", "write"))
    if effect not in set(get_args(EffectKind)):
        issues.append(PackIssue("error", "tool-effect", f"{tool_id}: unknown effect {effect!r}"))
    return issues


def validate_pack(pack_path: Path, *, ensure_importable: bool = True) -> PackValidationReport:
    """Validate a pack from source without installing it.

    Checks (each mirrors a production contract):
    1. ``pyproject.toml`` exists and declares a distribution name.
    2. The content root exists and contains at least one content subdir.
    3. The SDK tool catalog scan returns tools whose ``module_path`` imports
       in-process (the production dispatch contract).
    4. Tool IDs are unique within the pack.
    5. Every ``lisa.extensions.*`` entry point loads and returns.
    """

    if ensure_importable:
        _ensure_source_importable(pack_path)

    root = load_pack_content_root(pack_path)
    catalog = build_tool_catalog(pack_path)

    import tomllib  # stdlib, no cycle risk

    payload = tomllib.loads((pack_path / "pyproject.toml").read_text(encoding="utf-8"))
    dist_name = str(payload.get("project", {}).get("name", ""))
    provider_id = catalog and next(iter(catalog.values()), {}).get("tool_provider_id", "")
    report = PackValidationReport(
        pack_path=pack_path,
        provider_id=str(provider_id or dist_name),
    )

    if not dist_name:
        report.issues.append(
            PackIssue("error", "distribution", "pyproject.toml has no [project] name")
        )

    content_subdirs = [
        d for d in ("tools", "modes", "operations", "content") if (root / d).is_dir()
    ]
    if not content_subdirs:
        report.issues.append(
            PackIssue("warning", "content-root", f"no content subdirs found under {root}")
        )
    else:
        report.entry_points = load_pack_entry_points(pack_path)
        for group, entries in report.entry_points.items():
            for name, value in entries.items():
                if value is None:
                    report.issues.append(
                        PackIssue("error", "entry-point", f"{group}:{name} failed to load")
                    )

    seen: set[str] = set()
    for tool_id, tool_dict in sorted(catalog.items()):
        report.issues.extend(_validate_tool_entry(tool_id, tool_dict, seen))
    report.tools = catalog
    return report


def check_tool_yaml_schema(tool_id: str, tool_dict: dict[str, Any]) -> list[PackIssue]:
    """Validate one tool dict against the ``ToolDefinition`` contract shape.

    Kept separate so evals can grade individual tools without a full pack
    scan.
    """

    issues: list[PackIssue] = []
    if not tool_dict.get("tool_id"):
        issues.append(PackIssue("error", "tool-yaml", f"{tool_id}: missing tool_id"))
    if not tool_dict.get("description"):
        issues.append(PackIssue("warning", "tool-yaml", f"{tool_id}: missing description"))
    # Mirrors the kernel contract (kernel/tools/registry.py): timeout_seconds
    # >= 0, where 0 disables the wrapper timeout for self-bounding tools
    # (sdk/AUTHORING.md). Only negatives are invalid.
    if int(tool_dict.get("timeout_seconds", 30)) < 0:
        issues.append(PackIssue("error", "tool-yaml", f"{tool_id}: negative timeout_seconds"))
    if int(tool_dict.get("max_calls_per_turn", 0)) < 0:
        issues.append(PackIssue("error", "tool-yaml", f"{tool_id}: negative max_calls_per_turn"))
    return issues


def yaml_tools_on_disk(pack_path: Path) -> dict[str, Path]:
    """Map ``tool_id`` → ``.tool.yml`` path for every tool file on disk.

    Reads raw YAML (not the catalog) so schema issues that break catalog
    loading are still attributable to a specific file.
    """

    root = load_pack_content_root(pack_path)
    tools_dir = root / "tools"
    out: dict[str, Path] = {}
    if not tools_dir.is_dir():
        return out
    for yml_path in sorted(tools_dir.rglob("*.tool.yml")):
        try:
            metadata = yaml.safe_load(yml_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        tool_id = metadata.get("tool_id") if isinstance(metadata, dict) else None
        if isinstance(tool_id, str) and tool_id:
            out[tool_id] = yml_path
    return out

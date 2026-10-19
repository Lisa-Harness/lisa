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


def load_pack_content_root(pack_path: Path) -> Path:
    """Return the pack's content root (the directory holding content subdirs).

    The pack path is the distribution checkout (``pyproject.toml`` at top).
    Content roots registered via entry points point at the package directory
    (``lisa_packs/<name>``) which hosts ``tools/``/``modes/``/``operations/``.
    """

    pyproject = pack_path / "pyproject.toml"
    if pyproject.is_file():
        candidate = pack_path / "lisa_packs"
        if candidate.is_dir():
            children = [d for d in candidate.iterdir() if d.is_dir() and d.name != "__pycache__"]
            if len(children) == 1:
                return children[0]
            if len(children) > 1:
                # Multiple pack packages under the shared namespace: pick the
                # one whose name matches the distribution slug.
                import tomllib  # stdlib, no cycle risk

                payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
                dist_name = str(payload.get("project", {}).get("name", ""))
                slug = dist_name.removeprefix("lisa-pack-")
                for child in children:
                    if child.name == slug:
                        return child
                return children[0]
    return pack_path


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
    return json.loads(build_tool_catalog_json(tool_roots))


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


def _ensure_source_importable(pack_path: Path) -> list[Path]:
    """Make the pack's importable packages reachable from ``sys.path``.

    Standard src-less layout (``lisa_packs/`` at the checkout root) adds the
    checkout; a ``src/`` layout adds ``src/``. Returns the paths added.
    """

    added: list[Path] = []
    for candidate in (pack_path, pack_path / "src"):
        resolved = str(candidate.resolve())
        if candidate.is_dir() and resolved not in sys.path:
            sys.path.insert(0, resolved)
            added.append(candidate)
    importlib.invalidate_caches()
    namespace = sys.modules.get("lisa_packs")
    if namespace is not None and hasattr(namespace, "__path__"):
        namespace.__path__ = extend_path(list(namespace.__path__), namespace.__name__)
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

"""Host-layer ``from:`` and ``body:`` directive resolver for session contexts.

This module sits between session source resolution
(:func:`hosts.core.session_context.resolve_session_context_source`) and the
strict kernel compiler (:func:`kernel.session_plan.compile_session_plan`). It
consumes two host-only directives that a user may place in their
``session_context.md`` front matter:

``from:``
    Usable under any of the ``resources``, ``providers``, ``routing``, or
    ``context`` blocks. Pulls that *whole block* from a pack's bundled
    ``sessions/<name>.md``. Mutually exclusive with any other key in the same
    block. The block is the unit of inheritance — there is no per-field merge.

``body:``
    A top-level directive that controls the Markdown body. Three modes:

    - ``write`` (default when ``body:`` is omitted): the user's body is
      primary. Unchanged kernel behavior.
    - ``pull``: the user's body is ignored (may be empty); the pack session's
      body becomes primary.
    - ``context``: the pack session's body is prepended, wrapped in
      ``<pack_session_body>`` … ``</pack_session_body>``, followed by the
      user's body. Both must be non-empty.

Design constraints (see ``/memories/session/plan.md``):

- The kernel ``SessionManifestV2`` schema is untouched. ``from:`` and
  ``body:`` are stripped from the front matter before the kernel sees it, so
  the strict ``extra="forbid"`` schema rejects them if a pack session
  attempts to use them — this enforces "packs are leaves" for free.
- Identity (``schema_version``, ``name``, ``surface``) is always authored
  locally. There is no top-level ``from:``.
- No recursion: only user sessions may use ``from:``/``body:``. Pack sessions
  are leaf sources.
- Passthrough: when no directive is present the resolver returns the original
  text unchanged, so existing sessions behave identically.
- Hard cutover, no merge: the block is either fully pulled or fully local.

The resolver is pure: callers inject a ``pack_session_lookup`` callable
(``pack, session -> Path``) so the module is unit-testable without an
installed-pack environment.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from kernel.ipsar_fsm import parse_front_matter
from kernel.session_plan import SessionPlanError, compile_session_plan

logger = logging.getLogger(__name__)


# Blocks that may carry a ``from:`` directive. Each must be a top-level key of
# the v2 session manifest whose value is itself a mapping (a "block").
_FROM_BLOCK_NAMES: tuple[str, ...] = ("resources", "providers", "routing", "context", "execution")

# The single accepted wrapper tag for ``body: mode: context``.
_PACK_SESSION_BODY_TAG = "pack_session_body"


PackSessionLookup = Callable[..., Path]
"""Type of the injected pack-session path resolver.

Mirrors :func:`hosts.core.content_roots.resolve_pack_session_path`: called as
``lookup(pack=..., session=...)`` and returns the ``Path`` to the pack's
``sessions/<session>.md`` file (raising ``SessionPlanError`` on miss)."""


def resolve_session_from_directives(
    *,
    text: str,
    source_uri: str,
    pack_session_lookup: PackSessionLookup,
) -> str:
    """Resolve ``from:`` and ``body:`` directives and return recompiled text.

    Returns a single Markdown string (front matter + body) ready for
    :func:`kernel.session_plan.compile_session_plan`. When no directive is
    present the returned text is identical to the input.
    """

    try:
        front_matter, body = parse_front_matter(text)
    except Exception as exc:  # pragma: no cover - parser-specific exception
        raise SessionPlanError(f"invalid session front matter in {source_uri}: {exc}") from exc

    if not front_matter:
        # No front matter at all — let the kernel compiler raise its own
        # schema error. Nothing for this resolver to do.
        return text

    if not _has_directives(front_matter):
        return text

    # Compile the user manifest shape enough to know which blocks exist, but
    # do not validate it through SessionManifestV2 yet — we are about to
    # mutate it. We only need to know: which of the leaf blocks are present
    # as mappings, and whether a top-level ``body`` directive exists.
    resolved_front_matter = _resolve_blocks(front_matter, source_uri, pack_session_lookup)
    resolved_body, body_mode = _resolve_body(
        front_matter=front_matter,
        user_body=body,
        source_uri=source_uri,
        pack_session_lookup=pack_session_lookup,
    )
    # The ``body`` directive is host-only; strip it so the kernel schema
    # (extra="forbid") never sees it.
    resolved_front_matter.pop("body", None)

    return _emit_text(resolved_front_matter, resolved_body, body_mode)


def _has_directives(front_matter: dict[str, Any]) -> bool:
    """Return True if any leaf block uses ``from:`` or a ``body:`` key exists."""

    if "body" in front_matter:
        return True
    for name in _FROM_BLOCK_NAMES:
        block = front_matter.get(name)
        if isinstance(block, dict) and "from" in block:
            return True
    return False


def _resolve_blocks(
    front_matter: dict[str, Any],
    source_uri: str,
    pack_session_lookup: PackSessionLookup,
) -> dict[str, Any]:
    """Return a (possibly mutated) front-matter dict with ``from:`` blocks expanded."""

    resolved = dict(front_matter)
    for name in _FROM_BLOCK_NAMES:
        block = resolved.get(name)
        if not isinstance(block, dict) or "from" not in block:
            continue
        pack_block = _resolve_from_block(
            block_name=name,
            block=block,
            source_uri=source_uri,
            pack_session_lookup=pack_session_lookup,
        )
        resolved[name] = pack_block
    return resolved


def _resolve_from_block(
    *,
    block_name: str,
    block: dict[str, Any],
    source_uri: str,
    pack_session_lookup: PackSessionLookup,
) -> dict[str, Any]:
    """Validate and resolve one ``from:`` block directive.

    Mutual exclusivity: the block must contain exactly one key, ``from``.
    The ``from`` value must be ``{pack: <str>, session: <str>}``.
    """

    sibling_keys = [k for k in block if k != "from"]
    if sibling_keys:
        raise SessionPlanError(
            f"session {source_uri!r}: block {block_name!r} uses 'from:' and so "
            f"may not declare any other keys (found: {sorted(sibling_keys)})"
        )

    spec = block.get("from")
    if not isinstance(spec, dict):
        raise SessionPlanError(
            f"session {source_uri!r}: block {block_name!r} 'from:' must be a "
            f"mapping with 'pack' and 'session' keys"
        )
    pack = spec.get("pack")
    session = spec.get("session")
    if not isinstance(pack, str) or not isinstance(session, str):
        raise SessionPlanError(
            f"session {source_uri!r}: block {block_name!r} 'from:' requires "
            f"string 'pack' and 'session' values"
        )

    pack_plan = _load_pack_session(pack=pack, session=session, lookup=pack_session_lookup)
    pack_block_obj = getattr(pack_plan.manifest, block_name, None)
    if pack_block_obj is None:
        raise SessionPlanError(
            f"session {source_uri!r}: block {block_name!r} pulled from "
            f"pack '{pack}' session '{session}', but that pack session does "
            f"not declare a '{block_name}' block"
        )
    # Pydantic model_dump preserves tuples as lists (JSON-compatible), which
    # is what the kernel compiler re-validates against SessionManifestV2.
    dumped = pack_block_obj.model_dump(mode="json")
    assert isinstance(dumped, dict), "model_dump of a v2 block must be a dict"
    return dumped


def _resolve_body(
    *,
    front_matter: dict[str, Any],
    user_body: str,
    source_uri: str,
    pack_session_lookup: PackSessionLookup,
) -> tuple[str, str]:
    """Resolve the ``body:`` directive.

    Returns ``(resolved_body, mode)``. ``mode`` is one of ``write``,
    ``pull``, ``context`` and is passed to :func:`_emit_text` so it can apply
    the kernel compiler's non-empty-body rule correctly.
    """

    body_directive = front_matter.get("body")
    if body_directive is None:
        # No directive: standard kernel path. The user body must be non-empty
        # (the kernel enforces this). We do not relax it here.
        return user_body.strip(), "write"

    if not isinstance(body_directive, dict):
        raise SessionPlanError(f"session {source_uri!r}: 'body:' directive must be a mapping")
    mode = body_directive.get("mode", "write")
    if mode not in ("write", "pull", "context"):
        raise SessionPlanError(
            f"session {source_uri!r}: 'body:' mode must be one of "
            f"'write', 'pull', 'context' (got {mode!r})"
        )

    if mode == "write":
        # Explicit write is identical to the no-directive path.
        return user_body.strip(), "write"

    spec = body_directive.get("from")
    if not isinstance(spec, dict):
        raise SessionPlanError(
            f"session {source_uri!r}: 'body: mode: {mode}' requires a 'from:' "
            f"mapping with 'pack' and 'session' keys"
        )
    pack = spec.get("pack")
    session = spec.get("session")
    if not isinstance(pack, str) or not isinstance(session, str):
        raise SessionPlanError(
            f"session {source_uri!r}: 'body: from:' requires string 'pack' and 'session' values"
        )
    pack_plan = _load_pack_session(pack=pack, session=session, lookup=pack_session_lookup)
    pack_body = pack_plan.instructions.primary_body
    if not pack_body.strip():
        # Should be impossible — compile_session_plan rejects empty bodies —
        # but guard defensively against a future change.
        raise SessionPlanError(
            f"session {source_uri!r}: pack '{pack}' session '{session}' has "
            f"an empty body; cannot use it for body: mode: {mode}"
        )

    if mode == "pull":
        # The pack body replaces the user body. The user's body may be empty.
        return pack_body, "pull"

    # mode == "context": prepend the pack body, wrapped, then the user body.
    user_stripped = user_body.strip()
    if not user_stripped:
        raise SessionPlanError(
            f"session {source_uri!r}: 'body: mode: context' requires a "
            f"non-empty user body to follow the pack session body"
        )
    wrapped = f"<{_PACK_SESSION_BODY_TAG}>\n{pack_body}\n</{_PACK_SESSION_BODY_TAG}>"
    return f"{wrapped}\n{user_stripped}", "context"


# Per-resolver-call cache of compiled pack sessions. Keyed by (pack, session).
# A single user session may reference the same (pack, session) across several
# block leaves plus the body directive; compiling once per call is enough.
_PackSessionCacheKey = tuple[str, str]


def _load_pack_session(
    *,
    pack: str,
    session: str,
    lookup: PackSessionLookup,
    _cache: dict[_PackSessionCacheKey, Any] | None = None,
) -> Any:
    """Load and compile one pack session, memoizing within a resolver call."""

    if _cache is None:
        _cache = {}
    key = (pack, session)
    cached = _cache.get(key)
    if cached is not None:
        return cached
    path = lookup(pack=pack, session=session)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SessionPlanError(f"from: could not read pack session {path}: {exc}") from exc
    # Compiling through the strict kernel schema enforces "packs are leaves":
    # a pack session containing `from:` or `body:` fails here because the
    # schema's `extra="forbid"` rejects unknown keys.
    plan = compile_session_plan(text=text, source_uri=str(path.resolve()))
    _cache[key] = plan
    return plan


def _emit_text(front_matter: dict[str, Any], body: str, mode: str) -> str:
    """Re-emit front matter + body as a single Markdown string.

    ``mode`` is informational here; the body has already been assembled by
    :func:`_resolve_body`. We only need to satisfy the kernel compiler's
    non-empty-body rule for the ``write`` path.
    """

    if mode == "write" and not body.strip():
        # Preserve the kernel's existing behavior: an authored body must be
        # non-empty. Let the kernel raise the canonical error with the
        # source URI context by re-emitting the (possibly directive-free)
        # text and letting compile_session_plan handle it.
        pass
    fm_yaml = yaml.safe_dump(front_matter, sort_keys=False, allow_unicode=True)
    return f"---\n{fm_yaml}---\n{body}\n"


__all__ = ["PackSessionLookup", "resolve_session_from_directives"]

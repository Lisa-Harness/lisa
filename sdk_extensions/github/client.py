"""Read-only GitHub repo investigation client.

This module powers exact code lookup against the configured GitHub repository
without introducing any write capability. It reuses the existing GitHub token
and normalizes search results into the repo lookup response contract already
used elsewhere in the app.
"""

from __future__ import annotations

import base64
import binascii
import re
from collections.abc import Sequence
from typing import Literal

import httpx

from kernel.models import RepoLookupEvidence, RepoLookupResponse
from sdk_extensions.github.config import get_github_extension_settings
from sdk_extensions.github.models import (
    GitHubCodeSearchMatch,
    GitHubCodeSearchResponse,
    GitHubFileExcerpt,
    GitHubTreeEntry,
    GitHubTreeResponse,
)

_GITHUB_API_BASE = "https://api.github.com"
_MAX_QUERY_LENGTH = 240
_MAX_PATH_HINTS = 4
_MAX_RESULTS = 8
_MAX_TEXT_BYTES = 200_000
_SNIPPET_RADIUS = 2
_MAX_TREE_ENTRIES = 80
_DEFAULT_EXCERPT_LINES = 80
_MAX_EXCERPT_LINES = 200


class GitHubRepoLookupError(RuntimeError):
    """Raised when one repo lookup request is invalid or cannot be fulfilled."""


def is_configured() -> bool:
    """Return whether the GitHub repo lookup client has enough configuration."""
    github_settings = get_github_extension_settings()
    return bool(github_settings.token and github_settings.repo)


def _headers() -> dict[str, str]:
    github_settings = get_github_extension_settings()
    token = github_settings.token
    repo = github_settings.repo
    if not token or not repo:
        raise GitHubRepoLookupError(
            "GitHub repo lookup is not configured. Set GITHUB_TOKEN and GITHUB_REPO."
        )

    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.text-match+json, application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _normalize_query(query: str) -> str:
    normalized = " ".join(query.split())[:_MAX_QUERY_LENGTH].strip()
    if not normalized:
        raise GitHubRepoLookupError("Repo lookup requires a non-empty query.")
    return normalized


def _normalize_path_hints(path_hints: Sequence[str] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_hint in path_hints or []:
        hint = raw_hint.strip().strip("/")
        if not hint or hint in seen:
            continue
        if "\n" in hint or "\r" in hint:
            continue
        normalized.append(hint)
        seen.add(hint)
        if len(normalized) >= _MAX_PATH_HINTS:
            break
    return normalized


def _normalize_repo_path(path: str) -> str:
    normalized = path.strip().strip("/").replace("\\", "/")
    if not normalized or "\n" in normalized or "\r" in normalized:
        raise GitHubRepoLookupError("Repo path must be a non-empty single path.")
    return normalized


def _normalize_optional_repo_path(path: str) -> str:
    normalized = path.strip().strip("/").replace("\\", "/")
    if not normalized:
        return ""
    if "\n" in normalized or "\r" in normalized:
        raise GitHubRepoLookupError("Repo path must be a single path.")
    return normalized


def _query_terms(query: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for term in re.findall(r"[A-Za-z0-9_./-]+", query.lower()):
        if len(term) < 2 or term in seen:
            continue
        terms.append(term)
        seen.add(term)
        if len(terms) >= 8:
            break
    return terms


def _snippet_from_text(
    text: str,
    *,
    query: str,
    fragment: str,
) -> tuple[str, int | None, int | None, str]:
    lines = text.splitlines()
    if not lines:
        return "", None, None, ""

    query_terms = _query_terms(query)
    fragment_terms = _query_terms(fragment)
    best_index = -1
    best_score = -1

    for index, line in enumerate(lines):
        lower = line.lower()
        score = sum(1 for term in query_terms if term in lower)
        if score == 0 and fragment_terms:
            score = sum(1 for term in fragment_terms if term in lower)
        if score > best_score:
            best_index = index
            best_score = score

    if best_index < 0:
        best_index = next((index for index, line in enumerate(lines) if line.strip()), 0)

    start = max(best_index - _SNIPPET_RADIUS, 0)
    end = min(best_index + _SNIPPET_RADIUS + 1, len(lines))
    snippet = "\n".join(lines[start:end]).strip()
    match_line = lines[best_index].strip()
    return snippet, start + 1, end, match_line[:160]


def _decode_content(payload: dict[str, object]) -> str:
    encoding = str(payload.get("encoding") or "").lower()
    if encoding != "base64":
        raise GitHubRepoLookupError(f"Unsupported GitHub content encoding: {encoding or 'unknown'}")

    encoded_content = str(payload.get("content") or "")
    try:
        raw = base64.b64decode(encoded_content, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise GitHubRepoLookupError("GitHub content payload was not valid base64.") from exc

    if len(raw) > _MAX_TEXT_BYTES:
        raise GitHubRepoLookupError("GitHub content payload exceeded the lookup size limit.")
    if b"\x00" in raw:
        raise GitHubRepoLookupError("GitHub content payload appears to be binary.")

    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GitHubRepoLookupError("GitHub content payload was not valid UTF-8 text.") from exc


async def _get_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str],
    params: dict[str, str | int] | None = None,
) -> dict[str, object]:
    github_settings = get_github_extension_settings()
    response = await client.get(
        url,
        headers=headers,
        params=params,
        timeout=github_settings.timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict):
        return payload
    raise GitHubRepoLookupError("GitHub API response was not a JSON object.")


async def _resolve_ref(
    client: httpx.AsyncClient,
    *,
    headers: dict[str, str],
    ref: str,
) -> tuple[str, str, str]:
    github_settings = get_github_extension_settings()
    repo = github_settings.repo
    repo_payload = await _get_json(client, f"{_GITHUB_API_BASE}/repos/{repo}", headers=headers)
    default_branch = str(repo_payload.get("default_branch") or "main")

    resolved_ref = ref.strip() or github_settings.branch or default_branch
    commit_payload = await _get_json(
        client,
        f"{_GITHUB_API_BASE}/repos/{repo}/commits/{resolved_ref}",
        headers=headers,
    )
    git_sha = str(commit_payload.get("sha") or "")
    commit_data = commit_payload.get("commit") or {}
    if isinstance(commit_data, dict):
        tree_data = commit_data.get("tree") or {}
    else:
        tree_data = {}
    tree_sha = str(tree_data.get("sha") or "") if isinstance(tree_data, dict) else ""
    return resolved_ref, git_sha, tree_sha


def _search_query(query: str, *, path_hint: str = "") -> str:
    repo = get_github_extension_settings().repo
    if path_hint:
        return f"{query} repo:{repo} path:{path_hint}"
    return f"{query} repo:{repo}"


def _extract_fragment(item: dict[str, object]) -> str:
    text_matches = item.get("text_matches") or []
    if not isinstance(text_matches, list):
        return ""
    for match in text_matches:
        if isinstance(match, dict) and str(match.get("fragment") or "").strip():
            return str(match.get("fragment") or "").strip()
    return ""


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


async def search_code(
    query: str,
    *,
    path_hints: Sequence[str] | None = None,
    max_results: int = _MAX_RESULTS,
    ref: str = "",
    client: httpx.AsyncClient | None = None,
) -> GitHubCodeSearchResponse:
    """Search the configured GitHub repo read-only and return normalized matches."""

    normalized_query = _normalize_query(query)
    normalized_hints = _normalize_path_hints(path_hints)
    bounded_results = max(1, min(max_results, _MAX_RESULTS))
    headers = _headers()

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient()

    try:
        resolved_ref, git_sha, _tree_sha = await _resolve_ref(client, headers=headers, ref=ref)
        seen_paths: set[str] = set()
        matches: list[GitHubCodeSearchMatch] = []

        async def _collect(search_query: str) -> None:
            search_payload = await _get_json(
                client,
                f"{_GITHUB_API_BASE}/search/code",
                headers=headers,
                params={"q": search_query, "per_page": bounded_results},
            )
            items = search_payload.get("items") or []
            if not isinstance(items, list):
                return

            for item in items:
                if not isinstance(item, dict):
                    continue
                path = str(item.get("path") or "").strip()
                if not path or path in seen_paths:
                    continue
                seen_paths.add(path)
                matches.append(
                    GitHubCodeSearchMatch(
                        path=path,
                        fragment=_extract_fragment(item),
                    )
                )
                if len(matches) >= bounded_results:
                    break

        for path_hint in normalized_hints:
            await _collect(_search_query(normalized_query, path_hint=path_hint))
            if len(matches) >= bounded_results:
                break

        if not matches:
            await _collect(_search_query(normalized_query))

        return GitHubCodeSearchResponse(
            repo=get_github_extension_settings().repo.lower(),
            ref=resolved_ref,
            git_sha=git_sha,
            query=normalized_query,
            path_hints=normalized_hints,
            matches=matches,
        )
    finally:
        if owns_client:
            await client.aclose()


async def read_file_excerpt(
    path: str,
    *,
    ref: str = "",
    start_line: int = 1,
    end_line: int = 0,
    anchor_line: int = 0,
    context_radius: int = 20,
    client: httpx.AsyncClient | None = None,
) -> GitHubFileExcerpt:
    """Fetch a read-only file excerpt from the configured GitHub repo by ref."""

    normalized_path = _normalize_repo_path(path)
    headers = _headers()

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient()

    try:
        resolved_ref, git_sha, _tree_sha = await _resolve_ref(client, headers=headers, ref=ref)
        payload = await _get_json(
            client,
            f"{_GITHUB_API_BASE}/repos/{get_github_extension_settings().repo}/contents/{normalized_path}",
            headers=headers,
            params={"ref": resolved_ref},
        )
        text = _decode_content(payload)
        lines = text.splitlines()
        total_lines = len(lines)

        bounded_start = max(start_line, 1)
        bounded_end = end_line
        if anchor_line > 0:
            bounded_anchor = min(max(anchor_line, 1), max(total_lines, 1))
            bounded_start = max(1, bounded_anchor - max(context_radius, 0))
            bounded_end = bounded_anchor + max(context_radius, 0)
        elif bounded_end <= 0:
            bounded_end = bounded_start + _DEFAULT_EXCERPT_LINES - 1

        if total_lines > 0:
            bounded_start = min(bounded_start, total_lines)
            bounded_end = min(max(bounded_end, bounded_start), total_lines)
        else:
            bounded_end = bounded_start

        if bounded_end - bounded_start + 1 > _MAX_EXCERPT_LINES:
            bounded_end = bounded_start + _MAX_EXCERPT_LINES - 1

        excerpt_lines = [
            f"{line_number}: {lines[line_number - 1]}"
            for line_number in range(bounded_start, bounded_end + 1)
            if 1 <= line_number <= total_lines
        ]

        return GitHubFileExcerpt(
            repo=get_github_extension_settings().repo.lower(),
            ref=resolved_ref,
            git_sha=git_sha,
            path=normalized_path,
            start_line=bounded_start,
            end_line=bounded_end,
            total_lines=total_lines,
            excerpt="\n".join(excerpt_lines),
        )
    finally:
        if owns_client:
            await client.aclose()


async def list_tree(
    root: str = "",
    *,
    ref: str = "",
    max_entries: int = 40,
    client: httpx.AsyncClient | None = None,
) -> GitHubTreeResponse:
    """List direct child entries under one repo root or subdirectory."""

    normalized_root = _normalize_optional_repo_path(root)
    bounded_entries = max(1, min(max_entries, _MAX_TREE_ENTRIES))
    headers = _headers()

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient()

    try:
        resolved_ref, git_sha, tree_sha = await _resolve_ref(client, headers=headers, ref=ref)
        if not tree_sha:
            raise GitHubRepoLookupError("Could not resolve the tree for the requested Git ref.")

        payload = await _get_json(
            client,
            f"{_GITHUB_API_BASE}/repos/{get_github_extension_settings().repo}/git/trees/{tree_sha}",
            headers=headers,
            params={"recursive": 1},
        )
        raw_entries = payload.get("tree") or []
        if not isinstance(raw_entries, list):
            raise GitHubRepoLookupError("GitHub tree payload was not a list.")

        prefix = f"{normalized_root}/" if normalized_root else ""
        direct_entries: dict[str, GitHubTreeEntry] = {}

        for item in raw_entries:
            if not isinstance(item, dict):
                continue
            raw_path = str(item.get("path") or "").strip()
            if not raw_path:
                continue
            if normalized_root:
                if raw_path == normalized_root or not raw_path.startswith(prefix):
                    continue
                remaining = raw_path[len(prefix):]
            else:
                remaining = raw_path
            if not remaining:
                continue

            head, sep, _tail = remaining.partition("/")
            direct_path = f"{prefix}{head}" if prefix else head
            if not direct_path:
                continue

            existing = direct_entries.get(direct_path)
            if sep:
                direct_entries[direct_path] = GitHubTreeEntry(path=direct_path, entry_type="tree")
                continue

            entry_type: Literal["blob", "tree"] = "tree"
            if str(item.get("type") or "blob") != "tree":
                entry_type = "blob"
            if existing is not None and existing.entry_type == "tree":
                continue
            direct_entries[direct_path] = GitHubTreeEntry(
                path=direct_path,
                entry_type=entry_type,
                size=_coerce_int(item.get("size")) if entry_type == "blob" else None,
            )

        entries = sorted(
            direct_entries.values(),
            key=lambda entry: (entry.entry_type != "tree", entry.path.lower()),
        )[:bounded_entries]

        return GitHubTreeResponse(
            repo=get_github_extension_settings().repo.lower(),
            ref=resolved_ref,
            git_sha=git_sha,
            root=normalized_root,
            truncated=bool(payload.get("truncated")),
            entries=entries,
        )
    finally:
        if owns_client:
            await client.aclose()


async def lookup_repo(
    query: str,
    *,
    path_hints: Sequence[str] | None = None,
    max_results: int = _MAX_RESULTS,
    ref: str = "",
    request_id: str = "direct",
    client: httpx.AsyncClient | None = None,
) -> RepoLookupResponse:
    """Search the configured GitHub repo and return grounded exact-code evidence."""

    normalized_query = _normalize_query(query)
    normalized_hints = _normalize_path_hints(path_hints)
    bounded_results = max(1, min(max_results, _MAX_RESULTS))
    headers = _headers()

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient()

    try:
        resolved_ref, git_sha, _tree_sha = await _resolve_ref(client, headers=headers, ref=ref)
        seen_paths: set[str] = set()
        evidence: list[RepoLookupEvidence] = []

        for path_hint in normalized_hints:
            search_query = _search_query(normalized_query, path_hint=path_hint)
            search_payload = await _get_json(
                client,
                f"{_GITHUB_API_BASE}/search/code",
                headers=headers,
                params={"q": search_query, "per_page": bounded_results},
            )
            items = search_payload.get("items") or []
            if not isinstance(items, list):
                continue

            for item in items:
                if not isinstance(item, dict):
                    continue
                path = str(item.get("path") or "").strip()
                if not path or path in seen_paths:
                    continue

                seen_paths.add(path)
                fragment = _extract_fragment(item)

                try:
                    content_payload = await _get_json(
                        client,
                        f"{_GITHUB_API_BASE}/repos/{get_github_extension_settings().repo}/contents/{path}",
                        headers=headers,
                        params={"ref": resolved_ref},
                    )
                    text = _decode_content(content_payload)
                    snippet, line_start, line_end, summary_line = _snippet_from_text(
                        text,
                        query=normalized_query,
                        fragment=fragment,
                    )
                except GitHubRepoLookupError:
                    continue

                evidence.append(
                    RepoLookupEvidence(
                        path=path,
                        summary=summary_line or fragment[:160] or f"Matched query in {path}",
                        snippet=snippet,
                        line_start=line_start,
                        line_end=line_end,
                    )
                )
                if len(evidence) >= bounded_results:
                    break

            if len(evidence) >= bounded_results:
                break

        if not evidence:
            search_payload = await _get_json(
                client,
                f"{_GITHUB_API_BASE}/search/code",
                headers=headers,
                params={"q": _search_query(normalized_query), "per_page": bounded_results},
            )
            items = search_payload.get("items") or []
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    path = str(item.get("path") or "").strip()
                    if not path or path in seen_paths:
                        continue

                    seen_paths.add(path)
                    fragment = _extract_fragment(item)

                    try:
                        content_payload = await _get_json(
                            client,
                            f"{_GITHUB_API_BASE}/repos/{get_github_extension_settings().repo}/contents/{path}",
                            headers=headers,
                            params={"ref": resolved_ref},
                        )
                        text = _decode_content(content_payload)
                        snippet, line_start, line_end, summary_line = _snippet_from_text(
                            text,
                            query=normalized_query,
                            fragment=fragment,
                        )
                    except GitHubRepoLookupError:
                        continue

                    evidence.append(
                        RepoLookupEvidence(
                            path=path,
                            summary=summary_line or fragment[:160] or f"Matched query in {path}",
                            snippet=snippet,
                            line_start=line_start,
                            line_end=line_end,
                        )
                    )
                    if len(evidence) >= bounded_results:
                        break

        if not evidence:
            return RepoLookupResponse(
                request_id=request_id,
                status="not_found",
                repo=get_github_extension_settings().repo.lower(),
                git_sha=git_sha,
                answer_summary=(
                    f"No grounded repo matches found for {normalized_query!r} on {resolved_ref}."
                ),
                confidence=0.0,
            )

        confidence = min(0.95, 0.45 + (0.1 * len(evidence)))
        return RepoLookupResponse(
            request_id=request_id,
            status="answered",
            repo=get_github_extension_settings().repo.lower(),
            git_sha=git_sha,
            answer_summary=(
                f"Found {len(evidence)} grounded repo matches for {normalized_query!r} "
                f"on {resolved_ref} ({git_sha[:12]})."
            ),
            confidence=confidence,
            evidence=evidence,
        )
    finally:
        if owns_client:
            await client.aclose()

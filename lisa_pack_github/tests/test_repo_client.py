"""Ref-grounding tests for the GitHub repository investigation client."""

from __future__ import annotations

import base64

import httpx
import pytest
from pydantic import SecretStr

from lisa_pack_github.client import GitHubRepoLookupError, lookup_repo, search_code


def _configure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lisa_pack_github.config.settings.github_token", SecretStr("token"))
    monkeypatch.setattr("lisa_pack_github.config.settings.github_repo", "owner/repo")
    monkeypatch.setattr("lisa_pack_github.config.settings.github_branch", "")


def _response(request: httpx.Request, payload: dict[str, object]) -> httpx.Response:
    return httpx.Response(200, json=payload, request=request)


@pytest.mark.asyncio
async def test_code_search_rejects_non_default_requested_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Indexed GitHub code search cannot be presented as a feature-ref search."""

    _configure(monkeypatch)
    search_called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal search_called
        if request.url.path == "/repos/owner/repo":
            return _response(request, {"default_branch": "main"})
        if request.url.path == "/repos/owner/repo/commits/feature":
            return _response(
                request,
                {"sha": "feature-sha", "commit": {"tree": {"sha": "tree"}}},
            )
        if request.url.path == "/search/code":
            search_called = True
        return httpx.Response(404, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GitHubRepoLookupError, match="default branch"):
            await search_code("needle", ref="feature", client=client)

    assert search_called is False


@pytest.mark.asyncio
async def test_lookup_discards_nonmatching_requested_ref_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Search candidates are not evidence until their target-ref content matches."""

    _configure(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/repos/owner/repo":
            return _response(request, {"default_branch": "main"})
        if request.url.path == "/repos/owner/repo/commits/feature":
            return _response(
                request,
                {"sha": "feature-sha", "commit": {"tree": {"sha": "tree"}}},
            )
        if request.url.path == "/search/code":
            return _response(
                request,
                {
                    "items": [
                        {
                            "path": "src/removed.py",
                            "text_matches": [{"fragment": "needle"}],
                        }
                    ]
                },
            )
        if request.url.path == "/repos/owner/repo/contents/src/removed.py":
            content = base64.b64encode(b"feature content without the query").decode("ascii")
            return _response(request, {"encoding": "base64", "content": content})
        return httpx.Response(404, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await lookup_repo("needle", ref="feature", client=client)

    assert result.status == "not_found"
    assert result.evidence == []
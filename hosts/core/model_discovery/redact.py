"""Redaction and safe-error formatting for model discovery.

Single responsibility: turn any ``httpx``/generic exception into a
UI-safe string. No exception ``str()`` ever reaches the UI through this
module. This is the durable fix for the credential-leak class of bugs
where ``httpx.HTTPStatusError.__str__`` embeds the full request URL
(which, for Google's Gemini REST endpoint, contained the API key as a
``?key=...`` query parameter).
"""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlparse

# Query-string parameter names that carry secrets and must be redacted
# from any URL that might end up in a log or UI string.
_KEY_QUERY_PARAMS = frozenset(
    (
        "key",
        "api_key",
        "apikey",
        "access_token",
        "token",
        "secret",
    )
)


def redact_url(url: str) -> str:
    """Return ``url`` with any secret-bearing query params replaced.

    Non-secret params are preserved. If the URL has no query string it
    is returned unchanged.
    """

    parsed = urlparse(url)
    if not parsed.query:
        return url
    pairs = []
    for key, value in parse_qsl(parsed.query):
        if key.lower() in _KEY_QUERY_PARAMS:
            pairs.append((key, "***REDACTED***"))
        else:
            pairs.append((key, value))
    return parsed._replace(query=urlencode(pairs)).geturl()


def safe_http_error(exc: Exception, provider: str) -> str:
    """Return a UI-safe one-line description of ``exc``.

    Never includes the raw request URL, headers, or body. The provider
    name is included so the user knows which provider failed, but no
    host or path is leaked.
    """

    # Imported lazily so this module has no hard dependency on httpx
    # for callers that only want ``redact_url``.
    try:
        import httpx  # pylint: disable=import-outside-toplevel
    except ImportError:  # pragma: no cover - httpx is a core dep
        httpx = None  # type: ignore[assignment]

    if httpx is not None:
        if isinstance(exc, httpx.HTTPStatusError):
            response = exc.response
            reason = response.reason_phrase or "Unknown"
            return f"{provider}: HTTP {response.status_code} {reason}"
        if isinstance(exc, httpx.ConnectError):
            return f"{provider}: connection failed (check endpoint/host)"
        if isinstance(exc, httpx.TimeoutException):
            return f"{provider}: request timed out"
        if isinstance(exc, httpx.HTTPError):  # base httpx error
            return f"{provider}: HTTP transport error"
    if isinstance(exc, (ValueError, KeyError, TypeError)):
        return f"{provider}: unexpected response shape"
    return f"{provider}: unexpected error (see logs)"


def classify_status(exc: Exception) -> "str":
    """Map an exception to a :class:`ProbeStatus` value name.

    Returns the enum *name* (string) so this function has no hard import
    dependency on ``types`` (avoids a cycle). Callers wrap with
    ``ProbeStatus(...)``.
    """

    try:
        import httpx  # pylint: disable=import-outside-toplevel
    except ImportError:  # pragma: no cover
        httpx = None  # type: ignore[assignment]

    if httpx is not None:
        if isinstance(exc, httpx.ConnectError):
            return "network"
        if isinstance(exc, httpx.TimeoutException):
            return "network"
        if isinstance(exc, httpx.HTTPStatusError):
            code = exc.response.status_code
            if code in (401, 403):
                return "auth"
            if code == 404:
                return "not_found"
            if code == 429:
                return "rate_limited"
            return "unknown"
    if isinstance(exc, (ValueError, KeyError, TypeError)):
        return "schema"
    return "unknown"
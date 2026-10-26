"""Lifecycle management and provider-route discovery for local proxies."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from kernel.proxy_launch.compiler import ProxyClientConfig

__all__ = ["ProxyConnectionError", "ProxyPool", "ProxyRoute"]

logger = logging.getLogger(__name__)


class ProxyConnectionError(RuntimeError):
    """Raised when a required proxy cannot be started or adopted."""


@dataclass(frozen=True, slots=True)
class ProxyRoute:
    """A ready proxy endpoint selected for a provider gateway."""

    proxy_id: str
    protocol: str
    base_url: str


@dataclass(frozen=True, slots=True)
class ProxyHealth:
    """One proxy's liveness and routing view, for host introspection.

    ``stats`` holds the pack-declared live counters (``status_url_path``)
    fetched best-effort at gather time; ``None`` when the proxy declares
    no stats endpoint or the fetch failed.
    """

    proxy_id: str
    status: str  # warm | failed | skipped
    protocol: str
    providers: tuple[str, ...]
    base_url: str
    warm: bool
    required: bool
    status_url_path: str | None
    status_fields: tuple[str, ...]
    stats: dict[str, Any] | None
    last_error: str = ""


class ProxyPool:
    """Own warm local proxy processes and expose routes after readiness.

    Existing endpoints are adopted only when the declaration explicitly opts
    in. The default is host ownership because a live external process cannot
    receive the manifest's required upstream configuration retroactively.
    """

    _STARTUP_ATTEMPTS = 3
    _BACKOFF_SECONDS = (0.25, 0.5)

    def __init__(self, configs: list[ProxyClientConfig]) -> None:
        self._configs = list(configs)
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._routes: dict[str, ProxyRoute] = {}
        self._failed: set[str] = set()
        self._started = False

    @property
    def routes(self) -> dict[str, ProxyRoute]:
        """Return a copy of ready provider routes keyed by provider id."""
        return dict(self._routes)

    @property
    def failed_proxy_ids(self) -> frozenset[str]:
        """Return non-required proxy ids that failed to start."""
        return frozenset(self._failed)

    def health(self) -> list[ProxyHealth]:
        """Return a liveness view of every proxy this pool was asked to start.

        Includes skipped proxies (optional binaries absent at compile time
        never reach the pool; this covers compile-present failures only).
        The view is synchronous — no stats fetching — so callers can use
        it from render paths that must not await.
        """
        ready_ids = {route.proxy_id for route in self._routes.values()}
        views: list[ProxyHealth] = []
        for config in self._configs:
            if config.proxy_id in ready_ids:
                status = "warm"
            elif config.proxy_id in self._failed:
                status = "failed"
            else:
                status = "skipped"
            views.append(
                ProxyHealth(
                    proxy_id=config.proxy_id,
                    status=status,
                    protocol=config.protocol,
                    providers=config.providers,
                    base_url=config.base_url,
                    warm=config.warm,
                    required=config.required,
                    status_url_path=config.status_url_path,
                    status_fields=config.status_fields,
                    stats=None,
                )
            )
        return views

    async def fetch_stats(self, proxy_id: str) -> dict[str, Any] | None:
        """Fetch the pack-declared live stats for one proxy (best effort).

        Each declared field is a dotted path (``savings.total_tokens``)
        resolved against the JSON body. A path naming a scalar contributes
        one entry keyed by the path; a path naming an object contributes
        its scalar members keyed ``<path>.<member>``. Paths that miss or
        name lists are skipped. Returns ``None`` when the proxy declares
        no stats endpoint, is not warm, or the fetch fails. Never raises.
        """
        config = self._warm_config(proxy_id)
        if config is None or not config.status_url_path:
            return None
        payload = await self._get_json(config.port, config.status_url_path)
        if payload is None:
            return None
        if not config.status_fields:
            return payload
        return _select_stats_fields(payload, config.status_fields)

    def _warm_config(self, proxy_id: str) -> ProxyClientConfig | None:
        """Return the config for a warm proxy, or ``None`` when not ready."""
        ready_ids = {route.proxy_id for route in self._routes.values()}
        if proxy_id not in ready_ids:
            return None
        return next((c for c in self._configs if c.proxy_id == proxy_id), None)

    @staticmethod
    async def _get_json(port: int, path: str) -> dict[str, Any] | None:
        """GET one loopback JSON document, or ``None`` on any failure."""
        url = f"http://127.0.0.1:{port}{path}"
        try:
            async with httpx.AsyncClient(timeout=1.0) as client:
                response = await client.get(url)
        except httpx.HTTPError:
            return None
        if not response.is_success:
            return None
        try:
            payload = response.json()
        except ValueError:
            return None
        return payload if isinstance(payload, dict) else None

    def route_for(self, provider: str, protocol: str) -> ProxyRoute | None:
        """Return a ready route only when its wire protocol matches."""
        route = self._routes.get(provider.strip().lower())
        if route is None or route.protocol != protocol:
            return None
        return route

    async def start(self) -> None:
        """Start or adopt all warm proxies before provider clients are built."""
        if self._started:
            return
        warm_configs = [config for config in self._configs if config.warm]
        results = await asyncio.gather(
            *(self._start_one(config) for config in warm_configs),
            return_exceptions=True,
        )
        for config, result in zip(warm_configs, results, strict=True):
            if isinstance(result, BaseException):
                if config.required:
                    await self.aclose()
                    raise ProxyConnectionError(
                        f"Required proxy '{config.proxy_id}' failed to start: {result}"
                    ) from result
                logger.warning("Proxy %r failed to start: %s", config.proxy_id, result)
                self._failed.add(config.proxy_id)
                continue
            self._register_routes(config)
        self._started = True

    async def _start_one(self, config: ProxyClientConfig) -> None:
        if config.adopt_existing and await self._is_ready(config):
            logger.info("Adopting ready proxy %r at %s", config.proxy_id, config.base_url)
            return

        last_error: Exception | None = None
        for attempt in range(self._STARTUP_ATTEMPTS):
            # An empty env map must inherit the parent environment (same
            # semantics as ``env=None``): on Windows an explicit empty dict
            # strips ``SystemRoot`` and the child's winsock fails to
            # initialize (WinError 10106) — observed on CI runners.
            child_env = config.env or None
            process = await asyncio.create_subprocess_exec(
                config.command,
                *config.args,
                env=child_env,
                cwd=config.cwd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                await self._wait_ready(config, process)
            except Exception as exc:  # pylint: disable=broad-except
                last_error = exc
                await self._terminate(process)
                if attempt < self._STARTUP_ATTEMPTS - 1:
                    await asyncio.sleep(self._BACKOFF_SECONDS[attempt])
                continue
            self._processes[config.proxy_id] = process
            logger.info("Started proxy %r at %s", config.proxy_id, config.base_url)
            return
        raise ProxyConnectionError(
            f"Proxy '{config.proxy_id}' was not ready after "
            f"{self._STARTUP_ATTEMPTS} attempts: {last_error}"
        )

    async def _wait_ready(
        self,
        config: ProxyClientConfig,
        process: asyncio.subprocess.Process,
    ) -> None:
        deadline = asyncio.get_running_loop().time() + config.readiness_timeout_seconds
        while asyncio.get_running_loop().time() < deadline:
            if process.returncode is not None:
                stderr = await process.stderr.read() if process.stderr is not None else b""
                detail = stderr.decode("utf-8", errors="replace").strip()
                raise ProxyConnectionError(
                    f"Proxy '{config.proxy_id}' exited with {process.returncode}: {detail}"
                )
            if await self._is_ready(config):
                return
            await asyncio.sleep(0.1)
        raise ProxyConnectionError(
            f"Proxy '{config.proxy_id}' did not become ready at {config.base_url}"
        )

    async def _is_ready(self, config: ProxyClientConfig) -> bool:
        if config.readiness_path is None:
            return False
        url = f"http://127.0.0.1:{config.port}{config.readiness_path}"
        try:
            async with httpx.AsyncClient(timeout=0.5) as client:
                response = await client.get(url)
        except httpx.HTTPError:
            return False
        return response.is_success

    def _register_routes(self, config: ProxyClientConfig) -> None:
        route = ProxyRoute(
            proxy_id=config.proxy_id,
            protocol=config.protocol,
            base_url=config.base_url,
        )
        for provider in config.providers:
            previous = self._routes.get(provider)
            if previous is not None and previous.proxy_id != config.proxy_id:
                logger.warning(
                    "Proxy %r replaces route for provider %r from proxy %r",
                    config.proxy_id,
                    provider,
                    previous.proxy_id,
                )
            self._routes[provider] = route

    async def aclose(self) -> None:
        """Terminate only proxy processes spawned by this pool."""
        processes = list(self._processes.values())
        self._processes.clear()
        self._routes.clear()
        self._started = False
        await asyncio.gather(
            *(self._terminate(process) for process in processes),
            return_exceptions=True,
        )

    async def _terminate(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except TimeoutError:
            process.kill()
            await process.wait()


def _select_stats_fields(
    payload: dict[str, Any],
    fields: tuple[str, ...],
) -> dict[str, Any]:
    """Select declared dotted paths out of one stats JSON body.

    A scalar path contributes one entry; an object path contributes its
    scalar members as ``<path>.<member>``. Absent paths and non-scalar,
    non-object paths are skipped.
    """
    selected: dict[str, Any] = {}
    for field in fields:
        exists, value = _resolve_path(payload, field)
        if not exists:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            selected[field] = value
        elif isinstance(value, dict):
            for member, member_value in value.items():
                if isinstance(member_value, (str, int, float, bool)) or member_value is None:
                    selected[f"{field}.{member}"] = member_value
    return selected


def _resolve_path(payload: dict[str, Any], path: str) -> tuple[bool, Any]:
    """Resolve one dotted path (``a.b.c``) against a nested JSON body.

    Returns ``(exists, value)``: ``exists`` is False when any path segment
    is missing, so callers can distinguish an explicit ``null`` from an
    absent path.
    """
    current: Any = payload
    for segment in path.split("."):
        if not isinstance(current, dict) or segment not in current:
            return False, None
        current = current[segment]
    return True, current

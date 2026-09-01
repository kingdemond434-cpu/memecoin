import asyncio
from src.runtime.loop_local import loop_local_lock, loop_local_semaphore
import logging
import os
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Set
from urllib.parse import urlsplit

import aiohttp
import yaml
from web3 import AsyncWeb3
from web3.providers import AsyncHTTPProvider

logger = logging.getLogger(__name__)


def _host_of(url: str) -> str:
    """The host, for logs. Provider URLs carry the API key in the path or the
    query string, so the whole URL must never reach a log line."""
    try:
        return urlsplit(url).netloc or "unknown-host"
    except ValueError:
        return "unknown-host"


class ChainType(Enum):
    EVM = "evm"
    SOLANA = "solana"


@dataclass
class RPCEndpointConfig:
    url: str
    ws_url: Optional[str] = None
    weight: int = 1
    max_concurrent: int = 50
    timeout: float = 10.0
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class ChainConfig:
    name: str
    chain_id: Any
    chain_type: ChainType
    rpc_endpoints: List[RPCEndpointConfig]
    explorer_api: str
    explorer_key: str
    native_token: str
    decimals: int
    block_time: float
    factories: Dict[str, str]
    routers: Dict[str, str]
    base_tokens: List[str]
    min_liquidity_usd: float
    max_tax: float
    honeypot_check: bool
    programs: Dict[str, str] = field(default_factory=dict)


class RPCHealth(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


@dataclass
class EndpointHealth:
    endpoint: RPCEndpointConfig
    health: RPCHealth = RPCHealth.HEALTHY
    latency_ms: float = 0.0
    error_count: int = 0
    success_count: int = 0
    last_check: float = 0.0
    consecutive_failures: int = 0
    #: Wall-clock before which this endpoint must not be selected. A provider
    #: answering 429 has told us its quota is spent; re-selecting it inside
    #: that window spends nothing but the remaining quota of the retry budget.
    cooldown_until: float = 0.0
    last_status: int = 0
    #: Methods this endpoint refuses outright, learned at runtime.
    #:
    #: Free providers do not merely rate-limit, they carve out methods:
    #: publicnode answers getLatestBlockhash and getAccountInfo but 403s
    #: getTokenLargestAccounts ("Request blocked"), while leorpc serves
    #: getTokenLargestAccounts and 429s getAccountInfo. They are complementary
    #: rather than redundant, so a refusal has to disqualify an endpoint for
    #: ONE method instead of cooling it for all of them -- otherwise the pool
    #: is only ever as capable as its weakest member, and every call to a
    #: carved-out method burns the whole retry budget rediscovering the same
    #: 403.
    blocked_methods: Set[str] = field(default_factory=set)


class RPCManager:
    def __init__(self, chain_config: ChainConfig, max_concurrent_total: int = 100):
        self.chain_config = chain_config
        self.endpoints: List[EndpointHealth] = [
            EndpointHealth(ep) for ep in chain_config.rpc_endpoints
        ]
        self.max_concurrent_total = max_concurrent_total
        self._session: Optional[aiohttp.ClientSession] = None
        #: loop id -> that loop's session. See _session_for_loop.
        self._sessions: Dict[int, aiohttp.ClientSession] = {}
        self._ws_connections: Dict[str, Any] = {}
        # Loop-local, because the miners run on their own loop and every
        # RPC call they make used to append a waiter to a semaphore bound to
        # the MAIN loop, raise, and leave the waiter behind. 2,437 of them
        # were counted before the OOM killer took the process on 2026-09-01.
        # See src/runtime/loop_local.py.
        self._lock = loop_local_lock("rpc_manager.lock")
        self._health_check_task: Optional[asyncio.Task] = None
        self._request_semaphore = loop_local_semaphore(
            max_concurrent_total, "rpc_manager.requests")

    def _session_for_loop(self):
        """This loop's aiohttp session, created on first use by that loop.

        Same reasoning as the semaphore. A session binds to the loop that
        created it -- its connector, its timers, its timeout contexts -- and
        used from another it raises "Timeout context manager should be used
        inside a task" on the first request. The miners run on their own
        loop and share this manager, so a single session could never have
        served them.
        """
        import asyncio as _asyncio

        try:
            key = id(_asyncio.get_running_loop())
        except RuntimeError:
            key = 0
        existing = self._sessions.get(key)
        if existing is not None and not existing.closed:
            return existing
        created = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            connector=aiohttp.TCPConnector(limit=100, limit_per_host=20),
        )
        self._sessions[key] = created
        return created

    @property
    def session(self):
        """Always the calling loop's session. Every call site goes through here."""
        return self._session_for_loop()

    async def start(self):
        self._session = self._session_for_loop()
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        await self._initial_health_check()

    async def stop(self):
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        for ws in self._ws_connections.values():
            await ws.close()
        for session in list(self._sessions.values()):
            if not session.closed:
                try:
                    await session.close()
                except Exception:  # pragma: no cover - shutdown only
                    pass
        self._sessions.clear()
        if self._session and not self._session.closed:
            await self._session.close()

    async def _initial_health_check(self):
        await asyncio.gather(*[self._check_endpoint(e) for e in self.endpoints])

    async def _health_check_loop(self):
        while True:
            await asyncio.sleep(30)
            await asyncio.gather(*[self._check_endpoint(e) for e in self.endpoints])

    async def _check_endpoint(self, ep: EndpointHealth):
        start = time.time()
        method, params = self._health_probe()
        try:
            async with self.session.post(
                ep.endpoint.url,
                json={"jsonrpc": "2.0", "method": method, "params": params, "id": 1},
                headers=ep.endpoint.headers,
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if "result" in data and self._valid_health_result(data["result"]):
                        ep.health = RPCHealth.HEALTHY
                        ep.latency_ms = (time.time() - start) * 1000
                        ep.success_count += 1
                        ep.consecutive_failures = 0
                        ep.last_status = 200
                        ep.cooldown_until = 0.0
                        ep.last_check = time.time()
                        return
                    self._penalise(ep, resp.status)
                    ep.last_check = time.time()
                    return
                # The probe is also the cheapest place to learn a quota is
                # spent. Recording the cooldown here keeps a rate-limited
                # provider out of selection between probes instead of
                # rediscovering the 429 on every caller's hot path.
                cooldown = 0.0
                if resp.status == 429:
                    cooldown = self._retry_after_seconds(resp, 30.0)
                elif resp.status in (402, 403):
                    cooldown = 60.0
                self._penalise(ep, resp.status, cooldown)
                logger.warning("RPC health probe: %s answered HTTP %s",
                               _host_of(ep.endpoint.url), resp.status)
                ep.last_check = time.time()
                return
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            logger.debug("RPC health probe failed for %s", _host_of(ep.endpoint.url))
        self._penalise(ep, 0)
        ep.last_check = time.time()

    def _health_probe(self) -> tuple[str, List[Any]]:
        """Return a protocol-correct, inexpensive probe for this chain."""
        if self.chain_config.chain_type == ChainType.SOLANA:
            return "getHealth", []
        return "eth_blockNumber", []

    def _valid_health_result(self, result: Any) -> bool:
        if self.chain_config.chain_type == ChainType.SOLANA:
            return result == "ok"
        return isinstance(result, str) and result.startswith("0x")

    @staticmethod
    def _retry_after_seconds(resp: Any, default: float) -> float:
        """Honour a provider's own Retry-After, bounded so it cannot park an
        endpoint for hours on a malformed or hostile header."""
        raw = ""
        try:
            raw = str(resp.headers.get("Retry-After", "")).strip()
        except Exception:
            raw = ""
        if raw:
            try:
                return max(0.0, min(float(raw), 300.0))
            except ValueError:
                pass
        return default

    def _penalise(self, ep: EndpointHealth, status: int, cooldown: float = 0.0) -> None:
        """Record a transport-level refusal against an endpoint.

        A non-200 is a failure. Left uncounted it could not demote an endpoint,
        so a provider answering 429 to every call stayed HEALTHY, kept winning
        selection, and surfaced only as an unattributed "All RPC endpoints
        failed" -- the status that names the actual cause never reached a log.
        """
        ep.error_count += 1
        ep.consecutive_failures += 1
        ep.last_status = status
        if cooldown > 0:
            ep.cooldown_until = max(ep.cooldown_until, time.time() + cooldown)
        if ep.consecutive_failures >= 3:
            ep.health = RPCHealth.DOWN
        else:
            ep.health = RPCHealth.DEGRADED

    def _select_endpoint(self, prefer_ws: bool = False,
                         method: str = "") -> Optional[EndpointHealth]:
        now = time.time()
        candidates = [e for e in self.endpoints
                      if e.health != RPCHealth.DOWN and e.cooldown_until <= now
                      and not (method and method in e.blocked_methods)]
        if not candidates:
            # Everything is either down or cooling. Prefer the endpoint whose
            # cooldown expires soonest over failing the call outright: a stale
            # answer beats no answer for enrichment, and the caller still sees
            # the refusal if that endpoint is still rate limited.
            waiting = [e for e in self.endpoints
                       if e.health != RPCHealth.DOWN
                       and not (method and method in e.blocked_methods)]
            if not waiting:
                # Every endpoint that could serve this method is down or has
                # carved it out. Returning one that answers 403 forever would
                # dress a permanent refusal as a transient failure.
                return None
            candidates = [min(waiting, key=lambda e: e.cooldown_until)]
        if prefer_ws:
            candidates = [e for e in candidates if e.endpoint.ws_url]
        if not candidates:
            return None
        # Never let a credential-rejected/degraded endpoint win merely because
        # it has no measured latency yet. Degraded providers are a last resort
        # only when no healthy provider supports the requested transport.
        healthy = [e for e in candidates if e.health == RPCHealth.HEALTHY]
        candidates = healthy or candidates
        weights = [e.endpoint.weight * (1 / max(e.latency_ms, 1)) for e in candidates]
        return random.choices(candidates, weights=weights, k=1)[0]

    async def request(self, method: str, params: List[Any]) -> Any:
        async with self._request_semaphore:
            last_refusal = ""
            for attempt in range(3):
                ep = self._select_endpoint(method=method)
                if not ep:
                    raise RuntimeError(
                        f"No healthy RPC endpoint serves {method}")
                try:
                    async with self.session.post(
                        ep.endpoint.url,
                        json={"jsonrpc": "2.0", "method": method, "params": params, "id": 1},
                        headers=ep.endpoint.headers,
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if "result" in data:
                                ep.success_count += 1
                                ep.consecutive_failures = 0
                                ep.last_status = 200
                                ep.health = RPCHealth.HEALTHY
                                return data["result"]
                            if "error" in data:
                                raise RPCError(data["error"])
                            raise RPCError(
                                {"message": "response carried neither result nor error"})
                        # A refusal the transport reported rather than raised.
                        # 429 and 5xx are the providers' way of saying "not
                        # now"; both must cost this endpoint its health, and a
                        # spent quota must park it rather than be retried into.
                        cooldown = 0.0
                        if resp.status == 429:
                            cooldown = self._retry_after_seconds(resp, 30.0)
                        elif resp.status in (402, 403):
                            # A free provider carving out one expensive method
                            # -- publicnode's "Request blocked" on
                            # getTokenLargestAccounts -- is permanent for that
                            # method and irrelevant to the rest. Recorded per
                            # method so the endpoint keeps serving what it can
                            # instead of being cooled wholesale, which is what
                            # makes three partial free endpoints add up to one
                            # usable pool.
                            ep.blocked_methods.add(method)
                            logger.warning(
                                "RPC %s refuses %s (HTTP %s); excluded for that "
                                "method only", _host_of(ep.endpoint.url), method,
                                resp.status)
                        elif resp.status >= 500:
                            cooldown = self._retry_after_seconds(resp, 5.0)
                        self._penalise(ep, resp.status, cooldown)
                        last_refusal = (f"{_host_of(ep.endpoint.url)} HTTP "
                                        f"{resp.status}")
                        logger.warning(
                            "RPC %s refused %s with HTTP %s%s",
                            _host_of(ep.endpoint.url), method, resp.status,
                            f"; cooling {cooldown:.0f}s" if cooldown else "")
                        if attempt == 2:
                            break
                        await asyncio.sleep(0.1 * (attempt + 1))
                        continue
                except RPCError:
                    raise
                except Exception as e:
                    self._penalise(ep, 0)
                    last_refusal = f"{_host_of(ep.endpoint.url)} {type(e).__name__}: {e}"
                    if attempt == 2:
                        raise
                    await asyncio.sleep(0.1 * (attempt + 1))
            raise RuntimeError(
                f"All RPC endpoints failed for {method}"
                + (f"; last: {last_refusal}" if last_refusal else ""))

    async def batch_request(self, requests: List[Dict[str, Any]]) -> List[Any]:
        """Results aligned to ``requests`` by JSON-RPC id, not by arrival order.

        A server is explicitly permitted to return batch responses in any
        order, and may omit entries. Zipping the reply against the request list
        therefore attributes one call's result to another call -- here, one
        wallet's transaction to a different signature, which is a wrong feature
        rather than a missing one and nothing downstream could detect it.

        Missing ids come back as None so the caller sees a hole instead of a
        shifted list.
        """
        async with self._request_semaphore:
            ep = self._select_endpoint()
            if not ep:
                raise RuntimeError("No healthy RPC endpoints")
            async with self.session.post(
                ep.endpoint.url,
                json=requests,
                headers=ep.endpoint.headers,
            ) as resp:
                if resp.status != 200:
                    self._penalise(
                        ep, resp.status,
                        self._retry_after_seconds(resp, 30.0)
                        if resp.status == 429 else 0.0)
                    raise RPCError(
                        f"{_host_of(ep.endpoint.url)} HTTP {resp.status}")
                data = await resp.json()
                if not isinstance(data, list):
                    raise RPCError("batch reply was not a JSON-RPC array")
                ep.success_count += 1
                ep.consecutive_failures = 0
                ep.health = RPCHealth.HEALTHY
                by_id = {item.get("id"): item for item in data
                         if isinstance(item, dict)}
                return [(by_id.get(request.get("id")) or {}).get("result")
                        for request in requests]

    async def broadcast_request(self, method: str, params: List[Any], timeout: float = 1.5) -> Any:
        """Race one idempotent signed transaction across every healthy RPC path."""
        endpoints = [item for item in self.endpoints if item.health == RPCHealth.HEALTHY]
        if not endpoints:
            endpoints = [item for item in self.endpoints if item.health != RPCHealth.DOWN]
        if not endpoints:
            raise RuntimeError("No usable RPC endpoints")

        async def submit(item: EndpointHealth):
            async with self.session.post(
                item.endpoint.url,
                json={"jsonrpc": "2.0", "method": method, "params": params, "id": 1},
                headers=item.endpoint.headers,
            ) as resp:
                if resp.status != 200:
                    return None
                payload = await resp.json()
                return payload.get("result")

        tasks = [asyncio.create_task(submit(item)) for item in endpoints]
        done, pending = await asyncio.wait(tasks, timeout=timeout)
        # Requests were launched together; cancel only sockets still stalled
        # after the racing window. The identical signed transaction cannot fill twice.
        for task in pending:
            task.cancel()
        results = [task.result() for task in done if not task.cancelled() and task.exception() is None]
        return next((value for value in results if value), None)

    def get_ws_url(self) -> Optional[str]:
        ep = self._select_endpoint(prefer_ws=True)
        return ep.endpoint.ws_url if ep else None

    def get_ws_urls(self) -> List[str]:
        """Return transport candidates without exposing them through status output."""
        usable = [item for item in self.endpoints if item.endpoint.ws_url and item.health != RPCHealth.DOWN]
        healthy = [item for item in usable if item.health == RPCHealth.HEALTHY]
        degraded = [item for item in usable if item.health == RPCHealth.DEGRADED]
        return [item.endpoint.ws_url for item in healthy + degraded if item.endpoint.ws_url]

    def get_web3(self) -> AsyncWeb3:
        if self.chain_config.chain_type != ChainType.EVM:
            raise TypeError("get_web3() is only valid for EVM chains")
        ep = self._select_endpoint()
        if not ep:
            raise RuntimeError("No healthy RPC endpoints")
        return AsyncWeb3(AsyncHTTPProvider(ep.endpoint.url))

    def get_stats(self) -> Dict[str, Any]:
        return {
            "chain": self.chain_config.name,
            "endpoints": [
                {
                    # Provider keys commonly live in the query string or final
                    # URL path segment. Health/status output is also journaled,
                    # so expose only the origin and never credential-bearing
                    # path/query/fragment data.
                    "url": self._safe_endpoint_origin(e.endpoint.url),
                    "health": e.health.value,
                    "latency_ms": round(e.latency_ms, 2),
                    "success": e.success_count,
                    "errors": e.error_count,
                }
                for e in self.endpoints
            ],
        }

    @staticmethod
    def _safe_endpoint_origin(url: str) -> str:
        parsed = urlsplit(url)
        if not parsed.scheme or not parsed.hostname:
            return "REDACTED_ENDPOINT"
        host = parsed.hostname
        if parsed.port:
            host = f"{host}:{parsed.port}"
        return f"{parsed.scheme}://{host}"


class RPCError(Exception):
    def __init__(self, error: Any):
        self.error = error
        super().__init__(str(error))


class ChainRegistry:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        self.chains: Dict[str, ChainConfig] = {}
        self.rpc_managers: Dict[str, RPCManager] = {}
        self._parse_config(raw)

    def _parse_config(self, raw: Dict):
        globals_cfg = raw.get("global", {})
        for name, cfg in raw.get("chains", {}).items():
            chain_type = ChainType.SOLANA if name == "solana" else ChainType.EVM
            endpoints = []
            ws_urls = cfg.get("ws_urls", [])
            for i, endpoint_url in enumerate(cfg["rpc_urls"]):
                url = self._interpolate(endpoint_url)
                ws_url = self._interpolate(ws_urls[i]) if i < len(ws_urls) else None
                if self._has_unresolved_placeholder(url):
                    logger.info("Skipping %s RPC endpoint with missing environment value", name)
                    continue
                if ws_url and self._has_unresolved_placeholder(ws_url):
                    ws_url = None
                endpoints.append(RPCEndpointConfig(url=url, ws_url=ws_url, weight=1))
            if not endpoints:
                raise ValueError(f"No usable RPC endpoints configured for {name}")
            self.chains[name] = ChainConfig(
                name=cfg["name"],
                chain_id=cfg["chain_id"],
                chain_type=chain_type,
                rpc_endpoints=endpoints,
                explorer_api=cfg["explorer_api"],
                explorer_key=self._interpolate(cfg["explorer_key"]),
                native_token=cfg["native_token"],
                decimals=cfg["decimals"],
                block_time=cfg["block_time"],
                factories=cfg.get("factories", {}),
                routers=cfg.get("routers", {}),
                base_tokens=cfg.get("base_tokens", []),
                min_liquidity_usd=cfg["min_liquidity_usd"],
                max_tax=cfg["max_tax"],
                honeypot_check=cfg["honeypot_check"],
                programs=cfg.get("programs", {}),
            )

    def _interpolate(self, value: Optional[str]) -> Optional[str]:
        return os.path.expandvars(value) if value else value

    @staticmethod
    def _has_unresolved_placeholder(value: str) -> bool:
        return "${" in value

    async def start_all(self, enabled_chains: Optional[Iterable[str]] = None):
        selected = set(enabled_chains or self.chains.keys())
        for name, chain in self.chains.items():
            if name not in selected:
                continue
            self.rpc_managers[name] = RPCManager(chain)
            await self.rpc_managers[name].start()

    async def stop_all(self):
        for mgr in self.rpc_managers.values():
            await mgr.stop()

    async def stop(self):
        """Lifecycle alias used by the desk's uniform component shutdown."""
        await self.stop_all()

    def get_chain(self, name: str) -> Optional[ChainConfig]:
        return self.chains.get(name)

    def get_rpc(self, name: str) -> Optional[RPCManager]:
        return self.rpc_managers.get(name)

    def get_all_stats(self) -> Dict[str, Any]:
        return {name: mgr.get_stats() for name, mgr in self.rpc_managers.items()}

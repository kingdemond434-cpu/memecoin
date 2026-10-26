import asyncio
from src.runtime.loop_local import loop_local_lock, loop_local_semaphore
import logging
import math
import os
import random
import time
import weakref
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Set
from urllib.parse import urlsplit
from src.chains.rpc_budget import RPCBudget, RPCBudgetExhausted, RPCProviderCooling, SharedRPCCooldowns
from email.utils import parsedate_to_datetime
from datetime import timezone

import aiohttp
import yaml
from web3 import AsyncWeb3
from web3.providers import AsyncHTTPProvider

logger = logging.getLogger(__name__)


def _host_of(url: str) -> str:
    """The host, for logs. Provider URLs carry the API key in the path or the
    query string, so the whole URL must never reach a log line."""
    try:
        return urlsplit(url).hostname or "unknown-host"
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
        #: loop -> that loop's session, keyed on the LOOP OBJECT. See
        #: _session_for_loop for why not on its id.
        self._sessions: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
        #: The one session that can have no loop object to key on: the
        #: fallback used when nothing is running. Kept apart so it cannot
        #: collide with a real loop's entry.
        self._loopless_session: Optional[aiohttp.ClientSession] = None
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
        self._helius_budget = None
        shared_path = os.getenv('MEMECOIN_RPC_BUDGET_DB')
        self._shared_cooldowns = SharedRPCCooldowns(shared_path) if shared_path else None
        if os.getenv("HELIUS_API_KEY") and any(self._is_helius(e) for e in self.endpoints):
            self._helius_budget = RPCBudget(os.getenv("MEMECOIN_RPC_BUDGET_DB", "data/state/rpc_budget.sqlite"),
                os.environ["HELIUS_API_KEY"], rps=float(os.getenv("MEMECOIN_RPC_RPS", "10")),
                daily=int(os.environ["MEMECOIN_RPC_DAILY_UNITS"]) if os.getenv("MEMECOIN_RPC_DAILY_UNITS") else None,
                monthly=int(os.environ["MEMECOIN_RPC_MONTHLY_UNITS"]) if os.getenv("MEMECOIN_RPC_MONTHLY_UNITS") else None)

    @staticmethod
    def _is_helius(ep):
        return urlsplit(ep.endpoint.url).hostname in {"mainnet.helius-rpc.com", "devnet.helius-rpc.com"}

    async def _allocate(self, ep, methods):
        if self._cooldown_deadline(ep) > time.time():
            raise RPCProviderCooling(ep.cooldown_until)
        if self._helius_budget is not None and self._is_helius(ep):
            await self._helius_budget.acquire(methods)
        # A different manager/process may have received 429 while this call
        # waited for its allocation. Recheck before opening another socket.
        if self._cooldown_deadline(ep) > time.time():
            raise RPCProviderCooling(ep.cooldown_until)

    def _cooldown_deadline(self, ep):
        shared = getattr(self, '_shared_cooldowns', None)
        if shared is not None:
            ep.cooldown_until = max(ep.cooldown_until, shared.until(ep.endpoint))
        return ep.cooldown_until

    def _session_for_loop(self):
        """This loop's aiohttp session, created on first use by that loop.

        Same reasoning as the semaphore. A session binds to the loop that
        created it -- its connector, its timers, its timeout contexts -- and
        used from another it raises "Timeout context manager should be used
        inside a task" on the first request. The miners run on their own
        loop and share this manager, so a single session could never have
        served them.

        Keyed on the LOOP OBJECT, in a weak-keyed map, never on `id(loop)`.
        CPython reuses ids: once a loop is collected the next allocation can
        land on the same address, and a brand new loop then inherits the dead
        one's session -- a closed connector, or worse an open one bound to a
        loop that is gone. It is the same mistake the HttpClient fix made and
        the same one the loop-local semaphore was written to avoid, and it
        surfaces as an intermittent timeout-context error that nothing in the
        logs explains. Weak keys also make eviction automatic: a loop that
        goes away takes its entry with it.
        """
        import asyncio as _asyncio

        try:
            loop = _asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        self._retire_closed_loops()
        if loop is None:
            existing = self._loopless_session
        else:
            existing = self._sessions.get(loop)
        if existing is not None and not existing.closed:
            return existing
        created = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            connector=aiohttp.TCPConnector(limit=100, limit_per_host=20),
        )
        if loop is None:
            self._loopless_session = created
        else:
            self._sessions[loop] = created
        return created

    def _retire_closed_loops(self) -> None:
        """Drop the entry for any loop that has been closed.

        Weak keys alone are not enough here, and it is worth saying why: an
        `aiohttp.ClientSession` holds a reference to its own loop, so the
        VALUE of each entry strongly references its KEY. A weak-keyed map
        cannot collect a key its own value is pinning, so without this sweep
        every loop the desk ever ran would keep its session alive for the
        life of the process.

        The weak keying is still the fix that matters -- it is what stops a
        new loop inheriting a dead one's session when CPython reuses an
        address -- and this is what makes the memory follow.
        """
        dead = [loop for loop in list(self._sessions) if loop.is_closed()]
        for loop in dead:
            session = self._sessions.pop(loop, None)
            if session is not None and not session.closed:
                # Its loop is gone, so it cannot be closed properly any more.
                # Said out loud rather than dropped: an unclosed session is a
                # socket, and a caller that closes loops without closing
                # sessions should be told which one it was.
                logger.debug(
                    "retiring an RPC session whose loop closed under it")
                session._connector = None

    @property
    def session(self):
        """Always the calling loop's session. Every call site goes through here."""
        return self._session_for_loop()

    async def start(self):
        self._session = self._session_for_loop()
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        await self._initial_health_check()

    async def close_current_loop(self):
        """Release this worker's session without stopping the main RPC loop."""
        session = self._sessions.pop(asyncio.get_running_loop(), None)
        if session is not None and not session.closed:
            await session.close()

    async def stop(self):
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        for ws in self._ws_connections.values():
            await ws.close()
        for session in list(self._sessions.values()) + [self._loopless_session]:
            if session is not None and not session.closed:
                try:
                    await session.close()
                except Exception:  # pragma: no cover - shutdown only
                    pass
        self._sessions.clear()
        self._loopless_session = None
        if self._session and not self._session.closed:
            await self._session.close()
        if self._helius_budget is not None:
            self._helius_budget.close()
            self._helius_budget = None
        if self._shared_cooldowns is not None:
            self._shared_cooldowns.close()
            self._shared_cooldowns = None

    async def _initial_health_check(self):
        await asyncio.gather(*[self._check_endpoint(e) for e in self.endpoints])

    async def _health_check_loop(self):
        while True:
            await asyncio.sleep(30)
            await asyncio.gather(*[self._check_endpoint(e) for e in self.endpoints])

    async def _check_endpoint(self, ep: EndpointHealth):
        if self._cooldown_deadline(ep) > time.time(): return
        start = time.time()
        method, params = self._health_probe()
        try:
            await self._allocate(ep, [method])
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
        except RPCBudgetExhausted as exc:
            ep.cooldown_until = exc.retry_at
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
        """Honor finite delay/date headers, including genuine daily resets."""
        raw = ""
        try:
            raw = str(resp.headers.get("Retry-After", "")).strip()
        except Exception:
            raw = ""
        if raw:
            try:
                delay = float(raw)
                if math.isfinite(delay) and delay >= 0:
                    return delay
            except ValueError:
                try:
                    until = parsedate_to_datetime(raw)
                    if until.tzinfo is None: until = until.replace(tzinfo=timezone.utc)
                    return max(0., until.timestamp()-time.time())
                except (ValueError, TypeError, OverflowError):
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
            if self._shared_cooldowns is not None:
                self._shared_cooldowns.defer(ep.endpoint, ep.cooldown_until)
        if ep.consecutive_failures >= 3:
            ep.health = RPCHealth.DOWN
        else:
            ep.health = RPCHealth.DEGRADED

    def _select_endpoint(self, prefer_ws: bool = False,
                         method: str = "") -> Optional[EndpointHealth]:
        now = time.time()
        candidates = [e for e in self.endpoints
                      if e.health != RPCHealth.DOWN and self._cooldown_deadline(e) <= now
                      and not (method and method in e.blocked_methods)]
        if not candidates:
            # Retry-After is an actual refusal window, including when every
            # provider is cooling. Never turn it into another immediate call.
            return None
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
                        f"No healthy RPC endpoint serves {method}" + (f"; last: {last_refusal}" if last_refusal else ""))
                try:
                    await self._allocate(ep, [method])
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
                except RPCBudgetExhausted as exc:
                    ep.cooldown_until = exc.retry_at
                    last_refusal = ('RPC provider cooling after refusal' if isinstance(exc,RPCProviderCooling)
                                    else 'configured RPC allocation exhausted')
                    continue
                except Exception as e:
                    self._penalise(ep, 0)
                    last_refusal = f"{_host_of(ep.endpoint.url)} {type(e).__name__}"
                    if attempt == 2:
                        raise RPCError(last_refusal) from None
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
        if self._helius_budget is not None:
            chunk = 1 if any(r.get('method') == 'sendTransaction' for r in requests) else max(1, int(self._helius_budget.rps))
            if len(requests) > chunk:
                results = []
                for offset in range(0, len(requests), chunk):
                    results.extend(await self.batch_request(requests[offset:offset+chunk]))
                return results
        async with self._request_semaphore:
            ep = self._select_endpoint()
            if not ep:
                raise RuntimeError("No healthy RPC endpoints")
            await self._allocate(ep, [r["method"] for r in requests])
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
            if self._cooldown_deadline(item) > time.time(): return None
            await self._allocate(item, [method])
            async with self.session.post(
                item.endpoint.url,
                json={"jsonrpc": "2.0", "method": method, "params": params, "id": 1},
                headers=item.endpoint.headers,
            ) as resp:
                if resp.status != 200:
                    self._penalise(item,resp.status,self._retry_after_seconds(resp,30.)
                        if resp.status == 429 else 0.)
                    return None
                payload = await resp.json()
                return payload.get("result")

        tasks = [asyncio.create_task(submit(item)) for item in endpoints]
        done, pending = await asyncio.wait(tasks, timeout=timeout)
        # Requests were launched together; cancel only sockets still stalled
        # after the racing window. The identical signed transaction cannot fill twice.
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending,return_exceptions=True)
        results = [task.result() for task in done if not task.cancelled() and task.exception() is None]
        return next((value for value in results if value), None)

    def get_ws_url(self) -> Optional[str]:
        ep = self._select_endpoint(prefer_ws=True)
        return ep.endpoint.ws_url if ep else None

    def get_ws_urls(self) -> List[str]:
        """Return transport candidates without exposing them through status output."""
        usable = [item for item in self.endpoints if item.endpoint.ws_url and item.health != RPCHealth.DOWN
                  and self._cooldown_deadline(item) <= time.time()]
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
            "provider_budget": self._helius_budget.report() if self._helius_budget is not None else None,
            "shared_refusal_windows": self._shared_cooldowns is not None,
            "endpoints": [
                {
                    # Provider keys commonly live in the query string or final
                    # URL path segment. Health/status output is also journaled,
                    # so expose only the origin and never credential-bearing
                    # path/query/fragment data.
                    "url": self._safe_endpoint_origin(e.endpoint.url),
                    "health": e.health.value,
                    "available": e.health != RPCHealth.DOWN and self._cooldown_deadline(e) <= time.time(),
                    "cooldown_until": e.cooldown_until,
                    "latency_ms": round(e.latency_ms, 2),
                    "success": e.success_count,
                    "errors": e.error_count,
                    "last_http_status": e.last_status,
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
        self._start_locks = {}
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
            if name == 'solana':
                from src.chains.research_rpc import extra_solana_endpoints
                indexed = {endpoint.url: endpoint for endpoint in endpoints}
                for url, ws_url in extra_solana_endpoints():
                    if url in indexed:
                        if ws_url:
                            indexed[url].ws_url = ws_url
                    else:
                        endpoint = RPCEndpointConfig(url=url, ws_url=ws_url, weight=1)
                        endpoints.append(endpoint)
                        indexed[url] = endpoint
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

    async def start_chain(self, name: str) -> RPCManager:
        """Publish one manager only after startup; clean up failed/cancelled probes."""
        lock = self._start_locks.setdefault(name, loop_local_lock("chain.start"))
        async with lock:
            if name in self.rpc_managers:
                return self.rpc_managers[name]
            manager = RPCManager(self.chains[name])
            try:
                await manager.start()
            except BaseException:
                await manager.stop()
                raise
            self.rpc_managers[name] = manager
            return manager

    async def start_all(self, enabled_chains: Optional[Iterable[str]] = None):
        selected = set(self.chains if enabled_chains is None else enabled_chains)
        # This bounds concurrent connections, not the number of configured chains.
        slots = loop_local_semaphore(4, "chain.startup")

        async def start(name):
            async with slots:
                await self.start_chain(name)

        async with asyncio.TaskGroup() as tasks:
            for name in self.chains:
                if name in selected:
                    tasks.create_task(start(name))

    async def stop_all(self):
        for name, mgr in list(self.rpc_managers.items()):
            await mgr.stop()
            if self.rpc_managers.get(name) is mgr:
                del self.rpc_managers[name]

    async def stop(self):
        """Lifecycle alias used by the desk's uniform component shutdown."""
        await self.stop_all()

    def get_chain(self, name: str) -> Optional[ChainConfig]:
        return self.chains.get(name)

    def get_rpc(self, name: str) -> Optional[RPCManager]:
        return self.rpc_managers.get(name)

    def get_all_stats(self) -> Dict[str, Any]:
        return {name: mgr.get_stats() for name, mgr in self.rpc_managers.items()}

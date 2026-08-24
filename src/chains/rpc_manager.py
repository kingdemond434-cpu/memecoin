import asyncio
import json
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from urllib.parse import urlparse

import aiohttp
import yaml
from web3 import AsyncWeb3
from web3.providers import AsyncHTTPProvider, WebSocketProvider
from web3.types import RPCEndpoint, RPCResponse

logger = logging.getLogger(__name__)


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


class RPCManager:
    def __init__(self, chain_config: ChainConfig, max_concurrent_total: int = 100):
        self.chain_config = chain_config
        self.endpoints: List[EndpointHealth] = [
            EndpointHealth(ep) for ep in chain_config.rpc_endpoints
        ]
        self.max_concurrent_total = max_concurrent_total
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws_connections: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._health_check_task: Optional[asyncio.Task] = None
        self._request_semaphore = asyncio.Semaphore(max_concurrent_total)

    async def start(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            connector=aiohttp.TCPConnector(limit=100, limit_per_host=20),
        )
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
        if self._session:
            await self._session.close()

    async def _initial_health_check(self):
        await asyncio.gather(*[self._check_endpoint(e) for e in self.endpoints])

    async def _health_check_loop(self):
        while True:
            await asyncio.sleep(30)
            await asyncio.gather(*[self._check_endpoint(e) for e in self.endpoints])

    async def _check_endpoint(self, ep: EndpointHealth):
        start = time.time()
        try:
            async with self._session.post(
                ep.endpoint.url,
                json={"jsonrpc": "2.0", "method": "eth_blockNumber", "params": [], "id": 1},
                headers=ep.endpoint.headers,
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if "result" in data:
                        ep.health = RPCHealth.HEALTHY
                        ep.latency_ms = (time.time() - start) * 1000
                        ep.success_count += 1
                        ep.consecutive_failures = 0
                        return
        except Exception:
            pass
        ep.error_count += 1
        ep.consecutive_failures += 1
        if ep.consecutive_failures >= 3:
            ep.health = RPCHealth.DOWN
        elif ep.consecutive_failures >= 1:
            ep.health = RPCHealth.DEGRADED
        ep.last_check = time.time()

    def _select_endpoint(self, prefer_ws: bool = False) -> Optional[EndpointHealth]:
        healthy = [e for e in self.endpoints if e.health != RPCHealth.DOWN]
        if not healthy:
            return None
        if prefer_ws:
            healthy = [e for e in healthy if e.endpoint.ws_url]
        if not healthy:
            return None
        weights = [e.endpoint.weight * (1 / max(e.latency_ms, 1)) for e in healthy]
        return random.choices(healthy, weights=weights, k=1)[0]

    async def request(self, method: str, params: List[Any]) -> Any:
        async with self._request_semaphore:
            for attempt in range(3):
                ep = self._select_endpoint()
                if not ep:
                    raise RuntimeError("No healthy RPC endpoints")
                try:
                    async with self._session.post(
                        ep.endpoint.url,
                        json={"jsonrpc": "2.0", "method": method, "params": params, "id": 1},
                        headers=ep.endpoint.headers,
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if "result" in data:
                                ep.success_count += 1
                                return data["result"]
                            if "error" in data:
                                raise RPCError(data["error"])
                except Exception as e:
                    ep.error_count += 1
                    ep.consecutive_failures += 1
                    if ep.consecutive_failures >= 3:
                        ep.health = RPCHealth.DOWN
                    if attempt == 2:
                        raise
                    await asyncio.sleep(0.1 * (attempt + 1))
            raise RuntimeError("All RPC endpoints failed")

    async def batch_request(self, requests: List[Dict[str, Any]]) -> List[Any]:
        async with self._request_semaphore:
            ep = self._select_endpoint()
            if not ep:
                raise RuntimeError("No healthy RPC endpoints")
            async with self._session.post(
                ep.endpoint.url,
                json=requests,
                headers=ep.endpoint.headers,
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return [r.get("result") for r in data]
                raise RPCError(await resp.text())

    def get_ws_url(self) -> Optional[str]:
        ep = self._select_endpoint(prefer_ws=True)
        return ep.endpoint.ws_url if ep else None

    def get_web3(self) -> AsyncWeb3:
        ep = self._select_endpoint()
        if not ep:
            raise RuntimeError("No healthy RPC endpoints")
        return AsyncWeb3(AsyncHTTPProvider(ep.endpoint.url))

    def get_stats(self) -> Dict[str, Any]:
        return {
            "chain": self.chain_config.name,
            "endpoints": [
                {
                    "url": e.endpoint.url,
                    "health": e.health.value,
                    "latency_ms": round(e.latency_ms, 2),
                    "success": e.success_count,
                    "errors": e.error_count,
                }
                for e in self.endpoints
            ],
        }


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
            endpoints = [
                RPCEndpointConfig(
                    url=self._interpolate(ep),
                    ws_url=self._interpolate(cfg.get("ws_urls", [None])[i]) if i < len(cfg.get("ws_urls", [])) else None,
                    weight=1,
                )
                for i, ep in enumerate(cfg["rpc_urls"])
            ]
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

    def _interpolate(self, value: str) -> str:
        import os
        for key, val in os.environ.items():
            value = value.replace(f"${{{key}}}", val)
        return value

    async def start_all(self):
        for name, chain in self.chains.items():
            self.rpc_managers[name] = RPCManager(chain)
            await self.rpc_managers[name].start()

    async def stop_all(self):
        for mgr in self.rpc_managers.values():
            await mgr.stop()

    def get_chain(self, name: str) -> Optional[ChainConfig]:
        return self.chains.get(name)

    def get_rpc(self, name: str) -> Optional[RPCManager]:
        return self.rpc_managers.get(name)

    def get_all_stats(self) -> Dict[str, Any]:
        return {name: mgr.get_stats() for name, mgr in self.rpc_managers.items()}
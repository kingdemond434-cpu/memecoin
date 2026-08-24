import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from collections import deque

from web3 import AsyncWeb3
from web3.types import LogReceipt, TxData

from src.chains.rpc_manager import ChainConfig, ChainRegistry, RPCManager

logger = logging.getLogger(__name__)


class DetectionSource(Enum):
    MEMPOOL = "mempool"
    FACTORY = "factory"
    BLOCK = "block"
    SOCIAL = "social"


@dataclass
class TokenCandidate:
    address: str
    chain: str
    source: DetectionSource
    block_number: int
    tx_hash: Optional[str] = None
    deployer: Optional[str] = None
    factory: Optional[str] = None
    pair: Optional[str] = None
    base_token: Optional[str] = None
    initial_liquidity_usd: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseDetector(ABC):
    def __init__(self, chain_name: str, chain_config: ChainConfig, rpc: RPCManager, callback: Callable):
        self.chain_name = chain_name
        self.chain_config = chain_config
        self.rpc = rpc
        self.callback = callback
        self.running = False
        self._task: Optional[asyncio.Task] = None

    @abstractmethod
    async def start(self):
        pass

    @abstractmethod
    async def stop(self):
        pass


class MempoolDetector(BaseDetector):
    def __init__(self, chain_name: str, chain_config: ChainConfig, rpc: RPCManager, callback: Callable,
                 pending_tx_queue_size: int = 10000):
        super().__init__(chain_name, chain_config, rpc, callback)
        self.pending_tx_queue: asyncio.Queue = asyncio.Queue(maxsize=pending_tx_queue_size)
        self._seen_tx: Set[str] = set()
        self._ws = None
        self._process_task: Optional[asyncio.Task] = None

    async def start(self):
        self.running = True
        ws_url = self.rpc.get_ws_url()
        if not ws_url:
            logger.warning(f"No WS URL for {self.chain_name}, mempool detection disabled")
            return

        self._ws = await self._connect_ws(ws_url)
        self._process_task = asyncio.create_task(self._process_queue())
        asyncio.create_task(self._listen_mempool())

    async def stop(self):
        self.running = False
        if self._process_task:
            self._process_task.cancel()
        if self._ws:
            await self._ws.close()

    async def _connect_ws(self, url: str):
        import websockets
        ws = await websockets.connect(url, ping_interval=20, ping_timeout=10)
        await ws.send(json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_subscribe",
            "params": ["newPendingTransactions"]
        }))
        resp = await ws.recv()
        logger.info(f"Mempool subscription: {resp}")
        return ws

    async def _listen_mempool(self):
        import websockets
        while self.running:
            try:
                msg = await asyncio.wait_for(self._ws.recv(), timeout=30)
                data = json.loads(msg)
                if "params" in data and "result" in data["params"]:
                    tx_hash = data["params"]["result"]
                    if tx_hash not in self._seen_tx:
                        self._seen_tx.add(tx_hash)
                        if len(self._seen_tx) > 50000:
                            self._seen_tx.clear()
                        try:
                            self.pending_tx_queue.put_nowait(tx_hash)
                        except asyncio.QueueFull:
                            pass
            except asyncio.TimeoutError:
                await self._ws.ping()
            except Exception as e:
                logger.error(f"Mempool WS error: {e}")
                await asyncio.sleep(5)
                await self._reconnect()

    async def _reconnect(self):
        ws_url = self.rpc.get_ws_url()
        if ws_url:
            self._ws = await self._connect_ws(ws_url)

    async def _process_queue(self):
        while self.running:
            try:
                tx_hash = await asyncio.wait_for(self.pending_tx_queue.get(), timeout=1)
                await self._analyze_tx(tx_hash)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Process queue error: {e}")

    async def _analyze_tx(self, tx_hash: str):
        try:
            tx = await self.rpc.request("eth_getTransactionByHash", [tx_hash])
            if not tx or not tx.get("to"):
                return

            to_addr = tx["to"].lower()
            input_data = tx.get("input", "0x")

            for factory_name, factory_addr in self.chain_config.factories.items():
                if to_addr == factory_addr.lower():
                    if input_data.startswith("0x") and len(input_data) > 10:
                        selector = input_data[:10]
                        if selector in ["0x4e5a0b1d", "0x0d295980", "0x1c4f5b9d"]:
                            await self._handle_factory_call(tx, factory_name, factory_addr)
                            return

            for router_name, router_addr in self.chain_config.routers.items():
                if to_addr == router_addr.lower():
                    if input_data.startswith("0x") and len(input_data) > 10:
                        selector = input_data[:10]
                        if selector in ["0xf305d719", "0x8803dbee", "0x7ff36ab5", "0x5c11d795"]:
                            await self._handle_router_add_liquidity(tx, router_name, router_addr)
                            return

        except Exception as e:
            logger.debug(f"TX analysis error: {e}")

    async def _handle_factory_call(self, tx: TxData, factory_name: str, factory_addr: str):
        try:
            receipt = await self.rpc.request("eth_getTransactionReceipt", [tx["hash"]])
            if not receipt or receipt.get("status") != "0x1":
                return

            for log in receipt.get("logs", []):
                topics = log.get("topics", [])
                if len(topics) >= 3 and topics[0].lower() == "0x0d3648bd0f6ba80134a33ba9275ac585d9d315f0ad8355cdadefa3b5c4c54fae".lower():
                    token_addr = "0x" + topics[2][-40:]
                    pair_addr = "0x" + topics[1][-40:]
                    await self.callback(TokenCandidate(
                        address=token_addr,
                        chain=self.chain_name,
                        source=DetectionSource.FACTORY,
                        block_number=int(receipt["blockNumber"], 16),
                        tx_hash=tx["hash"],
                        deployer=tx["from"],
                        factory=factory_addr,
                        pair=pair_addr,
                    ))
                    return
        except Exception as e:
            logger.debug(f"Factory call handling error: {e}")

    async def _handle_router_add_liquidity(self, tx: TxData, router_name: str, router_addr: str):
        try:
            receipt = await self.rpc.request("eth_getTransactionReceipt", [tx["hash"]])
            if not receipt or receipt.get("status") != "0x1":
                return

            for log in receipt.get("logs", []):
                topics = log.get("topics", [])
                if len(topics) >= 3 and topics[0].lower() == "0x783cca1c0412dd0d695e784568c96da2e9c22ff989357a2e8b1d9b2b4e6b7118".lower():
                    token_a = "0x" + topics[1][-40:]
                    token_b = "0x" + topics[2][-40:]
                    pair_addr = log["address"]
                    
                    new_token = None
                    base_token = None
                    for bt in self.chain_config.base_tokens:
                        if token_a.lower() == bt.lower():
                            new_token, base_token = token_b, token_a
                            break
                        if token_b.lower() == bt.lower():
                            new_token, base_token = token_a, token_b
                            break
                    
                    if new_token:
                        await self.callback(TokenCandidate(
                            address=new_token,
                            chain=self.chain_name,
                            source=DetectionSource.MEMPOOL,
                            block_number=int(receipt["blockNumber"], 16),
                            tx_hash=tx["hash"],
                            deployer=tx["from"],
                            pair=pair_addr,
                            base_token=base_token,
                        ))
                        return
        except Exception as e:
            logger.debug(f"Router liquidity handling error: {e}")


class FactoryPoller(BaseDetector):
    def __init__(self, chain_name: str, chain_config: ChainConfig, rpc: RPCManager, callback: Callable,
                 poll_interval: float = 2.0):
        super().__init__(chain_name, chain_config, rpc, callback)
        self.poll_interval = poll_interval
        self._last_blocks: Dict[str, int] = {}
        self._pair_created_topic = "0x0d3648bd0f6ba80134a33ba9275ac585d9d315f0ad8355cdadefa3b5c4c54fae"

    async def start(self):
        self.running = True
        for factory_name, factory_addr in self.chain_config.factories.items():
            self._last_blocks[factory_name] = await self._get_latest_block()
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()

    async def _get_latest_block(self) -> int:
        block = await self.rpc.request("eth_blockNumber", [])
        return int(block, 16) if isinstance(block, str) else block

    async def _poll_loop(self):
        while self.running:
            try:
                for factory_name, factory_addr in self.chain_config.factories.items():
                    await self._poll_factory(factory_name, factory_addr)
            except Exception as e:
                logger.error(f"Factory poll error: {e}")
            await asyncio.sleep(self.poll_interval)

    async def _poll_factory(self, factory_name: str, factory_addr: str):
        from_block = self._last_blocks.get(factory_name, await self._get_latest_block())
        to_block = await self._get_latest_block()
        
        if to_block <= from_block:
            return

        logs = await self.rpc.request("eth_getLogs", [{
            "fromBlock": hex(from_block),
            "toBlock": hex(to_block),
            "address": factory_addr,
            "topics": [self._pair_created_topic]
        }])

        if logs:
            for log in logs:
                topics = log.get("topics", [])
                if len(topics) >= 3:
                    token_addr = "0x" + topics[2][-40:]
                    pair_addr = "0x" + topics[1][-40:]
                    tx_hash = log.get("transactionHash")
                    
                    tx = await self.rpc.request("eth_getTransactionByHash", [tx_hash]) if tx_hash else None
                    deployer = tx["from"] if tx else None
                    
                    await self.callback(TokenCandidate(
                        address=token_addr,
                        chain=self.chain_name,
                        source=DetectionSource.FACTORY,
                        block_number=int(log["blockNumber"], 16),
                        tx_hash=tx_hash,
                        deployer=deployer,
                        factory=factory_addr,
                        pair=pair_addr,
                    ))

        self._last_blocks[factory_name] = to_block


class SolanaTokenDetector(BaseDetector):
    def __init__(self, chain_name: str, chain_config: ChainConfig, rpc: RPCManager, callback: Callable):
        super().__init__(chain_name, chain_config, rpc, callback)
        self._ws = None
        self._known_mints: Set[str] = set()

    async def start(self):
        self.running = True
        ws_url = self.rpc.get_ws_url()
        if ws_url:
            self._ws = await self._connect_ws(ws_url)
            await self._subscribe_programs()
            asyncio.create_task(self._listen_ws())
        
        self._task = asyncio.create_task(self._poll_new_mints())

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
        if self._ws:
            await self._ws.close()

    async def _connect_ws(self, url: str):
        import websockets
        return await websockets.connect(url, ping_interval=20)

    async def _subscribe_programs(self):
        for prog_name, prog_id in self.chain_config.programs.items():
            await self._ws.send(json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "logsSubscribe",
                "params": [
                    {"mentions": [prog_id]},
                    {"commitment": "processed"}
                ]
            }))

    async def _listen_ws(self):
        while self.running:
            try:
                msg = await asyncio.wait_for(self._ws.recv(), timeout=30)
                data = json.loads(msg)
                if "params" in data and "result" in data["params"]:
                    await self._process_log(data["params"]["result"])
            except Exception as e:
                logger.error(f"Solana WS error: {e}")
                await asyncio.sleep(5)

    async def _process_log(self, log: Dict):
        try:
            if "value" in log and "logs" in log["value"]:
                for log_entry in log["value"]["logs"]:
                    if "initialize" in log_entry.lower() or "create" in log_entry.lower():
                        mint = self._extract_mint(log_entry)
                        if mint and mint not in self._known_mints:
                            self._known_mints.add(mint)
                            await self.callback(TokenCandidate(
                                address=mint,
                                chain=self.chain_name,
                                source=DetectionSource.FACTORY,
                                block_number=log.get("context", {}).get("slot", 0),
                                metadata={"log": log_entry}
                            ))
        except Exception as e:
            logger.debug(f"Solana log process error: {e}")

    def _extract_mint(self, log: str) -> Optional[str]:
        import re
        matches = re.findall(r'[1-9A-HJ-NP-Za-km-z]{32,44}', log)
        for m in matches:
            if len(m) >= 32:
                return m
        return None

    async def _poll_new_mints(self):
        while self.running:
            try:
                for prog_name, prog_id in self.chain_config.programs.items():
                    sigs = await self.rpc.request("getSignaturesForAddress", [prog_id, {"limit": 20}])
                    for sig_info in sigs:
                        sig = sig_info["signature"]
                        tx = await self.rpc.request("getTransaction", [sig, {"encoding": "jsonParsed"}])
                        if tx:
                            mint = self._extract_mint_from_tx(tx)
                            if mint and mint not in self._known_mints:
                                self._known_mints.add(mint)
                                await self.callback(TokenCandidate(
                                    address=mint,
                                    chain=self.chain_name,
                                    source=DetectionSource.FACTORY,
                                    block_number=tx.get("slot", 0),
                                    tx_hash=sig,
                                    metadata={"program": prog_name}
                                ))
            except Exception as e:
                logger.error(f"Solana poll error: {e}")
            await asyncio.sleep(5)

    def _extract_mint_from_tx(self, tx: Dict) -> Optional[str]:
        try:
            meta = tx.get("meta", {})
            post_balances = meta.get("postTokenBalances", [])
            for bal in post_balances:
                if bal.get("uiTokenAmount", {}).get("decimals", 0) > 0:
                    return bal.get("mint")
        except Exception:
            pass
        return None


class TokenDetectionEngine:
    def __init__(self, registry: ChainRegistry):
        self.registry = registry
        self.detectors: List[BaseDetector] = []
        self.candidates: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._seen_tokens: Dict[str, Set[str]] = {}
        self._dedup_window = 300

    def add_chain(self, chain_name: str, enable_mempool: bool = True, enable_factory: bool = True):
        chain = self.registry.get_chain(chain_name)
        rpc = self.registry.get_rpc(chain_name)
        if not chain or not rpc:
            return

        self._seen_tokens[chain_name] = set()

        if chain.chain_type.value == "evm":
            if enable_factory:
                self.detectors.append(FactoryPoller(chain_name, chain, rpc, self._on_candidate))
            if enable_mempool:
                self.detectors.append(MempoolDetector(chain_name, chain, rpc, self._on_candidate))
        elif chain.chain_type.value == "solana":
            self.detectors.append(SolanaTokenDetector(chain_name, chain, rpc, self._on_candidate))

    async def _on_candidate(self, candidate: TokenCandidate):
        key = f"{candidate.chain}:{candidate.address.lower()}"
        if key in self._seen_tokens.get(candidate.chain, set()):
            return
        
        self._seen_tokens.setdefault(candidate.chain, set()).add(key)
        if len(self._seen_tokens[candidate.chain]) > 10000:
            old = list(self._seen_tokens[candidate.chain])[:5000]
            for k in old:
                self._seen_tokens[candidate.chain].discard(k)
        
        try:
            self.candidates.put_nowait(candidate)
        except asyncio.QueueFull:
            logger.warning("Candidate queue full, dropping")

    async def start(self):
        for d in self.detectors:
            await d.start()

    async def stop(self):
        for d in self.detectors:
            await d.stop()

    async def get_candidate(self) -> TokenCandidate:
        return await self.candidates.get()
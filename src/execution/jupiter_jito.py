import asyncio
import base64
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import aiohttp
import nacl.signing
import nacl.encoding

from src.chains.rpc_manager import ChainConfig, RPCManager

logger = logging.getLogger(__name__)


class RouteType(Enum):
    JUPITER_V6 = "jupiter_v6"
    JUPITER_V4 = "jupiter_v4"
    RAYDIUM_DIRECT = "raydium_direct"
    ORCA_DIRECT = "orca_direct"
    METEORA_DLMM = "meteora_dlmm"
    JITO_BUNDLE = "jito_bundle"


class TransactionStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    LANDED = "landed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    REJECTED = "rejected"


@dataclass
class SwapQuote:
    input_mint: str
    output_mint: str
    input_amount: int
    output_amount: int
    price_impact_pct: float
    route: List[Dict]
    route_type: RouteType
    fees_bps: int
    min_output_amount: int
    quote_time: float = field(default_factory=time.time)
    raw_quote: Dict = field(default_factory=dict)


@dataclass
class SwapTransaction:
    transaction: str
    last_valid_block_height: int
    fee_payer: str
    status: TransactionStatus = TransactionStatus.PENDING
    signature: Optional[str] = None
    submitted_at: float = field(default_factory=time.time)
    landed_at: Optional[float] = None
    slot: Optional[int] = None
    error: Optional[str] = None
    priority_fee: int = 0
    jito_tip: int = 0
    route_type: RouteType = RouteType.JUPITER_V6


@dataclass
class ExecutionResult:
    success: bool
    signature: Optional[str] = None
    input_amount: int = 0
    output_amount: int = 0
    slippage_bps: int = 0
    fees_paid: int = 0
    priority_fee: int = 0
    jito_tip: int = 0
    latency_ms: int = 0
    route_type: RouteType = RouteType.JUPITER_V6
    error: Optional[str] = None
    landed: bool = False
    slot: Optional[int] = None


class JupiterClient:
    def __init__(self, base_url: str = "https://quote-api.jup.ag/v6"):
        self.base_url = base_url
        self._session: Optional[aiohttp.ClientSession] = None

    async def start(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            connector=aiohttp.TCPConnector(limit=20)
        )

    async def stop(self):
        if self._session:
            await self._session.close()

    async def get_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int = 100,
        swap_mode: str = "ExactIn",
        only_direct_routes: bool = False,
        as_legacy_transaction: bool = False,
        platform_fee_bps: int = 0
    ) -> Optional[SwapQuote]:
        try:
            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": str(amount),
                "slippageBps": str(slippage_bps),
                "swapMode": swap_mode,
                "onlyDirectRoutes": str(only_direct_routes).lower(),
                "asLegacyTransaction": str(as_legacy_transaction).lower(),
                "platformFeeBps": str(platform_fee_bps)
            }
            
            async with self._session.get(f"{self.base_url}/quote", params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return self._parse_quote(data, input_mint, output_mint, amount, RouteType.JUPITER_V6)
                else:
                    logger.error(f"Jupiter quote failed: {resp.status} - {await resp.text()}")
        except Exception as e:
            logger.error(f"Jupiter quote error: {e}")
        return None

    async def get_swap_transaction(
        self,
        quote: SwapQuote,
        user_public_key: str,
        wrap_unwrap_sol: bool = True,
        dynamic_compute_unit_limit: bool = True,
        prioritization_fee_lamports: Optional[int] = None,
        fee_account: Optional[str] = None
    ) -> Optional[SwapTransaction]:
        try:
            payload = {
                "quoteResponse": quote.raw_quote,
                "userPublicKey": user_public_key,
                "wrapAndUnwrapSol": wrap_unwrap_sol,
                "dynamicComputeUnitLimit": dynamic_compute_unit_limit,
            }
            
            if prioritization_fee_lamports:
                payload["prioritizationFeeLamports"] = prioritization_fee_lamports
            
            if fee_account:
                payload["feeAccount"] = fee_account
            
            async with self._session.post(
                f"{self.base_url}/swap",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return SwapTransaction(
                        transaction=data.get("swapTransaction", ""),
                        last_valid_block_height=int(data.get("lastValidBlockHeight", 0)),
                        fee_payer=user_public_key,
                        priority_fee=prioritization_fee_lamports or 0,
                        route_type=quote.route_type
                    )
                else:
                    logger.error(f"Jupiter swap failed: {resp.status} - {await resp.text()}")
        except Exception as e:
            logger.error(f"Jupiter swap error: {e}")
        return None

    def _parse_quote(self, data: Dict, input_mint: str, output_mint: str, amount: int, route_type: RouteType) -> SwapQuote:
        route_plan = data.get("routePlan", [])
        fees = data.get("swapUsdValue", 0)
        
        return SwapQuote(
            input_mint=input_mint,
            output_mint=output_mint,
            input_amount=amount,
            output_amount=int(data.get("outAmount", 0)),
            price_impact_pct=float(data.get("priceImpactPct", 0)),
            route=route_plan,
            route_type=route_type,
            fees_bps=int(data.get("swapUsdValue", 0) * 10000 / amount) if amount > 0 else 0,
            min_output_amount=int(data.get("outAmount", 0)) * 90 // 100,
            raw_quote=data
        )


class JitoClient:
    def __init__(self, jito_url: str = "https://mainnet.block-engine.jito.wtf"):
        self.jito_url = jito_url
        self._session: Optional[aiohttp.ClientSession] = None
        self._tip_accounts: List[str] = []

    async def start(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            connector=aiohttp.TCPConnector(limit=20)
        )
        await self._fetch_tip_accounts()

    async def stop(self):
        if self._session:
            await self._session.close()

    async def _fetch_tip_accounts(self):
        try:
            async with self._session.get(f"{self.jito_url}/api/v1/bundles/tip_accounts") as resp:
                if resp.status == 200:
                    self._tip_accounts = await resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch Jito tip accounts: {e}")
            self._tip_accounts = [
                "96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5",
                "HFqU5x63VTqvQss8hp11i4wVV8bD44pvwucfZ2bUfgRe",
                "Cw8CFyM9FkoMi7K7Crf6HNQqf4uEMzpKw6QNghXLvLkY",
                "ADaUMid9yfUytqMBgopwjb2DTLSokTSzL1zt6iGPaS49",
                "DfXygSm4jCyNCybVYYK6DwvWqjKee8pbDmJGcLWNDXjh",
                "ADuUkR4vqLUMWXxW9gh6D6L8pMSawimctcNZ5pGwDcEt",
                "DttWaMuVvTiduZRnguLF7jNxTgiMBZ1hyAumKUiL2KRL",
                "3AVi9TgZUoW8Z7t4wZj5d2Bt7u4Vz8wJ6Q8p9L2mN1sT"
            ]

    def get_random_tip_account(self) -> str:
        import random
        return random.choice(self._tip_accounts) if self._tip_accounts else ""

    async def send_bundle(
        self,
        transactions: List[str],
        tip_lamports: int = 100000
    ) -> Optional[str]:
        try:
            bundle_id = f"bundle_{int(time.time() * 1000)}"
            
            async with self._session.post(
                f"{self.jito_url}/api/v1/bundles",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "sendBundle",
                    "params": [transactions, {"encoding": "base64"}]
                },
                headers={"Content-Type": "application/json"}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("result")
                else:
                    logger.error(f"Jito bundle failed: {resp.status} - {await resp.text()}")
        except Exception as e:
            logger.error(f"Jito bundle error: {e}")
        return None

    async def get_bundle_status(self, bundle_id: str) -> Optional[Dict]:
        try:
            async with self._session.post(
                f"{self.jito_url}/api/v1/bundles",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getBundleStatuses",
                    "params": [[bundle_id]]
                }
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("result", {})
        except Exception as e:
            logger.error(f"Jito bundle status error: {e}")
        return None


class SolanaTransactionBuilder:
    def __init__(self, rpc: RPCManager, keypair: nacl.signing.SigningKey):
        self.rpc = rpc
        self.keypair = keypair
        self.public_key = keypair.verify_key.encode(encoder=nacl.encoding.Base64Encoder).decode()

    async def get_recent_blockhash(self) -> Optional[str]:
        try:
            result = await self.rpc.request("getLatestBlockhash", [{"commitment": "processed"}])
            return result.get("value", {}).get("blockhash")
        except Exception as e:
            logger.error(f"Failed to get recent blockhash: {e}")
        return None

    def sign_transaction(self, transaction_bytes: bytes) -> str:
        signed = self.keypair.sign(transaction_bytes)
        return base64.b64encode(signed.signature + transaction_bytes).decode()

    def sign_versioned_transaction(self, versioned_tx_bytes: bytes) -> str:
        return self.sign_transaction(versioned_tx_bytes)


class ExecutionEngine:
    def __init__(
        self,
        chain_config: ChainConfig,
        rpc: RPCManager,
        jupiter: JupiterClient,
        jito: JitoClient,
        tx_builder: SolanaTransactionBuilder,
        counterfactual_lab
    ):
        self.chain_config = chain_config
        self.rpc = rpc
        self.jupiter = jupiter
        self.jito = jito
        self.tx_builder = tx_builder
        self.counterfactual_lab = counterfactual_lab
        
        self.pending_txs: Dict[str, SwapTransaction] = {}
        self.execution_history: deque = deque(maxlen=10000)
        self.route_performance: Dict[RouteType, Dict] = defaultdict(lambda: {
            "total": 0, "landed": 0, "failed": 0, "avg_latency": 0, "avg_slippage": 0
        })
        
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._jito_poll_task: Optional[asyncio.Task] = None

    async def start(self):
        await self.jupiter.start()
        await self.jito.start()
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_pending())
        self._jito_poll_task = asyncio.create_task(self._poll_jito_bundles())

    async def stop(self):
        self._running = False
        for task in [self._monitor_task, self._jito_poll_task]:
            if task:
                task.cancel()
        await self.jupiter.stop()
        await self.jito.stop()

    async def execute_swap(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int = 100,
        priority_fee: int = 5000,
        jito_tip: int = 100000,
        use_jito: bool = False,
        route_type: RouteType = RouteType.JUPITER_V6
    ) -> ExecutionResult:
        start_time = time.time()
        
        quote = await self.jupiter.get_quote(
            input_mint, output_mint, amount, slippage_bps
        )
        if not quote:
            return ExecutionResult(success=False, error="No quote available")
        
        swap_tx = await self.jupiter.get_swap_transaction(
            quote, self.tx_builder.public_key,
            prioritization_fee_lamports=priority_fee
        )
        if not swap_tx:
            return ExecutionResult(success=False, error="Failed to build transaction")
        
        swap_tx.priority_fee = priority_fee
        swap_tx.jito_tip = jito_tip if use_jito else 0
        swap_tx.route_type = route_type
        
        signed_tx = self.tx_builder.sign_versioned_transaction(
            base64.b64decode(swap_tx.transaction)
        )
        
        if use_jito and jito_tip > 0:
            tip_account = self.jito.get_random_tip_account()
            bundle_id = await self.jito.send_bundle([signed_tx], jito_tip)
            if bundle_id:
                swap_tx.signature = bundle_id
                self.pending_txs[bundle_id] = swap_tx
                return ExecutionResult(
                    success=True,
                    signature=bundle_id,
                    input_amount=amount,
                    output_amount=quote.output_amount,
                    priority_fee=priority_fee,
                    jito_tip=jito_tip,
                    latency_ms=int((time.time() - start_time) * 1000),
                    route_type=route_type
                )
        
        sig = await self._send_raw_transaction(signed_tx)
        if sig:
            swap_tx.signature = sig
            swap_tx.status = TransactionStatus.SUBMITTED
            self.pending_txs[sig] = swap_tx
            return ExecutionResult(
                success=True,
                signature=sig,
                input_amount=amount,
                output_amount=quote.output_amount,
                priority_fee=priority_fee,
                jito_tip=0,
                latency_ms=int((time.time() - start_time) * 1000),
                route_type=route_type
            )
        
        return ExecutionResult(success=False, error="Failed to send transaction")

    async def _send_raw_transaction(self, signed_tx: str) -> Optional[str]:
        try:
            result = await self.rpc.request("sendTransaction", [
                signed_tx,
                {"encoding": "base64", "skipPreflight": False, "maxRetries": 3}
            ])
            return result
        except Exception as e:
            logger.error(f"Send transaction error: {e}")
        return None

    async def _monitor_pending(self):
        while self._running:
            try:
                to_remove = []
                for sig, tx in self.pending_txs.items():
                    if tx.status in [TransactionStatus.LANDED, TransactionStatus.FAILED]:
                        to_remove.append(sig)
                        continue
                    
                    if time.time() - tx.submitted_at > 60:
                        tx.status = TransactionStatus.TIMEOUT
                        to_remove.append(sig)
                        continue
                    
                    status = await self._check_transaction_status(sig)
                    if status == "confirmed":
                        tx.status = TransactionStatus.LANDED
                        tx.landed_at = time.time()
                        await self._record_landing(tx)
                        to_remove.append(sig)
                    elif status == "failed":
                        tx.status = TransactionStatus.FAILED
                        to_remove.append(sig)
                
                for sig in to_remove:
                    self.pending_txs.pop(sig, None)
            except Exception as e:
                logger.error(f"Monitor pending error: {e}")
            await asyncio.sleep(1)

    async def _check_transaction_status(self, signature: str) -> Optional[str]:
        try:
            result = await self.rpc.request("getSignatureStatuses", [[signature]])
            statuses = result.get("value", [])
            if statuses and statuses[0]:
                status = statuses[0]
                if status.get("confirmationStatus") == "confirmed":
                    return "confirmed"
                elif status.get("err"):
                    return "failed"
        except Exception:
            pass
        return None

    async def _poll_jito_bundles(self):
        while self._running:
            try:
                for sig, tx in list(self.pending_txs.items()):
                    if tx.route_type == RouteType.JITO_BUNDLE and tx.signature:
                        status = await self.jito.get_bundle_status(tx.signature)
                        if status:
                            bundle_status = status.get("value", [{}])[0]
                            if bundle_status.get("confirmation_status") == "confirmed":
                                tx.status = TransactionStatus.LANDED
                                tx.landed_at = time.time()
                                await self._record_landing(tx)
            except Exception as e:
                logger.error(f"Jito poll error: {e}")
            await asyncio.sleep(2)

    async def _record_landing(self, tx: SwapTransaction):
        result = ExecutionResult(
            success=True,
            signature=tx.signature,
            input_amount=0,
            output_amount=0,
            priority_fee=tx.priority_fee,
            jito_tip=tx.jito_tip,
            route_type=tx.route_type,
            landed=True,
            latency_ms=int((tx.landed_at - tx.submitted_at) * 1000) if tx.landed_at else 0
        )
        
        self.execution_history.append({
            "timestamp": time.time(),
            "result": result.__dict__,
            "tx": tx.__dict__
        })
        
        perf = self.route_performance[tx.route_type]
        perf["total"] += 1
        if result.landed:
            perf["landed"] += 1
        else:
            perf["failed"] += 1
        perf["avg_latency"] = (perf["avg_latency"] * (perf["total"] - 1) + result.latency_ms) / perf["total"]
        
        if self.counterfactual_lab:
            self.counterfactual_lab.record_execution(tx.signature or "", result.__dict__)

    async def execute_sell(
        self,
        token_mint: str,
        amount: int,
        slippage_bps: int = 500,
        priority_fee: int = 10000,
        jito_tip: int = 200000,
        use_jito: bool = True
    ) -> ExecutionResult:
        usdc = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        return await self.execute_swap(
            token_mint, usdc, amount, slippage_bps,
            priority_fee, jito_tip, use_jito
        )

    def get_route_stats(self) -> Dict:
        return {
            rt.value: {
                "total": stats["total"],
                "landed": stats["landed"],
                "failed": stats["failed"],
                "success_rate": stats["landed"] / max(stats["total"], 1),
                "avg_latency_ms": stats["avg_latency"]
            }
            for rt, stats in self.route_performance.items()
        }

    def get_recent_executions(self, limit: int = 100) -> List[Dict]:
        return list(self.execution_history)[-limit:]


class PriorityFeeOptimizer:
    def __init__(self):
        self.fee_history: deque = deque(maxlen=1000)
        self.landing_rates: Dict[int, float] = {}

    def record_attempt(self, priority_fee: int, landed: bool, latency_ms: int):
        self.fee_history.append({
            "fee": priority_fee,
            "landed": landed,
            "latency": latency_ms,
            "timestamp": time.time()
        })
        
        fees = [h for h in self.fee_history if h["fee"] == priority_fee]
        if fees:
            self.landing_rates[priority_fee] = sum(1 for f in fees if f["landed"]) / len(fees)

    def get_optimal_fee(self, expected_value: float, competition: float) -> int:
        base_fee = 5000
        
        if expected_value > 1000:
            base_fee = 20000
        elif expected_value > 100:
            base_fee = 10000
        
        if competition > 0.7:
            base_fee = int(base_fee * 1.5)
        elif competition > 0.5:
            base_fee = int(base_fee * 1.2)
        
        for fee in [5000, 10000, 20000, 50000, 100000]:
            rate = self.landing_rates.get(fee, 0)
            if rate > 0.8 and fee > base_fee:
                return fee
        
        return base_fee

    def get_jito_tip(self, expected_value: float, urgency: str) -> int:
        base_tip = 100000
        
        if urgency == "CRITICAL":
            base_tip = 500000
        elif urgency == "HIGH":
            base_tip = 300000
        elif urgency == "MEDIUM":
            base_tip = 200000
        
        if expected_value > 10000:
            base_tip = int(base_tip * 2)
        elif expected_value > 1000:
            base_tip = int(base_tip * 1.5)
        
        return base_tip
"""Solana execution with an immutable dry-run boundary.

The engine intentionally treats quote, submission, landing and fill as different
states. A quote is never recorded as a fill and live submission is impossible
while ``dry_run`` is true.
"""

import asyncio
import base64
import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import aiohttp
from solders.keypair import Keypair
from solders.message import to_bytes_versioned
from solders.transaction import VersionedTransaction

from src.chains.rpc_manager import ChainConfig, RPCManager

logger = logging.getLogger(__name__)

WSOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


class RouteType(Enum):
    JUPITER_V1 = "jupiter_v1"
    RAYDIUM_DIRECT = "raydium_direct"
    ORCA_DIRECT = "orca_direct"
    METEORA_DLMM = "meteora_dlmm"
    JITO_BUNDLE = "jito_bundle"


class TransactionStatus(Enum):
    QUOTED = "quoted"
    SIMULATED = "simulated"
    SUBMITTED = "submitted"
    LANDED = "landed"
    FILLED = "filled"
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
    route: List[Dict[str, Any]]
    route_type: RouteType
    fees_bps: int
    min_output_amount: int
    quote_time: float = field(default_factory=time.time)
    raw_quote: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SwapTransaction:
    transaction: str
    last_valid_block_height: int
    fee_payer: str
    quote: SwapQuote
    status: TransactionStatus = TransactionStatus.QUOTED
    signature: Optional[str] = None
    bundle_id: Optional[str] = None
    submitted_at: float = field(default_factory=time.time)
    landed_at: Optional[float] = None
    slot: Optional[int] = None
    error: Optional[str] = None
    priority_fee: int = 0
    jito_tip: int = 0
    route_type: RouteType = RouteType.JUPITER_V1


@dataclass
class ExecutionResult:
    success: bool
    status: TransactionStatus
    signature: Optional[str] = None
    bundle_id: Optional[str] = None
    input_amount: int = 0
    actual_input_amount: int = 0
    quoted_output_amount: int = 0
    filled_output_amount: int = 0
    native_balance_delta_lamports: int = 0
    slippage_bps: int = 0
    fees_paid: int = 0
    priority_fee: int = 0
    jito_tip: int = 0
    latency_ms: int = 0
    route_type: RouteType = RouteType.JUPITER_V1
    error: Optional[str] = None
    simulated: bool = False
    submitted: bool = False
    landed: bool = False
    filled: bool = False
    slot: Optional[int] = None

    @property
    def output_amount(self) -> int:
        """Compatibility alias that cannot confuse a live quote with a fill."""
        return self.filled_output_amount or (self.quoted_output_amount if self.simulated else 0)


class JupiterClient:
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.base_url = (base_url or os.getenv("JUPITER_API_URL") or "https://api.jup.ag/swap/v1").rstrip("/")
        self.api_key = api_key or os.getenv("JUPITER_API_KEY", "")
        self._session: Optional[aiohttp.ClientSession] = None
        # Jupiter's current free tier is 1 RPS with a key and keyless access is
        # 0.5 RPS. A single shared gate prevents the research observer, equity
        # marker and execution path from creating synchronized quote bursts.
        self._minimum_interval = 1.05 if self.api_key else 2.05
        self._next_request_at = 0.0
        self._rate_lock = asyncio.Lock()

    async def _enter_rate_limit(self):
        await self._rate_lock.acquire()
        try:
            delay = self._next_request_at - time.monotonic()
            if delay > 0:
                await asyncio.sleep(min(delay, 30.0))
            self._next_request_at = time.monotonic() + self._minimum_interval
        except BaseException:
            self._rate_lock.release()
            raise

    def _leave_rate_limit(self, retry_after: float = 0.0):
        if retry_after > 0:
            self._next_request_at = max(
                self._next_request_at, time.monotonic() + min(retry_after, 30.0),
            )
        self._rate_lock.release()

    async def start(self):
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
            connector=aiohttp.TCPConnector(limit=20),
            headers=headers,
        )

    async def stop(self):
        if self._session:
            await self._session.close()
            self._session = None

    async def get_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int = 100,
        swap_mode: str = "ExactIn",
        only_direct_routes: bool = False,
        as_legacy_transaction: bool = False,
        platform_fee_bps: int = 0,
    ) -> Optional[SwapQuote]:
        if not self._session:
            raise RuntimeError("Jupiter client is not started")
        if amount <= 0:
            return None
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(amount),
            "slippageBps": str(slippage_bps),
            "swapMode": swap_mode,
            "onlyDirectRoutes": str(only_direct_routes).lower(),
            "asLegacyTransaction": str(as_legacy_transaction).lower(),
            "platformFeeBps": str(platform_fee_bps),
            "restrictIntermediateTokens": "true",
        }
        await self._enter_rate_limit()
        try:
            async with self._session.get(f"{self.base_url}/quote", params=params) as resp:
                if resp.status != 200:
                    logger.warning("Jupiter quote DATA_BLOCKED: HTTP %s", resp.status)
                    if resp.status == 429:
                        try:
                            retry_after = float(resp.headers.get("Retry-After", self._minimum_interval * 2))
                        except ValueError:
                            retry_after = self._minimum_interval * 2
                        self._next_request_at = max(
                            self._next_request_at, time.monotonic() + min(retry_after, 30.0),
                        )
                    return None
                data = await resp.json()
                return self._parse_quote(data, input_mint, output_mint, amount)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            logger.warning("Jupiter quote DATA_BLOCKED: %s", exc)
            return None
        finally:
            self._leave_rate_limit()

    async def get_swap_transaction(
        self,
        quote: SwapQuote,
        user_public_key: str,
        *,
        priority_fee_lamports: int = 0,
        jito_tip_lamports: int = 0,
    ) -> Optional[SwapTransaction]:
        if not self._session:
            raise RuntimeError("Jupiter client is not started")
        payload: Dict[str, Any] = {
            "quoteResponse": quote.raw_quote,
            "userPublicKey": user_public_key,
            "wrapAndUnwrapSol": True,
            "dynamicComputeUnitLimit": True,
        }
        # Jupiter V1 accepts either an embedded priority fee or an embedded Jito
        # tip in /swap. Jito bundles only need the latter.
        if jito_tip_lamports > 0:
            payload["prioritizationFeeLamports"] = {"jitoTipLamports": jito_tip_lamports}
        elif priority_fee_lamports > 0:
            payload["prioritizationFeeLamports"] = priority_fee_lamports
        await self._enter_rate_limit()
        try:
            async with self._session.post(f"{self.base_url}/swap", json=payload) as resp:
                if resp.status != 200:
                    logger.warning("Jupiter transaction DATA_BLOCKED: HTTP %s", resp.status)
                    return None
                data = await resp.json()
                encoded = data.get("swapTransaction", "")
                if not encoded:
                    return None
                return SwapTransaction(
                    transaction=encoded,
                    last_valid_block_height=int(data.get("lastValidBlockHeight", 0)),
                    fee_payer=user_public_key,
                    quote=quote,
                    priority_fee=priority_fee_lamports,
                    jito_tip=jito_tip_lamports,
                    route_type=RouteType.JITO_BUNDLE if jito_tip_lamports else quote.route_type,
                )
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            logger.warning("Jupiter transaction DATA_BLOCKED: %s", exc)
            return None
        finally:
            self._leave_rate_limit()

    def _parse_quote(self, data: Dict[str, Any], input_mint: str, output_mint: str, amount: int) -> SwapQuote:
        out_amount = int(data.get("outAmount", 0))
        threshold = int(data.get("otherAmountThreshold", out_amount))
        route_plan = data.get("routePlan", [])
        route_fee = sum(int(step.get("swapInfo", {}).get("feeAmount", 0) or 0) for step in route_plan)
        return SwapQuote(
            input_mint=input_mint,
            output_mint=output_mint,
            input_amount=amount,
            output_amount=out_amount,
            price_impact_pct=float(data.get("priceImpactPct", 0) or 0),
            route=route_plan,
            route_type=RouteType.JUPITER_V1,
            fees_bps=int(route_fee * 10_000 / max(amount, 1)),
            min_output_amount=threshold,
            raw_quote=data,
        )


class JitoClient:
    """Jito Block Engine transport.

    Each JSON-RPC method has its own documented path; posting every method
    at ``/api/v1/bundles`` silently breaks status and tip-account lookups,
    which in turn makes a bundle look permanently unconfirmed. Endpoints are
    therefore routed per method rather than sharing one URL.

    A bundle id means the engine RECEIVED the bundle, never that it landed.
    """

    # Method -> documented path suffix on the Block Engine.
    METHOD_PATHS = {
        "sendBundle": "/api/v1/bundles",
        "getBundleStatuses": "/api/v1/getBundleStatuses",
        "getInflightBundleStatuses": "/api/v1/getInflightBundleStatuses",
        "getTipAccounts": "/api/v1/getTipAccounts",
        "sendTransaction": "/api/v1/transactions",
    }
    TIP_FLOOR_URL = "https://bundles.jito.wtf/api/v1/bundles/tip_floor"
    DEFAULT_REGIONS = (
        "https://mainnet.block-engine.jito.wtf",
        "https://dublin.mainnet.block-engine.jito.wtf",
        "https://amsterdam.mainnet.block-engine.jito.wtf",
        "https://frankfurt.mainnet.block-engine.jito.wtf",
        "https://london.mainnet.block-engine.jito.wtf",
        "https://ny.mainnet.block-engine.jito.wtf",
        "https://tokyo.mainnet.block-engine.jito.wtf",
    )

    def __init__(self, jito_url: Optional[str] = None, regions: Optional[List[str]] = None):
        configured = [value.strip().rstrip("/") for value in
                      os.getenv("JITO_BLOCK_ENGINE_URLS", "").split(",") if value.strip()]
        if regions:
            bases = [value.rstrip("/") for value in regions]
        elif configured:
            bases = configured
        elif jito_url:
            bases = [self._base_of(jito_url)]
        else:
            bases = list(self.DEFAULT_REGIONS)
        # Preserve order while removing duplicates so racing does not send the
        # same transaction to one region twice.
        self.regions: List[str] = list(dict.fromkeys(bases))
        self.jito_url = self.regions[0] + self.METHOD_PATHS["sendBundle"]
        # Which regions accepted a given bundle id, so status can be queried
        # where it was actually received.
        self._bundle_routes: Dict[str, List[str]] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._tip_accounts: List[str] = []

    @property
    def jito_urls(self) -> List[str]:
        """Bundle-submission URLs, one per configured region."""
        return [base + self.METHOD_PATHS["sendBundle"] for base in self.regions]

    @jito_urls.setter
    def jito_urls(self, urls: List[str]):
        self.regions = list(dict.fromkeys(self._base_of(url) for url in urls))
        self.jito_url = self.regions[0] + self.METHOD_PATHS["sendBundle"] if self.regions else ""

    @staticmethod
    def _base_of(url: str) -> str:
        trimmed = url.rstrip("/")
        for suffix in JitoClient.METHOD_PATHS.values():
            if trimmed.endswith(suffix):
                return trimmed[: -len(suffix)]
        return trimmed

    def endpoint(self, method: str, base: Optional[str] = None) -> str:
        return (base or self.regions[0]) + self.METHOD_PATHS[method]

    async def start(self):
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        self._tip_accounts = await self._rpc("getTipAccounts", []) or []

    async def stop(self):
        if self._session:
            await self._session.close()
            self._session = None

    async def _rpc(self, method: str, params: List[Any], base: Optional[str] = None) -> Any:
        return await self._rpc_at(self.endpoint(method, base), method, params)

    async def _rpc_at(self, url: str, method: str, params: List[Any]) -> Any:
        if not self._session:
            raise RuntimeError("Jito client is not started")
        try:
            async with self._session.post(
                url,
                json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                if data.get("error"):
                    logger.warning("Jito %s failed: %s", method, data["error"])
                    return None
                return data.get("result")
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            logger.warning("Jito %s DATA_BLOCKED: %s", method, exc)
            return None

    @staticmethod
    async def _first_valid(tasks: List["asyncio.Task"]) -> Any:
        """Return the first truthy result, cancelling the rest.

        asyncio.gather would block until every region answered, so the slowest
        relay set the latency of the whole submission even when a nearer one
        had already accepted. Racing to first receipt is the entire point of
        submitting to several regions.
        """
        pending = set(tasks)
        winner = None
        try:
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    try:
                        value = task.result()
                    except Exception:
                        continue
                    if value:
                        return value
                    winner = winner or None
        finally:
            for task in pending:
                task.cancel()
        return winner

    async def send_bundle(self, transactions: List[str]) -> Optional[str]:
        """Race one already-signed bundle across regions; first receipt wins.

        The same signed payload is idempotent on chain: whichever region wins
        the auction, the identical transaction lands at most once. Building a
        differently-signed payload per region would risk a double fill.
        """
        urls = self.jito_urls
        tasks = [asyncio.ensure_future(self._rpc_at(url, "sendBundle",
                                                    [transactions, {"encoding": "base64"}]))
                 for url in urls]
        bundle_id = await self._first_valid(tasks)
        if isinstance(bundle_id, str) and bundle_id:
            self._bundle_routes[bundle_id] = list(urls)
            return bundle_id
        return None

    async def send_transaction(self, transaction: str) -> Optional[str]:
        """Single-transaction lane; often lands as well as a 1-tx bundle."""
        tasks = [asyncio.ensure_future(
            self._rpc_at(self.endpoint("sendTransaction", base), "sendTransaction",
                         [transaction, {"encoding": "base64"}])) for base in self.regions]
        signature = await self._first_valid(tasks)
        return signature if isinstance(signature, str) and signature else None

    async def get_bundle_status(self, bundle_id: str) -> Optional[Dict[str, Any]]:
        """Landed status. Queries every region: only one auction accepted it."""
        results = await asyncio.gather(
            *(self._rpc("getBundleStatuses", [[bundle_id]], base) for base in self.regions),
            return_exceptions=True,
        )
        for value in results:
            if isinstance(value, dict) and (value.get("value") or []):
                return value
        return None

    async def get_inflight_bundle_status(self, bundle_id: str) -> Optional[Dict[str, Any]]:
        """Pre-landing state, distinguishing 'still pending' from 'dropped'."""
        results = await asyncio.gather(
            *(self._rpc("getInflightBundleStatuses", [[bundle_id]], base) for base in self.regions),
            return_exceptions=True,
        )
        for value in results:
            if isinstance(value, dict) and (value.get("value") or []):
                return value
        return None

    async def get_tip_floor_lamports(self, percentile: int = 75) -> Optional[int]:
        """Observed landed-tip floor in lamports, or None when unavailable.

        Bidding a hand-picked constant either overpays or silently never
        lands; this reports what actually cleared recently so the tip can be
        chosen against evidence.
        """
        if not self._session or percentile not in {25, 50, 75, 95, 99}:
            return None
        key = f"landed_tips_{percentile}th_percentile"
        try:
            async with self._session.get(self.TIP_FLOOR_URL) as resp:
                if resp.status != 200:
                    return None
                payload = await resp.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            logger.warning("Jito tip floor DATA_BLOCKED: %s", exc)
            return None
        row = payload[0] if isinstance(payload, list) and payload else payload
        if not isinstance(row, dict) or row.get(key) is None:
            return None
        return max(0, int(float(row[key]) * 1e9))  # reported in SOL


class SolanaTransactionBuilder:
    def __init__(self, rpc: RPCManager, keypair: Keypair):
        self.rpc = rpc
        self.keypair = keypair
        self.public_key = str(keypair.pubkey())

    def sign_versioned_transaction(self, versioned_tx_bytes: bytes) -> str:
        tx = VersionedTransaction.from_bytes(versioned_tx_bytes)
        required = tx.message.header.num_required_signatures
        signer_keys = list(tx.message.account_keys[:required])
        try:
            signer_index = signer_keys.index(self.keypair.pubkey())
        except ValueError as exc:
            raise ValueError("wallet is not a required signer of the Jupiter transaction") from exc
        signatures = list(tx.signatures)
        if len(signatures) != required:
            raise ValueError("malformed VersionedTransaction signature vector")
        signatures[signer_index] = self.keypair.sign_message(to_bytes_versioned(tx.message))
        signed = VersionedTransaction.populate(tx.message, signatures)
        if not signed.verify_with_results()[signer_index]:
            raise ValueError("VersionedTransaction signature verification failed")
        return base64.b64encode(bytes(signed)).decode("ascii")


class ExecutionEngine:
    def __init__(
        self,
        chain_config: ChainConfig,
        rpc: RPCManager,
        jupiter: JupiterClient,
        jito: JitoClient,
        tx_builder: SolanaTransactionBuilder,
        counterfactual_lab: Any,
        *,
        dry_run: bool = True,
        confirmation_timeout: float = 45.0,
    ):
        self.chain_config = chain_config
        self.rpc = rpc
        self.jupiter = jupiter
        self.jito = jito
        self.tx_builder = tx_builder
        self.counterfactual_lab = counterfactual_lab
        self.dry_run = bool(dry_run)
        self.confirmation_timeout = confirmation_timeout
        self.execution_history: deque = deque(maxlen=10_000)
        self.route_performance: Dict[RouteType, Dict[str, float]] = defaultdict(
            lambda: {"total": 0, "landed": 0, "filled": 0, "failed": 0, "avg_latency": 0}
        )

    async def start(self):
        await self.jupiter.start()
        await self.jito.start()

    async def stop(self):
        await self.jupiter.stop()
        await self.jito.stop()

    async def execute_swap(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int = 100,
        priority_fee: int = 5_000,
        jito_tip: int = 100_000,
        use_jito: bool = False,
        decision_id: Optional[str] = None,
    ) -> ExecutionResult:
        started = time.time()
        if amount <= 0 or slippage_bps <= 0 or slippage_bps > 2_000:
            return ExecutionResult(False, TransactionStatus.REJECTED, error="hard execution invariant failed")
        quote = await self.jupiter.get_quote(input_mint, output_mint, amount, slippage_bps)
        if not quote or quote.output_amount <= 0:
            return ExecutionResult(False, TransactionStatus.REJECTED, error="no executable quote")

        if self.dry_run:
            result = ExecutionResult(
                success=True,
                status=TransactionStatus.SIMULATED,
                input_amount=amount,
                actual_input_amount=amount,
                quoted_output_amount=quote.output_amount,
                slippage_bps=slippage_bps,
                latency_ms=int((time.time() - started) * 1000),
                route_type=quote.route_type,
                simulated=True,
            )
            self._record(result, decision_id)
            return result

        if os.getenv("ALLOW_LIVE_TRADING", "").lower() != "yes-i-understand":
            return ExecutionResult(
                False,
                TransactionStatus.REJECTED,
                error="live submission is locked; ALLOW_LIVE_TRADING acknowledgement absent",
            )

        if use_jito:
            # Bid against what actually cleared recently rather than a fixed
            # constant. One data-driven tip only: a differently signed
            # escalation ladder could double-fill if an earlier attempt lands
            # late, so the tip is chosen once, before signing.
            observed_tip = await self.jito.get_tip_floor_lamports(75)
            if observed_tip:
                jito_tip = min(max(jito_tip, observed_tip), 5_000_000)
        swap_tx = await self.jupiter.get_swap_transaction(
            quote,
            self.tx_builder.public_key,
            priority_fee_lamports=0 if use_jito else priority_fee,
            jito_tip_lamports=jito_tip if use_jito else 0,
        )
        if not swap_tx:
            return ExecutionResult(False, TransactionStatus.REJECTED, error="transaction build failed")
        try:
            signed_tx = self.tx_builder.sign_versioned_transaction(base64.b64decode(swap_tx.transaction))
        except (ValueError, TypeError) as exc:
            return ExecutionResult(False, TransactionStatus.REJECTED, error=f"signing failed: {exc}")

        signature: Optional[str] = None
        bundle_id: Optional[str] = None
        if use_jito:
            bundle_id = await self.jito.send_bundle([signed_tx])
            if not bundle_id:
                return ExecutionResult(False, TransactionStatus.FAILED, error="Jito bundle rejected")
            swap_tx.bundle_id = bundle_id
            swap_tx.route_type = RouteType.JITO_BUNDLE
            signature = await self._wait_for_bundle(bundle_id)
        else:
            signature = await self._send_raw_transaction(signed_tx)
        if not signature:
            return ExecutionResult(
                False,
                TransactionStatus.TIMEOUT,
                bundle_id=bundle_id,
                submitted=bool(bundle_id),
                error="submitted but no landed transaction was confirmed",
            )

        fill = await self._wait_for_fill(signature, input_mint, output_mint)
        status = (
            TransactionStatus.FILLED if fill.get("filled")
            else TransactionStatus.LANDED if fill.get("landed")
            else TransactionStatus.TIMEOUT
        )
        result = ExecutionResult(
            success=bool(fill.get("filled")),
            status=status,
            signature=signature,
            bundle_id=bundle_id,
            input_amount=amount,
            actual_input_amount=int(fill.get("input_amount", 0)),
            quoted_output_amount=quote.output_amount,
            filled_output_amount=int(fill.get("output_amount", 0)),
            native_balance_delta_lamports=int(fill.get("native_balance_delta_lamports", 0)),
            slippage_bps=slippage_bps,
            fees_paid=int(fill.get("fee", 0)),
            priority_fee=0 if use_jito else priority_fee,
            jito_tip=jito_tip if use_jito else 0,
            latency_ms=int((time.time() - started) * 1000),
            route_type=swap_tx.route_type,
            submitted=True,
            landed=bool(fill.get("landed")),
            filled=bool(fill.get("filled")),
            slot=fill.get("slot"),
            error=None if fill.get("filled") else "landed transaction had no verified output balance delta",
        )
        self._record(result, decision_id)
        return result

    async def _send_raw_transaction(self, signed_tx: str) -> Optional[str]:
        try:
            return await self.rpc.request(
                "sendTransaction",
                [signed_tx, {"encoding": "base64", "skipPreflight": False, "maxRetries": 3}],
            )
        except Exception as exc:
            logger.error("Send transaction failed: %s", exc)
            return None

    async def _wait_for_bundle(self, bundle_id: str) -> Optional[str]:
        deadline = time.monotonic() + self.confirmation_timeout
        while time.monotonic() < deadline:
            status = await self.jito.get_bundle_status(bundle_id)
            values = (status or {}).get("value") or []
            if values and values[0]:
                item = values[0]
                if item.get("err"):
                    return None
                confirmation = item.get("confirmationStatus") or item.get("confirmation_status")
                transactions = item.get("transactions") or []
                if confirmation in {"confirmed", "finalized"} and transactions:
                    return transactions[0]
            await asyncio.sleep(0.5)
        return None

    async def _wait_for_fill(self, signature: str, input_mint: str, output_mint: str) -> Dict[str, Any]:
        deadline = time.monotonic() + self.confirmation_timeout
        while time.monotonic() < deadline:
            try:
                tx = await self.rpc.request(
                    "getTransaction",
                    [signature, {"encoding": "jsonParsed", "commitment": "confirmed", "maxSupportedTransactionVersion": 0}],
                )
                if tx:
                    meta = tx.get("meta") or {}
                    if meta.get("err") is not None:
                        return {"landed": True, "filled": False, "slot": tx.get("slot"), "fee": meta.get("fee", 0)}
                    output = self._token_balance_delta(meta, output_mint, self.tx_builder.public_key)
                    input_used = self._token_balance_decrease(meta, input_mint, self.tx_builder.public_key)
                    native_delta = self._native_balance_delta(tx, self.tx_builder.public_key)
                    return {
                        "landed": True,
                        "filled": output > 0,
                        "output_amount": output,
                        "input_amount": input_used,
                        "native_balance_delta_lamports": native_delta,
                        "slot": tx.get("slot"),
                        "fee": meta.get("fee", 0),
                    }
            except Exception:
                pass
            await asyncio.sleep(0.5)
        return {"landed": False, "filled": False}

    @staticmethod
    def _token_balance_delta(meta: Dict[str, Any], mint: str, owner: str) -> int:
        def total(entries: List[Dict[str, Any]]) -> int:
            value = 0
            for item in entries or []:
                if item.get("mint") == mint and item.get("owner") == owner:
                    value += int(item.get("uiTokenAmount", {}).get("amount", 0) or 0)
            return value
        return max(0, total(meta.get("postTokenBalances", [])) - total(meta.get("preTokenBalances", [])))

    @staticmethod
    def _token_balance_decrease(meta: Dict[str, Any], mint: str, owner: str) -> int:
        def total(entries: List[Dict[str, Any]]) -> int:
            return sum(
                int(item.get("uiTokenAmount", {}).get("amount", 0) or 0)
                for item in entries or [] if item.get("mint") == mint and item.get("owner") == owner
            )
        return max(0, total(meta.get("preTokenBalances", [])) - total(meta.get("postTokenBalances", [])))

    @staticmethod
    def _native_balance_delta(tx: Dict[str, Any], owner: str) -> int:
        message = (((tx.get("transaction") or {}).get("message")) or {})
        keys = message.get("accountKeys") or []
        normalized = [item.get("pubkey") if isinstance(item, dict) else item for item in keys]
        try:
            index = normalized.index(owner)
        except ValueError:
            return 0
        meta = tx.get("meta") or {}
        before = meta.get("preBalances") or []
        after = meta.get("postBalances") or []
        if index >= len(before) or index >= len(after):
            return 0
        return int(after[index]) - int(before[index])

    def _record(self, result: ExecutionResult, decision_id: Optional[str]):
        result_data = result.__dict__.copy()
        result_data["status"] = result.status.value
        result_data["route_type"] = result.route_type.value
        data = {"timestamp": time.time(), "decision_id": decision_id, "result": result_data}
        self.execution_history.append(data)
        perf = self.route_performance[result.route_type]
        perf["total"] += 1
        perf["landed"] += int(result.landed)
        perf["filled"] += int(result.filled)
        perf["failed"] += int(not result.success)
        perf["avg_latency"] = (perf["avg_latency"] * (perf["total"] - 1) + result.latency_ms) / perf["total"]
        if self.counterfactual_lab:
            self.counterfactual_lab.record_execution(decision_id or result.signature or "unknown", result_data)

    async def execute_sell(self, token_mint: str, amount: int, **kwargs: Any) -> ExecutionResult:
        return await self.execute_swap(token_mint, USDC_MINT, amount, **kwargs)

    def get_route_stats(self) -> Dict[str, Dict[str, float]]:
        return {
            route.value: {
                **stats,
                "landing_rate": stats["landed"] / max(stats["total"], 1),
                "fill_rate": stats["filled"] / max(stats["total"], 1),
            }
            for route, stats in self.route_performance.items()
        }

    def get_recent_executions(self, limit: int = 100) -> List[Dict[str, Any]]:
        return list(self.execution_history)[-limit:]


class PriorityFeeOptimizer:
    def __init__(self):
        self.fee_history: deque = deque(maxlen=1_000)
        self.landing_rates: Dict[int, float] = {}

    def record_attempt(self, priority_fee: int, landed: bool, latency_ms: int):
        self.fee_history.append({"fee": priority_fee, "landed": landed, "latency": latency_ms})
        attempts = [item for item in self.fee_history if item["fee"] == priority_fee]
        self.landing_rates[priority_fee] = sum(item["landed"] for item in attempts) / len(attempts)

    def get_optimal_fee(self, expected_value: float, competition: float) -> int:
        base = 20_000 if expected_value > 1_000 else 10_000 if expected_value > 100 else 5_000
        if competition > 0.7:
            base = int(base * 1.5)
        elif competition > 0.5:
            base = int(base * 1.2)
        viable = [fee for fee, rate in self.landing_rates.items() if rate >= 0.8]
        return min(viable, default=base)

    def get_jito_tip(self, expected_value: float, urgency: str) -> int:
        tip = {"CRITICAL": 500_000, "HIGH": 300_000, "MEDIUM": 200_000}.get(urgency, 100_000)
        return int(tip * (2 if expected_value > 10_000 else 1.5 if expected_value > 1_000 else 1))

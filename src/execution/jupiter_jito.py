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
from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price
from solders.hash import Hash
from solders.instruction import AccountMeta as SoldersAccountMeta, Instruction
from solders.keypair import Keypair
from solders.message import MessageV0, to_bytes_versioned
from solders.pubkey import Pubkey
from solders.transaction import VersionedTransaction

from src.chains.pump_curve import quote_buy, quote_sell
from src.execution.landing_model import Attempt, LandingModel
from src.chains.blockhash import BlockhashCache
from src.execution.slot_value import urgency_adjusted_edge
from src.chains.pump_route import NativePumpRoute, PreparedInstruction, WSOL_MINT
from src.chains.pumpswap_curve import PumpSwapPoolState
from src.chains.pumpswap_curve import quote_buy as pool_quote_buy
from src.chains.pumpswap_curve import quote_sell as pool_quote_sell
from src.chains.pumpswap_route import PoolState, PumpSwapRoute
from src.chains.rpc_manager import ChainConfig, RPCManager

logger = logging.getLogger(__name__)

WSOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


class RouteType(Enum):
    PUMP_NATIVE = "pump_native"
    PUMPSWAP_NATIVE = "pumpswap_native"
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
    def __init__(self, rpc: RPCManager, keypair: Keypair,
                 blockhash_cache: Optional[BlockhashCache] = None):
        self.rpc = rpc
        self.keypair = keypair
        self.public_key = str(keypair.pubkey())
        # Continuously refreshed in the background. Fetching per transaction
        # put an RPC round trip inside the window the system exists to win,
        # and did it after the decision while the opportunity aged.
        self.blockhash_cache = blockhash_cache or BlockhashCache(rpc)
        # How often the cache could not vouch for its hash and the synchronous
        # fetch was paid anyway. Counted rather than hidden: a cache that
        # refuses on every trade has removed nothing.
        self.blockhash_fallbacks = 0
        self.last_blockhash_status = ""

    async def build_and_sign(self, instructions: List[Any], *,
                             compute_unit_limit: int = 0,
                             compute_unit_price_micro_lamports: int = 0) -> str:
        """Assemble a v0 transaction from our own instructions and sign it.

        The counterpart to `sign_versioned_transaction`, which signs somebody
        else's bytes. Here the message is built locally, so the signer is the
        fee payer by construction and there is no question of whether the
        wallet is a required signer of a transaction it did not compose.

        The compute budget instructions go first because the runtime reads
        them positionally, and they are set explicitly rather than left to the
        default: a Pump buy that lands with the default limit is a Pump buy
        that risks running out of compute in the middle of an init_if_needed,
        and paying for headroom is cheaper than losing the fill.
        """
        blockhash = await self._recent_blockhash()
        payer = self.keypair.pubkey()
        program_instructions: List[Instruction] = []
        if compute_unit_limit > 0:
            program_instructions.append(set_compute_unit_limit(int(compute_unit_limit)))
        if compute_unit_price_micro_lamports > 0:
            program_instructions.append(
                set_compute_unit_price(int(compute_unit_price_micro_lamports)))
        program_instructions.extend(instructions)
        message = MessageV0.try_compile(payer, program_instructions, [], blockhash)
        signed = VersionedTransaction(message, [self.keypair])
        return base64.b64encode(bytes(signed)).decode("ascii")

    async def _recent_blockhash(self) -> Hash:
        """A blockhash fresh enough to land, from cache when it can be vouched for.

        A stale blockhash is a transaction the cluster silently refuses, and
        that looks exactly like a transaction that lost a race -- so the cache
        does not merely hold the last value it saw. It refuses one it cannot
        vouch for (too old, or too near its lastValidBlockHeight), and a
        refusal falls through to the synchronous fetch. Slow is a cost;
        silently unlanded is a loss.
        """
        state = self.blockhash_cache.current()
        self.last_blockhash_status = state.status
        if state.ok:
            return Hash.from_string(state.blockhash)
        self.blockhash_fallbacks += 1
        if state.detail:
            logger.debug("blockhash cache refused (%s); fetching synchronously",
                         state.detail)
        response = await self.rpc.request(
            "getLatestBlockhash", [{"commitment": "confirmed"}])
        value = ((response or {}).get("value") or {}).get("blockhash")
        if not value:
            raise ValueError("no recent blockhash available")
        return Hash.from_string(str(value))

    def blockhash_report(self) -> Dict[str, Any]:
        """Whether the hot path is actually being served from cache."""
        return {**self.blockhash_cache.report(),
                "synchronous_fallbacks": self.blockhash_fallbacks,
                "last_status": self.last_blockhash_status}

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
        pump_route: Optional[NativePumpRoute] = None,
        pumpswap_route: Optional[PumpSwapRoute] = None,
    ):
        self.chain_config = chain_config
        self.rpc = rpc
        self.jupiter = jupiter
        self.jito = jito
        self.tx_builder = tx_builder
        self.counterfactual_lab = counterfactual_lab
        self.dry_run = bool(dry_run)
        self.confirmation_timeout = confirmation_timeout
        # The canonical T0 route: built locally from streamed curve state, with
        # no quote round trip. Jupiter keeps routing after migration, an
        # independent cross-check on our own pricing, and the fallback when the
        # curve state is stale -- and loses the job it should never have had,
        # which is being a mandatory dependency of a sub-second decision.
        self.pump_route = pump_route
        # The same route, on the other side of graduation. A migrated coin is
        # the same coin, and every piece of intelligence we hold about it
        # still applies -- so the one thing that should NOT change at
        # migration is whether we can price and execute it ourselves. Without
        # this, graduation silently demoted every position to a router round
        # trip, at exactly the point where the position is largest.
        self.pumpswap_route = pumpswap_route
        # Supplied by the desk, which owns the streamed curve state. The
        # engine does not subscribe to anything itself -- an execution
        # path that maintains its own view of the curve is a second
        # source of truth about the price we are about to trade at.
        self.curve_state_provider: Optional[Any] = None
        # Likewise for pool state: mint -> PumpSwapPoolState, maintained by
        # the desk off the PumpSwap event stream.
        self.pool_state_provider: Optional[Any] = None
        # Pool identity that the instruction needs but the reserve state does
        # not carry (vault addresses, coin_creator, mayhem flag). mint ->
        # PoolState, decoded from the Pool account.
        self.pool_account_provider: Optional[Any] = None
        self.native_route_attempts: Dict[str, int] = defaultdict(int)
        # Set explicitly rather than left to the runtime default: a Pump buy
        # that runs out of compute in the middle of an init_if_needed is a
        # lost fill, and headroom is cheaper than the fill.
        self.native_compute_unit_limit = 400_000
        # Stream-first reconciliation. The waiters are futures our own decode
        # path resolves when it sees one of our signatures; the intervals
        # govern the polling backstop, which starts tight and backs off rather
        # than sitting at a fixed half second.
        self._signature_waiters: Dict[str, Any] = {}
        self.reconcile_min_interval = 0.025
        self.reconcile_max_interval = 0.5
        self.stream_confirmations = 0
        self.poll_confirmations = 0
        # P(land | bid) from our own attempts. Every submission feeds it,
        # landed or not: a model fed only successes learns that everything
        # lands.
        self.landing_model = LandingModel()
        # What the last bid decision was, and whether it was measured
        # or fell back. Surfaced so a desk running on the ladder knows.
        self.last_bid: Dict[str, Any] = {}
        # Measured chain congestion in [0, 1], supplied by the runtime from
        # the network-health miner. Until this was wired the landing model
        # bucketed EVERY attempt as "unknown", so it could never learn that a
        # bid clearing in calm conditions misses in a rush -- which is the one
        # thing conditioning on congestion exists to learn.
        self.congestion_provider: Optional[Any] = None
        self.execution_history: deque = deque(maxlen=10_000)
        self.route_performance: Dict[RouteType, Dict[str, float]] = defaultdict(
            lambda: {"total": 0, "landed": 0, "filled": 0, "failed": 0, "avg_latency": 0}
        )

    async def start(self):
        await self.jupiter.start()
        await self.jito.start()
        # Started with the engine, not lazily on the first trade: a cache
        # whose first fetch happens under a decision has not moved the round
        # trip anywhere.
        cache = getattr(self.tx_builder, "blockhash_cache", None)
        if cache is not None:
            await cache.start()

    async def stop(self):
        await self.jupiter.stop()
        await self.jito.stop()
        cache = getattr(self.tx_builder, "blockhash_cache", None)
        if cache is not None:
            await cache.stop()

    def prepare_native_route(self, input_mint: str, output_mint: str, amount: int,
                             slippage_bps: int) -> Optional[PreparedInstruction]:
        """Build the instruction locally, or say why it could not be built.

        Two venues, one contract. Before graduation the trade is a bonding
        curve trade; after graduation it is a PumpSwap pool trade. Returns
        None only when NEITHER can answer -- that is the signal to fall back
        to a router, and it is deliberately narrow: a migrated coin used to
        take that path unconditionally, which meant graduation quietly ended
        native execution at the point where the position is largest.

        The protective bound is derived from the caller's slippage rather than
        invented: the sizing decision already chose the risk limit, and this
        only expresses it in the units the instruction takes.
        """
        buying = input_mint == WSOL_MINT
        mint = output_mint if buying else input_mint
        curve = self.curve_state_provider(mint) if self.curve_state_provider else None
        if curve is not None:
            return self._prepare_curve_route(mint, curve, buying, amount, slippage_bps)
        return self._prepare_pool_route(mint, buying, amount, slippage_bps)

    def _prepare_curve_route(self, mint: str, curve: Any, buying: bool, amount: int,
                             slippage_bps: int) -> Optional[PreparedInstruction]:
        """The pre-graduation route: `buy_v2` / `sell_v2` against the curve."""
        if self.pump_route is None:
            return None
        creator = getattr(curve, "creator", "")
        if not creator:
            return PreparedInstruction(
                status="DATA_BLOCKED",
                detail="curve state carries no creator; creator_vault underivable")
        user = self.tx_builder.public_key
        if buying:
            quote = quote_buy(curve, amount)
            if quote.data_status != "OK" or quote.output_amount <= 0:
                return PreparedInstruction(status="DATA_BLOCKED",
                                           detail=f"local buy quote: {quote.reason}")
            # amount is the token amount bought; max_sol_cost bounds the spend.
            return self.pump_route.build_buy(
                mint, creator, user, quote.output_amount,
                int(amount * (1 + slippage_bps / 10_000)))
        quote = quote_sell(curve, amount)
        if quote.data_status != "OK" or quote.output_amount <= 0:
            return PreparedInstruction(status="DATA_BLOCKED",
                                       detail=f"local sell quote: {quote.reason}")
        return self.pump_route.build_sell(
            mint, creator, user, amount,
            int(quote.output_amount * (1 - slippage_bps / 10_000)))

    def _prepare_pool_route(self, mint: str, buying: bool, amount: int,
                            slippage_bps: int) -> Optional[PreparedInstruction]:
        """The post-graduation route: PumpSwap `buy` / `sell` against the pool.

        Requires BOTH the decoded Pool account -- which carries the vaults, the
        coin_creator and the mayhem flag, none of which can be inferred from
        the event stream -- and live reserves. Missing either is a refusal,
        not a default: a fee recipient chosen from the wrong published set
        produces a transaction that is well formed and fails.
        """
        if self.pumpswap_route is None or self.pool_state_provider is None:
            return None
        reserves = self.pool_state_provider(mint)
        if reserves is None:
            return None
        account = self.pool_account_provider(mint) if self.pool_account_provider else None
        if account is None or not getattr(account, "ok", False):
            return PreparedInstruction(
                status="DATA_BLOCKED", venue="pumpswap",
                detail="pool account not decoded; vaults and coin_creator unavailable")
        user = self.tx_builder.public_key
        if buying:
            quote = pool_quote_buy(reserves, amount)
            if quote.data_status != "OK" or quote.output_amount <= 0:
                return PreparedInstruction(status="DATA_BLOCKED", venue="pumpswap",
                                           detail=f"local pool buy quote: {quote.reason}")
            # `buy` takes the base amount out and a cap on the QUOTE LEG, which
            # is the budget less the fee the protocol takes on top of it --
            # not the budget itself. Bounding with the budget would authorise
            # the pool to take the fee twice over.
            amm_leg = max(1, amount - int(quote.fee_amount))
            return self.pumpswap_route.build_buy(
                account, user, quote.output_amount,
                int(amm_leg * (1 + slippage_bps / 10_000)))
        quote = pool_quote_sell(reserves, amount)
        if quote.data_status != "OK" or quote.output_amount <= 0:
            return PreparedInstruction(status="DATA_BLOCKED", venue="pumpswap",
                                       detail=f"local pool sell quote: {quote.reason}")
        return self.pumpswap_route.build_sell(
            account, user, amount,
            int(quote.output_amount * (1 - slippage_bps / 10_000)))

    def native_route_report(self) -> Dict[str, Any]:
        """Whether the entry path is actually taking the local route.

        A native builder that exists and is never used is the same defect as
        one that was never written, and the only difference is that this one
        looks finished.
        """
        total = sum(self.native_route_attempts.values())
        prepared = self.native_route_attempts.get("prepared", 0)
        return {
            "status": "OK" if total else "DATA_BLOCKED",
            "attempts": total, "prepared": prepared,
            "prepared_share": (prepared / total) if total else None,
            "outcomes": dict(self.native_route_attempts),
            "landing_model": self.landing_model.report(),
            "last_bid": dict(self.last_bid),
            "reconciliation": {
                "stream_confirmations": self.stream_confirmations,
                "poll_confirmations": self.poll_confirmations,
                "watching": len(self._signature_waiters),
                "min_interval_s": self.reconcile_min_interval,
            },
            "route": self.pump_route.report() if self.pump_route else {
                "status": "DATA_BLOCKED", "detail": "no native route configured"},
            "pumpswap_route": self.pumpswap_route.report() if self.pumpswap_route else {
                "status": "DATA_BLOCKED", "detail": "no pumpswap route configured"},
            "pool_state_wired": self.pool_state_provider is not None,
            "pool_account_wired": self.pool_account_provider is not None,
            "blockhash": (self.tx_builder.blockhash_report()
                          if hasattr(self.tx_builder, "blockhash_report")
                          else {"status": "DATA_BLOCKED",
                                "detail": "builder holds no blockhash cache"}),
        }

    def _native_quote(self, input_mint: str, output_mint: str, amount: int,
                      venue: str = "pump_curve"):
        """The local quote backing a native trade. None when nothing can answer.

        The venue comes from the instruction that was actually built, not from
        a second look at the providers: quoting the curve for a trade whose
        accounts belong to the pool would be a valid-looking number about the
        wrong market.
        """
        if venue == "pumpswap":
            if self.pool_state_provider is None:
                return None
            buying = input_mint == WSOL_MINT
            reserves = self.pool_state_provider(output_mint if buying else input_mint)
            if reserves is None:
                return None
            quote = (pool_quote_buy(reserves, amount) if buying
                     else pool_quote_sell(reserves, amount))
            return quote if quote.data_status == "OK" and quote.output_amount > 0 else None
        if self.curve_state_provider is None:
            return None
        buying = input_mint == WSOL_MINT
        curve = self.curve_state_provider(output_mint if buying else input_mint)
        if curve is None:
            return None
        quote = quote_buy(curve, amount) if buying else quote_sell(curve, amount)
        if quote.data_status != "OK" or quote.output_amount <= 0:
            return None
        return quote

    async def _execute_native(self, native: PreparedInstruction, quote: Any,
                              amount: int, slippage_bps: int, started: float, *,
                              priority_fee: int, jito_tip: int, use_jito: bool,
                              decision_id: Optional[str],
                              input_mint: str = "", output_mint: str = "") -> ExecutionResult:
        """Sign and submit our own instruction. No quote call, no third-party build.

        The dry-run and live-lock gates sit here as well as on the Jupiter
        path, and deliberately AFTER construction: building the transaction is
        the part worth exercising in dry run, and a path whose safety gate is
        only on the other branch is a path that will one day be the branch
        taken.
        """
        # The route type follows the instruction. Reporting a pool fill as a
        # curve fill would put post-graduation execution quality into the
        # pre-graduation bucket, and the two are not the same market.
        route_type = (RouteType.PUMPSWAP_NATIVE if native.venue == "pumpswap"
                      else RouteType.PUMP_NATIVE)
        if self.dry_run:
            result = ExecutionResult(
                success=True, status=TransactionStatus.SIMULATED,
                input_amount=amount, actual_input_amount=amount,
                quoted_output_amount=quote.output_amount, slippage_bps=slippage_bps,
                latency_ms=int((time.time() - started) * 1000),
                route_type=route_type, simulated=True,
            )
            self.native_route_attempts["simulated"] += 1
            self.native_route_attempts[f"simulated:{native.venue}"] += 1
            self._record(result, decision_id)
            return result

        if os.getenv("ALLOW_LIVE_TRADING", "").lower() != "yes-i-understand":
            return ExecutionResult(
                False, TransactionStatus.REJECTED,
                error="live submission is locked; ALLOW_LIVE_TRADING acknowledgement absent")

        try:
            instruction = Instruction(
                Pubkey.from_string(native.program_id),
                bytes(native.data),
                [SoldersAccountMeta(Pubkey.from_string(meta.pubkey),
                                    meta.is_signer, meta.is_writable)
                 for meta in native.accounts],
            )
            signed = await self.tx_builder.build_and_sign(
                [instruction],
                compute_unit_limit=self.native_compute_unit_limit,
                # Jito bids through the tip account rather than the fee market,
                # so paying both would be paying twice for one race.
                compute_unit_price_micro_lamports=0 if use_jito else priority_fee,
            )
        except Exception as exc:
            self.native_route_attempts["build_failed"] += 1
            return ExecutionResult(False, TransactionStatus.REJECTED,
                                   error=f"native build failed: {exc}")

        self.native_route_attempts["submitted"] += 1
        result = await self._submit_signed(signed, amount, slippage_bps, started,
                                           jito_tip=jito_tip, use_jito=use_jito,
                                           route_type=route_type,
                                           input_mint=input_mint, output_mint=output_mint)
        result.quoted_output_amount = quote.output_amount
        self._record(result, decision_id)
        return result

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
        expected_edge_usd: float = 0.0,
        sol_price_usd: float = 0.0,
        slot_value: Optional[Any] = None,
    ) -> ExecutionResult:
        started = time.time()
        if amount <= 0 or slippage_bps <= 0 or slippage_bps > 2_000:
            return ExecutionResult(False, TransactionStatus.REJECTED, error="hard execution invariant failed")
        # The bid is chosen BEFORE the route branches, because the native
        # branch returns first and was therefore never reaching the landing
        # model at all -- the canonical fastest route was the one still using
        # the fixed ladder, while the learned bid applied only to the Jupiter
        # fallback. Exactly backwards.
        if use_jito:
            observed_tip = await self.jito.get_tip_floor_lamports(75)
            if observed_tip:
                jito_tip = min(max(jito_tip, observed_tip), 5_000_000)
            chosen = self.choose_bid(
                # The EDGE, not the notional. A $500 position is not $500 of
                # expected value, and bidding against the notional overpays
                # for a marginal trade and underpays for a good one -- the two
                # errors that matter, in the two directions that matter.
                expected_value_usd=float(expected_edge_usd or 0.0),
                sol_price_usd=float(sol_price_usd or 0.0),
                fallback_lamports=jito_tip,
                # How fast THIS opportunity decays. A slot costs almost
                # nothing on a launch drifting sideways and a third of the
                # edge on a curve moving every slot; one bid for both is the
                # error in both directions.
                slot_value=slot_value,
                congestion=self.current_congestion())
            if chosen.get("measured"):
                # The observed floor says what cleared; the curve says what is
                # worth paying. Under the floor is not a bid at all.
                jito_tip = min(max(observed_tip or 0, int(chosen["lamports"])), 5_000_000)
            self.last_bid = chosen

        native = self.prepare_native_route(input_mint, output_mint, amount, slippage_bps)
        if native is not None and native.ok:
            self.native_route_attempts["prepared"] += 1
            self.native_route_attempts[f"prepared:{native.venue}"] += 1
        elif native is not None:
            # Recorded rather than swallowed: a native route that is blocked on
            # every trade means the entry path is still paying two round trips
            # it was supposed to have stopped paying, and nothing else would
            # say so.
            self.native_route_attempts[f"blocked:{native.venue}:{native.status}"] += 1
        # The native route is taken when it builds, not merely prepared and
        # then ignored. Preparing an instruction and submitting somebody
        # else's transaction is the worst of both: it pays the construction
        # cost, keeps the round trips, and looks finished.
        if native is not None and native.ok:
            quote = self._native_quote(input_mint, output_mint, amount, native.venue)
            if quote is None:
                self.native_route_attempts["blocked:no_local_quote"] += 1
            else:
                return await self._execute_native(
                    native, quote, amount, slippage_bps, started,
                    priority_fee=priority_fee, jito_tip=jito_tip,
                    use_jito=use_jito, decision_id=decision_id,
                    input_mint=input_mint, output_mint=output_mint)

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

        # The tip was chosen once, before the route branched and before
        # signing: a differently signed escalation ladder could double-fill if
        # an earlier attempt lands late.
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

    async def _submit_signed(self, signed_tx: str, amount: int, slippage_bps: int,
                             started: float, *, jito_tip: int, use_jito: bool,
                             route_type: RouteType,
                             input_mint: str = "", output_mint: str = "") -> ExecutionResult:
        """Submit an already-signed transaction and reconcile what landed.

        Shared by the native route and anything else that composes its own
        transaction, so the submission, confirmation and fill-verification
        rules are written once. A second copy of this logic is a second place
        for "submitted" to drift away from "filled".
        """
        signature: Optional[str] = None
        bundle_id: Optional[str] = None
        if use_jito:
            bundle_id = await self.jito.send_bundle([signed_tx])
            if not bundle_id:
                return ExecutionResult(False, TransactionStatus.FAILED,
                                       error="Jito bundle rejected")
            signature = await self._wait_for_bundle(bundle_id)
        else:
            signature = await self._send_raw_transaction(signed_tx)
        if not signature:
            return ExecutionResult(
                False, TransactionStatus.TIMEOUT, bundle_id=bundle_id,
                submitted=bool(bundle_id),
                error="submitted but no landed transaction was confirmed")

        fill = await self._wait_for_fill(signature, input_mint, output_mint)
        status = (
            TransactionStatus.FILLED if fill.get("filled")
            else TransactionStatus.LANDED if fill.get("landed")
            else TransactionStatus.TIMEOUT
        )
        self.landing_model.record(Attempt(
            bid_lamports=int(jito_tip if use_jito else 0),
            landed=bool(fill.get("landed")),
            route=route_type.value,
            latency_ms=int((time.time() - started) * 1000),
            # Recorded with the conditions it was attempted under. An attempt
            # stored without them is pooled with every other, and the pooled
            # curve is the average of two regimes we never trade in.
            congestion=self.current_congestion()))
        return ExecutionResult(
            success=bool(fill.get("filled")), status=status, signature=signature,
            bundle_id=bundle_id, input_amount=amount,
            actual_input_amount=int(fill.get("input_amount", 0)),
            filled_output_amount=int(fill.get("output_amount", 0)),
            native_balance_delta_lamports=int(fill.get("native_balance_delta_lamports", 0)),
            slippage_bps=slippage_bps, fees_paid=int(fill.get("fee", 0)),
            jito_tip=jito_tip if use_jito else 0,
            latency_ms=int((time.time() - started) * 1000),
            route_type=RouteType.JITO_BUNDLE if use_jito else route_type,
            submitted=True, landed=bool(fill.get("landed")),
            filled=bool(fill.get("filled")), slot=fill.get("slot"),
            error=None if fill.get("filled")
            else "landed transaction had no verified output balance delta",
        )

    async def _send_raw_transaction(self, signed_tx: str) -> Optional[str]:
        try:
            return await self.rpc.request(
                "sendTransaction",
                [signed_tx, {"encoding": "base64", "skipPreflight": False, "maxRetries": 3}],
            )
        except Exception as exc:
            logger.error("Send transaction failed: %s", exc)
            return None

    def observe_signature(self, signature: str, slot: Optional[int] = None) -> bool:
        """Our own transaction has been seen on the stream.

        Called from the decode path, which already carries every signature on
        the programs we subscribe to. If one of them is ours, the transaction
        has landed and we know it at stream latency rather than at the next
        poll -- which is the difference between reconciling a fill in tens of
        milliseconds and reconciling it up to half a second late, on the one
        path where a stale position is the expensive kind.

        Synchronous and non-blocking, because a decode handler that awaits is
        a handler that drops the next event.
        """
        waiter = self._signature_waiters.get(signature)
        if waiter is None or waiter.done():
            return False
        waiter.set_result(slot)
        self.stream_confirmations += 1
        return True

    def _watch_signature(self, signature: str) -> "asyncio.Future":
        loop = asyncio.get_event_loop()
        waiter = loop.create_future()
        self._signature_waiters[signature] = waiter
        return waiter

    async def _await_landing(self, signature: str, deadline: float) -> bool:
        """Race the stream against a backing-off poll.

        The stream is the fast path and the poll is the backstop, not the
        other way round: a fixed 500ms poll detects a fill that landed at
        400ms somewhere between 100 and 500ms late, every time. The poll's
        first interval is short and then backs off, so the case where the
        stream misses it -- a route we do not subscribe to, a dropped
        connection -- still reconciles quickly rather than being punished for
        the stream's absence.
        """
        waiter = self._watch_signature(signature)
        interval = self.reconcile_min_interval
        try:
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                try:
                    await asyncio.wait_for(asyncio.shield(waiter),
                                           timeout=min(interval, remaining))
                    return True
                except asyncio.TimeoutError:
                    pass
                if await self._signature_landed(signature):
                    self.poll_confirmations += 1
                    return True
                interval = min(interval * 2, self.reconcile_max_interval)
            return False
        finally:
            self._signature_waiters.pop(signature, None)

    async def _signature_landed(self, signature: str) -> bool:
        """Has it landed? Cheapest question first, then the definitive one.

        `getSignatureStatuses` is a small call and answers directly. Where a
        node does not serve it, the presence of the transaction itself is the
        same answer arrived at more expensively -- and falling back rather
        than reporting "not landed" matters, because a missing status endpoint
        would otherwise look identical to a transaction that never landed.
        """
        try:
            response = await self.rpc.request(
                "getSignatureStatuses", [[signature], {"searchTransactionHistory": False}])
            values = ((response or {}).get("value") or [])
            status = values[0] if values else None
            if status:
                return str(status.get("confirmationStatus") or "") in {"confirmed", "finalized"}
        except Exception:
            pass
        try:
            tx = await self.rpc.request(
                "getTransaction",
                [signature, {"encoding": "jsonParsed", "commitment": "confirmed",
                             "maxSupportedTransactionVersion": 0}])
            return bool(tx)
        except Exception:
            return False

    async def _wait_for_bundle(self, bundle_id: str) -> Optional[str]:
        deadline = time.monotonic() + self.confirmation_timeout
        interval = self.reconcile_min_interval
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
            # Backing off rather than fixed: a bundle usually resolves in the
            # first few hundred milliseconds, and a fixed half-second cadence
            # finds that out up to half a second late every single time.
            await asyncio.sleep(interval)
            interval = min(interval * 2, self.reconcile_max_interval)
        return None

    async def _wait_for_fill(self, signature: str, input_mint: str, output_mint: str) -> Dict[str, Any]:
        """Reconcile what actually landed.

        Landing is detected by the stream where possible; the balance deltas
        still come from `getTransaction`, because knowing a transaction landed
        is not the same as knowing what it filled, and a position sized from
        an assumed fill is a position whose cost basis is fiction.
        """
        deadline = time.monotonic() + self.confirmation_timeout
        landed = await self._await_landing(signature, deadline)
        if not landed:
            return {"landed": False, "filled": False}
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
            # It landed; the details just have not propagated yet. Short waits,
            # because this is the window in which a position is open and
            # unaccounted for.
            await asyncio.sleep(self.reconcile_min_interval)
        return {"landed": True, "filled": False}

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

    def current_congestion(self) -> Optional[float]:
        """Measured congestion, or None. Never a default.

        None means unmeasured and the landing model buckets it as such. A
        default of "calm" would be the expensive direction of that error:
        it bids low into exactly the conditions where bidding low misses.
        """
        provider = self.congestion_provider
        if provider is None:
            return None
        try:
            value = provider()
        except Exception:
            return None
        return None if value is None else float(value)

    def choose_bid(self, expected_value_usd: float, sol_price_usd: float,
                   fallback_lamports: int, congestion: Optional[float] = None,
                   slot_value: Optional[Any] = None) -> Dict[str, Any]:
        """What to bid, from the landing curve where it can answer.

        The fallback ladder is used when the curve cannot, and the result says
        which happened -- so a desk running on guesses knows it is, rather
        than reading a number that looks measured.

        ``slot_value`` says how much of the edge one slot of delay destroys on
        THIS opportunity. Without it the bid is sized on the whole edge, which
        is what is at stake over the trade rather than over the race -- so an
        ordinary launch drifting sideways bids as hard as a curve moving every
        slot, which is indistinguishable from the fixed ladder this replaced.
        """
        raced = float(expected_value_usd)
        if slot_value is not None:
            raced = urgency_adjusted_edge(slot_value, expected_value_usd)
        recommendation = self.landing_model.recommend(
            raced, sol_price_usd, congestion)
        urgency = ({"slot_value": slot_value.to_dict(), "raced_value_usd": raced}
                   if slot_value is not None and hasattr(slot_value, "to_dict")
                   else {"slot_value": None, "raced_value_usd": raced})
        if recommendation.status == "OK":
            return {"lamports": recommendation.bid_lamports, "measured": True,
                    **urgency, **recommendation.to_dict()}
        return {"lamports": int(fallback_lamports), "measured": False,
                **urgency, **recommendation.to_dict()}

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

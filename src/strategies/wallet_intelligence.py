import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple
import json
import numpy as np

import aiohttp

from src.chains.rpc_manager import ChainConfig, RPCManager
from src.strategies.genealogy_graph import GenealogyGraph, WalletProfile, EntityType
from src.strategies.wallet_value import (
    FollowOutcome, WalletValue, WalletValueModel, executable_multiple,
)

logger = logging.getLogger(__name__)


class WalletRegime(Enum):
    GENERAL_HISTORY = "general_history"
    ULTRA_EARLY = "ultra_early"
    EARLY_CURVE = "early_curve"
    PRE_MIGRATION = "pre_migration"
    POST_MIGRATION = "post_migration"
    POLITICAL_MEME = "political_meme"
    HIGH_VOLUME_MANIA = "high_volume_mania"
    QUIET_REGIME = "quiet_regime"


@dataclass
class WalletRegimePerformance:
    wallet: str
    regime: WalletRegime
    trades: int = 0
    win_rate_2x: float = 0.0
    win_rate_5x: float = 0.0
    win_rate_10x: float = 0.0
    avg_entry_pct: float = 50.0
    avg_exit_pct: float = 50.0
    avg_hold_time: float = 0.0
    realized_pnl: float = 0.0
    rug_exposure: float = 0.0
    consistency: float = 0.0
    independence_score: float = 1.0
    last_updated: float = field(default_factory=time.time)


@dataclass
class WalletScore:
    wallet: str
    overall_score: float
    regime_scores: Dict[WalletRegime, float]
    early_entry_quality: float
    forward_return_quality: float
    consistency: float
    independence: float
    sample_size: int
    rug_exposure: float
    copy_crowding: float
    last_recalc: float = field(default_factory=time.time)
    rank: int = 0


class WalletIntelligenceEngine:
    def __init__(
        self,
        chain_config: ChainConfig,
        rpc: RPCManager,
        genealogy: GenealogyGraph,
        helius_key: str,
        min_trades_for_ranking: int = 20,
        recalc_interval_hours: int = 1
    ):
        self.chain_config = chain_config
        self.rpc = rpc
        self.genealogy = genealogy
        self.helius_key = helius_key
        self.min_trades = min_trades_for_ranking
        self.recalc_interval = recalc_interval_hours * 3600
        # A wallet the stream carried within this window needs no poll. Long
        # enough that a wallet trading normally stays stream-covered, short
        # enough that one going quiet on our programs is reconciled promptly.
        self.stream_coverage_seconds = 120.0
        self.max_reconcile_per_pass = 25
        self.reconcile_concurrency = 8
        self.reconcile_interval_seconds = 15.0
        
        self.regime_performances: Dict[str, Dict[WalletRegime, WalletRegimePerformance]] = defaultdict(dict)
        self.wallet_scores: Dict[str, WalletScore] = {}
        # What following a wallet is actually worth, in the same E[log W]
        # units as every other action the desk values. The composite score
        # above is kept for the callers that still read it and is no longer
        # what decides the watch list: its weights were chosen, and nothing
        # could tell you whether the list they produced made money.
        self.wallet_value = WalletValueModel()
        self.regime_classifier: Optional[Callable] = None
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._hunter_task: Optional[asyncio.Task] = None
        self._watcher_task: Optional[asyncio.Task] = None
        self._recalc_task: Optional[asyncio.Task] = None
        self._history_task: Optional[asyncio.Task] = None
        
        self._live_watch_wallets: Set[str] = set()
        self._recent_buys: deque = deque(maxlen=10000)
        self._recent_sells: deque = deque(maxlen=10000)
        self._seen_live_signatures: Set[str] = set()
        self._history_signatures: Dict[str, Set[str]] = defaultdict(set)
        self._social_wallet_candidates: Set[str] = set()
        self._token_launch_times: Dict[str, float] = {}
        self._token_migration_times: Dict[str, float] = {}
        self._history_candidates: deque = deque(maxlen=20_000)
        self._queued_history_wallets: Set[str] = set()
        self._history_evaluated_at: Dict[str, float] = {}
        self.data_status: Dict[str, str] = {}
        # When the geyser stream last carried each tracked wallet. A wallet the
        # stream covers does not need to be polled, and polling it anyway
        # spends the rate limit that the uncovered wallets need.
        self._stream_seen_at: Dict[str, float] = {}
        # Observation latency by path, so "our elite wallets are on a
        # two-second HTTP delay" is a measured number rather than a worry.
        self._observation_lag: Dict[str, Deque[float]] = {
            "stream": deque(maxlen=512), "poll": deque(maxlen=512)}
        
        self._helius_base = "https://api.helius.xyz/v0"

    async def start(self, initial_wallets: List[str] = None):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=50)
        )
        self._running = True
        
        if initial_wallets:
            self._live_watch_wallets.update(initial_wallets)
        
        self._hunter_task = asyncio.create_task(self._hunter_loop())
        self._watcher_task = asyncio.create_task(self._watcher_loop())
        self._recalc_task = asyncio.create_task(self._recalc_loop())
        self._history_task = asyncio.create_task(self._history_worker_loop())
        
        await self._initial_discovery()

    async def stop(self):
        self._running = False
        for task in [self._hunter_task, self._watcher_task, self._recalc_task, self._history_task]:
            if task:
                task.cancel()
        if self._session:
            await self._session.close()

    async def _initial_discovery(self):
        await self._discover_wallets_from_recent_launches()
        await self._recalculate_all_scores()

    async def _hunter_loop(self):
        while self._running:
            try:
                await self._discover_wallets_from_recent_launches()
                await self._discover_wallets_from_social()
                await self._recalculate_all_scores()
            except Exception as e:
                logger.error(f"Hunter loop error: {e}")
            await asyncio.sleep(self.recalc_interval)

    async def _watcher_loop(self):
        while self._running:
            try:
                await self._watch_live_wallets()
            except Exception as e:
                logger.error(f"Watcher loop error: {e}")
            # Reconciliation, not observation. The stream is the observation
            # path; hammering Helius every two seconds for wallets it already
            # covers bought nothing and spent the budget the uncovered ones
            # need.
            await asyncio.sleep(self.reconcile_interval_seconds)

    async def _recalc_loop(self):
        while self._running:
            try:
                await self._recalculate_all_scores()
                await self._update_live_watch_list()
            except Exception as e:
                logger.error(f"Recalc loop error: {e}")
            await asyncio.sleep(300)

    async def _history_worker_loop(self):
        """Continuously reconstruct observed buyer histories within free-provider limits."""
        while self._running:
            if not self._history_candidates:
                await asyncio.sleep(0.25)
                continue
            wallet, token = self._history_candidates.popleft()
            self._queued_history_wallets.discard(wallet)
            try:
                await self._evaluate_new_wallet(wallet, token)
                self._history_evaluated_at[wallet] = time.time()
            except Exception as exc:
                self.data_status[wallet] = f"DATA_BLOCKED: history worker: {exc}"
            await asyncio.sleep(1.1)

    def _queue_wallet_history(self, wallet: str, token: str):
        if not wallet or not 32 <= len(wallet) <= 44 or wallet in self._queued_history_wallets:
            return
        if time.time() - self._history_evaluated_at.get(wallet, 0) < 3600:
            return
        self._queued_history_wallets.add(wallet)
        self._history_candidates.append((wallet, token))

    async def _discover_wallets_from_recent_launches(self):
        # Launches are supplied by the validated Pump/Raydium program stream.
        # Helius does not expose a universal `/tokens/mintlist` endpoint.
        self.data_status["launch_discovery"] = "OK: program_stream"

    async def _analyze_token_early_buyers(self, token: str):
        if not token:
            return
        try:
            largest = await self.rpc.request("getTokenLargestAccounts", [token, {"commitment": "confirmed"}])
            token_accounts = [item.get("address") for item in (largest or {}).get("value", []) if item.get("address")]
            if not token_accounts:
                self.data_status[f"holders:{token}"] = "DATA_BLOCKED: no token accounts observed"
                return
            account_data = await self.rpc.request(
                "getMultipleAccounts", [token_accounts, {"encoding": "jsonParsed", "commitment": "confirmed"}],
            )
            for item in (account_data or {}).get("value", []):
                owner = (((item or {}).get("data") or {}).get("parsed") or {}).get("info", {}).get("owner")
                if owner and owner not in self.genealogy.wallets:
                    await self._evaluate_new_wallet(owner, token)
            self.data_status[f"holders:{token}"] = "OK"
        except Exception as e:
            self.data_status[f"holders:{token}"] = f"DATA_BLOCKED: {e}"
            # This was logged at debug, which the desk never enables by
            # default (LOG_LEVEL defaults to INFO) -- so every failure on
            # this path, for as long as it has existed, was invisible. It is
            # the entry point for the entire wallet-intelligence pipeline
            # (actor_graph, wallet_follow, capital_rotation all read zero
            # downstream of it), which is a strange thing to have been
            # debugging blind.
            logger.warning("Token holder analysis failed for %s: %s: %s",
                           token, type(e).__name__, e)

    async def analyze_token_early_buyers(self, token: str):
        """Public entrypoint used by the canonical launch stream."""
        await self._analyze_token_early_buyers(token)

    async def _evaluate_new_wallet(self, wallet: str, trigger_token: str):
        txs: List[Dict[str, Any]] = []
        try:
            if self.helius_key:
                async with self._session.get(
                    f"{self._helius_base}/addresses/{wallet}/transactions",
                    params={"api-key": self.helius_key, "limit": 100, "type": "SWAP"}
                ) as resp:
                    if resp.status == 200:
                        txs = await resp.json()
            if not txs:
                txs = await self._rpc_wallet_history(wallet)
                self.data_status[f"history_source:{wallet}"] = "OK: standard_solana_rpc"
            else:
                self.data_status[f"history_source:{wallet}"] = "OK: helius_enhanced"
            await self._build_wallet_history(wallet, txs)
        except Exception as e:
            self.data_status[wallet] = f"DATA_BLOCKED: wallet history unavailable: {e}"
            # Also silent at debug before now -- see the identical note on
            # _analyze_token_early_buyers's except block. This is the step
            # that actually fetches a wallet's trade history (Helius, then
            # the free RPC pool); a session/attribute error here (e.g. a
            # None self._session before start() has completed) would have
            # failed every single call and never shown once.
            logger.warning("Wallet history evaluation failed for %s: %s: %s",
                           wallet, type(e).__name__, e)

    #: Signatures per JSON-RPC batch. Providers cap batch size and oversized
    #: batches are rejected whole, so this stays well inside the common limit.
    RPC_HISTORY_BATCH = 25

    async def _rpc_wallet_history(self, wallet: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Provider-independent fallback using standard Solana transaction balance deltas.

        Batched, because this is the desk's single largest RPC consumer and it
        runs when the cheap path is already gone. `_analyze_token_early_buyers`
        calls it for up to twenty holders, so one token cost 20 x (1 + 50) =
        ~1,020 requests. Worse, it is a FALLBACK: it runs precisely when the
        Helius enhanced endpoint returned nothing, so the moment that provider
        hits its quota this path multiplies the same work by fifty and spends
        the remaining endpoints' quota too. That is the shape of the outage
        observed on 2026-08-28 -- all three Solana endpoints at 429, with
        getTransaction 75% of the refusals.

        One batch of 25 replaces 25 round trips. Same data, same commitment,
        ~50x fewer requests and one connection setup instead of fifty, which
        is latency the hot path was paying for as well as quota.
        """
        signatures = await self.rpc.request(
            "getSignaturesForAddress", [wallet, {"limit": limit, "commitment": "confirmed"}],
        )
        usable = [row for row in (signatures or [])
                  if row.get("signature") and not row.get("err")]
        if not usable:
            return []

        options = {"encoding": "jsonParsed", "commitment": "confirmed",
                   "maxSupportedTransactionVersion": 0}
        rows: List[Dict[str, Any]] = []
        for start in range(0, len(usable), self.RPC_HISTORY_BATCH):
            chunk = usable[start:start + self.RPC_HISTORY_BATCH]
            payload = [{"jsonrpc": "2.0", "id": index, "method": "getTransaction",
                        "params": [row["signature"], options]}
                       for index, row in enumerate(chunk)]
            try:
                results = await self.rpc.batch_request(payload)
            except Exception as exc:
                # Some free providers refuse batches outright -- publicnode
                # answers 403 to a JSON-RPC array while serving the same
                # method fine one call at a time. A refused batch therefore
                # degrades to bounded sequential singles through the
                # method-aware router (which knows who serves getTransaction)
                # instead of becoming a hole in this wallet's history that
                # reads downstream as "no trades".
                logger.debug("batched wallet history failed for %s (%s); "
                             "falling back to singles", wallet, exc)
                semaphore = asyncio.Semaphore(3)

                async def fetch_one(row):
                    async with semaphore:
                        try:
                            return await self.rpc.request(
                                "getTransaction", [row["signature"], options])
                        except Exception:
                            return None

                results = await asyncio.gather(
                    *(fetch_one(row) for row in chunk))
            for row, transaction in zip(chunk, results or []):
                converted = self._standard_tx_to_enhanced(wallet, row, transaction)
                if converted:
                    rows.append(converted)
        return rows

    @staticmethod
    def _standard_tx_to_enhanced(wallet: str, signature_row: Dict[str, Any], tx: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(tx, dict) or tx.get("meta") is None:
            return None
        meta = tx["meta"] or {}
        message = ((tx.get("transaction") or {}).get("message") or {})
        keys = message.get("accountKeys") or []
        addresses = [item.get("pubkey") if isinstance(item, dict) else item for item in keys]
        token_deltas: Dict[str, float] = defaultdict(float)
        for sign, balances in ((-1.0, meta.get("preTokenBalances") or []),
                               (1.0, meta.get("postTokenBalances") or [])):
            for balance in balances:
                if balance.get("owner") != wallet or not balance.get("mint"):
                    continue
                ui = ((balance.get("uiTokenAmount") or {}).get("uiAmountString"))
                try:
                    token_deltas[balance["mint"]] += sign * float(ui or 0)
                except (TypeError, ValueError):
                    continue
        transfers = []
        for mint, delta in token_deltas.items():
            if abs(delta) <= 1e-12:
                continue
            transfers.append({
                "mint": mint, "tokenAmount": abs(delta),
                "toUserAccount": wallet if delta > 0 else None,
                "fromUserAccount": wallet if delta < 0 else None,
            })
        native = []
        if wallet in addresses:
            index = addresses.index(wallet)
            pre = meta.get("preBalances") or []
            post = meta.get("postBalances") or []
            if index < len(pre) and index < len(post):
                delta = int(post[index]) - int(pre[index])
                # Remove the payer fee so buys are not understated.
                if index == 0 and delta < 0:
                    delta += int(meta.get("fee", 0) or 0)
                if delta:
                    native.append({
                        "amount": abs(delta),
                        "toUserAccount": wallet if delta > 0 else None,
                        "fromUserAccount": wallet if delta < 0 else None,
                    })
        if not transfers or not native:
            return None
        return {
            "signature": signature_row.get("signature"),
            "timestamp": tx.get("blockTime") or signature_row.get("blockTime") or 0,
            "tokenTransfers": transfers, "nativeTransfers": native,
        }

    async def _build_wallet_history(self, wallet: str, txs: List[Dict]):
        if wallet not in self.genealogy.wallets:
            self.genealogy.wallets[wallet] = WalletProfile(
                address=wallet,
                entity_type=EntityType.WALLET,
                first_seen=time.time(),
                last_seen=time.time()
            )
        
        wp = self.genealogy.wallets[wallet]
        
        unseen = [tx for tx in txs if tx.get("signature") and tx.get("signature") not in self._history_signatures[wallet]]
        reconstructed = self._reconstruct_wallet_trades(wallet, unseen)
        self._history_signatures[wallet].update(tx.get("signature") for tx in unseen if tx.get("signature"))
        wp.tx_count += reconstructed["swap_count"]
        wp.last_seen = max([float(tx.get("timestamp", 0) or 0) for tx in txs] + [wp.last_seen])
        classified_trades = 0
        for trade in reconstructed["closed_trades"]:
            token = trade["token"]
            if token not in wp.launches_participated:
                wp.launches_participated.append(token)
            self._attach_launch_relative_regime(trade)
            regime = self._classify_regime(trade)
            if regime is not None:
                await self._update_regime_performance(wallet, regime, trade)
                classified_trades += 1
        if classified_trades:
            self.data_status[wallet] = "OK"
        elif reconstructed["closed_trades"]:
            self.data_status[wallet] = "DATA_BLOCKED: closed trades lack PIT regime labels"
        else:
            self.data_status[wallet] = "DATA_BLOCKED: no complete comparable round trips"

    def _reconstruct_wallet_trades(self, wallet: str, txs: List[Dict]) -> Dict[str, Any]:
        """Reconstruct actual FIFO round trips from Helius transfer deltas."""
        positions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        closed: List[Dict[str, Any]] = []
        signatures: Set[str] = set()
        for tx in sorted(txs, key=lambda item: float(item.get("timestamp", 0) or 0)):
            normalized = self._normalize_swap(wallet, tx)
            if not normalized:
                continue
            signatures.add(normalized["signature"])
            token = normalized["token"]
            if normalized["side"] == "buy":
                positions[token].append({
                    "amount": normalized["amount"],
                    "remaining": normalized["amount"],
                    "cost": normalized["base_value"],
                    "base_unit": normalized["base_unit"],
                    "timestamp": normalized["timestamp"],
                })
                continue
            remaining_sell = normalized["amount"]
            total_sold = normalized["amount"]
            allocated_cost = 0.0
            first_entry = None
            comparable = True
            for lot in positions[token]:
                if remaining_sell <= 0:
                    break
                used = min(lot["remaining"], remaining_sell)
                if used <= 0:
                    continue
                allocated_cost += lot["cost"] * used / max(lot["amount"], 1e-12)
                first_entry = lot["timestamp"] if first_entry is None else min(first_entry, lot["timestamp"])
                comparable = comparable and lot["base_unit"] == normalized["base_unit"]
                lot["remaining"] -= used
                remaining_sell -= used
            matched = total_sold - remaining_sell
            if matched <= 0 or allocated_cost <= 0 or not comparable:
                continue
            proceeds = normalized["base_value"] * matched / max(total_sold, 1e-12)
            closed.append({
                "token": token,
                "entry_timestamp": first_entry,
                "timestamp": normalized["timestamp"],
                "entry_timestamp": first_entry if first_entry is not None else normalized["timestamp"],
                "multiple": proceeds / allocated_cost,
                "realized_pnl": proceeds - allocated_cost,
                "hold_time": normalized["timestamp"] - (first_entry or normalized["timestamp"]),
                "base_unit": normalized["base_unit"],
                "data_status": "OK",
            })
        return {"swap_count": len(signatures), "closed_trades": closed}

    @staticmethod
    def _normalize_swap(wallet: str, tx: Dict) -> Optional[Dict[str, Any]]:
        transfers = tx.get("tokenTransfers", []) or []
        received = [item for item in transfers if item.get("toUserAccount") == wallet]
        sent = [item for item in transfers if item.get("fromUserAccount") == wallet]
        stable_mints = {
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        }
        token_received = next((item for item in received if item.get("mint") not in stable_mints), None)
        token_sent = next((item for item in sent if item.get("mint") not in stable_mints), None)
        side = "buy" if token_received else "sell" if token_sent else None
        token_transfer = token_received or token_sent
        if not side or not token_transfer:
            return None
        base_value = 0.0
        base_unit = ""
        native = tx.get("nativeTransfers", []) or []
        native_in = sum(float(item.get("amount", 0) or 0) for item in native if item.get("toUserAccount") == wallet) / 1e9
        native_out = sum(float(item.get("amount", 0) or 0) for item in native if item.get("fromUserAccount") == wallet) / 1e9
        if max(native_in, native_out) > 0:
            base_value = native_out if side == "buy" else native_in
            base_unit = "SOL"
        else:
            stable = [item for item in transfers if item.get("mint") in stable_mints]
            stable_in = sum(float(item.get("tokenAmount", 0) or 0) for item in stable if item.get("toUserAccount") == wallet)
            stable_out = sum(float(item.get("tokenAmount", 0) or 0) for item in stable if item.get("fromUserAccount") == wallet)
            base_value = stable_out if side == "buy" else stable_in
            base_unit = "USD_STABLE"
        amount = float(token_transfer.get("tokenAmount", 0) or 0)
        signature = tx.get("signature")
        if not signature or amount <= 0 or base_value <= 0:
            return None
        return {
            "signature": signature,
            "token": token_transfer.get("mint"),
            "side": side,
            "amount": amount,
            "base_value": base_value,
            "base_unit": base_unit,
            "timestamp": float(tx.get("timestamp", 0) or 0),
        }

    def _extract_token_from_tx(self, tx: Dict) -> Optional[str]:
        try:
            for transfer in tx.get("tokenTransfers", []):
                if transfer.get("fromUserAccount") != transfer.get("toUserAccount"):
                    return transfer.get("mint")
        except Exception:
            pass
        return None

    # Must match the thresholds _classify_regime applies to its own PIT
    # registry: the same label assigned by two paths with different cutoffs
    # is the training-serving skew problem in miniature.
    ULTRA_EARLY_SECONDS = 10.0
    EARLY_CURVE_SECONDS = 300.0

    def _attach_launch_relative_regime(self, trade: Dict[str, Any]):
        """Classify the two regimes that a real detected launch timestamp can
        honestly support. A launch time is only known for tokens this desk
        itself observed being created (GenealogyGraph.token_launch_times);
        Helius history alone never establishes launch-relative timing, so
        every other regime (narrative, volume, migration-phase) stays
        unclassified rather than guessed.
        """
        if trade.get("regime") is not None:
            return
        launch_times = getattr(self.genealogy, "token_launch_times", None) or {}
        launch_time = launch_times.get(trade["token"])
        entry_time = trade.get("entry_timestamp")
        if launch_time is None or entry_time is None or entry_time < launch_time:
            return
        elapsed = entry_time - launch_time
        if elapsed <= self.ULTRA_EARLY_SECONDS:
            trade["regime"] = WalletRegime.ULTRA_EARLY
        elif elapsed <= self.EARLY_CURVE_SECONDS:
            trade["regime"] = WalletRegime.EARLY_CURVE

    def _classify_regime(self, tx: Dict) -> Optional[WalletRegime]:
        if self.regime_classifier:
            return self.regime_classifier(tx)
        supplied = tx.get("regime")
        if supplied:
            try:
                return supplied if isinstance(supplied, WalletRegime) else WalletRegime(str(supplied))
            except ValueError:
                return None
        token = str(tx.get("token", ""))
        entry_at = float(tx.get("entry_timestamp", tx.get("timestamp", 0)) or 0)
        migration_at = self._token_migration_times.get(token)
        if migration_at is not None and entry_at:
            return WalletRegime.PRE_MIGRATION if entry_at < migration_at else WalletRegime.POST_MIGRATION
        launch_at = self._token_launch_times.get(token)
        if launch_at is not None and entry_at >= launch_at:
            elapsed = entry_at - launch_at
            if elapsed <= 10:
                return WalletRegime.ULTRA_EARLY
            if elapsed <= 300:
                return WalletRegime.EARLY_CURVE
        # This label asserts only a verified closed round trip. It does not
        # fabricate launch-relative timing; dedicated regimes still require PIT
        # launch/migration timestamps.
        return WalletRegime.GENERAL_HISTORY if tx.get("data_status") == "OK" else None

    def record_token_lifecycle(self, token: str, *, launch_at: Optional[float] = None,
                               migration_at: Optional[float] = None):
        """Register observed PIT lifecycle timestamps used to classify wallet skill."""
        if not token:
            return
        if launch_at is not None:
            self._token_launch_times[token] = float(launch_at)
        if migration_at is not None:
            self._token_migration_times[token] = float(migration_at)

    async def _update_regime_performance(self, wallet: str, regime: WalletRegime, tx: Dict):
        if regime not in self.regime_performances[wallet]:
            self.regime_performances[wallet][regime] = WalletRegimePerformance(
                wallet=wallet, regime=regime
            )
        
        perf = self.regime_performances[wallet][regime]
        perf.trades += 1
        
        multiple = tx.get("multiple", 1)
        if multiple >= 2:
            perf.win_rate_2x = (perf.win_rate_2x * (perf.trades - 1) + 1) / perf.trades
        else:
            perf.win_rate_2x = (perf.win_rate_2x * (perf.trades - 1)) / perf.trades
        
        if multiple >= 5:
            perf.win_rate_5x = (perf.win_rate_5x * (perf.trades - 1) + 1) / perf.trades
        else:
            perf.win_rate_5x = (perf.win_rate_5x * (perf.trades - 1)) / perf.trades
        
        if multiple >= 10:
            perf.win_rate_10x = (perf.win_rate_10x * (perf.trades - 1) + 1) / perf.trades
        else:
            perf.win_rate_10x = (perf.win_rate_10x * (perf.trades - 1)) / perf.trades
        
        if tx.get("entry_pct") is not None:
            entry_pct = tx["entry_pct"]
            perf.avg_entry_pct = (perf.avg_entry_pct * (perf.trades - 1) + entry_pct) / perf.trades
        if tx.get("exit_pct") is not None:
            exit_pct = tx["exit_pct"]
            perf.avg_exit_pct = (perf.avg_exit_pct * (perf.trades - 1) + exit_pct) / perf.trades
        
        hold_time = tx.get("hold_time", 0)
        perf.avg_hold_time = (perf.avg_hold_time * (perf.trades - 1) + hold_time) / perf.trades
        
        pnl = tx.get("realized_pnl", 0)
        perf.realized_pnl = (perf.realized_pnl * (perf.trades - 1) + pnl) / perf.trades
        
        if tx.get("rugged") is not None:
            rugged = bool(tx["rugged"])
            perf.rug_exposure = (perf.rug_exposure * (perf.trades - 1) + int(rugged)) / perf.trades
        
        perf.last_updated = time.time()

    async def _discover_wallets_from_social(self):
        candidates = list(self._social_wallet_candidates)[:100]
        self._social_wallet_candidates.difference_update(candidates)
        for wallet in candidates:
            await self._evaluate_new_wallet(wallet, "social_discovery")

    def register_social_wallet(self, wallet: str):
        if wallet and 32 <= len(wallet) <= 44:
            self._social_wallet_candidates.add(wallet)

    def stale_watch_wallets(self, now: Optional[float] = None) -> List[str]:
        """Tracked wallets the stream has NOT covered recently, oldest first.

        The poll used to walk all hundred watched wallets every two seconds,
        sequentially. Even at fifty milliseconds a request that is five
        seconds of work inside a two-second loop, so the hundredth wallet was
        never on a two-second delay -- it was on whatever the queue happened
        to be, and nobody was measuring which.

        The stream already carries every trade on the programs we subscribe
        to, at stream latency. So the poll stops being the primary path and
        becomes what it should always have been: reconciliation for the
        wallets the stream cannot see, which are the ones trading somewhere we
        do not subscribe. Ordering by staleness means the budget goes to the
        wallets we know least about rather than to whichever came first out of
        a set.
        """
        now = time.time() if now is None else float(now)
        stale = [(self._stream_seen_at.get(wallet, 0.0), wallet)
                 for wallet in self._live_watch_wallets
                 if now - self._stream_seen_at.get(wallet, 0.0) > self.stream_coverage_seconds]
        stale.sort()
        return [wallet for _, wallet in stale]

    async def _watch_live_wallets(self):
        if not self.helius_key:
            self.data_status["live_wallet_watch"] = "DATA_BLOCKED: HELIUS_API_KEY missing"
            return
        if not self._live_watch_wallets:
            return

        wallets = self.stale_watch_wallets()[:self.max_reconcile_per_pass]
        if not wallets:
            self.data_status["live_wallet_watch"] = "OK: stream covers every watched wallet"
            return

        semaphore = asyncio.Semaphore(self.reconcile_concurrency)

        async def reconcile(wallet: str) -> None:
            async with semaphore:
                try:
                    async with self._session.get(
                        f"{self._helius_base}/addresses/{wallet}/transactions",
                        params={
                            "api-key": self.helius_key,
                            "limit": 20,
                            "type": "SWAP",
                            "commitment": "processed"
                        }
                    ) as resp:
                        if resp.status == 200:
                            txs = await resp.json()
                            for tx in txs:
                                await self._process_live_transaction(wallet, tx)
                except Exception as e:
                    # Also promoted from debug: a wallet on the live watch
                    # list going silently unpolled has the same downstream
                    # effect as never having been watched at all.
                    logger.warning("Live watch poll failed for %s: %s: %s",
                                   wallet, type(e).__name__, e)

        # Concurrently, because one slow wallet used to delay every wallet
        # behind it, and the delay was invisible.
        await asyncio.gather(*(reconcile(wallet) for wallet in wallets),
                             return_exceptions=True)
        self.data_status["live_wallet_watch"] = (
            f"OK: reconciled {len(wallets)} of {len(self._live_watch_wallets)} watched wallets")

    def coverage_report(self, now: Optional[float] = None) -> Dict[str, Any]:
        """How each watched wallet is actually being observed, and how late."""
        now = time.time() if now is None else float(now)
        watched = len(self._live_watch_wallets)
        streamed = sum(1 for wallet in self._live_watch_wallets
                       if now - self._stream_seen_at.get(wallet, 0.0)
                       <= self.stream_coverage_seconds)
        lags = {}
        for path, samples in self._observation_lag.items():
            lags[path] = {
                "observations": len(samples),
                "median_s": (float(np.median(samples)) if samples else None),
                "p90_s": (float(np.quantile(samples, 0.9)) if samples else None),
            }
        return {
            "status": "OK" if watched else "DATA_BLOCKED",
            "watched": watched,
            "stream_covered": streamed,
            "awaiting_reconciliation": max(0, watched - streamed),
            "stream_coverage_seconds": self.stream_coverage_seconds,
            "observation_lag": lags,
        }

    async def _process_live_transaction(self, wallet: str, tx: Dict):
        sig = tx.get("signature")
        if not sig or sig in self._seen_live_signatures:
            return
        self._seen_live_signatures.add(sig)
        if len(self._seen_live_signatures) > 100_000:
            self._seen_live_signatures = set(list(self._seen_live_signatures)[-50_000:])
        
        token = self._extract_token_from_tx(tx)
        if not token:
            return
        
        side = self._determine_side(wallet, tx)
        
        event = {
            "wallet": wallet,
            "token": token,
            "side": side,
            "amount": self._extract_amount(tx),
            "price": self._extract_price(tx),
            "timestamp": tx.get("timestamp", time.time()),
            "signature": sig,
            "slot": tx.get("slot", 0),
            "wallet_score": self.wallet_scores.get(wallet, WalletScore(wallet, 0, {}, 0, 0, 0, 0, 0, 0, 0)).overall_score
        }
        
        if side == "buy":
            self._recent_buys.append(event)
        else:
            self._recent_sells.append(event)
        self._note_observation_lag("poll", event["timestamp"])

    def record_live_trade(self, token: str, event: Dict[str, Any]):
        """Register a decoded stream trade without pretending its limit is a fill price."""
        side = event.get("side")
        if side not in {"buy", "sell"}:
            return
        record = {
            "wallet": event.get("wallet", ""), "token": token, "side": side,
            "amount": float(event.get("amount", 0) or 0), "price": float(event.get("price", 0) or 0),
            "timestamp": float(event.get("timestamp", time.time())), "signature": event.get("signature"),
            "data_status": "OK" if event.get("price") else "DATA_BLOCKED_PRICE",
        }
        (self._recent_buys if side == "buy" else self._recent_sells).append(record)
        wallet = record["wallet"]
        if wallet:
            # The stream carried this wallet, so the poll does not have to.
            self._stream_seen_at[wallet] = time.time()
            self._note_observation_lag("stream", record["timestamp"])
        if side == "buy":
            self._queue_wallet_history(wallet, token)

    def _note_observation_lag(self, path: str, observed_event_time: Any) -> None:
        """Seconds between a trade happening and us seeing it, by path.

        Recorded rather than assumed, because the whole argument for moving
        wallet monitoring onto the stream rests on this number, and an
        argument that cannot be checked is a preference.
        """
        try:
            lag = time.time() - float(observed_event_time)
        except (TypeError, ValueError):
            return
        if 0 <= lag < 3600:
            self._observation_lag[path].append(lag)

    def _determine_side(self, wallet: str, tx: Dict) -> str:
        try:
            for transfer in tx.get("tokenTransfers", []):
                if transfer.get("fromUserAccount") == wallet:
                    return "sell"
                if transfer.get("toUserAccount") == wallet:
                    return "buy"
        except Exception:
            pass
        return "unknown"

    def _extract_amount(self, tx: Dict) -> float:
        try:
            for transfer in tx.get("tokenTransfers", []):
                return float(transfer.get("tokenAmount", 0))
        except Exception:
            pass
        return 0.0

    def _extract_price(self, tx: Dict) -> float:
        try:
            native_transfers = tx.get("nativeTransfers", [])
            token_transfers = tx.get("tokenTransfers", [])
            if native_transfers and token_transfers:
                sol_amount = float(native_transfers[0].get("amount", 0)) / 1e9
                token_amount = float(token_transfers[0].get("tokenAmount", 0))
                if token_amount > 0:
                    return sol_amount / token_amount
        except Exception:
            pass
        return 0.0

    async def _recalculate_all_scores(self):
        all_wallets = set(self.genealogy.wallets.keys()) | set(self.regime_performances.keys())
        
        scores = []
        for wallet in all_wallets:
            score = self._calculate_wallet_score(wallet)
            if score:
                scores.append(score)
        
        scores.sort(key=lambda s: s.overall_score, reverse=True)
        for i, score in enumerate(scores):
            score.rank = i + 1
            self.wallet_scores[score.wallet] = score
        
        logger.info(f"Recalculated scores for {len(scores)} wallets, top 10: {[s.wallet[:8] for s in scores[:10]]}")

    def _calculate_wallet_score(self, wallet: str) -> Optional[WalletScore]:
        wp = self.genealogy.wallets.get(wallet)
        if not wp or wp.tx_count < self.min_trades:
            return None
        
        regimes = self.regime_performances.get(wallet, {})
        if not regimes:
            return None
        
        regime_scores = {}
        for regime, perf in regimes.items():
            if perf.trades < 5:
                continue
            
            early_quality = max(0, 1 - perf.avg_entry_pct / 100)
            forward_quality = (perf.win_rate_2x * 0.3 + perf.win_rate_5x * 0.5 + perf.win_rate_10x * 0.2)
            consistency = 1 - abs(perf.win_rate_5x - perf.win_rate_2x * 0.5)
            
            independence = perf.independence_score
            
            sample_factor = min(1.0, perf.trades / 100)
            
            regime_score = (
                early_quality * 0.25 +
                forward_quality * 0.30 +
                consistency * 0.20 +
                independence * 0.15 +
                sample_factor * 0.10
            ) * (1 - perf.rug_exposure * 0.5)
            
            regime_scores[regime] = max(0, regime_score)
        
        if not regime_scores:
            return None
        
        overall = np.mean(list(regime_scores.values()))
        
        early_entry = np.mean([max(0, 1 - p.avg_entry_pct / 100) for p in regimes.values() if p.trades >= 5])
        forward_return = np.mean([p.win_rate_5x for p in regimes.values() if p.trades >= 5])
        consistency = np.mean([1 - abs(p.win_rate_5x - p.win_rate_2x * 0.5) for p in regimes.values() if p.trades >= 5])
        independence = np.mean([p.independence_score for p in regimes.values() if p.trades >= 5])
        sample_size = sum(p.trades for p in regimes.values())
        rug_exposure = np.mean([p.rug_exposure for p in regimes.values() if p.trades >= 5])
        
        copy_crowding = self._estimate_copy_crowding(wallet)
        
        return WalletScore(
            wallet=wallet,
            overall_score=overall * (1 - copy_crowding * 0.3),
            regime_scores=regime_scores,
            early_entry_quality=early_entry,
            forward_return_quality=forward_return,
            consistency=consistency,
            independence=independence,
            sample_size=sample_size,
            rug_exposure=rug_exposure,
            copy_crowding=copy_crowding
        )

    def _estimate_copy_crowding(self, wallet: str) -> float:
        wp = self.genealogy.wallets.get(wallet)
        if not wp:
            return 0.0
        
        follower_count = len(wp.related_wallets)
        return min(1.0, follower_count / 100)

    async def _update_live_watch_list(self):
        # Measured value first. A wallet with a positive lower bound has been
        # shown to grow capital when followed at fills we could actually get;
        # the composite score has been shown to be above 0.5, which is a fact
        # about the formula rather than about the wallet.
        measured = [value.wallet for value in
                    self.wallet_value.rank(limit=200, followable_only=True)]
        remaining = 200 - len(measured)
        fallback = [
            s.wallet for s in sorted(self.wallet_scores.values(), key=lambda x: x.overall_score, reverse=True)
            if s.overall_score > 0.5 and s.sample_size >= self.min_trades
            and s.wallet not in set(measured)
        ][:max(0, remaining)]
        # Watching a wallet costs a subscription, not capital, so the unproven
        # ones still fill the list -- that is how they accumulate the outcomes
        # that would prove them. What they do NOT get is to outrank a wallet
        # whose value has been measured.
        top_wallets = measured + fallback
        
        self._live_watch_wallets = set(top_wallets)
        
        for wallet in top_wallets[:50]:
            if wallet not in self.genealogy.wallets:
                self.genealogy.wallets[wallet] = WalletProfile(
                    address=wallet,
                    entity_type=EntityType.WALLET,
                    first_seen=time.time(),
                    last_seen=time.time()
                )
            self.genealogy.wallets[wallet].is_smart_money = True

    def get_wallet_score(self, wallet: str) -> Optional[WalletScore]:
        return self.wallet_scores.get(wallet)

    def is_watched(self, wallet: str) -> bool:
        """Whether this wallet is on the live watch list.

        The gate on opening a follow: measuring what following EVERY wallet
        would have returned is measuring the market, and the model would then
        rank the market rather than the wallets we chose to watch.
        """
        return bool(wallet) and wallet in self._live_watch_wallets

    def get_wallet_value(self, wallet: str, regime: Optional[WalletRegime] = None) -> WalletValue:
        """Forward E[log W] of following this wallet, or why there is none.

        DATA_BLOCKED below the sample floor, never a default. A wallet we have
        not followed enough times has no value estimate, and reporting a low
        one instead invites treating "unmeasured" as "measured and poor".
        """
        return self.wallet_value.value(wallet, regime.value if regime else "")

    def record_follow_outcome(self, outcome: FollowOutcome) -> bool:
        """One measured result of following a wallet."""
        return self.wallet_value.record(outcome)

    def get_top_wallets_by_value(self, limit: int = 20,
                                 regime: Optional[WalletRegime] = None) -> List[WalletValue]:
        """Ranked by lower confidence bound, so six lucky trades cannot lead."""
        return self.wallet_value.rank(limit=limit, regime=regime.value if regime else "")

    def get_regime_performance(self, wallet: str, regime: WalletRegime) -> Optional[WalletRegimePerformance]:
        return self.regime_performances.get(wallet, {}).get(regime)

    def get_top_wallets(self, regime: Optional[WalletRegime] = None, limit: int = 20) -> List[WalletScore]:
        scores = list(self.wallet_scores.values())
        
        if regime:
            scores = [s for s in scores if regime in s.regime_scores and s.regime_scores[regime] > 0.5]
        
        scores.sort(key=lambda s: s.overall_score, reverse=True)
        return scores[:limit]

    def get_wallet_signal(self, wallet: str, token: str, regime: WalletRegime) -> Dict[str, Any]:
        score = self.wallet_scores.get(wallet)
        if not score:
            return {"signal": 0, "confidence": 0, "reason": "no_score"}
        
        regime_perf = self.regime_performances.get(wallet, {}).get(regime)
        if not regime_perf or regime_perf.trades < 5:
            return {"signal": 0, "confidence": 0, "reason": "insufficient_regime_data"}
        
        regime_score = score.regime_scores.get(regime, 0)
        
        signal = regime_score * (1 - regime_perf.rug_exposure)
        confidence = min(1.0, regime_perf.trades / 50) * score.consistency
        
        return {
            "signal": signal,
            "confidence": confidence,
            "regime_score": regime_score,
            "win_rate_5x": regime_perf.win_rate_5x,
            "avg_entry_pct": regime_perf.avg_entry_pct,
            "avg_exit_pct": regime_perf.avg_exit_pct,
            "rugged_rate": regime_perf.rug_exposure,
            "sample_size": regime_perf.trades
        }

    def get_coordinated_activity(self, window_seconds: int = 300) -> List[Dict]:
        now = time.time()
        cutoff = now - window_seconds
        
        buys_by_token = defaultdict(list)
        for buy in self._recent_buys:
            if buy["timestamp"] > cutoff:
                buys_by_token[buy["token"]].append(buy)
        
        coordinated = []
        for token, buys in buys_by_token.items():
            if len(buys) >= 3:
                wallet_scores = [b["wallet_score"] for b in buys if b["wallet_score"] > 0]
                if wallet_scores:
                    avg_score = np.mean(wallet_scores)
                    if avg_score > 0.6:
                        coordinated.append({
                            "token": token,
                            "wallets": [b["wallet"] for b in buys],
                            "avg_wallet_score": avg_score,
                            "buy_count": len(buys),
                            "time_span": max(b["timestamp"] for b in buys) - min(b["timestamp"] for b in buys),
                            "total_sol": sum(b["amount"] * b["price"] for b in buys)
                        })
        
        return sorted(coordinated, key=lambda x: x["avg_wallet_score"], reverse=True)

    def get_stats(self) -> Dict:
        return {
            "tracked_wallets": len(self.genealogy.wallets),
            "scored_wallets": len(self.wallet_scores),
            "live_watching": len(self._live_watch_wallets),
            "history_queue": len(self._history_candidates),
            "histories_evaluated": len(self._history_evaluated_at),
            "regimes_tracked": len(WalletRegime),
            "recent_buys": len(self._recent_buys),
            "recent_sells": len(self._recent_sells),
            "top_10": [{"wallet": s.wallet[:8], "score": round(s.overall_score, 3)} for s in self.get_top_wallets(limit=10)],
            # The measured ranking, alongside the composite one. Where they
            # disagree, the measured one is the one with evidence behind it.
            "wallet_value": self.wallet_value.report(),
            "data_blocked_wallets": sum(1 for status in self.data_status.values() if status.startswith("DATA_BLOCKED")),
            "data_status": dict(self.data_status),
        }

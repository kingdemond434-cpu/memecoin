import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import json
import numpy as np

import aiohttp

from src.chains.rpc_manager import ChainConfig, RPCManager
from src.strategies.genealogy_graph import GenealogyGraph, WalletProfile, EntityType

logger = logging.getLogger(__name__)


class WalletRegime(Enum):
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
        
        self.regime_performances: Dict[str, Dict[WalletRegime, WalletRegimePerformance]] = defaultdict(dict)
        self.wallet_scores: Dict[str, WalletScore] = {}
        self.regime_classifier: Optional[Callable] = None
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._hunter_task: Optional[asyncio.Task] = None
        self._watcher_task: Optional[asyncio.Task] = None
        self._recalc_task: Optional[asyncio.Task] = None
        
        self._live_watch_wallets: Set[str] = set()
        self._recent_buys: deque = deque(maxlen=10000)
        self._recent_sells: deque = deque(maxlen=10000)
        self._seen_live_signatures: Set[str] = set()
        self._history_signatures: Dict[str, Set[str]] = defaultdict(set)
        self._social_wallet_candidates: Set[str] = set()
        self.data_status: Dict[str, str] = {}
        
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
        
        await self._initial_discovery()

    async def stop(self):
        self._running = False
        for task in [self._hunter_task, self._watcher_task, self._recalc_task]:
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
            await asyncio.sleep(2)

    async def _recalc_loop(self):
        while self._running:
            try:
                await self._recalculate_all_scores()
                await self._update_live_watch_list()
            except Exception as e:
                logger.error(f"Recalc loop error: {e}")
            await asyncio.sleep(300)

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
            logger.debug("Token holder analysis error: %s", e)

    async def analyze_token_early_buyers(self, token: str):
        """Public entrypoint used by the canonical launch stream."""
        await self._analyze_token_early_buyers(token)

    async def _evaluate_new_wallet(self, wallet: str, trigger_token: str):
        if not self.helius_key:
            self.data_status[wallet] = "DATA_BLOCKED: HELIUS_API_KEY missing"
            return
        try:
            async with self._session.get(
                f"{self._helius_base}/addresses/{wallet}/transactions",
                params={"api-key": self.helius_key, "limit": 100, "type": "SWAP"}
            ) as resp:
                if resp.status == 200:
                    txs = await resp.json()
                    await self._build_wallet_history(wallet, txs)
        except Exception as e:
            logger.debug(f"Wallet eval error: {e}")

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
        for trade in reconstructed["closed_trades"]:
            token = trade["token"]
            if token not in wp.launches_participated:
                wp.launches_participated.append(token)
            await self._update_regime_performance(wallet, self._classify_regime(trade), trade)
        self.data_status[wallet] = "OK" if reconstructed["closed_trades"] else "DATA_BLOCKED"

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
                "timestamp": normalized["timestamp"],
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

    def _classify_regime(self, tx: Dict) -> WalletRegime:
        slot = tx.get("slot", 0)
        timestamp = tx.get("timestamp", time.time())
        
        if self.regime_classifier:
            return self.regime_classifier(tx)
        
        return WalletRegime.EARLY_CURVE

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

    async def _watch_live_wallets(self):
        if not self.helius_key:
            self.data_status["live_wallet_watch"] = "DATA_BLOCKED: HELIUS_API_KEY missing"
            return
        if not self._live_watch_wallets:
            return
        
        wallets = list(self._live_watch_wallets)[:100]
        
        for wallet in wallets:
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
                logger.debug(f"Live watch error for {wallet}: {e}")

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
        top_wallets = [
            s.wallet for s in sorted(self.wallet_scores.values(), key=lambda x: x.overall_score, reverse=True)
            if s.overall_score > 0.5 and s.sample_size >= self.min_trades
        ][:200]
        
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
            "regimes_tracked": len(WalletRegime),
            "recent_buys": len(self._recent_buys),
            "recent_sells": len(self._recent_sells),
            "top_10": [{"wallet": s.wallet[:8], "score": round(s.overall_score, 3)} for s in self.get_top_wallets(limit=10)],
            "data_blocked_wallets": sum(1 for status in self.data_status.values() if status.startswith("DATA_BLOCKED")),
            "data_status": dict(self.data_status),
        }

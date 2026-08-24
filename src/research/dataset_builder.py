import asyncio
import logging
import time
import json
import gzip
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from src.chains.rpc_manager import ChainConfig, RPCManager
from src.strategies.genealogy_graph import GenealogyGraph
from src.strategies.wallet_intelligence import WalletIntelligenceEngine
from src.strategies.social_intelligence import SocialIntelligenceEngine
from src.strategies.prelaunch_intent import PrelaunchIntentModel
from src.strategies.information_graph import InformationLeadGraph, CounterfactualExecutionLab
from src.strategies.rug_hazard import ContinuousRugHazardModel
from src.strategies.champion_challenger import ChampionChallengerFramework

logger = logging.getLogger(__name__)


class SnapshotTimepoint(Enum):
    PRELAUNCH = "prelaunch"
    T0 = "t0"
    T1S = "t1s"
    T3S = "t3s"
    T5S = "t5s"
    T10S = "t10s"
    T30S = "t30s"
    T1M = "t1m"
    T3M = "t3m"
    T5M = "t5m"
    T15M = "t15m"
    T1H = "t1h"


@dataclass
class LaunchSnapshot:
    token: str
    chain: str
    snapshot_time: SnapshotTimepoint
    timestamp: float
    
    deployer_features: Dict[str, Any] = field(default_factory=dict)
    wallet_features: Dict[str, Any] = field(default_factory=dict)
    flow_features: Dict[str, Any] = field(default_factory=dict)
    liquidity_features: Dict[str, Any] = field(default_factory=dict)
    social_features: Dict[str, Any] = field(default_factory=dict)
    token_features: Dict[str, Any] = field(default_factory=dict)
    market_features: Dict[str, Any] = field(default_factory=dict)
    entity_graph_features: Dict[str, Any] = field(default_factory=dict)
    
    label_2x: Optional[bool] = None
    label_5x: Optional[bool] = None
    label_10x: Optional[bool] = None
    label_50x: Optional[bool] = None
    label_migration: Optional[bool] = None
    label_rug: Optional[bool] = None
    label_rug_time: Optional[float] = None
    max_multiple: Optional[float] = None
    time_to_peak: Optional[float] = None
    max_drawdown: Optional[float] = None
    feasible_exit_multiple: Optional[float] = None
    realized_pnl: Optional[float] = None


@dataclass
class LaunchEpisode:
    token: str
    chain: str
    created_at: float
    deployer: str
    factory: str
    pair: str
    base_token: str
    
    snapshots: Dict[SnapshotTimepoint, LaunchSnapshot] = field(default_factory=dict)
    final_outcome: Dict[str, Any] = field(default_factory=dict)
    execution_attempts: List[Dict] = field(default_factory=list)
    counterfactuals: List[Dict] = field(default_factory=list)
    market_observations: List[Dict] = field(default_factory=list)
    risk_report: Dict[str, Any] = field(default_factory=dict)
    prelaunch_status: str = "DATA_BLOCKED"


class PointInTimeDatasetBuilder:
    def __init__(
        self,
        chain_config: ChainConfig,
        rpc: RPCManager,
        genealogy: GenealogyGraph,
        wallet_intel: WalletIntelligenceEngine,
        social_intel: SocialIntelligenceEngine,
        prelaunch: PrelaunchIntentModel,
        info_graph: InformationLeadGraph,
        rug_hazard: ContinuousRugHazardModel,
        champion_challenger: ChampionChallengerFramework,
        storage_path: str = "data/launch_episodes"
    ):
        self.chain_config = chain_config
        self.rpc = rpc
        self.genealogy = genealogy
        self.wallet_intel = wallet_intel
        self.social_intel = social_intel
        self.prelaunch = prelaunch
        self.info_graph = info_graph
        self.rug_hazard = rug_hazard
        self.champion_challenger = champion_challenger
        self.storage_path = storage_path
        
        self.active_episodes: Dict[str, LaunchEpisode] = {}
        self.completed_episodes: Dict[str, LaunchEpisode] = {}
        self.outcome_index: Dict[str, Dict[str, Any]] = {}
        
        self.snapshot_times = {
            SnapshotTimepoint.PRELAUNCH: -1,
            SnapshotTimepoint.T0: 0,
            SnapshotTimepoint.T1S: 1,
            SnapshotTimepoint.T3S: 3,
            SnapshotTimepoint.T5S: 5,
            SnapshotTimepoint.T10S: 10,
            SnapshotTimepoint.T30S: 30,
            SnapshotTimepoint.T1M: 60,
            SnapshotTimepoint.T3M: 180,
            SnapshotTimepoint.T5M: 300,
            SnapshotTimepoint.T15M: 900,
            SnapshotTimepoint.T1H: 3600,
        }
        
        self._running = False
        self._snapshot_task: Optional[asyncio.Task] = None
        self._flush_task: Optional[asyncio.Task] = None
        self._snapshot_inflight: Set[Tuple[str, SnapshotTimepoint]] = set()
        self._capture_tasks: Set[asyncio.Task] = set()

    async def start(self):
        import os
        os.makedirs(self.storage_path, exist_ok=True)
        self._load_outcome_index()
        self._load_active_checkpoints()
        self._running = True
        self._snapshot_task = asyncio.create_task(self._snapshot_loop())
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def stop(self):
        self._running = False
        for task in [self._snapshot_task, self._flush_task]:
            if task:
                task.cancel()
        for task in list(self._capture_tasks):
            task.cancel()
        if self._capture_tasks:
            await asyncio.gather(*self._capture_tasks, return_exceptions=True)
        await self._flush_active()
        await self._flush_all()

    def start_episode(
        self,
        token: str,
        deployer: str,
        factory: str,
        pair: str,
        base_token: str,
        *,
        detected_at: Optional[float] = None,
        prelaunch_context: Optional[Dict[str, Any]] = None,
    ):
        if token in self.active_episodes:
            return
        
        episode = LaunchEpisode(
            token=token,
            chain=self.chain_config.name,
            created_at=detected_at or time.time(),
            deployer=deployer,
            factory=factory,
            pair=pair,
            base_token=base_token
        )
        self.active_episodes[token] = episode
        context = prelaunch_context or {}
        as_of = float(context.get("as_of", 0) or 0)
        valid_prelaunch = bool(context and as_of and as_of <= episode.created_at)
        episode.prelaunch_status = "OK" if valid_prelaunch else "DATA_BLOCKED"
        if not valid_prelaunch:
            context = {}
        episode.snapshots[SnapshotTimepoint.PRELAUNCH] = LaunchSnapshot(
            token=token,
            chain=self.chain_config.name,
            snapshot_time=SnapshotTimepoint.PRELAUNCH,
            timestamp=as_of if valid_prelaunch else episode.created_at,
            deployer_features=context.get("deployer_features", {"status": "DATA_BLOCKED"}),
            wallet_features=context.get("wallet_features", {"status": "DATA_BLOCKED"}),
            social_features=context.get("social_features", {"status": "DATA_BLOCKED"}),
            entity_graph_features=context.get("entity_graph_features", {"status": "DATA_BLOCKED"}),
            market_features={"status": episode.prelaunch_status},
        )

    async def _snapshot_loop(self):
        while self._running:
            try:
                now = time.time()
                for token, episode in list(self.active_episodes.items()):
                    elapsed = now - episode.created_at
                    
                    for sp, target_time in self.snapshot_times.items():
                        if sp == SnapshotTimepoint.PRELAUNCH:
                            continue
                        
                        key = (token, sp)
                        if sp not in episode.snapshots and key not in self._snapshot_inflight and elapsed >= target_time:
                            self._snapshot_inflight.add(key)
                            task = asyncio.create_task(self._capture_snapshot(token, sp))
                            self._capture_tasks.add(task)
                            task.add_done_callback(self._capture_tasks.discard)
                    
                    if elapsed >= self.snapshot_times[SnapshotTimepoint.T1H]:
                        final_key = (token, SnapshotTimepoint.T1H)
                        if SnapshotTimepoint.T1H not in episode.snapshots:
                            if final_key not in self._snapshot_inflight:
                                self._snapshot_inflight.add(final_key)
                                await self._capture_snapshot(token, SnapshotTimepoint.T1H)
                            continue
                        await self._finalize_episode(token)
            except Exception as e:
                logger.error(f"Snapshot loop error: {e}")
            await asyncio.sleep(1)

    async def _capture_snapshot(self, token: str, snapshot_time: SnapshotTimepoint):
        if token not in self.active_episodes:
            return
        
        episode = self.active_episodes[token]
        if snapshot_time in episode.snapshots:
            return
        
        key = (token, snapshot_time)
        try:
            as_of = time.time()
            snapshot = LaunchSnapshot(
                token=token,
                chain=self.chain_config.name,
                snapshot_time=snapshot_time,
                timestamp=as_of,
            )
            
            snapshot.deployer_features = await self._capture_deployer_features(episode, as_of)
            snapshot.wallet_features = await self._capture_wallet_features(episode, as_of)
            snapshot.flow_features = await self._capture_flow_features(episode, as_of)
            snapshot.liquidity_features = await self._capture_liquidity_features(episode, as_of)
            snapshot.social_features = await self._capture_social_features(episode, as_of)
            snapshot.token_features = await self._capture_token_features(episode, as_of)
            snapshot.market_features = await self._capture_market_features(episode, as_of)
            snapshot.entity_graph_features = await self._capture_entity_graph_features(episode, as_of)
            
            episode.snapshots[snapshot_time] = snapshot
            
        except Exception as e:
            logger.error(f"Snapshot capture failed for {token} {snapshot_time}: {e}")
        finally:
            self._snapshot_inflight.discard(key)

    async def _capture_deployer_features(self, episode: LaunchEpisode, as_of: float) -> Dict[str, Any]:
        dp = self.genealogy.get_deployer_profile(episode.deployer)
        if not dp:
            return {"has_profile": False}
        
        return {
            "has_profile": True,
            "prior_launches": len(dp.tokens_created),
            "rug_rate": dp.rug_rate,
            "success_rate": dp.success_rate,
            "avg_max_multiple": dp.avg_max_multiple,
            "funding_wallet_count": len(dp.funding_wallets),
            "operational_wallet_count": len(dp.operational_wallets),
            "ownership_renounced_rate": dp.ownership_renounced_rate,
            "liquidity_locked_rate": dp.liquidity_locked_rate
        }

    async def _capture_wallet_features(self, episode: LaunchEpisode, as_of: float) -> Dict[str, Any]:
        smart_wallets = self.wallet_intel.get_top_wallets(limit=50)
        
        initial_buyers = []
        smart_buyers = 0
        insider_buyers = 0
        total_sol_volume = 0
        
        for buy in self.wallet_intel._recent_buys:
            observed_at = float(buy.get("timestamp", 0) or 0)
            if buy["token"] == episode.token and episode.created_at <= observed_at <= as_of:
                initial_buyers.append(buy)
                total_sol_volume += buy["amount"] * buy["price"]
                
                ws = self.wallet_intel.get_wallet_score(buy["wallet"])
                if ws and ws.overall_score > 0.7:
                    smart_buyers += 1
                if ws and ws.is_insider:
                    insider_buyers += 1
        
        return {
            "initial_buyer_count": len(initial_buyers),
            "smart_buyer_count": smart_buyers,
            "insider_buyer_count": insider_buyers,
            "total_sol_volume": total_sol_volume,
            "buyer_diversity": len(set(b["wallet"] for b in initial_buyers)) / max(len(initial_buyers), 1)
        }

    async def _capture_flow_features(self, episode: LaunchEpisode, as_of: float) -> Dict[str, Any]:
        observations = [
            item for item in episode.market_observations
            if item.get("type") == "trade" and float(item.get("timestamp", 0) or 0) <= as_of
        ]
        if not observations:
            return {"status": "DATA_BLOCKED", "reason": "no_trade_observations"}
        observations.sort(key=lambda item: item.get("timestamp", 0))
        now = as_of
        buys_10s = [item for item in observations if item.get("side") == "buy" and now - item.get("timestamp", 0) <= 10]
        buys_prev = [item for item in observations if item.get("side") == "buy" and 10 < now - item.get("timestamp", 0) <= 20]
        wallets = [item.get("wallet") for item in buys_10s if item.get("wallet")]
        sell_10s = [item for item in observations if item.get("side") == "sell" and now - item.get("timestamp", 0) <= 10]
        buy_notionals = [float(item.get("notional_sol", 0) or 0) for item in buys_10s
                         if item.get("notional_sol") is not None]
        sell_notionals = [float(item.get("notional_sol", 0) or 0) for item in sell_10s
                          if item.get("notional_sol") is not None]
        total_notional = sum(buy_notionals) + sum(sell_notionals)
        positive_sizes = np.array([value for value in buy_notionals if value > 0], dtype=float)
        if positive_sizes.size:
            weights = positive_sizes / positive_sizes.sum()
            order_flow_entropy = float(-np.sum(weights * np.log(np.clip(weights, 1e-12, 1))))
            order_flow_entropy /= max(float(np.log(len(weights))), 1.0)
        else:
            order_flow_entropy = None
        slots = [item.get("slot") for item in buys_10s if item.get("slot")]
        largest_slot = max((slots.count(slot) for slot in set(slots)), default=0)
        return {
            "status": "OK",
            "buy_velocity": len(buys_10s) / 10,
            "buy_acceleration": (len(buys_10s) - len(buys_prev)) / 10,
            "organic_ratio": len(set(wallets)) / max(len(wallets), 1),
            "bundle_concentration": largest_slot / max(len(buys_10s), 1),
            "unique_buyers_10s": len(set(wallets)),
            "repeated_buy_ratio": 1 - len(set(wallets)) / max(len(wallets), 1),
            "buy_notional_sol_10s": sum(buy_notionals) if buy_notionals else None,
            "sell_notional_sol_10s": sum(sell_notionals) if sell_notionals else None,
            "buy_sell_notional_imbalance": ((sum(buy_notionals) - sum(sell_notionals)) / total_notional
                                             if total_notional else None),
            "order_flow_entropy": order_flow_entropy,
            "economics_status": "OK" if total_notional else "DATA_BLOCKED",
            "observed_trade_count": len(observations),
        }

    async def _capture_liquidity_features(self, episode: LaunchEpisode, as_of: float) -> Dict[str, Any]:
        observed = [
            item for item in episode.market_observations
            if item.get("liquidity_usd") is not None and float(item.get("timestamp", 0) or 0) <= as_of
        ]
        if not observed:
            return {"status": "DATA_BLOCKED", "reason": "liquidity_not_observed"}
        latest = max(observed, key=lambda item: item.get("timestamp", 0))
        return {
            "status": "OK",
            "liquidity_usd": latest.get("liquidity_usd"),
            "liquidity_locked": latest.get("liquidity_locked"),
            "lp_burned_pct": latest.get("lp_burned_pct"),
            "route_feasible": latest.get("route_feasible"),
            "price_impact_pct": latest.get("price_impact_pct"),
        }

    async def _capture_social_features(self, episode: LaunchEpisode, as_of: float) -> Dict[str, Any]:
        social_signal = self.social_intel.get_token_social_signal(episode.token, as_of=as_of)
        return social_signal

    async def _capture_token_features(self, episode: LaunchEpisode, as_of: float) -> Dict[str, Any]:
        if not episode.risk_report:
            return {"status": "DATA_BLOCKED", "reason": "risk_report_not_recorded"}
        return {
            "status": episode.risk_report.get("data_status", "OK"),
            "ownership_renounced": episode.risk_report.get("ownership_renounced"),
            "can_mint": episode.risk_report.get("can_mint"),
            "can_freeze": episode.risk_report.get("can_freeze"),
            "top_10_pct": episode.risk_report.get("top_10_pct"),
            "token_extensions": episode.risk_report.get("token_extensions", []),
            "sell_route_feasible": episode.risk_report.get("sell_route_feasible"),
        }

    async def _capture_market_features(self, episode: LaunchEpisode, as_of: float) -> Dict[str, Any]:
        observed = [
            item for item in episode.market_observations
            if item.get("sol_price_usd") and float(item.get("timestamp", 0) or 0) <= as_of
        ]
        recent_launches = [ep for ep in self.active_episodes.values() if 0 <= as_of - ep.created_at <= 3600]
        if not observed:
            return {"status": "DATA_BLOCKED", "meme_launch_rate_1h": len(recent_launches)}
        latest = max(observed, key=lambda item: item.get("timestamp", 0))
        eligible_launches = [ep for ep in self.active_episodes.values() if ep.created_at <= as_of]
        return {
            "status": "OK",
            "sol_price_usd": latest.get("sol_price_usd"),
            "meme_launch_rate_1h": len(recent_launches),
            "graduation_rate_observed": sum(
                any(item.get("migrated") and float(item.get("timestamp", 0) or 0) <= as_of
                    for item in ep.market_observations)
                for ep in eligible_launches
            ) / max(len(eligible_launches), 1),
        }

    async def _capture_entity_graph_features(self, episode: LaunchEpisode, as_of: float) -> Dict[str, Any]:
        dp = self.genealogy.get_deployer_profile(episode.deployer)
        cluster_risk = 0
        
        if dp and dp.wallet_profile and dp.wallet_profile.cluster_id:
            cluster = self.genealogy.find_cluster(dp.wallet_profile.cluster_id)
            if cluster:
                cluster_risk = 1 if cluster.risk_level == "critical" else 0.5 if cluster.risk_level == "high" else 0
        
        funding_risks = []
        if dp:
            for wallet in dp.funding_wallets:
                profile = self.genealogy.get_wallet_profile(wallet)
                if profile:
                    funding_risks.append(1 - profile.trust_score)
        return {
            "deployer_cluster_risk": cluster_risk,
            "funding_wallet_risk": max(funding_risks) if funding_risks else None,
            "creator_genealogy_depth": len(dp.funding_wallets) if dp else None,
            "status": "OK" if dp else "DATA_BLOCKED",
        }

    async def _finalize_episode(self, token: str):
        if token not in self.active_episodes:
            return
        
        episode = self.active_episodes.pop(token)
        
        episode.final_outcome = await self._determine_final_outcome(episode)
        self.outcome_index[token] = dict(episode.final_outcome)
        self._persist_outcome_index()
        
        for snapshot in episode.snapshots.values():
            labels = episode.final_outcome
            if labels.get("status") == "DATA_BLOCKED":
                continue
            snapshot.label_2x = labels.get("max_multiple", 0) >= 2
            snapshot.label_5x = labels.get("max_multiple", 0) >= 5
            snapshot.label_10x = labels.get("max_multiple", 0) >= 10
            snapshot.label_50x = labels.get("max_multiple", 0) >= 50
            snapshot.label_migration = labels.get("migrated", False)
            snapshot.label_rug = labels.get("rugged", False)
            snapshot.label_rug_time = labels.get("rug_time")
            snapshot.max_multiple = labels.get("max_multiple")
            snapshot.time_to_peak = labels.get("time_to_peak")
            snapshot.max_drawdown = labels.get("max_drawdown")
            snapshot.feasible_exit_multiple = labels.get("feasible_exit_multiple")
            snapshot.realized_pnl = labels.get("realized_pnl")
        
        self.completed_episodes[token] = episode
        self._active_checkpoint_path(token).unlink(missing_ok=True)
        if episode.final_outcome.get("status") == "OK":
            await self.genealogy.update_token_outcome(token, {**episode.final_outcome, "deployer": episode.deployer})
            if hasattr(self.social_intel, "record_token_outcome"):
                self.social_intel.record_token_outcome(token, episode.final_outcome)
            for attempt in episode.execution_attempts:
                hypothesis_id = attempt.get("hypothesis_id")
                if hypothesis_id:
                    self.champion_challenger.record_forward_result(
                        hypothesis_id,
                        {"pnl": attempt.get("realized_pnl_usd", 0), "elogw": attempt.get("elogw", 0)},
                    )

    async def _determine_final_outcome(self, episode: LaunchEpisode) -> Dict[str, Any]:
        prices = sorted(
            [item for item in episode.market_observations
             if float(item.get("price_usd", item.get("price_multiple", 0)) or 0) > 0],
            key=lambda item: item.get("timestamp", 0),
        )
        if not prices:
            return {"status": "DATA_BLOCKED", "reason": "no_point_in_time_price_observations"}
        if all(float(item.get("price_multiple", 0) or 0) > 0 for item in prices):
            multiples = [float(item["price_multiple"]) for item in prices]
        else:
            entry = float(prices[0]["price_usd"])
            multiples = [float(item["price_usd"]) / entry for item in prices]
        peak_index = int(np.argmax(multiples))
        running_peak = multiples[0]
        max_drawdown = 0.0
        inferred_rug = None
        for multiple, item in zip(multiples, prices):
            running_peak = max(running_peak, multiple)
            drawdown = 1 - multiple / max(running_peak, 1e-12)
            max_drawdown = max(max_drawdown, drawdown)
            if inferred_rug is None and drawdown >= 0.90:
                inferred_rug = item
        feasible = [
            multiple for multiple, item in zip(multiples, prices)
            if item.get("route_feasible", item.get("feasible")) is True
            and float(item.get("price_impact_pct", 1) or 1) <= 0.15
        ]
        explicit_rugs = sorted(
            (item for item in episode.market_observations if item.get("rugged")),
            key=lambda item: float(item.get("timestamp", float("inf")) or float("inf")),
        )
        explicit_rug = explicit_rugs[0] if explicit_rugs else None
        rug_observation = explicit_rug or inferred_rug
        migration_types = {"migration", "token_migrated", "graduation"}
        migrated = any(
            item.get("migrated") is True or str(item.get("type", "")).lower() in migration_types
            for item in episode.market_observations
        )
        realized = sum(float(item.get("realized_pnl_usd", 0) or 0) for item in episode.execution_attempts)
        return {
            "status": "OK",
            "max_multiple": max(multiples),
            "migrated": migrated,
            "rugged": rug_observation is not None,
            "rug_time": max(0.0, float(rug_observation.get("timestamp", episode.created_at)) - episode.created_at)
                        if rug_observation else None,
            "time_to_peak": prices[peak_index].get("timestamp", episode.created_at) - episode.created_at,
            "max_drawdown": max_drawdown,
            "feasible_exit_multiple": max(feasible) if feasible else None,
            "realized_pnl": realized,
            "observations": len(prices),
        }

    def record_market_observation(self, token: str, observation: Dict[str, Any]):
        episode = self.active_episodes.get(token) or self.completed_episodes.get(token)
        if not episode:
            return
        record = {**observation, "timestamp": observation.get("timestamp", time.time())}
        episode.market_observations.append(record)

    def record_risk_report(self, token: str, report: Dict[str, Any]):
        episode = self.active_episodes.get(token)
        if episode:
            episode.risk_report = report

    def get_outcome(self, token: str) -> Dict[str, Any]:
        if token in self.outcome_index:
            return self.outcome_index[token]
        episode = self.completed_episodes.get(token)
        if episode:
            return episode.final_outcome
        active = self.active_episodes.get(token)
        return {"status": "PENDING"} if active else {"status": "DATA_BLOCKED", "reason": "unknown_token"}

    def _load_outcome_index(self):
        path = Path(self.storage_path) / "outcome_index.json"
        if not path.exists():
            return
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self.outcome_index = {str(token): dict(outcome) for token, outcome in loaded.items()
                                      if isinstance(outcome, dict)}
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Outcome index is unreadable; preserving the file and starting empty: %s", exc)

    def _persist_outcome_index(self):
        path = Path(self.storage_path) / "outcome_index.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.outcome_index, separators=(",", ":")), encoding="utf-8")
        temporary.replace(path)

    def _active_checkpoint_path(self, token: str) -> Path:
        return Path(self.storage_path) / "active" / f"{token}.json.gz"

    def _load_active_checkpoints(self):
        active_dir = Path(self.storage_path) / "active"
        if not active_dir.exists():
            return
        for path in active_dir.glob("*.json.gz"):
            try:
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    episode = self._episode_from_dict(json.load(handle))
                if episode.token and episode.token not in self.outcome_index:
                    self.active_episodes[episode.token] = episode
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                logger.error("Active PIT checkpoint is unreadable; preserving %s: %s", path, exc)

    def record_execution_attempt(self, token: str, attempt: Dict):
        if token in self.active_episodes:
            self.active_episodes[token].execution_attempts.append(attempt)
        elif token in self.completed_episodes:
            self.completed_episodes[token].execution_attempts.append(attempt)

    def record_counterfactual(self, token: str, counterfactual: Dict):
        if token in self.active_episodes:
            self.active_episodes[token].counterfactuals.append(counterfactual)
        elif token in self.completed_episodes:
            self.completed_episodes[token].counterfactuals.append(counterfactual)

    async def _flush_loop(self):
        while self._running:
            try:
                await self._flush_active()
                await self._flush_completed()
            except Exception as e:
                logger.error(f"Flush error: {e}")
            await asyncio.sleep(60)

    async def _flush_active(self):
        for episode in list(self.active_episodes.values()):
            path = self._active_checkpoint_path(episode.token)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            with gzip.open(temporary, "wt", encoding="utf-8") as handle:
                json.dump(self._episode_to_dict(episode), handle, separators=(",", ":"))
            temporary.replace(path)

    async def _flush_completed(self):
        if not self.completed_episodes:
            return
        
        to_flush = list(self.completed_episodes.items())[:100]
        for token, episode in to_flush:
            await self._write_episode(episode)
            del self.completed_episodes[token]

    async def _flush_all(self):
        for token, episode in list(self.completed_episodes.items()):
            await self._write_episode(episode)
        self.completed_episodes.clear()

    async def _write_episode(self, episode: LaunchEpisode):
        import os
        date_str = datetime.fromtimestamp(episode.created_at).strftime("%Y-%m-%d")
        filename = f"{self.storage_path}/{date_str}/{episode.token}.json.gz"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        data = self._episode_to_dict(episode)

        temporary = f"{filename}.tmp"
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            json.dump(data, handle, separators=(",", ":"))
        Path(temporary).replace(filename)

    @staticmethod
    def _episode_to_dict(episode: LaunchEpisode) -> Dict[str, Any]:
        return {
            "token": episode.token,
            "chain": episode.chain,
            "created_at": episode.created_at,
            "deployer": episode.deployer,
            "factory": episode.factory,
            "pair": episode.pair,
            "base_token": episode.base_token,
            "snapshots": {
                sp.value: {
                    "token": s.token,
                    "chain": s.chain,
                    "snapshot_time": s.snapshot_time.value,
                    "timestamp": s.timestamp,
                    "deployer_features": s.deployer_features,
                    "wallet_features": s.wallet_features,
                    "flow_features": s.flow_features,
                    "liquidity_features": s.liquidity_features,
                    "social_features": s.social_features,
                    "token_features": s.token_features,
                    "market_features": s.market_features,
                    "entity_graph_features": s.entity_graph_features,
                    "labels": {
                        "label_2x": s.label_2x,
                        "label_5x": s.label_5x,
                        "label_10x": s.label_10x,
                        "label_50x": s.label_50x,
                        "label_migration": s.label_migration,
                        "label_rug": s.label_rug,
                        "label_rug_time": s.label_rug_time,
                        "max_multiple": s.max_multiple,
                        "time_to_peak": s.time_to_peak,
                        "max_drawdown": s.max_drawdown,
                        "feasible_exit_multiple": s.feasible_exit_multiple,
                        "realized_pnl": s.realized_pnl
                    }
                }
                for sp, s in episode.snapshots.items()
            },
            "final_outcome": episode.final_outcome,
            "execution_attempts": episode.execution_attempts,
            "counterfactuals": episode.counterfactuals,
            "market_observations": episode.market_observations,
            "risk_report": episode.risk_report,
            "prelaunch_status": episode.prelaunch_status,
        }

    @staticmethod
    def _episode_from_dict(data: Dict[str, Any]) -> LaunchEpisode:
        episode = LaunchEpisode(
            token=str(data["token"]), chain=str(data.get("chain", "solana")),
            created_at=float(data["created_at"]), deployer=str(data.get("deployer", "")),
            factory=str(data.get("factory", "")), pair=str(data.get("pair", "")),
            base_token=str(data.get("base_token", "")),
        )
        for key, raw in (data.get("snapshots") or {}).items():
            try:
                timepoint = SnapshotTimepoint(key)
            except ValueError:
                continue
            labels = raw.get("labels") or {}
            episode.snapshots[timepoint] = LaunchSnapshot(
                token=str(raw.get("token", episode.token)), chain=str(raw.get("chain", episode.chain)),
                snapshot_time=timepoint, timestamp=float(raw.get("timestamp", episode.created_at)),
                deployer_features=dict(raw.get("deployer_features") or {}),
                wallet_features=dict(raw.get("wallet_features") or {}),
                flow_features=dict(raw.get("flow_features") or {}),
                liquidity_features=dict(raw.get("liquidity_features") or {}),
                social_features=dict(raw.get("social_features") or {}),
                token_features=dict(raw.get("token_features") or {}),
                market_features=dict(raw.get("market_features") or {}),
                entity_graph_features=dict(raw.get("entity_graph_features") or {}),
                label_2x=labels.get("label_2x"), label_5x=labels.get("label_5x"),
                label_10x=labels.get("label_10x"), label_50x=labels.get("label_50x"),
                label_migration=labels.get("label_migration"), label_rug=labels.get("label_rug"),
                label_rug_time=labels.get("label_rug_time"), max_multiple=labels.get("max_multiple"),
                time_to_peak=labels.get("time_to_peak"), max_drawdown=labels.get("max_drawdown"),
                feasible_exit_multiple=labels.get("feasible_exit_multiple"),
                realized_pnl=labels.get("realized_pnl"),
            )
        episode.final_outcome = dict(data.get("final_outcome") or {})
        episode.execution_attempts = list(data.get("execution_attempts") or [])
        episode.counterfactuals = list(data.get("counterfactuals") or [])
        episode.market_observations = list(data.get("market_observations") or [])
        episode.risk_report = dict(data.get("risk_report") or {})
        episode.prelaunch_status = str(data.get("prelaunch_status", "DATA_BLOCKED"))
        return episode

    def get_training_data(self, target_label: str, timepoints: List[SnapshotTimepoint] = None) -> Tuple[np.ndarray, np.ndarray]:
        if timepoints is None:
            timepoints = [SnapshotTimepoint.T10S, SnapshotTimepoint.T30S, SnapshotTimepoint.T1M]
        
        X_list = []
        y_list = []
        
        for episode in self.completed_episodes.values():
            for tp in timepoints:
                if tp in episode.snapshots:
                    snapshot = episode.snapshots[tp]
                    label_val = getattr(snapshot, f"label_{target_label}", None)
                    if label_val is not None:
                        features = self._flatten_features(snapshot)
                        X_list.append(features)
                        y_list.append(float(label_val))
        
        if not X_list:
            return np.array([]), np.array([])
        
        return np.array(X_list), np.array(y_list)

    def _flatten_features(self, snapshot: LaunchSnapshot) -> np.ndarray:
        feature_dict = {}
        for category in ["deployer_features", "wallet_features", "flow_features", 
                        "liquidity_features", "social_features", "token_features",
                        "market_features", "entity_graph_features"]:
            feat_dict = getattr(snapshot, category, {})
            for k, v in feat_dict.items():
                if isinstance(v, (int, float, bool)):
                    feature_dict[f"{category}.{k}"] = float(v) if isinstance(v, bool) else v
        
        return np.array([feature_dict[key] for key in sorted(feature_dict)])

    def get_stats(self) -> Dict:
        return {
            "active_episodes": len(self.active_episodes),
            "completed_episodes": len(self.completed_episodes),
            "indexed_outcomes": len(self.outcome_index),
            "storage_path": self.storage_path
        }

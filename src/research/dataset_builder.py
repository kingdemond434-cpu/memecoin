import asyncio
import logging
import time
import json
import gzip
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
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

    async def start(self):
        import os
        os.makedirs(self.storage_path, exist_ok=True)
        self._running = True
        self._snapshot_task = asyncio.create_task(self._snapshot_loop())
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def stop(self):
        self._running = False
        for task in [self._snapshot_task, self._flush_task]:
            if task:
                task.cancel()
        await self._flush_all()

    def start_episode(self, token: str, deployer: str, factory: str, pair: str, base_token: str):
        if token in self.active_episodes:
            return
        
        episode = LaunchEpisode(
            token=token,
            chain=self.chain_config.name,
            created_at=time.time(),
            deployer=deployer,
            factory=factory,
            pair=pair,
            base_token=base_token
        )
        self.active_episodes[token] = episode
        
        asyncio.create_task(self._capture_prelaunch_snapshot(episode))

    async def _capture_prelaunch_snapshot(self, episode: LaunchEpisode):
        await asyncio.sleep(0.1)
        await self._capture_snapshot(episode.token, SnapshotTimepoint.PRELAUNCH)

    async def _snapshot_loop(self):
        while self._running:
            try:
                now = time.time()
                for token, episode in list(self.active_episodes.items()):
                    elapsed = now - episode.created_at
                    
                    for sp, target_time in self.snapshot_times.items():
                        if sp == SnapshotTimepoint.PRELAUNCH:
                            continue
                        
                        if sp not in episode.snapshots and elapsed >= target_time - 2:
                            asyncio.create_task(self._capture_snapshot(token, sp))
                    
                    if elapsed > 3600:
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
        
        try:
            snapshot = LaunchSnapshot(
                token=token,
                chain=self.chain_config.name,
                snapshot_time=snapshot_time,
                timestamp=time.time()
            )
            
            snapshot.deployer_features = await self._capture_deployer_features(episode)
            snapshot.wallet_features = await self._capture_wallet_features(episode)
            snapshot.flow_features = await self._capture_flow_features(episode)
            snapshot.liquidity_features = await self._capture_liquidity_features(episode)
            snapshot.social_features = await self._capture_social_features(episode)
            snapshot.token_features = await self._capture_token_features(episode)
            snapshot.market_features = await self._capture_market_features(episode)
            snapshot.entity_graph_features = await self._capture_entity_graph_features(episode)
            
            episode.snapshots[snapshot_time] = snapshot
            
        except Exception as e:
            logger.error(f"Snapshot capture failed for {token} {snapshot_time}: {e}")

    async def _capture_deployer_features(self, episode: LaunchEpisode) -> Dict[str, Any]:
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

    async def _capture_wallet_features(self, episode: LaunchEpisode) -> Dict[str, Any]:
        smart_wallets = self.wallet_intel.get_top_wallets(limit=50)
        
        initial_buyers = []
        smart_buyers = 0
        insider_buyers = 0
        total_sol_volume = 0
        
        for buy in self.wallet_intel._recent_buys:
            if buy["token"] == episode.token:
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

    async def _capture_flow_features(self, episode: LaunchEpisode) -> Dict[str, Any]:
        return {
            "buy_velocity": 0,
            "buy_acceleration": 0,
            "organic_ratio": 0,
            "bundle_concentration": 0
        }

    async def _capture_liquidity_features(self, episode: LaunchEpisode) -> Dict[str, Any]:
        return {
            "liquidity_usd": 0,
            "liquidity_locked": False,
            "lp_burned_pct": 0
        }

    async def _capture_social_features(self, episode: LaunchEpisode) -> Dict[str, Any]:
        social_signal = self.social_intel.get_token_social_signal(episode.token)
        return social_signal

    async def _capture_token_features(self, episode: LaunchEpisode) -> Dict[str, Any]:
        return {
            "buy_tax": 0,
            "sell_tax": 0,
            "ownership_renounced": False,
            "can_mint": False,
            "can_freeze": False
        }

    async def _capture_market_features(self, episode: LaunchEpisode) -> Dict[str, Any]:
        return {
            "sol_price_usd": 0,
            "sol_24h_change": 0,
            "meme_launch_rate_1h": 0,
            "graduation_rate_24h": 0
        }

    async def _capture_entity_graph_features(self, episode: LaunchEpisode) -> Dict[str, Any]:
        dp = self.genealogy.get_deployer_profile(episode.deployer)
        cluster_risk = 0
        
        if dp and dp.wallet_profile and dp.wallet_profile.cluster_id:
            cluster = self.genealogy.find_cluster(dp.wallet_profile.cluster_id)
            if cluster:
                cluster_risk = 1 if cluster.risk_level == "critical" else 0.5 if cluster.risk_level == "high" else 0
        
        return {
            "deployer_cluster_risk": cluster_risk,
            "funding_wallet_risk": 0,
            "creator_genealogy_depth": 0
        }

    async def _finalize_episode(self, token: str):
        if token not in self.active_episodes:
            return
        
        episode = self.active_episodes.pop(token)
        
        episode.final_outcome = await self._determine_final_outcome(episode)
        
        for snapshot in episode.snapshots.values():
            labels = episode.final_outcome
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

    async def _determine_final_outcome(self, episode: LaunchEpisode) -> Dict[str, Any]:
        return {
            "max_multiple": 0,
            "migrated": False,
            "rugged": False,
            "rug_time": None,
            "time_to_peak": 0,
            "max_drawdown": 0,
            "feasible_exit_multiple": 0,
            "realized_pnl": 0
        }

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
                await self._flush_completed()
            except Exception as e:
                logger.error(f"Flush error: {e}")
            await asyncio.sleep(300)

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
        
        data = {
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
            "counterfactuals": episode.counterfactuals
        }
        
        with gzip.open(filename, 'wt') as f:
            json.dump(data, f)

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
        
        return np.array(list(feature_dict.values()))

    def get_stats(self) -> Dict:
        return {
            "active_episodes": len(self.active_episodes),
            "completed_episodes": len(self.completed_episodes),
            "storage_path": self.storage_path
        }
import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import json
import hashlib

import aiohttp
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

from src.chains.rpc_manager import ChainConfig, RPCManager
from src.strategies.genealogy_graph import GenealogyGraph, DeployerProfile
from src.strategies.wallet_intelligence import WalletIntelligenceEngine

logger = logging.getLogger(__name__)


class IntentSignal(Enum):
    FUNDING_ARRIVAL = "funding_arrival"
    WALLET_CLUSTER_CREATION = "wallet_cluster_creation"
    METADATA_CREATION = "metadata_creation"
    SOCIAL_ACCOUNT_CREATION = "social_account_creation"
    DEPLOYER_ACTIVATION = "deployer_activation"
    INFRASTRUCTURE_INTERACTION = "infrastructure_interaction"
    NARRATIVE_ACCELERATION = "narrative_acceleration"
    REPEAT_CREATOR_PATTERN = "repeat_creator_pattern"


@dataclass
class PrelaunchSignal:
    entity: str
    signal_type: IntentSignal
    strength: float
    timestamp: float
    evidence: Dict[str, Any]
    related_entities: Set[str] = field(default_factory=set)


@dataclass
class EntityIntentProfile:
    entity: str
    entity_type: str
    first_seen: float
    last_active: float
    
    funding_events: List[Dict] = field(default_factory=list)
    wallet_clusters: List[Dict] = field(default_factory=list)
    metadata_creations: List[Dict] = field(default_factory=list)
    social_creations: List[Dict] = field(default_factory=list)
    deployer_activations: List[Dict] = field(default_factory=list)
    infrastructure_interactions: List[Dict] = field(default_factory=list)
    
    prior_launches: List[str] = field(default_factory=list)
    prior_success_rate: float = 0
    prior_rug_rate: float = 0
    
    intent_score: float = 0
    launch_probability_1h: float = 0
    launch_probability_6h: float = 0
    launch_probability_24h: float = 0
    
    cluster_id: Optional[str] = None
    risk_level: str = "unknown"


class PrelaunchIntentModel:
    def __init__(
        self,
        chain_config: ChainConfig,
        rpc: RPCManager,
        genealogy: GenealogyGraph,
        wallet_intel: WalletIntelligenceEngine,
        helius_key: str
    ):
        self.chain_config = chain_config
        self.rpc = rpc
        self.genealogy = genealogy
        self.wallet_intel = wallet_intel
        self.helius_key = helius_key
        
        self.entities: Dict[str, EntityIntentProfile] = {}
        self.signals: deque = deque(maxlen=10000)
        self.pending_launches: Dict[str, Dict] = {}
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._scoring_task: Optional[asyncio.Task] = None
        
        self._model: Optional[CalibratedClassifierCV] = None
        self._is_trained = False
        self._feature_names = [
            "funding_velocity", "wallet_cluster_size", "metadata_count",
            "social_count", "deployer_age", "infra_interactions",
            "prior_success_rate", "prior_rug_rate", "cluster_risk",
            "narrative_velocity", "creator_frequency"
        ]

    async def start(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=50)
        )
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        self._scoring_task = asyncio.create_task(self._scoring_loop())
        await self._load_historical_entities()

    async def stop(self):
        self._running = False
        for task in [self._monitor_task, self._scoring_task]:
            if task:
                task.cancel()
        if self._session:
            await self._session.close()

    async def _load_historical_entities(self):
        for deployer_addr, dp in self.genealogy.deployers.items():
            entity = EntityIntentProfile(
                entity=deployer_addr,
                entity_type="deployer",
                first_seen=dp.wallet_profile.first_seen if dp.wallet_profile else time.time(),
                last_active=dp.wallet_profile.last_seen if dp.wallet_profile else time.time(),
                prior_launches=dp.tokens_created,
                prior_success_rate=dp.success_rate,
                prior_rug_rate=dp.rug_rate
            )
            self.entities[deployer_addr] = entity

    async def _monitor_loop(self):
        while self._running:
            try:
                await self._scan_new_funding()
                await self._scan_wallet_clusters()
                await self._scan_metadata()
                await self._scan_social_accounts()
                await self._scan_deployer_activation()
                await self._scan_infrastructure()
            except Exception as e:
                logger.error(f"Prelaunch monitor error: {e}")
            await asyncio.sleep(60)

    async def _scoring_loop(self):
        while self._running:
            try:
                await self._score_all_entities()
                await self._predict_imminent_launches()
            except Exception as e:
                logger.error(f"Prelaunch scoring error: {e}")
            await asyncio.sleep(300)

    async def _scan_new_funding(self):
        try:
            async with self._session.get(
                f"https://api.helius.xyz/v0/transactions",
                params={
                    "api-key": self.helius_key,
                    "type": "TRANSFER",
                    "limit": 200,
                    "commitment": "processed"
                }
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for tx in data:
                        await self._process_funding_transfer(tx)
        except Exception as e:
            logger.debug(f"Funding scan error: {e}")

    async def _process_funding_transfer(self, tx: Dict):
        try:
            native_transfers = tx.get("nativeTransfers", [])
            for transfer in native_transfers:
                amount = float(transfer.get("amount", 0)) / 1e9
                if amount < 0.5:
                    continue
                
                from_addr = transfer.get("fromUserAccount")
                to_addr = transfer.get("toUserAccount")
                
                if not from_addr or not to_addr:
                    continue
                
                is_new_wallet = to_addr not in self.genealogy.wallets
                is_known_funder = from_addr in self.genealogy.wallets
                
                if is_new_wallet or is_known_funder:
                    signal = PrelaunchSignal(
                        entity=to_addr,
                        signal_type=IntentSignal.FUNDING_ARRIVAL,
                        strength=min(amount / 10, 1.0),
                        timestamp=tx.get("timestamp", time.time()),
                        evidence={
                            "from": from_addr,
                            "amount_sol": amount,
                            "tx_sig": tx.get("signature"),
                            "is_new_wallet": is_new_wallet
                        }
                    )
                    self.signals.append(signal)
                    await self._update_entity_signal(to_addr, signal)
                    
        except Exception as e:
            logger.debug(f"Funding transfer process error: {e}")

    async def _scan_wallet_clusters(self):
        pass

    async def _scan_metadata(self):
        pass

    async def _scan_social_accounts(self):
        pass

    async def _scan_deployer_activation(self):
        for deployer_addr, dp in self.genealogy.deployers.items():
            if deployer_addr in self.entities:
                entity = self.entities[deployer_addr]
                if dp.wallet_profile and dp.wallet_profile.last_seen > entity.last_active:
                    entity.last_active = dp.wallet_profile.last_seen
                    signal = PrelaunchSignal(
                        entity=deployer_addr,
                        signal_type=IntentSignal.DEPLOYER_ACTIVATION,
                        strength=0.5,
                        timestamp=dp.wallet_profile.last_seen,
                        evidence={"recent_tx_count": dp.wallet_profile.tx_count}
                    )
                    self.signals.append(signal)

    async def _scan_infrastructure(self):
        pass

    async def _update_entity_signal(self, entity_addr: str, signal: PrelaunchSignal):
        if entity_addr not in self.entities:
            self.entities[entity_addr] = EntityIntentProfile(
                entity=entity_addr,
                entity_type="unknown",
                first_seen=signal.timestamp,
                last_active=signal.timestamp
            )
        
        entity = self.entities[entity_addr]
        entity.last_active = signal.timestamp
        
        if signal.signal_type == IntentSignal.FUNDING_ARRIVAL:
            entity.funding_events.append(signal.evidence)
        elif signal.signal_type == IntentSignal.WALLET_CLUSTER_CREATION:
            entity.wallet_clusters.append(signal.evidence)
        elif signal.signal_type == IntentSignal.METADATA_CREATION:
            entity.metadata_creations.append(signal.evidence)
        elif signal.signal_type == IntentSignal.SOCIAL_ACCOUNT_CREATION:
            entity.social_creations.append(signal.evidence)
        elif signal.signal_type == IntentSignal.DEPLOYER_ACTIVATION:
            entity.deployer_activations.append(signal.evidence)
        elif signal.signal_type == IntentSignal.INFRASTRUCTURE_INTERACTION:
            entity.infrastructure_interactions.append(signal.evidence)

    async def _score_all_entities(self):
        now = time.time()
        for entity in self.entities.values():
            if now - entity.last_active > 86400:
                continue
            
            features = self._extract_features(entity)
            entity.intent_score = self._calculate_intent_score(features)
            entity.launch_probability_1h = self._estimate_launch_prob(entity, 3600)
            entity.launch_probability_6h = self._estimate_launch_prob(entity, 21600)
            entity.launch_probability_24h = self._estimate_launch_prob(entity, 86400)
            
            if entity.intent_score > 0.7:
                entity.risk_level = "high_intent"
            elif entity.intent_score > 0.4:
                entity.risk_level = "medium_intent"
            else:
                entity.risk_level = "low_intent"

    def _extract_features(self, entity: EntityIntentProfile) -> Dict[str, float]:
        now = time.time()
        recent_window = 3600
        
        recent_funding = [f for f in entity.funding_events if now - f.get("timestamp", 0) < recent_window]
        funding_velocity = len(recent_funding) / max(recent_window / 3600, 1)
        
        wallet_cluster_size = sum(len(c.get("wallets", [])) for c in entity.wallet_clusters)
        
        metadata_count = len(entity.metadata_creations)
        social_count = len(entity.social_creations)
        
        deployer_age = (now - entity.first_seen) / 86400 if entity.first_seen else 0
        
        infra_interactions = len(entity.infrastructure_interactions)
        
        cluster_risk = 0
        if entity.cluster_id:
            cluster = self.genealogy.clusters.get(entity.cluster_id)
            if cluster:
                cluster_risk = 1 if cluster.risk_level == "critical" else 0.5 if cluster.risk_level == "high" else 0
        
        narrative_velocity = self._get_narrative_velocity(entity.entity)
        
        creator_frequency = len(entity.prior_launches) / max(deployer_age, 1) if deployer_age > 0 else 0
        
        return {
            "funding_velocity": funding_velocity,
            "wallet_cluster_size": wallet_cluster_size,
            "metadata_count": metadata_count,
            "social_count": social_count,
            "deployer_age": deployer_age,
            "infra_interactions": infra_interactions,
            "prior_success_rate": entity.prior_success_rate,
            "prior_rug_rate": entity.prior_rug_rate,
            "cluster_risk": cluster_risk,
            "narrative_velocity": narrative_velocity,
            "creator_frequency": creator_frequency
        }

    def _calculate_intent_score(self, features: Dict[str, float]) -> float:
        score = 0.0
        
        score += min(features["funding_velocity"] / 5, 1) * 0.25
        score += min(features["wallet_cluster_size"] / 10, 1) * 0.20
        score += min(features["metadata_count"] / 3, 1) * 0.10
        score += min(features["social_count"] / 3, 1) * 0.10
        score += min(features["infra_interactions"] / 5, 1) * 0.10
        
        score += features["prior_success_rate"] * 0.15
        score -= features["prior_rug_rate"] * 0.20
        score -= features["cluster_risk"] * 0.15
        
        score += min(features["narrative_velocity"] / 10, 1) * 0.10
        score += min(features["creator_frequency"] / 2, 1) * 0.05
        
        if features["deployer_age"] < 1:
            score += 0.1
        elif features["deployer_age"] > 30:
            score += 0.05
        
        return max(0, min(1, score))

    def _estimate_launch_prob(self, entity: EntityIntentProfile, horizon: float) -> float:
        base = entity.intent_score
        
        if entity.prior_success_rate > 0.3:
            base *= 1.3
        elif entity.prior_rug_rate > 0.5:
            base *= 0.5
        
        time_factor = min(horizon / 3600, 24) / 24
        return min(1, base * (0.3 + 0.7 * time_factor))

    def _get_narrative_velocity(self, entity: str) -> float:
        return 0.0

    async def _predict_imminent_launches(self):
        high_intent = [
            e for e in self.entities.values()
            if e.intent_score > 0.7 and e.launch_probability_1h > 0.5
        ]
        
        for entity in high_intent:
            if entity.entity not in self.pending_launches:
                self.pending_launches[entity.entity] = {
                    "entity": entity.entity,
                    "intent_score": entity.intent_score,
                    "launch_prob_1h": entity.launch_probability_1h,
                    "launch_prob_6h": entity.launch_probability_6h,
                    "risk_level": entity.risk_level,
                    "detected_at": time.time(),
                    "features": self._extract_features(entity)
                }

    def get_imminent_launches(self, min_prob: float = 0.5) -> List[Dict]:
        return [
            v for v in self.pending_launches.values()
            if v["launch_prob_1h"] >= min_prob
        ]

    def get_entity_profile(self, entity: str) -> Optional[EntityIntentProfile]:
        return self.entities.get(entity)

    def get_top_entities(self, limit: int = 20) -> List[EntityIntentProfile]:
        entities = sorted(
            [e for e in self.entities.values() if e.intent_score > 0.3],
            key=lambda x: x.intent_score,
            reverse=True
        )
        return entities[:limit]

    def train_launch_predictor(self, historical_data: List[Dict]):
        if len(historical_data) < 50:
            return
        
        X = []
        y = []
        for record in historical_data:
            features = record.get("features", {})
            X.append([features.get(f, 0) for f in self._feature_names])
            y.append(1 if record.get("launched_within_1h", False) else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        base = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        self._model = CalibratedClassifierCV(base, method='isotonic', cv=3)
        self._model.fit(X, y)
        self._is_trained = True
        logger.info(f"Trained pre-launch predictor on {len(X)} samples")

    def predict_launch_probability(self, entity: str, horizon_hours: float = 1) -> float:
        if not self._is_trained or entity not in self.entities:
            return 0.0
        
        entity_obj = self.entities[entity]
        features = self._extract_features(entity_obj)
        X = np.array([[features.get(f, 0) for f in self._feature_names]])
        
        try:
            prob = self._model.predict_proba(X)[0, 1]
            return float(prob)
        except Exception:
            return entity_obj.launch_probability_1h if horizon_hours <= 1 else (
                entity_obj.launch_probability_6h if horizon_hours <= 6 else entity_obj.launch_probability_24h
            )

    def get_stats(self) -> Dict:
        high = sum(1 for e in self.entities.values() if e.intent_score > 0.7)
        medium = sum(1 for e in self.entities.values() if 0.4 < e.intent_score <= 0.7)
        return {
            "tracked_entities": len(self.entities),
            "high_intent": high,
            "medium_intent": medium,
            "pending_launches": len(self.pending_launches),
            "recent_signals": len(self.signals),
            "model_trained": self._is_trained
        }
import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

from src.chains.rpc_manager import ChainConfig, RPCManager
from src.strategies.genealogy_graph import GenealogyGraph
from src.strategies.wallet_intelligence import WalletIntelligenceEngine
from src.strategies.information_graph import AdversarialAdaptationDetector

logger = logging.getLogger(__name__)


class HazardTrigger(Enum):
    CREATOR_TRANSFER = "creator_transfer"
    INSIDER_SELL = "insider_sell"
    SMART_WALLET_EXIT = "smart_wallet_exit"
    LIQUIDITY_WITHDRAWAL = "liquidity_withdrawal"
    CONCENTRATION_CHANGE = "concentration_change"
    BUY_DECELERATION = "buy_deceleration"
    SELL_ACCELERATION = "sell_acceleration"
    HOLDER_DISTRIBUTION = "holder_distribution"
    VOLUME_COLLAPSE = "volume_collapse"
    ROUTE_DEGRADATION = "route_degradation"
    SOCIAL_VELOCITY_COLLAPSE = "social_velocity_collapse"
    FAILED_MIGRATION = "failed_migration"
    DEV_WALLET_ACTIVATION = "dev_wallet_activation"
    BUNDLE_DETECTION = "bundle_detection"


@dataclass
class HazardSignal:
    trigger: HazardTrigger
    strength: float
    confidence: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HazardState:
    token: str
    chain: str
    current_hazard: float = 0.0
    hazard_30s: float = 0.0
    hazard_5m: float = 0.0
    hazard_30m: float = 0.0
    signals: List[HazardSignal] = field(default_factory=list)
    last_update: float = field(default_factory=time.time)
    exit_recommended: bool = False
    exit_urgency: str = "none"


class ContinuousRugHazardModel:
    def __init__(
        self,
        chain_config: ChainConfig,
        rpc: RPCManager,
        genealogy: GenealogyGraph,
        wallet_intel: WalletIntelligenceEngine,
        adversarial: AdversarialAdaptationDetector
    ):
        self.chain_config = chain_config
        self.rpc = rpc
        self.genealogy = genealogy
        self.wallet_intel = wallet_intel
        self.adversarial = adversarial
        
        self.hazard_states: Dict[str, HazardState] = {}
        self.hazard_models: Dict[str, CalibratedClassifierCV] = {}
        self.is_trained = False
        
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._model_task: Optional[asyncio.Task] = None
        
        self._feature_names = [
            "creator_sol_balance_change",
            "insider_sell_volume",
            "smart_wallet_exit_count",
            "smart_wallet_exit_rate",
            "liquidity_change_pct",
            "lp_burn_rate",
            "top10_concentration_change",
            "deployer_pct_change",
            "buy_velocity_change",
            "sell_velocity_change",
            "buy_sell_ratio",
            "volume_change_pct",
            "price_impact_change",
            "holder_count_change",
            "fresh_wallet_pct_change",
            "route_availability",
            "social_velocity_change",
            "migration_progress",
            "dev_wallet_tx_count",
            "bundle_wallet_count"
        ]
        
        self.trigger_weights = {
            HazardTrigger.CREATOR_TRANSFER: 0.25,
            HazardTrigger.INSIDER_SELL: 0.30,
            HazardTrigger.SMART_WALLET_EXIT: 0.20,
            HazardTrigger.LIQUIDITY_WITHDRAWAL: 0.25,
            HazardTrigger.CONCENTRATION_CHANGE: 0.15,
            HazardTrigger.BUY_DECELERATION: 0.15,
            HazardTrigger.SELL_ACCELERATION: 0.20,
            HazardTrigger.HOLDER_DISTRIBUTION: 0.15,
            HazardTrigger.VOLUME_COLLAPSE: 0.20,
            HazardTrigger.ROUTE_DEGRADATION: 0.10,
            HazardTrigger.SOCIAL_VELOCITY_COLLAPSE: 0.10,
            HazardTrigger.FAILED_MIGRATION: 0.35,
            HazardTrigger.DEV_WALLET_ACTIVATION: 0.20,
            HazardTrigger.BUNDLE_DETECTION: 0.30
        }

    async def start(self):
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        self._model_task = asyncio.create_task(self._model_update_loop())
        await self._load_historical_model()

    async def stop(self):
        self._running = False
        for task in [self._monitor_task, self._model_task]:
            if task:
                task.cancel()

    async def _load_historical_model(self):
        pass

    async def _monitor_loop(self):
        while self._running:
            try:
                await self._update_all_hazards()
            except Exception as e:
                logger.error(f"Hazard monitor error: {e}")
            await asyncio.sleep(2)

    async def _model_update_loop(self):
        while self._running:
            try:
                await self._retrain_models()
            except Exception as e:
                logger.error(f"Hazard model retrain error: {e}")
            await asyncio.sleep(3600)

    async def _update_all_hazards(self):
        for token, state in list(self.hazard_states.items()):
            if time.time() - state.last_update > 300:
                await self._compute_hazard(token)

    async def _compute_hazard(self, token: str) -> HazardState:
        if token not in self.hazard_states:
            self.hazard_states[token] = HazardState(token=token, chain=self.chain_config.name)
        
        state = self.hazard_states[token]
        
        signals = await self._collect_hazard_signals(token)
        state.signals = signals[-50:]
        
        hazard_components = {}
        for signal in signals:
            weight = self.trigger_weights.get(signal.trigger, 0.1)
            adjusted_strength = signal.strength * signal.confidence
            adjusted_strength *= self._get_adaptive_weight(signal.trigger, adjusted_strength)
            hazard_components[signal.trigger.value] = adjusted_strength * weight
        
        base_hazard = sum(hazard_components.values())
        
        if self.is_trained and token in self.hazard_models:
            features = await self._extract_hazard_features(token)
            if features is not None:
                try:
                    model_hazard = self.hazard_models[token].predict_proba([features])[0, 1]
                    base_hazard = 0.7 * base_hazard + 0.3 * model_hazard
                except Exception:
                    pass
        
        state.current_hazard = min(1.0, base_hazard)
        state.hazard_30s = self._project_hazard(state.current_hazard, 30)
        state.hazard_5m = self._project_hazard(state.current_hazard, 300)
        state.hazard_30m = self._project_hazard(state.current_hazard, 1800)
        
        state.exit_recommended = state.hazard_30s > 0.6 or state.hazard_5m > 0.75
        state.exit_urgency = self._get_urgency(state.hazard_30s, state.hazard_5m)
        state.last_update = time.time()
        
        return state

    def _get_adaptive_weight(self, trigger: HazardTrigger, strength: float) -> float:
        feature_name = trigger.value
        return self.adversarial.get_adaptive_weight(feature_name, 1.0)

    def _project_hazard(self, current: float, seconds: int) -> float:
        return min(1.0, current * (1 + seconds / 300))

    def _get_urgency(self, hazard_30s: float, hazard_5m: float) -> str:
        if hazard_30s > 0.8 or hazard_5m > 0.9:
            return "CRITICAL"
        elif hazard_30s > 0.6 or hazard_5m > 0.75:
            return "HIGH"
        elif hazard_30s > 0.4 or hazard_5m > 0.5:
            return "MEDIUM"
        elif hazard_30s > 0.2 or hazard_5m > 0.3:
            return "LOW"
        return "NONE"

    async def _collect_hazard_signals(self, token: str) -> List[HazardSignal]:
        signals = []
        
        creator_signals = await self._check_creator_activity(token)
        signals.extend(creator_signals)
        
        insider_signals = await self._check_insider_selling(token)
        signals.extend(insider_signals)
        
        smart_exit_signals = await self._check_smart_wallet_exits(token)
        signals.extend(smart_exit_signals)
        
        liq_signals = await self._check_liquidity_changes(token)
        signals.extend(liq_signals)
        
        conc_signals = await self._check_concentration_changes(token)
        signals.extend(conc_signals)
        
        flow_signals = await self._check_flow_changes(token)
        signals.extend(flow_signals)
        
        route_signals = await self._check_route_health(token)
        signals.extend(route_signals)
        
        social_signals = await self._check_social_changes(token)
        signals.extend(social_signals)
        
        return signals

    async def _check_creator_activity(self, token: str) -> List[HazardSignal]:
        signals = []
        
        return signals

    async def _check_insider_selling(self, token: str) -> List[HazardSignal]:
        signals = []
        
        return signals

    async def _check_smart_wallet_exits(self, token: str) -> List[HazardSignal]:
        signals = []
        
        smart_wallets = self.wallet_intel.get_top_wallets(limit=50)
        for ws in smart_wallets:
            recent_sells = [s for s in self.wallet_intel._recent_sells 
                          if s["wallet"] == ws.wallet and s["token"] == token]
            
            if recent_sells:
                total_sold = sum(s["amount"] * s["price"] for s in recent_sells)
                signal = HazardSignal(
                    trigger=HazardTrigger.SMART_WALLET_EXIT,
                    strength=min(total_sold / 1000, 1.0),
                    confidence=ws.overall_score,
                    timestamp=time.time(),
                    metadata={"wallet": ws.wallet, "sell_count": len(recent_sells)}
                )
                signals.append(signal)
        
        return signals

    async def _check_liquidity_changes(self, token: str) -> List[HazardSignal]:
        return []

    async def _check_concentration_changes(self, token: str) -> List[HazardSignal]:
        return []

    async def _check_flow_changes(self, token: str) -> List[HazardSignal]:
        return []

    async def _check_route_health(self, token: str) -> List[HazardSignal]:
        return []

    async def _check_social_changes(self, token: str) -> List[HazardSignal]:
        return []

    async def _extract_hazard_features(self, token: str) -> Optional[np.ndarray]:
        return None

    async def _retrain_models(self):
        pass

    def get_hazard(self, token: str) -> Optional[HazardState]:
        return self.hazard_states.get(token)

    def should_exit(self, token: str, position: Dict) -> Tuple[bool, str, float]:
        state = self.hazard_states.get(token)
        if not state:
            return False, "no_hazard_data", 0.0
        
        if state.exit_recommended:
            exit_pct = 1.0 if state.exit_urgency == "CRITICAL" else 0.5
            return True, state.exit_urgency, exit_pct
        
        return False, "hold", 0.0

    def get_stats(self) -> Dict:
        critical = sum(1 for s in self.hazard_states.values() if s.exit_urgency == "CRITICAL")
        high = sum(1 for s in self.hazard_states.values() if s.exit_urgency == "HIGH")
        return {
            "tracked_tokens": len(self.hazard_states),
            "critical": critical,
            "high": high,
            "model_trained": self.is_trained
        }
"""Continuous, evidence-backed Solana exit hazard tracking."""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from src.chains.rpc_manager import ChainConfig, RPCManager
from src.strategies.genealogy_graph import GenealogyGraph
from src.strategies.information_graph import AdversarialAdaptationDetector
from src.strategies.wallet_intelligence import WalletIntelligenceEngine

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
    exit_urgency: str = "NONE"
    data_status: str = "DATA_BLOCKED"
    blocked_reason: str = "no_market_observations"


class ContinuousRugHazardModel:
    """Combines observed flow, route, holder and liquidity deterioration."""

    def __init__(self, chain_config: ChainConfig, rpc: RPCManager, genealogy: GenealogyGraph,
                 wallet_intel: WalletIntelligenceEngine, adversarial: AdversarialAdaptationDetector):
        self.chain_config = chain_config
        self.rpc = rpc
        self.genealogy = genealogy
        self.wallet_intel = wallet_intel
        self.adversarial = adversarial
        self.hazard_states: Dict[str, HazardState] = {}
        self.observations: Dict[str, Deque[Dict[str, Any]]] = defaultdict(lambda: deque(maxlen=5_000))
        self.token_metadata: Dict[str, Dict[str, Any]] = {}
        self.is_trained = False
        self.data_status = "DATA_BLOCKED"
        self.data_status_detail = "no versioned chronological hazard training artifact"
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self.trigger_weights = {
            HazardTrigger.CREATOR_TRANSFER: 0.38, HazardTrigger.INSIDER_SELL: 0.38,
            HazardTrigger.SMART_WALLET_EXIT: 0.24, HazardTrigger.LIQUIDITY_WITHDRAWAL: 0.45,
            HazardTrigger.CONCENTRATION_CHANGE: 0.20, HazardTrigger.BUY_DECELERATION: 0.15,
            HazardTrigger.SELL_ACCELERATION: 0.22, HazardTrigger.VOLUME_COLLAPSE: 0.20,
            HazardTrigger.ROUTE_DEGRADATION: 0.45, HazardTrigger.SOCIAL_VELOCITY_COLLAPSE: 0.08,
            HazardTrigger.FAILED_MIGRATION: 0.30, HazardTrigger.DEV_WALLET_ACTIVATION: 0.25,
            HazardTrigger.BUNDLE_DETECTION: 0.25,
        }

    async def start(self):
        self._running = True
        await self._load_historical_model()
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop(self):
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                return

    async def _load_historical_model(self):
        self.data_status = "DATA_BLOCKED"
        self.data_status_detail = "no versioned chronological hazard training artifact"

    def register_token(self, token: str, metadata: Optional[Dict[str, Any]] = None) -> HazardState:
        state = self.hazard_states.setdefault(token, HazardState(token=token, chain=self.chain_config.name))
        if metadata:
            self.token_metadata[token] = {**self.token_metadata.get(token, {}), **metadata}
        return state

    def record_observation(self, token: str, observation: Dict[str, Any]):
        if not token or not isinstance(observation, dict):
            return
        item = dict(observation)
        item.setdefault("timestamp", time.time())
        item.setdefault("type", "unknown")
        self.register_token(token)
        self.observations[token].append(item)

    async def _monitor_loop(self):
        while self._running:
            try:
                for token in list(self.hazard_states):
                    await self._compute_hazard(token)
            except Exception as exc:
                logger.error("Hazard monitor error: %s", exc)
            await asyncio.sleep(2)

    async def _compute_hazard(self, token: str) -> HazardState:
        state = self.register_token(token)
        observations = list(self.observations.get(token, ()))
        if not observations:
            state.data_status = "DATA_BLOCKED"
            state.blocked_reason = "no_market_observations"
            state.last_update = time.time()
            return state
        signals = await self._collect_hazard_signals(token)
        state.signals = signals[-50:]
        survival = 1.0
        for signal in signals:
            weight = self.trigger_weights.get(signal.trigger, 0.10)
            adaptive = self.adversarial.get_adaptive_weight(signal.trigger.value, 1.0)
            component = float(np.clip(signal.strength * signal.confidence * weight * adaptive, 0, 0.95))
            survival *= 1.0 - component
        state.current_hazard = float(np.clip(1.0 - survival, 0, 1))
        state.hazard_30s = self._project_hazard(state.current_hazard, 30)
        state.hazard_5m = self._project_hazard(state.current_hazard, 300)
        state.hazard_30m = self._project_hazard(state.current_hazard, 1_800)
        state.exit_urgency = self._get_urgency(state.hazard_30s, state.hazard_5m)
        state.exit_recommended = state.exit_urgency in {"HIGH", "CRITICAL"}
        state.data_status = "OK"
        state.blocked_reason = ""
        state.last_update = time.time()
        return state

    @staticmethod
    def _project_hazard(current: float, seconds: int) -> float:
        persistence = 1.0 - np.exp(-seconds / 600.0)
        return float(np.clip(current + (1.0 - current) * current * persistence, 0, 1))

    @staticmethod
    def _get_urgency(hazard_30s: float, hazard_5m: float) -> str:
        if hazard_30s >= 0.80 or hazard_5m >= 0.90:
            return "CRITICAL"
        if hazard_30s >= 0.60 or hazard_5m >= 0.75:
            return "HIGH"
        if hazard_30s >= 0.40 or hazard_5m >= 0.50:
            return "MEDIUM"
        if hazard_30s >= 0.20 or hazard_5m >= 0.30:
            return "LOW"
        return "NONE"

    async def _collect_hazard_signals(self, token: str) -> List[HazardSignal]:
        observations = list(self.observations.get(token, ()))
        if not observations:
            return []
        now = time.time()
        recent = [item for item in observations if now - float(item.get("timestamp", now)) <= 60]
        prior = [item for item in observations if 60 < now - float(item.get("timestamp", now)) <= 300]
        signals: List[HazardSignal] = []
        mapping = {"creator_transfer": HazardTrigger.CREATOR_TRANSFER,
                   "dev_wallet_activation": HazardTrigger.DEV_WALLET_ACTIVATION,
                   "failed_migration": HazardTrigger.FAILED_MIGRATION, "bundle": HazardTrigger.BUNDLE_DETECTION}
        for item in recent:
            if item.get("type") in mapping:
                signals.append(self._signal(mapping[item["type"]], item.get("strength", 1), item.get("confidence", 0.8), item))

        trades_recent = [item for item in recent if item.get("type") == "trade"]
        trades_prior = [item for item in prior if item.get("type") == "trade"]
        buy_recent = sum(self._notional(item) for item in trades_recent if item.get("side") == "buy")
        sell_recent = sum(self._notional(item) for item in trades_recent if item.get("side") == "sell")
        buy_prior = sum(self._notional(item) for item in trades_prior if item.get("side") == "buy") / 4.0
        sell_prior = sum(self._notional(item) for item in trades_prior if item.get("side") == "sell") / 4.0
        total_recent, total_prior = buy_recent + sell_recent, buy_prior + sell_prior
        if total_recent > 0 and sell_recent / total_recent >= 0.65:
            signals.append(self._signal(HazardTrigger.SELL_ACCELERATION, sell_recent / total_recent, 0.85, {"sell_share": sell_recent / total_recent}))
        elif len(trades_recent) >= 4:
            # Pump instruction arguments expose a limit, not the actual quote
            # paid. Until balance-delta enrichment arrives, trade counts are a
            # lower-confidence fallback and are never presented as notional.
            sell_count = sum(item.get("side") == "sell" for item in trades_recent)
            sell_share = sell_count / len(trades_recent)
            if sell_share >= 0.75:
                signals.append(self._signal(
                    HazardTrigger.SELL_ACCELERATION, sell_share, 0.55,
                    {"sell_share_by_count": sell_share, "measurement": "count_fallback"},
                ))
        if buy_prior > 0 and buy_recent < buy_prior * 0.35:
            signals.append(self._signal(HazardTrigger.BUY_DECELERATION, 1 - buy_recent / buy_prior, 0.75, {}))
        if total_prior > 0 and total_recent < total_prior * 0.25:
            signals.append(self._signal(HazardTrigger.VOLUME_COLLAPSE, 1 - total_recent / total_prior, 0.70, {}))

        for item in self._latest_by_type(observations).values():
            event_type = item.get("type")
            if event_type == "liquidity" and float(item.get("change_pct", 0)) <= -0.15:
                signals.append(self._signal(HazardTrigger.LIQUIDITY_WITHDRAWAL, min(abs(float(item["change_pct"])), 1), 0.95, item))
            elif event_type == "concentration" and float(item.get("top10_change_pct", 0)) >= 0.10:
                signals.append(self._signal(HazardTrigger.CONCENTRATION_CHANGE, min(float(item["top10_change_pct"]) * 3, 1), 0.80, item))
            elif event_type == "route":
                feasible, impact = item.get("feasible"), float(item.get("price_impact_pct", 0) or 0)
                if feasible is False or impact >= 0.15:
                    signals.append(self._signal(HazardTrigger.ROUTE_DEGRADATION, 1 if feasible is False else min(impact * 3, 1), 0.98, item))
            elif event_type == "social" and float(item.get("velocity_change_pct", 0)) <= -0.70:
                signals.append(self._signal(HazardTrigger.SOCIAL_VELOCITY_COLLAPSE, abs(float(item["velocity_change_pct"])), 0.50, item))

        smart_wallets = {score.wallet: score for score in self.wallet_intel.get_top_wallets(limit=50)}
        for item in trades_recent:
            if item.get("side") != "sell":
                continue
            score = smart_wallets.get(item.get("wallet"))
            if score:
                strength = min(self._notional(item) / 1_000, 1) if self._notional(item) else 0.25
                signals.append(self._signal(HazardTrigger.SMART_WALLET_EXIT, strength, score.overall_score, item))
            if item.get("is_insider"):
                strength = min(self._notional(item) / 1_000, 1) if self._notional(item) else 0.25
                signals.append(self._signal(HazardTrigger.INSIDER_SELL, strength, 0.90, item))
        return signals

    @staticmethod
    def _notional(item: Dict[str, Any]) -> float:
        if item.get("notional_usd") is not None:
            return max(0.0, float(item["notional_usd"]))
        return max(0.0, float(item.get("amount", 0) or 0) * float(item.get("price", 0) or 0))

    @staticmethod
    def _latest_by_type(observations: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        latest: Dict[str, Dict[str, Any]] = {}
        for item in observations:
            key = str(item.get("type", "unknown"))
            if key not in latest or float(item.get("timestamp", 0)) >= float(latest[key].get("timestamp", 0)):
                latest[key] = item
        return latest

    @staticmethod
    def _signal(trigger: HazardTrigger, strength: Any, confidence: Any, metadata: Dict[str, Any]) -> HazardSignal:
        return HazardSignal(trigger, float(np.clip(float(strength or 0), 0, 1)),
                            float(np.clip(float(confidence or 0), 0, 1)),
                            float(metadata.get("timestamp", time.time())), dict(metadata))

    def get_hazard(self, token: str) -> Optional[HazardState]:
        return self.hazard_states.get(token)

    def should_exit(self, token: str, position: Dict[str, Any]) -> Tuple[bool, str, float]:
        state = self.hazard_states.get(token)
        if not state or state.data_status != "OK":
            return False, "DATA_BLOCKED", 0.0
        if state.exit_urgency == "CRITICAL":
            return True, "CRITICAL", 1.0
        if state.exit_urgency == "HIGH":
            return True, "HIGH", 0.50
        return False, "hold", 0.0

    def get_stats(self) -> Dict[str, Any]:
        states = list(self.hazard_states.values())
        return {"tracked_tokens": len(states), "critical": sum(s.exit_urgency == "CRITICAL" for s in states),
                "high": sum(s.exit_urgency == "HIGH" for s in states),
                "data_blocked_tokens": sum(s.data_status == "DATA_BLOCKED" for s in states),
                "observations": sum(len(items) for items in self.observations.values()),
                "model_trained": self.is_trained, "model_status": self.data_status,
                "model_status_detail": self.data_status_detail}

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from src.chains.rpc_manager import ChainConfig, RPCManager
from src.strategies.genealogy_graph import GenealogyGraph
from src.strategies.wallet_intelligence import WalletIntelligenceEngine, WalletRegime
from src.strategies.social_intelligence import SocialIntelligenceEngine, SocialPlatform
from src.strategies.prelaunch_intent import PrelaunchIntentModel, IntentSignal

logger = logging.getLogger(__name__)


class LeadEventType(Enum):
    DEPLOYER_ACTIVITY = "deployer_activity"
    ELITE_WALLET_BUY = "elite_wallet_buy"
    WALLET_CLUSTER_BUY = "wallet_cluster_buy"
    OBSCURE_X_MENTION = "obscure_x_mention"
    SECOND_WALLET_CLUSTER = "second_wallet_cluster"
    MID_KOL_MENTION = "mid_kol_mention"
    RETAIL_PILE_IN = "retail_pile_in"
    MIGRATION = "migration"
    DEV_SELL = "dev_sell"
    SMART_WALLET_EXIT = "smart_wallet_exit"


@dataclass
class LeadEvent:
    token: str
    event_type: LeadEventType
    source: str
    source_type: str
    timestamp: float
    lead_time_seconds: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LeadPath:
    token: str
    sequence: List[LeadEventType]
    timestamps: List[float]
    sources: List[str]
    total_lead_time: float
    outcome: Dict[str, Any]


class InformationLeadGraph:
    def __init__(
        self,
        chain_config: ChainConfig,
        rpc: RPCManager,
        genealogy: GenealogyGraph,
        wallet_intel: WalletIntelligenceEngine,
        social_intel: SocialIntelligenceEngine,
        prelaunch: PrelaunchIntentModel
    ):
        self.chain_config = chain_config
        self.rpc = rpc
        self.genealogy = genealogy
        self.wallet_intel = wallet_intel
        self.social_intel = social_intel
        self.prelaunch = prelaunch
        
        self.lead_events: Dict[str, List[LeadEvent]] = defaultdict(list)
        self.completed_paths: List[LeadPath] = []
        
        self.source_rankings: Dict[str, Dict[LeadEventType, float]] = defaultdict(dict)
        self.path_patterns: Dict[str, int] = defaultdict(int)
        
        self._running = False
        self._analysis_task: Optional[asyncio.Task] = None

    async def start(self):
        self._running = True
        self._analysis_task = asyncio.create_task(self._analysis_loop())

    async def stop(self):
        self._running = False
        if self._analysis_task:
            self._analysis_task.cancel()

    def record_event(self, token: str, event_type: LeadEventType, source: str, 
                     source_type: str, timestamp: float, metadata: Dict = None):
        event = LeadEvent(
            token=token,
            event_type=event_type,
            source=source,
            source_type=source_type,
            timestamp=timestamp,
            metadata=metadata or {}
        )
        self.lead_events[token].append(event)

    async def _analysis_loop(self):
        while self._running:
            try:
                await self._analyze_completed_tokens()
                await self._update_source_rankings()
                await self._mine_path_patterns()
            except Exception as e:
                logger.error(f"Lead graph analysis error: {e}")
            await asyncio.sleep(300)

    async def _analyze_completed_tokens(self):
        now = time.time()
        for token, events in list(self.lead_events.items()):
            if len(events) < 3:
                continue
            
            if now - max(e.timestamp for e in events) > 3600:
                await self._finalize_path(token)
                del self.lead_events[token]

    async def _finalize_path(self, token: str):
        events = sorted(self.lead_events[token], key=lambda x: x.timestamp)
        if len(events) < 2:
            return
        
        first_time = events[0].timestamp
        for event in events:
            event.lead_time_seconds = event.timestamp - first_time
        
        outcome = await self._get_token_outcome(token)
        
        path = LeadPath(
            token=token,
            sequence=[e.event_type for e in events],
            timestamps=[e.timestamp for e in events],
            sources=[e.source for e in events],
            total_lead_time=events[-1].timestamp - first_time,
            outcome=outcome
        )
        
        self.completed_paths.append(path)
        if len(self.completed_paths) > 10000:
            self.completed_paths = self.completed_paths[-5000:]

    async def _get_token_outcome(self, token: str) -> Dict[str, Any]:
        return {
            "max_multiple": 0,
            "rugged": False,
            "migrated": False,
            "time_to_peak": 0
        }

    async def _update_source_rankings(self):
        type_performance = defaultdict(lambda: defaultdict(list))
        
        for path in self.completed_paths:
            if not path.outcome.get("rugged", False) and path.outcome.get("max_multiple", 0) >= 2:
                for i, (event_type, source) in enumerate(zip(path.sequence, path.sources)):
                    lead_time = path.timestamps[i] - path.timestamps[0]
                    type_performance[event_type][source].append({
                        "lead_time": lead_time,
                        "multiple": path.outcome.get("max_multiple", 1)
                    })
        
        for event_type, sources in type_performance.items():
            for source, performances in sources.items():
                if len(performances) >= 3:
                    avg_lead = np.mean([p["lead_time"] for p in performances])
                    avg_mult = np.mean([p["multiple"] for p in performances])
                    consistency = 1 - (np.std([p["lead_time"] for p in performances]) / max(avg_lead, 1))
                    
                    score = (1 / max(avg_lead / 60, 0.5)) * 0.4 + min(avg_mult / 10, 1) * 0.4 + consistency * 0.2
                    self.source_rankings[source][event_type] = max(0, min(1, score))

    async def _mine_path_patterns(self):
        for path in self.completed_paths:
            pattern_key = "->".join([e.value for e in path.sequence])
            self.path_patterns[pattern_key] += 1

    def get_lead_sequence(self, token: str) -> List[LeadEvent]:
        return sorted(self.lead_events.get(token, []), key=lambda x: x.timestamp)

    def get_source_ranking(self, event_type: LeadEventType, limit: int = 10) -> List[Tuple[str, float]]:
        rankings = []
        for source, types in self.source_rankings.items():
            if event_type in types:
                rankings.append((source, types[event_type]))
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings[:limit]

    def get_common_patterns(self, min_count: int = 3) -> List[Tuple[str, int]]:
        patterns = [(p, c) for p, c in self.path_patterns.items() if c >= min_count]
        patterns.sort(key=lambda x: x[1], reverse=True)
        return patterns[:20]

    def predict_next_event(self, token: str) -> List[Tuple[LeadEventType, float]]:
        events = self.get_lead_sequence(token)
        if not events:
            return []
        
        current_seq = tuple(e.event_type for e in events)
        
        next_probs = defaultdict(float)
        for path in self.completed_paths:
            path_seq = tuple(path.sequence)
            for i in range(len(path_seq) - len(current_seq) + 1):
                if path_seq[i:i+len(current_seq)] == current_seq and i + len(current_seq) < len(path_seq):
                    next_event = path_seq[i + len(current_seq)]
                    next_probs[next_event] += 1
        
        total = sum(next_probs.values())
        if total == 0:
            return []
        
        return [(evt, count/total) for evt, count in sorted(next_probs.items(), key=lambda x: x[1], reverse=True)]

    def get_stats(self) -> Dict:
        return {
            "active_tokens": len(self.lead_events),
            "completed_paths": len(self.completed_paths),
            "ranked_sources": len(self.source_rankings),
            "known_patterns": len(self.path_patterns),
            "top_patterns": self.get_common_patterns()[:10]
        }


class CounterfactualExecutionLab:
    def __init__(self):
        self.counterfactuals: List[Dict] = []
        self.execution_policies: Dict[str, Dict] = {}

    def record_decision(self, token: str, signal_snapshot: Dict, decision: Dict):
        pass

    def record_execution(self, token: str, execution_result: Dict):
        pass

    def add_counterfactual(self, token: str, policy_name: str, 
                           actual_result: Dict, counterfactual_result: Dict):
        record = {
            "token": token,
            "timestamp": time.time(),
            "policy": policy_name,
            "actual": actual_result,
            "counterfactual": counterfactual_result,
            "delta_pnl": counterfactual_result.get("pnl", 0) - actual_result.get("pnl", 0),
            "delta_slippage": counterfactual_result.get("slippage", 0) - actual_result.get("slippage", 0),
            "delta_landed": counterfactual_result.get("landed", False) != actual_result.get("landed", False)
        }
        self.counterfactuals.append(record)
        if len(self.counterfactuals) > 50000:
            self.counterfactuals = self.counterfactuals[-25000:]

    def evaluate_policies(self) -> Dict[str, Dict]:
        policy_stats = defaultdict(lambda: {"count": 0, "delta_pnl": 0, "win_rate": 0, "landed_improvement": 0})
        
        for cf in self.counterfactuals:
            policy = cf["policy"]
            stats = policy_stats[policy]
            stats["count"] += 1
            stats["delta_pnl"] += cf["delta_pnl"]
            if cf["delta_pnl"] > 0:
                stats["win_rate"] += 1
            if cf["delta_landed"]:
                stats["landed_improvement"] += 1
        
        for policy, stats in policy_stats.items():
            if stats["count"] > 0:
                stats["avg_delta_pnl"] = stats["delta_pnl"] / stats["count"]
                stats["win_rate"] = stats["win_rate"] / stats["count"]
                stats["landed_improvement_rate"] = stats["landed_improvement"] / stats["count"]
        
        return dict(policy_stats)

    def get_best_policy(self, base_policy: str) -> Optional[str]:
        stats = self.evaluate_policies()
        base_stats = stats.get(base_policy, {})
        
        best = None
        best_improvement = 0
        for policy, s in stats.items():
            if policy != base_policy and s["count"] >= 20:
                improvement = s.get("avg_delta_pnl", 0) - base_stats.get("avg_delta_pnl", 0)
                if improvement > best_improvement and s["win_rate"] > 0.5:
                    best_improvement = improvement
                    best = policy
        
        return best


class AdversarialAdaptationDetector:
    def __init__(self):
        self.feature_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.feature_weights: Dict[str, float] = {}
        self.fakeability_scores: Dict[str, float] = {}
        self.adaptation_alerts: List[Dict] = []

    def set_fakeability(self, feature: str, score: float):
        self.fakeability_scores[feature] = max(0, min(1, score))

    def record_feature_value(self, token: str, feature: str, value: float, outcome: Dict):
        self.feature_history[feature].append({
            "token": token,
            "value": value,
            "outcome_multiple": outcome.get("max_multiple", 1),
            "rugged": outcome.get("rugged", False),
            "timestamp": time.time()
        })

    def detect_adaptation(self, feature: str, window: int = 1000) -> Dict:
        history = list(self.feature_history[feature])
        if len(history) < window:
            return {"detected": False, "reason": "insufficient_data"}
        
        recent = history[-window:]
        older = history[-window*2:-window] if len(history) >= window*2 else history[:-window]
        
        recent_values = [h["value"] for h in recent]
        older_values = [h["value"] for h in older]
        
        recent_outcomes = [h["outcome_multiple"] for h in recent]
        older_outcomes = [h["outcome_multiple"] for h in older]
        
        recent_rugs = sum(1 for h in recent if h["rugged"]) / len(recent)
        older_rugs = sum(1 for h in older if h["rugged"]) / len(older)
        
        value_shift = abs(np.mean(recent_values) - np.mean(older_values)) / max(np.std(older_values), 0.001)
        outcome_degradation = np.mean(older_outcomes) - np.mean(recent_outcomes)
        rug_increase = recent_rugs - older_rugs
        
        fakeability = self.fakeability_scores.get(feature, 0.5)
        
        adaptation_score = (value_shift * 0.3 + 
                          max(0, outcome_degradation) * 0.4 + 
                          max(0, rug_increase) * 0.3) * (1 + fakeability)
        
        detected = adaptation_score > 1.5 and fakeability > 0.3
        
        if detected:
            alert = {
                "feature": feature,
                "adaptation_score": adaptation_score,
                "value_shift": value_shift,
                "outcome_degradation": outcome_degradation,
                "rug_increase": rug_increase,
                "fakeability": fakeability,
                "timestamp": time.time()
            }
            self.adaptation_alerts.append(alert)
            if len(self.adaptation_alerts) > 1000:
                self.adaptation_alerts = self.adaptation_alerts[-500:]
        
        return {
            "detected": detected,
            "adaptation_score": adaptation_score,
            "value_shift": value_shift,
            "outcome_degradation": outcome_degradation,
            "rug_increase": rug_increase,
            "fakeability": fakeability,
            "recommendation": "reduce_weight" if detected else "maintain"
        }

    def get_adaptive_weight(self, feature: str, base_weight: float) -> float:
        if feature not in self.feature_weights:
            self.feature_weights[feature] = base_weight
        
        detection = self.detect_adaptation(feature)
        if detection["detected"]:
            reduction = min(0.8, detection["adaptation_score"] / 3)
            self.feature_weights[feature] *= (1 - reduction)
            logger.warning(f"Reduced weight for {feature} by {reduction:.1%} due to adversarial adaptation")
        
        return max(0.01, self.feature_weights[feature])

    def get_alerts(self, limit: int = 20) -> List[Dict]:
        return sorted(self.adaptation_alerts, key=lambda x: x["timestamp"], reverse=True)[:limit]

    def get_all_weights(self) -> Dict[str, float]:
        return dict(self.feature_weights)
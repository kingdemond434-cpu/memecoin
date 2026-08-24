import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import json
import hashlib
import numpy as np

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    DISCOVERED = "discovered"
    SPEC_FROZEN = "spec_frozen"
    HISTORICAL_REPLAY = "historical_replay"
    STAGE_A_SURVIVOR = "stage_a_survivor"
    CHRONOLOGICAL_OOS = "chronological_oos"
    FORWARD_SHADOW = "forward_shadow"
    EVIDENCE_READY = "evidence_ready"
    CANARY = "canary"
    LIVE = "live"
    REJECTED = "rejected"
    HIBERNATED = "hibernated"
    DECAYING = "decaying"
    RETIRED = "retired"
    DATA_BLOCKED = "data_blocked"


@dataclass
class HypothesisSpec:
    hypothesis_id: str
    mechanism: str
    target: str
    features: List[str]
    feature_hash: str
    model_type: str
    model_params: Dict[str, Any]
    training_window: str
    threshold: float
    sizing_rule: Dict[str, Any]
    exit_rule: Dict[str, Any]
    execution_policy: Dict[str, Any]
    fakeability: Dict[str, float]
    cost_model: Dict[str, float]
    falsifier: str
    kill_thesis: str
    source_provenance: str
    trial_family: str
    created_at: float
    frozen_at: Optional[float] = None
    hash: str = ""
    
    def __post_init__(self):
        if not self.hash:
            content = json.dumps({
                "mechanism": self.mechanism,
                "target": self.target,
                "features": self.features,
                "model_type": self.model_type,
                "model_params": self.model_params,
                "threshold": self.threshold
            }, sort_keys=True)
            self.hash = hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class TrialResult:
    hypothesis_id: str
    stage: str
    samples: int
    metrics: Dict[str, float]
    oos_metrics: Dict[str, float]
    portfolio_impact: float
    passed: bool
    timestamp: float
    notes: str = ""


@dataclass
class ChampionModel:
    hypothesis: HypothesisSpec
    promoted_at: float
    forward_performance: deque = field(default_factory=lambda: deque(maxlen=1000))
    decay_score: float = 0.0
    last_updated: float = field(default_factory=time.time)
    status: str = "LIVE"


class ChampionChallengerFramework:
    def __init__(
        self,
        min_oos_samples: int = 50,
        min_portfolio_impact: float = 0.0001,
        shadow_duration_hours: int = 168,
        canary_duration_hours: int = 72,
        decay_window: int = 500,
        decay_threshold: float = -0.001
    ):
        self.min_oos_samples = min_oos_samples
        self.min_portfolio_impact = min_portfolio_impact
        self.shadow_duration = shadow_duration_hours * 3600
        self.canary_duration = canary_duration_hours * 3600
        self.decay_window = decay_window
        self.decay_threshold = decay_threshold
        
        self.hypotheses: Dict[str, HypothesisSpec] = {}
        self.trial_results: List[TrialResult] = []
        self.champions: Dict[str, ChampionModel] = {}
        self.challengers: Dict[str, Dict] = {}
        self.shadow_models: Dict[str, Dict] = {}
        self.canary_models: Dict[str, Dict] = {}
        
        self._running = False
        self._evaluation_task: Optional[asyncio.Task] = None

    async def start(self):
        self._running = True
        self._evaluation_task = asyncio.create_task(self._evaluation_loop())

    async def stop(self):
        self._running = False
        if self._evaluation_task:
            self._evaluation_task.cancel()

    def submit_hypothesis(self, hypothesis: HypothesisSpec) -> str:
        if hypothesis.hypothesis_id in self.hypotheses:
            return "duplicate"
        
        self.hypotheses[hypothesis.hypothesis_id] = hypothesis
        self.challengers[hypothesis.hypothesis_id] = {
            "hypothesis": hypothesis,
            "submitted_at": time.time(),
            "status": ModelStatus.DISCOVERED.value,
        }
        return "accepted"

    def mark_data_blocked(self, hypothesis_id: str, reason: str):
        if hypothesis_id not in self.hypotheses:
            return
        challenger = self.challengers.setdefault(hypothesis_id, {"hypothesis": self.hypotheses[hypothesis_id]})
        challenger["status"] = ModelStatus.DATA_BLOCKED.value
        challenger["reason"] = reason

    def is_live(self, hypothesis_id: str) -> bool:
        champion = self.champions.get(hypothesis_id)
        return bool(champion and champion.status == "LIVE")

    def freeze_hypothesis(self, hypothesis_id: str) -> bool:
        if hypothesis_id not in self.hypotheses:
            return False
        
        hyp = self.hypotheses[hypothesis_id]
        hyp.frozen_at = time.time()
        return True

    def record_trial_result(self, result: TrialResult):
        self.trial_results.append(result)
        if len(self.trial_results) > 10000:
            self.trial_results = self.trial_results[-5000:]

    async def _evaluation_loop(self):
        while self._running:
            try:
                await self._evaluate_challengers()
                await self._evaluate_shadow_models()
                await self._evaluate_canary_models()
                await self._monitor_champion_decay()
            except Exception as e:
                logger.error(f"Champion/challenger evaluation error: {e}")
            await asyncio.sleep(300)

    async def _evaluate_challengers(self):
        for hyp_id, challenger in list(self.challengers.items()):
            if hyp_id not in self.hypotheses:
                continue
            
            hyp = self.hypotheses[hyp_id]
            results = [r for r in self.trial_results if r.hypothesis_id == hyp_id]
            
            if not results:
                continue
            
            latest = results[-1]
            
            if latest.stage == "CHRONOLOGICAL_OOS":
                if (latest.samples >= self.min_oos_samples and
                    latest.oos_metrics.get("elogw", 0) > self.min_portfolio_impact and
                    latest.passed):
                    
                    self.shadow_models[hyp_id] = {
                        "hypothesis": hyp,
                        "started_at": time.time(),
                        "forward_results": deque(maxlen=1000),
                        "oos_metrics": latest.oos_metrics
                    }
                    del self.challengers[hyp_id]
                    logger.info(f"Promoted {hyp_id} to SHADOW")

    async def _evaluate_shadow_models(self):
        now = time.time()
        for hyp_id, shadow in list(self.shadow_models.items()):
            if now - shadow["started_at"] >= self.shadow_duration:
                forward_results = list(shadow["forward_results"])
                if len(forward_results) >= self.min_oos_samples:
                    avg_elogw = np.mean([r.get("elogw", 0) for r in forward_results])
                    win_rate = sum(1 for r in forward_results if r.get("pnl", 0) > 0) / len(forward_results)
                    
                    if avg_elogw > self.min_portfolio_impact and win_rate > 0.35:
                        self.canary_models[hyp_id] = {
                            "hypothesis": shadow["hypothesis"],
                            "started_at": now,
                            "position_count": 0,
                            "total_pnl": 0.0,
                            "forward_results": deque(maxlen=1000)
                        }
                        logger.info(f"Promoted {hyp_id} to CANARY")
                    else:
                        self._retire_hypothesis(hyp_id, "shadow_failed")
                    
                    del self.shadow_models[hyp_id]

    async def _evaluate_canary_models(self):
        now = time.time()
        for hyp_id, canary in list(self.canary_models.items()):
            if now - canary["started_at"] >= self.canary_duration:
                if canary["position_count"] >= 20:
                    avg_pnl = canary["total_pnl"] / max(canary["position_count"], 1)
                    win_rate = sum(1 for r in canary["forward_results"] if r.get("pnl", 0) > 0) / max(len(canary["forward_results"]), 1)
                    
                    if avg_pnl > 0 and win_rate > 0.3:
                        champion = ChampionModel(
                            hypothesis=canary["hypothesis"],
                            promoted_at=now
                        )
                        self.champions[hyp_id] = champion
                        logger.info(f"Promoted {hyp_id} to LIVE CHAMPION")
                    else:
                        self._retire_hypothesis(hyp_id, "canary_failed")
                else:
                    self._retire_hypothesis(hyp_id, "insufficient_canary_trades")
                
                del self.canary_models[hyp_id]

    def _retire_hypothesis(self, hyp_id: str, reason: str):
        if hyp_id in self.hypotheses:
            self.hypotheses[hyp_id].status = ModelStatus.RETIRED
            logger.info(f"Retired {hyp_id}: {reason}")

    async def _monitor_champion_decay(self):
        for hyp_id, champion in list(self.champions.items()):
            if len(champion.forward_performance) >= self.decay_window:
                recent = list(champion.forward_performance)[-self.decay_window:]
                avg_elogw = np.mean([r.get("elogw", 0) for r in recent])
                win_rate = sum(1 for r in recent if r.get("pnl", 0) > 0) / len(recent)
                
                champion.decay_score = avg_elogw
                champion.last_updated = time.time()
                
                if avg_elogw < self.decay_threshold or win_rate < 0.25:
                    champion.status = "DECAYING"
                    logger.warning(f"Champion {hyp_id} decaying: elogw={avg_elogw:.6f}, wr={win_rate:.2f}")
                
                if avg_elogw < self.decay_threshold * 2 and win_rate < 0.2:
                    champion.status = "HIBERNATED"
                    logger.warning(f"Champion {hyp_id} hibernated")

    def record_forward_result(self, hyp_id: str, result: Dict):
        if hyp_id in self.champions:
            self.champions[hyp_id].forward_performance.append(result)
        elif hyp_id in self.shadow_models:
            self.shadow_models[hyp_id]["forward_results"].append(result)
        elif hyp_id in self.canary_models:
            canary = self.canary_models[hyp_id]
            canary["forward_results"].append(result)
            if "pnl" in result:
                canary["position_count"] += 1
                canary["total_pnl"] += result["pnl"]

    def get_live_champions(self) -> List[ChampionModel]:
        return [c for c in self.champions.values() if c.status == "LIVE"]

    def get_champion_for_target(self, target: str) -> Optional[ChampionModel]:
        for champ in self.champions.values():
            if champ.hypothesis.target == target and champ.status == "LIVE":
                return champ
        return None

    def get_stats(self) -> Dict:
        return {
            "total_hypotheses": len(self.hypotheses),
            "challengers": len(self.challengers),
            "data_blocked": len([c for c in self.challengers.values() if c.get("status") == ModelStatus.DATA_BLOCKED.value]),
            "shadow_models": len(self.shadow_models),
            "canary_models": len(self.canary_models),
            "live_champions": len(self.get_live_champions()),
            "decaying_champions": len([c for c in self.champions.values() if c.status == "DECAYING"]),
            "hibernated_champions": len([c for c in self.champions.values() if c.status == "HIBERNATED"])
        }


class HypothesisRegistry:
    def __init__(self):
        self.hypotheses: Dict[str, HypothesisSpec] = {}
        self.families: Dict[str, List[str]] = defaultdict(list)
        self.graveyard: List[Dict] = []

    def register(self, hypothesis: HypothesisSpec):
        self.hypotheses[hypothesis.hypothesis_id] = hypothesis
        self.families[hypothesis.trial_family].append(hypothesis.hypothesis_id)

    def get(self, hypothesis_id: str) -> Optional[HypothesisSpec]:
        return self.hypotheses.get(hypothesis_id)

    def get_family(self, family: str) -> List[HypothesisSpec]:
        return [self.hypotheses[hid] for hid in self.families.get(family, []) if hid in self.hypotheses]

    def graveyard_entry(self, hypothesis_id: str, reason: str, metrics: Dict):
        if hypothesis_id in self.hypotheses:
            entry = {
                "hypothesis_id": hypothesis_id,
                "spec": self.hypotheses[hypothesis_id].__dict__,
                "reason": reason,
                "final_metrics": metrics,
                "timestamp": time.time()
            }
            self.graveyard.append(entry)
            del self.hypotheses[hypothesis_id]

    def find_similar(self, features: List[str], threshold: float = 0.8) -> List[HypothesisSpec]:
        similar = []
        feature_set = set(features)
        for hyp in self.hypotheses.values():
            overlap = len(feature_set & set(hyp.features)) / max(len(feature_set | set(hyp.features)), 1)
            if overlap >= threshold:
                similar.append(hyp)
        return similar


class TrialLedger:
    def __init__(self):
        self.trials: List[Dict] = []
        self.multiplicity_groups: Dict[str, List[str]] = defaultdict(list)

    def record_trial(self, trial: Dict):
        self.trials.append(trial)
        if len(self.trials) > 50000:
            self.trials = self.trials[-25000:]

    def record_multiplicity_group(self, group_id: str, hypothesis_ids: List[str]):
        self.multiplicity_groups[group_id].extend(hypothesis_ids)

    def get_recent_trials(self, hours: int = 24) -> List[Dict]:
        cutoff = time.time() - hours * 3600
        return [t for t in self.trials if t.get("timestamp", 0) > cutoff]

    def get_family_stats(self, family: str) -> Dict:
        trials = [t for t in self.trials if t.get("family") == family]
        if not trials:
            return {"count": 0}
        
        passed = sum(1 for t in trials if t.get("passed", False))
        return {
            "total": len(trials),
            "passed": passed,
            "pass_rate": passed / len(trials),
            "avg_elogw": np.mean([t.get("elogw", 0) for t in trials]) if trials else 0
        }

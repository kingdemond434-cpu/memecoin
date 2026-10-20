import asyncio
import dataclasses
import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import json
import hashlib
from pathlib import Path
import numpy as np
from src.research.research_contracts import canonical_candidate_identity, validate_exact_candidate

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
    status: str = ModelStatus.DISCOVERED.value
    schema_version: int = 2
    parent_ids: List[str] = field(default_factory=list)
    agent_identity: str = "legacy_unattributed"
    exact_candidate: Optional[Dict[str, Any]] = None
    frozen_hash: str = ""
    admission_status: str = "LEGACY_NONEXACT"

    def __post_init__(self):
        if not self.hash:
            content = json.dumps({
                "mechanism": self.mechanism,
                "target": self.target,
                "features": self.features,
                "feature_hash": self.feature_hash,
                "model_type": self.model_type,
                "model_params": self.model_params,
                "training_window": self.training_window,
                "threshold": self.threshold,
                "sizing_rule": self.sizing_rule,
                "exit_rule": self.exit_rule,
                "execution_policy": self.execution_policy,
                "cost_model": self.cost_model,
                "falsifier": self.falsifier,
                "kill_thesis": self.kill_thesis,
            }, sort_keys=True)
            self.hash = hashlib.sha256(content.encode()).hexdigest()[:16]

    def content_hash(self) -> str:
        body = dataclasses.asdict(self)
        for key in ("hash", "frozen_hash", "frozen_at", "status", "admission_status"):
            body.pop(key, None)
        return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode()).hexdigest()


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
    spec_hash: str = ""
    evidence_status: str = "LEGACY_UNBOUND"


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
        decay_threshold: float = -0.001,
        state_path: Optional[str] = None,
    ):
        self.min_oos_samples = min_oos_samples
        self.min_portfolio_impact = min_portfolio_impact
        self.shadow_duration = shadow_duration_hours * 3600
        self.canary_duration = canary_duration_hours * 3600
        self.decay_window = decay_window
        self.decay_threshold = decay_threshold
        self.state_path = Path(state_path) if state_path else None
        
        # The registry and the trial ledger. Both classes existed, neither
        # was ever constructed, so every one of their methods was dead: no
        # hypothesis was registered, no trial recorded, no multiplicity group
        # declared. That matters more than it sounds -- the gauntlet's CSCV
        # pass prices SELECTION, and selection is how many candidates were
        # tried for the one that won. With nothing counting the candidates,
        # the probability of backtest overfitting was being computed over a
        # population nobody had enumerated.
        self.registry = HypothesisRegistry()
        self.trial_ledger = TrialLedger()
        self.hypotheses: Dict[str, HypothesisSpec] = {}
        self.trial_results: List[TrialResult] = []
        self.champions: Dict[str, ChampionModel] = {}
        self.challengers: Dict[str, Dict] = {}
        self.shadow_models: Dict[str, Dict] = {}
        self.canary_models: Dict[str, Dict] = {}
        self.frozen_specs: Dict[str, Dict[str, Any]] = {}
        self.research_components: Dict[str, Dict[str, Any]] = {}
        
        self._running = False
        self._evaluation_task: Optional[asyncio.Task] = None
        self._load_state()

    async def start(self):
        self._running = True
        self._evaluation_task = asyncio.create_task(self._evaluation_loop())

    async def stop(self):
        self._running = False
        if self._evaluation_task:
            self._evaluation_task.cancel()
            try:
                await self._evaluation_task
            except asyncio.CancelledError:
                pass
        self._save_state()

    def submit_hypothesis(self, hypothesis: HypothesisSpec) -> str:
        hypothesis.admission_status = (
            validate_exact_candidate(hypothesis.exact_candidate).status
            if hypothesis.exact_candidate is not None else "LEGACY_NONEXACT")
        if hypothesis.hypothesis_id in self.hypotheses:
            existing = self.hypotheses[hypothesis.hypothesis_id]
            if existing.content_hash() == hypothesis.content_hash():
                return "duplicate"
            # Never inherit promotion evidence across a feature/model schema change.
            self.challengers.pop(hypothesis.hypothesis_id, None)
            self.shadow_models.pop(hypothesis.hypothesis_id, None)
            self.canary_models.pop(hypothesis.hypothesis_id, None)
            self.champions.pop(hypothesis.hypothesis_id, None)
            # Retain the old version and every outcome-inspecting trial. Only
            # evidence bound to the new frozen hash may be consumed below.
            self.registry.graveyard_entry(hypothesis.hypothesis_id, "superseded",
                                          self.trial_ledger.get_family_stats(existing.trial_family))
            hypothesis.frozen_at = None
            hypothesis.frozen_hash = ""
        
        # Registered before it is a challenger, and grouped with whatever it
        # resembles. Two hypotheses over the same features tested on the same
        # data are one trial repeated, and the multiplicity group is what says
        # so to anything pricing selection later.
        self.registry.register(hypothesis)
        siblings = [
            other.hypothesis_id
            for other in self.registry.find_similar(hypothesis.features)
            if other.hypothesis_id != hypothesis.hypothesis_id]
        if siblings:
            self.trial_ledger.record_multiplicity_group(
                hypothesis.trial_family, siblings + [hypothesis.hypothesis_id])
        self.hypotheses[hypothesis.hypothesis_id] = hypothesis
        self.challengers[hypothesis.hypothesis_id] = {
            "hypothesis": hypothesis,
            "submitted_at": time.time(),
            "status": ModelStatus.DISCOVERED.value,
            "novelty": self.candidate_niche(hypothesis.hypothesis_id),
        }
        self._save_state()
        return "accepted"

    def mark_data_blocked(self, hypothesis_id: str, reason: str):
        if hypothesis_id not in self.hypotheses:
            return
        champion = self.champions.get(hypothesis_id)
        if champion:
            champion.status = "HIBERNATED"
        self.shadow_models.pop(hypothesis_id, None)
        self.canary_models.pop(hypothesis_id, None)
        challenger = self.challengers.setdefault(hypothesis_id, {"hypothesis": self.hypotheses[hypothesis_id]})
        challenger["status"] = ModelStatus.DATA_BLOCKED.value
        challenger["reason"] = reason
        self.hypotheses[hypothesis_id].status = ModelStatus.DATA_BLOCKED.value
        self._save_state()

    def candidate_niche(self, hypothesis_id: str) -> Dict[str, Any]:
        hypothesis = self.hypotheses.get(hypothesis_id)
        if hypothesis is None or hypothesis.exact_candidate is None:
            return {"status": "LEGACY_NONEXACT", "candidate_id": hypothesis_id}
        try:
            signatures = canonical_candidate_identity(hypothesis.exact_candidate)
        except ValueError:
            return {"status": "BLOCKED", "candidate_id": hypothesis_id}
        program = self.registry.program_niches.get(signatures["program_signature"], {})
        family = self.registry.parameter_niches.get(signatures["parameter_family_signature"], {})
        program_representative = program.get("representative")
        family_representative = family.get("representative")
        version = {"hypothesis_id": hypothesis_id, "spec_hash": hypothesis.content_hash()}
        program_first = next(iter(program.get("versions", [])), None)
        family_first = next(iter(family.get("versions", [])), None)
        disposition = ("DUPLICATE" if program_first != version
                       else "PARAMETER_NEIGHBOUR" if family_first != version else "NOVEL_PROGRAM")
        return {"status": "INDEXED", "candidate_id": hypothesis_id, **signatures,
                "disposition": disposition, "representative": family_representative,
                "validation_representative": program_representative,
                "members": list(family.get("members", [])),
                "program_members": list(program.get("members", [])),
                "contributes_independent_breadth": family_first == version,
                "validation_job_family": signatures["parameter_family_signature"]}

    def is_live(self, hypothesis_id: str) -> bool:
        champion = self.champions.get(hypothesis_id)
        return bool(champion and champion.status == "LIVE" and self._exact_frozen_valid(champion.hypothesis))

    def freeze_hypothesis(self, hypothesis_id: str) -> bool:
        if hypothesis_id not in self.hypotheses:
            return False
        
        hyp = self.hypotheses[hypothesis_id]
        if hyp.frozen_hash:
            return self._frozen_valid(hyp)
        if hyp.exact_candidate is not None:
            admission = validate_exact_candidate(hyp.exact_candidate)
            hyp.admission_status = admission.status
            if not admission.ok:
                self.mark_data_blocked(hypothesis_id, "; ".join(admission.reasons))
                return False
            if (hyp.exact_candidate["candidate_id"] != hyp.hypothesis_id
                    or hyp.exact_candidate["trial_family"] != hyp.trial_family
                    or hyp.exact_candidate["parent_ids"] != hyp.parent_ids):
                self.mark_data_blocked(hypothesis_id, "candidate identity/genealogy mismatch")
                return False
        hyp.frozen_at = time.time()
        hyp.frozen_hash = hyp.content_hash()
        hyp.hash = hyp.frozen_hash
        self.frozen_specs[hyp.frozen_hash] = dataclasses.asdict(hyp)
        self._save_state()
        return True

    @staticmethod
    def _frozen_valid(hypothesis: HypothesisSpec) -> bool:
        try:
            return bool(hypothesis.frozen_at and hypothesis.frozen_hash
                        and hypothesis.frozen_hash == hypothesis.content_hash()
                        and (hypothesis.exact_candidate is None or
                             validate_exact_candidate(hypothesis.exact_candidate).ok))
        except (ValueError, TypeError):
            return False

    @classmethod
    def _exact_frozen_valid(cls, hypothesis: HypothesisSpec) -> bool:
        return bool(hypothesis.exact_candidate is not None
                    and hypothesis.admission_status == "EXACT_CANDIDATE"
                    and cls._frozen_valid(hypothesis))

    def record_trial_result(self, result: TrialResult):
        hypothesis = self.hypotheses.get(result.hypothesis_id)
        # Missing legacy bindings remain visible research history. A changed
        # frozen program can never inherit a successful trial's authority.
        if hypothesis and hypothesis.frozen_hash:
            if (not self._frozen_valid(hypothesis)
                    or (result.spec_hash and result.spec_hash != hypothesis.frozen_hash)
                    or result.timestamp < hypothesis.frozen_at):
                result.evidence_status = "REJECTED_BINDING"
            elif hypothesis.admission_status == "EXACT_CANDIDATE" and not result.spec_hash:
                result.evidence_status = "LEGACY_UNBOUND"
            else:
                result.spec_hash = hypothesis.frozen_hash
                result.evidence_status = ("BOUND" if hypothesis.admission_status == "EXACT_CANDIDATE"
                                          else "LEGACY_NONEXACT")
        self.trial_ledger.record_trial({
            "hypothesis_id": result.hypothesis_id,
            "family": (hypothesis.trial_family if hypothesis else ""),
            "stage": result.stage, "samples": result.samples,
            "passed": bool(result.passed and result.evidence_status != "REJECTED_BINDING"),
            "elogw": float(result.metrics.get("elogw", 0.0) or 0.0),
            "portfolio_impact": float(result.portfolio_impact),
            "timestamp": result.timestamp, "spec_hash": result.spec_hash,
            "evidence_status": result.evidence_status})
        self.trial_results.append(result)
        self._save_state()
        return result.evidence_status == "BOUND"

    async def _evaluation_loop(self):
        while self._running:
            try:
                await self._evaluate_challengers()
                await self._evaluate_shadow_models()
                await self._evaluate_canary_models()
                await self._monitor_champion_decay()
                self._save_state()
            except Exception as e:
                logger.error(f"Champion/challenger evaluation error: {e}")
            await asyncio.sleep(300)

    async def _evaluate_challengers(self):
        for hyp_id, challenger in list(self.challengers.items()):
            if hyp_id not in self.hypotheses:
                continue
            
            hyp = self.hypotheses[hyp_id]
            if not self._exact_frozen_valid(hyp):
                self.mark_data_blocked(hyp_id, "exact frozen candidate required before promotion")
                continue
            results = [r for r in self.trial_results if r.hypothesis_id == hyp_id
                       and r.evidence_status == "BOUND" and r.spec_hash == hyp.frozen_hash]
            
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
                    hyp.status = ModelStatus.FORWARD_SHADOW.value
                    del self.challengers[hyp_id]
                    logger.info(f"Promoted {hyp_id} to SHADOW")

    async def _evaluate_shadow_models(self):
        now = time.time()
        for hyp_id, shadow in list(self.shadow_models.items()):
            if not self._exact_frozen_valid(shadow["hypothesis"]):
                self.mark_data_blocked(hyp_id, "frozen candidate content changed")
                continue
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
                        shadow["hypothesis"].status = ModelStatus.CANARY.value
                        logger.info(f"Promoted {hyp_id} to CANARY")
                    else:
                        self._retire_hypothesis(hyp_id, "shadow_failed")
                    
                    del self.shadow_models[hyp_id]

    async def _evaluate_canary_models(self):
        now = time.time()
        for hyp_id, canary in list(self.canary_models.items()):
            if not self._exact_frozen_valid(canary["hypothesis"]):
                self.mark_data_blocked(hyp_id, "frozen candidate content changed")
                continue
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
                        canary["hypothesis"].status = ModelStatus.LIVE.value
                        logger.info(f"Promoted {hyp_id} to LIVE CHAMPION")
                    else:
                        self._retire_hypothesis(hyp_id, "canary_failed")
                else:
                    self._retire_hypothesis(hyp_id, "insufficient_canary_trades")
                
                del self.canary_models[hyp_id]

    def _retire_hypothesis(self, hyp_id: str, reason: str):
        if hyp_id in self.hypotheses:
            self.hypotheses[hyp_id].status = ModelStatus.RETIRED.value
            # What was tried and failed is evidence too. A graveyard nobody
            # writes to means the same idea gets rediscovered, retested and
            # recounted as a fresh trial every time.
            self.registry.graveyard_entry(
                hyp_id, reason, self.trial_ledger.get_family_stats(
                    self.hypotheses[hyp_id].trial_family))
            logger.info(f"Retired {hyp_id}: {reason}")
            self._save_state()

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
        hypothesis = self.hypotheses.get(hyp_id)
        if hypothesis is None or not self._exact_frozen_valid(hypothesis):
            return False
        if result.get("spec_hash") != hypothesis.frozen_hash:
            return False
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
        self._save_state()
        return True

    @staticmethod
    def _json_value(value: Any) -> Any:
        if dataclasses.is_dataclass(value):
            return ChampionChallengerFramework._json_value(dataclasses.asdict(value))
        if isinstance(value, deque):
            return [ChampionChallengerFramework._json_value(item) for item in value]
        if isinstance(value, dict):
            return {str(key): ChampionChallengerFramework._json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [ChampionChallengerFramework._json_value(item) for item in value]
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _bucket_state(self, bucket: Dict[str, Dict]) -> Dict[str, Dict]:
        packed = {}
        for hyp_id, entry in bucket.items():
            packed[hyp_id] = {
                key: self._json_value(value)
                for key, value in entry.items()
                if key != "hypothesis"
            }
        return packed

    @staticmethod
    def _research_checkpoint(state: Any) -> Dict[str, Any]:
        if not isinstance(state, dict):
            raise ValueError("research checkpoint must be an object")
        forbidden = {"authority", "live_authority", "allow_live_trading", "live_lock",
                     "operator_ack", "private_key", "signer", "submit_transaction",
                     "allocate_live_capital", "disable_veto", "lower_gates"}
        def inspect(value: Any) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    if not isinstance(key, str) or key.lower() in forbidden:
                        raise ValueError("research checkpoints cannot alter authority")
                    inspect(item)
            elif isinstance(value, list):
                for item in value:
                    inspect(item)
        inspect(state)
        return json.loads(json.dumps(state, sort_keys=True, allow_nan=False))

    def get_research_component(self, name: str) -> Optional[Dict[str, Any]]:
        value = self.research_components.get(name)
        return self._research_checkpoint(value) if value is not None else None

    def checkpoint_research_component(self, name: str, state: Dict[str, Any]) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("named research component required")
        self.research_components[name] = self._research_checkpoint(state)
        self._save_state()

    def _save_state(self):
        if not self.state_path:
            return
        state = {
            "schema_version": 2,
            "hypotheses": {key: self._json_value(value) for key, value in self.hypotheses.items()},
            "trial_results": self._json_value(self.trial_results),
            "frozen_specs": self.frozen_specs,
            "research_components": self.research_components,
            "challengers": self._bucket_state(self.challengers),
            "shadow_models": self._bucket_state(self.shadow_models),
            "canary_models": self._bucket_state(self.canary_models),
            "registry": {
                "active_ids": sorted(self.registry.hypotheses),
                "families": dict(self.registry.families),
                "graveyard": self._json_value(self.registry.graveyard),
                "genealogy": self.registry.genealogy,
                "program_niches": self.registry.program_niches,
                "parameter_niches": self.registry.parameter_niches,
            },
            "trial_ledger": {"trials": self._json_value(self.trial_ledger.trials),
                             "multiplicity_groups": dict(self.trial_ledger.multiplicity_groups)},
            "champions": {
                key: {
                    "promoted_at": value.promoted_at,
                    "forward_performance": self._json_value(value.forward_performance),
                    "decay_score": value.decay_score,
                    "last_updated": value.last_updated,
                    "status": value.status,
                }
                for key, value in self.champions.items()
            },
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.state_path)

    def _load_state(self):
        if not self.state_path or not self.state_path.exists():
            return
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("schema_version") not in (1, 2):
                raise ValueError("unsupported champion state schema")
            self.hypotheses = {
                key: HypothesisSpec(**value) for key, value in state.get("hypotheses", {}).items()
            }
            self.trial_results = [TrialResult(**value) for value in state.get("trial_results", [])]
            self.frozen_specs = state.get("frozen_specs", {})
            self.research_components = {
                name: self._research_checkpoint(value)
                for name, value in state.get("research_components", {}).items()}
            registry = state.get("registry", {})
            active = set(registry.get("active_ids", self.hypotheses))
            for hyp_id, hypothesis in self.hypotheses.items():
                if hyp_id in active and hypothesis.status != ModelStatus.RETIRED.value:
                    self.registry.register(hypothesis)
            self.registry.families = defaultdict(list, registry.get("families", self.registry.families))
            self.registry.graveyard = registry.get("graveyard", [])
            self.registry.genealogy = registry.get("genealogy", self.registry.genealogy)
            if "program_niches" in registry and "parameter_niches" in registry:
                self.registry.program_niches = registry["program_niches"]
                self.registry.parameter_niches = registry["parameter_niches"]
            else:
                for raw in list(self.frozen_specs.values()) + [row["spec"] for row in self.registry.graveyard]:
                    self.registry.index_candidate(HypothesisSpec(**raw))
            ledger = state.get("trial_ledger")
            if ledger is not None:
                self.trial_ledger.trials = ledger.get("trials", [])
                self.trial_ledger.multiplicity_groups = defaultdict(list, ledger.get("multiplicity_groups", {}))
            else:
                # Migrate old state without inventing a frozen evidence seal.
                for result in self.trial_results:
                    hypothesis = self.hypotheses.get(result.hypothesis_id)
                    self.trial_ledger.record_trial({
                        **dataclasses.asdict(result),
                        "family": hypothesis.trial_family if hypothesis else "",
                        "elogw": result.metrics.get("elogw", 0.0),
                    })
                for hypothesis in self.registry.hypotheses.values():
                    siblings = [other.hypothesis_id for other in self.registry.find_similar(hypothesis.features)]
                    if len(siblings) > 1:
                        self.trial_ledger.record_multiplicity_group(hypothesis.trial_family, siblings)
            self.challengers = self._restore_bucket(state.get("challengers", {}))
            self.shadow_models = self._restore_bucket(state.get("shadow_models", {}), forward=True)
            self.canary_models = self._restore_bucket(state.get("canary_models", {}), forward=True)
            self.champions = {}
            for hyp_id, value in state.get("champions", {}).items():
                hypothesis = self.hypotheses.get(hyp_id)
                if not hypothesis:
                    continue
                self.champions[hyp_id] = ChampionModel(
                    hypothesis=hypothesis,
                    promoted_at=float(value["promoted_at"]),
                    forward_performance=deque(value.get("forward_performance", []), maxlen=1000),
                    decay_score=float(value.get("decay_score", 0.0)),
                    last_updated=float(value.get("last_updated", time.time())),
                    status=str(value.get("status", "LIVE")),
                )
            for hypothesis in self.hypotheses.values():
                if not self._frozen_valid(hypothesis):
                    if hypothesis.frozen_hash:
                        hypothesis.admission_status = "BLOCKED_TAMPER"
                    elif hypothesis.exact_candidate is None:
                        hypothesis.admission_status = "LEGACY_NONEXACT"
            for bucket in (self.shadow_models, self.canary_models):
                for hyp_id, entry in list(bucket.items()):
                    hypothesis = entry["hypothesis"]
                    if hypothesis.admission_status != "EXACT_CANDIDATE" or not self._frozen_valid(hypothesis):
                        del bucket[hyp_id]
                        self.challengers[hyp_id] = {"hypothesis": hypothesis,
                            "status": ModelStatus.DATA_BLOCKED.value,
                            "reason": "restored legacy or unbound evidence requires exact frozen reevaluation"}
                        hypothesis.status = ModelStatus.DATA_BLOCKED.value
            for champion in self.champions.values():
                if (champion.hypothesis.admission_status != "EXACT_CANDIDATE"
                        or not self._frozen_valid(champion.hypothesis)):
                    champion.status = "HIBERNATED"
        except Exception as exc:
            logger.error("Ignoring unreadable champion/challenger state %s: %s", self.state_path, exc)

    def _restore_bucket(self, raw: Dict[str, Dict], *, forward: bool = False) -> Dict[str, Dict]:
        restored = {}
        for hyp_id, value in raw.items():
            hypothesis = self.hypotheses.get(hyp_id)
            if not hypothesis:
                continue
            entry = dict(value)
            entry["hypothesis"] = hypothesis
            if forward:
                entry["forward_results"] = deque(entry.get("forward_results", []), maxlen=1000)
            restored[hyp_id] = entry
        return restored

    def get_live_champions(self) -> List[ChampionModel]:
        return [c for c in self.champions.values() if self.is_live(c.hypothesis.hypothesis_id)]

    def get_champion_for_target(self, target: str) -> Optional[ChampionModel]:
        for champ in self.champions.values():
            if champ.hypothesis.target == target and self.is_live(champ.hypothesis.hypothesis_id):
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
            "hibernated_champions": len([c for c in self.champions.values() if c.status == "HIBERNATED"]),
            "retired": len([h for h in self.hypotheses.values() if h.status == ModelStatus.RETIRED.value]),
            # How many candidates were tried, by family, and how many passed.
            # This is the denominator selection has to be priced against: a
            # family with one survivor out of forty is a different claim from
            # a family with one out of two.
            "trials_last_24h": len(self.trial_ledger.get_recent_trials(24)),
            "families": {
                family: self.trial_ledger.get_family_stats(family)
                for family in sorted({
                    h.trial_family for h in self.hypotheses.values()
                    if h.trial_family})},
            "family_members": {
                family: len(self.registry.get_family(family))
                for family in sorted({
                    h.trial_family for h in self.hypotheses.values()
                    if h.trial_family})},
            "multiplicity_groups": {
                group: len(set(members))
                for group, members in sorted(
                    self.trial_ledger.multiplicity_groups.items())},
            "raw_exact_variants": len({version["spec_hash"] for niche in self.registry.program_niches.values()
                                       for version in niche["versions"]}),
            "niche_scope": "all_research_history",
            "effective_executable_programs": len(self.registry.program_niches),
            "effective_independent_breadth": len(self.registry.parameter_niches),
            "candidate_niches": {hid: self.candidate_niche(hid) for hid in sorted(self.hypotheses)
                                 if self.hypotheses[hid].exact_candidate is not None},
            "live_champion_targets": sorted({
                champion.hypothesis.target
                for champion in self.get_live_champions()
                if self.get_champion_for_target(champion.hypothesis.target)}),
        }


class HypothesisRegistry:
    def __init__(self):
        self.hypotheses: Dict[str, HypothesisSpec] = {}
        self.families: Dict[str, List[str]] = defaultdict(list)
        self.graveyard: List[Dict] = []
        self.genealogy: Dict[str, List[str]] = {}
        self.program_niches: Dict[str, Dict[str, Any]] = {}
        self.parameter_niches: Dict[str, Dict[str, Any]] = {}

    def register(self, hypothesis: HypothesisSpec):
        self.hypotheses[hypothesis.hypothesis_id] = hypothesis
        if hypothesis.hypothesis_id not in self.families[hypothesis.trial_family]:
            self.families[hypothesis.trial_family].append(hypothesis.hypothesis_id)
        self.genealogy[hypothesis.hypothesis_id] = list(hypothesis.parent_ids)
        self.index_candidate(hypothesis)

    def index_candidate(self, hypothesis: HypothesisSpec) -> None:
        if hypothesis.exact_candidate is None:
            return
        try:
            signatures = canonical_candidate_identity(hypothesis.exact_candidate)
        except ValueError:
            return
        for index, key in ((self.program_niches, "program_signature"),
                           (self.parameter_niches, "parameter_family_signature")):
            niche = index.setdefault(signatures[key], {
                "representative": hypothesis.hypothesis_id, "members": [], "versions": []})
            if hypothesis.hypothesis_id not in niche["members"]:
                niche["members"].append(hypothesis.hypothesis_id)
            version = {"hypothesis_id": hypothesis.hypothesis_id,
                       "spec_hash": hypothesis.content_hash()}
            if version not in niche["versions"]:
                niche["versions"].append(version)

    def get(self, hypothesis_id: str) -> Optional[HypothesisSpec]:
        return self.hypotheses.get(hypothesis_id)

    def get_family(self, family: str) -> List[HypothesisSpec]:
        return [self.hypotheses[hid] for hid in self.families.get(family, []) if hid in self.hypotheses]

    def graveyard_entry(self, hypothesis_id: str, reason: str, metrics: Dict):
        if hypothesis_id in self.hypotheses:
            entry = {
                "hypothesis_id": hypothesis_id,
                "spec": dataclasses.asdict(self.hypotheses[hypothesis_id]),
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

    def record_multiplicity_group(self, group_id: str, hypothesis_ids: List[str]):
        self.multiplicity_groups[group_id] = sorted(set(
            self.multiplicity_groups[group_id]) | set(hypothesis_ids))

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

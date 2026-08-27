import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import json
import hashlib
import os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


class PredictionTarget(Enum):
    P_2X = "p_2x"
    P_5X = "p_5x"
    P_10X = "p_10x"
    P_20X = "p_20x"
    P_50X = "p_50x"
    # The survival curve used to end at 50x, which meant a 500x and an
    # ordinary 50x were the same number to every consumer -- the sizing, the
    # monster machine, the allocator. They are not the same event, and the
    # book is carried by the difference.
    P_100X = "p_100x"
    P_250X = "p_250x"
    P_500X = "p_500x"
    P_MIGRATION = "p_migration"
    P_RUG_30S = "p_rug_30s"
    P_RUG_5M = "p_rug_5m"
    EXPECTED_SLIPPAGE = "expected_slippage"
    EXPECTED_HOLD_TIME = "expected_hold_time"
    EXPECTED_FEASIBLE_MULTIPLE = "expected_feasible_multiple"


# The cumulative survival levels, in order. Everything that walks the curve --
# monotonicity, the disjoint bins, the labels -- reads this one tuple, so a new
# rung cannot be added to one and forgotten by the others.
SURVIVAL_LEVELS: Tuple[Tuple[PredictionTarget, float], ...] = (
    (PredictionTarget.P_2X, 2.0),
    (PredictionTarget.P_5X, 5.0),
    (PredictionTarget.P_10X, 10.0),
    (PredictionTarget.P_20X, 20.0),
    (PredictionTarget.P_50X, 50.0),
    (PredictionTarget.P_100X, 100.0),
    (PredictionTarget.P_250X, 250.0),
    (PredictionTarget.P_500X, 500.0),
)

# What a launch IS, at the age the decision is being made.
#
# A pooled model trained across every horizon learns the average launch, and
# there is no such thing. At 100ms almost nothing is observable except who
# funded whom and how fast the first buys arrived; by five minutes the holder
# distribution and the whole social response exist, and the features that
# mattered at 100ms have been overwritten by their own consequences. The same
# column means a different thing in each regime, so one model reconciling them
# is not learning a launch -- it is learning a blend of four.
#
# Bands are closed below and open above.
# One-hot regimes. Fixed and ordered so the feature schema hash is stable, and
# an unrecognised regime lights nothing rather than being folded into a
# neighbour it has nothing to do with.
REGIME_NAMES: Tuple[str, ...] = ("bull", "bear", "chop", "euphoria")


#: The age bands, and the number of models the desk maintains.
#:
#: Four, and not five. Splitting a band makes each sub-band's model see a more
#: homogeneous world AND halves the rows it is fitted on, and below a certain
#: amount of evidence the second effect wins outright -- two confidently wrong
#: models where there was one honestly uncertain one. The split will still
#: look justified, because any partition of a finite sample shows some
#: difference between its halves.
#:
#: So this table does not grow on intuition. `src/research/band_split.py`
#: answers whether a cut is warranted -- both sides clearing the training
#: floor on their own, and their outcomes differing beyond noise on the target
#: the entry decision turns on -- and the shadow trainer reports its verdict
#: on every run under `split_warrants`. Adding a band means recording that
#: report next to the edit.
AGE_BANDS: Tuple[Tuple[str, float, float], ...] = (
    ("flash", 0.0, 0.5),
    ("early", 0.5, 5.0),
    ("forming", 5.0, 60.0),
    ("mature", 60.0, float("inf")),
)


def band_for(age_seconds: float) -> str:
    """Which brain owns a decision at this age."""
    age = max(0.0, float(age_seconds))
    for name, low, high in AGE_BANDS:
        if low <= age < high:
            return name
    return AGE_BANDS[-1][0]


CLASSIFICATION_TARGETS = {target for target, _ in SURVIVAL_LEVELS} | {
    PredictionTarget.P_MIGRATION, PredictionTarget.P_RUG_30S, PredictionTarget.P_RUG_5M,
}


@dataclass
class PredictionFeatures:
    token: str
    chain: str
    timestamp: float
    
    deployer_rug_rate: float = 0
    deployer_success_rate: float = 0
    deployer_avg_multiple: float = 0
    deployer_cluster_risk: float = 0
    funding_wallet_risk: float = 0
    funding_wallet_reuse: float = 0
    wallet_quality_weighted_flow: float = 0
    sybil_discount: float = 0
    smart_wallet_sync_evidence: float = 0
    
    initial_buyers: int = 0
    smart_buyers: int = 0
    insider_buyers: int = 0
    buyer_acceleration: float = 0
    buy_velocity: float = 0
    sol_volume: float = 0
    organic_ratio: float = 0
    bundle_concentration: float = 0
    
    liquidity_usd: float = 0
    liquidity_locked: bool = False
    buy_tax: float = 0
    sell_tax: float = 0
    ownership_renounced: bool = False
    can_mint: bool = False
    can_freeze: bool = False
    
    social_velocity: float = 0
    social_acceleration: float = 0
    social_price_disagreement: float = 0
    social_credibility: float = 0
    chain_before_social: float = 0
    cross_platform: bool = False
    
    narrative_novelty: float = 0
    narrative_momentum: float = 0
    
    holder_concentration: float = 0
    holder_concentration_delta: float = 0
    holder_concentration_velocity: float = 0
    top_10_pct: float = 0
    top_20_pct: float = 0
    top_20_delta_pct: float = 0
    deployer_pct: float = 0
    insider_pct: float = 0
    bundler_pct: float = 0
    fresh_wallet_pct: float = 0
    whale_pct: float = 0
    connected_cluster_pct: float = 0
    dev_recent_sells: float = 0
    dev_sell_supply_pct: float = 0
    dev_hard_veto_count: float = 0
    capital_rotation_flow: float = 0
    token_extension_risk: float = 0
    meme_launch_rate_1h: float = 0
    sol_change_24h: float = 0
    btc_change_24h: float = 0
    sol_btc_beta: float = 0
    solana_tvl_change: float = 0
    priority_fee_p90: float = 0
    fee_pressure: float = 0
    
    regime: str = "unknown"
    time_since_launch: float = 0
    data_coverage: float = 0
    wallet_history_available: bool = False
    social_available: bool = False
    social_price_disagreement_available: bool = False
    holder_trajectory_available: bool = False
    holder_owner_enrichment_available: bool = False
    actor_flow_available: bool = False
    dev_state_available: bool = False
    capital_rotation_available: bool = False
    coordination_available: bool = False
    flow_available: bool = False
    
    def to_array(self) -> np.ndarray:
        return np.array([
            self.deployer_rug_rate,
            self.deployer_success_rate,
            self.deployer_avg_multiple / 100,
            self.deployer_cluster_risk,
            self.funding_wallet_risk,
            self.funding_wallet_reuse,
            float(np.clip(np.log1p(max(0.0, self.wallet_quality_weighted_flow)) / 10, 0, 1)),
            self.sybil_discount,
            float(np.clip(self.smart_wallet_sync_evidence / 10, 0, 1)),
            min(self.initial_buyers / 50, 1),
            min(self.smart_buyers / 10, 1),
            min(self.insider_buyers / 10, 1),
            self.buyer_acceleration,
            min(self.buy_velocity / 100, 1),
            min(self.sol_volume / 100, 1),
            self.organic_ratio,
            self.bundle_concentration,
            min(self.liquidity_usd / 50000, 1),
            float(self.liquidity_locked),
            self.buy_tax / 100,
            self.sell_tax / 100,
            float(self.ownership_renounced),
            float(self.can_mint),
            float(self.can_freeze),
            min(self.social_velocity / 10, 1),
            min(self.social_acceleration / 5, 1),
            float(np.clip(self.social_price_disagreement, -5, 5) / 5),
            self.social_credibility,
            self.chain_before_social,
            float(self.cross_platform),
            self.narrative_novelty,
            self.narrative_momentum,
            self.holder_concentration,
            self.holder_concentration_delta,
            self.holder_concentration_velocity,
            self.top_10_pct / 100,
            self.top_20_pct / 100,
            self.top_20_delta_pct / 100,
            self.deployer_pct / 100,
            self.insider_pct / 100,
            self.bundler_pct / 100,
            self.fresh_wallet_pct / 100,
            self.whale_pct / 100,
            self.connected_cluster_pct / 100,
            min(self.dev_recent_sells / 10, 1),
            self.dev_sell_supply_pct / 100,
            min(self.dev_hard_veto_count, 1),
            float(np.clip(np.sign(self.capital_rotation_flow)
                          * np.log1p(abs(self.capital_rotation_flow)) / 10, -1, 1)),
            self.token_extension_risk,
            min(self.meme_launch_rate_1h / 500, 1),
            np.clip(self.sol_change_24h / 100, -1, 1),
            np.clip(self.btc_change_24h / 100, -1, 1),
            np.clip(self.sol_btc_beta / 5, -1, 1),
            np.clip(self.solana_tvl_change, -1, 1),
            min(self.priority_fee_p90 / 100_000, 1),
            min(self.fee_pressure / 20, 1),
            self.data_coverage,
            float(self.wallet_history_available),
            float(self.social_available),
            float(self.social_price_disagreement_available),
            float(self.holder_trajectory_available),
            float(self.holder_owner_enrichment_available),
            float(self.actor_flow_available),
            float(self.dev_state_available),
            float(self.capital_rotation_available),
            float(self.coordination_available),
            float(self.flow_available),
            # Age was carried on the dataclass and never reached the array, so
            # every model was blind to whether it was being asked about a
            # hundred-millisecond launch or an hour-old one. Log-scaled
            # because the difference between 50ms and 500ms is the whole
            # decision while the difference between 50 and 55 minutes is not.
            float(np.clip(np.log1p(max(0.0, self.time_since_launch)) / 10.0, 0, 1)),
            *(float(self.regime == name) for name in REGIME_NAMES),
        ], dtype=np.float32)


@dataclass
class MultiHeadPrediction:
    token: str
    chain: str
    timestamp: float
    
    p_2x: float = 0
    p_5x: float = 0
    p_10x: float = 0
    p_20x: float = 0
    p_50x: float = 0
    p_100x: float = 0
    p_250x: float = 0
    p_500x: float = 0
    p_migration: float = 0
    p_rug_30s: float = 0
    p_rug_5m: float = 0
    expected_slippage: float = 0
    expected_hold_time: float = 0
    expected_feasible_multiple: float = 0
    
    model_version: str = ""
    feature_hash: str = ""
    calibration_version: str = ""
    # Which age brain answered, and whether it was the one fitted to this age.
    # Carried on the prediction rather than looked up later, so a decision
    # made on a labelled pooled fallback is distinguishable afterwards from
    # one made on the band's own model -- promotion reads this.
    age_band: str = ""
    band_status: str = ""


class MultiHeadPredictor:
    ARTIFACT_VERSION = 5

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.models: Dict[PredictionTarget, Any] = {}
        self.calibrators: Dict[PredictionTarget, Any] = {}
        self.feature_names = [
            "deployer_rug_rate", "deployer_success_rate", "deployer_avg_multiple",
            "deployer_cluster_risk", "funding_wallet_risk", "funding_wallet_reuse",
            "wallet_quality_weighted_flow", "sybil_discount",
            "smart_wallet_sync_evidence", "initial_buyers",
            "smart_buyers", "insider_buyers", "buyer_acceleration", "buy_velocity",
            "sol_volume", "organic_ratio", "bundle_concentration", "liquidity_usd",
            "liquidity_locked", "buy_tax", "sell_tax", "ownership_renounced",
            "can_mint", "can_freeze", "social_velocity", "social_acceleration",
            "social_price_disagreement",
            "social_credibility", "chain_before_social", "cross_platform",
            "narrative_novelty", "narrative_momentum", "holder_concentration",
            "holder_concentration_delta", "holder_concentration_velocity",
            "top_10_pct", "top_20_pct", "top_20_delta_pct", "deployer_pct",
            "insider_pct", "bundler_pct", "fresh_wallet_pct", "whale_pct",
            "connected_cluster_pct", "token_extension_risk", "meme_launch_rate_1h",
            "dev_recent_sells", "dev_sell_supply_pct", "dev_hard_veto_count",
            "capital_rotation_flow",
            "sol_change_24h", "btc_change_24h", "sol_btc_beta", "solana_tvl_change",
            "priority_fee_p90", "fee_pressure",
            "data_coverage", "wallet_history_available",
            "social_available", "social_price_disagreement_available",
            "holder_trajectory_available", "holder_owner_enrichment_available",
            "actor_flow_available",
            "dev_state_available", "capital_rotation_available",
            "coordination_available", "flow_available",
            "time_since_launch",
            *(f"regime_{name}" for name in REGIME_NAMES),
        ]
        self.model_version = "1.0"
        self._training_data: Dict[PredictionTarget, List[Tuple[np.ndarray, float, float]]] = defaultdict(list)
        self._is_trained = False
        self.validation_report: Dict[str, Any] = {}

    def initialize_models(self):
        for target in PredictionTarget:
            if target in CLASSIFICATION_TARGETS:
                self.models[target] = GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    random_state=42
                )
            else:
                self.models[target] = GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    random_state=42
                )

    def add_training_sample(self, features: PredictionFeatures, labels: Dict[PredictionTarget, float]):
        X = features.to_array()
        for target, y in labels.items():
            if target in self.models:
                self._training_data[target].append((X, y, features.timestamp))

    def train(self, min_samples: int = 100) -> Dict[str, Any]:
        results = {}
        
        for target, data in self._training_data.items():
            if len(data) < min_samples:
                logger.warning(f"Insufficient samples for {target.value}: {len(data)} < {min_samples}")
                results[target.value] = {"status": "insufficient_data", "samples": len(data)}
                continue
            
            ordered = sorted(data, key=lambda item: item[2])
            X = np.array([d[0] for d in ordered])
            y = np.array([d[1] for d in ordered])
            
            try:
                model = self.models[target]
                if target in CLASSIFICATION_TARGETS:
                    split = max(1, int(len(X) * 0.8))
                    X_fit, y_fit, X_cal, y_cal = X[:split], y[:split], X[split:], y[split:]
                    if len(np.unique(y_fit)) < 2:
                        raise ValueError("chronological fit window requires both classes")
                    model.fit(X_fit, y_fit)
                    calibration = "raw_probability"
                    if len(X_cal) >= 10 and len(np.unique(y_cal)) >= 2:
                        iso_reg = IsotonicRegression(out_of_bounds='clip')
                        iso_reg.fit(model.predict_proba(X_cal)[:, 1], y_cal)
                        self.calibrators[target] = iso_reg
                        calibration = "isotonic_chronological"
                else:
                    model.fit(X, y)
                    calibration = "not_applicable"
                
                results[target.value] = {
                    "status": "trained", "samples": len(data), "calibration": calibration,
                }
                logger.info(f"Trained {target.value} on {len(data)} samples")
                
            except Exception as e:
                logger.error(f"Training failed for {target.value}: {e}")
                results[target.value] = {"status": "failed", "error": str(e)}
        
        trained_targets = {PredictionTarget(key) for key, result in results.items() if result.get("status") == "trained"}
        required = set(PredictionTarget)
        self._is_trained = required.issubset(trained_targets)
        self.model_version = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return results

    def predict(self, features: PredictionFeatures) -> Optional[MultiHeadPrediction]:
        if not self._is_trained:
            return None
        
        X = features.to_array().reshape(1, -1)
        feature_hash = hashlib.md5(X.tobytes()).hexdigest()[:16]
        
        pred = MultiHeadPrediction(
            token=features.token,
            chain=features.chain,
            timestamp=features.timestamp,
            model_version=self.model_version,
            feature_hash=feature_hash
        )
        
        for target, model in self.models.items():
            try:
                if target in CLASSIFICATION_TARGETS:
                    raw = model.predict_proba(X)[:, 1]
                    calibrator = self.calibrators.get(target)
                    prob = calibrator.predict(raw)[0] if calibrator else raw[0]
                    setattr(pred, target.value, float(np.clip(prob, 0, 1)))
                else:
                    val = model.predict(X)[0]
                    if target == PredictionTarget.EXPECTED_SLIPPAGE:
                        val = np.clip(val, 0, 1)
                    elif target == PredictionTarget.EXPECTED_HOLD_TIME:
                        val = max(0, val)
                    elif target == PredictionTarget.EXPECTED_FEASIBLE_MULTIPLE:
                        val = np.clip(val, 0.02, SURVIVAL_LEVELS[-1][1])
                    setattr(pred, target.value, float(val))
            except Exception as e:
                logger.error(f"Prediction failed for {target.value}: {e}")
                return None
        
        self._enforce_nested_monotonicity(pred)
        return pred

    def predict_batch(self, features_list: List[PredictionFeatures]) -> List[Optional[MultiHeadPrediction]]:
        if not self._is_trained or not features_list:
            return [None] * len(features_list)
        
        X = np.array([f.to_array() for f in features_list])
        results = []
        
        for i, features in enumerate(features_list):
            feature_hash = hashlib.md5(X[i].tobytes()).hexdigest()[:16]
            pred = MultiHeadPrediction(
                token=features.token,
                chain=features.chain,
                timestamp=features.timestamp,
                model_version=self.model_version,
                feature_hash=feature_hash
            )
            
            failed = False
            for target, model in self.models.items():
                try:
                    if target in CLASSIFICATION_TARGETS:
                        raw = model.predict_proba(X[i:i+1])[:, 1]
                        calibrator = self.calibrators.get(target)
                        prob = calibrator.predict(raw)[0] if calibrator else raw[0]
                        setattr(pred, target.value, float(np.clip(prob, 0, 1)))
                    else:
                        val = model.predict(X[i:i+1])[0]
                        if target == PredictionTarget.EXPECTED_SLIPPAGE:
                            val = np.clip(val, 0, 1)
                        elif target == PredictionTarget.EXPECTED_HOLD_TIME:
                            val = max(0, val)
                        elif target == PredictionTarget.EXPECTED_FEASIBLE_MULTIPLE:
                            val = np.clip(val, 0.02, SURVIVAL_LEVELS[-1][1])
                        setattr(pred, target.value, float(val))
                except Exception as e:
                    logger.error(f"Batch prediction failed for {target.value}: {e}")
                    failed = True
                    break
            
            if failed:
                results.append(None)
            else:
                self._enforce_nested_monotonicity(pred)
                results.append(pred)
        
        return results

    def save(self, path: str, validation_report: Optional[Dict[str, Any]] = None):
        import joblib
        if not self._is_trained:
            raise RuntimeError("refusing to save an incomplete model bundle")
        if not validation_report or validation_report.get("status") != "PASSED":
            raise RuntimeError("refusing to save a model without passed chronological validation")
        data = {
            "artifact_version": self.ARTIFACT_VERSION,
            "models": self.models,
            "calibrators": self.calibrators,
            "model_version": self.model_version,
            "feature_names": self.feature_names,
            "feature_schema_hash": hashlib.sha256("\n".join(self.feature_names).encode()).hexdigest(),
            "validation_report": validation_report,
            "trained_at": time.time(),
        }
        joblib.dump(data, path)
        logger.info(f"Saved models to {path}")

    def load(self, path: str):
        import joblib
        data = joblib.load(path)
        if data.get("artifact_version") != self.ARTIFACT_VERSION:
            raise ValueError("unsupported or unvalidated model artifact")
        expected_hash = hashlib.sha256("\n".join(self.feature_names).encode()).hexdigest()
        if data.get("feature_schema_hash") != expected_hash or data.get("feature_names") != self.feature_names:
            raise ValueError("model feature schema mismatch")
        if (data.get("validation_report") or {}).get("status") != "PASSED":
            raise ValueError("model artifact lacks passed chronological validation")
        self.models = data["models"]
        self.calibrators = data["calibrators"]
        self.model_version = data["model_version"]
        self.validation_report = dict(data["validation_report"])
        self._is_trained = True
        logger.info(f"Loaded models from {path}, version: {self.model_version}")

    def load_latest(self) -> bool:
        """Load the newest persisted model bundle; never invent bootstrap scores."""
        if not os.path.isdir(self.model_dir):
            return False
        candidates = [
            os.path.join(self.model_dir, name)
            for name in os.listdir(self.model_dir)
            if name.startswith("multihead-shadow-") and name.endswith((".joblib", ".pkl"))
        ]
        if not candidates:
            return False
        for candidate in sorted(candidates, key=os.path.getmtime, reverse=True):
            try:
                self.load(candidate)
                required = set(PredictionTarget)
                missing = required.difference(self.models)
                if missing or not CLASSIFICATION_TARGETS.issubset(self.calibrators):
                    raise ValueError(f"missing trained heads/calibrators: {sorted(item.value for item in missing)}")
                return True
            except Exception as exc:
                logger.error("Model bundle rejected (%s): %s", candidate, exc)
                self._is_trained = False
        return False

    @staticmethod
    def _enforce_nested_monotonicity(prediction: MultiHeadPrediction):
        """P(>=x) cannot rise as x rises. Heads are trained independently, so it can.

        An untrained tail head reads zero, which the accumulate below turns
        into "nothing above that rung", and the bins collapse the excess back
        into the last rung the model can actually see. That is the correct
        degradation: a model with no evidence about 500x must not be given a
        500x number, and must not have its 50x conviction deleted either.
        """
        names = [target.value for target, _ in SURVIVAL_LEVELS]
        levels = np.clip([getattr(prediction, name) for name in names], 0, 1)
        levels = np.minimum.accumulate(levels)
        for name, value in zip(names, levels):
            setattr(prediction, name, float(value))

    def get_feature_importance(self, target: PredictionTarget) -> Dict[str, float]:
        model = self.models.get(target)
        if not model or not hasattr(model, 'feature_importances_'):
            return {}
        
        importances = model.feature_importances_
        
        return dict(zip(self.feature_names, importances.tolist()))


class ElogwEngine:
    """Portfolio decision engine using disjoint tail-probability bins."""

    # Bounds on the state-dependent giveback allowance. A rule that can reach
    # zero is a hair trigger; one that can reach 1.0 is not a guard.
    GIVEBACK_FLOOR_SCALE = 0.5
    GIVEBACK_CEILING_SCALE = 1.6
    MAX_GIVEBACK_PCT = 0.60
    # The harvest hurdle rises; it never becomes a cap on the day.
    MAX_HARVEST_MULTIPLE = 4.0
    # Small-account concentration is widened, and still bounded.
    SMALL_ACCOUNT_MAX_SCALE = 3.0
    MAX_SMALL_ACCOUNT_POSITION_PCT = 0.20

    def __init__(
        self,
        predictor: MultiHeadPredictor,
        risk_aversion: float = 1.0,
        max_position_pct: float = 0.05,
        max_position_usd: float = 500.0,
        max_portfolio_risk: float = 0.10,
        max_total_exposure_pct: float = 0.30,
        max_concurrent_positions: int = 10,
        max_daily_loss_usd: float = 1_000.0,
        max_daily_loss_pct: Optional[float] = None,
        daily_giveback_pct: Optional[float] = None,
        daily_giveback_arm_pct: float = 0.5,
        min_edge_bps: float = 50,
        max_liquidity_fraction: float = 0.01,
        uncertainty_penalty: float = 0.15,
        drawdown_aversion_lambda: float = 3.0,
        harvest_trigger_ratio: float = 1.5,
        harvest_slope: float = 0.5,
        small_account_mode: bool = False,
        small_account_negligible_share: float = 0.002,
    ):
        self.predictor = predictor
        self.risk_aversion = max(risk_aversion, 0.1)
        self.max_position_pct = max_position_pct
        self.max_position_usd = max_position_usd
        self.max_portfolio_risk = max_portfolio_risk
        self.max_total_exposure_pct = max_total_exposure_pct
        self.max_concurrent_positions = max_concurrent_positions
        # A capacity floor, not a view on the trade: below this the book
        # cannot absorb the smallest position the sizing engine produces, so
        # there is no size at which the question becomes interesting.
        self.min_liquidity_usd = 5_000.0
        self.max_daily_loss = max_daily_loss_usd
        self.max_daily_loss_pct = max_daily_loss_pct
        self.daily_giveback_pct = daily_giveback_pct
        self.daily_giveback_arm_pct = daily_giveback_arm_pct
        self.min_edge_bps = min_edge_bps
        self.max_liquidity_fraction = max_liquidity_fraction
        self.harvest_trigger_ratio = max(0.0, harvest_trigger_ratio)
        self.harvest_slope = max(0.0, harvest_slope)
        self.small_account_mode = bool(small_account_mode)
        self.small_account_negligible_share = max(0.0, small_account_negligible_share)
        self._opportunity_quality: Optional[float] = None
        self._opportunity_uncertainty: float = 0.0
        self.uncertainty_penalty = uncertainty_penalty
        self.drawdown_aversion_lambda = max(0.0, drawdown_aversion_lambda)

        self.portfolio_value = 0.0
        self.open_positions: Dict[str, Dict] = {}
        self.daily_pnl = 0.0
        self.kill_switch_active = False
        self._pnl_day = self._utc_day()
        self._day_start_equity = 0.0
        self._daily_peak_pnl = 0.0

    @staticmethod
    def _utc_day() -> int:
        return int(time.time() // 86_400)

    def _roll_day_if_needed(self):
        """Reset the daily budget at the UTC day boundary.

        Without this the counter is not daily at all: it accumulates for the
        life of the process, so a month of profit silently finances a loss far
        past the configured limit, and a tripped switch never re-arms.
        """
        today = self._utc_day()
        if today != self._pnl_day:
            self._pnl_day = today
            self.daily_pnl = 0.0
            self.kill_switch_active = False
            self._daily_peak_pnl = 0.0
            self._day_start_equity = max(0.0, self.portfolio_value)
        elif self._day_start_equity <= 0:
            self._day_start_equity = max(0.0, self.portfolio_value)

    def daily_loss_limit(self) -> float:
        """The day's loss budget in USD.

        A percentage limit is anchored to equity at the START of the day, not
        current equity: anchoring to a shrinking balance would tighten the
        budget precisely as losses mount, halting the book early on a normal
        drawdown instead of at the level actually configured.
        """
        if self.max_daily_loss_pct is None:
            return self.max_daily_loss
        anchor = self._day_start_equity if self._day_start_equity > 0 else self.portfolio_value
        return max(0.0, float(self.max_daily_loss_pct) * max(0.0, anchor))

    def giveback_floor(self) -> Optional[float]:
        """Rising floor under the day's banked gains, or None when unarmed.

        A pure loss limit lets a +30% day round-trip to zero without ever
        tripping, because net PnL never goes negative. This ratchets a floor
        under the intraday peak instead: upside stays uncapped, but a fixed
        share of realized gains cannot be handed back.

        It stays unarmed until the peak is a meaningful fraction of the loss
        budget, so ordinary noise around break-even cannot halt the book.
        """
        if self.daily_giveback_pct is None:
            return None
        arm_at = self.daily_loss_limit() * max(0.0, self.daily_giveback_arm_pct)
        if self._daily_peak_pnl <= 0 or self._daily_peak_pnl < arm_at:
            return None
        return self._daily_peak_pnl * (1.0 - self.giveback_allowance())

    def giveback_allowance(self) -> float:
        """How much of the day's peak may be handed back, given today's edge.

        A fixed 35% is the wrong answer in both directions. On a day where the
        remaining opportunity set is exceptional, refusing to tolerate normal
        volatility halts the book in front of the trades worth taking. On a
        day where edge has collapsed, 35% is far more than should be risked to
        find out that it has.

        So the allowance moves with the quality of what is still available:

            allowance = base * quality_scale

        where quality is the observed edge of the current opportunity slate
        relative to the hurdle the engine already applies. It is clamped to a
        band around the configured base, because a state-dependent rule that
        can reach zero becomes a hair trigger and one that can reach one is
        not a guard at all.

        Uncertainty tightens it: a slate whose edge estimates are wide is not
        the same as a slate that is genuinely good, and treating them alike is
        how a bad day gets financed by a confident-looking model.
        """
        base = float(self.daily_giveback_pct or 0.0)
        if self._opportunity_quality is None:
            return base
        quality = float(np.clip(self._opportunity_quality, 0.0, 2.0))
        confidence = float(np.clip(1.0 - self._opportunity_uncertainty, 0.0, 1.0))
        scale = 0.5 + 0.5 * quality * confidence
        return float(np.clip(base * scale, base * self.GIVEBACK_FLOOR_SCALE,
                             min(base * self.GIVEBACK_CEILING_SCALE, self.MAX_GIVEBACK_PCT)))

    def observe_opportunity_set(self, quality: Optional[float],
                                uncertainty: float = 0.0) -> None:
        """Record how good the currently available opportunities look.

        ``quality`` is the best available edge divided by the engine's own
        hurdle: 1.0 means "exactly at the hurdle", 2.0 means "twice as good as
        we normally require". None means it was not measured, and the
        allowance falls back to the configured base rather than to an
        optimistic guess.
        """
        self._opportunity_quality = None if quality is None else float(quality)
        self._opportunity_uncertainty = float(np.clip(uncertainty, 0.0, 1.0))

    def harvest_hurdle_bps(self) -> float:
        """The edge a NEW trade must clear, raised after an exceptional run.

        This is not a profit cap and must never become one. The book never
        stops trading, and an extraordinary opportunity still clears an
        extraordinary hurdle. What rises is the price of taking ordinary risk
        once an unusual amount of the day's gain is already banked, because
        the marginal utility of one more ordinary trade has fallen while the
        amount it can undo has grown.
        """
        limit = self.daily_loss_limit()
        if limit <= 0 or self.daily_pnl <= 0:
            return self.min_edge_bps
        # Measured against the loss budget, which is the natural unit for "how
        # big is today relative to what we are willing to risk".
        harvest_ratio = self.daily_pnl / limit
        if harvest_ratio < self.harvest_trigger_ratio:
            return self.min_edge_bps
        excess = harvest_ratio - self.harvest_trigger_ratio
        return float(self.min_edge_bps * min(self.MAX_HARVEST_MULTIPLE,
                                             1.0 + excess * self.harvest_slope))

    def small_account_concentration(self, liquidity_usd: float) -> float:
        """Position ceiling as a fraction of equity, widened while capital is tiny.

        The liquidity-fraction cap exists so a position cannot be a large
        share of a pool. When equity is small enough that even a maximum
        position is a negligible share of the pool, that cap is not binding on
        anything real, and holding the book to the same per-position
        percentage as a large account gives up the one structural advantage a
        small account has. The cap tightens automatically as equity grows,
        because the same trade stops being negligible.

        The exposure ceiling, portfolio risk budget and daily-loss switch are
        untouched. This widens one ceiling, never the ones that bound ruin.
        """
        if not self.small_account_mode or liquidity_usd <= 0 or self.portfolio_value <= 0:
            return self.max_position_pct
        max_position_usd = self.portfolio_value * self.max_position_pct
        pool_share = max_position_usd / liquidity_usd
        if pool_share >= self.small_account_negligible_share:
            return self.max_position_pct
        return float(min(self.max_position_pct * self.SMALL_ACCOUNT_MAX_SCALE,
                         self.MAX_SMALL_ACCOUNT_POSITION_PCT))

    @staticmethod
    def probability_bins(prediction: MultiHeadPrediction) -> List[Tuple[str, float, float]]:
        """Convert the cumulative survival curve into disjoint outcomes.

        Returns ``(name, probability, gross return)``. Every bucket uses its
        conservative lower bound rather than an optimistic midpoint.

        When the tail heads are untrained they read zero, monotonicity pushes
        every rung above them to zero too, and the mass collapses into the
        highest rung the model can actually see -- which is exactly the old
        behaviour, at the old conservative bound. Extending the curve cannot
        manufacture tail conviction that was never trained.
        """
        MultiHeadPredictor._enforce_nested_monotonicity(prediction)
        levels = [(getattr(prediction, target.value), multiple)
                  for target, multiple in SURVIVAL_LEVELS]
        # Each bucket pays its own LOWER bound, never a midpoint. A 10x-to-20x
        # bucket paying 14x would be inventing the shape of the distribution
        # inside a bucket the model was never asked about; paying 9x cannot be
        # an overstatement of what the bucket contains.
        survival = [("under_2x", max(0.0, 1.0 - levels[0][0]), -0.35)]
        for index, (probability, multiple) in enumerate(levels):
            higher = levels[index + 1][0] if index + 1 < len(levels) else 0.0
            upper = levels[index + 1][1] if index + 1 < len(levels) else None
            name = (f"{multiple:g}x_to_{upper:g}x" if upper is not None
                    else f"{multiple:g}x_plus")
            survival.append((name, max(0.0, probability - higher), multiple - 1.0))
        if prediction.expected_feasible_multiple > 0:
            feasible_return = float(np.clip(prediction.expected_feasible_multiple - 1,
                                            -0.98, SURVIVAL_LEVELS[-1][1] - 1.0))
            survival = [
                (name, probability, min(outcome, feasible_return) if outcome > 0 else outcome)
                for name, probability, outcome in survival
            ]
        p_rug = float(np.clip(max(prediction.p_rug_30s, prediction.p_rug_5m), 0, 1))
        bins = [(name, probability * (1 - p_rug), outcome) for name, probability, outcome in survival]
        bins.append(("rug", p_rug, -0.98))
        total = sum(probability for _, probability, _ in bins)
        return [(name, probability / total, outcome) for name, probability, outcome in bins] if total else []

    def _growth_inputs(self, prediction: MultiHeadPrediction):
        """Shared (probabilities, net returns, normalised entropy) for one prediction."""
        bins = self.probability_bins(prediction)
        if not bins or prediction.p_2x <= 0:
            return None
        execution_cost = 0.003 + max(0.0, prediction.expected_slippage)
        probabilities = np.array([probability for _, probability, _ in bins], dtype=float)
        returns = np.array([outcome - execution_cost for _, _, outcome in bins], dtype=float)
        entropy = -float(np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1))))
        entropy /= max(np.log(len(probabilities)), 1e-12)
        return probabilities, returns, entropy

    def log_growth_at_fraction(self, prediction: MultiHeadPrediction, fraction: float) -> float:
        """E[log W] for committing exactly ``fraction`` of equity.

        The optimiser answers "what is the best size?"; this answers "what is
        this specific size worth?". Cross-sectional comparison needs the
        second question: when freed capital funds less than a candidate's
        optimum, reusing the optimum's number would claim an edge at a size
        that was never evaluated. E[log W] is not linear in size, so that is
        an invention rather than an approximation.
        """
        inputs = self._growth_inputs(prediction)
        if inputs is None or fraction < 0:
            return -float("inf")
        probabilities, returns, entropy = inputs
        wealth = 1 + fraction * returns
        if np.any(wealth <= 0):
            return -float("inf")
        if self.drawdown_aversion_lambda > 0 and fraction > 0:
            drawdown_moment = float(np.sum(probabilities * wealth ** (-self.drawdown_aversion_lambda)))
            if drawdown_moment > 1.0 + 1e-12:
                return -float("inf")
        value = float(np.sum(probabilities * np.log(wealth)))
        value -= self.uncertainty_penalty * entropy * fraction
        return value / self.risk_aversion

    def exposure_cap(self, liquidity_usd: float) -> float:
        """Largest fraction of equity this token may take, across all ceilings."""
        if self.portfolio_value <= 0 or liquidity_usd <= 0:
            return 0.0
        return min(
            self.small_account_concentration(liquidity_usd),
            self.max_position_usd / self.portfolio_value,
            liquidity_usd * self.max_liquidity_fraction / self.portfolio_value,
        )

    def calculate_expected_log_growth(
        self,
        prediction: MultiHeadPrediction,
        sol_price_usd: float,
        liquidity_usd: float,
    ) -> Tuple[float, float, float]:
        if self.portfolio_value <= 0 or sol_price_usd <= 0 or liquidity_usd <= 0:
            return -float("inf"), 0.0, 0.0
        if self._growth_inputs(prediction) is None:
            return -float("inf"), 0.0, 0.0
        cap = self.exposure_cap(liquidity_usd)
        if cap <= 0:
            return -float("inf"), 0.0, 0.0
        fractions = np.linspace(0, cap, 401)
        growth = [self.log_growth_at_fraction(prediction, float(f)) for f in fractions]
        index = int(np.argmax(growth))
        fraction = float(fractions[index])
        position_value = self.portfolio_value * fraction
        return float(growth[index]), fraction, position_value / sol_price_usd

    def marginal_log_growth(
        self,
        prediction: MultiHeadPrediction,
        held_cost_fraction: float,
        current_multiple: float,
        added_fraction: float,
    ) -> float:
        """E[log W] for adding ``added_fraction`` on top of an open position.

        A single all-at-once entry has to commit before the evidence that
        actually separates a launch has arrived. Scaling in needs the marginal
        quantity, not the from-scratch optimum: the slice already held rides
        from its own entry, while new capital enters at today's price, so both
        slices share the same forward return but not the same basis.

        Wealth is normalised to 1 at the current instant:
            cash            = 1 - held_cost - added
            position value  = held_cost * current_multiple + added
        and the forward return R applies to the position from here.
        """
        held_cost = max(0.0, float(held_cost_fraction))
        added = max(0.0, float(added_fraction))
        multiple = max(0.0, float(current_multiple))
        bins = self.probability_bins(prediction)
        if not bins:
            return -float("inf")

        cash = 1.0 - held_cost - added
        if cash < 0:
            return -float("inf")
        # A flat book is a legitimate baseline worth log(1) = 0, not an error:
        # it is exactly what "adding nothing" must score against.
        position_now = held_cost * multiple + added
        if position_now < 0:
            return -float("inf")

        execution_cost = 0.003 + max(0.0, prediction.expected_slippage)
        probabilities = np.array([probability for _, probability, _ in bins], dtype=float)
        # Only the newly added slice pays entry cost again.
        returns = np.array([outcome for _, _, outcome in bins], dtype=float)
        wealth = cash + position_now * (1.0 + returns) - added * execution_cost
        if np.any(wealth <= 0):
            return -float("inf")

        # The drawdown constraint governs capital being COMMITTED, not capital
        # already committed. Applying it at added == 0 asks "may I open this
        # position?" of a position that is already open, and answers -inf --
        # which is not a value the baseline can be compared against. It made
        # plan_scale_in bail out ("baseline not finite") for exactly the held
        # positions whose tail is worst, so a position could never be added to
        # once its own held state tripped the bound, regardless of whether the
        # addition itself was sound.
        if self.drawdown_aversion_lambda > 0 and added > 0:
            drawdown_moment = float(np.sum(probabilities * wealth ** (-self.drawdown_aversion_lambda)))
            if drawdown_moment > 1.0 + 1e-12:
                return -float("inf")

        entropy = -float(np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1))))
        entropy /= max(np.log(len(probabilities)), 1e-12)
        value = float(np.sum(probabilities * np.log(wealth)))
        value -= self.uncertainty_penalty * entropy * added
        return value / self.risk_aversion

    def plan_scale_in(
        self,
        prediction: MultiHeadPrediction,
        held_cost_usd: float,
        current_multiple: float,
        liquidity_usd: float,
        portfolio_value: Optional[float] = None,
        steps: int = 200,
    ) -> Tuple[float, float]:
        """Additional exposure to add now, as (fraction_of_equity, delta_elogw).

        Returns (0, 0) when no addition improves expected log growth — the
        stop-scaling condition. Capital is deployed as evidence arrives rather
        than guessed at T0, and the same rule cuts the position off the moment
        the marginal contribution turns negative.
        """
        if portfolio_value is not None:
            self.portfolio_value = max(0.0, portfolio_value)
        if self.portfolio_value <= 0 or liquidity_usd <= 0:
            return 0.0, 0.0

        held_fraction = max(0.0, held_cost_usd) / self.portfolio_value
        headroom = self.exposure_cap(liquidity_usd) - held_fraction
        if headroom <= 0:
            return 0.0, 0.0

        baseline = self.marginal_log_growth(prediction, held_fraction, current_multiple, 0.0)
        if not np.isfinite(baseline):
            return 0.0, 0.0

        best_fraction, best_gain = 0.0, 0.0
        for candidate in np.linspace(0.0, headroom, max(2, steps))[1:]:
            gain = self.marginal_log_growth(prediction, held_fraction, current_multiple, float(candidate))
            if not np.isfinite(gain):
                continue
            if gain - baseline > best_gain:
                best_fraction, best_gain = float(candidate), gain - baseline
        return best_fraction, best_gain

    #: Gates that express an OPINION about whether a trade is worth taking.
    #: Every one of them is a threshold on a quantity the action-value engine
    #: already prices, so applying them before Q means two components deciding
    #: the same question from different numbers -- and the threshold always
    #: wins, because it runs first.
    ECONOMIC_GATES = frozenset({
        "rug_risk_too_high", "insufficient_upside", "slippage_too_high",
        "edge_below_threshold",
    })

    def admissible(
        self,
        prediction: MultiHeadPrediction,
        sol_price_usd: float,
        liquidity_usd: float,
        portfolio_value: float,
    ) -> Tuple[bool, Dict]:
        """Hard admissibility. Safety, ruin limits, data integrity, capacity.

        Nothing here is an opinion about whether the trade is good. These are
        facts about the account and the measurement: a model that is not
        trained cannot price anything, a book at its exposure ceiling has no
        room, and a negative slippage reading is a broken instrument rather
        than a cheap fill. Refusing on any of them is refusing to answer, not
        answering no.

        Kept OUTSIDE the objective on purpose. A limit the objective can
        outbid has stopped being a limit, and every one of these is a case
        where the correct response is refusing rather than resizing.
        """
        self.portfolio_value = max(0.0, portfolio_value)
        self._roll_day_if_needed()
        if not self.predictor._is_trained:
            return False, {"reason": "DATA_BLOCKED", "detail": "no validated multi-head model bundle"}
        if self.kill_switch_active or self.daily_pnl <= -self.daily_loss_limit():
            self.kill_switch_active = True
            return False, {"reason": "daily_loss_kill_switch", "daily_pnl": self.daily_pnl}
        if self.portfolio_value <= 0 or sol_price_usd <= 0:
            return False, {"reason": "DATA_BLOCKED", "detail": "wallet equity or SOL/USD unavailable"}
        if len(self.open_positions) >= self.max_concurrent_positions:
            return False, {"reason": "max_concurrent_positions"}
        if prediction.expected_slippage < 0:
            # Not a cheap fill. A negative estimate is an instrument reading
            # something it cannot read, and pricing a trade on it would be
            # pricing on noise.
            return False, {"reason": "DATA_BLOCKED", "detail": "negative slippage estimate"}
        if liquidity_usd < self.min_liquidity_usd:
            # Capacity, not opinion: below this the book cannot absorb the
            # smallest position the sizing engine will produce, so there is no
            # size at which the question becomes interesting.
            return False, {"reason": "liquidity_too_low", "liquidity_usd": liquidity_usd}
        return True, {"admissible": True}

    def size_candidate(
        self,
        prediction: MultiHeadPrediction,
        sol_price_usd: float,
        liquidity_usd: float,
    ) -> Dict:
        """The size a candidate would take, and what that size is worth.

        Separate from admissibility and from Q. Sizing owns how much; Q owns
        whether. Fusing them is what let a Kelly hurdle quietly become the
        entry policy.
        """
        elogw, fraction, size_sol = self.calculate_expected_log_growth(
            prediction, sol_price_usd, liquidity_usd)
        position_value = size_sol * sol_price_usd
        p_rug = max(prediction.p_rug_30s, prediction.p_rug_5m)
        return {
            "elogw": elogw,
            "kelly_fraction": fraction,
            "position_size_sol": size_sol,
            "position_value_usd": position_value,
            "risk_contribution": fraction * p_rug,
            "edge_bps": elogw * 10_000,
            "hurdle_bps": self.harvest_hurdle_bps(),
            "p_rug": p_rug,
            "probability_bins": self.probability_bins(prediction),
            "drawdown_aversion_lambda": self.drawdown_aversion_lambda,
        }

    def portfolio_room(self, sized: Dict) -> Tuple[bool, Dict]:
        """Whether the book has room for this size. A ruin limit, not a view."""
        position_value = float(sized.get("position_value_usd", 0.0))
        current_exposure = sum(
            float(pos.get("remaining_cost_usd", pos.get("cost_basis_usd", 0)))
            for pos in self.open_positions.values())
        if current_exposure + position_value > self.portfolio_value * self.max_total_exposure_pct:
            return False, {"reason": "total_exposure_limit"}
        current_risk = sum(float(pos.get("risk_contribution", 0))
                           for pos in self.open_positions.values())
        if current_risk + float(sized.get("risk_contribution", 0.0)) > self.max_portfolio_risk:
            return False, {"reason": "portfolio_risk_limit"}
        return True, {}

    def should_trade(
        self,
        prediction: MultiHeadPrediction,
        sol_price_usd: float,
        liquidity_usd: float,
        portfolio_value: float,
    ) -> Tuple[bool, Dict]:
        """Admissibility, the legacy economic gates, then sizing.

        Retained for callers that have no action-value engine to defer to --
        the shadow trainer, the replay harness, anything scoring a historical
        decision the way it was actually made. The live desk does NOT use it:
        it composes `admissible`, `size_candidate` and `portfolio_room` and
        lets Q answer the economic question, because a threshold on P(2x) and
        an expected-log-growth objective are two opinions about one thing and
        the threshold wins by running first.
        """
        allowed, detail = self.admissible(
            prediction, sol_price_usd, liquidity_usd, portfolio_value)
        if not allowed:
            return False, detail

        if prediction.p_rug_30s > 0.40 or prediction.p_rug_5m > 0.50:
            return False, {"reason": "rug_risk_too_high"}
        if prediction.p_2x < 0.10 or prediction.p_5x < 0.05:
            return False, {"reason": "insufficient_upside"}
        if prediction.expected_slippage > 0.15:
            return False, {"reason": "slippage_too_high"}

        sized = self.size_candidate(prediction, sol_price_usd, liquidity_usd)
        if not np.isfinite(sized["elogw"]) or sized["edge_bps"] < sized["hurdle_bps"]:
            return False, {"reason": "edge_below_threshold",
                           "edge_bps": sized["edge_bps"],
                           "hurdle_bps": sized["hurdle_bps"],
                           "harvesting": sized["hurdle_bps"] > self.min_edge_bps}

        has_room, room_detail = self.portfolio_room(sized)
        if not has_room:
            return False, room_detail
        return True, sized

    def update_position(self, token: str, position_data: Dict):
        if token in self.open_positions:
            raise ValueError(f"position already exists for {token}")
        required = {"size_tokens", "remaining_cost_usd", "risk_contribution"}
        missing = required.difference(position_data)
        if missing:
            raise ValueError(f"position missing risk/accounting fields: {sorted(missing)}")
        position_data.setdefault("initial_size_tokens", int(position_data["size_tokens"]))
        position_data.setdefault("initial_risk_contribution", float(position_data["risk_contribution"]))
        self.open_positions[token] = position_data

    def reduce_position(self, token: str, sold_tokens: int, allocated_cost_usd: float):
        position = self.open_positions[token]
        position["size_tokens"] = max(0, int(position["size_tokens"]) - int(sold_tokens))
        position["remaining_cost_usd"] = max(0.0, float(position["remaining_cost_usd"]) - allocated_cost_usd)
        original = max(float(position.get("initial_size_tokens", position["size_tokens"])), 1)
        initial_risk = float(position.get("initial_risk_contribution", position["risk_contribution"]))
        position["risk_contribution"] = initial_risk * position["size_tokens"] / original
        if position["size_tokens"] == 0:
            self.close_position(token)

    def close_position(self, token: str):
        self.open_positions.pop(token, None)

    def update_pnl(self, pnl: float):
        self._roll_day_if_needed()
        self.daily_pnl += float(pnl)
        self._daily_peak_pnl = max(self._daily_peak_pnl, self.daily_pnl)
        if self.daily_pnl <= -self.daily_loss_limit():
            self.kill_switch_active = True
            logger.critical("Daily loss kill switch activated: %.2f (limit %.2f)",
                            self.daily_pnl, self.daily_loss_limit())
            return
        floor = self.giveback_floor()
        if floor is not None and self.daily_pnl <= floor:
            self.kill_switch_active = True
            logger.critical(
                "Daily giveback guard activated: pnl %.2f fell to the %.2f floor under a %.2f peak",
                self.daily_pnl, floor, self._daily_peak_pnl,
            )

    def get_portfolio_state(self) -> Dict:
        exposure = sum(float(pos.get("remaining_cost_usd", 0)) for pos in self.open_positions.values())
        return {
            "portfolio_value": self.portfolio_value,
            "daily_pnl": self.daily_pnl,
            "open_positions": len(self.open_positions),
            "exposure_usd": exposure,
            "portfolio_risk": sum(float(pos.get("risk_contribution", 0)) for pos in self.open_positions.values()),
            "max_daily_loss": self.daily_loss_limit(),
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "day_start_equity": self._day_start_equity,
            "kill_switch_active": self.kill_switch_active,
            "risk_budget_remaining": max(0.0, self.daily_loss_limit() + self.daily_pnl),
            "daily_peak_pnl": self._daily_peak_pnl,
            "daily_giveback_floor": self.giveback_floor(),
        }

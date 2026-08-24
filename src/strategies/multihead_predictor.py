import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import json
import hashlib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


class PredictionTarget(Enum):
    P_2X = "p_2x"
    P_5X = "p_5x"
    P_10X = "p_10x"
    P_MIGRATION = "p_migration"
    P_RUG_30S = "p_rug_30s"
    P_RUG_5M = "p_rug_5m"
    EXPECTED_SLIPPAGE = "expected_slippage"
    EXPECTED_HOLD_TIME = "expected_hold_time"


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
    social_credibility: float = 0
    chain_before_social: float = 0
    cross_platform: bool = False
    
    narrative_novelty: float = 0
    narrative_momentum: float = 0
    
    holder_concentration: float = 0
    top_10_pct: float = 0
    deployer_pct: float = 0
    
    regime: str = "unknown"
    time_since_launch: float = 0
    
    def to_array(self) -> np.ndarray:
        return np.array([
            self.deployer_rug_rate,
            self.deployer_success_rate,
            self.deployer_avg_multiple / 100,
            self.deployer_cluster_risk,
            self.funding_wallet_risk,
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
            self.social_credibility,
            self.chain_before_social,
            float(self.cross_platform),
            self.narrative_novelty,
            self.narrative_momentum,
            self.holder_concentration,
            self.top_10_pct / 100,
            self.deployer_pct / 100,
        ], dtype=np.float32)


@dataclass
class MultiHeadPrediction:
    token: str
    chain: str
    timestamp: float
    
    p_2x: float = 0
    p_5x: float = 0
    p_10x: float = 0
    p_migration: float = 0
    p_rug_30s: float = 0
    p_rug_5m: float = 0
    expected_slippage: float = 0
    expected_hold_time: float = 0
    
    model_version: str = ""
    feature_hash: str = ""
    calibration_version: str = ""


class MultiHeadPredictor:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.models: Dict[PredictionTarget, Any] = {}
        self.calibrators: Dict[PredictionTarget, Any] = {}
        self.feature_names = [
            "deployer_rug_rate", "deployer_success_rate", "deployer_avg_multiple",
            "deployer_cluster_risk", "funding_wallet_risk", "initial_buyers",
            "smart_buyers", "insider_buyers", "buyer_acceleration", "buy_velocity",
            "sol_volume", "organic_ratio", "bundle_concentration", "liquidity_usd",
            "liquidity_locked", "buy_tax", "sell_tax", "ownership_renounced",
            "can_mint", "can_freeze", "social_velocity", "social_acceleration",
            "social_credibility", "chain_before_social", "cross_platform",
            "narrative_novelty", "narrative_momentum", "holder_concentration",
            "top_10_pct", "deployer_pct"
        ]
        self.model_version = "1.0"
        self._training_data: Dict[PredictionTarget, List[Tuple[np.ndarray, float]]] = defaultdict(list)
        self._is_trained = False

    def initialize_models(self):
        for target in PredictionTarget:
            if target in [PredictionTarget.P_2X, PredictionTarget.P_5X, 
                         PredictionTarget.P_10X, PredictionTarget.P_MIGRATION,
                         PredictionTarget.P_RUG_30S, PredictionTarget.P_RUG_5M]:
                base_model = GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    random_state=42
                )
                self.models[target] = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
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
                self._training_data[target].append((X, y))

    def train(self, min_samples: int = 100) -> Dict[str, Any]:
        results = {}
        
        for target, data in self._training_data.items():
            if len(data) < min_samples:
                logger.warning(f"Insufficient samples for {target.value}: {len(data)} < {min_samples}")
                results[target.value] = {"status": "insufficient_data", "samples": len(data)}
                continue
            
            X = np.array([d[0] for d in data])
            y = np.array([d[1] for d in data])
            
            try:
                model = self.models[target]
                model.fit(X, y)
                
                if hasattr(model, 'calibrated_classifiers_'):
                    self.calibrators[target] = model
                else:
                    iso_reg = IsotonicRegression(out_of_bounds='clip')
                    preds = model.predict(X)
                    iso_reg.fit(preds, y)
                    self.calibrators[target] = iso_reg
                
                results[target.value] = {"status": "trained", "samples": len(data)}
                logger.info(f"Trained {target.value} on {len(data)} samples")
                
            except Exception as e:
                logger.error(f"Training failed for {target.value}: {e}")
                results[target.value] = {"status": "failed", "error": str(e)}
        
        self._is_trained = True
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
                if target in self.calibrators:
                    calibrator = self.calibrators[target]
                    if hasattr(calibrator, 'predict_proba'):
                        prob = calibrator.predict_proba(X)[0, 1]
                    else:
                        raw = model.predict(X)[0]
                        prob = calibrator.predict([raw])[0]
                    setattr(pred, target.value, float(np.clip(prob, 0, 1)))
                else:
                    val = model.predict(X)[0]
                    if target == PredictionTarget.EXPECTED_SLIPPAGE:
                        val = np.clip(val, 0, 1)
                    elif target == PredictionTarget.EXPECTED_HOLD_TIME:
                        val = max(0, val)
                    setattr(pred, target.value, float(val))
            except Exception as e:
                logger.error(f"Prediction failed for {target.value}: {e}")
                setattr(pred, target.value, 0.0)
        
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
            
            for target, model in self.models.items():
                try:
                    if target in self.calibrators:
                        calibrator = self.calibrators[target]
                        if hasattr(calibrator, 'predict_proba'):
                            prob = calibrator.predict_proba(X[i:i+1])[0, 1]
                        else:
                            raw = model.predict(X[i:i+1])[0]
                            prob = calibrator.predict([raw])[0]
                        setattr(pred, target.value, float(np.clip(prob, 0, 1)))
                    else:
                        val = model.predict(X[i:i+1])[0]
                        if target == PredictionTarget.EXPECTED_SLIPPAGE:
                            val = np.clip(val, 0, 1)
                        elif target == PredictionTarget.EXPECTED_HOLD_TIME:
                            val = max(0, val)
                        setattr(pred, target.value, float(val))
                except Exception as e:
                    logger.error(f"Batch prediction failed for {target.value}: {e}")
                    setattr(pred, target.value, 0.0)
            
            results.append(pred)
        
        return results

    def save(self, path: str):
        import joblib
        data = {
            "models": self.models,
            "calibrators": self.calibrators,
            "model_version": self.model_version,
            "feature_names": self.feature_names
        }
        joblib.dump(data, path)
        logger.info(f"Saved models to {path}")

    def load(self, path: str):
        import joblib
        data = joblib.load(path)
        self.models = data["models"]
        self.calibrators = data["calibrators"]
        self.model_version = data["model_version"]
        self.feature_names = data["feature_names"]
        self._is_trained = True
        logger.info(f"Loaded models from {path}, version: {self.model_version}")

    def get_feature_importance(self, target: PredictionTarget) -> Dict[str, float]:
        model = self.models.get(target)
        if not model or not hasattr(model, 'feature_importances_'):
            return {}
        
        if hasattr(model, 'calibrated_classifiers_'):
            base_model = model.calibrated_classifiers_[0].base_estimator
            importances = base_model.feature_importances_
        else:
            importances = model.feature_importances_
        
        return dict(zip(self.feature_names, importances.tolist()))


class ElogwEngine:
    def __init__(
        self,
        predictor: MultiHeadPredictor,
        risk_aversion: float = 1.0,
        max_position_pct: float = 0.05,
        max_portfolio_risk: float = 0.1,
        min_edge_bps: float = 50,
        rug_penalty_multiplier: float = 3.0,
        slippage_penalty_multiplier: float = 2.0,
        uncertainty_penalty: float = 0.5
    ):
        self.predictor = predictor
        self.risk_aversion = risk_aversion
        self.max_position_pct = max_position_pct
        self.max_portfolio_risk = max_portfolio_risk
        self.min_edge_bps = min_edge_bps
        self.rug_penalty = rug_penalty_multiplier
        self.slippage_penalty = slippage_penalty_multiplier
        self.uncertainty_penalty = uncertainty_penalty
        
        self.portfolio_value = 10000.0
        self.open_positions: Dict[str, Dict] = {}
        self.daily_pnl = 0.0
        self.max_daily_loss = 0.05 * self.portfolio_value

    def calculate_expected_log_growth(self, prediction: MultiHeadPrediction, 
                                       position_size_sol: float,
                                       current_price: float,
                                       liquidity_usd: float) -> float:
        p_2x = prediction.p_2x
        p_5x = prediction.p_5x
        p_10x = prediction.p_10x
        p_rug_30s = prediction.p_rug_30s
        p_rug_5m = prediction.p_rug_5m
        p_migration = prediction.p_migration
        expected_slippage = prediction.expected_slippage
        expected_hold_time = prediction.expected_hold_time
        
        if p_2x == 0 and p_5x == 0 and p_10x == 0:
            return -float('inf')
        
        p_rug = max(p_rug_30s, p_rug_5m)
        
        outcomes = []
        probs = []
        
        p_loss = p_rug
        if p_loss > 0:
            outcomes.append(-0.95)
            probs.append(p_loss)
        
        p_1x = max(0, 1 - p_2x - p_5x - p_10x - p_loss)
        if p_1x > 0:
            outcomes.append(0)
            probs.append(p_1x)
        
        if p_2x > 0:
            outcomes.append(1.0)
            probs.append(p_2x * 0.6)
        
        if p_5x > 0:
            outcomes.append(4.0)
            probs.append(p_5x * 0.3)
        
        if p_10x > 0:
            outcomes.append(9.0)
            probs.append(p_10x * 0.1)
        
        total_p = sum(probs)
        if total_p == 0:
            return -float('inf')
        probs = [p / total_p for p in probs]
        
        fees_bps = 30
        slippage_cost = expected_slippage * self.slippage_penalty
        execution_cost = (fees_bps + slippage_cost * 10000) / 10000
        
        net_outcomes = [o - execution_cost for o in outcomes]
        
        kelly_fractions = []
        for i, (outcome, prob) in enumerate(zip(net_outcomes, probs)):
            if outcome > 0:
                kelly_f = (prob * outcome - (1 - prob) * abs(min(net_outcomes))) / outcome
                kelly_fractions.append(max(0, kelly_f))
            else:
                kelly_fractions.append(0)
        
        avg_kelly = np.mean(kelly_fractions) if kelly_fractions else 0
        kelly_fraction = avg_kelly / self.risk_aversion
        
        uncertainty = 1 - (p_2x + p_5x + p_10x)
        kelly_fraction *= (1 - uncertainty * self.uncertainty_penalty)
        
        if p_rug > 0.2:
            kelly_fraction *= (1 - p_rug * self.rug_penalty)
        
        kelly_fraction = max(0, min(kelly_fraction, self.max_position_pct))
        
        position_value = self.portfolio_value * kelly_fraction
        position_size_sol = position_value / current_price if current_price > 0 else 0
        
        expected_log_growth = 0
        for outcome, prob in zip(net_outcomes, probs):
            if outcome > -1:
                expected_log_growth += prob * np.log(1 + kelly_fraction * outcome)
        
        expected_log_growth -= self.uncertainty_penalty * uncertainty * kelly_fraction
        expected_log_growth -= p_rug * self.rug_penalty * kelly_fraction
        
        return expected_log_growth, kelly_fraction, position_size_sol

    def should_trade(self, prediction: MultiHeadPrediction, 
                     current_price: float, liquidity_usd: float,
                     portfolio_value: float) -> Tuple[bool, Dict]:
        self.portfolio_value = portfolio_value
        
        if prediction.p_rug_30s > 0.4 or prediction.p_rug_5m > 0.5:
            return False, {"reason": "rug_risk_too_high", "p_rug_30s": prediction.p_rug_30s, "p_rug_5m": prediction.p_rug_5m}
        
        if prediction.p_2x < 0.1 and prediction.p_5x < 0.05:
            return False, {"reason": "insufficient_upside", "p_2x": prediction.p_2x, "p_5x": prediction.p_5x}
        
        if prediction.expected_slippage > 0.15:
            return False, {"reason": "slippage_too_high", "expected_slippage": prediction.expected_slippage}
        
        if liquidity_usd < 5000:
            return False, {"reason": "liquidity_too_low", "liquidity_usd": liquidity_usd}
        
        expected_log_growth, kelly_fraction, position_size = self.calculate_expected_log_growth(
            prediction, 0, current_price, liquidity_usd
        )
        
        if expected_log_growth <= 0:
            return False, {"reason": "negative_expected_growth", "elogw": expected_log_growth}
        
        edge_bps = expected_log_growth * 10000
        if edge_bps < self.min_edge_bps:
            return False, {"reason": "edge_below_threshold", "edge_bps": edge_bps, "threshold": self.min_edge_bps}
        
        current_portfolio_risk = sum(pos.get("risk_contribution", 0) for pos in self.open_positions.values())
        position_risk = kelly_fraction * (prediction.p_rug_30s + prediction.p_rug_5m)
        
        if current_portfolio_risk + position_risk > self.max_portfolio_risk:
            return False, {"reason": "portfolio_risk_limit", "current_risk": current_portfolio_risk, "position_risk": position_risk}
        
        return True, {
            "elogw": expected_log_growth,
            "kelly_fraction": kelly_fraction,
            "position_size_sol": position_size,
            "position_value_usd": portfolio_value * kelly_fraction,
            "edge_bps": edge_bps,
            "p_rug": max(prediction.p_rug_30s, prediction.p_rug_5m),
            "expected_slippage": prediction.expected_slippage
        }

    def update_position(self, token: str, position_data: Dict):
        self.open_positions[token] = position_data

    def close_position(self, token: str):
        self.open_positions.pop(token, None)

    def update_pnl(self, pnl: float):
        self.daily_pnl += pnl
        if self.daily_pnl < -self.max_daily_loss:
            logger.warning(f"Daily loss limit approached: {self.daily_pnl:.2f}")

    def get_portfolio_state(self) -> Dict:
        return {
            "portfolio_value": self.portfolio_value,
            "daily_pnl": self.daily_pnl,
            "open_positions": len(self.open_positions),
            "portfolio_risk": sum(pos.get("risk_contribution", 0) for pos in self.open_positions.values()),
            "max_daily_loss": self.max_daily_loss,
            "risk_budget_remaining": max(0, self.max_daily_loss + self.daily_pnl)
        }
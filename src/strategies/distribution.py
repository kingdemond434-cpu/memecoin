"""Detecting that the crowd is turning, before the price says so.

A trailing stop is a lagging exit by construction: it fires on price decline,
so it can only ever bank after the decline has already happened. On a token
whose top forms in four seconds, that is most of the giveback.

Distribution detection is a different question from rug detection, and it is
worth keeping the two apart. A rug is an act by an identifiable party --
liquidity pulled, mint abused, creator dumping inventory. Distribution is a
change in the *composition of demand*: the marginal buyer gets worse, the
skilled money leaves, purchase sizes shrink while purchase counts rise, sells
stop being absorbed. Nothing illegitimate happens. The token simply runs out
of people willing to pay more, and everyone holding finds out at the same time
unless something was watching the composition rather than the price.

The features here are deliberately all flow- and actor-composition-based, and
deliberately exclude price drawdown entirely. Including it would let the model
reach the right answer for the wrong reason during training -- drawdown is the
most predictive single feature of "price is about to be lower", and a model
that leans on it has learned to be a trailing stop with extra steps, which is
the thing being replaced. `test_price_collapse_alone_is_not_distribution`
pins that.

Calibration discipline matches the rest of the repository: an untrained
detector reports DATA_BLOCKED and no probability. The uncalibrated
``evidence_score`` is always available for shadow logging and research, and is
never to be read as a probability -- it is a weighted sum of normalised
signals, useful for ranking and for building the training set, not for sizing.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DISTRIBUTION_SCHEMA_VERSION = "v1"

DISTRIBUTION_FEATURE_NAMES = (
    "smart_wallet_exit_rate",
    "creator_linked_sell_share",
    "buyer_quality_decline",
    "avg_buy_size_decline",
    "avg_sell_size_growth",
    "buyer_count_growth_with_shrinking_size",
    "buy_acceleration_rollover",
    "sell_absorption_failure",
    "new_buyer_saturation",
    "social_chain_divergence",
)

# Horizons the detector answers over. Distribution on a newborn launch resolves
# in seconds, so a 30s-minimum horizon would be answering a question nobody
# asked.
DISTRIBUTION_HORIZONS = (1.0, 3.0, 10.0)


@dataclass
class DistributionReading:
    """What the detector currently believes, and how sure it is allowed to be."""

    status: str
    evidence_score: float = 0.0
    contributions: Dict[str, float] = field(default_factory=dict)
    features: Dict[str, float] = field(default_factory=dict)
    coverage: float = 0.0
    probabilities: Dict[float, float] = field(default_factory=dict)
    detail: str = ""

    @property
    def calibrated(self) -> bool:
        return self.status == "OK" and bool(self.probabilities)

    def probability(self, horizon: float) -> Optional[float]:
        """P(distribution begins within ``horizon``), or None if uncalibrated.

        Returns None rather than the evidence score, so that a caller cannot
        accidentally size a position off an uncalibrated number by forgetting
        to check ``calibrated``.
        """
        return self.probabilities.get(float(horizon))


def _notional(item: Dict[str, Any]) -> float:
    for key in ("notional_usd", "amount_usd", "sol_amount", "amount"):
        value = item.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
            return float(value)
    return 0.0


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _decline(recent: float, prior: float) -> float:
    """How far ``recent`` has fallen below ``prior``, in [0, 1]."""
    if prior <= 0:
        return 0.0
    return float(np.clip(1.0 - recent / prior, 0.0, 1.0))


def _growth(recent: float, prior: float) -> float:
    """How far ``recent`` has risen above ``prior``, saturating at 1."""
    if prior <= 0:
        return 0.0
    return float(np.clip(recent / prior - 1.0, 0.0, 1.0))


def distribution_features(
    observations: Sequence[Dict[str, Any]],
    as_of: float,
    recent_window: float = 15.0,
    prior_window: float = 60.0,
) -> Tuple[Dict[str, float], float]:
    """Point-in-time feature map plus the fraction of it actually observed.

    Windows are short on purpose. Distribution on a newborn Pump launch is a
    seconds-scale event, and a 5-minute window averages the turn away entirely.

    Returns ``(features, coverage)``. Coverage is the share of features backed
    by data that was actually present rather than defaulted to zero, so a
    caller can tell "no distribution" apart from "nothing was recorded".
    """
    eligible = [item for item in observations
                if float(item.get("timestamp", 0) or 0) <= as_of]
    trades = [item for item in eligible if item.get("type") == "trade"]
    recent = [item for item in trades if 0 <= as_of - float(item.get("timestamp", 0) or 0) <= recent_window]
    prior = [item for item in trades
             if recent_window < as_of - float(item.get("timestamp", 0) or 0) <= recent_window + prior_window]

    recent_buys = [item for item in recent if item.get("side") == "buy"]
    recent_sells = [item for item in recent if item.get("side") == "sell"]
    prior_buys = [item for item in prior if item.get("side") == "buy"]
    prior_sells = [item for item in prior if item.get("side") == "sell"]

    observed: Dict[str, bool] = {}
    features: Dict[str, float] = {name: 0.0 for name in DISTRIBUTION_FEATURE_NAMES}

    # 1. Skilled money leaving. Not "someone sold" -- wallets whose historical
    #    early-launch record is good, weighted by that record.
    skilled_sells = [item for item in recent_sells if item.get("wallet_skill") is not None]
    skilled_buys = [item for item in recent_buys if item.get("wallet_skill") is not None]
    if skilled_sells or skilled_buys:
        observed["smart_wallet_exit_rate"] = True
        sold = sum(float(item["wallet_skill"]) * _notional(item) for item in skilled_sells)
        bought = sum(float(item["wallet_skill"]) * _notional(item) for item in skilled_buys)
        if sold + bought > 0:
            features["smart_wallet_exit_rate"] = float(sold / (sold + bought))

    # 2. Selling by the creator or by wallets sharing its funder. The same
    #    dollar of selling means something very different depending on who
    #    signed it.
    linked_sells = [item for item in recent_sells if item.get("creator_linked") is not None]
    if linked_sells:
        observed["creator_linked_sell_share"] = True
        linked = sum(_notional(item) for item in linked_sells if item.get("creator_linked"))
        total_sell = sum(_notional(item) for item in recent_sells)
        if total_sell > 0:
            features["creator_linked_sell_share"] = float(linked / total_sell)

    # 3. The marginal buyer getting worse. A wave that is still growing in
    #    count while its average participant quality falls is late-stage.
    recent_quality = [float(item["wallet_skill"]) for item in recent_buys
                      if item.get("wallet_skill") is not None]
    prior_quality = [float(item["wallet_skill"]) for item in prior_buys
                     if item.get("wallet_skill") is not None]
    if recent_quality and prior_quality:
        observed["buyer_quality_decline"] = True
        features["buyer_quality_decline"] = _decline(_mean(recent_quality), _mean(prior_quality))

    # 4/5/6. Size structure. Shrinking buys alongside a rising buy count is the
    #    signature of retail arriving after the move, which is the cohort that
    #    cannot hold it up.
    recent_buy_sizes = [_notional(item) for item in recent_buys if _notional(item) > 0]
    prior_buy_sizes = [_notional(item) for item in prior_buys if _notional(item) > 0]
    recent_sell_sizes = [_notional(item) for item in recent_sells if _notional(item) > 0]
    prior_sell_sizes = [_notional(item) for item in prior_sells if _notional(item) > 0]

    if recent_buy_sizes and prior_buy_sizes:
        observed["avg_buy_size_decline"] = True
        size_decline = _decline(_mean(recent_buy_sizes), _mean(prior_buy_sizes))
        features["avg_buy_size_decline"] = size_decline

        recent_rate = len(recent_buys) / max(recent_window, 1e-9)
        prior_rate = len(prior_buys) / max(prior_window, 1e-9)
        count_growth = _growth(recent_rate, prior_rate)
        observed["buyer_count_growth_with_shrinking_size"] = True
        # The conjunction, not either alone: more buyers is bullish, smaller
        # buys is ambiguous, and more-but-smaller is the exhaustion pattern.
        features["buyer_count_growth_with_shrinking_size"] = float(count_growth * size_decline)

    if recent_sell_sizes and prior_sell_sizes:
        observed["avg_sell_size_growth"] = True
        features["avg_sell_size_growth"] = _growth(_mean(recent_sell_sizes), _mean(prior_sell_sizes))

    # 7. Rollover, not deceleration. Deceleration says the second derivative is
    #    negative; rollover says the first derivative is about to be.
    if len(recent) >= 4 and prior_buys:
        observed["buy_acceleration_rollover"] = True
        half = recent_window / 2.0
        newer = [item for item in recent_buys
                 if as_of - float(item.get("timestamp", 0) or 0) <= half]
        older = [item for item in recent_buys
                 if half < as_of - float(item.get("timestamp", 0) or 0) <= recent_window]
        newer_rate = len(newer) / max(half, 1e-9)
        older_rate = len(older) / max(half, 1e-9)
        prior_rate = len(prior_buys) / max(prior_window, 1e-9)
        # Both legs need a margin. Trade counts are small integers over short
        # windows, so a genuinely flat stream wobbles by 15-20% from bucket
        # boundaries alone; a rollover detector without a margin fires on that
        # noise continuously and is worth nothing.
        accelerating_before = older_rate > prior_rate * 1.25
        rolling_now = newer_rate < older_rate * 0.75
        if accelerating_before and rolling_now:
            features["buy_acceleration_rollover"] = _decline(newer_rate, older_rate)

    # 8. Absorption. In a healthy move a large sell is bought back within
    #    seconds. When it is not, the bid has gone.
    absorption = [item for item in eligible
                  if item.get("type") == "absorption" and item.get("recovered") is not None]
    if absorption:
        observed["sell_absorption_failure"] = True
        failed = sum(1 for item in absorption if not item.get("recovered"))
        features["sell_absorption_failure"] = float(failed / len(absorption))

    # 9. Saturation: the share of buyers who have never touched this token.
    #    Late in a move nearly every buyer is new, which means the pool of
    #    people left to convert is nearly empty.
    flagged = [item for item in recent_buys if item.get("first_time_buyer") is not None]
    if flagged:
        observed["new_buyer_saturation"] = True
        features["new_buyer_saturation"] = float(
            sum(1 for item in flagged if item.get("first_time_buyer")) / len(flagged))

    # 10. Divergence. Social attention still climbing while on-chain quality
    #     falls is the crowd arriving after the money left.
    social = [item for item in eligible if item.get("type") == "social_velocity"
              and item.get("velocity") is not None]
    if social and (recent_quality and prior_quality):
        observed["social_chain_divergence"] = True
        social.sort(key=lambda item: float(item.get("timestamp", 0) or 0))
        recent_social = [float(item["velocity"]) for item in social
                         if as_of - float(item.get("timestamp", 0) or 0) <= recent_window]
        prior_social = [float(item["velocity"]) for item in social
                        if recent_window < as_of - float(item.get("timestamp", 0) or 0) <= recent_window + prior_window]
        if recent_social and prior_social:
            social_growth = _growth(_mean(recent_social), _mean(prior_social))
            features["social_chain_divergence"] = float(
                social_growth * features["buyer_quality_decline"])

    coverage = len(observed) / len(DISTRIBUTION_FEATURE_NAMES)
    return features, coverage


# Weights for the uncalibrated evidence score. These are not probabilities and
# are not learned; they exist to rank states and to seed a training set. The
# ordering reflects how directly each signal implicates the marginal buyer:
# skilled money leaving and creator-linked selling are causes, size structure
# and saturation are symptoms.
_EVIDENCE_WEIGHTS: Dict[str, float] = {
    "smart_wallet_exit_rate": 0.20,
    "creator_linked_sell_share": 0.18,
    "buyer_quality_decline": 0.14,
    "sell_absorption_failure": 0.12,
    "buy_acceleration_rollover": 0.10,
    "buyer_count_growth_with_shrinking_size": 0.09,
    "avg_sell_size_growth": 0.07,
    "new_buyer_saturation": 0.05,
    "social_chain_divergence": 0.03,
    "avg_buy_size_decline": 0.02,
}


class DistributionDetector:
    """Answers P(distribution begins within 1s / 3s / 10s).

    Until a chronologically validated model is loaded it answers DATA_BLOCKED
    and supplies only the uncalibrated evidence score. That is not a
    limitation to be worked around: a probability that was never validated
    against outcomes, fed into a sizing rule, is a fabricated number with a
    position attached to it.
    """

    def __init__(self, min_coverage: float = 0.3):
        self.min_coverage = min_coverage
        self._model: Optional[Any] = None
        self._model_version: str = ""
        self._feature_names: Tuple[str, ...] = DISTRIBUTION_FEATURE_NAMES

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def load_model(self, model: Any, feature_names: Sequence[str], version: str) -> bool:
        """Adopt a trained model, refusing one built on different features.

        A model trained under one feature order and served under another is
        silently wrong in a way no test of either component alone can catch.
        """
        if tuple(feature_names) != DISTRIBUTION_FEATURE_NAMES:
            logger.warning("distribution model rejected: feature names do not match schema %s",
                           DISTRIBUTION_SCHEMA_VERSION)
            return False
        if not hasattr(model, "predict_proba"):
            logger.warning("distribution model rejected: no predict_proba")
            return False
        self._model, self._model_version = model, str(version)
        return True

    def evaluate(self, observations: Sequence[Dict[str, Any]], as_of: float) -> DistributionReading:
        features, coverage = distribution_features(observations, as_of)
        contributions = {name: _EVIDENCE_WEIGHTS[name] * features[name]
                         for name in DISTRIBUTION_FEATURE_NAMES}
        evidence = float(sum(contributions.values()))

        if coverage < self.min_coverage:
            return DistributionReading(
                status="DATA_BLOCKED", evidence_score=evidence, contributions=contributions,
                features=features, coverage=coverage,
                detail=f"coverage {coverage:.2f} below {self.min_coverage:.2f}; "
                       "too little of the flow was observed to say anything",
            )
        if not self.is_trained:
            return DistributionReading(
                status="DATA_BLOCKED", evidence_score=evidence, contributions=contributions,
                features=features, coverage=coverage,
                detail="no chronologically validated distribution model",
            )

        vector = np.asarray([[features[name] for name in DISTRIBUTION_FEATURE_NAMES]], dtype=float)
        probabilities: Dict[float, float] = {}
        try:
            raw = self._model.predict_proba(vector)[0]
        except Exception as exc:  # pragma: no cover - defensive
            return DistributionReading(
                status="DATA_BLOCKED", evidence_score=evidence, contributions=contributions,
                features=features, coverage=coverage, detail=f"model inference failed: {exc}",
            )
        base = float(raw[1]) if len(raw) > 1 else float(raw[0])
        for horizon in DISTRIBUTION_HORIZONS:
            # The model is trained on the shortest horizon; longer horizons are
            # the survival extension of the same instantaneous rate, the same
            # projection the rug-hazard model uses.
            rate = -math.log(max(1e-9, 1.0 - min(base, 1 - 1e-9))) / DISTRIBUTION_HORIZONS[0]
            probabilities[horizon] = float(1.0 - math.exp(-rate * horizon))
        return DistributionReading(
            status="OK", evidence_score=evidence, contributions=contributions,
            features=features, coverage=coverage, probabilities=probabilities,
            detail=f"model {self._model_version}",
        )

    @staticmethod
    def top_contributors(reading: DistributionReading, limit: int = 3) -> List[Tuple[str, float]]:
        """The signals actually driving this reading, for forensics and logs."""
        ranked = sorted(reading.contributions.items(), key=lambda item: item[1], reverse=True)
        return [item for item in ranked if item[1] > 0][:limit]

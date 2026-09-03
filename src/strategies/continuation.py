"""Will this position double again from where it is now?

The desk had a `continuation` number and it was `max(p_5x, p_10x)` -- both
unconditional survival probabilities measured FROM LAUNCH. At 8x, `p_5x` is
answering "will this reach 5x", a question the position settled an hour ago,
and the answer is nearly one. The number that decides whether a runner is held
through a drawdown was therefore approximately constant above 5x and carried
no information about the only thing that matters there: whether there is more
to come.

The right quantity is conditional. For the maximum multiple M a launch will
ever reach, the model's survival curve gives S(x) = P(M >= x), and

    P(M >= 2m | M >= m) = S(2m) / S(m)

is exactly "will it double from here", read off the curve the model already
produces. It is defined at every multiple, it means the same thing at 1.5x as
at 80x, and it needs no new head -- only the existing one, read correctly.

Two things this module refuses to do, both of which are how a conditional
probability becomes false conviction:

**It will not divide two uncalibrated numbers.** A raw gradient-boosting score
is an ordering. A ratio of two orderings is not a probability, and it is a
number of exactly the right shape to be believed.

**It will not divide two numbers built from a handful of examples.** If the
50x head saw eleven positives, S(50) is noise, and S(100)/S(50) is a ratio of
noise that will happily read 0.9. Heads below a positive-count floor do not
participate, so the curve simply ends and the answer is DATA_BLOCKED.

DATA_BLOCKED is the useful answer here. Every consumer treats an absent
continuation as no conviction, which restores the ordinary trailing stop --
the behaviour the desk has today, arrived at deliberately rather than by
accident.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.strategies.multihead_predictor import (
    PredictionTarget, SURVIVAL_LEVELS)

CONTINUATION_SCHEMA_VERSION = "v1"

#: Every launch reaches 1x by definition -- it is the price it opened at --
#: so the curve is anchored there rather than starting at the 2x head. Without
#: the anchor there is no segment covering a position at 1.4x, which is where
#: conviction is worth the most because nothing has been given back yet.
CURVE_ANCHOR: Tuple[float, float] = (1.0, 1.0)

#: A head that saw fewer positives than this does not join the curve.
#: Thirty is not a magic number and is not claimed to be one; it is the point
#: below which a binomial proportion's own confidence interval is wider than
#: the differences this ratio is being asked to resolve. The gauntlet is what
#: will eventually replace it with a measured floor.
DEFAULT_MIN_POSITIVES = 30

#: Below this the denominator is not a probability, it is a rounding artefact,
#: and the ratio it produces is unbounded in the flattering direction.
DEFAULT_MIN_CONDITIONING_SURVIVAL = 1e-4

#: How much further the position must go to count as "continuing". Two, so the
#: question is always "double from here" regardless of where here is.
DEFAULT_HORIZON = 2.0


@dataclass(frozen=True)
class Continuation:
    """One conditional continuation reading, with its own provenance."""

    status: str
    probability: Optional[float] = None
    from_multiple: float = 0.0
    target_multiple: float = 0.0
    #: True only when every head the reading touched was isotonically
    #: calibrated AND cleared the positive-count floor. The monster override
    #: is reachable from nothing else.
    calibrated: bool = False
    survival_from: Optional[float] = None
    survival_target: Optional[float] = None
    basis: str = ""
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "OK" and self.probability is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": CONTINUATION_SCHEMA_VERSION,
            "status": self.status, "probability": self.probability,
            "from_multiple": self.from_multiple,
            "target_multiple": self.target_multiple,
            "calibrated": self.calibrated,
            "survival_from": self.survival_from,
            "survival_target": self.survival_target,
            "basis": self.basis, "detail": self.detail,
        }


def _blocked(reason: str, **fields: Any) -> Continuation:
    return Continuation(status="DATA_BLOCKED", detail=reason, **fields)


class ContinuationModel:
    """Reads a conditional continuation probability off the survival curve."""

    def __init__(self, *,
                 min_positives: int = DEFAULT_MIN_POSITIVES,
                 min_conditioning_survival: float = DEFAULT_MIN_CONDITIONING_SURVIVAL,
                 horizon: float = DEFAULT_HORIZON):
        self.min_positives = int(min_positives)
        self.min_conditioning_survival = float(min_conditioning_survival)
        self.horizon = float(horizon)

    # -- the curve ---------------------------------------------------------

    def usable_levels(self, predictor: Any) -> List[Tuple[float, PredictionTarget]]:
        """Survival rungs whose head is calibrated and saw enough positives.

        Walked in order and STOPPED at the first unusable rung rather than
        skipped past. The rungs are nested -- P(>=100x) is a subset of
        P(>=50x) -- so bridging over a rung the model cannot support would
        interpolate a segment across a gap the evidence does not cover.
        """
        usable: List[Tuple[float, PredictionTarget]] = []
        for target, level in SURVIVAL_LEVELS:
            if not self._head_usable(predictor, target):
                break
            usable.append((float(level), target))
        return usable

    def _head_usable(self, predictor: Any, target: PredictionTarget) -> bool:
        checker = getattr(predictor, "is_calibrated", None)
        if not callable(checker) or not checker(target):
            return False
        counter = getattr(predictor, "head_positives", None)
        if not callable(counter):
            return False
        positives = counter(target)
        # None means the bundle predates the record. Unknown is not enough.
        return positives is not None and positives >= self.min_positives

    def curve(self, predictor: Any, prediction: Any
              ) -> List[Tuple[float, float]]:
        """[(multiple, P(max >= multiple))], anchored at (1.0, 1.0).

        Clamped monotone decreasing on the way out. The predictor enforces
        nested monotonicity already; this does not trust it, because a curve
        that ticks upward makes a conditional probability exceed one and the
        consumer of that number is an override on a stop loss.
        """
        points: List[Tuple[float, float]] = [CURVE_ANCHOR]
        previous = CURVE_ANCHOR[1]
        for level, target in self.usable_levels(predictor):
            raw = getattr(prediction, target.value, None)
            if raw is None:
                break
            value = min(previous, max(0.0, float(raw)))
            points.append((level, value))
            previous = value
        return points

    @staticmethod
    def survival_at(curve: Sequence[Tuple[float, float]], multiple: float
                    ) -> Tuple[Optional[float], str]:
        """S(multiple), interpolated log-linearly, with the segment named.

        Log-log because the launch tail is a power law -- P(>=m) = k * m^-a is
        a straight line in these coordinates -- and linear interpolation
        between 10x and 20x on a raw axis overstates the middle of that
        segment by a factor that grows with the span.

        Never extrapolated past the last usable rung. Beyond it the model has
        no evidence, and a conditional probability read off an extrapolation
        is the model's opinion about its own opinion.
        """
        if not curve or multiple <= 0:
            return None, "no curve"
        if multiple <= curve[0][0]:
            return curve[0][1], f"at or below {curve[0][0]:g}x"
        if multiple > curve[-1][0]:
            return None, f"beyond the last measured rung ({curve[-1][0]:g}x)"
        for (low_x, low_y), (high_x, high_y) in zip(curve, curve[1:]):
            if multiple > high_x:
                continue
            basis = f"{low_x:g}x-{high_x:g}x"
            if low_y <= 0.0 or high_y <= 0.0:
                # log(0) has no interpolation. A zero rung is a real answer --
                # the head never saw its class -- and everything at or past it
                # is zero too.
                return (0.0 if multiple > low_x else low_y), basis
            span = math.log(high_x) - math.log(low_x)
            if span <= 0:
                return low_y, basis
            weight = (math.log(multiple) - math.log(low_x)) / span
            log_y = ((1.0 - weight) * math.log(low_y)
                     + weight * math.log(high_y))
            return math.exp(log_y), basis
        return curve[-1][1], f"at {curve[-1][0]:g}x"

    # -- the reading -------------------------------------------------------

    def evaluate(self, predictor: Any, prediction: Any, multiple: float, *,
                 horizon: Optional[float] = None) -> Continuation:
        """P(the position reaches `horizon` times its current multiple).

        Conditional on it having already reached where it is, which is the
        only version of the question that is still open.
        """
        horizon = float(self.horizon if horizon is None else horizon)
        if prediction is None:
            return _blocked("no prediction for this position")
        if not getattr(predictor, "_is_trained", False):
            return _blocked("the predictor is not trained")
        if multiple <= 0:
            return _blocked(f"a position cannot be at {multiple}x")
        curve = self.curve(predictor, prediction)
        if len(curve) < 2:
            return _blocked(
                "no survival head is both calibrated and above the "
                f"{self.min_positives}-positive floor; the curve is only its "
                "anchor")

        target_multiple = multiple * horizon
        survival_from, from_basis = self.survival_at(curve, multiple)
        survival_target, target_basis = self.survival_at(curve, target_multiple)
        common = dict(from_multiple=float(multiple),
                      target_multiple=float(target_multiple),
                      survival_from=survival_from,
                      survival_target=survival_target,
                      basis=f"{from_basis} -> {target_basis}")
        if survival_from is None:
            return _blocked(
                f"the curve does not reach {multiple:g}x ({from_basis})",
                **common)
        if survival_target is None:
            return _blocked(
                f"the curve does not reach {target_multiple:g}x "
                f"({target_basis})", **common)
        if survival_from < self.min_conditioning_survival:
            # Dividing by this would not be a conditional probability, it
            # would be a ratio of two numbers the model rounds to nothing.
            return _blocked(
                f"P(>={multiple:g}x) is {survival_from:.2e}, below the "
                f"{self.min_conditioning_survival:.0e} conditioning floor; a "
                "ratio of two rounding artefacts is not a conviction",
                **common)
        probability = min(1.0, max(0.0, survival_target / survival_from))
        return Continuation(status="OK", probability=probability,
                            calibrated=True, **common)


#: Used when a caller has no configured model. A bare default is safe here in
#: a way it usually is not: every gate in `evaluate` is a REFUSAL, so a
#: predictor that cannot answer `is_calibrated` or `head_positives` yields
#: DATA_BLOCKED and grants nothing. The fallback cannot be more permissive
#: than a configured model, only less informed.
DEFAULT_MODEL = ContinuationModel()


def position_multiple(position: Dict[str, Any]) -> float:
    """Where this position currently is, from whichever field carries it."""
    try:
        multiple = float(position.get("current_multiple", 0.0) or 0.0)
    except (TypeError, ValueError):
        multiple = 0.0
    if multiple > 0:
        return multiple
    try:
        entry = float(position.get("entry_price", 0.0) or 0.0)
        price = float(position.get("current_price", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return (price / entry) if entry > 0 else 0.0


def read_position_continuation(model: Optional[ContinuationModel],
                               predictor: Any,
                               position: Dict[str, Any]) -> Continuation:
    """One conditional reading for an open position.

    Reads `prediction_object` -- the refreshed prediction, not the entry-time
    one. Holding a runner through a drawdown on entry-time evidence is holding
    on a belief that every trade since entry may have contradicted, and those
    trades are exactly what separates a 20x from a distribution phase.
    """
    return (model or DEFAULT_MODEL).evaluate(
        predictor, position.get("prediction_object"),
        position_multiple(position))

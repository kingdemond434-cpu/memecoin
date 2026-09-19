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

#: Rungs needed before a power law may be fitted. Two points always fit a
#: line perfectly and say nothing about whether it IS a line; three is the
#: smallest number that can disagree with the model.
MIN_TAIL_FIT_RUNGS = 3

#: How far past the last MEASURED rung the fitted tail may be trusted, as a
#: multiple of that rung. A launch tail is a power law over the range anyone
#: has measured; eight times beyond the last observation is an opinion about
#: the model rather than a reading from it.
DEFAULT_TAIL_REACH = 8.0

#: Worst root-mean-square residual, in log survival, that still counts as "a
#: power law describes these rungs". Above it the curve is some other shape
#: and extrapolating along a straight line would invent the part that
#: matters most.
DEFAULT_MAX_TAIL_RESIDUAL = 0.35


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
    #: True when this came from a power law FITTED to the measured rungs
    #: rather than read between two of them. Carried so a consumer can treat
    #: it as the weaker evidence it is, and so the gauntlet can score
    #: decisions made on it separately from decisions made on measurement.
    extrapolated: bool = False
    tail_alpha: Optional[float] = None
    tail_residual: Optional[float] = None

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
            "extrapolated": self.extrapolated,
            "tail_alpha": self.tail_alpha,
            "tail_residual": self.tail_residual,
        }


def _blocked(reason: str, **fields: Any) -> Continuation:
    return Continuation(status="DATA_BLOCKED", detail=reason, **fields)


class ContinuationModel:
    """Reads a conditional continuation probability off the survival curve."""

    def __init__(self, *,
                 min_positives: int = DEFAULT_MIN_POSITIVES,
                 min_conditioning_survival: float = DEFAULT_MIN_CONDITIONING_SURVIVAL,
                 horizon: float = DEFAULT_HORIZON,
                 extrapolate_tail: bool = True,
                 tail_reach: float = DEFAULT_TAIL_REACH,
                 max_tail_residual: float = DEFAULT_MAX_TAIL_RESIDUAL):
        self.min_positives = int(min_positives)
        self.min_conditioning_survival = float(min_conditioning_survival)
        self.horizon = float(horizon)
        self.extrapolate_tail = bool(extrapolate_tail)
        self.tail_reach = float(tail_reach)
        self.max_tail_residual = float(max_tail_residual)

    # -- the fitted tail ---------------------------------------------------

    def fit_tail(self, curve: Sequence[Tuple[float, float]]
                 ) -> Optional[Tuple[float, float]]:
        """(alpha, rms residual) for S(m) = k * m ** -alpha, or None.

        The head of the curve is measured and the tail is where the money is,
        and those are the same sentence read twice. On a 32,542-launch corpus
        the 50x head sees thirteen positives and the 100x head sees five, so
        both fall below the positive floor and the curve simply ends at 20x --
        which means the conditional continuation goes DATA_BLOCKED above 10x,
        and conviction can never engage on the runner it exists for. A 100x
        passes that point and the model goes blind at exactly the wrong
        instant.

        More data does not fix it soon: thirty positives at 50x needs about
        2.3x this corpus, and at 100x about 6.5x. That is months.

        So the shape is fitted instead of the level being guessed. A launch
        tail is a power law -- straight in log-log -- and the rungs that ARE
        reliable determine its exponent by least squares. Refused on fewer
        than three rungs, because two points fit a line perfectly and cannot
        disagree with the model, and refused when the residual says these
        rungs are not a straight line at all.
        """
        # The anchor is excluded. S(1) = 1 is definitional -- every launch
        # reaches the price it opened at -- and it does not sit on the power
        # law, so including it drags the exponent badly: on a curve whose
        # true alpha is 1.23 it fits 2.02 with a residual of 0.85, which the
        # guard below rejects outright.
        points = [(math.log(level), math.log(value))
                  for level, value in curve
                  if level > CURVE_ANCHOR[0] and value > 0]
        if len(points) < MIN_TAIL_FIT_RUNGS:
            return None
        count = len(points)
        mean_x = sum(x for x, _ in points) / count
        mean_y = sum(y for _, y in points) / count
        variance = sum((x - mean_x) ** 2 for x, _ in points)
        if variance <= 0:
            return None
        slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / variance
        intercept = mean_y - slope * mean_x
        residual = math.sqrt(
            sum((y - (slope * x + intercept)) ** 2 for x, y in points) / count)
        alpha = -slope
        if alpha <= 0:
            # A curve that RISES with the multiple is not a survival curve.
            return None
        return alpha, residual

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

    def report(self, predictor: Any, prediction: Any = None) -> Dict[str, Any]:
        """Which rungs are usable, and therefore what this can answer about.

        The practical question after a training run is not "did it train" but
        "how far up the curve can it see". A model whose last usable rung is
        10x can answer for positions below 5x and is DATA_BLOCKED above them
        -- so conviction never engages on exactly the runners it was built
        for, and nothing else on the status page would say so.
        """
        if not getattr(predictor, "_is_trained", False):
            return {"schema": CONTINUATION_SCHEMA_VERSION,
                    "status": "DATA_BLOCKED", "detail": "predictor not trained"}
        usable = self.usable_levels(predictor)
        rungs = []
        for target, level in SURVIVAL_LEVELS:
            checker = getattr(predictor, "is_calibrated", None)
            counter = getattr(predictor, "head_positives", None)
            rungs.append({
                "level": float(level), "head": target.value,
                "calibrated": bool(callable(checker) and checker(target)),
                "positives": (counter(target) if callable(counter) else None),
                "usable": any(item[0] == float(level) for item in usable),
            })
        top = usable[-1][0] if usable else None
        fit = self.fit_tail(self.curve(predictor, prediction)) if (
            self.extrapolate_tail and prediction is not None) else None
        reach = top if top is None else (
            top * self.tail_reach
            if fit and fit[1] <= self.max_tail_residual else top)
        answerable = None if reach is None else reach / self.horizon
        return {
            "schema": CONTINUATION_SCHEMA_VERSION,
            "status": "OK" if usable else "DATA_BLOCKED",
            "usable_rungs": [item[0] for item in usable],
            "highest_usable_multiple": top,
            "highest_answerable_survival": reach,
            "tail_alpha": (fit[0] if fit else None),
            "tail_residual": (fit[1] if fit else None),
            "answerable_up_to_multiple": answerable,
            "min_positives": self.min_positives,
            "horizon": self.horizon,
            "rungs": rungs,
            "detail": ("" if usable else
                       "no survival head is both calibrated and above the "
                       f"{self.min_positives}-positive floor; conviction "
                       "cannot be granted and every runner exits on the "
                       "ordinary trail"),
        }

    def survival(self, curve: Sequence[Tuple[float, float]], multiple: float,
                 fit: Optional[Tuple[float, float]] = None
                 ) -> Tuple[Optional[float], str, bool]:
        """S(multiple), extending past the last rung along the fitted tail.

        Returns (value, basis, extrapolated). Interpolation inside the
        measured range is untouched and never marked extrapolated; the fitted
        law is used only beyond the last rung, only when the fit describes
        the measured rungs, and only as far as `tail_reach` past them.
        """
        value, basis = self.survival_at(curve, multiple)
        if value is not None or not curve:
            return value, basis, False
        last_x, last_y = curve[-1]
        if not self.extrapolate_tail or multiple <= last_x or last_y <= 0:
            return None, basis, False
        if multiple > last_x * self.tail_reach:
            return None, (f"{multiple:g}x is more than {self.tail_reach:g}x "
                          f"past the last measured rung ({last_x:g}x)"), False
        if fit is None:
            fit = self.fit_tail(curve)
        if fit is None:
            return None, f"{basis}; no power law could be fitted", False
        alpha, residual = fit
        if residual > self.max_tail_residual:
            return None, (f"{basis}; the measured rungs are not a power law "
                          f"(residual {residual:.2f})"), False
        # S(m) = S(last) * (m / last) ** -alpha, anchored on the last rung the
        # desk actually measured rather than on the fitted intercept, so the
        # extrapolation starts from an observation.
        return (last_y * (multiple / last_x) ** -alpha,
                f"fitted tail past {last_x:g}x (alpha {alpha:.2f})", True)

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
        fit = self.fit_tail(curve) if self.extrapolate_tail else None
        survival_from, from_basis, from_extrapolated = self.survival(
            curve, multiple, fit)
        survival_target, target_basis, target_extrapolated = self.survival(
            curve, target_multiple, fit)
        extrapolated = bool(from_extrapolated or target_extrapolated)
        common = dict(from_multiple=float(multiple),
                      target_multiple=float(target_multiple),
                      survival_from=survival_from,
                      survival_target=survival_target,
                      extrapolated=extrapolated,
                      tail_alpha=(fit[0] if fit else None),
                      tail_residual=(fit[1] if fit else None),
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

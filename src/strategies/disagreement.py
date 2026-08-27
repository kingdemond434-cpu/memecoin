"""Model disagreement as an explicit uncertainty that shrinks the bet.

The desk carries several independent readings of the same launch: the monster
model on tail potential, the actor graph on who is buying, source intelligence
on where it came from, the hazard model on whether it is about to die, the
tradeability frontier on whether it can be left, the distribution detector on
whether someone is selling into the buying. When they agree, the picture is one
thing seen six ways. When they do not, something is wrong with the picture --
and which of them is wrong is exactly what cannot be known at the time.

The usual answer is a vote: five of six agree, so trade. That is wrong twice
over. It discards the information in the disagreement, and it produces the same
size whether the views were unanimous or barely carried -- so the position is
largest precisely when the evidence is most contested, because the bullish
views are loudest there.

The right answer is that disagreement is variance. The E[log W] sizing already
knows what to do with variance: a distribution whose mean is uncertain is
sized as if its mean were worse, because the growth-optimal fraction falls in
the parameter uncertainty as well as in the outcome uncertainty. So this
produces a sigma, and sizing shrinks:

    q_effective = q_kelly * shrink(sigma)

with `shrink` at 1 under unanimity and falling toward a floor as the views
scatter. Nothing is rejected for disagreeing. A contested launch is a smaller
position, not an abandoned one -- and a launch every view calls good is the one
that gets full size, which is the ordering that was backwards before.

Two things it deliberately refuses.

An ABSENT view is not a disagreeing view, and it is not an agreeing one
either. A model that could not answer is unmeasured; counting silence as
consent is how a desk with five broken models trades as if it had six
confirmations. Below a floor of participating views the reading is
DATA_BLOCKED and sizing is left to its own conservatism.

And a view that is CONFIDENTLY BEARISH is not the same as views that scatter.
Unanimous pessimism is agreement, and it belongs to Q, which will decline the
trade on its merits. This measures dispersion, not level.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DISAGREEMENT_SCHEMA_VERSION = "v1"

# Fewest participating views before dispersion means anything. Two views
# agreeing is not consensus, it is a coincidence with a sample size of two.
DEFAULT_MIN_VIEWS = 3

# The smallest multiple sizing may be shrunk to. Not zero: total disagreement
# is a reason to be small, and Q remains the thing that decides whether to be
# in at all. A shrink that could reach zero would be a second veto competing
# with the objective.
DEFAULT_FLOOR = 0.25

# How fast size falls in dispersion. At sigma = 0.5 -- views spread across the
# whole range -- this puts the multiplier near the floor.
DEFAULT_SENSITIVITY = 3.0


@dataclass(frozen=True)
class View:
    """One model's reading of one launch, on a common [0, 1] bullish scale.

    ``weight`` is how much this view is worth when it speaks, not how bullish
    it is. A calibrated model and an uncalibrated heuristic both get a vote
    here; they should not get the same vote.
    """

    name: str
    value: Optional[float]
    weight: float = 1.0
    status: str = "OK"
    detail: str = ""

    @property
    def participates(self) -> bool:
        return (self.status == "OK" and self.value is not None
                and math.isfinite(float(self.value)) and self.weight > 0)


@dataclass
class DisagreementReading:
    """How much the views scatter, and what that does to size."""

    status: str
    sigma: float = 0.0
    mean: float = 0.0
    shrink: float = 1.0
    participating: int = 0
    absent: int = 0
    views: Dict[str, Optional[float]] = field(default_factory=dict)
    # The pair furthest apart. Named because "the models disagree" is not
    # actionable and "monster says 0.9, distribution says 0.1" is.
    widest: Tuple[str, str, float] = ("", "", 0.0)
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "OK"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": DISAGREEMENT_SCHEMA_VERSION, "status": self.status,
            "sigma": round(self.sigma, 6), "mean": round(self.mean, 6),
            "shrink": round(self.shrink, 6),
            "participating": self.participating, "absent": self.absent,
            "views": {name: (round(value, 4) if value is not None else None)
                      for name, value in self.views.items()},
            "widest_disagreement": {"a": self.widest[0], "b": self.widest[1],
                                    "gap": round(self.widest[2], 4)},
            "detail": self.detail,
        }


class DisagreementModel:
    """Turns a set of views into a sizing multiplier."""

    def __init__(self, *, min_views: int = DEFAULT_MIN_VIEWS,
                 floor: float = DEFAULT_FLOOR,
                 sensitivity: float = DEFAULT_SENSITIVITY):
        self.min_views = max(2, int(min_views))
        self.floor = min(1.0, max(0.0, float(floor)))
        self.sensitivity = max(0.0, float(sensitivity))

    def read(self, views: Sequence[View]) -> DisagreementReading:
        """Dispersion across the participating views, and the shrink it earns."""
        present = [view for view in views if view.participates]
        absent = [view for view in views if not view.participates]
        recorded = {view.name: (float(view.value) if view.participates else None)
                    for view in views}
        if len(present) < self.min_views:
            # Silence is not consent. A desk with five broken models must not
            # trade as though it had six confirmations.
            return DisagreementReading(
                status="DATA_BLOCKED", participating=len(present),
                absent=len(absent), views=recorded, shrink=1.0,
                detail=(f"{len(present)} views answered; {self.min_views} needed "
                        "before dispersion means anything"))

        total_weight = sum(view.weight for view in present)
        mean = sum(view.weight * float(view.value) for view in present) / total_weight
        variance = sum(view.weight * (float(view.value) - mean) ** 2
                       for view in present) / total_weight
        sigma = math.sqrt(max(0.0, variance))

        widest = ("", "", 0.0)
        for index, first in enumerate(present):
            for second in present[index + 1:]:
                gap = abs(float(first.value) - float(second.value))
                if gap > widest[2]:
                    widest = (first.name, second.name, gap)

        return DisagreementReading(
            status="OK", sigma=sigma, mean=mean, shrink=self.shrink_for(sigma),
            participating=len(present), absent=len(absent), views=recorded,
            widest=widest,
            detail=(f"{len(present)} views, sigma {sigma:.3f}"
                    + (f", widest {widest[0]} vs {widest[1]} ({widest[2]:.2f})"
                       if widest[2] > 0 else "")))

    def shrink_for(self, sigma: float) -> float:
        """Size multiplier for this much dispersion.

        Exponential rather than linear, and floored. Linear would let moderate
        disagreement barely register while total disagreement still sized a
        third of the book; the exponential puts the cost where the
        disagreement actually is.
        """
        if self.sensitivity <= 0:
            return 1.0
        raw = math.exp(-self.sensitivity * max(0.0, float(sigma)))
        return float(self.floor + (1.0 - self.floor) * raw)


def views_from_intelligence(intelligence: Dict[str, Any],
                            prediction: Any = None) -> List[View]:
    """Read the desk's own reports as views on a common bullish scale.

    Everything is mapped so that HIGHER MEANS MORE BULLISH, including the ones
    that natively measure danger -- a hazard of 0.9 becomes a view of 0.1.
    Without that flip the variance would measure the mixture of conventions
    rather than the disagreement, and two models saying the same thing in
    opposite units would look like the widest disagreement in the set.
    """
    views: List[View] = []

    def add(name: str, value: Optional[float], weight: float = 1.0,
            status: str = "OK", invert: bool = False) -> None:
        if value is None or status != "OK":
            views.append(View(name=name, value=None, weight=weight,
                              status="DATA_BLOCKED"))
            return
        bounded = min(1.0, max(0.0, float(value)))
        views.append(View(name=name, value=(1.0 - bounded) if invert else bounded,
                          weight=weight))

    monster = intelligence.get("monster") or {}
    add("monster", monster.get("probability"), weight=1.0,
        status=str(monster.get("status", "DATA_BLOCKED")))

    flow = (intelligence.get("actor") or {}).get("smart_flow") or {}
    add("actor", flow.get("discount"), weight=1.0,
        status=str(flow.get("status", "DATA_BLOCKED")))

    swarm = (intelligence.get("actor") or {}).get("swarm") or {}
    add("swarm", swarm.get("probability"), weight=0.8,
        status=str(swarm.get("status", "DATA_BLOCKED")))

    source = intelligence.get("source") or {}
    add("source", source.get("credibility"), weight=0.8,
        status=str(source.get("status", "DATA_BLOCKED")))

    hazard = intelligence.get("hazard") or {}
    add("rug", hazard.get("hazard_30s"), weight=1.2,
        status=str(hazard.get("status", "DATA_BLOCKED")), invert=True)

    capacity = intelligence.get("capacity") or {}
    add("capacity", capacity.get("ratio"), weight=1.0,
        status=str(capacity.get("status", "DATA_BLOCKED")))

    distribution = intelligence.get("distribution") or {}
    add("distribution", distribution.get("probability"), weight=1.0,
        status=str(distribution.get("status", "DATA_BLOCKED")), invert=True)

    return views

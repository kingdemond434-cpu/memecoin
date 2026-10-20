"""When the desk cannot look at every launch, choose WHICH it drops.

Under a burst -- a trending narrative, a coordinated wave, the minute after
a major listing -- launches arrive faster than the desk can evaluate them,
and something has to give. What used to give was whatever arrived last:

    if len(self._candidate_pipelines) >= max_candidate_pipelines:
        record DATA_BLOCKED_candidate_pipeline_saturated
        return

That is a queue discipline, not a decision. It sheds by ARRIVAL ORDER, so
during the exact minute when the best launch of the day is most likely to
appear, the desk's rule for handling it is "were you 99th or 101st". A
burst is not noise to be survived; it is when the opportunities are.

The alternative is to shed by what the launch is worth looking at, using
only what is free at dispatch -- the deployer's record, whether a source
named it, whether the venue is verified. None of that is a prediction of
the outcome. It is a prediction of whether SPENDING A PIPELINE SLOT on this
launch is worth more than spending it on the median launch, which is a much
easier question and the only one that has to be answered in microseconds.

Three properties this has to keep:

**It never sheds when there is room.** Below saturation every candidate is
admitted, whatever its priority. A shedding rule that starts filtering
early is a policy change wearing a capacity argument.

**The bar rises with pressure, not with opinion.** At 80% full the desk
declines the bottom decile; at 100% it declines everything below the
median. The quantile comes from the priorities actually seen recently, so
a quiet hour cannot inherit a busy hour's bar.

**A shed launch is a recorded decision.** Not a dropped one. It carries its
priority and the bar it failed, so the counterfactual corpus can later ask
the only question that matters about this whole mechanism: were the
launches it shed the ones that went on to run? If they were, the priors are
wrong and this makes that visible instead of invisible.
"""

from __future__ import annotations

import bisect
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional
from collections import deque

logger = logging.getLogger(__name__)

LOAD_SHEDDING_SCHEMA_VERSION = "v1"

#: Below this fraction of capacity nothing is shed, at any priority.
SHED_FLOOR = 0.80

#: The quantile declined at full capacity. Half: at the point where the desk
#: is certainly dropping something, it drops the below-median half rather
#: than the most recent half.
MAX_SHED_QUANTILE = 0.50

#: How many recent priorities the bar is computed from. Long enough that one
#: unusual launch cannot move the bar, short enough that the bar tracks the
#: hour the desk is actually in.
PRIORITY_WINDOW = 2048

#: Until this many priorities have been seen there is no distribution to
#: take a quantile of, and nothing is shed. An empty desk must not start by
#: refusing launches on the strength of a sample of four.
MIN_PRIORITIES = 64


@dataclass
class ShedDecision:
    admitted: bool
    priority: float
    bar: Optional[float]
    utilisation: float
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "admitted": self.admitted,
            "priority": round(self.priority, 4),
            "bar": round(self.bar, 4) if self.bar is not None else None,
            "utilisation": round(self.utilisation, 3),
            "reason": self.reason,
        }


class EconomicLoadShedder:
    """Admits candidates while there is room, and by worth when there is not."""

    def __init__(self, capacity: int,
                 shed_floor: float = SHED_FLOOR,
                 max_quantile: float = MAX_SHED_QUANTILE,
                 window: int = PRIORITY_WINDOW):
        self.capacity = max(1, int(capacity))
        self.shed_floor = float(shed_floor)
        self.max_quantile = float(max_quantile)
        self._priorities: Deque[float] = deque(maxlen=int(window))
        self.admitted = 0
        self.shed = 0
        self.shed_by_priority: List[float] = []
        self.peak_utilisation = 0.0
        self.last_shed_at = 0.0

    # --- the priors ------------------------------------------------------

    def priority(self, features: Dict[str, Any]) -> float:
        """How much a pipeline slot spent here is worth, from free evidence.

        Deliberately NOT a prediction of the launch's return. It is a
        prediction of whether this launch repays a slot better than the
        median launch does, and the inputs are exactly those already in hand
        at dispatch: nothing here may cost a network call, because the whole
        point is to decide before the expensive path begins.

        Unknown inputs contribute nothing rather than a default. A launch
        the desk knows nothing about scores the base rate, which is correct:
        most launches are exactly that, and treating ignorance as a negative
        would shed every launch from a deployer never seen before -- which
        is most of the ones worth catching.
        """
        score = 0.0
        # A deployer the desk has scored before. Signed, because a known-bad
        # deployer is worth a slot LESS than an unknown one, not the same.
        deployer = features.get("deployer_score")
        if deployer is not None:
            try:
                score += max(-1.0, min(1.0, float(deployer)))
            except (TypeError, ValueError):
                pass
        # Somebody named it before it launched. The strongest free signal
        # there is, because it is the only one that is not derivable from
        # the launch transaction itself.
        if features.get("named_by_source"):
            score += 1.0
        if features.get("named_actor"):
            score += 1.0
        # A venue whose decoder has been verified against real transactions.
        # An unverified venue's launches are evidence about the REGISTRY,
        # not trading candidates, so a slot spent there buys less.
        if features.get("venue_verified"):
            score += 0.4
        # Funding structure visible in the launch transaction itself.
        funders = features.get("funding_wallets") or ()
        if funders:
            score += 0.15
        # A deployer with prior launches is a known quantity either way, and
        # a known quantity is worth more than a coin flip.
        history = features.get("deployer_launches")
        if history:
            score += 0.15
        # The structural signals sum to less than one pre-launch naming, on
        # purpose. Venue, funding shape and deployer history are all derived
        # from the launch transaction, so they are correlated with each other
        # and with nothing outside it; somebody talking about a mint before
        # it existed is information the chain does not contain, and it should
        # not be outvoted by three readings of the same document.
        return score

    # --- the decision ----------------------------------------------------

    def _bar(self, utilisation: float) -> Optional[float]:
        if len(self._priorities) < MIN_PRIORITIES:
            return None
        if utilisation <= self.shed_floor:
            return None
        # Linear from the floor to full: no shedding at the floor, the
        # configured quantile at capacity.
        span = max(1e-9, 1.0 - self.shed_floor)
        fraction = min(1.0, (utilisation - self.shed_floor) / span)
        quantile = self.max_quantile * fraction
        if quantile <= 0.0:
            return None
        ordered = sorted(self._priorities)
        index = min(len(ordered) - 1, int(quantile * len(ordered)))
        return ordered[index]

    def admit(self, features: Dict[str, Any], in_flight: int) -> ShedDecision:
        """Whether to spend a pipeline slot on this launch."""
        priority = self.priority(features)
        self._priorities.append(priority)
        utilisation = in_flight / self.capacity
        self.peak_utilisation = max(self.peak_utilisation, utilisation)
        if in_flight >= self.capacity:
            # Genuinely full. Nothing can be admitted, whatever it scores --
            # the slot does not exist.
            self.shed += 1
            self.last_shed_at = time.time()
            self.shed_by_priority.append(priority)
            del self.shed_by_priority[:-256]
            return ShedDecision(
                False, priority, None, utilisation,
                f"every one of {self.capacity} pipeline slots is in use")
        bar = self._bar(utilisation)
        if bar is None or priority >= bar:
            self.admitted += 1
            return ShedDecision(True, priority, bar, utilisation)
        self.shed += 1
        self.last_shed_at = time.time()
        self.shed_by_priority.append(priority)
        del self.shed_by_priority[:-256]
        return ShedDecision(
            False, priority, bar, utilisation,
            f"priority {priority:.2f} below the {bar:.2f} bar at "
            f"{utilisation:.0%} of capacity")

    def report(self) -> Dict[str, Any]:
        ordered = sorted(self._priorities)
        median = ordered[len(ordered) // 2] if ordered else None
        return {
            "schema": LOAD_SHEDDING_SCHEMA_VERSION,
            "status": "OK",
            "capacity": self.capacity,
            "admitted": self.admitted,
            "shed": self.shed,
            "shed_rate": (self.shed / (self.shed + self.admitted)
                          if (self.shed + self.admitted) else 0.0),
            "peak_utilisation": round(self.peak_utilisation, 3),
            "priority_samples": len(self._priorities),
            "median_priority": median,
            "median_shed_priority": (
                sorted(self.shed_by_priority)[len(self.shed_by_priority) // 2]
                if self.shed_by_priority else None),
            "seconds_since_shed": (round(time.time() - self.last_shed_at, 1)
                                   if self.last_shed_at else None),
            "detail": ("under saturation the desk declines the least "
                       "promising launches rather than the most recent; "
                       "every decline is a recorded decision carrying its "
                       "priority, so whether the shed launches were the good "
                       "ones is answerable afterwards"),
        }

"""Screens that shrink a position instead of discarding the launch.

The census measured what the filters cost, and the answer was severe: the
great majority of resolved monsters were discarded upstream, before anything
could form a view on them. That is the most expensive failure mode available
to a strategy whose returns are dominated by rare large outcomes, because the
losses it prevents are bounded at one position and the gains it forgoes are
not.

The mistake is structural, not parametric. A screen answers a yes/no question
-- "are the holders too concentrated" -- and a yes/no answer cannot express
the thing that is actually true, which is that concentration makes a token
worse rather than untradeable. Tuning the threshold does not fix that; it only
moves which launches fall off the cliff.

So a screen becomes evidence. Each returns a SIZE MULTIPLIER in (0, 1], and
the multipliers compose. A launch with concentrated holders, no source touch
and a correlated First25 does not get three rejections: it gets a position a
small fraction of full size, which is what "three independent reasons for
concern" actually means under a log-wealth objective.

Two things are deliberately NOT softened.

**Hard vetoes remain hard.** Some conditions are not "worse", they are
"untradeable at any size": an unsellable token, a freeze authority that is
live and has been used, a route that cannot be built, the daily-loss kill
switch. Those still reject, and they are enumerated here so the distinction
is visible rather than implied.

**The floor is not zero.** A launch whose multipliers compose below the floor
is rejected -- not because the screen said no, but because the position it
justifies is too small to clear its own execution cost. That is an economic
statement, and it is the honest form of "no": we are not declining because a
rule fired, we are declining because there is no size at which this pays.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

SCREEN_POLICY_SCHEMA_VERSION = "v1"

#: Below this composed multiplier the position cannot clear its own costs, and
#: the launch is declined on economics rather than on a rule.
DEFAULT_SIZE_FLOOR = 0.05


class Verdict(Enum):
    FULL = "full"
    REDUCED = "reduced"
    #: Declined because no size pays, not because a rule fired.
    UNECONOMIC = "uneconomic"
    #: Untradeable at any size.
    VETOED = "vetoed"


@dataclass
class ScreenReading:
    """One screen's contribution: a multiplier, or a veto."""

    name: str
    multiplier: float = 1.0
    veto: bool = False
    reason: str = ""
    #: How well this screen's own input was measured, from the fallback
    #: ladder. A screen firing on a PRIOR should not cut size as hard as one
    #: firing on a reading, or the desk ends up sized by its own ignorance.
    confidence: float = 1.0

    def effective(self) -> float:
        """The multiplier, softened toward 1.0 by how little we actually know.

        A screen that is certain cuts fully. A screen resting on a proxy cuts
        proportionally less -- because the alternative is letting an inferred
        fact shrink a position as hard as a measured one, which is how a desk
        becomes maximally cautious exactly where it is least informed.
        """
        if self.veto:
            return 0.0
        clamped = max(0.0, min(1.0, float(self.multiplier)))
        weight = max(0.0, min(1.0, float(self.confidence)))
        return 1.0 - (1.0 - clamped) * weight

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "multiplier": round(self.multiplier, 4),
                "effective": round(self.effective(), 4), "veto": self.veto,
                "confidence": round(self.confidence, 4), "reason": self.reason}


@dataclass
class ScreenOutcome:
    """What the screens jointly decided, and what each contributed."""

    verdict: Verdict
    size_multiplier: float
    readings: List[ScreenReading] = field(default_factory=list)
    reason: str = ""

    @property
    def rejected(self) -> bool:
        return self.verdict in (Verdict.UNECONOMIC, Verdict.VETOED)

    @property
    def census_reason(self) -> str:
        """What to record in the launch census when this declines.

        A reduced position is NOT a screen for census purposes -- the launch
        reached a decision. Only a genuine decline is attributed, so the
        missed-monster attribution keeps measuring what was thrown away
        rather than what was merely sized down.
        """
        if not self.rejected:
            return ""
        vetoes = [item.name for item in self.readings if item.veto]
        if vetoes:
            return f"veto_{vetoes[0]}"
        worst = min(self.readings, key=lambda item: item.effective(),
                    default=None)
        return f"uneconomic_{worst.name}" if worst else "uneconomic"

    def to_dict(self) -> Dict[str, Any]:
        return {"verdict": self.verdict.value,
                "size_multiplier": round(self.size_multiplier, 4),
                "reason": self.reason,
                "readings": [item.to_dict() for item in self.readings]}


def graded(name: str, value: Optional[float], *, benign: float, severe: float,
           floor: float = 0.15, confidence: float = 1.0,
           reason: str = "") -> ScreenReading:
    """Turn a continuous measurement into a multiplier rather than a cliff.

    ``benign`` is the level at which the reading costs nothing; ``severe`` is
    where it costs everything down to ``floor``. Between them the multiplier
    falls smoothly, which is the whole point: a token at 0.41 concentration
    and one at 0.39 should differ by two percent of size, not by everything.

    An unmeasured value returns a multiplier of 1.0 at LOW confidence rather
    than a punitive one. That looks backwards and is not: an unmeasured fact
    must not masquerade as a benign one, so the caller sees full size at low
    confidence and the confidence itself shrinks the position elsewhere --
    rather than this screen inventing a severity it did not observe.
    """
    if value is None:
        return ScreenReading(name=name, multiplier=1.0, confidence=0.0,
                             reason=f"{name} unmeasured; not treated as benign")
    span = severe - benign
    if abs(span) < 1e-12:
        share = 1.0 if value >= severe else 0.0
    else:
        share = (float(value) - benign) / span
    share = max(0.0, min(1.0, share))
    multiplier = 1.0 - share * (1.0 - floor)
    return ScreenReading(name=name, multiplier=multiplier, confidence=confidence,
                         reason=reason or f"{name}={value:.4g}")


def veto(name: str, reason: str) -> ScreenReading:
    """A condition that is untradeable at any size."""
    return ScreenReading(name=name, multiplier=0.0, veto=True, reason=reason)


class ScreenPolicy:
    """Composes screen readings into one size multiplier or one decline."""

    def __init__(self, *, size_floor: float = DEFAULT_SIZE_FLOOR):
        self.size_floor = float(size_floor)
        self.full = 0
        self.reduced = 0
        self.uneconomic = 0
        self.vetoed = 0
        #: Size actually forgone per screen, in multiplier terms. This is the
        #: number that says which screen is expensive -- it accumulates on
        #: every launch a screen touches, not only the ones it rejects.
        self.forgone: Dict[str, float] = {}
        self.touched: Dict[str, int] = {}

    def evaluate(self, readings: Sequence[ScreenReading]) -> ScreenOutcome:
        rows = list(readings or ())
        for reading in rows:
            self.touched[reading.name] = self.touched.get(reading.name, 0) + 1
            self.forgone[reading.name] = (self.forgone.get(reading.name, 0.0)
                                          + (1.0 - reading.effective()))
        vetoes = [item for item in rows if item.veto]
        if vetoes:
            self.vetoed += 1
            return ScreenOutcome(
                verdict=Verdict.VETOED, size_multiplier=0.0, readings=rows,
                reason=f"untradeable at any size: {vetoes[0].reason}")

        multiplier = 1.0
        for reading in rows:
            multiplier *= reading.effective()

        if multiplier < self.size_floor:
            self.uneconomic += 1
            return ScreenOutcome(
                verdict=Verdict.UNECONOMIC, size_multiplier=multiplier,
                readings=rows,
                reason=(f"composed size {multiplier:.3f} is below the "
                        f"{self.size_floor:.2f} floor; no size clears its own "
                        "execution cost"))
        if multiplier >= 0.999:
            self.full += 1
            return ScreenOutcome(verdict=Verdict.FULL, size_multiplier=1.0,
                                 readings=rows, reason="no screen reduced this")
        self.reduced += 1
        worst = min(rows, key=lambda item: item.effective())
        return ScreenOutcome(
            verdict=Verdict.REDUCED, size_multiplier=multiplier, readings=rows,
            reason=f"reduced to {multiplier:.3f}, chiefly by {worst.name}")

    def report(self) -> Dict[str, Any]:
        """Which screens are expensive, measured in size rather than in counts.

        A screen that rejects rarely but cuts every launch it touches by half
        is more expensive than one that rejects often and cuts nothing else,
        and a count-based report cannot see that.
        """
        total = self.full + self.reduced + self.uneconomic + self.vetoed
        ranked = sorted(
            ((name, cost, self.touched.get(name, 0))
             for name, cost in self.forgone.items()),
            key=lambda row: row[1], reverse=True)
        return {
            "schema": SCREEN_POLICY_SCHEMA_VERSION,
            "status": "OK" if total else "DATA_BLOCKED",
            "detail": ("" if total else "no launch has been screened yet"),
            "evaluated": total,
            "full_size": self.full,
            "reduced": self.reduced,
            "declined_uneconomic": self.uneconomic,
            "declined_veto": self.vetoed,
            # The share that would have been thrown away under hard screens
            # and is now merely sized down. This is the leak being repaired.
            "kept_but_reduced_share": (self.reduced / total) if total else None,
            "size_floor": self.size_floor,
            "costliest_screens": [
                {"screen": name, "size_forgone": round(cost, 3),
                 "launches_touched": touched,
                 "mean_cut": round(cost / touched, 4) if touched else None}
                for name, cost, touched in ranked[:10]],
        }

"""What one slot of delay actually costs THIS opportunity.

The landing model already bids economically: it learns P(land | bid) from our
own attempts and picks the bid that maximises expected value rather than
paying a fixed ladder. That is the right shape, and it is missing one term.

It treats the edge as a constant. It is not. The whole value of a trade is
that some of it is available now and less of it will be available later, and
how much less is a property of the specific opportunity:

    V_slot = E[dlogW | land now] - E[dlogW | land one slot later]

For an ordinary launch drifting sideways, one slot costs almost nothing and
bidding as though it were urgent is burning fee on a race not worth winning.
For a T0 monster setup where the curve is moving every slot, one slot can cost
a third of the edge, and bidding the same as the ordinary launch loses the
trade that mattered.

So decay is MEASURED from the thing that is actually decaying -- how fast this
token's own price is moving right now -- rather than inferred from a category.
A "monster" label is a claim about the outcome; the slope of the curve over
the last few seconds is a measurement of the present, and it is the present
that a slot of delay is spent in.

Two directions this refuses to bid up.

A price falling fast is not urgency to BUY. The same slope that makes a rising
token expensive to miss makes a falling one cheaper to wait for, and treating
the magnitude of the move as urgency in both directions would systematically
overpay to catch falling knives. Direction is read against the side.

And an EXIT is not priced from the slope at all. What an exit races is the
hazard, not the drift: leaving a position that is about to become unsellable
is worth the position, whatever the price is doing this second. That number
comes from the escape model, which already knows it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

SLOT_VALUE_SCHEMA_VERSION = "v1"

# One Solana slot. What "one slot later" means in seconds, and therefore the
# unit the decay is expressed in.
SLOT_SECONDS = 0.4

# Marks older than this say nothing about the current slope. A token that
# moved 40% two minutes ago and has been flat since is not decaying now.
DEFAULT_MARK_MAX_AGE_S = 10.0

# Fewest marks before a slope is a slope. Two points through noise is a line
# through noise.
DEFAULT_MIN_MARKS = 3

# Cap on the share of edge one slot may be said to cost. A decay estimate that
# can reach 1.0 would authorise bidding the entire edge on a single slot,
# which turns any measurement error straight into overpayment.
DEFAULT_MAX_DECAY = 0.35


@dataclass
class SlotValue:
    """The cost of one slot of delay, and where the number came from."""

    status: str
    # Share of the edge lost per slot, in [0, max_decay].
    decay_per_slot: float = 0.0
    # The edge that survives landing one slot late.
    value_now_usd: float = 0.0
    value_next_slot_usd: float = 0.0
    slope_per_second: Optional[float] = None
    marks_used: int = 0
    source: str = ""
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "OK"

    @property
    def slot_cost_usd(self) -> float:
        """What winning this slot rather than the next one is worth."""
        return max(0.0, self.value_now_usd - self.value_next_slot_usd)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": SLOT_VALUE_SCHEMA_VERSION, "status": self.status,
            "decay_per_slot": round(self.decay_per_slot, 6),
            "value_now_usd": round(self.value_now_usd, 6),
            "value_next_slot_usd": round(self.value_next_slot_usd, 6),
            "slot_cost_usd": round(self.slot_cost_usd, 6),
            "slope_per_second": self.slope_per_second,
            "marks_used": self.marks_used, "source": self.source,
            "detail": self.detail,
        }


class SlotValueModel:
    """How fast the edge on one opportunity is decaying, right now."""

    def __init__(self, *, slot_seconds: float = SLOT_SECONDS,
                 mark_max_age_s: float = DEFAULT_MARK_MAX_AGE_S,
                 min_marks: int = DEFAULT_MIN_MARKS,
                 max_decay: float = DEFAULT_MAX_DECAY):
        self.slot_seconds = float(slot_seconds)
        self.mark_max_age_s = float(mark_max_age_s)
        self.min_marks = max(2, int(min_marks))
        self.max_decay = min(1.0, max(0.0, float(max_decay)))

    def from_marks(self, marks: Sequence[Tuple[float, float]], *,
                   expected_edge_usd: float, buying: bool = True,
                   now: Optional[float] = None) -> SlotValue:
        """Decay from this token's own recent price path.

        ``marks`` are (timestamp, price_multiple), oldest first. The slope is
        taken in LOG space, because what decays is a multiplicative return and
        a linear slope would call the same proportional move twice as urgent
        at twice the price.
        """
        import time as _time

        now = _time.time() if now is None else now
        fresh = [(stamp, value) for stamp, value in marks
                 if value > 0 and 0 <= now - stamp <= self.mark_max_age_s]
        edge = max(0.0, float(expected_edge_usd))
        if len(fresh) < self.min_marks:
            return SlotValue(
                status="DATA_BLOCKED", value_now_usd=edge, value_next_slot_usd=edge,
                marks_used=len(fresh), source="marks",
                detail=(f"{len(fresh)} fresh marks; {self.min_marks} needed before "
                        "a slope is a slope"))
        fresh.sort()
        span = fresh[-1][0] - fresh[0][0]
        if span <= 0:
            return SlotValue(status="DATA_BLOCKED", value_now_usd=edge,
                             value_next_slot_usd=edge, marks_used=len(fresh),
                             source="marks", detail="every mark shares one timestamp")
        slope = (math.log(fresh[-1][1]) - math.log(fresh[0][1])) / span

        # Direction matters. A price falling fast is not urgency to buy -- the
        # same slope that makes a rising token expensive to miss makes a
        # falling one cheaper to wait for.
        against_us = slope > 0 if buying else slope < 0
        if not against_us:
            return SlotValue(
                status="OK", decay_per_slot=0.0, value_now_usd=edge,
                value_next_slot_usd=edge, slope_per_second=slope,
                marks_used=len(fresh), source="marks",
                detail="the move is in our favour; a slot of delay costs nothing")

        decay = min(self.max_decay, abs(slope) * self.slot_seconds)
        return SlotValue(
            status="OK", decay_per_slot=decay, value_now_usd=edge,
            value_next_slot_usd=edge * (1.0 - decay), slope_per_second=slope,
            marks_used=len(fresh), source="marks",
            detail=f"log slope {slope:+.4f}/s costs {decay:.1%} of the edge per slot")

    def from_hazard(self, hazard_per_second: Optional[float], *,
                    expected_edge_usd: float) -> SlotValue:
        """Decay for an EXIT, which races the hazard rather than the drift.

        Leaving a position that is about to become unsellable is worth the
        position, whatever the price is doing this second -- so the slope has
        nothing to say here and the hazard has everything.
        """
        edge = max(0.0, float(expected_edge_usd))
        if hazard_per_second is None or not math.isfinite(float(hazard_per_second)):
            return SlotValue(status="DATA_BLOCKED", value_now_usd=edge,
                             value_next_slot_usd=edge, source="hazard",
                             detail="hazard rate not measured")
        rate = max(0.0, float(hazard_per_second))
        # Probability the thing we are escaping happens during this slot.
        decay = min(self.max_decay, 1.0 - math.exp(-rate * self.slot_seconds))
        return SlotValue(
            status="OK", decay_per_slot=decay, value_now_usd=edge,
            value_next_slot_usd=edge * (1.0 - decay), source="hazard",
            detail=f"hazard {rate:.4f}/s puts {decay:.1%} of this exit at risk per slot")


def urgency_adjusted_edge(slot: SlotValue, expected_edge_usd: float) -> float:
    """The edge to bid against, given how fast it is decaying.

    The bid should be sized on what winning THIS SLOT is worth, which is the
    edge times how much of it a slot of delay destroys -- not on the whole
    edge, which is what is at stake over the whole trade rather than over the
    race. Bidding the whole edge on every slot overpays on the ordinary launch
    and is indistinguishable from the fixed ladder it replaced.

    An unmeasured decay returns the edge unchanged, so an unmeasured slot value
    can only ever leave the bid where the landing model would have put it.
    """
    edge = max(0.0, float(expected_edge_usd))
    if not slot.ok or slot.decay_per_slot <= 0:
        return edge
    return edge * min(1.0, max(0.0, slot.decay_per_slot))

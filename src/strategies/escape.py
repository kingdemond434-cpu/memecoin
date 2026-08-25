"""Whether we could actually get out, which usually matters more than whether it rugs.

A single ``p_rug`` conflates two questions that call for opposite actions. "80%
chance this eventually dies" and "35% chance of catastrophic collapse in the
next second" are both high rug probabilities and only one of them is a reason
not to be in the trade right now. A token that will certainly die in twenty
minutes, with a 60% chance of another 3x first, is a position; a token with a
35% chance of collapsing in the next second is not, whatever its eventual
distribution looks like.

The variable that is missing from almost every rug model is escape. A predicted
5x is worth nothing if the probability of our sell landing before the collapse
is near zero, and a modest predicted gain on a deep, fast-to-exit token can be
worth more than a huge one on a token we would be trapped in. So escape is
modelled explicitly, per size:

    P(escape) = P(the transaction lands before the event) * (share of the
                position the venue can absorb at that moment)

Both factors are necessary. A transaction that lands into a pool that cannot
absorb the size has not escaped; neither has a perfectly sized order that
arrives after the liquidity is gone. Multiplying them means either one going to
zero takes the whole thing to zero, which is the correct behaviour and not what
a model tracking only one of them does.

Hazards are also split by mechanism, because they are not interchangeable
evidence. Creator selling, funder-linked selling, liquidity removal,
sellability loss and migration failure have different lead times and different
escape prospects, and averaging them into one number destroys exactly the
distinction the exit needs.
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ESCAPE_SCHEMA_VERSION = "v1"

# The horizons that actually differ in what they imply for action. Sub-second
# entries exist because on a newborn launch they are where the decision lives.
HAZARD_HORIZONS: Tuple[float, ...] = (0.1, 0.5, 1.0, 3.0, 10.0, 30.0, 300.0)


class HazardMechanism(Enum):
    """Distinct ways a position dies. Not interchangeable evidence."""

    CREATOR_SELLING = "creator_selling"
    FUNDER_LINKED_SELLING = "funder_linked_selling"
    INSIDER_CLUSTER_EXIT = "insider_cluster_exit"
    LIQUIDITY_REMOVAL = "liquidity_removal"
    SELLABILITY_LOSS = "sellability_loss"
    AUTHORITY_ABUSE = "authority_abuse"
    MIGRATION_FAILURE = "migration_failure"


# Mechanisms that make a position unsellable rather than merely worth less.
# Escape from these is not a matter of speed -- there is nothing to sell into.
UNESCAPABLE_MECHANISMS = frozenset({
    HazardMechanism.SELLABILITY_LOSS,
    HazardMechanism.AUTHORITY_ABUSE,
    HazardMechanism.LIQUIDITY_REMOVAL,
})


@dataclass
class HazardCurve:
    """Per-mechanism instantaneous rates, and the horizon probabilities they imply."""

    status: str
    rates: Dict[HazardMechanism, float] = field(default_factory=dict)
    detail: str = ""

    @property
    def total_rate(self) -> float:
        """Combined rate. Competing risks add; probabilities do not."""
        return float(sum(max(0.0, rate) for rate in self.rates.values()))

    def probability_within(self, seconds: float) -> Optional[float]:
        """P(any mechanism fires within ``seconds``)."""
        if self.status != "OK" or seconds <= 0:
            return None
        return float(1.0 - math.exp(-self.total_rate * seconds))

    def unescapable_rate(self) -> float:
        """Rate of the mechanisms that speed cannot save a position from."""
        return float(sum(max(0.0, rate) for mechanism, rate in self.rates.items()
                         if mechanism in UNESCAPABLE_MECHANISMS))

    def curve(self, horizons: Sequence[float] = HAZARD_HORIZONS) -> Dict[float, Optional[float]]:
        return {float(horizon): self.probability_within(horizon) for horizon in horizons}

    def dominant(self) -> Optional[HazardMechanism]:
        if self.status != "OK" or not self.rates:
            return None
        return max(self.rates.items(), key=lambda item: item[1])[0]

    def report(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "rates": {mechanism.value: rate for mechanism, rate in self.rates.items()},
            "curve": {str(horizon): value for horizon, value in self.curve().items()},
            "dominant": self.dominant().value if self.dominant() else None,
            "detail": self.detail,
        }


def hazard_curve_from_probabilities(
    probabilities: Dict[HazardMechanism, Tuple[float, float]],
) -> HazardCurve:
    """Build rates from (probability, horizon) pairs, one per mechanism.

    Each model reports at whatever horizon it was trained on; converting to an
    instantaneous rate is what makes them combinable at all. Adding
    probabilities from different horizons directly is a category error that
    quietly overstates short-horizon risk and understates long.
    """
    rates: Dict[HazardMechanism, float] = {}
    for mechanism, (probability, horizon) in probabilities.items():
        if probability is None or horizon is None or horizon <= 0:
            continue
        clamped = float(np.clip(probability, 0.0, 1.0 - 1e-9))
        rates[mechanism] = -math.log(1.0 - clamped) / float(horizon)
    if not rates:
        return HazardCurve(status="DATA_BLOCKED",
                           detail="no mechanism reported a probability at a usable horizon")
    return HazardCurve(status="OK", rates=rates, detail=f"{len(rates)} mechanisms")


@dataclass
class EscapeEstimate:
    status: str
    probability: float = 0.0
    fillable_share: float = 0.0
    land_probability: float = 0.0
    expected_latency_s: float = 0.0
    detail: str = ""


def escape_probability(
    position_size: int,
    sellable_size: Optional[int],
    expected_latency_s: float,
    hazard: HazardCurve,
    landing_probability: float = 1.0,
) -> EscapeEstimate:
    """P(we get this position out before the event), for one size.

    Two independent ways to fail, multiplied rather than averaged so either one
    going to zero takes the result to zero:

      - the transaction has to land before the hazard fires, which depends on
        latency and on the rate of the mechanisms speed can outrun;
      - the venue has to be able to absorb the size when it arrives.

    Mechanisms in UNESCAPABLE_MECHANISMS are excluded from the race entirely.
    Once liquidity is gone or the mint is frozen there is nothing to sell into,
    so a faster transaction changes nothing, and letting speed discount those
    would produce exactly the false confidence that gets a position trapped.
    """
    if position_size <= 0:
        return EscapeEstimate(status="DATA_BLOCKED", detail="no position to escape")
    if sellable_size is None:
        # Unknown capacity is not full capacity. A caller that cannot measure
        # depth has to treat the position as untradeable, not as liquid.
        return EscapeEstimate(status="DATA_BLOCKED",
                              detail="exit capacity not measured; escape is unknown")
    if hazard.status != "OK":
        return EscapeEstimate(status="DATA_BLOCKED", detail="no hazard curve")
    if expected_latency_s < 0:
        return EscapeEstimate(status="DATA_BLOCKED", detail="negative latency supplied")

    fillable = float(np.clip(sellable_size / position_size, 0.0, 1.0))
    # Only the mechanisms a faster transaction can actually beat enter the
    # race. The unescapable ones are deliberately excluded rather than
    # discounted by latency: once the mint is frozen or the liquidity is gone
    # there is nothing to sell into, so letting a shorter latency raise the
    # estimate would manufacture confidence in exactly the situation that traps
    # a position. Their effect reaches this number through `fillable`, which
    # goes to zero when the venue can no longer absorb anything.
    escapable_rate = max(0.0, hazard.total_rate - hazard.unescapable_rate())
    survives_race = math.exp(-escapable_rate * max(0.0, expected_latency_s))
    land = float(np.clip(landing_probability, 0.0, 1.0))

    probability = float(fillable * survives_race * land)
    return EscapeEstimate(
        status="OK", probability=probability, fillable_share=fillable,
        land_probability=land * survives_race,
        expected_latency_s=float(expected_latency_s),
        detail=(f"{fillable:.1%} fillable, {survives_race:.1%} outruns the "
                f"escapable hazard, {hazard.unescapable_rate():.4f}/s unescapable"),
    )


@dataclass
class RideVerdict:
    status: str
    action: str = "reject"
    e_log_ride: float = 0.0
    e_log_reject: float = 0.0
    escape: Optional[EscapeEstimate] = None
    detail: str = ""


def ride_or_reject(
    upside_multiple: float,
    upside_probability: float,
    hazard: HazardCurve,
    escape: EscapeEstimate,
    position_fraction: float,
    horizon_s: float,
    residual_multiple_on_failure: float = 0.02,
) -> RideVerdict:
    """Whether a token likely to die is still worth being in right now.

    "Likely to eventually rug" and "about to rug" call for opposite actions,
    and a binary safe/unsafe gate cannot express the difference. The comparison
    is in log wealth over the horizon:

        ride:   upside if it runs and we get out, residual if it does not
        reject: the book unchanged

    Escape enters as the probability of realising the upside at all. A 5x we
    cannot exit is not a 5x, so an escape probability near zero drives the ride
    branch to the failure outcome regardless of how large the predicted move
    is. That is the whole point of computing it separately.
    """
    if hazard.status != "OK" or escape.status != "OK":
        return RideVerdict(status="DATA_BLOCKED",
                           detail="hazard or escape unavailable; not a tradeable state")
    if not 0 < position_fraction <= 1 or horizon_s <= 0 or upside_multiple <= 0:
        return RideVerdict(status="DATA_BLOCKED", detail="inputs out of range")

    event = hazard.probability_within(horizon_s)
    if event is None:
        return RideVerdict(status="DATA_BLOCKED", detail="hazard horizon unavailable")
    survives = 1.0 - event
    p_up = float(np.clip(upside_probability, 0.0, 1.0))

    def wealth(multiple: float) -> float:
        return (1.0 - position_fraction) + position_fraction * max(0.0, multiple)

    # The move happens and the position is out in time.
    captured = p_up * survives * escape.probability
    # The move happens but the position is caught; only the fillable share got
    # out, at the pre-event price, and the remainder is residual.
    trapped_after_run = p_up * survives * (1.0 - escape.probability)
    trapped_value = (escape.fillable_share * upside_multiple
                     + (1.0 - escape.fillable_share) * residual_multiple_on_failure)
    # The move does not happen, or the event front-runs it.
    failed = max(0.0, 1.0 - captured - trapped_after_run)

    e_log_ride = (
        captured * math.log(max(1e-12, wealth(upside_multiple)))
        + trapped_after_run * math.log(max(1e-12, wealth(trapped_value)))
        + failed * math.log(max(1e-12, wealth(residual_multiple_on_failure)))
    )
    e_log_reject = 0.0
    action = "ride" if e_log_ride > e_log_reject else "reject"
    return RideVerdict(
        status="OK", action=action, e_log_ride=float(e_log_ride),
        e_log_reject=float(e_log_reject), escape=escape,
        detail=(f"P(event within {horizon_s:g}s)={event:.3f}, "
                f"P(escape)={escape.probability:.3f}"),
    )


def liquidation_ladder(
    position_size: int,
    frontier: Any,
    hazard: HazardCurve,
    expected_latency_s: float,
    slices: Sequence[float] = (0.10, 0.25, 0.50, 0.75, 1.00),
    acceptable_impact: float = 0.10,
) -> Dict[str, Any]:
    """What each slice of the position could actually get out, right now.

    A chart can look healthy while executable exit liquidity quietly rots, and
    the only way to see that is to keep asking the question at every size
    rather than at one.
    """
    sellable = frontier.size_at(acceptable_impact) if getattr(frontier, "ok", False) else None
    rungs = []
    for share in slices:
        size = int(position_size * share)
        estimate = escape_probability(size, sellable, expected_latency_s, hazard)
        rungs.append({
            "share": float(share), "size": size, "status": estimate.status,
            "escape_probability": estimate.probability,
            "fillable_share": estimate.fillable_share,
        })
    return {
        "status": "OK" if sellable is not None and hazard.status == "OK" else "DATA_BLOCKED",
        "sellable_at_impact": sellable, "rungs": rungs,
        "hazard": hazard.report(),
    }

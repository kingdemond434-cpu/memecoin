"""Keeping capital available for the rare event, and scaling into it as it proves.

Two failures sit either side of a genuinely exceptional launch, and a system
tuned to avoid one walks into the other.

The first is being fully invested in mediocrity when it arrives. Detection is
worthless if every dollar is committed to launches that were merely positive
when the once-a-cycle event appears; the allocator will correctly refuse to
displace ten adequate positions fast enough to matter. So the reserve is not a
fixed cash percentage -- a dumb constant is a permanent tax paid for an event
that mostly does not happen -- but a function of how likely an exceptional
event currently is. Quiet weeks hold almost nothing back; a week where a
verified global attention event is propagating holds back a lot.

The second is taking the huge position at T0, before anything is proven. The
life-changing trade is rarely the one that bet maximum size on the first
observation. It is the one that entered small, escalated as authenticity and
independent demand were established, and escalated again as liquidity expanded
enough to hold real money. Capacity escalation makes position size a function
of proven evidence and observed depth rather than of initial conviction.

The two interact deliberately: the reserve exists so the escalation has
something to escalate WITH.

Nothing here can raise a risk limit. The reserve only ever withholds capital,
and the escalation ceiling is bounded by observed depth and by every existing
exposure control. A module whose job is to argue for a very large position
must not also be able to raise the ceiling on one.
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MEGA_EVENT_SCHEMA_VERSION = "v1"


class AudienceTier(Enum):
    """Who has encountered this yet. Ordered by how far the wave has travelled."""

    TRENCHES = "solana_trenches"
    CRYPTO_NATIVE = "crypto_native"
    MAINSTREAM_CRYPTO = "mainstream_crypto"
    MAINSTREAM_PUBLIC = "mainstream_public"
    INTERNATIONAL = "international_public"
    EXCHANGE_LISTED = "exchange_access"


# Rough share of an eventual mega-event audience each tier represents. These
# are priors for reasoning about REMAINING reach, not measurements, and they
# are labelled as such wherever they are consumed.
_TIER_SHARE = {
    AudienceTier.TRENCHES: 0.02,
    AudienceTier.CRYPTO_NATIVE: 0.08,
    AudienceTier.MAINSTREAM_CRYPTO: 0.20,
    AudienceTier.MAINSTREAM_PUBLIC: 0.35,
    AudienceTier.INTERNATIONAL: 0.20,
    AudienceTier.EXCHANGE_LISTED: 0.15,
}


@dataclass
class RemainingAudience:
    """How much of the eventual audience has not encountered this yet.

    The number that separates "+10x and finished" from "+10x with 95% of the
    audience still ahead of it". A monster exit that reasons about price
    without this cannot tell those apart, and they call for opposite actions.
    """

    status: str
    reached: List[AudienceTier] = field(default_factory=list)
    remaining_share: float = 0.0
    detail: str = ""

    @property
    def exhausted(self) -> bool:
        return self.status == "OK" and self.remaining_share <= 0.05


def remaining_audience(reached: Sequence[AudienceTier]) -> RemainingAudience:
    """Share of the audience still unreached, from the tiers observed so far.

    Tiers must be OBSERVED, not assumed. An empty observation set is
    DATA_BLOCKED rather than "nobody has heard of it yet", because those two
    states justify opposite position sizes and only one of them is knowable
    from having seen nothing.
    """
    if not reached:
        return RemainingAudience(status="DATA_BLOCKED",
                                 detail="no audience tier was observed")
    covered = sum(_TIER_SHARE.get(tier, 0.0) for tier in set(reached))
    remaining = float(np.clip(1.0 - covered, 0.0, 1.0))
    return RemainingAudience(
        status="OK", reached=sorted(set(reached), key=lambda tier: tier.value),
        remaining_share=remaining,
        detail=(f"{len(set(reached))} tiers reached; {remaining:.0%} of the modelled "
                "audience is prior-based, not measured"),
    )


@dataclass
class ReserveDecision:
    status: str
    reserve_fraction: float = 0.0
    reason: str = ""
    event_probability: Optional[float] = None


class MegaEventReserve:
    """Withholds capital in proportion to how likely an exceptional event is.

    A fixed cash buffer is a tax paid every week for an event that happens
    twice a year. This one is near zero when nothing is propagating and rises
    only on evidence, so the cost is concentrated in the periods where the
    option is actually worth something.

    It can only ever withhold. There is no path here that frees capital beyond
    what the ordinary limits already allow.
    """

    def __init__(self, baseline_fraction: float = 0.0, max_fraction: float = 0.35,
                 arm_probability: float = 0.05):
        self.baseline_fraction = max(0.0, baseline_fraction)
        self.max_fraction = max(self.baseline_fraction, max_fraction)
        # Below this the event evidence is indistinguishable from a normal
        # week and paying for the option is not justified.
        self.arm_probability = max(0.0, arm_probability)

    def decide(self, event_probability: Optional[float],
               authenticated: bool = False) -> ReserveDecision:
        if event_probability is None:
            return ReserveDecision(status="DATA_BLOCKED",
                                   reserve_fraction=self.baseline_fraction,
                                   reason="no event probability was measured; "
                                          "holding the baseline only")
        probability = float(np.clip(event_probability, 0.0, 1.0))
        if probability < self.arm_probability:
            return ReserveDecision(status="OK", reserve_fraction=self.baseline_fraction,
                                   reason="no elevated event probability",
                                   event_probability=probability)
        span = max(1e-9, 1.0 - self.arm_probability)
        scaled = (probability - self.arm_probability) / span
        # An unauthenticated event is a reason to hold capital ready, not a
        # reason to hold as much as a verified one: most viral stories never
        # produce a token worth funding.
        weight = 1.0 if authenticated else 0.6
        fraction = self.baseline_fraction + (self.max_fraction - self.baseline_fraction) * scaled * weight
        return ReserveDecision(
            status="OK", reserve_fraction=float(min(self.max_fraction, fraction)),
            reason=("verified event propagating" if authenticated
                    else "unverified event propagating"),
            event_probability=probability,
        )

    def deployable_equity(self, equity_usd: float, decision: ReserveDecision) -> float:
        return float(max(0.0, equity_usd * (1.0 - decision.reserve_fraction)))


@dataclass
class EscalationStep:
    name: str
    target_fraction: float
    reason: str


@dataclass
class EscalationPlan:
    status: str
    target_fraction: float = 0.0
    step: str = ""
    reason: str = ""
    capacity_capped: bool = False


# The ladder. Each rung requires the previous rung's evidence plus one new
# fact, so size grows with what has been PROVEN rather than with conviction.
ESCALATION_LADDER: Tuple[Tuple[str, str], ...] = (
    ("probe", "detected"),
    ("authenticated", "authenticity_proven"),
    ("independent_demand", "independent_buyers_arrived"),
    ("liquidity_expanding", "depth_grew"),
    ("mass_adoption", "audience_still_ahead"),
)


def plan_capacity_escalation(
    evidence: Dict[str, bool],
    held_fraction: float,
    max_fraction_by_step: Dict[str, float],
    executable_fraction: Optional[float],
) -> EscalationPlan:
    """Largest position justified by the evidence proven so far.

    The ladder is strictly ordered: a rung is only reached when every rung
    below it holds. Allowing a later rung to be satisfied on its own would let
    a large position be justified by a single impressive-looking signal, which
    is the failure mode the ladder exists to prevent.

    The result is capped by what the venue can actually absorb. A size the
    market cannot fill is not a position, it is a slippage estimate.
    """
    if executable_fraction is None:
        # Unknown depth is not unlimited depth.
        return EscalationPlan(status="DATA_BLOCKED",
                              reason="executable size was never measured")
    reached = "none"
    for step, requirement in ESCALATION_LADDER:
        if not evidence.get(requirement):
            break
        reached = step
    if reached == "none":
        return EscalationPlan(status="OK", target_fraction=0.0, step="none",
                              reason="no rung of the ladder is satisfied")

    target = float(max_fraction_by_step.get(reached, 0.0))
    capped = min(target, float(executable_fraction))
    # Escalation never shrinks a position; banking is a separate decision made
    # by the exit policy, and letting two components both move size down is how
    # a runner gets sold twice.
    target_fraction = max(float(held_fraction), capped)
    return EscalationPlan(
        status="OK", target_fraction=target_fraction, step=reached,
        reason=f"evidence supports the '{reached}' rung",
        capacity_capped=capped < target,
    )

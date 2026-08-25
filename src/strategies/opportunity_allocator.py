"""Cross-sectional capital allocation across every live launch at once.

The engine this sits on top of answers one token at a time: "does this trade
clear the hurdle?" That question has a silent failure mode. A book with ten
mediocre positions that each individually cleared the hurdle will refuse an
eleventh that is five times better, because the exposure ceiling is already
spent -- and it will keep refusing for as long as the mediocre ones are held.
Thresholding tokens independently makes capital a first-come resource rather
than a contested one.

The fix is to score the whole universe on the same axis and let the best
opportunity win:

    score_i = E[dlog W_i] / (capital_i * seconds_i)

which is growth per dollar per second. Dividing by capital makes a small edge
on a small position comparable to a large edge on a large one; dividing by
expected holding time prices opportunity cost, so a +3% trade that ties up
capital for 45 minutes correctly loses to a +1% trade that recycles it in 90
seconds. The objective is still total geometric growth -- this is only the
statement of it that accounts for the fact that capital is finite and time
is not free.

Two properties matter more than the ranking itself:

Replacement is priced, not free. Displacing an incumbent costs its exit
slippage plus the challenger's entry cost, and that round trip is charged to
the challenger before the comparison is made. Without it the allocator churns
the book on noise, which is a reliable way to convert a real edge into fees.

Missing inputs block rather than default. An expected holding time nobody
predicted is not "assume 60 seconds"; a candidate whose depth was never
observed is not "assume it can absorb the trade". Both are DATA_BLOCKED and
simply do not rank. A ranking that silently invents its denominator will
confidently point capital at whichever token has the least information.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Round-trip cost floor charged to any displacement. Below this the allocator
# is trading fees for rank noise.
DEFAULT_REPLACEMENT_COST_PCT = 0.02


@dataclass
class Opportunity:
    """One rankable use of the next dollar.

    ``elogw`` is the expected log-growth contribution of committing
    ``capital_usd`` to this opportunity, over ``expected_hold_seconds``. For a
    fresh candidate it is the from-scratch E[log W]; for an open position it is
    the forward E[log W] of continuing to hold it. ``None`` on either of the
    two denominators means the input was never observed.
    """

    token: str
    elogw: float
    capital_usd: float
    expected_hold_seconds: Optional[float]
    liquidity_usd: Optional[float]
    sleeve: str = "unassigned"
    is_open_position: bool = False
    held_multiple: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Re-prices this opportunity at a different capital commitment. Supplied by
    # callers that can actually recompute E[log W] at another size. Without it
    # the allocator refuses to fund a candidate at a size its edge was never
    # evaluated at, rather than rescaling a number that does not rescale.
    elogw_at: Optional[Callable[[float], float]] = None

    @property
    def blocked_reason(self) -> Optional[str]:
        if not math.isfinite(self.elogw):
            return "DATA_BLOCKED_ELOGW"
        if self.expected_hold_seconds is None:
            return "DATA_BLOCKED_HOLD_TIME"
        if self.expected_hold_seconds <= 0:
            return "DATA_BLOCKED_HOLD_TIME"
        if self.liquidity_usd is None:
            return "DATA_BLOCKED_LIQUIDITY"
        if self.liquidity_usd <= 0:
            return "DATA_BLOCKED_LIQUIDITY"
        if self.capital_usd <= 0:
            return "DATA_BLOCKED_CAPITAL"
        return None

    @property
    def growth_velocity(self) -> float:
        """E[dlog W] per second -- the secondary ranking variable.

        Reported separately from the primary score because two opportunities
        can share a score while differing wildly in how fast they release the
        capital, and that difference is what determines how many further bets
        the book gets to make today.
        """
        if self.blocked_reason:
            return float("-inf")
        return self.elogw / float(self.expected_hold_seconds)

    @property
    def score(self) -> float:
        """E[dlog W] per dollar per second."""
        if self.blocked_reason:
            return float("-inf")
        return self.elogw / (self.capital_usd * float(self.expected_hold_seconds))


@dataclass
class Displacement:
    """A proposal to close an incumbent so a strictly better challenger can run."""

    incumbent: Opportunity
    challenger: Opportunity
    incumbent_score: float
    challenger_score_after_cost: float
    round_trip_cost_usd: float
    freed_capital_usd: float

    @property
    def score_gain(self) -> float:
        return self.challenger_score_after_cost - self.incumbent_score


@dataclass
class Slate:
    """The ranked universe at one instant, plus what it could not rank."""

    ranked: List[Opportunity]
    blocked: List[Tuple[Opportunity, str]]
    displacements: List[Displacement]

    @property
    def best(self) -> Optional[Opportunity]:
        return self.ranked[0] if self.ranked else None

    def by_sleeve(self) -> Dict[str, List[Opportunity]]:
        grouped: Dict[str, List[Opportunity]] = {}
        for opportunity in self.ranked:
            grouped.setdefault(opportunity.sleeve, []).append(opportunity)
        return grouped

    def report(self) -> Dict[str, Any]:
        return {
            "ranked": len(self.ranked),
            "blocked": len(self.blocked),
            "blocked_reasons": sorted({reason for _, reason in self.blocked}),
            "displacements": len(self.displacements),
            "best_token": self.best.token if self.best else None,
            "best_score": self.best.score if self.best else None,
            "sleeves": {name: len(items) for name, items in self.by_sleeve().items()},
        }


class OpportunityAllocator:
    """Ranks the live universe and proposes capital moves within it.

    The allocator proposes; it never executes and never touches risk limits.
    Every displacement it returns is still subject to the exposure ceiling,
    the daily-loss kill switch and the live-capital lock downstream. That
    separation is deliberate: a component whose job is to argue for deploying
    capital must not also be the component that decides how much is allowed.
    """

    def __init__(
        self,
        replacement_cost_pct: float = DEFAULT_REPLACEMENT_COST_PCT,
        min_displacement_gain_ratio: float = 1.5,
        max_displacements_per_cycle: int = 2,
    ):
        self.replacement_cost_pct = max(0.0, replacement_cost_pct)
        # A challenger must beat the incumbent by a multiple, not a hair. Rank
        # differences inside estimation error are not information.
        self.min_displacement_gain_ratio = max(1.0, min_displacement_gain_ratio)
        self.max_displacements_per_cycle = max(0, max_displacements_per_cycle)

    def rank(self, opportunities: Sequence[Opportunity]) -> Slate:
        ranked: List[Opportunity] = []
        blocked: List[Tuple[Opportunity, str]] = []
        for opportunity in opportunities:
            reason = opportunity.blocked_reason
            if reason:
                blocked.append((opportunity, reason))
            else:
                ranked.append(opportunity)
        ranked.sort(key=lambda item: (item.score, item.growth_velocity), reverse=True)
        displacements = self._propose_displacements(ranked)
        return Slate(ranked=ranked, blocked=blocked, displacements=displacements)

    def _propose_displacements(self, ranked: Sequence[Opportunity]) -> List[Displacement]:
        incumbents = [item for item in ranked if item.is_open_position]
        challengers = [item for item in ranked if not item.is_open_position]
        if not incumbents or not challengers:
            return []

        # Worst incumbent first, best challenger first: the only pairing that
        # can be worth the round trip.
        incumbents.sort(key=lambda item: item.score)
        challengers.sort(key=lambda item: item.score, reverse=True)

        proposals: List[Displacement] = []
        used_challengers = set()
        for incumbent in incumbents:
            if len(proposals) >= self.max_displacements_per_cycle:
                break
            for challenger in challengers:
                if challenger.token in used_challengers or challenger.token == incumbent.token:
                    continue
                proposal = self._price_displacement(incumbent, challenger)
                if proposal is None:
                    continue
                proposals.append(proposal)
                used_challengers.add(challenger.token)
                break
        return proposals

    def _price_displacement(
        self, incumbent: Opportunity, challenger: Opportunity
    ) -> Optional[Displacement]:
        # Capital actually released is the incumbent's mark, not its cost: a
        # position sitting at 3x frees three times what it tied up, and one at
        # 0.4x frees far less than the allocator would otherwise believe.
        freed = incumbent.capital_usd * max(0.0, incumbent.held_multiple)
        if freed <= 0:
            return None

        # The challenger's E[log W] was computed at one specific size, and
        # E[log W] is not linear in capital. When the freed capital funds less
        # than that size the number has to be recomputed, not rescaled: only a
        # caller that owns the growth model can do that, so without an
        # ``elogw_at`` the displacement is refused rather than estimated.
        if challenger.liquidity_usd is None:
            return None
        deployable = min(freed, challenger.capital_usd, challenger.liquidity_usd)
        if deployable <= 0:
            return None
        challenger_elogw = challenger.elogw
        if deployable < challenger.capital_usd:
            if challenger.elogw_at is None:
                return None
            challenger_elogw = challenger.elogw_at(deployable)
            if not math.isfinite(challenger_elogw):
                return None

        # Charge the challenger for both legs: getting the incumbent out and
        # getting the challenger in. The score lives in log-wealth, so the
        # dollar cost is converted to the multiplicative haircut it actually
        # represents rather than subtracted from a log quantity.
        round_trip_cost = (freed + deployable) * self.replacement_cost_pct
        cost_fraction = min(0.99, round_trip_cost / max(deployable, 1e-9))
        charged = Opportunity(
            token=challenger.token,
            elogw=challenger_elogw + math.log1p(-cost_fraction),
            capital_usd=deployable,
            expected_hold_seconds=challenger.expected_hold_seconds,
            liquidity_usd=challenger.liquidity_usd,
            sleeve=challenger.sleeve,
        )
        if charged.blocked_reason or charged.score <= 0:
            return None
        if charged.score < incumbent.score * self.min_displacement_gain_ratio:
            return None
        if charged.score <= incumbent.score:
            return None
        return Displacement(
            incumbent=incumbent,
            challenger=challenger,
            incumbent_score=incumbent.score,
            challenger_score_after_cost=charged.score,
            round_trip_cost_usd=round_trip_cost,
            freed_capital_usd=freed,
        )

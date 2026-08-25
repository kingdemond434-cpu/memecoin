"""How much can actually be traded, at what impact, in each direction.

"Liquidity is $X" is not a tradeable fact. It does not say how much can be
bought without moving the price, it does not say how much can be sold, and on
a bonding curve those two numbers are not even close to each other. A position
marked at 20x that can only exit 4% of itself inside an acceptable impact is
not a 20x; it is a 20x on 4% of the size and something much worse on the rest.

So capacity is a curve, not a scalar. For each side the frontier answers:

    q*(1%), q*(3%), q*(5%), q*(10%)

the largest trade whose price impact stays inside each bound. Entry and exit
are computed separately because they are different questions with different
answers, and it is always the exit one that gets discovered too late.

The single most important consumer is ``exit_capacity_ratio``, which converts
"how much of this position is really liquidatable" into the number the
hold-versus-exit comparison discounts upside by. Its default must never be
1.0. Assuming a position is fully sellable until proven otherwise is how a
theoretical return becomes a real loss, so an unmeasurable frontier is
DATA_BLOCKED and the caller has to decide what to do about not knowing.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DEFAULT_IMPACT_BOUNDS: Tuple[float, ...] = (0.01, 0.03, 0.05, 0.10)

# A quote function takes a size and returns (ok, price_impact_pct). ``ok`` is
# False when the venue could not answer at that size at all, which is itself
# information -- it is not zero impact.
QuoteFn = Callable[[int], Tuple[bool, float]]


@dataclass
class Frontier:
    """Executable size at each impact bound, for one side of one venue."""

    status: str
    side: str
    sizes: Dict[float, int] = field(default_factory=dict)
    ceiling: int = 0
    detail: str = ""
    # True when the frontier was computed from virtual reserves alone. The
    # constant-product impact is exact from virtual reserves, but only the
    # REAL reserves cap what the curve can physically pay out, so without them
    # every number here is an upper bound on executable size. Saying so is the
    # difference between an optimistic estimate and a silent one.
    upper_bound_only: bool = False

    @property
    def ok(self) -> bool:
        return self.status == "OK"

    def size_at(self, impact: float) -> Optional[int]:
        return self.sizes.get(float(impact))

    def impact_for(self, size: int) -> Optional[float]:
        """Tightest measured bound this size fits inside, or None if it exceeds all.

        None means "worse than the widest bound measured", not "no impact".
        """
        for bound in sorted(self.sizes):
            if size <= self.sizes[bound]:
                return bound
        return None


def minimum_quotable(quote: QuoteFn, ceiling: int) -> Optional[int]:
    """Smallest size the venue will actually quote.

    Not always 1. A bonding curve legitimately refuses a one-lamport trade
    because the output rounds to zero, and that refusal is a rounding artefact
    at the bottom of the range, not a depth signal. Treating it as "the venue
    will not quote" would report every healthy curve as unquotable, so the
    probe walks up geometrically until something answers.
    """
    size = 1
    while size <= ceiling:
        try:
            ok, _ = quote(size)
        except Exception:  # pragma: no cover - defensive
            return None
        if ok:
            return size
        size *= 4
    return None


def largest_size_within(quote: QuoteFn, ceiling: int, max_impact: float,
                        floor: Optional[int] = None, iterations: int = 64) -> int:
    """Binary search the largest size whose impact stays within ``max_impact``.

    Impact is monotone in size for constant-product curves, but only above the
    integer-rounding region: on a real Pump curve a 64-unit sell reports 3.3%
    impact purely from quantisation while a 1,000,000-unit sell reports 1.0%.
    So the search starts from zero and never requires the low endpoint to
    satisfy the bound -- demanding that would make one rounding artefact at
    the bottom of the range report the whole curve as unexecutable. The probes
    descend from the ceiling and only reach the noisy region when capacity is
    genuinely near zero, which is the answer anyway.
    """
    if ceiling <= 0 or max_impact <= 0:
        return 0
    low, high = 0, int(ceiling)
    steps = 0
    while low < high and steps < iterations:
        steps += 1
        mid = (low + high + 1) // 2
        ok, impact = quote(mid)
        if ok and impact <= max_impact:
            low = mid
        else:
            high = mid - 1
    return low


def build_frontier(
    quote: QuoteFn,
    ceiling: int,
    side: str,
    bounds: Sequence[float] = DEFAULT_IMPACT_BOUNDS,
) -> Frontier:
    """Executable size at each bound for one side.

    A frontier that is zero everywhere is reported as OK with zeros, not as
    blocked: "nothing is executable at an acceptable impact" is a measurement.
    Blocked is reserved for "the venue would not answer at any size".
    """
    if ceiling <= 0:
        return Frontier(status="DATA_BLOCKED", side=side,
                        detail="no ceiling supplied; venue inventory unknown")
    try:
        floor = minimum_quotable(quote, ceiling)
    except Exception as exc:  # pragma: no cover - defensive
        return Frontier(status="DATA_BLOCKED", side=side, detail=f"quote failed: {exc}")
    if floor is None:
        return Frontier(status="DATA_BLOCKED", side=side,
                        detail="venue would not quote at any size up to its ceiling")

    sizes = {float(bound): largest_size_within(quote, ceiling, float(bound))
             for bound in bounds}
    return Frontier(status="OK", side=side, sizes=sizes, ceiling=int(ceiling),
                    detail=f"{len(sizes)} bounds measured against ceiling {ceiling}")


def exit_capacity_ratio(
    position_size: int,
    frontier: Frontier,
    acceptable_impact: float = 0.10,
) -> Tuple[str, float]:
    """Share of the position that is genuinely liquidatable, as (status, ratio).

    There is deliberately no permissive default. A caller that cannot measure
    the frontier gets DATA_BLOCKED and has to decide, rather than silently
    inheriting 1.0 -- which would tell the hold-versus-exit comparison that
    every position is fully sellable, exactly when it is not.
    """
    if position_size <= 0:
        return "DATA_BLOCKED", 0.0
    if not frontier.ok:
        return "DATA_BLOCKED", 0.0
    sellable = frontier.size_at(acceptable_impact)
    if sellable is None:
        # Asking for a bound that was never measured must not fall through to
        # a neighbouring bound: that would silently answer a different question.
        return "DATA_BLOCKED", 0.0
    ratio = float(min(1.0, sellable / position_size))
    return ("OK_UPPER_BOUND" if frontier.upper_bound_only else "OK"), ratio


@dataclass
class TradeabilityReport:
    """Both sides of one token at one instant."""

    entry: Frontier
    exit: Frontier

    @property
    def ok(self) -> bool:
        return self.entry.ok and self.exit.ok

    def round_trip_size(self, impact: float) -> Optional[int]:
        """Largest size that both sides can handle within ``impact``.

        The binding side is the exit, essentially always. Sizing to the entry
        frontier is how a position gets opened that cannot be closed.
        """
        entry = self.entry.size_at(impact)
        exit_side = self.exit.size_at(impact)
        if entry is None or exit_side is None:
            return None
        return min(entry, exit_side)

    def asymmetry(self, impact: float) -> Optional[float]:
        """Exit capacity over entry capacity. Below 1 means easier in than out."""
        entry = self.entry.size_at(impact)
        exit_side = self.exit.size_at(impact)
        if not entry or exit_side is None:
            return None
        return float(exit_side / entry)

    def report(self) -> Dict[str, object]:
        return {
            "entry_status": self.entry.status, "exit_status": self.exit.status,
            "entry_sizes": dict(self.entry.sizes), "exit_sizes": dict(self.exit.sizes),
            "round_trip_5pct": self.round_trip_size(0.05),
            "asymmetry_5pct": self.asymmetry(0.05),
        }


def curve_tradeability(state, quote_buy_fn, quote_sell_fn,
                       bounds: Sequence[float] = DEFAULT_IMPACT_BOUNDS) -> TradeabilityReport:
    """Both frontiers for a Pump bonding curve, from local state only.

    No RPC round trip: the curve state already fully determines both sides, so
    this is answerable in the same microseconds the decision has to be made in.
    """
    def buy_quote(lamports: int) -> Tuple[bool, float]:
        quote = quote_buy_fn(state, int(lamports))
        return (quote.data_status == "OK", float(quote.price_impact_pct))

    def sell_quote(tokens: int) -> Tuple[bool, float]:
        quote = quote_sell_fn(state, int(tokens))
        return (quote.data_status == "OK", float(quote.price_impact_pct))

    sol_ceiling = max(0, int(getattr(state, "virtual_sol_reserves", 0)))
    token_ceiling = max(int(getattr(state, "real_token_reserves", 0)),
                        int(getattr(state, "virtual_token_reserves", 0)))
    upper_bound_only = int(getattr(state, "real_sol_reserves", 0)) <= 0
    entry = build_frontier(buy_quote, sol_ceiling, "entry", bounds)
    exit_side = build_frontier(sell_quote, token_ceiling, "exit", bounds)
    entry.upper_bound_only = upper_bound_only
    exit_side.upper_bound_only = upper_bound_only
    return TradeabilityReport(entry=entry, exit=exit_side)

"""What a millisecond is worth, measured rather than assumed.

Everything this desk has done for speed -- signer IPC from 505 to 235
microseconds, the chain decoded in Rust, the network taken out from in front
of the T0 decision -- has been done on the belief that being earlier is
better. That belief is almost certainly true and has never been priced. And
the question is not rhetorical: if 250ms costs a tenth of a percent of entry
price, most of the fast path is a rounding error and the effort belongs
elsewhere; if it costs eight percent, latency IS the strategy and nothing
else deserves attention until it is exhausted.

The obvious way to find out is to delay some decisions on purpose and
compare. That is the wrong instrument here, for two reasons. It costs money
once the desk is live -- deliberately entering late on a real position is
paying for information the data already contains -- and it cannot run at
all right now, because nothing has been entered. So the delay would produce
its first reading exactly when it starts being expensive.

The data already contains it. Every launch is snapshotted at 0, 10, 25, 50,
100, 250 and 500 milliseconds and at one second, and each snapshot carries
the bonding curve's reserves. For a constant-product curve the marginal
price IS the reserve ratio, so the price a decision at T+d would have paid
is not an estimate: it is arithmetic on numbers already recorded.

    entering at d instead of 0 multiplies the outcome by  p_0 / p_d
    so the cost in log terms is exactly                   -ln(p_d / p_0)

which composes with E[log W] the way every other cost in this desk does.

Two readings, and the second is the one that matters. Across ALL launches
the median drift is small, because most launches go nowhere and their curves
barely move in the first half second. Across the launches that RAN, the
first half second is when the curve moves most -- so a mean that mixes them
understates the cost of being slow by exactly the amount that matters. The
ledger reports both and never collapses them.
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

LATENCY_VALUE_SCHEMA_VERSION = "v1"

#: The delays priced. The short rungs are below the desk's own decode time
#: and are there to bound the answer from beneath: if 10ms already costs
#: something measurable, the remaining microsecond work has a ceiling worth
#: knowing before any more of it is done.
DELAY_LADDER_MS: Tuple[int, ...] = (10, 25, 50, 100, 250, 500, 1000)

#: A launch counts as having RUN at this multiple. Not a prediction target --
#: the split exists because the cost of being late is concentrated entirely
#: in the launches whose price moved, and an average over the rest hides it.
RAN_MULTIPLE = 2.0

#: Below this many resolved launches in a bucket the answer is noise, and a
#: noisy answer about whether to spend a month on latency is worse than
#: DATA_BLOCKED, because it will be acted on.
MIN_SAMPLES = 200


def marginal_price(sol_reserves: float, token_reserves: float) -> Optional[float]:
    """SOL per token at the margin. The curve's price, not an estimate.

    A constant-product curve prices at the reserve ratio by definition, so
    this is arithmetic rather than a model, and it is the same arithmetic
    the program itself does.
    """
    try:
        sol = float(sol_reserves)
        tokens = float(token_reserves)
    except (TypeError, ValueError):
        return None
    if sol <= 0 or tokens <= 0:
        return None
    return sol / tokens


@dataclass
class DelayBucket:
    delay_ms: int
    log_deltas: List[float] = field(default_factory=list)
    log_deltas_that_ran: List[float] = field(default_factory=list)
    #: Launches where the price at the delay could not be read at all. Kept
    #: because a bucket that is 90% unreadable is not a measurement, however
    #: confident the 10% looks.
    unreadable: int = 0

    def _summary(self, samples: Sequence[float]) -> Dict[str, Any]:
        """Signed log deltas in, an operator's cost percentage out."""
        if len(samples) < MIN_SAMPLES:
            return {
                "status": "DATA_BLOCKED",
                "samples": len(samples),
                "needed": MIN_SAMPLES,
                "reason": ("too few resolved launches to price this delay; a "
                           "noisy answer here would be acted on"),
            }
        ordered = sorted(samples)
        median = statistics.median(ordered)
        return {
            "status": "OK",
            "samples": len(ordered),
            "median_log_delta": round(median, 6),
            # The same number as a price move, which is what an operator
            # reads: POSITIVE means entering this late costs this much of
            # the entry. The sign is flipped from the log delta on purpose --
            # a cost is naturally read as a positive quantity.
            "median_cost_pct": round((math.exp(median) - 1.0) * -100.0, 4),
            "mean_log_delta": round(statistics.fmean(ordered), 6),
            # The BAD tail, which for a signed delta is the low end: p10 is
            # the tenth-worst-case launch, not the tenth-best.
            "p10_log_delta": round(ordered[int(0.10 * (len(ordered) - 1))], 6),
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "delay_ms": self.delay_ms,
            "unreadable": self.unreadable,
            # Both, never collapsed. Across all launches the drift is small
            # because most launches go nowhere; across the ones that ran it
            # is not, and that is the population a decision is about.
            "all_launches": self._summary(self.log_deltas),
            "launches_that_ran": self._summary(self.log_deltas_that_ran),
        }


class LatencyValueLedger:
    """Prices the delay ladder from snapshots the desk already takes."""

    def __init__(self, ladder: Sequence[int] = DELAY_LADDER_MS,
                 ran_multiple: float = RAN_MULTIPLE):
        self.ladder = tuple(int(delay) for delay in ladder)
        self.ran_multiple = float(ran_multiple)
        self.buckets: Dict[int, DelayBucket] = {
            delay: DelayBucket(delay) for delay in self.ladder}
        self.launches = 0
        self.launches_that_ran = 0
        self.without_t0 = 0

    def observe(self, prices_by_delay_ms: Dict[int, Optional[float]],
                outcome_multiple: Optional[float]) -> bool:
        """One resolved launch. `prices_by_delay_ms` must include 0.

        Returns whether it contributed anything, so a caller can tell "no
        data yet" from "the data is there and says nothing".
        """
        base = prices_by_delay_ms.get(0)
        if not base or base <= 0:
            # No price at T0 is no baseline, and a cost measured against a
            # baseline we invented is not a measurement.
            self.without_t0 += 1
            return False
        self.launches += 1
        ran = (outcome_multiple is not None
               and float(outcome_multiple) >= self.ran_multiple)
        if ran:
            self.launches_that_ran += 1
        contributed = False
        for delay in self.ladder:
            bucket = self.buckets[delay]
            price = prices_by_delay_ms.get(delay)
            if not price or price <= 0:
                bucket.unreadable += 1
                continue
            # Entering at `delay` instead of 0 multiplies the outcome by
            # p_0 / p_delay, so the LOG RETURN DELTA is -ln(p_delay / p_0).
            #
            # The sign is worth stating because it is easy to read backwards:
            # NEGATIVE means the delay hurt (the price had risen by then and
            # the same SOL bought fewer tokens), POSITIVE means it helped.
            # Launches where being slow helped are kept -- dropping them
            # would manufacture the answer this exists to measure.
            cost = -math.log(price / base)
            bucket.log_deltas.append(cost)
            if ran:
                bucket.log_deltas_that_ran.append(cost)
            contributed = True
        return contributed

    def observe_snapshots(self, snapshots: Dict[Any, Dict[str, Any]],
                          outcome_multiple: Optional[float]) -> bool:
        """Convenience over the dataset builder's own snapshot rows.

        Takes ``{offset_seconds: liquidity_features}`` and reads the curve
        reserves out of each. Any row whose provenance is the protocol
        INVARIANT rather than an observation is skipped: the initialisation
        constants are identical at every offset, so including them would
        report a cost of exactly zero for every delay and drown the real
        measurements in launches that were never observed at all.
        """
        prices: Dict[int, Optional[float]] = {}
        for offset_s, features in snapshots.items():
            if not isinstance(features, dict):
                continue
            if features.get("provenance") == "INVARIANT":
                continue
            if features.get("status") != "OK":
                continue
            price = marginal_price(features.get("curve_sol_reserves"),
                                   features.get("curve_token_reserves"))
            prices[int(round(float(offset_s) * 1000))] = price
        return self.observe(prices, outcome_multiple)

    def cost_of(self, delay_ms: int, *, on_launches_that_ran: bool = True
                ) -> Optional[float]:
        """The measured log return delta for that delay, or None if unpriced.

        NEGATIVE means the delay hurt. See `observe` for why the sign runs
        this way rather than the other.
        """
        bucket = self.buckets.get(int(delay_ms))
        if bucket is None:
            return None
        samples = (bucket.log_deltas_that_ran if on_launches_that_ran
                   else bucket.log_deltas)
        if len(samples) < MIN_SAMPLES:
            return None
        return float(statistics.median(samples))

    def verdict(self) -> str:
        """One line an operator can act on, or an honest refusal.

        This is the whole point of the module. The decision it informs is
        "should the next month go to latency or to alpha", and that decision
        wants a sentence, not a table.
        """
        priced = [(delay, self.cost_of(delay)) for delay in self.ladder]
        priced = [(delay, cost) for delay, cost in priced if cost is not None]
        if not priced:
            return ("DATA_BLOCKED: no delay has enough resolved launches to be "
                    f"priced (need {MIN_SAMPLES} each)")
        worst_delay, worst = max(priced, key=lambda row: abs(row[1]))
        pct = (math.exp(worst) - 1.0) * -100.0
        return (f"{worst_delay}ms of delay costs {pct:.2f}% of entry price on "
                f"the median launch that ran; priced from "
                f"{self.launches_that_ran} such launches")

    def report(self) -> Dict[str, Any]:
        return {
            "schema": LATENCY_VALUE_SCHEMA_VERSION,
            "status": "OK" if self.launches_that_ran >= MIN_SAMPLES
                      else "DATA_BLOCKED",
            "launches": self.launches,
            "launches_that_ran": self.launches_that_ran,
            "ran_multiple": self.ran_multiple,
            "without_t0_price": self.without_t0,
            "verdict": self.verdict(),
            "delays": [self.buckets[delay].as_dict() for delay in self.ladder],
            "detail": ("what being late costs, computed from curve reserves "
                       "already recorded at each snapshot rung rather than by "
                       "delaying live decisions -- which would produce its "
                       "first reading exactly when it starts costing money"),
        }

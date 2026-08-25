"""P(land | bid, conditions), estimated from our own attempts.

Tip and priority-fee selection was a lookup table: 500,000 lamports if the
urgency string said CRITICAL, doubled if the position was large. Those numbers
are somebody's guess about a market that reprices every slot, and the fee
history the optimiser kept was consulted only to find fees with an 80% landing
rate -- then it took the CHEAPEST of them, which is the wrong end: the cheapest
fee clearing 80% is the one closest to failing.

What actually matters is the trade-off. A bid that lands is worth the position;
a bid that does not is worth nothing, and overbidding is a pure transfer. So
the question is P(land | bid) under current conditions, and the right bid is
the one that maximises expected value net of the bid -- which for a large
expected edge is a large bid and for a marginal one is no bid at all.

Nothing here is a trained model in the machine-learning sense, and it does not
pretend to be. It is an empirical landing curve built from our own attempts,
bucketed by bid and conditioned on congestion, with an explicit sample floor
below which it refuses to answer rather than extrapolating from four fills.
Refusing is what keeps it honest: a landing curve fitted to a handful of
attempts will confidently recommend whatever those attempts happened to pay.

The conditioning is deliberately coarse. Congestion is bucketed rather than
continuous because the data available per bucket is small, and a finely
conditioned estimate over few samples is a worse estimate that looks better.
Route and region are recorded but not conditioned on until each has enough
attempts to earn it -- and that promotion is a decision made from the data,
not from a comment.
"""

import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

LANDING_MODEL_SCHEMA_VERSION = "v1"

# Below this many attempts in a bucket the curve is not a curve.
MIN_BUCKET_ATTEMPTS = 20
# Bid buckets in lamports, ascending. Geometric because landing probability
# responds to the ORDER of magnitude of a bid, not to linear increments.
BID_BUCKETS: Tuple[int, ...] = (
    0, 10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 2_500_000, 5_000_000)

CONGESTION_BUCKETS: Tuple[Tuple[str, float], ...] = (
    ("calm", 0.33), ("busy", 0.66), ("contested", 1.01))


def bid_bucket(lamports: int) -> int:
    """The bucket a bid falls into, as its lower bound."""
    chosen = BID_BUCKETS[0]
    for bound in BID_BUCKETS:
        if lamports >= bound:
            chosen = bound
    return chosen


def congestion_bucket(congestion: Optional[float]) -> str:
    if congestion is None:
        return "unknown"
    value = float(np.clip(congestion, 0.0, 1.0))
    for name, upper in CONGESTION_BUCKETS:
        if value < upper:
            return name
    return CONGESTION_BUCKETS[-1][0]


@dataclass
class Attempt:
    bid_lamports: int
    landed: bool
    congestion: Optional[float] = None
    route: str = ""
    region: str = ""
    latency_ms: int = 0


@dataclass
class LandingEstimate:
    status: str
    probability: Optional[float] = None
    attempts: int = 0
    bucket: int = 0
    congestion: str = "unknown"
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "probability": self.probability,
                "attempts": self.attempts, "bucket": self.bucket,
                "congestion": self.congestion, "detail": self.detail}


@dataclass
class BidRecommendation:
    status: str
    bid_lamports: int = 0
    expected_value_usd: Optional[float] = None
    probability: Optional[float] = None
    detail: str = ""
    fallback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "bid_lamports": self.bid_lamports,
                "expected_value_usd": self.expected_value_usd,
                "probability": self.probability, "detail": self.detail,
                "fallback": self.fallback}


class LandingModel:
    """An empirical landing curve over our own attempts."""

    def __init__(self, capacity: int = 20_000,
                 min_bucket_attempts: int = MIN_BUCKET_ATTEMPTS):
        self.capacity = max(1, int(capacity))
        self.min_bucket_attempts = max(1, int(min_bucket_attempts))
        self._attempts: Deque[Attempt] = deque(maxlen=self.capacity)
        self._counts: Dict[Tuple[str, int], List[int]] = defaultdict(lambda: [0, 0])
        self._route_counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0])

    def record(self, attempt: Attempt) -> None:
        """One attempt. Landed or not -- both are evidence, and only one is fun.

        A model fed only successes learns that everything lands.
        """
        if len(self._attempts) == self._attempts.maxlen:
            evicted = self._attempts[0]
            key = (congestion_bucket(evicted.congestion), bid_bucket(evicted.bid_lamports))
            counts = self._counts[key]
            counts[0] = max(0, counts[0] - 1)
            counts[1] = max(0, counts[1] - int(evicted.landed))
        self._attempts.append(attempt)
        key = (congestion_bucket(attempt.congestion), bid_bucket(attempt.bid_lamports))
        counts = self._counts[key]
        counts[0] += 1
        counts[1] += int(attempt.landed)
        if attempt.route:
            route = self._route_counts[attempt.route]
            route[0] += 1
            route[1] += int(attempt.landed)

    def probability(self, bid_lamports: int,
                    congestion: Optional[float] = None) -> LandingEstimate:
        """P(land) at this bid under these conditions, or DATA_BLOCKED.

        Falls back from the conditioned bucket to the unconditioned one before
        refusing, because "we have never traded at this bid in a contested
        slot" is a much narrower gap than "we have never traded at this bid".
        """
        bucket = bid_bucket(int(bid_lamports))
        name = congestion_bucket(congestion)
        for key, label in (((name, bucket), name), (("unknown", bucket), "unconditioned")):
            total, landed = self._counts.get(key, [0, 0])
            if total >= self.min_bucket_attempts:
                return LandingEstimate(status="OK", probability=landed / total,
                                       attempts=total, bucket=bucket, congestion=label)
        pooled_total = sum(total for (_, bid), (total, _) in self._counts.items()
                           if bid == bucket)
        pooled_landed = sum(landed for (_, bid), (_, landed) in self._counts.items()
                            if bid == bucket)
        if pooled_total >= self.min_bucket_attempts:
            return LandingEstimate(status="OK", probability=pooled_landed / pooled_total,
                                   attempts=pooled_total, bucket=bucket,
                                   congestion="pooled")
        return LandingEstimate(
            status="DATA_BLOCKED", attempts=pooled_total, bucket=bucket,
            congestion=name,
            detail=f"need {self.min_bucket_attempts} attempts at this bid, "
                   f"have {pooled_total}")

    def recommend(self, expected_value_usd: float, sol_price_usd: float,
                  congestion: Optional[float] = None,
                  max_bid_lamports: int = 5_000_000) -> BidRecommendation:
        """The bid maximising `P(land) * edge - bid`.

        A bid that lands is worth the position; a bid that does not is worth
        nothing, and overbidding is a pure transfer. That is the whole
        calculation, and it says something the lookup table could not: for a
        marginal edge the right bid is small or zero, and for a large one it
        is larger than any fixed ladder would allow.

        Returns DATA_BLOCKED when the curve cannot answer at any bid, so the
        caller falls back to its configured ladder KNOWING it is doing so.
        """
        if expected_value_usd <= 0 or sol_price_usd <= 0:
            return BidRecommendation(status="REJECTED", bid_lamports=0,
                                     detail="no edge to bid for")
        best: Optional[Tuple[float, int, float]] = None
        measured = 0
        for bid in BID_BUCKETS:
            if bid > max_bid_lamports:
                break
            estimate = self.probability(bid, congestion)
            if estimate.status != "OK" or estimate.probability is None:
                continue
            measured += 1
            bid_usd = (bid / 1e9) * float(sol_price_usd)
            value = estimate.probability * float(expected_value_usd) - bid_usd
            if best is None or value > best[0]:
                best = (value, bid, estimate.probability)
        if best is None:
            return BidRecommendation(
                status="DATA_BLOCKED", bid_lamports=0, fallback=True,
                detail=f"no bid bucket has {self.min_bucket_attempts} attempts yet")
        value, bid, probability = best
        if value <= 0:
            # Every bid loses money at this edge. Not bidding is an answer.
            return BidRecommendation(
                status="OK", bid_lamports=0, expected_value_usd=value,
                probability=probability,
                detail="no bid clears its own cost at this edge")
        return BidRecommendation(status="OK", bid_lamports=bid,
                                 expected_value_usd=value, probability=probability,
                                 detail=f"{measured} measured buckets")

    def report(self) -> Dict[str, Any]:
        curve = []
        for bid in BID_BUCKETS:
            total = sum(count for (_, bucket), (count, _) in self._counts.items()
                        if bucket == bid)
            landed = sum(count for (_, bucket), (_, count) in self._counts.items()
                         if bucket == bid)
            if total:
                curve.append({"bid_lamports": bid, "attempts": total,
                              "landing_rate": landed / total})
        return {
            "schema": LANDING_MODEL_SCHEMA_VERSION,
            "status": "OK" if curve else "DATA_BLOCKED",
            "attempts": len(self._attempts),
            "min_bucket_attempts": self.min_bucket_attempts,
            "curve": curve,
            "routes": {name: {"attempts": total, "landing_rate": landed / total}
                       for name, (total, landed) in self._route_counts.items() if total},
        }

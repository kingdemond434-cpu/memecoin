"""Deliberate bid exploration, so the landing curve can learn its own shape.

The landing model estimates P(land | bid, congestion) from our own attempts.
Those attempts are produced by bidding what the model currently recommends,
which means the only region it ever observes is the region it already
believes in. That is a closed loop: an early over-estimate of what a bid
costs is never contradicted, because nothing ever bids less and lands.

Standard exploit-only failure, and it is worse here than usual because the
thing being estimated is the price of a scarce good. If the model believes
200k lamports is needed and 60k would have done, every trade overpays by
140k for ever and nothing in the data will say so.

So a small, budgeted fraction of attempts deliberately bid somewhere other
than the recommendation, chosen to fill the emptiest cell of the
(congestion x bid) grid rather than at random. Random exploration wastes the
budget re-sampling regions already dense with observations.

Four safety properties, because exploration on a live desk spends real money:

**Never on exits.** An exit that misses is a position held through the event
it was escaping. The upside of learning is bounded; the downside is not.
Exploration is for entries and adds only.

**Never on high-edge trades.** The cost of missing scales with the edge at
stake, so exploration is spent on marginal opportunities where a miss costs
little -- which is also where over-bidding is most wasteful, so the budget
lands where the learning is worth most per dollar risked.

**Bounded in both directions.** An exploratory bid is capped as a multiple of
the recommendation and never exceeds an absolute ceiling. Exploring upward
without a cap is how a fee model gets taught by one catastrophic bid.

**Budgeted per hour, and the budget is spend, not count.** Ten cheap probes
and ten expensive ones are not the same experiment. What is capped is the
lamports knowingly put at risk beyond the recommendation.
"""

from __future__ import annotations

import logging
import random
import time

from src.execution.landing_model import (
    CONGESTION_BUCKETS as _LANDING_BUCKETS,
    congestion_bucket as _landing_congestion_bucket,
)
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BID_EXPLORER_SCHEMA_VERSION = "v1"

#: Bid buckets, as multiples of the current recommendation. Below 1.0 is the
#: valuable direction -- it is what discovers that we are overpaying -- so the
#: grid is denser there.
DEFAULT_BID_MULTIPLES: Tuple[float, ...] = (0.35, 0.5, 0.7, 0.85, 1.25, 1.6)

#: Congestion buckets, IMPORTED from the landing model rather than restated.
#: A parallel set of labels here would mean the cell this explorer fills is
#: not the cell the model records against, so the exploration budget would be
#: spent covering a grid nothing consults -- and the two would drift apart
#: silently the first time either was retuned.
CONGESTION_BUCKETS: Tuple[str, ...] = (
    ("unknown",) + tuple(name for name, _upper in _LANDING_BUCKETS))

#: Attempts per cell beyond which it is considered covered and stops being a
#: target. Enough for a landing rate to mean something; not so many that the
#: budget is spent perfecting one cell.
DEFAULT_COVERAGE_TARGET = 25

#: Ceiling on exploratory spend per hour, in lamports beyond what the
#: recommendation would have cost. Deliberately small: this is a research
#: budget, not a strategy.
DEFAULT_HOURLY_BUDGET_LAMPORTS = 2_000_000

#: An opportunity worth more than this is never explored on.
DEFAULT_MAX_EDGE_USD = 25.0

#: Hard ceiling on any single exploratory bid.
DEFAULT_MAX_BID_LAMPORTS = 1_500_000


def congestion_bucket(congestion: Optional[float]) -> str:
    """Label a congestion reading, exactly as the landing model labels it.

    Delegated rather than reimplemented. None stays 'unknown' rather than
    'calm': an unmeasured chain is not a quiet one.
    """
    return _landing_congestion_bucket(congestion)


@dataclass
class ExplorationChoice:
    """One decision about whether to deviate, and why."""

    explore: bool
    bid_lamports: int
    reason: str
    cell: str = ""
    multiple: float = 1.0
    extra_spend_lamports: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"explore": self.explore, "bid_lamports": self.bid_lamports,
                "reason": self.reason, "cell": self.cell,
                "multiple": self.multiple,
                "extra_spend_lamports": self.extra_spend_lamports}


class BidExplorer:
    """Chooses when to bid off-policy, and tracks what that has bought us."""

    def __init__(self, *, bid_multiples: Tuple[float, ...] = DEFAULT_BID_MULTIPLES,
                 coverage_target: int = DEFAULT_COVERAGE_TARGET,
                 hourly_budget_lamports: int = DEFAULT_HOURLY_BUDGET_LAMPORTS,
                 max_edge_usd: float = DEFAULT_MAX_EDGE_USD,
                 max_bid_lamports: int = DEFAULT_MAX_BID_LAMPORTS,
                 rng: Optional[random.Random] = None):
        self.bid_multiples = tuple(bid_multiples)
        self.coverage_target = max(1, int(coverage_target))
        self.hourly_budget_lamports = max(0, int(hourly_budget_lamports))
        self.max_edge_usd = float(max_edge_usd)
        self.max_bid_lamports = int(max_bid_lamports)
        self._rng = rng or random.Random()
        # (congestion_bucket, bid_multiple) -> attempts, lands
        self._cells: Dict[Tuple[str, float], List[int]] = {}
        self._spent_lamports = 0
        self._window_started = time.time()
        self.explorations = 0
        self.declined = 0
        self.last_choice: Optional[ExplorationChoice] = None

    # --- the decision ----------------------------------------------------

    def choose(self, *, recommended_lamports: int, edge_usd: float,
               congestion: Optional[float], is_exit: bool,
               measured: bool, now: Optional[float] = None) -> ExplorationChoice:
        """Whether to deviate from the recommendation on this attempt.

        ``measured`` says whether the recommendation came from the landing
        curve or from the fallback ladder. Exploring around a fallback is
        pointless -- there is no model to inform -- and worse, it spends the
        budget teaching a curve that is not being consulted.
        """
        now = time.time() if now is None else now
        self._roll_window(now)
        recommended = max(0, int(recommended_lamports))

        if is_exit:
            return self._decline(recommended,
                                 "exits are never explored; a missed exit is "
                                 "the event it was escaping")
        if not measured:
            return self._decline(recommended,
                                 "recommendation is the fallback ladder; there "
                                 "is no curve to inform")
        if recommended <= 0:
            return self._decline(recommended, "no bid to deviate from")
        if float(edge_usd) > self.max_edge_usd:
            return self._decline(
                recommended,
                f"edge ${float(edge_usd):.2f} above the ${self.max_edge_usd:.2f} "
                "exploration ceiling; too expensive to learn on")

        bucket = congestion_bucket(congestion)
        target = self._emptiest_cell(bucket)
        if target is None:
            return self._decline(recommended,
                                 f"every {bucket} cell has reached coverage")

        multiple = target
        bid = int(recommended * multiple)
        bid = max(1, min(bid, self.max_bid_lamports))
        # Only the UPWARD deviation is a cost. Bidding less than recommended
        # risks a miss, not a spend, and that risk is already bounded by the
        # edge ceiling above.
        extra = max(0, bid - recommended)
        if extra and self._spent_lamports + extra > self.hourly_budget_lamports:
            return self._decline(
                recommended,
                f"exploration budget spent ({self._spent_lamports} of "
                f"{self.hourly_budget_lamports} lamports this hour)")

        self._spent_lamports += extra
        self.explorations += 1
        choice = ExplorationChoice(
            explore=True, bid_lamports=bid,
            reason=f"filling the emptiest {bucket} cell at {multiple:.2f}x",
            cell=f"{bucket}@{multiple:.2f}", multiple=multiple,
            extra_spend_lamports=extra)
        self.last_choice = choice
        return choice

    def _decline(self, recommended: int, reason: str) -> ExplorationChoice:
        self.declined += 1
        choice = ExplorationChoice(explore=False, bid_lamports=recommended,
                                   reason=reason)
        self.last_choice = choice
        return choice

    def _emptiest_cell(self, bucket: str) -> Optional[float]:
        """The least-observed bid multiple for this congestion bucket.

        Ties broken randomly so a restart does not always begin by filling the
        same cell, which would leave the grid lopsided in exactly the same way
        every time.
        """
        candidates = [(self._cells.get((bucket, multiple), [0, 0])[0], multiple)
                      for multiple in self.bid_multiples]
        under = [item for item in candidates if item[0] < self.coverage_target]
        if not under:
            return None
        fewest = min(count for count, _ in under)
        tied = [multiple for count, multiple in under if count == fewest]
        return self._rng.choice(tied)

    # --- what exploration bought -----------------------------------------

    def record(self, *, cell: str, landed: bool) -> None:
        """One exploratory attempt's outcome, into the coverage grid."""
        if not cell or "@" not in cell:
            return
        bucket, _, multiple_text = cell.partition("@")
        try:
            multiple = float(multiple_text)
        except ValueError:
            return
        entry = self._cells.setdefault((bucket, multiple), [0, 0])
        entry[0] += 1
        if landed:
            entry[1] += 1

    def _roll_window(self, now: float) -> None:
        if now - self._window_started >= 3_600.0:
            self._window_started = now
            self._spent_lamports = 0

    def report(self) -> Dict[str, Any]:
        """Coverage of the grid, and what it has cost.

        The useful line is `cells_covered`: until it is most of the grid, the
        landing model's estimates outside the exploited region are priors
        wearing the clothes of measurements.
        """
        total_cells = len(CONGESTION_BUCKETS) * len(self.bid_multiples)
        covered = sum(1 for counts in self._cells.values()
                      if counts[0] >= self.coverage_target)
        rows = []
        for (bucket, multiple), (attempts, lands) in sorted(self._cells.items()):
            rows.append({"cell": f"{bucket}@{multiple:.2f}", "attempts": attempts,
                         "landed": lands,
                         "landing_rate": (lands / attempts) if attempts else None})
        return {
            "schema": BID_EXPLORER_SCHEMA_VERSION,
            "status": "OK" if self.explorations else "DATA_BLOCKED",
            "detail": ("" if self.explorations else
                       "no exploratory bid placed yet; the landing curve is "
                       "observed only where it is already exploited"),
            "explorations": self.explorations,
            "declined": self.declined,
            "cells_covered": covered,
            "cells_total": total_cells,
            "coverage_target_per_cell": self.coverage_target,
            "budget_spent_this_hour": self._spent_lamports,
            "hourly_budget": self.hourly_budget_lamports,
            "cells": rows,
            "last_choice": (self.last_choice.to_dict() if self.last_choice else None),
        }

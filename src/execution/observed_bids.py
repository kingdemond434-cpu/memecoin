"""What the competition actually pays to land, measured off the same stream.

The landing model conditions on bid, congestion, leader and account
contention, and chooses a bid economically. It also contains almost no
data, because a bid model learns from ATTEMPTS and the desk in DRY_RUN
makes none. That is the correct behaviour and it is also a chicken and egg:
the model cannot be good until the desk trades, and the desk should not
trade on a model that is not good.

There is a free corpus sitting in the stream the desk already consumes.
Every buy and sell on the program, from every other operator, arrives with
its own ComputeBudget instructions attached -- the compute unit price they
chose and the limit they set. Thousands an hour, on exactly the launches
this desk is deciding about, at exactly the ages that matter.

What that corpus is, and what it is NOT, decides how it may be used:

**It is the distribution of WINNING bids.** Every transaction visible on
the stream landed, by construction. So this measures what was sufficient,
per launch age and congestion -- a competitive reference the desk can price
against.

**It is not P(land | bid).** The transactions that paid too little and were
dropped are invisible here, and no amount of data fixes that: it is
survivorship, exactly the bias the reconstruction path is stamped for. A
model fitted to this and read as a landing probability would be confidently
wrong in the direction that says every bid works.

So it is reported as a bid DISTRIBUTION with the censoring stated, and it
conditions the desk's bid the way a market quote conditions a price -- as
what others paid, never as what we would have needed.
"""

from __future__ import annotations

import logging
import struct
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

OBSERVED_BIDS_SCHEMA_VERSION = "v1"

#: The ComputeBudget program. Not an Anchor program: its instructions are a
#: single leading tag byte followed by a fixed little-endian payload, which
#: is why this decoder is nine lines rather than a discriminator table.
COMPUTE_BUDGET_PROGRAM = "ComputeBudget111111111111111111111111111111"

#: Instruction tags, from the published ComputeBudget instruction enum.
TAG_REQUEST_HEAP_FRAME = 1
TAG_SET_COMPUTE_UNIT_LIMIT = 2
TAG_SET_COMPUTE_UNIT_PRICE = 3

#: Launch-age buckets, in seconds. The first three are where a snipe lives
#: and where the bid war actually happens; the rest exist so the early
#: numbers have something to be compared against.
AGE_BUCKETS: Tuple[float, ...] = (1.0, 3.0, 10.0, 60.0, 600.0)

#: Samples kept per bucket. Enough for a stable p90, bounded so a busy day
#: cannot grow this without limit.
BUCKET_SAMPLES = 4096

#: Below this many samples a percentile is not reported. A bid taken from
#: nine observations is a number that will be acted on and should not be.
MIN_SAMPLES = 100


@dataclass
class ComputeBudget:
    """What a transaction asked the runtime for."""

    unit_price_micro_lamports: Optional[int] = None
    unit_limit: Optional[int] = None

    @property
    def stated(self) -> bool:
        """Whether the sender said anything at all about priority.

        A transaction with no ComputeBudget instruction is not bidding zero
        in any meaningful sense -- it is taking the default -- and mixing
        those into a bid distribution would drag every percentile toward a
        number nobody chose.
        """
        return self.unit_price_micro_lamports is not None

    @property
    def priority_lamports(self) -> Optional[float]:
        """What the priority actually costs, which is price TIMES limit.

        A unit price alone is not comparable across transactions: paying
        1,000 micro-lamports per unit on a 40,000-unit budget and on a
        400,000-unit budget are ten times apart in what the leader
        receives, and the leader is ordering by the total.
        """
        if self.unit_price_micro_lamports is None:
            return None
        limit = self.unit_limit if self.unit_limit else 200_000
        return self.unit_price_micro_lamports * limit / 1_000_000.0


def decode_compute_budget(instructions: Sequence[Tuple[str, bytes]]
                          ) -> ComputeBudget:
    """Read the priority a transaction declared, from its own instructions.

    Takes `(program_id, data)` pairs so the caller keeps ownership of how
    keys are resolved -- this module has no business knowing how a
    transaction is shaped, only what these two instructions mean.
    """
    budget = ComputeBudget()
    for program_id, data in instructions:
        if program_id != COMPUTE_BUDGET_PROGRAM or not data:
            continue
        tag = data[0]
        try:
            if tag == TAG_SET_COMPUTE_UNIT_PRICE and len(data) >= 9:
                budget.unit_price_micro_lamports = int(
                    struct.unpack_from("<Q", data, 1)[0])
            elif tag == TAG_SET_COMPUTE_UNIT_LIMIT and len(data) >= 5:
                budget.unit_limit = int(struct.unpack_from("<I", data, 1)[0])
        except struct.error:
            # A malformed budget instruction is not a reason to lose the
            # transaction it rode in on.
            continue
    return budget


@dataclass
class BidBucket:
    label: str
    samples: Deque[float] = field(default_factory=lambda: deque(maxlen=BUCKET_SAMPLES))
    #: Transactions that declared no priority at all. Counted separately,
    #: never as a zero bid -- see ComputeBudget.stated.
    silent: int = 0

    def percentile(self, fraction: float) -> Optional[float]:
        if len(self.samples) < MIN_SAMPLES:
            return None
        ordered = sorted(self.samples)
        index = min(len(ordered) - 1, max(0, int(fraction * len(ordered))))
        return float(ordered[index])

    def as_dict(self) -> Dict[str, Any]:
        return {
            "bucket": self.label,
            "samples": len(self.samples),
            "silent": self.silent,
            "silent_share": (self.silent / (self.silent + len(self.samples))
                             if (self.silent or self.samples) else None),
            "p50_lamports": self.percentile(0.50),
            "p90_lamports": self.percentile(0.90),
            "p99_lamports": self.percentile(0.99),
            "status": "OK" if len(self.samples) >= MIN_SAMPLES else "DATA_BLOCKED",
        }


class ObservedBidCorpus:
    """The bid distribution of transactions that landed, by launch age."""

    def __init__(self, age_buckets: Sequence[float] = AGE_BUCKETS):
        self.age_buckets = tuple(age_buckets)
        self.buckets: Dict[str, BidBucket] = {
            label: BidBucket(label) for label in self._labels()}
        self.observed = 0
        self.without_age = 0

    def _labels(self) -> List[str]:
        labels = []
        previous = 0.0
        for edge in self.age_buckets:
            labels.append(f"{previous:g}-{edge:g}s")
            previous = edge
        labels.append(f">{previous:g}s")
        return labels

    def _label_for(self, age_s: float) -> str:
        previous = 0.0
        for edge in self.age_buckets:
            if age_s < edge:
                return f"{previous:g}-{edge:g}s"
            previous = edge
        return f">{previous:g}s"

    def observe(self, budget: ComputeBudget, age_s: Optional[float]) -> bool:
        """One landed transaction on a launch of known age."""
        if age_s is None or age_s < 0:
            # Without an age this says nothing about the bid war at T0,
            # which is the only part of the distribution worth having.
            self.without_age += 1
            return False
        bucket = self.buckets[self._label_for(float(age_s))]
        if not budget.stated:
            bucket.silent += 1
            return False
        priority = budget.priority_lamports
        if priority is None or priority <= 0:
            bucket.silent += 1
            return False
        bucket.samples.append(priority)
        self.observed += 1
        return True

    def reference_bid(self, age_s: float,
                      fraction: float = 0.90) -> Optional[float]:
        """What others paid at this age, in lamports of priority.

        A REFERENCE, not a requirement: everything in this corpus landed, so
        the transactions that paid less and were dropped are invisible. It
        answers "what did the competition pay", which is a market quote, and
        never "what would we have needed", which is a landing probability
        this data structurally cannot support.
        """
        return self.buckets[self._label_for(float(age_s))].percentile(fraction)

    def report(self) -> Dict[str, Any]:
        measured = [bucket for bucket in self.buckets.values()
                    if len(bucket.samples) >= MIN_SAMPLES]
        return {
            "schema": OBSERVED_BIDS_SCHEMA_VERSION,
            "status": "OK" if measured else "DATA_BLOCKED",
            "observed": self.observed,
            "without_age": self.without_age,
            "buckets": [self.buckets[label].as_dict() for label in self._labels()],
            "censoring": ("every transaction here LANDED, so this is the "
                          "distribution of winning bids and not P(land|bid); "
                          "the ones that paid too little are invisible and no "
                          "amount of data fixes that"),
            "detail": ("what other operators paid to land on the launches "
                       "this desk was deciding about, read off the same "
                       "stream, at the ages that matter"),
        }

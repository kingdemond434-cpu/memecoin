"""What following a wallet is actually worth, measured forward.

The wallet ranking was a hand-weighted composite: early-entry quality at 0.25,
forward return at 0.30, consistency at 0.20, independence at 0.15, sample size
at 0.10, multiplied by `1 - rug_exposure * 0.5` and `1 - copy_crowding * 0.3`,
and a wallet was "top" above 0.5. Every one of those numbers was chosen rather
than measured, and nothing in the system could tell you whether following the
resulting list made or lost money.

This asks the only question that has an answer: **if we had followed this
wallet, at the fill we could actually have got, what would it have done to our
capital?** That is E[log W], the same objective the desk maximises everywhere
else -- so a wallet's value is directly comparable to every other action's
value instead of being a score on its own private scale.

Three things make it *executable* rather than flattering:

* The entry is OURS, not theirs. A wallet that buys at the first slot and is
  up 3x by the time its transaction is visible to us has no edge we can take;
  the outcome recorded here starts from the price we could have paid.
* The exit is bounded by capacity. Realising a 40x that only existed for a
  size we could not have sold is not a return, it is a screenshot.
* Fees and the rug are in the number. A rug is a -100% observation, not a
  separate "rug_exposure" penalty applied with a coefficient somebody picked.

And the ranking is by a LOWER CONFIDENCE BOUND, not the mean. A wallet with
six lucky trades otherwise tops every list, forever, which is exactly how a
follow-the-smart-money system ends up following noise.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

WALLET_VALUE_SCHEMA_VERSION = "v1"

# Below this a wallet has no value estimate at all -- not a low one. Twelve
# followed trades is already generous for a heavy-tailed distribution; it is
# set here as the point below which the confidence bound would be so wide that
# reporting it invites reading the mean instead.
DEFAULT_MIN_SAMPLES = 12

# One-sided normal quantile for the confidence bound. 1.645 is 95%: we rank on
# what the wallet is worth in the bad case, because the cost of following a
# wallet that got lucky is paid in capital and the cost of skipping a good one
# is paid in opportunity.
DEFAULT_Z = 1.645

# How many followed outcomes are kept per wallet. Bounded, and oldest-first:
# a wallet that was good two years ago and is now exit liquidity should stop
# being ranked on the two-year-old trades.
DEFAULT_HISTORY = 500

# A total loss is not -inf. log(0) would make one rug dominate any amount of
# evidence, which is arithmetically true of a full-bankroll bet and false of
# how the desk actually sizes -- so the floor is what a position sized at the
# desk's cap can actually lose.
MIN_MULTIPLE = 1e-4


@dataclass(frozen=True)
class FollowOutcome:
    """One trade we could have taken by following a wallet, and what it did.

    ``executable_multiple`` is the whole point: it is measured from OUR
    achievable entry to OUR achievable exit, not from theirs.
    """

    wallet: str
    token: str
    observed_at: float
    executable_multiple: float
    regime: str = ""
    # What fraction of our intended size the exit could actually absorb. A
    # trade we could only half exit realised half the move and held the rest
    # into whatever came next.
    capacity_ratio: float = 1.0
    rugged: bool = False
    # Seconds between the wallet's entry being visible and ours landing. Kept
    # so a wallet whose edge decays inside our latency is identifiable as
    # exactly that, rather than as a wallet with no edge.
    follow_latency_s: Optional[float] = None
    data_status: str = "OK"

    @property
    def log_return(self) -> float:
        """log of the multiple, floored. The quantity E[log W] averages."""
        return math.log(max(MIN_MULTIPLE, float(self.executable_multiple)))


@dataclass
class WalletValue:
    """A wallet's forward value, or why it does not have one yet."""

    status: str
    wallet: str = ""
    regime: str = ""
    samples: int = 0
    # E[log(1+r)] per followed trade.
    mean_log_return: float = 0.0
    # The rankable number: the mean's lower confidence bound, shrunk toward
    # the population. Positive means following this wallet grew capital with
    # confidence, not merely on average.
    lower_bound: float = 0.0
    stdev: float = 0.0
    rug_rate: float = 0.0
    median_capacity: float = 1.0
    median_latency_s: Optional[float] = None
    last_outcome_at: float = 0.0
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "OK"

    @property
    def followable(self) -> bool:
        """Whether following this wallet is positive-growth with confidence."""
        return self.ok and self.lower_bound > 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": WALLET_VALUE_SCHEMA_VERSION, "status": self.status,
            "wallet": self.wallet, "regime": self.regime, "samples": self.samples,
            "mean_log_return": round(self.mean_log_return, 6),
            "lower_bound": round(self.lower_bound, 6),
            "stdev": round(self.stdev, 6), "rug_rate": round(self.rug_rate, 4),
            "median_capacity": round(self.median_capacity, 4),
            "median_latency_s": self.median_latency_s,
            "followable": self.followable, "detail": self.detail,
        }


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


class WalletValueModel:
    """Forward E[log W] per wallet, ranked by lower confidence bound.

    Holds outcomes, not scores. Everything reported is derived on demand from
    the recorded trades, so there is no cached number that can drift away from
    the evidence that produced it.
    """

    def __init__(self, *, min_samples: int = DEFAULT_MIN_SAMPLES,
                 z: float = DEFAULT_Z, history: int = DEFAULT_HISTORY,
                 shrinkage_strength: float = 10.0):
        self.min_samples = max(2, int(min_samples))
        self.z = float(z)
        self.history = int(history)
        # Pseudo-observations of the population mean blended into each
        # wallet's estimate. It is what stops a wallet with fifteen trades and
        # one 60x from being ranked as though that 60x were its expectation.
        self.shrinkage_strength = max(0.0, float(shrinkage_strength))
        self._outcomes: Dict[str, Deque[FollowOutcome]] = defaultdict(
            lambda: deque(maxlen=self.history))
        self.rejected = 0

    # --- recording -------------------------------------------------------

    def record(self, outcome: FollowOutcome) -> bool:
        """Add one followed outcome. Returns whether it was accepted.

        An outcome whose own data status is blocked is REFUSED, not recorded
        at its face value: a trade whose executable multiple could not be
        measured is not a trade that returned 1.0, and averaging it in as one
        would pull every wallet toward break-even with fictional evidence.
        """
        if outcome.data_status != "OK" or not outcome.wallet:
            self.rejected += 1
            return False
        if outcome.executable_multiple <= 0 and not outcome.rugged:
            self.rejected += 1
            return False
        self._outcomes[outcome.wallet].append(outcome)
        return True

    def record_many(self, outcomes: Sequence[FollowOutcome]) -> int:
        return sum(1 for outcome in outcomes if self.record(outcome))

    # --- estimation ------------------------------------------------------

    def population_mean(self) -> Optional[float]:
        """Mean log return across every recorded outcome.

        The shrinkage target. A wallet with few trades is pulled toward what
        following a random tracked wallet does, which is the honest prior:
        not zero, and not its own six lucky trades.
        """
        total, count = 0.0, 0
        for outcomes in self._outcomes.values():
            for outcome in outcomes:
                total += outcome.log_return
                count += 1
        return (total / count) if count else None

    def value(self, wallet: str, regime: str = "",
              now: Optional[float] = None) -> WalletValue:
        """This wallet's forward value, or DATA_BLOCKED with the reason."""
        outcomes = [item for item in self._outcomes.get(wallet, ())
                    if not regime or item.regime == regime]
        if len(outcomes) < self.min_samples:
            return WalletValue(
                status="DATA_BLOCKED", wallet=wallet, regime=regime,
                samples=len(outcomes),
                detail=f"{len(outcomes)} followed outcomes; {self.min_samples} needed")

        returns = [outcome.log_return for outcome in outcomes]
        count = len(returns)
        mean = sum(returns) / count
        variance = sum((value - mean) ** 2 for value in returns) / max(1, count - 1)
        stdev = math.sqrt(max(0.0, variance))

        prior = self.population_mean()
        if prior is None or self.shrinkage_strength <= 0:
            shrunk = mean
        else:
            weight = count / (count + self.shrinkage_strength)
            shrunk = weight * mean + (1 - weight) * prior
        lower = shrunk - self.z * stdev / math.sqrt(count)

        return WalletValue(
            status="OK", wallet=wallet, regime=regime, samples=count,
            mean_log_return=mean, lower_bound=lower, stdev=stdev,
            rug_rate=sum(1 for outcome in outcomes if outcome.rugged) / count,
            median_capacity=_median([outcome.capacity_ratio for outcome in outcomes]),
            median_latency_s=(_median([outcome.follow_latency_s for outcome in outcomes
                                       if outcome.follow_latency_s is not None])
                              or None),
            last_outcome_at=max(outcome.observed_at for outcome in outcomes))

    def rank(self, limit: int = 50, regime: str = "",
             followable_only: bool = True) -> List[WalletValue]:
        """Wallets ordered by lower bound, best first.

        Not by mean. Ranking on the mean is what puts a wallet with six lucky
        trades at the top of the list and keeps it there.
        """
        values = [self.value(wallet, regime) for wallet in self._outcomes]
        usable = [value for value in values
                  if value.ok and (value.followable or not followable_only)]
        usable.sort(key=lambda value: value.lower_bound, reverse=True)
        return usable[:limit]

    def report(self, now: Optional[float] = None) -> Dict[str, Any]:
        """Whether the model can rank anything yet, and on what evidence."""
        tracked = len(self._outcomes)
        observations = sum(len(items) for items in self._outcomes.values())
        estimable = [self.value(wallet) for wallet in self._outcomes]
        ok = [value for value in estimable if value.ok]
        followable = [value for value in ok if value.followable]
        return {
            "schema": WALLET_VALUE_SCHEMA_VERSION,
            "status": "OK" if ok else "DATA_BLOCKED",
            "detail": ("" if ok else
                       f"{tracked} wallets tracked, none with {self.min_samples} "
                       "followed outcomes yet; no wallet is ranked"),
            "wallets_tracked": tracked, "observations": observations,
            "wallets_estimable": len(ok), "wallets_followable": len(followable),
            "rejected_outcomes": self.rejected,
            "min_samples": self.min_samples,
            "population_mean_log_return": self.population_mean(),
            "top": [value.to_dict() for value in self.rank(limit=10)],
        }


def executable_multiple(their_entry_price: float, our_entry_price: float,
                        our_exit_price: float, *, fee_bps: float = 0.0,
                        capacity_ratio: float = 1.0,
                        remainder_multiple: Optional[float] = None
                        ) -> Tuple[float, str]:
    """What following this trade would have returned US, and its data status.

    ``their_entry_price`` is not used in the arithmetic and is required
    anyway: the caller has to have looked it up, and a caller that cannot
    distinguish their fill from ours is a caller that will eventually pass
    theirs.

    When only part of the position could be exited, the rest is NOT marked at
    the exit price -- selling it is what moves that price, which is why it
    could not be sold. So a partial exit needs ``remainder_multiple``: what
    the stuck fraction actually became. Without it the outcome is
    DATA_BLOCKED, because the difference between a 10x on a tenth of the
    position and a 10x on all of it is the entire question.
    """
    if our_entry_price <= 0 or our_exit_price < 0 or their_entry_price <= 0:
        return 0.0, "DATA_BLOCKED: prices not measured"
    if not 0.0 < capacity_ratio <= 1.0:
        return 0.0, "DATA_BLOCKED: capacity ratio out of range"
    gross = our_exit_price / our_entry_price
    net = gross * (1 - max(0.0, fee_bps) / 10_000)
    if capacity_ratio >= 1.0:
        return max(0.0, net), "OK"
    if remainder_multiple is None:
        return 0.0, ("DATA_BLOCKED: only "
                     f"{capacity_ratio:.0%} was liquidatable and the remainder "
                     "was never marked")
    remainder = max(0.0, float(remainder_multiple))
    return (max(0.0, capacity_ratio * net + (1 - capacity_ratio) * remainder), "OK")

"""The landing curve the Attempt record was built for, and the bid it implies.

`landing_model.Attempt` has carried `leader`, `account_contention`,
`blockhash_age_slots` and `slot_value` since it was written, with a docstring
saying in as many words that they exist "for models that do not exist yet". The
models still did not exist. Every bid the desk placed was priced off a curve
conditioned on one thing -- congestion -- while the record on disk held the
three variables that actually decide whether a launch snipe lands.

This is those models.

**Conditioning, with the backoff named.** P(land) is looked up on the most
specific cell that has support -- leader x contention x bid -- and falls back
through contention x bid, congestion x bid, and finally the pooled bid curve.
Every estimate says which rung answered it, because a probability from the
pooled curve and one from this validator in this contention state are different
claims and a caller that cannot tell them apart will size them the same.

**Monotonicity is enforced, not assumed.** Empirical buckets are not monotone
in bid: on a few hundred attempts the 200k bucket routinely lands more often
than the 500k one, purely by sampling. An optimiser handed that curve picks the
bump and reports a confident recommendation to underbid. Pool-adjacent-violators
projects the curve onto the monotone cone, which is the smallest correction that
makes "more lamports never lands less often" true.

**Blockhash age is a separate hazard, not another bucket.** A stale blockhash
does not lose a race; it expires, and the transaction is rejected regardless of
what it paid. Folding that into the bid curve would teach the model that high
bids fail, when what failed was the clock. It multiplies:

    P(land) = P(win the slot | bid, leader, contention) x P(not yet expired | age)

**The bid is where the next lamport stops buying a lamport.** `recommend`
already maximised `P*edge - bid` by scanning; what it could not say is the
MARGINAL value of the next lamport, which is the quantity the desk actually
wants -- it makes "bid more on this one" a measurement rather than a policy.
The optimiser walks the curve and stops at the first rung where dEV/dlamport
falls below 1.0: past that point a lamport spent returns less than a lamport.

**Paper and real are never pooled.** `Attempt.real` exists and its own comment
says a curve fitted across both "describes neither" -- and the existing model
pools them anyway. In dry run nothing is submitted, so paper attempts carry
whatever the simulator decided; mixing them into the curve that prices real
bids is how a desk learns a landing rate it has never observed. Every estimate
here declares which population it came from.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.execution.landing_model import (BID_BUCKETS, Attempt, bid_bucket,
                                         congestion_bucket, contention_bucket)

logger = logging.getLogger(__name__)

BID_CURVE_SCHEMA_VERSION = "v1"

#: Attempts a cell needs before it may answer. Conditioned cells are small by
#: construction -- there are thousands of validators -- so this is the number
#: below which a rate is a decimal point rather than a measurement.
DEFAULT_MIN_CELL = 30

#: Bid rungs the optimiser scans, in lamports.
#:
#: These are the landing model's OWN bid buckets, not a finer ladder. A finer
#: one is worse than useless: two rungs inside one bucket return the identical
#: probability, so the marginal between them is exactly zero, and a walk that
#: stops when the marginal falls below 1.0 stops on that artefact before it
#: reaches a rung the curve can actually distinguish. The curve's resolution is
#: the bucket; scanning below it measures the bucketing, not the network.
DEFAULT_BID_RUNGS: Tuple[int, ...] = tuple(BID_BUCKETS)

#: Solana blockhashes are valid for 150 slots. Past that the transaction is
#: rejected for an expired hash however much it paid.
BLOCKHASH_VALIDITY_SLOTS = 150

#: Below this many observations the expiry hazard is taken from the published
#: validity window rather than fitted, and said so.
DEFAULT_MIN_EXPIRY_SAMPLES = 50


def _rate(counts: Sequence[int]) -> Optional[float]:
    total = counts[0] if counts else 0
    return (counts[1] / total) if total else None


def isotonic(points: Sequence[Tuple[int, float, int]]
             ) -> List[Tuple[int, float, int]]:
    """Project (bid, rate, weight) onto a non-decreasing curve, by PAVA.

    Weighted pool-adjacent-violators. Where two adjacent rungs disagree with
    the monotone constraint they are merged into their weighted mean, which is
    the least-squares-optimal monotone fit and, unlike smoothing, changes
    nothing where the data already behaves.
    """
    blocks: List[List[float]] = []   # [sum(weight*value), weight, max_bid]
    for bid, value, weight in sorted(points, key=lambda item: item[0]):
        if weight <= 0:
            continue
        blocks.append([float(value) * weight, float(weight), float(bid)])
        while len(blocks) >= 2 and (
                blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]):
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([left[0] + right[0], left[1] + right[1], right[2]])
    result: List[Tuple[int, float, int]] = []
    index = 0
    ordered = sorted(points, key=lambda item: item[0])
    for total, weight, _ in blocks:
        mean = total / weight
        consumed = 0.0
        while index < len(ordered) and consumed < weight - 1e-9:
            bid, _value, item_weight = ordered[index]
            if item_weight <= 0:
                index += 1
                continue
            result.append((int(bid), float(mean), int(item_weight)))
            consumed += item_weight
            index += 1
    return result


@dataclass
class ConditionedEstimate:
    """P(land), and exactly which population produced it."""

    status: str
    probability: Optional[float] = None
    attempts: int = 0
    #: Which backoff rung answered: leader_contention, contention, congestion,
    #: pooled -- or the reason nothing could.
    basis: str = ""
    population: str = "real"
    expiry_survival: Optional[float] = None
    detail: str = ""

    @property
    def landed_probability(self) -> Optional[float]:
        """P(land) including the expiry hazard, which is what a caller wants."""
        if self.probability is None:
            return None
        if self.expiry_survival is None:
            return self.probability
        return self.probability * self.expiry_survival

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["landed_probability"] = self.landed_probability
        return data


@dataclass
class MarginalRung:
    bid_lamports: int
    probability: Optional[float]
    expected_value_lamports: Optional[float]
    #: dEV / dlamport between this rung and the previous one. Above 1.0 the
    #: next lamport is worth spending; below it, it is a transfer.
    marginal: Optional[float] = None
    basis: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BidDecision:
    status: str
    bid_lamports: int = 0
    expected_value_lamports: Optional[float] = None
    probability: Optional[float] = None
    basis: str = ""
    marginal_at_choice: Optional[float] = None
    rungs: List[MarginalRung] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["rungs"] = [rung.to_dict() for rung in self.rungs]
        return data


class BlockhashExpiryModel:
    """P(the hash is still valid) as a function of its age in slots.

    Fitted from attempts where the failure names an expiry; falls back to the
    published 150-slot window when there is not enough evidence, and says which
    it used. A desk that has never had a hash expire should not conclude that
    hashes never expire.
    """

    def __init__(self, *, validity_slots: int = BLOCKHASH_VALIDITY_SLOTS,
                 min_samples: int = DEFAULT_MIN_EXPIRY_SAMPLES):
        self.validity_slots = int(validity_slots)
        self.min_samples = int(min_samples)
        #: age bucket -> [attempts, expired]
        self._counts: Dict[int, List[int]] = {}

    @staticmethod
    def _bucket(age_slots: int) -> int:
        return max(0, int(age_slots) // 10 * 10)

    def observe(self, attempt: Attempt) -> bool:
        if attempt.blockhash_age_slots is None:
            return False
        bucket = self._bucket(attempt.blockhash_age_slots)
        counts = self._counts.setdefault(bucket, [0, 0])
        counts[0] += 1
        expired = (not attempt.landed
                   and "expire" in (attempt.failure or "").lower())
        counts[1] += int(expired)
        return True

    def survival(self, age_slots: Optional[int]) -> Tuple[Optional[float], str]:
        if age_slots is None:
            return None, "unmeasured"
        age = int(age_slots)
        if age >= self.validity_slots:
            return 0.0, "past the published validity window"
        counts = self._counts.get(self._bucket(age))
        if counts and counts[0] >= self.min_samples:
            return max(0.0, 1.0 - counts[1] / counts[0]), "fitted"
        # Linear decay to the published cliff. Deliberately crude and
        # deliberately stated: it is a stand-in until the desk has expired
        # enough transactions to fit one, not a claim about the network.
        return max(0.0, 1.0 - age / float(self.validity_slots)), "published_window"


class ConditionedLandingCurve:
    """Landing probability conditioned on what actually decides a launch snipe.

    Reads the same `Attempt` records `LandingModel` does, and keeps its own
    cells so the two can disagree visibly rather than one silently reshaping
    the other.
    """

    def __init__(self, *, min_cell: int = DEFAULT_MIN_CELL,
                 real_only: bool = True):
        self.min_cell = max(1, int(min_cell))
        #: The whole point of separating them. A curve fitted across paper and
        #: real describes neither, and it is the real one that prices bids.
        self.real_only = bool(real_only)
        self.expiry = BlockhashExpiryModel()
        self._leader_contention: Dict[Tuple[str, int, int], List[int]] = {}
        self._contention: Dict[Tuple[int, int], List[int]] = {}
        self._congestion: Dict[Tuple[str, int], List[int]] = {}
        self._pooled: Dict[int, List[int]] = {}
        self.observed = 0
        self.skipped_population = 0

    # -- ingest ----------------------------------------------------------

    def record(self, attempt: Attempt) -> bool:
        if self.real_only and not attempt.real:
            self.skipped_population += 1
            return False
        self.observed += 1
        bucket = bid_bucket(int(attempt.bid_lamports))
        landed = int(bool(attempt.landed))

        def _bump(store: Dict[Any, List[int]], key: Any) -> None:
            counts = store.setdefault(key, [0, 0])
            counts[0] += 1
            counts[1] += landed

        _bump(self._pooled, bucket)
        _bump(self._congestion, (congestion_bucket(attempt.congestion), bucket))
        if attempt.account_contention is not None:
            contention = contention_bucket(attempt.account_contention)
            _bump(self._contention, (contention, bucket))
            if attempt.leader:
                _bump(self._leader_contention,
                      (attempt.leader, contention, bucket))
        self.expiry.observe(attempt)
        return True

    def extend(self, attempts: Iterable[Attempt]) -> int:
        return sum(int(self.record(attempt)) for attempt in attempts)

    # -- lookup ----------------------------------------------------------

    def estimate(self, bid_lamports: int, *, leader: str = "",
                 account_contention: Optional[int] = None,
                 congestion: Optional[float] = None,
                 blockhash_age_slots: Optional[int] = None
                 ) -> ConditionedEstimate:
        bucket = bid_bucket(int(bid_lamports))
        population = "real" if self.real_only else "mixed"
        survival, survival_basis = self.expiry.survival(blockhash_age_slots)

        ladder: List[Tuple[str, Optional[List[int]]]] = []
        if leader and account_contention is not None:
            ladder.append(("leader_contention", self._leader_contention.get(
                (leader, contention_bucket(account_contention), bucket))))
        if account_contention is not None:
            ladder.append(("contention", self._contention.get(
                (contention_bucket(account_contention), bucket))))
        ladder.append(("congestion", self._congestion.get(
            (congestion_bucket(congestion), bucket))))
        ladder.append(("pooled", self._pooled.get(bucket)))

        for basis, counts in ladder:
            if counts and counts[0] >= self.min_cell:
                return ConditionedEstimate(
                    status="OK", probability=_rate(counts),
                    attempts=counts[0], basis=basis, population=population,
                    expiry_survival=survival,
                    detail=f"expiry from {survival_basis}")
        have = (self._pooled.get(bucket) or [0, 0])[0]
        return ConditionedEstimate(
            status="DATA_BLOCKED", attempts=have, basis="none",
            population=population, expiry_survival=survival,
            detail=(f"no cell with {self.min_cell} attempts at bid bucket "
                    f"{bucket}; the most populated had {have}"))

    def monotone_curve(self, *, leader: str = "",
                       account_contention: Optional[int] = None,
                       congestion: Optional[float] = None,
                       rungs: Sequence[int] = DEFAULT_BID_RUNGS
                       ) -> List[Tuple[int, float, str]]:
        """The landing curve across bid rungs, forced non-decreasing.

        Unenforced, the empirical curve is not monotone at these sample sizes,
        and an optimiser handed a non-monotone curve confidently recommends
        underbidding into whichever bucket got lucky.
        """
        raw: List[Tuple[int, float, int]] = []
        bases: Dict[int, str] = {}
        # One rung per distinct bid bucket, whatever ladder a caller hands in.
        # Two rungs inside one bucket carry the same probability by
        # construction, and a zero marginal between them is an artefact of the
        # bucketing rather than a statement about landing.
        seen_buckets: set = set()
        deduped: List[int] = []
        for rung in sorted(set(int(value) for value in rungs)):
            bucket = bid_bucket(rung)
            if bucket in seen_buckets:
                continue
            seen_buckets.add(bucket)
            deduped.append(rung)
        for rung in deduped:
            estimate = self.estimate(rung, leader=leader,
                                     account_contention=account_contention,
                                     congestion=congestion)
            if estimate.probability is None:
                continue
            raw.append((int(rung), float(estimate.probability),
                        max(1, estimate.attempts)))
            bases[int(rung)] = estimate.basis
        if not raw:
            return []
        return [(bid, value, bases.get(bid, ""))
                for bid, value, _ in isotonic(raw)]

    # -- the decision ----------------------------------------------------

    def optimise(self, *, edge_lamports: float, leader: str = "",
                 account_contention: Optional[int] = None,
                 congestion: Optional[float] = None,
                 blockhash_age_slots: Optional[int] = None,
                 max_bid_lamports: int = DEFAULT_BID_RUNGS[-1],
                 rungs: Sequence[int] = DEFAULT_BID_RUNGS) -> BidDecision:
        """Bid until the next lamport buys less than a lamport.

        `edge_lamports` is what landing is worth -- the position's expected
        value if the transaction lands, in the same unit as the bid, because a
        marginal rate is only meaningful when numerator and denominator share
        a unit.
        """
        if edge_lamports <= 0:
            return BidDecision(status="REJECTED",
                               detail="no edge to bid for")
        curve = self.monotone_curve(leader=leader,
                                    account_contention=account_contention,
                                    congestion=congestion, rungs=rungs)
        if not curve:
            return BidDecision(
                status="DATA_BLOCKED",
                detail="the landing curve cannot answer at any bid rung; the "
                       "caller must fall back to its configured ladder and "
                       "know that it is doing so")
        survival, _ = self.expiry.survival(blockhash_age_slots)
        scale = 1.0 if survival is None else survival

        walked: List[MarginalRung] = []
        best_bid = 0
        best_value: Optional[float] = None
        chosen_marginal: Optional[float] = None
        previous: Optional[Tuple[int, float]] = None
        for bid, probability, basis in curve:
            if bid > max_bid_lamports:
                break
            landed = probability * scale
            value = landed * float(edge_lamports) - float(bid)
            marginal: Optional[float] = None
            if previous is not None:
                delta_bid = bid - previous[0]
                if delta_bid > 0:
                    marginal = ((landed - previous[1]) * float(edge_lamports)
                                / delta_bid)
            walked.append(MarginalRung(
                bid_lamports=bid, probability=landed,
                expected_value_lamports=value, marginal=marginal, basis=basis))
            if best_value is None or value > best_value:
                best_value = value
                best_bid = bid
                chosen_marginal = marginal
            previous = (bid, landed)
            # Past the point where a lamport returns less than a lamport, every
            # further rung is a transfer. Stop walking rather than scanning the
            # whole ladder and rediscovering the same maximum.
            if marginal is not None and marginal < 1.0:
                break

        if best_value is None or best_value <= 0:
            return BidDecision(
                status="REJECTED", bid_lamports=0, rungs=walked,
                expected_value_lamports=best_value,
                detail="no bid on the curve has positive expected value; the "
                       "edge does not cover what landing costs")
        estimate = self.estimate(
            best_bid, leader=leader, account_contention=account_contention,
            congestion=congestion, blockhash_age_slots=blockhash_age_slots)
        return BidDecision(
            status="OK", bid_lamports=best_bid,
            expected_value_lamports=best_value,
            probability=estimate.landed_probability, basis=estimate.basis,
            marginal_at_choice=chosen_marginal, rungs=walked)

    # -- reporting -------------------------------------------------------

    def report(self) -> Dict[str, Any]:
        return {
            "schema": BID_CURVE_SCHEMA_VERSION,
            "observed": self.observed,
            "skipped_wrong_population": self.skipped_population,
            "real_only": self.real_only,
            "cells": {
                "leader_contention": len(self._leader_contention),
                "contention": len(self._contention),
                "congestion": len(self._congestion),
                "pooled": len(self._pooled),
            },
            "min_cell": self.min_cell,
            "expiry_buckets": len(self.expiry._counts),
        }

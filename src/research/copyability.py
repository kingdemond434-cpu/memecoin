"""The Copyability Gauntlet: what a wallet has to survive to be worth copying.

A wallet's realised PnL is the least transferable thing about it. It was earned
with that wallet's capital, at that wallet's latency, in that wallet's market,
and a follower inherits none of those. A wallet can show +500% and be
completely uncopyable because it bought before there was liquidity and anybody
arriving after it pays 20% more for the same tokens. Ranking on wallet PnL is
how a follower becomes exit liquidity for the wallet it admires.

So nothing here scores a wallet on what the wallet made. Every number is what
OUR capital would have done following it, and the rankable one is a lower
confidence bound on that, at the latency the desk actually has.

Six refusals worth stating, because each of them is a way this measurement
flatters itself if you let it:

**The desk's latency, not the best one.** `FollowVerdict.best_delay` picks the
delay with the highest mean, which is selection on the data: given five delays
and noise, one of them wins. The score here is fixed at the desk's measured
latency and the rest of the curve is reported as sensitivity, which is what it
is.

**The lower bound, not the mean.** A wallet with six lucky trades otherwise
tops every list forever. The bound is what the wallet is worth in the bad case,
and the bad case is the one paid for in capital.

**One trade is not an edge.** A wallet whose whole record is one 200x and
twenty-nine losses has a fine mean. `single_trade_profit_share` and the
drop-the-best-trade recomputation exist to find exactly that, and a wallet
whose bound goes negative without its best trade is reported as a wallet with
one trade, not as a survivor.

**Speed is not information.** A wallet whose follow return collapses between
50ms and 1s was never sharing anything; it was arriving early. That is a real
and common property of profitable wallets, and it makes them uncopyable rather
than unprofitable. `edge_is_speed` names it instead of burying it in an
average.

**Unmeasured is not zero, anywhere.** A wallet with no resolved decisions, a
month with no trades, an entry whose liquidity was never observed: each is
DATA_BLOCKED and says so. The alternative -- treating absence as a neutral
observation -- pulls every wallet toward the population mean with evidence that
does not exist.

**The corpus is selected.** Wallets get discovered by winning. Any statistic
over a set of celebrated addresses is biased upward by that selection, and the
warning travels with the report rather than living in a README.

Nothing here decides anything. It produces verdicts; the allocator reads them.
"""

from __future__ import annotations

import logging
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.research.benchmark_wallets import (
    FOLLOW_DELAYS_S, SELECTION_WARNING, WalletDecision)
from src.research.gauntlet import (
    LOG_FLOOR, Observation, Verdict, bootstrap_lower_bound, max_drawdown,
    probability_of_backtest_overfitting)

logger = logging.getLogger(__name__)

COPYABILITY_SCHEMA_VERSION = "v1"

#: Resolved decisions below which a wallet has no verdict. Thirty is already
#: thin for a distribution this heavy-tailed; below it every test here is
#: theatre wearing a number's clothes.
MIN_RESOLVED_DECISIONS = 30

#: Distinct calendar months the record must span. A wallet that traded one
#: excellent fortnight has been measured against one market, and memecoin
#: markets do not stay the same for two months at a time.
MIN_DISTINCT_MONTHS = 3

#: Share of total profit one trade may contribute. Above this the wallet is a
#: single event with a history attached.
MAX_SINGLE_TRADE_PROFIT_SHARE = 0.5

#: Gross wins over gross losses, both in log space. 1.0 is break-even before
#: the confidence bound has an opinion.
MIN_PROFIT_FACTOR = 1.2

#: Peak-to-trough decline of the cumulative follower log-growth curve. A
#: follower does not get to skip the drawdown by knowing it ends.
MAX_FOLLOWER_DRAWDOWN = 1.0

#: Share of followed trades that ended in a rug. Priced as -100% observations
#: in the return series too -- this is a second, blunter guard for the case
#: where a wallet's positive mean is carried by survivors of its own rugs.
MAX_RUG_RATE = 0.4

#: Probability of backtest overfitting, from CSCV over the follower returns.
MAX_COPY_PBO = 0.5

#: Fraction of the edge that may be lost between the fastest and slowest
#: simulated delay before the wallet's edge is called speed rather than
#: information.
MAX_EDGE_DECAY_SHARE = 0.7

#: The desk's own follow latency, in seconds, used as the scoring delay until
#: a measured one is supplied. Deliberately not the fastest grid point.
DEFAULT_DESK_LATENCY_S = 0.25

#: Cost of one followed round trip, as a fraction. Entry plus exit fee plus
#: expected slippage; overridden from the desk's measured cost model.
DEFAULT_COST_PER_ROUND_TRIP = 0.04

#: Cost multipliers for the sensitivity pass.
COST_MULTIPLIERS: Tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)


def _log_return(multiple: Optional[float], cost: float) -> Optional[float]:
    """Log return of following, net of one round trip. None stays None."""
    if multiple is None or multiple <= 0:
        return None
    return max(LOG_FLOOR, math.log(multiple) + math.log(max(1e-9, 1.0 - cost)))


def _month_key(timestamp: float) -> str:
    return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).strftime("%Y-%m")


def profit_factor(returns: Sequence[float]) -> Optional[float]:
    """Gross gains over gross losses. None when there is nothing to divide by.

    A wallet with no losing followed trades does not have an infinite profit
    factor; it has a sample too small to have met a loss yet, and returning
    None says so rather than printing inf next to a real number.
    """
    gains = sum(value for value in returns if value > 0)
    losses = -sum(value for value in returns if value < 0)
    if losses <= 0:
        return None
    return float(gains / losses)


def single_trade_profit_share(returns: Sequence[float]) -> Optional[float]:
    """Share of gross profit contributed by the single best followed trade."""
    gains = [value for value in returns if value > 0]
    if not gains:
        return None
    total = sum(gains)
    if total <= 0:
        return None
    return float(max(gains) / total)


def cumulative_drawdown(returns: Sequence[float]) -> Optional[float]:
    """Worst peak-to-trough decline of cumulative follower log growth.

    Returned as a POSITIVE magnitude in log units, because the criterion it is
    compared against is a positive limit and a sign flip in the middle of a
    threshold check is a bug waiting for a bad week to surface it.
    """
    if not returns:
        return None
    return abs(max_drawdown(list(returns)))


@dataclass
class CopyabilityCriteria:
    """Every threshold in one object, so a verdict can carry what produced it."""

    min_resolved: int = MIN_RESOLVED_DECISIONS
    min_months: int = MIN_DISTINCT_MONTHS
    max_single_trade_share: float = MAX_SINGLE_TRADE_PROFIT_SHARE
    min_profit_factor: float = MIN_PROFIT_FACTOR
    max_drawdown: float = MAX_FOLLOWER_DRAWDOWN
    max_rug_rate: float = MAX_RUG_RATE
    max_pbo: float = MAX_COPY_PBO
    max_edge_decay_share: float = MAX_EDGE_DECAY_SHARE
    desk_latency_s: float = DEFAULT_DESK_LATENCY_S
    cost_per_round_trip: float = DEFAULT_COST_PER_ROUND_TRIP
    #: Minimum liquidity at the wallet's entry, in USD, for a decision to be
    #: copyable at all. Zero disables the filter; None-valued observations are
    #: always counted as unmeasured rather than as passing.
    min_entry_liquidity_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CopyabilityVerdict:
    """What following this wallet is worth, and every reason it might not be."""

    status: str
    wallet: str = ""
    verdict: str = Verdict.DATA_BLOCKED.value

    #: THE rankable number: lower confidence bound on follower log growth per
    #: trade, at the desk's latency, net of costs. Everything else is why.
    follower_lower_bound: Optional[float] = None
    follower_mean: Optional[float] = None
    #: The wallet's OWN mean log return. Carried for contrast and never used
    #: in scoring -- the gap between this and follower_mean is the whole point.
    wallet_own_mean: Optional[float] = None

    resolved: int = 0
    scored: int = 0
    months: int = 0
    months_positive: Optional[float] = None

    latency_curve: Dict[float, Optional[float]] = field(default_factory=dict)
    latency_lower_curve: Dict[float, Optional[float]] = field(default_factory=dict)
    edge_decay: Optional[float] = None
    edge_decay_share: Optional[float] = None
    edge_is_speed: Optional[bool] = None

    cost_curve: Dict[float, Optional[float]] = field(default_factory=dict)
    cost_survival_multiple: Optional[float] = None

    profit_factor: Optional[float] = None
    max_drawdown: Optional[float] = None
    single_trade_share: Optional[float] = None
    #: The lower bound recomputed with the best trade removed. A wallet whose
    #: edge does not survive this has one trade, not an edge.
    lower_bound_without_best: Optional[float] = None
    rug_rate: Optional[float] = None
    median_entry_liquidity_usd: Optional[float] = None
    median_buyer_rank: Optional[float] = None
    pbo: Optional[float] = None

    reasons: List[str] = field(default_factory=list)
    blocked: List[str] = field(default_factory=list)
    criteria: Dict[str, Any] = field(default_factory=dict)
    detail: str = ""

    @property
    def copyable(self) -> bool:
        return self.verdict == Verdict.SURVIVOR.value

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["schema"] = COPYABILITY_SCHEMA_VERSION
        data["selection_warning"] = SELECTION_WARNING
        data["copyable"] = self.copyable
        for key in ("latency_curve", "latency_lower_curve", "cost_curve"):
            data[key] = {f"{k:g}": v for k, v in getattr(self, key).items()}
        return data


class CopyabilityGauntlet:
    """Turns one wallet's decisions into a verdict about copying it."""

    def __init__(self, criteria: Optional[CopyabilityCriteria] = None,
                 *, delays: Sequence[float] = FOLLOW_DELAYS_S,
                 cost_multipliers: Sequence[float] = COST_MULTIPLIERS):
        self.criteria = criteria or CopyabilityCriteria()
        self.delays = tuple(sorted(float(value) for value in delays))
        self.cost_multipliers = tuple(cost_multipliers)

    # -- conversion -------------------------------------------------------

    def observations(self, decisions: Sequence[WalletDecision],
                     mechanism: str = "") -> List[Observation]:
        """One gauntlet Observation per resolved decision, across the grid.

        Exists so a wallet is testable by exactly the machinery every other
        mechanism is tested by. A wallet that cannot be expressed as
        observations is a wallet whose claim cannot be checked.
        """
        rows: List[Observation] = []
        for decision in decisions:
            if not decision.resolved:
                continue
            by_latency: Dict[float, Optional[float]] = {}
            for delay in self.delays:
                multiple = decision.follow_multiple(delay)
                # Absent, not zero: a delay at which no fill was reachable did
                # not return nothing, it did not happen.
                by_latency[delay] = (None if multiple is None or multiple <= 0
                                     else float(multiple) - 1.0)
            rows.append(Observation(
                mechanism=mechanism or f"follow:{decision.wallet}",
                timestamp=float(decision.entered_at),
                regime=str(decision.state.get("regime", "unknown")),
                source_family=str(decision.state.get("source_family", "")),
                cohort=_month_key(decision.entered_at),
                net_return_by_latency=by_latency,
                cost_fraction=float(self.criteria.cost_per_round_trip)))
        return rows

    # -- pieces -----------------------------------------------------------

    def _returns_at(self, decisions: Sequence[WalletDecision], delay: float,
                    cost_multiplier: float = 1.0) -> List[float]:
        cost = min(0.99, self.criteria.cost_per_round_trip * cost_multiplier)
        values = [_log_return(row.follow_multiple(delay), cost) for row in decisions]
        return [value for value in values if value is not None]

    def _monthly(self, decisions: Sequence[WalletDecision], delay: float
                 ) -> Dict[str, List[float]]:
        cost = self.criteria.cost_per_round_trip
        months: Dict[str, List[float]] = {}
        for row in decisions:
            value = _log_return(row.follow_multiple(delay), cost)
            if value is None:
                continue
            months.setdefault(_month_key(row.entered_at), []).append(value)
        return months

    def _aligned_delay_matrix(self, decisions: Sequence[WalletDecision]
                              ) -> List[List[float]]:
        """delays x decisions, keeping only decisions filled at every delay."""
        cost = self.criteria.cost_per_round_trip
        columns: List[List[float]] = []
        for row in sorted(decisions, key=lambda item: item.entered_at):
            values = [_log_return(row.follow_multiple(delay), cost)
                      for delay in self.delays]
            if any(value is None for value in values):
                continue
            columns.append([value for value in values if value is not None])
        if not columns:
            return []
        return [[column[index] for column in columns]
                for index in range(len(self.delays))]

    def _median_state(self, decisions: Sequence[WalletDecision], key: str
                      ) -> Optional[float]:
        values = [float(row.state[key]) for row in decisions
                  if row.state.get(key) is not None]
        return float(statistics.median(values)) if values else None

    # -- verdict ----------------------------------------------------------

    def evaluate(self, wallet: str, decisions: Sequence[WalletDecision],
                 *, wallet_own_mean: Optional[float] = None) -> CopyabilityVerdict:
        """The whole question, answered or explicitly refused."""
        criteria = self.criteria
        resolved = [row for row in decisions if row.resolved]
        verdict = CopyabilityVerdict(
            status="DATA_BLOCKED", wallet=wallet, resolved=len(resolved),
            wallet_own_mean=wallet_own_mean, criteria=criteria.to_dict())

        if len(resolved) < criteria.min_resolved:
            verdict.blocked.append("insufficient_resolved_decisions")
            verdict.detail = (f"{len(resolved)} resolved decisions, below the "
                              f"{criteria.min_resolved} needed for a verdict")
            return verdict

        scoring = self._returns_at(resolved, criteria.desk_latency_s)
        verdict.scored = len(scoring)
        if len(scoring) < criteria.min_resolved:
            verdict.blocked.append("insufficient_fills_at_desk_latency")
            verdict.detail = (
                f"only {len(scoring)} of {len(resolved)} decisions had a "
                f"reachable fill at {criteria.desk_latency_s:g}s")
            return verdict

        verdict.status = "OK"
        verdict.follower_mean = float(sum(scoring) / len(scoring))
        verdict.follower_lower_bound = bootstrap_lower_bound(scoring)

        # Latency: the sensitivity curve, and whether the edge IS the speed.
        for delay in self.delays:
            returns = self._returns_at(resolved, delay)
            enough = len(returns) >= criteria.min_resolved
            verdict.latency_curve[delay] = (
                float(sum(returns) / len(returns)) if enough else None)
            verdict.latency_lower_curve[delay] = (
                bootstrap_lower_bound(returns) if enough else None)
        measured = {k: v for k, v in verdict.latency_curve.items() if v is not None}
        if len(measured) >= 2:
            fastest = measured[min(measured)]
            slowest = measured[max(measured)]
            verdict.edge_decay = float(fastest - slowest)
            if fastest > 0:
                verdict.edge_decay_share = float((fastest - slowest) / fastest)
                verdict.edge_is_speed = (
                    verdict.edge_decay_share > criteria.max_edge_decay_share)
        else:
            verdict.blocked.append("latency_sensitivity")

        # Cost: how much worse execution can get before the bound dies.
        for multiplier in self.cost_multipliers:
            returns = self._returns_at(resolved, criteria.desk_latency_s, multiplier)
            verdict.cost_curve[multiplier] = (
                bootstrap_lower_bound(returns)
                if len(returns) >= criteria.min_resolved else None)
        survivors = [m for m, bound in sorted(verdict.cost_curve.items())
                     if bound is not None and bound > 0]
        verdict.cost_survival_multiple = max(survivors) if survivors else None

        # Shape of the record.
        verdict.profit_factor = profit_factor(scoring)
        verdict.max_drawdown = cumulative_drawdown(scoring)
        verdict.single_trade_share = single_trade_profit_share(scoring)
        if len(scoring) > 1:
            best = max(scoring)
            trimmed = list(scoring)
            trimmed.remove(best)
            verdict.lower_bound_without_best = bootstrap_lower_bound(trimmed)

        months = self._monthly(resolved, criteria.desk_latency_s)
        verdict.months = len(months)
        if months:
            positive = sum(1 for values in months.values()
                           if sum(values) / len(values) > 0)
            verdict.months_positive = float(positive / len(months))

        rugs = [row for row in resolved if row.state.get("rugged") is not None]
        if rugs:
            verdict.rug_rate = float(
                sum(1 for row in rugs if row.state.get("rugged")) / len(rugs))
        else:
            verdict.blocked.append("rug_rate")

        verdict.median_entry_liquidity_usd = self._median_state(
            resolved, "liquidity_usd")
        if verdict.median_entry_liquidity_usd is None:
            verdict.blocked.append("entry_liquidity")
        ranks = [float(row.buyer_rank) for row in resolved
                 if row.buyer_rank is not None]
        verdict.median_buyer_rank = float(statistics.median(ranks)) if ranks else None

        # The selection actually on offer here is WHICH DELAY to believe:
        # given five of them and noise, one wins in sample. CSCV over the
        # per-delay series answers whether that winner holds out of sample.
        # Candidates must be aligned in time, so only decisions with a fill at
        # every delay enter the matrix.
        matrix = self._aligned_delay_matrix(resolved)
        verdict.pbo = (probability_of_backtest_overfitting(matrix)
                       if len(matrix) >= 2 else None)
        if verdict.pbo is None:
            verdict.blocked.append("pbo")

        self._apply_criteria(verdict)
        return verdict

    def _apply_criteria(self, verdict: CopyabilityVerdict) -> None:
        """Turn measurements into a verdict, naming every failure.

        Ordered so the first reason in the list is the one worth reading. A
        wallet that fails on the bound is not interesting for its drawdown.
        """
        criteria = self.criteria
        reasons = verdict.reasons

        if verdict.follower_lower_bound is None:
            verdict.blocked.append("follower_lower_bound")
        elif verdict.follower_lower_bound <= 0:
            reasons.append("follower_lower_bound_not_positive")

        if verdict.lower_bound_without_best is not None \
                and verdict.lower_bound_without_best <= 0 \
                and (verdict.follower_lower_bound or 0) > 0:
            reasons.append("edge_is_one_trade")
        if verdict.single_trade_share is not None \
                and verdict.single_trade_share > criteria.max_single_trade_share:
            reasons.append("profit_concentrated_in_one_trade")

        if verdict.edge_is_speed:
            reasons.append("edge_is_speed_not_information")
        if verdict.months < criteria.min_months:
            reasons.append("record_too_short_in_months")
        if verdict.months_positive is not None and verdict.months_positive < 0.5:
            reasons.append("most_months_unprofitable_to_follow")

        if verdict.profit_factor is not None \
                and verdict.profit_factor < criteria.min_profit_factor:
            reasons.append("profit_factor_below_floor")
        if verdict.max_drawdown is not None \
                and verdict.max_drawdown > criteria.max_drawdown:
            reasons.append("follower_drawdown_beyond_limit")
        if verdict.rug_rate is not None and verdict.rug_rate > criteria.max_rug_rate:
            reasons.append("rug_exposure_beyond_limit")
        if verdict.pbo is not None and verdict.pbo > criteria.max_pbo:
            reasons.append("selection_overfit")
        if criteria.min_entry_liquidity_usd > 0 \
                and verdict.median_entry_liquidity_usd is not None \
                and verdict.median_entry_liquidity_usd < criteria.min_entry_liquidity_usd:
            reasons.append("enters_below_copyable_liquidity")
        if verdict.cost_survival_multiple is None \
                and "follower_lower_bound" not in verdict.blocked:
            reasons.append("does_not_survive_its_own_costs")

        # A blocked measurement is never a pass. The verdict says DATA_BLOCKED
        # and the caller decides what to do about not knowing, which is the
        # only honest place for that decision to live.
        fatal_blocks = {"follower_lower_bound", "insufficient_resolved_decisions",
                        "insufficient_fills_at_desk_latency"}
        if fatal_blocks & set(verdict.blocked):
            verdict.verdict = Verdict.DATA_BLOCKED.value
        elif not reasons:
            verdict.verdict = Verdict.SURVIVOR.value
        elif "follower_lower_bound_not_positive" in reasons:
            verdict.verdict = Verdict.KILL.value
        else:
            verdict.verdict = Verdict.FRAGILE.value

        verdict.detail = (
            f"{verdict.scored} scored follows at {criteria.desk_latency_s:g}s "
            f"across {verdict.months} months")


class CopyabilityBoard:
    """Every studied wallet, ranked by what following it is worth to us."""

    def __init__(self, gauntlet: Optional[CopyabilityGauntlet] = None):
        self.gauntlet = gauntlet or CopyabilityGauntlet()

    def build(self, decisions_by_wallet: Dict[str, Sequence[WalletDecision]],
              own_means: Optional[Dict[str, float]] = None
              ) -> List[CopyabilityVerdict]:
        own_means = own_means or {}
        verdicts = [self.gauntlet.evaluate(wallet, rows,
                                           wallet_own_mean=own_means.get(wallet))
                    for wallet, rows in decisions_by_wallet.items()]
        # Survivors first, then by the bound. A DATA_BLOCKED wallet sorts last
        # and never above a measured one, whatever its headline looks like.
        order = {Verdict.SURVIVOR.value: 0, Verdict.FRAGILE.value: 1,
                 Verdict.KILL.value: 2, Verdict.DATA_BLOCKED.value: 3}
        verdicts.sort(key=lambda item: (order.get(item.verdict, 9),
                                        -(item.follower_lower_bound or -9e9)))
        return verdicts

    def report(self, decisions_by_wallet: Dict[str, Sequence[WalletDecision]],
               own_means: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        verdicts = self.build(decisions_by_wallet, own_means)
        survivors = [item for item in verdicts if item.copyable]
        # The contrast the whole module exists to make visible.
        gaps = [item.wallet_own_mean - item.follower_mean for item in verdicts
                if item.wallet_own_mean is not None and item.follower_mean is not None]
        return {
            "status": "OK" if verdicts else "DATA_BLOCKED",
            "schema": COPYABILITY_SCHEMA_VERSION,
            "selection_warning": SELECTION_WARNING,
            "generated_at": time.time(),
            "wallets": len(verdicts),
            "survivors": len(survivors),
            "median_wallet_minus_follower_log_return": (
                float(statistics.median(gaps)) if gaps else None),
            "by_wallet": [item.to_dict() for item in verdicts],
        }

"""Turning every mistake into a ranked research task.

Improving "accuracy" is the wrong objective for a book whose returns are
dominated by a handful of outcomes. Accuracy weights a thousand ordinary
rejections equally with the one 30x that was passed on, when the second cost
more geometric growth than the first thousand produced. The useful question is
not "how often were we right" but "where did wealth actually leak out", and
then to spend engineering on the largest leak rather than the most annoying
one.

Six leaks, each measured in the currency it actually costs:

  MISSED_MONSTER    rejected, then exploded
  PREMATURE_EXIT    sold at 3x, went to 30x
  RUG_LOSS          entered something that collapsed
  UNDER_SIZED       right call, took a fraction of what was executable
  OVER_SIZED        right call, impact and exit capacity destroyed it
  EXECUTION_MISS    right call, the transaction did not land

Two disciplines make the output trustworthy rather than merely interesting:

Forgone growth is measured in log wealth, not dollars or multiples. A missed
50x on a token that could absorb $200 is a smaller leak than an under-sized 3x
on one that could absorb $50,000, and only the log-wealth view ranks them
correctly. Every leak is capped by what was actually EXECUTABLE -- credit and
blame are both limited to sizes that could have been filled, or the ledger
becomes a fantasy about trades nobody could have made.

Attribution is exhaustive and non-overlapping. Every realised trade lands in
exactly one bucket and the buckets sum to the book's actual PnL. An
attribution that does not reconcile is a story, and a story will happily
allocate a quarter's engineering to a leak that does not exist.
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ATTRIBUTION_SCHEMA_VERSION = "v1"


class Leak(Enum):
    MISSED_MONSTER = "missed_monster"
    PREMATURE_EXIT = "premature_exit"
    RUG_LOSS = "rug_loss"
    UNDER_SIZED = "under_sized"
    OVER_SIZED = "over_sized"
    EXECUTION_MISS = "execution_miss"


@dataclass
class Finding:
    """One measured leak, with enough context to become a research task."""

    leak: Leak
    token: str
    forgone_log_growth: float
    detail: str
    evidence: Dict[str, Any] = field(default_factory=dict)


def _log_growth(fraction: float, multiple: float) -> float:
    """E[log W] of committing ``fraction`` of the book at ``multiple`` return.

    Wealth after the trade is (1 - fraction) + fraction * multiple. Working in
    log wealth is what makes a missed 50x on a $200-capacity token comparable
    to an under-sized 3x on a $50,000 one.
    """
    if fraction <= 0:
        return 0.0
    wealth = (1.0 - fraction) + fraction * max(0.0, multiple)
    if wealth <= 0:
        return -float("inf")
    return math.log(wealth)


def executable_fraction(capacity_usd: Optional[float], equity_usd: float,
                        max_position_pct: float) -> Optional[float]:
    """Share of the book that could genuinely have gone into this token.

    None when capacity was never observed. Every leak below is capped by this,
    because blaming the book for not taking a size the venue could not fill is
    measuring a trade nobody could have made.
    """
    if capacity_usd is None or capacity_usd < 0 or equity_usd <= 0:
        return None
    return float(min(max_position_pct, capacity_usd / equity_usd))


def missed_monster(trade: Dict[str, Any], equity_usd: float,
                   max_position_pct: float) -> Optional[Finding]:
    """A rejected launch that subsequently ran."""
    if trade.get("entered"):
        return None
    feasible = float(trade.get("max_feasible_multiple", 0) or 0)
    if feasible <= 1.0:
        return None
    fraction = executable_fraction(trade.get("capacity_usd"), equity_usd, max_position_pct)
    if fraction is None:
        return None
    return Finding(
        leak=Leak.MISSED_MONSTER, token=str(trade.get("token", "")),
        forgone_log_growth=_log_growth(fraction, feasible),
        detail=f"rejected for {trade.get('rejection_reason', 'unknown')}; "
               f"reached {feasible:.1f}x feasible",
        evidence={"rejection_reason": trade.get("rejection_reason"),
                  "max_feasible_multiple": feasible,
                  "executable_fraction": fraction,
                  # What the bot actually knew at decision time -- the only
                  # thing a fix can be built against.
                  "decision_features": trade.get("decision_features")},
    )


def premature_exit(trade: Dict[str, Any], equity_usd: float,
                   max_position_pct: float) -> Optional[Finding]:
    """Exited into a move that kept going."""
    if not trade.get("entered"):
        return None
    realized = float(trade.get("realized_multiple", 0) or 0)
    feasible = float(trade.get("max_feasible_multiple", 0) or 0)
    if feasible <= realized or realized <= 0:
        return None
    fraction = executable_fraction(trade.get("capacity_usd"), equity_usd, max_position_pct)
    if fraction is None:
        return None
    actual_fraction = min(fraction, float(trade.get("position_fraction", fraction) or fraction))
    forgone = _log_growth(actual_fraction, feasible) - _log_growth(actual_fraction, realized)
    if forgone <= 0:
        return None
    return Finding(
        leak=Leak.PREMATURE_EXIT, token=str(trade.get("token", "")),
        forgone_log_growth=forgone,
        detail=f"exited at {realized:.1f}x on {trade.get('exit_reason', 'unknown')}; "
               f"{feasible:.1f}x was feasible",
        evidence={"exit_reason": trade.get("exit_reason"),
                  "realized_multiple": realized, "max_feasible_multiple": feasible,
                  "tail_capture": realized / feasible},
    )


def rug_loss(trade: Dict[str, Any]) -> Optional[Finding]:
    """Entered something that collapsed, and how early it was knowable."""
    if not trade.get("entered") or not trade.get("rugged"):
        return None
    realized = float(trade.get("realized_multiple", 0) or 0)
    fraction = float(trade.get("position_fraction", 0) or 0)
    if fraction <= 0:
        return None
    forgone = -_log_growth(fraction, realized)
    return Finding(
        leak=Leak.RUG_LOSS, token=str(trade.get("token", "")),
        forgone_log_growth=max(0.0, forgone),
        detail=(f"rugged; earliest observable warning "
                f"{trade.get('earliest_warning_seconds')}s before collapse"),
        evidence={"earliest_warning_seconds": trade.get("earliest_warning_seconds"),
                  "exit_reason": trade.get("exit_reason"),
                  "realized_multiple": realized},
    )


def sizing_leak(trade: Dict[str, Any], equity_usd: float,
                max_position_pct: float) -> Optional[Finding]:
    """Right call, wrong size -- in either direction."""
    if not trade.get("entered"):
        return None
    realized = float(trade.get("realized_multiple", 0) or 0)
    taken = float(trade.get("position_fraction", 0) or 0)
    executable = executable_fraction(trade.get("capacity_usd"), equity_usd, max_position_pct)
    if executable is None or taken <= 0 or realized <= 0:
        return None

    if realized > 1.0 and taken < executable:
        forgone = _log_growth(executable, realized) - _log_growth(taken, realized)
        if forgone <= 0:
            return None
        return Finding(
            leak=Leak.UNDER_SIZED, token=str(trade.get("token", "")),
            forgone_log_growth=forgone,
            detail=f"took {taken:.3%} of the book where {executable:.3%} was executable",
            evidence={"taken_fraction": taken, "executable_fraction": executable,
                      "realized_multiple": realized},
        )
    if taken > executable:
        # The position exceeded what the venue could absorb, so part of the
        # realised multiple was never really available on that size.
        forgone = _log_growth(executable, realized) - _log_growth(taken, realized)
        return Finding(
            leak=Leak.OVER_SIZED, token=str(trade.get("token", "")),
            forgone_log_growth=max(0.0, forgone),
            detail=f"took {taken:.3%} where only {executable:.3%} was executable",
            evidence={"taken_fraction": taken, "executable_fraction": executable,
                      "realized_multiple": realized},
        )
    return None


def execution_miss(trade: Dict[str, Any], equity_usd: float,
                   max_position_pct: float) -> Optional[Finding]:
    """The decision was right and the transaction did not land.

    Distinct from a prediction failure, and the distinction matters: one is
    fixed with a model and the other with tips, regions and code.
    """
    if trade.get("entered") or not trade.get("attempted"):
        return None
    feasible = float(trade.get("max_feasible_multiple", 0) or 0)
    if feasible <= 1.0:
        return None
    fraction = executable_fraction(trade.get("capacity_usd"), equity_usd, max_position_pct)
    if fraction is None:
        return None
    return Finding(
        leak=Leak.EXECUTION_MISS, token=str(trade.get("token", "")),
        forgone_log_growth=_log_growth(fraction, feasible),
        detail=f"decision taken but not filled: {trade.get('failure_reason', 'unknown')}",
        evidence={"failure_reason": trade.get("failure_reason"),
                  "max_feasible_multiple": feasible},
    )


def find_leaks(trades: Sequence[Dict[str, Any]], equity_usd: float,
               max_position_pct: float = 0.05) -> List[Finding]:
    """Every measurable leak across a set of trades, largest first."""
    findings: List[Finding] = []
    for trade in trades:
        for finding in (
            missed_monster(trade, equity_usd, max_position_pct),
            execution_miss(trade, equity_usd, max_position_pct),
            premature_exit(trade, equity_usd, max_position_pct),
            rug_loss(trade),
            sizing_leak(trade, equity_usd, max_position_pct),
        ):
            if finding is not None and math.isfinite(finding.forgone_log_growth):
                findings.append(finding)
    findings.sort(key=lambda item: item.forgone_log_growth, reverse=True)
    return findings


def rank_research(findings: Sequence[Finding], top_n: int = 5) -> Dict[str, Any]:
    """Where the next engineering hour is worth most.

    Grouped by leak and then by the reason inside it, because "we lose most to
    missed monsters" is not actionable and "we lose most to missed monsters
    rejected for insufficient_upside" is.
    """
    by_leak: Dict[str, float] = defaultdict(float)
    by_reason: Dict[Tuple[str, str], float] = defaultdict(float)
    for finding in findings:
        by_leak[finding.leak.value] += finding.forgone_log_growth
        reason = str(finding.evidence.get("rejection_reason")
                     or finding.evidence.get("exit_reason")
                     or finding.evidence.get("failure_reason") or "unattributed")
        by_reason[(finding.leak.value, reason)] += finding.forgone_log_growth
    total = sum(by_leak.values())
    ranked_reasons = sorted(by_reason.items(), key=lambda item: item[1], reverse=True)
    return {
        "total_forgone_log_growth": total,
        "by_leak": dict(sorted(by_leak.items(), key=lambda item: item[1], reverse=True)),
        "share_by_leak": ({name: value / total for name, value in by_leak.items()}
                          if total > 0 else {}),
        "top_causes": [{"leak": leak, "reason": reason, "forgone_log_growth": value}
                       for (leak, reason), value in ranked_reasons[:top_n]],
        "worst_tokens": [{"token": item.token, "leak": item.leak.value,
                          "forgone_log_growth": item.forgone_log_growth,
                          "detail": item.detail}
                         for item in findings[:top_n]],
    }


@dataclass
class LedgerEntry:
    mechanism: str
    realized_pnl_usd: float
    trades: int


def alpha_ledger(trades: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Realised PnL attributed to the mechanism that produced each trade.

    Exhaustive and non-overlapping: every realised trade lands in exactly one
    bucket and the buckets sum to the book's PnL. An attribution that does not
    reconcile is a story, and a story will happily send a quarter of
    engineering at a leak that does not exist. Trades whose mechanism was never
    recorded go to an explicit "unattributed" bucket rather than being dropped,
    because silently dropping them is how the sum stops reconciling.
    """
    buckets: Dict[str, List[float]] = defaultdict(list)
    for trade in trades:
        if not trade.get("entered") or trade.get("realized_pnl_usd") is None:
            continue
        mechanism = str(trade.get("mechanism") or "unattributed")
        buckets[mechanism].append(float(trade["realized_pnl_usd"]))

    entries = [LedgerEntry(mechanism=name, realized_pnl_usd=float(sum(values)),
                           trades=len(values))
               for name, values in buckets.items()]
    entries.sort(key=lambda item: item.realized_pnl_usd, reverse=True)
    total = sum(entry.realized_pnl_usd for entry in entries)
    book = sum(float(trade["realized_pnl_usd"]) for trade in trades
               if trade.get("entered") and trade.get("realized_pnl_usd") is not None)
    return {
        "entries": [{"mechanism": entry.mechanism, "realized_pnl_usd": entry.realized_pnl_usd,
                     "trades": entry.trades} for entry in entries],
        "total_usd": total,
        "reconciles": abs(total - book) < 1e-6,
        "concentration": (max((entry.realized_pnl_usd for entry in entries), default=0.0) / total
                          if total > 0 else None),
    }


def tail_contribution(trades: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Share of total growth coming from the best 10%, 1% and 0.1% of trades.

    A book whose growth is 90% one mechanism is less diversified than its trade
    count suggests, and an exit policy that lifts win rate while removing the
    top decile destroys wealth while every conventional metric improves.
    """
    growths = sorted(
        (math.log(max(1e-9, float(trade.get("wealth_multiple", 0) or 0)))
         for trade in trades
         if trade.get("entered") and float(trade.get("wealth_multiple", 0) or 0) > 0),
        reverse=True,
    )
    if not growths:
        return {"status": "DATA_BLOCKED", "detail": "no trades with a wealth multiple"}
    positive_total = sum(value for value in growths if value > 0)
    if positive_total <= 0:
        return {"status": "DATA_BLOCKED", "detail": "no positive-growth trades"}

    def share(top_fraction: float) -> Optional[float]:
        count = int(len(growths) * top_fraction)
        if count < 1:
            # Reporting a share computed from zero trades as 0.0 would read as
            # "the top 0.1% contributed nothing" about a population too small
            # to have a top 0.1%.
            return None
        return float(sum(value for value in growths[:count] if value > 0) / positive_total)

    return {
        "status": "OK", "sample": len(growths),
        "total_log_growth": float(sum(growths)),
        "top_10pct_share": share(0.10),
        "top_1pct_share": share(0.01),
        "top_01pct_share": share(0.001),
    }


class EdgeDecayMonitor:
    """Live health per mechanism, with a sample-size floor before any verdict.

    A mechanism that has produced eight trades has not "degraded" -- it has not
    been measured. Declaring an edge dead on a small sample retires exactly the
    regime-specific edges that go quiet and then come back, which is the most
    expensive kind of mistake this monitor can make. So a verdict below the
    floor is MEASURING, never DEGRADED, and DEGRADED is a demotion rather than
    a deletion.
    """

    HEALTHY = "HEALTHY"
    WEAKENING = "WEAKENING"
    DEGRADED = "DEGRADED"
    MEASURING = "MEASURING"

    def __init__(self, min_trades: int = 30, weakening_ratio: float = 0.5,
                 degraded_ratio: float = 0.0):
        self.min_trades = max(2, min_trades)
        self.weakening_ratio = weakening_ratio
        self.degraded_ratio = degraded_ratio
        self._history: Dict[str, List[float]] = defaultdict(list)
        self._baseline: Dict[str, float] = {}

    def record(self, mechanism: str, log_growth: float) -> None:
        self._history[mechanism].append(float(log_growth))

    def set_baseline(self, mechanism: str, mean_log_growth: float) -> None:
        """The forward performance this mechanism was promoted on."""
        self._baseline[mechanism] = float(mean_log_growth)

    def health(self, mechanism: str) -> Dict[str, Any]:
        samples = self._history.get(mechanism, [])
        if len(samples) < self.min_trades:
            return {"mechanism": mechanism, "status": self.MEASURING,
                    "sample": len(samples),
                    "detail": f"{len(samples)} of {self.min_trades} trades needed for a verdict"}
        recent = float(np.mean(samples[-self.min_trades:]))
        baseline = self._baseline.get(mechanism)
        if baseline is None or baseline <= 0:
            status = self.HEALTHY if recent > 0 else self.DEGRADED
            return {"mechanism": mechanism, "status": status, "sample": len(samples),
                    "recent_mean_log_growth": recent,
                    "detail": "no promotion baseline recorded; judged on sign alone"}
        ratio = recent / baseline
        if ratio <= self.degraded_ratio:
            status = self.DEGRADED
        elif ratio < self.weakening_ratio:
            status = self.WEAKENING
        else:
            status = self.HEALTHY
        return {"mechanism": mechanism, "status": status, "sample": len(samples),
                "recent_mean_log_growth": recent, "baseline": baseline, "ratio": ratio,
                "detail": f"recent forward growth is {ratio:.2f}x the promotion baseline"}

    def report(self) -> Dict[str, Any]:
        return {mechanism: self.health(mechanism) for mechanism in sorted(self._history)}

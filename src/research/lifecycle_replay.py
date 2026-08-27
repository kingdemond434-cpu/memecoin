"""Replaying the whole launch, not one row per token with a final return.

A memecoin launch reduced to (features, outcome) throws away the only thing
worth knowing about it. The question is never "did this token go up"; it is
"at each instant, what could we actually have bought, how much, at what price,
and what could we then actually have sold". Those are different questions and
only the second one has money in it.

So the unit of evaluation here is a launch lifecycle, replayed on a grid:

    delay x size x exit rule

Every cell answers with an EXECUTABLE result -- entry filled at the price the
curve would have given for that size at that instant, exit credited only for
what the curve could have absorbed. A replay that credits the peak price on
unlimited size is a fantasy that ranks the thinnest tokens highest, which is
exactly backwards.

Two disciplines make the output usable rather than merely impressive:

Point-in-time, enforced structurally. A decision at T+250ms sees observations
timestamped at or before T+250ms and nothing else. This is checked rather than
intended: `test_replay_is_point_in_time` appends a future observation and
asserts the result does not move.

Missing means missing. A cell with no observation to price against is
DATA_BLOCKED and is excluded from every aggregate, rather than scoring zero.
Scoring it zero would make a launch nobody could have traded look like a
launch that was traded and broke even, and there is no way to tell those apart
downstream.

The scoreboard is deliberately sniper-native. Sharpe over a book whose returns
come from a handful of tokens describes a distribution nobody experienced;
"net SOL per 100 launches observed" and "share of feasible 10x captured" are
the numbers that decide whether this works.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

LIFECYCLE_REPLAY_SCHEMA_VERSION = "v1"

# The grid the decision actually lives on. Sub-second entries are here because
# on a newborn launch that is where the edge is won or lost, and a grid that
# starts at 1s cannot see it.
DEFAULT_DELAYS_S: Tuple[float, ...] = (
    0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 3.0, 5.0, 10.0, 30.0,
)

DEFAULT_SIZES_SOL: Tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)


@dataclass
class Mark:
    """One observed, executable price point in a launch's life."""

    timestamp: float
    multiple: float
    # Largest size that could actually have been traded here, in SOL. None
    # means depth was never observed at this instant -- not that it was zero.
    executable_sol: Optional[float] = None
    feasible: bool = True


@dataclass
class Lifecycle:
    """The movie for one launch: every mark, in order, with its outcome."""

    token: str
    created_at: float
    marks: List[Mark] = field(default_factory=list)
    migrated: bool = False
    rugged: bool = False
    rug_time: Optional[float] = None

    def marks_upto(self, offset_s: float) -> List[Mark]:
        """Marks at or before ``created_at + offset_s``. The PIT boundary."""
        cutoff = self.created_at + offset_s
        return [mark for mark in self.marks if mark.timestamp <= cutoff]

    def marks_after(self, offset_s: float) -> List[Mark]:
        cutoff = self.created_at + offset_s
        return [mark for mark in self.marks if mark.timestamp > cutoff]

    def entry_mark(self, delay_s: float) -> Optional[Mark]:
        """The mark a buyer arriving at ``delay_s`` would actually have filled at.

        The LAST mark at or before the delay, not the nearest: a buyer at
        T+250ms cannot fill at a price that printed at T+400ms, however much
        closer it is.
        """
        eligible = [mark for mark in self.marks_upto(delay_s) if mark.feasible]
        return eligible[-1] if eligible else None

    def peak_feasible_multiple(self, after_s: float) -> Optional[float]:
        """Highest multiple that was both printed and executable after entry."""
        eligible = [mark.multiple for mark in self.marks_after(after_s)
                    if mark.feasible and (mark.executable_sol or 0) > 0]
        return max(eligible) if eligible else None


@dataclass
class Cell:
    """One (delay, size, exit rule) result for one launch."""

    token: str
    delay_s: float
    size_sol: float
    exit_rule: str
    status: str
    entry_multiple: Optional[float] = None
    exit_multiple: Optional[float] = None
    filled_sol: float = 0.0
    net_sol: float = 0.0
    max_feasible_multiple: Optional[float] = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "OK"

    @property
    def tail_capture(self) -> Optional[float]:
        """Realised over maximum FEASIBLE, on this cell."""
        if not self.ok or not self.max_feasible_multiple or self.exit_multiple is None:
            return None
        if self.max_feasible_multiple <= 0:
            return None
        return float(self.exit_multiple / self.max_feasible_multiple)


# An exit rule takes the marks available after entry and returns the mark it
# exits on, or None to hold to the end. Pure and deterministic, so the same
# rule replays offline exactly as it would run live.
ExitRule = Callable[[Sequence[Mark], float], Optional[Mark]]


def hold_to_end(marks: Sequence[Mark], entry_multiple: float) -> Optional[Mark]:
    """The baseline every other rule has to beat."""
    feasible = [mark for mark in marks if mark.feasible]
    return feasible[-1] if feasible else None


def fixed_take_profit(target: float) -> ExitRule:
    def rule(marks: Sequence[Mark], entry_multiple: float) -> Optional[Mark]:
        for mark in marks:
            if mark.feasible and mark.multiple / max(entry_multiple, 1e-9) >= target:
                return mark
        return hold_to_end(marks, entry_multiple)
    return rule


def trailing_stop(ratio: float) -> ExitRule:
    def rule(marks: Sequence[Mark], entry_multiple: float) -> Optional[Mark]:
        high = entry_multiple
        for mark in marks:
            if not mark.feasible:
                continue
            high = max(high, mark.multiple)
            if mark.multiple <= high * ratio:
                return mark
        return hold_to_end(marks, entry_multiple)
    return rule


DEFAULT_EXIT_RULES: Dict[str, ExitRule] = {
    "hold": hold_to_end,
    "tp_2x": fixed_take_profit(2.0),
    "tp_5x": fixed_take_profit(5.0),
    "trail_70": trailing_stop(0.70),
    "trail_50": trailing_stop(0.50),
}


def replay_cell(
    lifecycle: Lifecycle,
    delay_s: float,
    size_sol: float,
    exit_rule_name: str,
    exit_rule: ExitRule,
    round_trip_cost: float = 0.02,
) -> Cell:
    """One cell of the grid, priced against what was actually executable."""
    entry = lifecycle.entry_mark(delay_s)
    if entry is None:
        return Cell(lifecycle.token, delay_s, size_sol, exit_rule_name,
                    status="DATA_BLOCKED",
                    detail=f"no feasible mark at or before T+{delay_s:g}s")
    if entry.executable_sol is None:
        # Unobserved depth is not unlimited depth.
        return Cell(lifecycle.token, delay_s, size_sol, exit_rule_name,
                    status="DATA_BLOCKED", entry_multiple=entry.multiple,
                    detail="entry depth was never observed")

    filled = min(size_sol, entry.executable_sol)
    if filled <= 0:
        return Cell(lifecycle.token, delay_s, size_sol, exit_rule_name,
                    status="OK", entry_multiple=entry.multiple, filled_sol=0.0,
                    net_sol=0.0, detail="nothing fillable at this size")

    forward = lifecycle.marks_after(delay_s)
    exit_mark = exit_rule(forward, entry.multiple)
    if exit_mark is None:
        return Cell(lifecycle.token, delay_s, size_sol, exit_rule_name,
                    status="DATA_BLOCKED", entry_multiple=entry.multiple,
                    filled_sol=filled, detail="no feasible exit mark after entry")
    if exit_mark.executable_sol is None:
        return Cell(lifecycle.token, delay_s, size_sol, exit_rule_name,
                    status="DATA_BLOCKED", entry_multiple=entry.multiple,
                    filled_sol=filled, detail="exit depth was never observed")

    # Credit only what the curve could absorb on the way out. A replay that
    # credits the printed price on unlimited size ranks the thinnest tokens
    # highest, which is exactly backwards.
    sellable = min(filled, exit_mark.executable_sol)
    gross_multiple = exit_mark.multiple / max(entry.multiple, 1e-9)
    proceeds = sellable * gross_multiple + (filled - sellable) * 0.02
    net = proceeds * (1.0 - round_trip_cost) - filled

    return Cell(
        lifecycle.token, delay_s, size_sol, exit_rule_name, status="OK",
        entry_multiple=entry.multiple, exit_multiple=gross_multiple,
        filled_sol=filled, net_sol=net,
        max_feasible_multiple=(
            (lifecycle.peak_feasible_multiple(delay_s) or entry.multiple)
            / max(entry.multiple, 1e-9)),
        detail=f"filled {filled:.4f} SOL, sold {sellable:.4f}",
    )


def replay_lifecycle(
    lifecycle: Lifecycle,
    delays: Sequence[float] = DEFAULT_DELAYS_S,
    sizes: Sequence[float] = DEFAULT_SIZES_SOL,
    exit_rules: Optional[Dict[str, ExitRule]] = None,
    round_trip_cost: float = 0.02,
) -> List[Cell]:
    """Every cell of the grid for one launch."""
    rules = exit_rules or DEFAULT_EXIT_RULES
    return [
        replay_cell(lifecycle, delay, size, name, rule, round_trip_cost)
        for delay in delays
        for size in sizes
        for name, rule in rules.items()
    ]


def _mean(values: Sequence[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def sniper_scoreboard(cells: Sequence[Cell], launches_observed: int) -> Dict[str, Any]:
    """The numbers that decide whether this works, in sniper terms.

    Deliberately not Sharpe. A book whose returns come from a handful of
    tokens has a Sharpe that describes a distribution nobody experienced.
    Every aggregate here excludes DATA_BLOCKED cells rather than scoring them
    zero, and reports how many were excluded, because a launch nobody could
    have traded and a launch that broke even are different facts.
    """
    priced = [cell for cell in cells if cell.ok]
    blocked = len(cells) - len(priced)
    if not priced:
        return {"status": "DATA_BLOCKED", "cells": len(cells), "blocked": blocked,
                "detail": "no cell could be priced against observed depth"}

    by_delay: Dict[float, List[float]] = {}
    by_size: Dict[float, List[float]] = {}
    by_rule: Dict[str, List[float]] = {}
    for cell in priced:
        by_delay.setdefault(cell.delay_s, []).append(cell.net_sol)
        by_size.setdefault(cell.size_sol, []).append(cell.net_sol)
        by_rule.setdefault(cell.exit_rule, []).append(cell.net_sol)

    captures = [cell.tail_capture for cell in priced if cell.tail_capture is not None]
    tenx_available = [cell for cell in priced
                      if (cell.max_feasible_multiple or 0) >= 10.0]
    tenx_captured = [cell for cell in tenx_available if (cell.exit_multiple or 0) >= 5.0]

    return {
        "status": "OK",
        "cells": len(cells),
        "priced": len(priced),
        "blocked": blocked,
        "launches_observed": launches_observed,
        # Opportunity extraction, which is what a sniper is actually measured on.
        "net_sol_per_100_launches": (
            sum(cell.net_sol for cell in priced) / max(launches_observed, 1) * 100),
        "net_sol_per_priced_cell": _mean([cell.net_sol for cell in priced]),
        "tail_capture_mean": _mean(captures),
        "share_of_10x_captured_above_5x": (
            len(tenx_captured) / len(tenx_available) if tenx_available else None),
        # Speed value: how much edge each extra millisecond of delay costs.
        "net_sol_by_delay": {delay: _mean(values) for delay, values in sorted(by_delay.items())},
        # Capacity: where size stops paying.
        "net_sol_by_size": {size: _mean(values) for size, values in sorted(by_size.items())},
        "net_sol_by_exit_rule": {name: _mean(values) for name, values in sorted(by_rule.items())},
    }


def delay_decay(scoreboard: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """How much of the edge is gone by each delay, relative to T0.

    The number that decides whether the next engineering hour belongs in
    latency or in prediction. If the T+1s row still holds most of the edge,
    shaving microseconds is not where the money is.
    """
    by_delay = scoreboard.get("net_sol_by_delay") or {}
    if not by_delay:
        return None
    zero = by_delay.get(0.0)
    if zero is None or zero <= 0:
        # With no positive edge at T0 there is nothing for a delay to decay
        # from, and a ratio against a non-positive base is meaningless.
        return None
    return {delay: (value / zero if value is not None else None)
            for delay, value in sorted(by_delay.items())}


def lifecycle_from_episode(episode: Dict[str, Any]) -> Optional[Lifecycle]:
    """Build a Lifecycle from a persisted point-in-time episode.

    Only observations carrying an explicit multiple are used, and depth is
    taken only where it was recorded. An observation that priced nothing
    contributes nothing rather than contributing a zero.
    """
    created_at = float(episode.get("created_at", 0) or 0)
    if created_at <= 0:
        return None
    marks: List[Mark] = []
    for item in episode.get("market_observations") or []:
        timestamp = item.get("timestamp")
        multiple = item.get("price_multiple")
        if timestamp is None or multiple is None:
            continue
        try:
            multiple = float(multiple)
        except (TypeError, ValueError):
            continue
        if multiple <= 0:
            continue
        depth = item.get("executable_sol")
        marks.append(Mark(
            timestamp=float(timestamp), multiple=multiple,
            executable_sol=(float(depth) if depth is not None else None),
            feasible=item.get("feasible", True) is not False,
        ))
    if not marks:
        return None
    marks.sort(key=lambda mark: mark.timestamp)
    outcome = episode.get("final_outcome") or {}
    return Lifecycle(
        token=str(episode.get("token", "")), created_at=created_at, marks=marks,
        migrated=bool(outcome.get("migrated")), rugged=bool(outcome.get("rugged")),
        rug_time=(float(outcome["rug_time"]) if outcome.get("rug_time") is not None else None),
    )

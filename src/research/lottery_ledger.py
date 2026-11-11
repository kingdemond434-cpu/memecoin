"""Counterfactual lottery tickets on launches the core desk declined.

The ASYMMETRIC_TAIL sleeve can only take a sub-position of a trade the core
gate already accepted, so the lottery never sees the launches the core turns
down for a low 2x probability, which is exactly where a cheap ticket on a fat
tail lives. This ledger answers the question that has to come first: had the
desk bought a small fixed ticket on every declined, safety-clean launch whose
model still priced a real tail, what would the sleeve have made?

Each admitted launch gets one virtual ticket at the curve price at decision
time. Half is sold the first time the price reaches `harvest_at` (booked at
exactly that multiple, never at the peak); the runner is marked to the last
observed price, closed at zero on a detected rug, and closed at its last mark
when the horizon ends. A round-trip cost is charged on every ticket.

What this is not: a fill. Prices are bonding-curve marks from the stream, a
ticket this size would move a thin curve, and post-migration prices are not
followed (the runner keeps its last curve mark). So results are an upper
bound on what the sleeve could have earned, labelled as such, and the
sleeve's real budget is still owned by `asymmetric_tail`. No authority.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

SCHEMA = "lottery_counterfactual/1"


@dataclass(frozen=True)
class LotteryPolicy:
    ticket_fraction: float = 0.001          # of equity, same as the tail sleeve's per-token cap
    min_tail_value: float = 1.0             # admit when p_10x * 10 >= this (gross break-even)
    harvest_at: float = 3.0
    harvest_fraction: float = 0.5
    round_trip_cost: float = 0.04           # fees + slippage on a thin curve, both legs
    horizon_seconds: float = 7 * 86400.0
    max_tickets_per_day: int = 200
    max_open: int = 5000


@dataclass
class Ticket:
    token: str
    opened_at: float
    entry_multiple: float                   # curve multiple at decision time
    cost_usd: float
    p_10x: float
    reason: str
    world_event: str = ""
    peak: float = 1.0                       # ticket multiple, ratchets up
    last: float = 1.0
    harvested: bool = False
    proceeds_usd: float = 0.0
    closed_at: Optional[float] = None
    close_reason: str = ""


#: Rejections that mean the launch must not be bought at any size.
_NEVER = ("DATA_BLOCKED", "safety_veto", "daily_loss_kill_switch")


class LotteryLedger:
    def __init__(self, policy: Optional[LotteryPolicy] = None):
        self.policy = policy or LotteryPolicy()
        self.open: Dict[str, Ticket] = {}
        self.closed: List[Ticket] = []
        self.skipped: Counter = Counter()
        self._day_counts: Counter = Counter()

    @staticmethod
    def _day(at: float) -> str:
        return datetime.fromtimestamp(at, timezone.utc).date().isoformat()

    def consider(self, token: str, *, at: float, entry_multiple: Optional[float],
                 p_10x: float, reason: str, safety_clear: bool, equity_usd: float,
                 world_event: str = "") -> bool:
        """File a ticket on a declined launch, or count why not."""
        policy = self.policy
        why = None
        if token in self.open or any(t.token == token for t in self.closed[-1000:]):
            why = "already_ticketed"
        elif any(str(reason).startswith(prefix) for prefix in _NEVER):
            why = "hard_reject"
        elif not safety_clear:
            why = "safety_not_clear"
        elif not entry_multiple or not math.isfinite(entry_multiple) or entry_multiple <= 0:
            why = "no_entry_mark"
        elif not math.isfinite(equity_usd) or equity_usd <= 0:
            why = "no_equity"
        elif p_10x * 10 < policy.min_tail_value:
            why = "tail_too_thin"
        elif len(self.open) >= policy.max_open:
            why = "max_open"
        elif self._day_counts[self._day(at)] >= policy.max_tickets_per_day:
            why = "daily_ticket_cap"
        if why:
            self.skipped[why] += 1
            return False
        self._day_counts[self._day(at)] += 1
        self.open[token] = Ticket(token, at, float(entry_multiple),
                                  equity_usd * policy.ticket_fraction, float(p_10x),
                                  str(reason)[:80], world_event=str(world_event)[:80])
        return True

    def mark(self, token: str, curve_multiple: float, at: float) -> None:
        ticket = self.open.get(token)
        if ticket is None or not math.isfinite(curve_multiple) or curve_multiple <= 0:
            return
        if at - ticket.opened_at > self.policy.horizon_seconds:
            self._close(ticket, at, "horizon")
            return
        multiple = curve_multiple / ticket.entry_multiple
        ticket.last = multiple
        ticket.peak = max(ticket.peak, multiple)
        if not ticket.harvested and multiple >= self.policy.harvest_at:
            ticket.harvested = True
            ticket.proceeds_usd += (ticket.cost_usd * self.policy.harvest_fraction
                                    * self.policy.harvest_at)

    def rugged(self, token: str, at: float) -> None:
        ticket = self.open.get(token)
        if ticket is not None:
            ticket.last = 0.0
            self._close(ticket, at, "rug")

    def expire(self, now: float) -> int:
        due = [t for t in self.open.values() if now - t.opened_at > self.policy.horizon_seconds]
        for ticket in due:
            self._close(ticket, now, "horizon")
        return len(due)

    def _runner_fraction(self, ticket: Ticket) -> float:
        return 1 - self.policy.harvest_fraction if ticket.harvested else 1.0

    def _close(self, ticket: Ticket, at: float, why: str) -> None:
        ticket.proceeds_usd += ticket.cost_usd * self._runner_fraction(ticket) * ticket.last
        ticket.closed_at, ticket.close_reason = at, why
        self.open.pop(ticket.token, None)
        self.closed.append(ticket)
        if len(self.closed) > 50_000:
            del self.closed[:10_000]

    def report(self) -> Dict[str, Any]:
        policy = self.policy
        closed = self.closed
        cost = sum(t.cost_usd for t in closed)
        charges = cost * policy.round_trip_cost
        proceeds = sum(t.proceeds_usd for t in closed)
        open_mark = sum(t.proceeds_usd + t.cost_usd * self._runner_fraction(t) * t.last
                        for t in self.open.values())
        log_growth = [math.log(max(t.proceeds_usd / t.cost_usd - policy.round_trip_cost, 1e-9))
                      for t in closed if t.cost_usd > 0]
        return {
            "schema": SCHEMA,
            "scope": ("counterfactual upper bound: curve marks, harvest booked at the threshold, "
                      "runner at last curve mark, no price impact; no authority"),
            "policy": asdict(policy),
            "open_tickets": len(self.open), "closed_tickets": len(closed),
            "closed_cost_usd": cost, "closed_proceeds_usd": proceeds,
            "closed_net_usd": proceeds - cost - charges,
            "closed_money_multiple": (proceeds - charges) / cost if cost else None,
            "mean_log_return_per_ticket": (sum(log_growth) / len(log_growth)) if log_growth else None,
            "open_marked_value_usd": open_mark,
            "peak_at_least": {f"{m}x": sum(t.peak >= m for t in closed + list(self.open.values()))
                              for m in (3, 10, 100)},
            "closed_by_reason": dict(Counter(t.close_reason for t in closed)),
            "event_matched_tickets": sum(bool(t.world_event) for t in closed + list(self.open.values())),
            "skipped": dict(self.skipped),
        }

    def state(self) -> Dict[str, Any]:
        return {"schema": SCHEMA, "open": [asdict(t) for t in self.open.values()],
                "closed": [asdict(t) for t in self.closed[-5000:]],
                "skipped": dict(self.skipped), "days": dict(self._day_counts)}

    def load(self, state: Dict[str, Any]) -> None:
        if not state or state.get("schema") != SCHEMA:
            return
        self.open = {row["token"]: Ticket(**row) for row in state.get("open", [])}
        self.closed = [Ticket(**row) for row in state.get("closed", [])]
        self.skipped = Counter(state.get("skipped", {}))
        self._day_counts = Counter(state.get("days", {}))

"""Counterfactual lottery tickets on launches the core desk declined.

The ASYMMETRIC_TAIL sleeve can only take a sub-position of a trade the core
gate already accepted, so the lottery never sees the launches the core turns
down for a low 2x probability, which is exactly where a cheap ticket on a fat
tail lives. This ledger answers the question that has to come first: had the
desk bought a small fixed ticket on every declined, safety-clean launch whose
model still priced a real tail, what would the sleeve have made?

Each admitted launch gets one virtual ticket at the ABSOLUTE curve price at
decision time (never a multiple of some in-memory anchor, which the desk
resets when an episode leaves memory or the process restarts). Half is sold the first time the price reaches `harvest_at` (booked at
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

SCHEMA = "lottery_counterfactual/2"


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
    entry_price: float                      # absolute curve price at decision time
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
    last_mark_at: Optional[float] = None
    migrated: bool = False


#: Rejections that mean the launch must not be bought at any size.
_NEVER = ("DATA_BLOCKED", "safety_veto", "daily_loss_kill_switch")


class LotteryLedger:
    def __init__(self, policy: Optional[LotteryPolicy] = None):
        self.policy = policy or LotteryPolicy()
        self.open: Dict[str, Ticket] = {}
        self.closed: List[Ticket] = []          # recent, for inspection only
        self.skipped: Counter = Counter()
        self._day_counts: Counter = Counter()
        # Running totals over EVERY closed ticket, persisted, so the report
        # does not shrink when the inspection list is trimmed or restored.
        self.totals: Dict[str, Any] = {
            "closed": 0, "cost": 0.0, "proceeds": 0.0, "log_ticket": 0.0,
            "log_equity": 0.0, "peak3": 0, "peak10": 0, "peak100": 0,
            "reasons": {}, "event_matched": 0, "unmarked_after_migration": 0}

    @staticmethod
    def _day(at: float) -> str:
        return datetime.fromtimestamp(at, timezone.utc).date().isoformat()

    def consider(self, token: str, *, at: float, entry_price: Optional[float],
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
        elif not entry_price or not math.isfinite(entry_price) or entry_price <= 0:
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
        self.open[token] = Ticket(token, at, float(entry_price),
                                  equity_usd * policy.ticket_fraction, float(p_10x),
                                  str(reason)[:80], world_event=str(world_event)[:80])
        return True

    def mark(self, token: str, curve_price: float, at: float) -> bool:
        """Mark a ticket to an absolute curve price. True when it just harvested."""
        ticket = self.open.get(token)
        if ticket is None or not math.isfinite(curve_price) or curve_price <= 0:
            return False
        if at - ticket.opened_at > self.policy.horizon_seconds:
            self._close(ticket, at, "horizon")
            return False
        multiple = curve_price / ticket.entry_price
        ticket.last, ticket.last_mark_at = multiple, at
        ticket.peak = max(ticket.peak, multiple)
        if not ticket.harvested and multiple >= self.policy.harvest_at:
            ticket.harvested = True
            ticket.proceeds_usd += (ticket.cost_usd * self.policy.harvest_fraction
                                    * self.policy.harvest_at)
            return True
        return False

    def migrated(self, token: str) -> None:
        """Curve marks stop here; the runner keeps its last curve mark."""
        ticket = self.open.get(token)
        if ticket is not None:
            ticket.migrated = True

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
        policy = self.policy
        ticket.proceeds_usd += ticket.cost_usd * self._runner_fraction(ticket) * ticket.last
        ticket.closed_at, ticket.close_reason = at, why
        self.open.pop(ticket.token, None)
        self.closed.append(ticket)
        if len(self.closed) > 5_000:
            del self.closed[:1_000]
        totals = self.totals
        money = ticket.proceeds_usd / ticket.cost_usd - policy.round_trip_cost
        totals["closed"] += 1
        totals["cost"] += ticket.cost_usd
        totals["proceeds"] += ticket.proceeds_usd
        # Per ticket: log of the ticket's own money multiple. A total loss
        # is unbounded below, so it is floored and labelled, not an E[log W].
        totals["log_ticket"] += math.log(max(money, 1e-9))
        # Per equity: the growth this ticket contributed at its real size,
        # log(1 + f (m - 1)), which IS the E[log W] quantity.
        totals["log_equity"] += math.log(1 + policy.ticket_fraction * (money - 1))
        for multiple, key in ((3, "peak3"), (10, "peak10"), (100, "peak100")):
            totals[key] += ticket.peak >= multiple
        totals["reasons"][why] = totals["reasons"].get(why, 0) + 1
        totals["event_matched"] += bool(ticket.world_event)
        totals["unmarked_after_migration"] += why == "horizon" and ticket.migrated

    def report(self) -> Dict[str, Any]:
        policy, totals = self.policy, self.totals
        cost, proceeds, count = totals["cost"], totals["proceeds"], totals["closed"]
        charges = cost * policy.round_trip_cost
        open_tickets = list(self.open.values())
        return {
            "schema": SCHEMA,
            "scope": ("counterfactual upper bound: curve marks, harvest booked at the threshold, "
                      "runner at last curve mark (held after migration), no price impact; "
                      "no authority"),
            "policy": asdict(policy),
            "open_tickets": len(open_tickets), "closed_tickets": count,
            "closed_cost_usd": cost, "closed_proceeds_usd": proceeds,
            "closed_net_usd": proceeds - cost - charges,
            "closed_money_multiple": (proceeds - charges) / cost if cost else None,
            "mean_log_ticket_multiple_floored": totals["log_ticket"] / count if count else None,
            "mean_log_equity_growth_per_ticket": totals["log_equity"] / count if count else None,
            "open_marked_value_usd": sum(t.proceeds_usd + t.cost_usd * self._runner_fraction(t) * t.last
                                         for t in open_tickets),
            "peak_at_least": {
                "3x": totals["peak3"] + sum(t.peak >= 3 for t in open_tickets),
                "10x": totals["peak10"] + sum(t.peak >= 10 for t in open_tickets),
                "100x": totals["peak100"] + sum(t.peak >= 100 for t in open_tickets)},
            "closed_by_reason": dict(totals["reasons"]),
            "unmarked_after_migration": totals["unmarked_after_migration"]
                                        + sum(t.migrated for t in open_tickets),
            "event_matched_tickets": totals["event_matched"] + sum(bool(t.world_event) for t in open_tickets),
            "skipped": dict(self.skipped),
        }

    def state(self) -> Dict[str, Any]:
        return {"schema": SCHEMA, "open": [asdict(t) for t in self.open.values()],
                "closed": [asdict(t) for t in self.closed[-1000:]],
                "totals": self.totals,
                "skipped": dict(self.skipped), "days": dict(self._day_counts)}

    def load(self, state: Dict[str, Any]) -> None:
        if not state or state.get("schema") != SCHEMA:
            return
        self.open = {row["token"]: Ticket(**row) for row in state.get("open", [])}
        self.closed = [Ticket(**row) for row in state.get("closed", [])]
        self.skipped = Counter(state.get("skipped", {}))
        self._day_counts = Counter(state.get("days", {}))
        if state.get("totals"):
            self.totals = {**self.totals, **state["totals"]}

"""Bounded ASYMMETRIC_TAIL shadow proposals through the canonical allocator.

This module owns no execution, promotion or file-writing authority. Its state
is included in the desk's existing checkpoint and its proposals still require
the ordinary risk, allocation and execution path. Gross losses consume a
fixed lifetime research budget; the arrival of a monster is never assumed.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from src.strategies.opportunity_allocator import Opportunity, OpportunityAllocator, Slate


SLEEVE = "ASYMMETRIC_TAIL"
SCHEMA_VERSION = "v1"


@dataclass(frozen=True)
class TailSleevePolicy:
    total_budget_fraction: float = 0.03
    per_token_fraction: float = 0.001
    daily_loss_fraction: float = 0.005
    weekly_loss_fraction: float = 0.015
    actor_fraction: float = 0.005
    narrative_fraction: float = 0.01
    chain_fraction: float = 0.02
    max_open_positions: int = 30
    harvest_at_multiple: float = 3.0
    harvest_fraction: float = 0.5
    runner_retention_fraction: float = 0.1
    max_hold_seconds: float = 7 * 86400.0

    def __post_init__(self) -> None:
        for name in ("total_budget_fraction", "per_token_fraction",
                     "daily_loss_fraction", "weekly_loss_fraction", "actor_fraction",
                     "narrative_fraction", "chain_fraction", "harvest_fraction"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be finite and in (0, 1]")
        if not 0 <= self.runner_retention_fraction < 1:
            raise ValueError("runner_retention_fraction must be in [0, 1)")
        if self.harvest_fraction > 1 - self.runner_retention_fraction:
            raise ValueError("harvest would remove the declared runner")
        if (not math.isfinite(self.harvest_at_multiple) or self.harvest_at_multiple <= 1
                or not math.isfinite(self.max_hold_seconds) or self.max_hold_seconds <= 0):
            raise ValueError("harvest multiple and holding time must be finite and positive")
        if (isinstance(self.max_open_positions, bool)
                or not isinstance(self.max_open_positions, int) or self.max_open_positions < 1):
            raise ValueError("max_open_positions must be a positive integer")


@dataclass
class ShadowTailPosition:
    token: str
    chain: str
    actor_id: str
    narrative_id: str
    initial_cost_usd: float
    remaining_cost_usd: float
    opened_at: float
    remaining_fraction: float = 1.0
    proceeds_usd: float = 0.0
    harvested: bool = False


@dataclass
class TailProposal:
    slate: Slate
    proposed: List[Opportunity]
    blocked: Dict[str, str]
    shadow_only: bool = True

    def report(self) -> Dict[str, Any]:
        return {"sleeve": SLEEVE, "shadow_only": True,
                "canonical_slate": self.slate.report(),
                "proposed": [item.token for item in self.proposed],
                "blocked": dict(self.blocked)}


class AsymmetricTailSleeve:
    """Propose bounded shadow tickets; never submit or authorise capital."""

    def __init__(self, equity_usd: float, *, policy: Optional[TailSleevePolicy] = None,
                 allocator: Optional[OpportunityAllocator] = None):
        if not math.isfinite(equity_usd) or equity_usd <= 0:
            raise ValueError("a measured positive equity is required")
        self.equity_usd = float(equity_usd)
        self.policy = policy or TailSleevePolicy()
        self.allocator = allocator if allocator is not None else OpportunityAllocator()
        self.positions: Dict[str, ShadowTailPosition] = {}
        self.losses: List[Dict[str, float]] = []
        self.outcomes: List[Dict[str, Any]] = []
        self.shutdown_reason = ""
        self.realized_pnl_usd = 0.0
        self._peak_equity_usd = self.equity_usd
        self._max_drawdown = 0.0
        self._last_time: Optional[float] = None

    @property
    def budget_usd(self) -> float:
        return self.equity_usd * self.policy.total_budget_fraction

    @property
    def gross_loss_usd(self) -> float:
        return sum(item["loss_usd"] for item in self.losses)

    def _time(self, now: float) -> None:
        if not math.isfinite(now) or now < 0:
            raise ValueError("a finite UTC timestamp is required")
        if self._last_time is not None and now < self._last_time:
            raise ValueError("shadow state cannot be updated backwards in time")

    @staticmethod
    def _window(now: float, *, weekly: bool) -> str:
        at = datetime.fromtimestamp(now, timezone.utc)
        year, week, _ = at.isocalendar()
        return f"{year}-W{week:02d}" if weekly else at.date().isoformat()

    def _window_loss(self, now: float, *, weekly: bool) -> float:
        key = self._window(now, weekly=weekly)
        return sum(item["loss_usd"] for item in self.losses
                   if self._window(item["at"], weekly=weekly) == key)

    def _headroom(self, opportunity: Opportunity, now: float,
                  provisional: Sequence[Opportunity] = ()) -> float:
        costs = [(p.remaining_cost_usd, p.chain, p.actor_id, p.narrative_id)
                 for p in self.positions.values()]
        costs += [(p.capital_usd, p.metadata["chain"], p.metadata["actor_id"],
                   p.metadata["narrative_id"]) for p in provisional]
        if len(costs) >= self.policy.max_open_positions:
            return 0.0
        at_risk = sum(row[0] for row in costs)
        policy = self.policy
        limits = [
            self.equity_usd * policy.per_token_fraction,
            self.budget_usd - self.gross_loss_usd - at_risk,
            self.equity_usd * policy.daily_loss_fraction
                - self._window_loss(now, weekly=False) - at_risk,
            self.equity_usd * policy.weekly_loss_fraction
                - self._window_loss(now, weekly=True) - at_risk,
        ]
        for index, key, fraction in ((1, "chain", policy.chain_fraction),
                                     (2, "actor_id", policy.actor_fraction),
                                     (3, "narrative_id", policy.narrative_fraction)):
            used = sum(row[0] for row in costs if row[index] == opportunity.metadata[key])
            limits.append(self.equity_usd * fraction - used)
        return max(0.0, min(limits))

    def _blocked(self, item: Opportunity) -> Optional[str]:
        if self.shutdown_reason:
            return "SHUTDOWN:" + self.shutdown_reason
        if item.token in self.positions:
            return "already_open"
        if item.is_open_position:
            return "incumbent_is_not_a_new_ticket"
        for key in ("chain", "actor_id", "narrative_id"):
            if not isinstance(item.metadata.get(key), str) or not item.metadata[key].strip():
                return "DATA_BLOCKED_" + key
        if item.metadata.get("safety_clear") is not True:
            return "DATA_BLOCKED_safety"
        if item.metadata.get("exit_ready") is not True:
            return "DATA_BLOCKED_exit"
        if not math.isfinite(item.capital_usd) or item.blocked_reason:
            return item.blocked_reason or "DATA_BLOCKED_CAPITAL"
        if item.elogw <= 0:
            return "cash_has_greater_log_growth"
        return None

    @staticmethod
    def _bounded(item: Opportunity, ceiling: float) -> Optional[Opportunity]:
        size = min(item.capital_usd, ceiling)
        if size <= 1e-9:
            return None
        elogw = item.elogw
        if size < item.capital_usd:
            if item.elogw_at is None:
                return None
            try:
                elogw = float(item.elogw_at(size))
            except Exception:
                return None
        if not math.isfinite(elogw) or elogw <= 0:
            return None
        return replace(item, capital_usd=size, elogw=elogw, sleeve=SLEEVE,
                       metadata={**item.metadata, "shadow_only": True})

    def propose(self, opportunities: Sequence[Opportunity], *, now: float,
                incumbents: Sequence[Opportunity] = ()) -> TailProposal:
        """Return a jointly bounded batch ranked by the desk's allocator.

        Proposals reserve nothing. The canonical consumer must explicitly
        record a shadow entry, at which point every budget is checked again.
        Other sleeves' incumbents remain in the canonical comparison.
        """
        self._time(now)
        candidates: List[Opportunity] = []
        blocked: Dict[str, str] = {}
        seen = set()
        for item in opportunities:
            reason = "duplicate_ticket" if item.token in seen else self._blocked(item)
            seen.add(item.token)
            if reason:
                blocked[item.token] = reason
                continue
            ceiling = self._headroom(item, now)
            bounded = self._bounded(item, ceiling)
            if bounded is None:
                blocked[item.token] = ("budget_exhausted" if ceiling <= 1e-9
                                       else "DATA_BLOCKED_size_repricing")
            else:
                candidates.append(bounded)
        slate = self.allocator.rank([*incumbents, *candidates])
        proposed: List[Opportunity] = []
        eligible = {item.token for item in candidates}
        for item in slate.ranked:
            if item.is_open_position or item.token not in eligible:
                continue
            bounded = self._bounded(item, self._headroom(item, now, proposed))
            if bounded is None:
                blocked[item.token] = "batch_budget_or_repricing_limit"
            else:
                proposed.append(bounded)
        return TailProposal(slate, proposed, blocked)

    def record_shadow_entry(self, item: Opportunity, *, now: float) -> bool:
        """Account only for a shadow entry admitted by the canonical consumer."""
        self._time(now)
        if self._blocked(item) or item.sleeve != SLEEVE:
            return False
        if item.capital_usd > self._headroom(item, now) + 1e-9:
            return False
        self.positions[item.token] = ShadowTailPosition(
            item.token, item.metadata["chain"], item.metadata["actor_id"],
            item.metadata["narrative_id"], item.capital_usd, item.capital_usd, now)
        self._last_time = now
        return True

    def record_shadow_exit(self, token: str, *, fraction: float,
                           proceeds_usd: float, now: float) -> bool:
        """Record measured shadow proceeds, after every modeled exit cost."""
        self._time(now)
        if (not math.isfinite(fraction) or not 0 < fraction <= 1
                or not math.isfinite(proceeds_usd) or proceeds_usd < 0):
            raise ValueError("exit fraction/proceeds must be finite and nonnegative")
        position = self.positions.get(token)
        if position is None:
            return False
        cost = position.remaining_cost_usd * fraction
        pnl = proceeds_usd - cost
        self.realized_pnl_usd += pnl
        if pnl < 0:
            self.losses.append({"at": now, "loss_usd": -pnl})
        position.proceeds_usd += proceeds_usd
        position.remaining_cost_usd -= cost
        position.remaining_fraction *= 1 - fraction
        position.harvested = True
        if fraction == 1:
            self.outcomes.append({"token": token, "at": now,
                "multiple": position.proceeds_usd / position.initial_cost_usd,
                "pnl_usd": position.proceeds_usd - position.initial_cost_usd,
                "chain": position.chain, "actor_id": position.actor_id,
                "narrative_id": position.narrative_id,
                "holding_seconds": now - position.opened_at})
            del self.positions[token]
        wealth = self.equity_usd + self.realized_pnl_usd
        self._peak_equity_usd = max(self._peak_equity_usd, wealth)
        self._max_drawdown = max(self._max_drawdown, 1 - wealth / self._peak_equity_usd)
        self._last_time = now
        if self.gross_loss_usd >= self.budget_usd - 1e-9:
            self.shutdown("lifetime_loss_budget_exhausted")
        return True

    def exit_plan(self, token: str, multiple: Optional[float], *, now: float,
                  known_safety_veto: bool = False) -> Dict[str, Any]:
        self._time(now)
        position = self.positions.get(token)
        result = {"sleeve": SLEEVE, "shadow_only": True, "token": token,
                  "action": "HOLD", "exit_fraction": 0.0,
                  "runner_retention_fraction": self.policy.runner_retention_fraction}
        if position is None:
            return {**result, "status": "DATA_BLOCKED", "reason": "no_shadow_position"}
        if known_safety_veto or self.shutdown_reason or now - position.opened_at >= self.policy.max_hold_seconds:
            return {**result, "status": "OK", "action": "EXIT", "exit_fraction": 1.0,
                    "reason": ("hard_safety_veto" if known_safety_veto else
                               self.shutdown_reason or "holding_time_budget_exhausted")}
        if multiple is None or not math.isfinite(multiple) or multiple < 0:
            return {**result, "status": "DATA_BLOCKED", "reason": "unmeasured_executable_multiple"}
        if multiple >= self.policy.harvest_at_multiple and not position.harvested:
            return {**result, "status": "OK", "action": "HARVEST",
                    "exit_fraction": self.policy.harvest_fraction,
                    "reason": "retain_bounded_runner_after_partial_profits"}
        return {**result, "status": "OK", "reason": "within_declared_holding_budget"}

    def shutdown(self, reason: str) -> None:
        if not reason.strip():
            raise ValueError("shutdown requires a named reason")
        self.shutdown_reason = reason

    def report(self, *, now: float) -> Dict[str, Any]:
        self._time(now)
        open_risk = sum(p.remaining_cost_usd for p in self.positions.values())
        measured = bool(self.outcomes or self.losses or self.realized_pnl_usd)
        gains = sorted((max(0.0, row["pnl_usd"]) for row in self.outcomes), reverse=True)
        total_gains = sum(gains)
        monsters = [row["at"] for row in self.outcomes if row["multiple"] >= 10]
        concentration = {}
        for name in ("chain", "actor_id", "narrative_id"):
            values: Dict[str, float] = {}
            for position in self.positions.values():
                key = getattr(position, name)
                values[key] = values.get(key, 0.0) + position.remaining_cost_usd
            concentration[name] = values
        return {"schema": SCHEMA_VERSION, "sleeve": SLEEVE, "shadow_only": True,
            "live_authority": False, "status": "SHUTDOWN" if self.shutdown_reason else "SHADOW_ONLY",
            "shutdown_reason": self.shutdown_reason, "policy": asdict(self.policy),
            "budget_usd": self.budget_usd, "gross_loss_usd": self.gross_loss_usd,
            "open_risk_usd": open_risk,
            "remaining_lifetime_risk_usd": max(0.0, self.budget_usd - self.gross_loss_usd - open_risk),
            "daily_loss_usd": self._window_loss(now, weekly=False),
            "weekly_loss_usd": self._window_loss(now, weekly=True),
            "realized_pnl_usd": self.realized_pnl_usd if measured else None,
            "net_log_growth": math.log1p(self.realized_pnl_usd / self.equity_usd) if measured else None,
            "max_drawdown": self._max_drawdown if measured else None,
            "probability_of_ruin": None,
            "ruin_status": "UNMEASURED_requires_independent_stress_paths",
            "top_trade_return_share": {str(n): sum(gains[:n]) / total_gains if total_gains > 0 else None
                                       for n in (1, 5, 10)},
            "time_between_monsters_seconds": [b - a for a, b in zip(monsters, monsters[1:])]
                if len(monsters) > 1 else None,
            "closed_shadow_trades": len(self.outcomes), "open_shadow_positions": len(self.positions),
            "concentration_usd": concentration, "capital_tied_up_usd": open_risk,
            "exit_reliability": None, "bet_return_correlation": None,
            "metric_scope": "realized shadow proceeds; open maximum-loss risk reported separately"}

    def state(self) -> Dict[str, Any]:
        """For the canonical desk checkpoint; this object writes no files."""
        return {"schema": SCHEMA_VERSION, "equity_usd": self.equity_usd,
            "policy": asdict(self.policy), "positions": [asdict(p) for p in self.positions.values()],
            "losses": [dict(row) for row in self.losses],
            "outcomes": [dict(row) for row in self.outcomes],
            "realized_pnl_usd": self.realized_pnl_usd,
            "peak_equity_usd": self._peak_equity_usd, "max_drawdown": self._max_drawdown,
            "shutdown_reason": self.shutdown_reason, "last_time": self._last_time}

    def restore(self, state: Dict[str, Any]) -> None:
        """Refuse incompatible checkpoints rather than silently renew risk."""
        if (state.get("schema") != SCHEMA_VERSION
                or state.get("equity_usd") != self.equity_usd
                or state.get("policy") != asdict(self.policy)):
            raise ValueError("tail checkpoint schema/equity/policy mismatch")
        positions = [ShadowTailPosition(**row) for row in state["positions"]]
        losses = [dict(row) for row in state["losses"]]
        if any(not math.isfinite(row["loss_usd"]) or row["loss_usd"] < 0
               or not math.isfinite(row["at"]) for row in losses):
            raise ValueError("invalid shadow loss checkpoint")
        if len({p.token for p in positions}) != len(positions):
            raise ValueError("duplicate shadow positions")
        for p in positions:
            if (not all(isinstance(getattr(p, name), str) and getattr(p, name).strip()
                        for name in ("token", "chain", "actor_id", "narrative_id"))
                    or not all(math.isfinite(getattr(p, name)) for name in
                               ("initial_cost_usd", "remaining_cost_usd", "opened_at",
                                "remaining_fraction", "proceeds_usd"))
                    or p.initial_cost_usd <= 0
                    or not 0 < p.remaining_cost_usd <= p.initial_cost_usd
                    or not 0 < p.remaining_fraction <= 1
                    or p.proceeds_usd < 0 or p.opened_at < 0):
                raise ValueError("invalid shadow position checkpoint")
        realized = float(state["realized_pnl_usd"])
        peak = float(state["peak_equity_usd"])
        drawdown = float(state["max_drawdown"])
        if (not all(math.isfinite(value) for value in (realized, peak, drawdown))
                or realized <= -self.equity_usd or peak < self.equity_usd
                or not 0 <= drawdown <= 1):
            raise ValueError("invalid shadow return checkpoint")
        last_time = state["last_time"]
        if last_time is not None and (not math.isfinite(last_time) or last_time < 0):
            raise ValueError("invalid shadow checkpoint timestamp")
        if (sum(row["loss_usd"] for row in losses)
                + sum(p.remaining_cost_usd for p in positions) > self.budget_usd + 1e-8):
            raise ValueError("shadow checkpoint exceeds the lifetime risk budget")
        self.positions = {p.token: p for p in positions}
        self.losses = losses
        self.outcomes = [dict(row) for row in state["outcomes"]]
        self.realized_pnl_usd = realized
        self._peak_equity_usd = peak
        self._max_drawdown = drawdown
        self.shutdown_reason = str(state["shutdown_reason"])
        self._last_time = last_time

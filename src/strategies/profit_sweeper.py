"""Segregated-profit planning with no transaction authority."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ProfitIsolationPolicy:
    working_capital_usd: float
    sweep_trigger_usd: float
    isolation_fraction: float = 1.0

    def __post_init__(self):
        if self.working_capital_usd < 0 or self.sweep_trigger_usd < 0:
            raise ValueError("capital and trigger must be non-negative")
        if not 0 < self.isolation_fraction <= 1:
            raise ValueError("isolation_fraction must be in (0, 1]")

    def plan(self, *, equity_usd: Optional[float], cold_destination: str = "",
             dry_run: bool = True) -> Dict[str, Any]:
        if equity_usd is None or equity_usd < 0:
            return {"status": "DATA_BLOCKED", "detail": "equity is not measured"}
        surplus = max(0.0, float(equity_usd) - self.working_capital_usd)
        if surplus < self.sweep_trigger_usd:
            return {"status": "HOLD", "surplus_usd": surplus,
                    "working_capital_usd": self.working_capital_usd}
        if not cold_destination:
            return {"status": "DATA_BLOCKED", "surplus_usd": surplus,
                    "detail": "cold destination is not configured"}
        amount = surplus * self.isolation_fraction
        return {
            "status": "PAPER_PLAN" if dry_run else "AWAITING_SEPARATE_AUTHORIZATION",
            "destination": cold_destination, "amount_usd": amount,
            "surplus_usd": surplus, "working_capital_usd": self.working_capital_usd,
            "transaction_created": False,
            "detail": "planning only; this module cannot sign or submit transfers",
        }

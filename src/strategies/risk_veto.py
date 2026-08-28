"""One fail-closed safety authority, separate from alpha ranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


DANGEROUS_EXTENSIONS = frozenset({
    "transfer_fee_config", "default_account_state", "non_transferable",
    "permanent_delegate", "transfer_hook", "confidential_transfer_mint",
    "confidential_transfer_fee_config", "confidential_mint_burn", "pausable",
})


@dataclass(frozen=True)
class RiskVetoResult:
    status: str
    reasons: List[str] = field(default_factory=list)
    unmeasured: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def clear(self) -> bool:
        return self.status == "CLEAR"

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "clear": self.clear,
                "reasons": list(self.reasons),
                "unmeasured": list(self.unmeasured),
                "evidence": dict(self.evidence)}


class RiskVeto:
    """Turns observed non-negotiable safety facts into a hard veto.

    No alpha score is accepted here.  A model cannot vote an active freeze
    authority away, and a missing route is not a low score: it is an unknown
    ability to leave the position.
    """

    def __init__(self, *, require_complete_safety: bool = True,
                 max_exit_impact_pct: float = 0.20,
                 max_liquidity_fraction: float = 0.01):
        self.require_complete_safety = bool(require_complete_safety)
        self.max_exit_impact_pct = max(0.0, float(max_exit_impact_pct))
        self.max_liquidity_fraction = max(0.0, float(max_liquidity_fraction))

    def evaluate(self, report: Any, *, dev_state: Optional[Dict[str, Any]] = None,
                 position_value_usd: Optional[float] = None,
                 liquidity_usd: Optional[float] = None,
                 exit_capacity_ratio: Optional[float] = None,
                 connected_holder_pct: Optional[float] = None,
                 max_connected_holder_pct: float = 80.0) -> RiskVetoResult:
        reasons: List[str] = []
        unmeasured: List[str] = []
        checks = getattr(report, "checks", {}) or {}
        evidence: Dict[str, Any] = {
            "risk_level": getattr(getattr(report, "risk_level", None), "value", None),
            "risk_score": getattr(report, "score", None),
        }
        status = str(getattr(report, "data_status", "DATA_BLOCKED"))
        if status != "OK":
            unmeasured.extend(getattr(report, "blocked_checks", ()) or ("safety_report",))
        if bool(getattr(report, "can_mint", False)):
            reasons.append("mint_authority_active")
        if bool(getattr(report, "can_freeze", False)):
            reasons.append("freeze_authority_active")

        extensions = set(getattr(report, "token_extensions", ()) or ())
        dangerous = sorted(extensions & DANGEROUS_EXTENSIONS)
        reasons.extend(f"dangerous_token_extension:{name}" for name in dangerous)

        route = checks.get("sell_route") or {}
        route_feasible = getattr(report, "sell_route_feasible", None)
        if route_feasible is False:
            reasons.append("sell_route_unavailable")
        elif route_feasible is None:
            unmeasured.append("sell_route")
        impact = route.get("price_impact_pct")
        if impact is not None and float(impact) > self.max_exit_impact_pct:
            reasons.append("catastrophic_exit_price_impact")

        level = evidence["risk_level"]
        if level in {"high", "critical", "honeypot", "rugged"}:
            reasons.append(f"native_risk_level:{level}")

        dev_state = dev_state or {}
        reasons.extend(f"developer:{reason}" for reason in dev_state.get("hard_vetoes", ()))

        if connected_holder_pct is None:
            unmeasured.append("connected_holder_concentration")
        elif float(connected_holder_pct) >= float(max_connected_holder_pct):
            reasons.append("extreme_connected_holder_concentration")

        if position_value_usd is not None:
            if liquidity_usd is None or float(liquidity_usd) <= 0:
                unmeasured.append("exit_liquidity")
            elif float(position_value_usd) > float(liquidity_usd) * self.max_liquidity_fraction:
                reasons.append("position_exceeds_exit_liquidity_limit")
            if exit_capacity_ratio is None:
                unmeasured.append("exit_capacity_ratio")
            elif float(exit_capacity_ratio) < 1.0:
                reasons.append("full_position_not_executable_at_impact_limit")

        reasons = sorted(set(reasons))
        unmeasured = sorted(set(unmeasured) - set(getattr(report, "blocked_checks", ()) or ())) \
            + sorted(set(getattr(report, "blocked_checks", ()) or ()))
        evidence.update({"dangerous_extensions": dangerous,
                         "route_feasible": route_feasible,
                         "route_price_impact_pct": impact,
                         "position_value_usd": position_value_usd,
                         "liquidity_usd": liquidity_usd,
                         "exit_capacity_ratio": exit_capacity_ratio,
                         "connected_holder_pct": connected_holder_pct})
        if reasons:
            return RiskVetoResult("VETO", reasons, unmeasured, evidence)
        if unmeasured and self.require_complete_safety:
            return RiskVetoResult("DATA_BLOCKED", reasons, unmeasured, evidence)
        return RiskVetoResult("CLEAR", reasons, unmeasured, evidence)

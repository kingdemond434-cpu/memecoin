"""What the desk froze before it acted, and what it learned afterwards.

Two records, deliberately in one place because they are the two halves of
the same claim. The evidence packet is written BEFORE the action it
justifies: every measurement the entry thesis rested on, with each one
labelled by whether it was measured or unavailable. The calibration record
is written afterwards, comparing what was predicted to what happened.

The ordering is the point. A row amended after the outcome is known leaks
the future into its own features, and a model trained on that looks
extraordinary in backtest and fails forward. So the thesis is sealed at
decision time and never rewritten -- an unavailable measurement stays
DATA_BLOCKED rather than becoming a zero that reads like a fact.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from src.research.calibration import Provenance
from src.research.launch_census import rug_mechanism_monster_threshold
from src.research.trade_evidence import evidence_packet
from src.runtime.serialisation import jsonable as _jsonable

logger = logging.getLogger(__name__)


class EvidenceRecording:
    """Freezes the entry thesis, and records how it turned out."""

    def _record_calibration(self, payload: Dict[str, Any]) -> None:
        """Score every stated probability against what actually happened.

        Without this the calibration harness is inert -- it can measure, and
        nothing feeds it. Each pair is stamped with where it came from, so a
        model validated on shadow decisions is never mistaken for one proven
        on real fills.

        Probabilities the desk did not state are skipped rather than defaulted.
        A model that was never consulted has no prediction to score, and
        scoring it against a default would manufacture calibration evidence
        out of the model's absence.
        """
        provenance = (Provenance.FORWARD_REAL.value
                      if payload.get("entered") and not self.dry_run
                      else Provenance.SHADOW.value)
        when = float(payload.get("timestamp", time.time()))
        rugged = payload.get("rugged")
        stated = payload.get("stated_probabilities") or {}

        # Rug hazards, at each horizon the desk states one for.
        for model, key in (("rug_30s", "p_rug_30s"), ("rug_5m", "p_rug_5m")):
            probability = stated.get(key)
            if probability is None or rugged is None:
                continue
            self.calibration.record(model, float(probability), bool(rugged),
                                    at=when, provenance=provenance)

        # Monster: did the token reach the multiple the desk said it might?
        monster_p = stated.get("p_monster")
        multiple = payload.get("max_feasible_multiple")
        if monster_p is not None and multiple is not None:
            self.calibration.record(
                "monster_p", float(monster_p),
                float(multiple) >= rug_mechanism_monster_threshold(),
                at=when, provenance=provenance)

        # Landing: only scoreable on an attempt, and only real when the
        # attempt spent real money.
        landing_p = stated.get("p_land")
        if landing_p is not None and payload.get("attempted"):
            self.calibration.record(
                "landing_p", float(landing_p), bool(payload.get("entered")),
                at=when,
                provenance=(Provenance.FORWARD_REAL.value if not self.dry_run
                            else Provenance.SHADOW.value))

        # Escape: stated when a hazard exit was chosen, scored on whether the
        # position actually got out before the catastrophe.
        escape_p = stated.get("p_escape")
        if escape_p is not None and payload.get("escape_attempted"):
            self.calibration.record(
                "escape_p", float(escape_p), bool(payload.get("escaped")),
                at=when, provenance=provenance)



    #: Read once at first request and cached. The file ships with the repo,
    #: so a missing one is a broken install rather than a runtime condition,
    #: and it says so instead of serving a blank page.
    _dashboard_cache: Optional[str] = None












    def _record_trade_evidence_packet(
        self, token: str, candidate: Any, risk: Any, liquidity: float,
        trade_info: Dict[str, Any], intelligence: Dict[str, Any], *,
        decision: str, veto: Dict[str, Any],
    ) -> None:
        """Freeze the complete measured entry thesis before execution."""
        if not self.trade_evidence:
            return
        curve = self._latest_curve_state.get(token)
        pool = self._latest_pool_state.get(token)
        if curve is not None:
            bonding = {
                "status": "OK", "venue": "pump_fun",
                "virtual_token_reserves": curve.virtual_token_reserves,
                "virtual_sol_reserves": curve.virtual_sol_reserves,
                "real_token_reserves": curve.real_token_reserves,
                "real_sol_reserves": curve.real_sol_reserves,
                "complete": curve.complete,
            }
        elif pool is not None:
            bonding = {"status": "OK", "venue": "pump_swap", "pool": pool.pool,
                       "base_reserves": pool.base_reserves,
                       "quote_reserves": pool.quote_reserves}
        else:
            bonding = {"status": "DATA_BLOCKED", "detail": "no native venue state"}
        checks = getattr(risk, "checks", {}) or {}
        actors = intelligence.get("actors") or {
            "status": "DATA_BLOCKED", "detail": "actor graph unavailable"}
        packet = evidence_packet(
            mint=token, timestamp=time.time(), bonding_curve=bonding,
            liquidity={"status": "OK" if liquidity > 0 else "DATA_BLOCKED",
                       "liquidity_usd": liquidity},
            sellability=checks.get("sell_route") or {
                "status": "DATA_BLOCKED", "detail": "sell route not measured"},
            authorities={"status": getattr(risk, "data_status", "DATA_BLOCKED"),
                         "mint_active": getattr(risk, "can_mint", None),
                         "freeze_active": getattr(risk, "can_freeze", None),
                         "extensions": getattr(risk, "token_extensions", ())},
            holder_distribution=self.holder_trajectory.state(token),
            wallet_clusters=actors,
            dev_wallet=self.dev_wallet_monitor.state(token),
            smart_wallet_flow=actors.get("smart_flow") or {
                "status": "DATA_BLOCKED", "detail": "quality flow unavailable"},
            social_velocity=intelligence.get("social") or {
                "status": "DATA_BLOCKED", "detail": "social velocity unavailable"},
            entry_cost=self._cost_model(token),
            exit_liquidity={"status": "OK" if liquidity > 0 else "DATA_BLOCKED",
                            "liquidity_usd": liquidity,
                            "capacity_limit_fraction": self.risk_veto.max_liquidity_fraction},
            risk_vetoes=veto.get("reasons", ()),
            expected_edge=(float(trade_info["elogw"])
                           if trade_info.get("elogw") is not None else None),
            position_size=(float(trade_info["position_value_usd"])
                           if trade_info.get("position_value_usd") is not None else None),
            exit_plan={"status": self.exit_policy_status,
                       "policy": _jsonable(self.exit_policy)},
            decision=decision,
        )
        self.trade_evidence.record("candidate_decision", packet)

    def _prelaunch_context(self, deployer: str, detected_at: float) -> Optional[Dict[str, Any]]:
        profile = self.prelaunch.get_entity_profile(deployer) if deployer else None
        if not profile or profile.last_active > detected_at:
            return None
        return {
            "as_of": profile.last_active,
            "deployer_features": {"prior_launches": len(profile.prior_launches),
                                  "prior_success_rate": profile.prior_success_rate,
                                  "prior_rug_rate": profile.prior_rug_rate},
            "wallet_features": {"cluster_id": profile.cluster_id or ""},
            "social_features": {"social_creations": len(profile.social_creations)},
            "entity_graph_features": {"intent_score": profile.intent_score},
        }

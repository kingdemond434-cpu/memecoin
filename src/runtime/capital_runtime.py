"""Desk integration for the canonical portfolio opportunity allocator."""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Tuple

from src.strategies.opportunity_allocator import Opportunity

logger = logging.getLogger(__name__)


class CapitalRuntime:
    async def _contest_for_capital(
        self, token: str, candidate: Any, prediction: Any,
        liquidity: float, trade_info: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        """Let a rejected-on-capacity candidate contest the weakest position.

        A hurdle applied one token at a time makes capital a first-come
        resource: ten mediocre positions that each cleared it will lock out an
        eleventh that is far better, for as long as they are held. This asks
        the only question that actually matters -- is this the best place for
        the next dollar right now -- and, if it clearly is, proposes closing
        the worst incumbent to fund it.

        It proposes; it does not widen any limit. The freed capital still has
        to clear ``should_trade`` afterwards, so the exposure ceiling, the
        portfolio risk budget and the daily-loss kill switch all still apply.
        """
        elogw, fraction, size_sol = self.elogw_engine.calculate_expected_log_growth(
            prediction, self.sol_price_usd, liquidity)
        if not math.isfinite(elogw) or size_sol <= 0:
            return False, {**trade_info, "contest": "DATA_BLOCKED_CHALLENGER_ELOGW"}

        equity = max(self.wallet_equity_usd, 1e-9)
        challenger = Opportunity(
            token=token, elogw=elogw, capital_usd=size_sol * self.sol_price_usd,
            expected_hold_seconds=(float(prediction.expected_hold_time)
                                   if prediction.expected_hold_time > 0 else None),
            liquidity_usd=liquidity, sleeve=candidate.metadata.get("sleeve", "t0_sniper"),
            # Freed capital rarely matches the optimal size exactly, so the
            # allocator is given the means to re-price this candidate at
            # whatever a displacement would actually fund.
            elogw_at=lambda usd: self.elogw_engine.log_growth_at_fraction(
                prediction, min(usd / equity, self.elogw_engine.exposure_cap(liquidity))),
        )
        incumbents = await self._incumbent_opportunities()
        cash = max(0.0, self.wallet_equity_usd - sum(
            item.capital_usd * item.held_multiple for item in incumbents))
        slate = self.opportunity_allocator.rank(incumbents + [challenger],
            cash_available_usd=cash, cash_horizon_seconds=challenger.expected_hold_seconds)
        self.last_slate_report = slate.report()
        self._feed_opportunity_quality(slate)

        move = next((item for item in slate.displacements
                     if item.challenger.token == token), None)
        if move is None:
            return False, {**trade_info, "contest": "no_incumbent_worth_displacing",
                           "slate": self.last_slate_report}

        incumbent_position = self.elogw_engine.open_positions.get(move.incumbent.token)
        if incumbent_position is None:
            return False, {**trade_info, "contest": "incumbent_closed_before_displacement"}
        logger.info("DISPLACE %s (score=%.3e) for %s (score=%.3e, cost=$%.2f)",
                    move.incumbent.token, move.incumbent_score, token,
                    move.challenger_score_after_cost, move.round_trip_cost_usd)
        await self._execute_exit(move.incumbent.token, incumbent_position, 1.0,
                                 f"displaced_by_{token}")
        if move.incumbent.token in self.elogw_engine.open_positions:
            # The exit did not actually clear the position (no fill, or an
            # unverified balance change). Capital was never freed, so the
            # challenger must not be funded as though it had been.
            return False, {**trade_info, "contest": "displacement_exit_did_not_fill"}

        await self._refresh_portfolio_state()
        return self.elogw_engine.should_trade(
            prediction, self.sol_price_usd, liquidity, self.wallet_equity_usd)

    async def _incumbent_opportunities(self) -> List[Opportunity]:
        """Score every open position on its FORWARD growth, not its history.

        What a position has already made is sunk. The only thing that competes
        with a new candidate is what it is expected to add from here, per
        dollar still tied up, per second it stays tied up.
        """
        opportunities: List[Opportunity] = []
        for held_token, position in self.elogw_engine.open_positions.items():
            prediction = position.get("prediction_object")
            held_cost = float(position.get("remaining_cost_usd", 0) or 0)
            liquidity = position.get("liquidity_usd")
            if prediction is None or held_cost <= 0:
                continue
            multiple = float(position.get("high_water_multiple", 1.0) or 1.0)
            held_fraction = (held_cost / self.elogw_engine.portfolio_value
                             if self.elogw_engine.portfolio_value > 0 else 0.0)
            forward = self.elogw_engine.marginal_log_growth(prediction, held_fraction, multiple, 0.0)
            hold_time = float(getattr(prediction, "expected_hold_time", 0) or 0)
            elapsed = max(0.0, time.time() - float(position.get("entry_time", time.time())))
            opportunities.append(Opportunity(
                token=held_token,
                elogw=forward,
                capital_usd=held_cost,
                # Remaining expected hold, not total: capital already committed
                # for 200s of a 300s expected hold is 100s from being free.
                expected_hold_seconds=(max(1.0, hold_time - elapsed) if hold_time > 0 else None),
                liquidity_usd=(float(liquidity) if liquidity else None),
                sleeve=str(position.get("sleeve", "t0_sniper")),
                is_open_position=True,
                held_multiple=multiple,
            ))
        return opportunities


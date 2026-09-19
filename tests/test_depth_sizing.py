"""The position ceiling was a constant standing in for a measurement.

`max_liquidity_fraction: 0.01` says a position may not exceed 1% of
liquidity. On a constant-product bonding curve that number is not pricing
what it looks like it is pricing: a buy followed immediately by a sell
returns to the same point on the curve, so the round trip costs the fee and
nothing else AT ANY SIZE. 0.5% of the pool and 50% of the pool both retain
98.01% with a 100bps-per-leg fee. The fraction was never a cost control.

What it was crudely standing in for is exit -- how much of the position can
be turned back into SOL inside an impact we would accept, bounded by the real
SOL the curve is physically holding. The desk can measure that exactly from
state it already streams (`sell_capacity_lamports` existed for precisely this
and had no caller outside tests), so it is measured.

Two failures are pinned here alongside it, both found while checking what the
cap was really doing:

  * `_local_liquidity` preferred real reserves and fell back to virtual, so a
    fresh Pump curve reported 30 SOL of depth and the same curve one 0.1 SOL
    buy later reported 0.1. Depth fell thirtyfold because somebody bought,
    and it stayed under the $5,000 minimum-liquidity gate until roughly 25
    SOL had traded -- past the window the desk exists to trade.
  * Applying the flat fraction ON TOP of the measurement would veto positions
    the measurement had just cleared, which is how a better instrument makes
    a system more timid instead of more accurate.
"""

from types import SimpleNamespace

import pytest

from src.chains.pump_curve import (
    BondingCurveState, LAMPORTS_PER_SOL, quote_buy, quote_sell)
from src.execution.tradeability import curve_depth_usd
from src.research.dataset_builder import (
    PUMP_INITIAL_VIRTUAL_SOL, PUMP_INITIAL_VIRTUAL_TOKEN, PUMP_TOKEN_TOTAL_SUPPLY)
from src.runtime.depth import DepthResolution
from src.strategies.multihead_predictor import (
    ElogwEngine, MultiHeadPredictor, MultiHeadPrediction)
from src.strategies.risk_veto import RiskVeto

SOL_USD = 200.0


def _curve(sol_bought: float = 0.0) -> BondingCurveState:
    """A Pump curve after ``sol_bought`` SOL has actually been bought in."""
    state = BondingCurveState(
        virtual_token_reserves=PUMP_INITIAL_VIRTUAL_TOKEN,
        virtual_sol_reserves=PUMP_INITIAL_VIRTUAL_SOL,
        real_token_reserves=793_100_000_000_000,
        real_sol_reserves=0,
        token_total_supply=PUMP_TOKEN_TOTAL_SUPPLY,
        complete=False)
    if sol_bought > 0:
        lamports = int(sol_bought * LAMPORTS_PER_SOL)
        quote = quote_buy(state, lamports)
        net = lamports - quote.fee_amount
        state.virtual_sol_reserves += net
        state.virtual_token_reserves -= quote.output_amount
        state.real_sol_reserves += net
        state.real_token_reserves -= quote.output_amount
    return state


def _desk(curve, sol_price=SOL_USD, impact=0.10):
    desk = DepthResolution()
    desk._latest_curve_state = {"mint": curve} if curve is not None else {}
    desk.sol_price_usd = sol_price
    desk.global_config = {"acceptable_exit_impact": impact}
    return desk


class TestMeasuredExitDepth:
    def test_a_curve_holding_no_real_sol_has_no_measurable_exit(self):
        """At T0 exit liquidity is a forecast about flow that has not arrived.

        The virtual reserves would happily quote a sale the curve cannot fund.
        Refusing is the only honest answer; the caller falls back to its
        declared assumption knowing that is what it is doing.
        """
        assert curve_depth_usd(_curve(0.0), 0.10, SOL_USD) is None

    def test_measured_depth_matches_an_independent_round_trip(self):
        state = _curve(20.0)
        depth = curve_depth_usd(state, 0.10, SOL_USD)
        assert depth is not None
        # Independently: sell exactly that notional back and confirm the
        # impact really does land inside the bound it was measured at.
        tokens = quote_buy(state, int(depth / SOL_USD * LAMPORTS_PER_SOL)).output_amount
        assert quote_sell(state, tokens).price_impact_pct <= 0.10 + 1e-9

    def test_measurement_is_more_permissive_than_the_flat_fraction(self):
        """And that is the point: the constant was not conservative, it was
        uninformed. Both are ceilings on the same quantity; only one of them
        looked at the curve."""
        state = _curve(20.0)
        measured = curve_depth_usd(state, 0.10, SOL_USD)
        flat = (state.virtual_sol_reserves / LAMPORTS_PER_SOL) * SOL_USD * 0.01
        assert measured > flat

    def test_depth_is_refused_rather_than_guessed_without_a_curve(self):
        assert _desk(None)._measured_depth_usd("mint") is None

    def test_a_completed_curve_has_migrated_and_is_not_measured_here(self):
        state = _curve(20.0)
        state.complete = True
        assert curve_depth_usd(state, 0.10, SOL_USD) is None


class TestLiquidityIsMonotoneInCurveProgress:
    def test_liquidity_does_not_collapse_when_the_first_buyer_arrives(self):
        """The regression that mattered: buying used to shrink reported depth."""
        at_t0 = _desk(_curve(0.0))._local_liquidity("mint")
        after = _desk(_curve(0.1))._local_liquidity("mint")
        assert after >= at_t0

    def test_liquidity_rises_with_every_further_buy(self):
        readings = [_desk(_curve(x))._local_liquidity("mint")
                    for x in (0.0, 0.1, 1.0, 5.0, 20.0, 60.0)]
        assert readings == sorted(readings)

    def test_a_young_curve_clears_the_minimum_liquidity_gate(self):
        """$5,000 is the configured floor. A curve one buy old used to read
        $20 against it, which rejected the entire T0 population."""
        assert _desk(_curve(0.1))._local_liquidity("mint") >= 5_000.0


def _engine(**kwargs):
    predictor = MultiHeadPredictor()
    predictor._is_trained = True
    engine = ElogwEngine(predictor, min_edge_bps=-1, drawdown_aversion_lambda=0,
                         max_position_pct=0.05, max_liquidity_fraction=0.01,
                         **kwargs)
    engine.portfolio_value = 10_000.0
    return engine


def _prediction():
    return MultiHeadPrediction("mint", "solana", 0, p_2x=0.85, p_5x=0.6, p_10x=0.3,
                               p_50x=0.05, p_rug_30s=0.01, p_rug_5m=0.02,
                               expected_slippage=0.01)


class TestTheCeilingUsesMeasurementWhenThereIsOne:
    def test_measured_depth_replaces_the_flat_fraction_rather_than_joining_it(self):
        engine = _engine()
        ceilings = engine.exposure_ceilings(10_000.0, depth_usd=900.0)
        assert engine.CEILING_MEASURED_DEPTH in ceilings
        assert engine.CEILING_POOL_DEPTH not in ceilings
        assert ceilings[engine.CEILING_MEASURED_DEPTH] == pytest.approx(0.09)

    def test_the_flat_fraction_remains_the_fallback_when_nothing_was_measured(self):
        engine = _engine()
        ceilings = engine.exposure_ceilings(10_000.0, depth_usd=None)
        assert engine.CEILING_POOL_DEPTH in ceilings
        assert engine.CEILING_MEASURED_DEPTH not in ceilings

    def test_an_unmeasurable_depth_is_never_read_as_unlimited(self):
        """None and 0.0 both fall back to the declared fraction. Neither is a
        licence to size without a ceiling."""
        engine = _engine()
        for value in (None, 0.0):
            cap = engine.exposure_cap(10_000.0, depth_usd=value)
            assert cap == pytest.approx(engine.exposure_cap(10_000.0))

    def test_measurement_raises_the_cap_it_replaces(self):
        engine = _engine()
        flat = engine.exposure_cap(10_000.0)
        measured = engine.exposure_cap(10_000.0, depth_usd=900.0)
        assert measured > flat

    def test_the_capacity_report_counts_both_kinds_of_depth_bind(self):
        engine = _engine()
        engine.exposure_cap(10_000.0)
        engine.exposure_cap(10_000.0, depth_usd=1.0)
        report = engine.capacity_report()
        assert report["depth_bound_share"] == pytest.approx(1.0)
        assert engine.CEILING_MEASURED_DEPTH in report["bound_by"]
        assert engine.CEILING_POOL_DEPTH in report["bound_by"]

    def test_the_size_says_which_ceiling_bound_it_and_on_what_evidence(self):
        # $300 of measured depth on $10k of equity is 3%, inside the 5%
        # concentration ceiling, so depth is what actually binds here.
        engine = _engine()
        sized = engine.size_candidate(_prediction(), SOL_USD, 10_000.0, depth_usd=300.0)
        assert sized["binding_ceiling"] == engine.CEILING_MEASURED_DEPTH
        assert sized["measured_depth_usd"] == 300.0
        assert sized["kelly_fraction"] <= 0.03 + 1e-9


class TestTheVetoDoesNotOverrideTheMeasurement:
    def _report(self):
        return SimpleNamespace(data_status="OK", checks={}, risk_level=None,
                               score=0, blocked_checks=())

    def test_a_position_inside_measured_depth_is_not_vetoed_by_the_fraction(self):
        veto = RiskVeto(require_complete_safety=False, max_liquidity_fraction=0.01)
        result = veto.evaluate(self._report(), position_value_usd=500.0,
                               liquidity_usd=10_000.0, depth_usd=900.0)
        assert "position_exceeds_exit_liquidity_limit" not in result.reasons
        assert "position_exceeds_measured_exit_depth" not in result.reasons

    def test_a_position_beyond_measured_depth_is_still_vetoed(self):
        veto = RiskVeto(require_complete_safety=False, max_liquidity_fraction=0.01)
        result = veto.evaluate(self._report(), position_value_usd=1_500.0,
                               liquidity_usd=10_000.0, depth_usd=900.0)
        assert "position_exceeds_measured_exit_depth" in result.reasons

    def test_without_a_measurement_the_flat_fraction_still_binds(self):
        veto = RiskVeto(require_complete_safety=False, max_liquidity_fraction=0.01)
        result = veto.evaluate(self._report(), position_value_usd=500.0,
                               liquidity_usd=10_000.0, depth_usd=None)
        assert "position_exceeds_exit_liquidity_limit" in result.reasons

    def test_the_veto_records_which_depth_it_judged_against(self):
        veto = RiskVeto(require_complete_safety=False, max_liquidity_fraction=0.01)
        result = veto.evaluate(self._report(), position_value_usd=500.0,
                               liquidity_usd=10_000.0, depth_usd=900.0)
        assert result.evidence["measured_exit_depth_usd"] == 900.0

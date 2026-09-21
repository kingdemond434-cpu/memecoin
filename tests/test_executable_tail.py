"""A 1000x on a chart is not a 1000x in a position.

The number a memecoin prints is a market-cap multiple. Realising it means
selling into the venue that produced it, and the sale moves the venue. A tail
strategy that optimises the printed multiple is optimising a screenshot.

Two things fall out of the arithmetic here and both are findings rather than
features. A Pump bonding curve completes at roughly 85 SOL of real reserves,
which from a T0 entry is 14.4x -- so the curve cannot price a 20x exit, let
alone a 1000x one, and every rung above that needs the migrated pool. And
without observed pool state those rungs are unmeasurable, which is the entire
range the strategy exists for.
"""

import math
import time

import pytest

from src.chains.pump_curve import (
    BondingCurveState, LAMPORTS_PER_SOL, quote_buy)
from src.chains.pumpswap_curve import PumpSwapPoolState
from src.execution.executable_tail import (
    DEFAULT_TAIL_LADDER, capturable_upside, executable_tail_curve,
    migration_multiple, pool_at_migration, project_curve, project_pool)
from src.research.dataset_builder import (
    PUMP_INITIAL_VIRTUAL_SOL, PUMP_INITIAL_VIRTUAL_TOKEN,
    PUMP_TOKEN_TOTAL_SUPPLY)

CURVE = [(1.0, 1.0), (2.0, 0.35), (5.0, 0.12), (10.0, 0.05), (20.0, 0.02),
         (50.0, 0.006), (100.0, 0.002), (250.0, 0.0008), (500.0, 0.0003)]


def _entered(sol=0.3):
    """A curve after we bought `sol` into it at T0, plus our position."""
    state = BondingCurveState(
        virtual_token_reserves=PUMP_INITIAL_VIRTUAL_TOKEN,
        virtual_sol_reserves=PUMP_INITIAL_VIRTUAL_SOL,
        real_token_reserves=793_100_000_000_000, real_sol_reserves=0,
        token_total_supply=PUMP_TOKEN_TOTAL_SUPPLY, complete=False)
    lamports = int(sol * LAMPORTS_PER_SOL)
    buy = quote_buy(state, lamports)
    net = lamports - buy.fee_amount
    state.virtual_sol_reserves += net
    state.virtual_token_reserves -= buy.output_amount
    state.real_sol_reserves += net
    state.real_token_reserves -= buy.output_amount
    return state, buy.output_amount, lamports


def _pool(base=206_900_000_000_000, quote=85_000_000_000):
    return PumpSwapPoolState(
        pool="p", base_mint="b", quote_mint="q", base_reserves=base,
        quote_reserves=quote, virtual_quote_reserves=0,
        total_fee_bps=100, updated_at=time.time())


def _survival(model):
    return lambda multiple: model.survival(CURVE, multiple)


class TestTheProjectionIsExactCurveArithmetic:
    def test_reserves_scale_as_the_square_root_of_the_price(self):
        """price = S**2 / k on a constant product, so S grows as sqrt(m).
        Checked against the curve's own quote rather than asserted."""
        state, _tokens, _cost = _entered()
        before = state.price_sol_per_token
        for multiple in (2.0, 5.0, 10.0):
            projected = project_curve(state, multiple)
            assert projected.price_sol_per_token / before == pytest.approx(
                multiple, rel=1e-6)
            assert (projected.virtual_sol_reserves / state.virtual_sol_reserves
                    == pytest.approx(math.sqrt(multiple), rel=1e-6))

    def test_the_product_is_conserved(self):
        state, _tokens, _cost = _entered()
        product = state.virtual_sol_reserves * state.virtual_token_reserves
        projected = project_curve(state, 7.0)
        assert (projected.virtual_sol_reserves
                * projected.virtual_token_reserves) == pytest.approx(
                    product, rel=1e-6)

    def test_every_lamport_of_the_move_becomes_real_reserves(self):
        """Which is what bounds what a seller can actually be paid."""
        state, _tokens, _cost = _entered()
        projected = project_curve(state, 4.0)
        growth = projected.virtual_sol_reserves - state.virtual_sol_reserves
        assert (projected.real_sol_reserves - state.real_sol_reserves
                == pytest.approx(growth, rel=1e-9))

    def test_a_pool_projects_by_the_same_law(self):
        pool = _pool()
        before = pool.price_quote_per_base
        projected = project_pool(pool, 9.0)
        assert projected.price_quote_per_base / before == pytest.approx(
            9.0, rel=1e-6)


class TestTheCurveCannotPriceTheTail:
    def test_migration_from_a_t0_entry_is_about_fourteen_times(self):
        state, _tokens, _cost = _entered()
        assert migration_multiple(state) == pytest.approx(14.4, abs=0.3)

    def test_projecting_past_migration_is_refused(self):
        """That curve is not a venue any more; the token has left it."""
        state, _tokens, _cost = _entered()
        assert project_curve(state, 10.0) is not None
        assert project_curve(state, 50.0) is None

    def test_without_a_pool_every_rung_above_migration_is_blocked(self):
        state, tokens, cost = _entered()
        rungs = executable_tail_curve(state, tokens, cost)
        above = [rung for rung in rungs if rung.multiple > 20.0]
        assert above
        assert all(rung.status == "DATA_BLOCKED" for rung in above)
        assert all(rung.venue == "migrated_or_unmeasurable" for rung in above)

    def test_the_modelled_migration_pool_is_off_by_default(self):
        """The curve sells 99.98% of its inventory by migration, so a pool
        seeded from the remainder holds less base than one small position.
        That is an artefact of the seeding assumption, not a fact about the
        venue, and no verified seeding figure was available."""
        state, _tokens, _cost = _entered()
        at_migration = pool_at_migration(state)
        assert at_migration.base_reserves < 200_000_000_000
        rungs = executable_tail_curve(state, *_entered()[1:])
        assert not any(rung.venue == "modelled_pool" for rung in rungs)


class TestTheLadderAcrossBothVenues:
    def _rungs(self):
        from src.strategies.continuation import ContinuationModel
        state, tokens, cost = _entered()
        return executable_tail_curve(
            state, tokens, cost, survival=_survival(ContinuationModel()),
            pool=_pool())

    def test_every_rung_prices_when_a_pool_is_observed(self):
        rungs = self._rungs()
        assert all(rung.status == "OK" for rung in rungs)
        assert {rung.venue for rung in rungs} == {"bonding_curve", "pool"}

    def test_exitable_fraction_falls_as_the_multiple_rises(self):
        """The whole point. A position that is 100% exitable at 10x is not
        100% exitable at 1000x, and the difference is the tail it keeps."""
        by_multiple = {rung.multiple: rung.exitable_fraction
                       for rung in self._rungs()}
        assert by_multiple[10.0] == pytest.approx(1.0)
        assert by_multiple[100.0] < by_multiple[20.0]
        assert by_multiple[1000.0] < by_multiple[100.0]

    def test_the_realised_multiple_is_far_below_the_printed_one(self):
        by_multiple = {rung.multiple: rung.net_multiple for rung in self._rungs()}
        assert by_multiple[1000.0] < 400.0

    def test_the_pool_is_projected_from_where_it_already_is(self):
        """Projecting by the whole multiple compounds the move the token has
        already made onto itself: it reported a 20x rung realising 114x,
        because the pool was already 14.4x above entry before the projection
        multiplied it again."""
        by_multiple = {rung.multiple: rung.net_multiple for rung in self._rungs()}
        assert by_multiple[20.0] < 20.0
        assert by_multiple[50.0] < 50.0

    def test_capturable_is_the_product_of_all_three(self):
        rung = next(item for item in self._rungs() if item.multiple == 100.0)
        assert rung.capturable == pytest.approx(
            rung.survival * rung.exitable_fraction * (rung.net_multiple - 1.0))


class TestTheSummaryRefusesToHideItsCoverage:
    def test_coverage_is_reported_beside_the_total(self):
        """A capturable upside summed over three rungs of ten is not a small
        answer, it is a mostly unanswered one."""
        state, tokens, cost = _entered()
        summary = capturable_upside(executable_tail_curve(state, tokens, cost))
        assert summary["rungs"] == len(DEFAULT_TAIL_LADDER)
        assert summary["executable_rungs"] <= 3
        assert summary["status"] == "DATA_BLOCKED"

    def test_no_position_prices_nothing(self):
        state, _tokens, _cost = _entered()
        rungs = executable_tail_curve(state, 0, 0)
        assert all(rung.status == "DATA_BLOCKED" for rung in rungs)

    def test_a_missing_curve_blocks_every_rung(self):
        rungs = executable_tail_curve(None, 1000, 1000)
        assert all(rung.status == "DATA_BLOCKED" for rung in rungs)
        assert capturable_upside(rungs)["capturable_upside"] is None

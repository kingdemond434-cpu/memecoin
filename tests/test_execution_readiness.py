"""Nine execution reads that were built and never called.

The exit template for this mint, its staged ladder, the impact bound a size
fits inside, what the competition paid at this launch age, which feed is
carrying, who leads the slot we would land in, whether the challenger route
has earned a fill, and whether a trained exit policy exists at all. Every one
answers a question the desk asks implicitly at entry and then guesses at.

Two are promoted to decisional here, and the tests say which is which --
because a readiness snapshot that blurs "this changed the trade" with "this
was written down" teaches the forward ledger nothing about either.
"""

from types import SimpleNamespace

import pytest

from src.runtime.execution_readiness import (
    BID_FLOOR_FRACTION, bid_floor, exit_readiness_for, landing_context,
    pre_trade_readiness, route_choice, size_impact)


def _desk(**kwargs):
    base = {"exit_readiness": None, "staged_exits": None, "feed_race": None,
            "leader_schedule": None, "observed_bids": None,
            "raptor_shadow": None, "_latest_curve_state": {},
            "_latest_slot": 0}
    base.update(kwargs)
    return SimpleNamespace(**base)


class TestTheBidFloorIsDecisional:
    def test_a_bid_under_the_observed_market_is_raised(self):
        desk = _desk(observed_bids=SimpleNamespace(
            reference_bid=lambda age, fraction=0.90: 100_000.0))
        assert bid_floor(desk, 5.0, 1_000) == int(100_000 * BID_FLOOR_FRACTION)

    def test_a_bid_above_the_market_is_left_alone(self):
        desk = _desk(observed_bids=SimpleNamespace(
            reference_bid=lambda age, fraction=0.90: 100_000.0))
        assert bid_floor(desk, 5.0, 500_000) == 500_000

    def test_the_floor_sits_under_the_market_not_on_it(self):
        """Everything in that corpus LANDED, so it is biased upward: matching
        it exactly overpays on every launch nobody else wanted."""
        assert BID_FLOOR_FRACTION < 1.0

    def test_no_corpus_means_no_floor_rather_than_a_guess(self):
        assert bid_floor(_desk(), 5.0, 1_000) == 1_000

    def test_a_corpus_that_raises_does_not_break_the_trade(self):
        def boom(age, fraction=0.90):
            raise RuntimeError("corpus unavailable")
        desk = _desk(observed_bids=SimpleNamespace(reference_bid=boom))
        assert bid_floor(desk, 5.0, 1_000) == 1_000


class TestTheRouteChoiceIsDecisional:
    def test_an_unpromoted_challenger_does_not_carry_the_fill(self):
        desk = _desk(raptor_shadow=SimpleNamespace(
            should_route_through_challenger=lambda: False))
        assert route_choice(desk)["route"] == "incumbent"

    def test_a_promoted_challenger_does(self):
        desk = _desk(raptor_shadow=SimpleNamespace(
            should_route_through_challenger=lambda: True))
        assert route_choice(desk)["route"] == "challenger"

    def test_no_shadow_at_all_routes_to_the_incumbent(self):
        result = route_choice(_desk())
        assert result["route"] == "incumbent"
        assert result["status"] == "DATA_BLOCKED"


class TestTheRecordedReads:
    def test_an_unprepared_exit_is_reported_as_unprepared(self):
        readiness = exit_readiness_for(_desk(), "mint")
        assert readiness["template_ready"] is False
        assert readiness["ladder_ready"] is False

    def test_a_prepared_exit_reports_its_rungs(self):
        desk = _desk(
            exit_readiness=SimpleNamespace(template_for=lambda t, now=None: object()),
            staged_exits=SimpleNamespace(
                ladder_for=lambda t: SimpleNamespace(rungs=[1, 2, 3])))
        readiness = exit_readiness_for(desk, "mint")
        assert readiness["template_ready"] is True
        assert readiness["ladder_rungs"] == 3

    def test_a_size_beyond_every_measured_bound_is_blocked_not_zero(self):
        """None is not "no impact". It is "worse than anything measured",
        which should make a position smaller and instead made it invisible."""
        from src.chains.pump_curve import BondingCurveState
        from src.research.dataset_builder import (
            PUMP_INITIAL_VIRTUAL_SOL, PUMP_INITIAL_VIRTUAL_TOKEN,
            PUMP_TOKEN_TOTAL_SUPPLY)
        state = BondingCurveState(
            virtual_token_reserves=PUMP_INITIAL_VIRTUAL_TOKEN,
            virtual_sol_reserves=PUMP_INITIAL_VIRTUAL_SOL,
            real_token_reserves=793_100_000_000_000,
            real_sol_reserves=1_000_000_000,
            token_total_supply=PUMP_TOKEN_TOTAL_SUPPLY, complete=False)
        desk = _desk(_latest_curve_state={"mint": state})
        small = size_impact(desk, "mint", 1_000_000)
        enormous = size_impact(desk, "mint", 10 ** 15)
        assert small["status"] == "OK"
        assert small["impact_bound"] is not None
        assert enormous["impact_bound"] is None
        assert "size down" in enormous["detail"]

    def test_no_curve_state_blocks_rather_than_reporting_zero_impact(self):
        assert size_impact(_desk(), "mint", 1_000)["status"] == "DATA_BLOCKED"

    def test_no_size_is_not_a_measurement(self):
        assert size_impact(_desk(), "mint", 0)["status"] == "DATA_BLOCKED"

    def test_landing_context_reads_all_three_sources(self):
        desk = _desk(
            feed_race=SimpleNamespace(best_feed=lambda: "yellowstone"),
            leader_schedule=SimpleNamespace(
                node_for=lambda slot: SimpleNamespace(identity="val1")),
            observed_bids=SimpleNamespace(
                reference_bid=lambda age, fraction=0.90: 42_000.0),
            _latest_slot=99)
        context = landing_context(desk, 5.0)
        assert context["best_feed"] == "yellowstone"
        assert context["leader_identity"] == "val1"
        assert context["reference_bid_lamports"] == 42_000.0

    def test_an_unknown_slot_does_not_invent_a_leader(self):
        context = landing_context(_desk(), 5.0)
        assert context["leader_known"] is False
        assert context["leader_identity"] is None


class TestTheSnapshotSeparatesActedFromRecorded:
    def test_the_two_halves_are_named_apart(self):
        snapshot = pre_trade_readiness(_desk(), "mint", 1_000, 5.0)
        assert set(snapshot["decisional"]) == {"route"}
        assert set(snapshot["recorded"]) == {"exit", "size", "landing"}

    def test_the_entry_path_takes_the_snapshot_and_the_floor(self):
        """The wiring, not the reads. Eight of these worked perfectly and
        were called by nothing."""
        import ast
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1] / "src" / "main.py"
                  ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        called = {node.func.id for node in ast.walk(tree)
                  if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name)}
        assert {"bid_floor", "pre_trade_readiness"} <= called

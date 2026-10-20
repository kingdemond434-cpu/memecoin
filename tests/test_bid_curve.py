"""The conditioned landing curve and the marginal-lamport bid.

The failures worth testing are the ones that produce a confident wrong bid: a
non-monotone empirical curve that makes underbidding look optimal, an expiry
folded into the bid term so high bids look like failures, and a paper attempt
pooled into the curve that prices real money.
"""

import pytest

from src.execution.bid_curve import (
    BLOCKHASH_VALIDITY_SLOTS, ConditionedLandingCurve, DEFAULT_BID_RUNGS,
    isotonic)
from src.execution.landing_model import Attempt


def _attempt(bid, landed, *, leader="", contention=None, congestion=None,
             age=None, real=True, failure=""):
    return Attempt(bid_lamports=int(bid), landed=bool(landed), leader=leader,
                   account_contention=contention, congestion=congestion,
                   blockhash_age_slots=age, real=real, failure=failure)


def _fill(curve, bid, landed_count, total, **kwargs):
    for index in range(total):
        curve.record(_attempt(bid, index < landed_count, **kwargs))


# --- monotonicity --------------------------------------------------------

def test_isotonic_leaves_an_already_monotone_curve_alone():
    points = [(0, 0.1, 100), (100, 0.3, 100), (200, 0.6, 100)]
    assert [round(value, 6) for _, value, _ in isotonic(points)] == [0.1, 0.3, 0.6]


def test_isotonic_pools_a_violating_pair_into_their_weighted_mean():
    points = [(0, 0.1, 100), (100, 0.8, 100), (200, 0.4, 100)]
    values = [round(value, 6) for _, value, _ in isotonic(points)]
    assert values == [0.1, 0.6, 0.6]   # (0.8 + 0.4) / 2


def test_isotonic_respects_weight():
    # The 0.9 came from four attempts; the 0.2 from four hundred. The pooled
    # value must sit next to the evidence, not halfway between the numbers.
    points = [(0, 0.05, 400), (100, 0.9, 4), (200, 0.2, 400)]
    values = [value for _, value, _ in isotonic(points)]
    assert values[1] == pytest.approx(values[2])
    assert values[1] < 0.3


def test_isotonic_ignores_zero_weight_rungs():
    assert isotonic([(0, 0.5, 0), (100, 0.6, 10)]) == [(100, 0.6, 10)]


def test_the_curve_never_lands_less_often_for_more_lamports():
    curve = ConditionedLandingCurve(min_cell=10)
    # Empirically non-monotone: the 25k bucket got lucky.
    _fill(curve, 10_000, 10, 100)
    _fill(curve, 25_000, 90, 100)
    _fill(curve, 100_000, 40, 100)
    rendered = [value for _, value, _ in curve.monotone_curve()]
    assert rendered == sorted(rendered), "an optimiser would pick the bump"


def test_the_optimiser_would_have_underbid_on_the_raw_curve():
    curve = ConditionedLandingCurve(min_cell=10)
    _fill(curve, 10_000, 10, 100)
    _fill(curve, 25_000, 90, 100)     # the lucky bucket
    _fill(curve, 100_000, 95, 100)
    decision = curve.optimise(edge_lamports=1_000_000)
    assert decision.status == "OK"
    # Without the monotone projection the 25k rung looks nearly as good as
    # 100k for a quarter of the price and wins outright.
    assert decision.bid_lamports >= 25_000


# --- conditioning and its backoff ---------------------------------------

def test_the_most_specific_populated_cell_answers():
    curve = ConditionedLandingCurve(min_cell=10)
    _fill(curve, 100_000, 5, 50, leader="valA", contention=8)
    _fill(curve, 100_000, 45, 50, leader="valB", contention=8)
    a = curve.estimate(100_000, leader="valA", account_contention=8)
    b = curve.estimate(100_000, leader="valB", account_contention=8)
    assert a.basis == b.basis == "leader_contention"
    assert a.probability < b.probability


def test_an_unseen_leader_falls_back_and_says_so():
    curve = ConditionedLandingCurve(min_cell=10)
    _fill(curve, 100_000, 25, 50, leader="valA", contention=8)
    estimate = curve.estimate(100_000, leader="never-seen",
                              account_contention=8)
    assert estimate.status == "OK"
    assert estimate.basis == "contention"


def test_it_falls_all_the_way_to_pooled_before_refusing():
    curve = ConditionedLandingCurve(min_cell=10)
    _fill(curve, 100_000, 25, 50)     # no leader, no contention
    estimate = curve.estimate(100_000, leader="x", account_contention=3)
    assert estimate.basis in ("congestion", "pooled")
    assert estimate.probability == pytest.approx(0.5)


def test_a_thin_cell_is_data_blocked_and_names_what_it_had():
    curve = ConditionedLandingCurve(min_cell=100)
    _fill(curve, 100_000, 2, 5)
    estimate = curve.estimate(100_000)
    assert estimate.status == "DATA_BLOCKED"
    assert estimate.probability is None
    assert "the most populated had 5" in estimate.detail


def test_contention_is_the_term_that_decides_a_launch_snipe():
    curve = ConditionedLandingCurve(min_cell=10)
    _fill(curve, 100_000, 48, 50, contention=0)    # nobody else writing
    _fill(curve, 100_000, 3, 50, contention=32)    # a hot launch
    quiet = curve.estimate(100_000, account_contention=0)
    hot = curve.estimate(100_000, account_contention=32)
    assert quiet.probability > 0.8
    assert hot.probability < 0.2


# --- the population boundary --------------------------------------------

def test_paper_attempts_are_refused_from_the_curve_that_prices_real_bids():
    curve = ConditionedLandingCurve(min_cell=10)
    for _ in range(500):
        assert curve.record(_attempt(100_000, True, real=False)) is False
    assert curve.observed == 0
    assert curve.skipped_population == 500
    assert curve.estimate(100_000).status == "DATA_BLOCKED"


def test_a_mixed_curve_is_possible_but_must_be_asked_for():
    curve = ConditionedLandingCurve(min_cell=10, real_only=False)
    _fill(curve, 100_000, 40, 50, real=False)
    estimate = curve.estimate(100_000)
    assert estimate.status == "OK"
    assert estimate.population == "mixed"


def test_real_and_paper_do_not_contaminate_each_other():
    curve = ConditionedLandingCurve(min_cell=10)
    _fill(curve, 100_000, 50, 50, real=False)   # paper: everything lands
    _fill(curve, 100_000, 5, 50, real=True)     # real: almost nothing does
    assert curve.estimate(100_000).probability == pytest.approx(0.1)


# --- blockhash expiry ----------------------------------------------------

def test_expiry_is_a_separate_multiplier_not_a_bid_bucket():
    curve = ConditionedLandingCurve(min_cell=10)
    _fill(curve, 100_000, 40, 50)
    fresh = curve.estimate(100_000, blockhash_age_slots=0)
    stale = curve.estimate(100_000, blockhash_age_slots=120)
    # The race probability is identical; only the survival term differs.
    assert fresh.probability == stale.probability
    assert fresh.landed_probability > stale.landed_probability


def test_a_hash_past_the_validity_window_cannot_land_at_any_price():
    curve = ConditionedLandingCurve(min_cell=10)
    _fill(curve, 5_000_000, 50, 50)
    estimate = curve.estimate(5_000_000,
                              blockhash_age_slots=BLOCKHASH_VALIDITY_SLOTS)
    assert estimate.landed_probability == 0.0


def test_an_unmeasured_age_does_not_invent_a_survival_term():
    curve = ConditionedLandingCurve(min_cell=10)
    _fill(curve, 100_000, 40, 50)
    estimate = curve.estimate(100_000, blockhash_age_slots=None)
    assert estimate.expiry_survival is None
    assert estimate.landed_probability == estimate.probability


def test_the_expiry_curve_is_fitted_once_there_is_evidence():
    curve = ConditionedLandingCurve(min_cell=10)
    # Sixty attempts at age ~20, none of which expired.
    for index in range(60):
        curve.record(_attempt(100_000, True, age=20))
    survival, basis = curve.expiry.survival(20)
    assert basis == "fitted"
    assert survival == pytest.approx(1.0)
    # The published-window stand-in would have said 1 - 20/150 = 0.867.
    assert survival > 0.9


def test_expired_failures_are_read_out_of_the_failure_text():
    curve = ConditionedLandingCurve(min_cell=10)
    for index in range(60):
        expired = index < 30
        curve.record(_attempt(100_000, not expired, age=100,
                              failure="Blockhash expired" if expired else ""))
    survival, basis = curve.expiry.survival(100)
    assert basis == "fitted"
    assert survival == pytest.approx(0.5)


# --- the marginal-lamport decision --------------------------------------

def test_the_bid_stops_where_a_lamport_stops_buying_a_lamport():
    curve = ConditionedLandingCurve(min_cell=10)
    # Landing saturates by 100k: paying more buys almost nothing.
    _fill(curve, 10_000, 10, 100)
    _fill(curve, 50_000, 60, 100)
    _fill(curve, 100_000, 90, 100)
    _fill(curve, 200_000, 91, 100)
    _fill(curve, 400_000, 92, 100)
    decision = curve.optimise(edge_lamports=1_000_000)
    assert decision.status == "OK"
    assert decision.bid_lamports == 100_000
    assert decision.marginal_at_choice is None or decision.marginal_at_choice >= 1.0
    beyond = [rung for rung in decision.rungs if rung.bid_lamports > 100_000]
    assert all(rung.marginal is not None and rung.marginal < 1.0
               for rung in beyond)


def test_a_bigger_edge_justifies_a_bigger_bid():
    curve = ConditionedLandingCurve(min_cell=10)
    _fill(curve, 10_000, 10, 100)
    _fill(curve, 50_000, 60, 100)
    _fill(curve, 100_000, 90, 100)
    small = curve.optimise(edge_lamports=80_000)
    large = curve.optimise(edge_lamports=5_000_000)
    assert large.bid_lamports >= small.bid_lamports


def test_an_edge_smaller_than_the_cheapest_useful_bid_is_rejected():
    curve = ConditionedLandingCurve(min_cell=10)
    _fill(curve, 100_000, 90, 100)
    _fill(curve, 400_000, 95, 100)
    decision = curve.optimise(edge_lamports=1_000)
    assert decision.status == "REJECTED"
    assert decision.bid_lamports == 0
    assert "does not cover" in decision.detail


def test_no_edge_is_rejected_before_the_curve_is_consulted():
    decision = ConditionedLandingCurve().optimise(edge_lamports=0)
    assert decision.status == "REJECTED"
    assert decision.rungs == []


def test_an_unanswerable_curve_is_data_blocked_not_a_guess():
    decision = ConditionedLandingCurve(min_cell=100).optimise(
        edge_lamports=1_000_000)
    assert decision.status == "DATA_BLOCKED"
    assert decision.bid_lamports == 0
    assert "fall back to its configured ladder" in decision.detail


def test_a_stale_blockhash_lowers_the_bid_it_is_worth_paying():
    curve = ConditionedLandingCurve(min_cell=10)
    _fill(curve, 10_000, 10, 100)
    _fill(curve, 50_000, 60, 100)
    _fill(curve, 100_000, 95, 100)
    fresh = curve.optimise(edge_lamports=300_000, blockhash_age_slots=0)
    stale = curve.optimise(edge_lamports=300_000, blockhash_age_slots=140)
    assert fresh.status == "OK"
    # Paying to win a race you will lose to the clock is a pure transfer.
    assert stale.bid_lamports <= fresh.bid_lamports


def test_the_max_bid_is_respected():
    curve = ConditionedLandingCurve(min_cell=10)
    for rung in DEFAULT_BID_RUNGS[1:]:
        _fill(curve, rung, 99, 100)
    decision = curve.optimise(edge_lamports=100_000_000,
                              max_bid_lamports=50_000)
    assert decision.bid_lamports <= 50_000


def test_the_decision_carries_the_walk_for_an_operator_to_read():
    curve = ConditionedLandingCurve(min_cell=10)
    _fill(curve, 10_000, 20, 100)
    _fill(curve, 50_000, 70, 100)
    _fill(curve, 100_000, 90, 100)
    decision = curve.optimise(edge_lamports=1_000_000)
    assert len(decision.rungs) >= 3
    assert decision.rungs[0].marginal is None    # nothing before the first
    assert all(rung.basis for rung in decision.rungs)


def test_the_report_says_which_population_it_refused():
    curve = ConditionedLandingCurve(min_cell=10)
    _fill(curve, 100_000, 5, 10, real=True)
    _fill(curve, 100_000, 5, 10, real=False)
    report = curve.report()
    assert report["observed"] == 10
    assert report["skipped_wrong_population"] == 10
    assert report["real_only"] is True

"""The canonical event record and the per-cell edge.

The failures worth testing are the ones that produce a confident number from
nothing: a rate off four samples, a return that forgot the latency cost, a
lead measured against a clock that ran backwards, an edge that exists in one
regime being reported as an edge.
"""

import math

import pytest

from src.research.insider_event import (
    ACHIEVABLE_LATENCIES_S, AchievableEntry, ClaimType, EdgeKey, InsiderEvent,
    LawfulAccess, Mechanism, SourceEdgeLedger, from_source_post,
    wilson_interval)


def _event(index, *, ret=0.5, latency_mult=1.02, cost=0.02, rugged=False,
           access=LawfulAccess.PUBLIC, regime="hot", mentions=0,
           size=1.0, feasible=True, observed=None, source_at=None,
           horizon=30.0, multiple=None, source="chan-a"):
    stamp = 1_756_000_000.0 + index * 600.0
    event = InsiderEvent(
        event_id=f"ev-{index}", source_id=source,
        source_at=source_at if source_at is not None else stamp,
        observed_at=observed if observed is not None else stamp + 4.0,
        lawful_access=access, token="MINT", mint="MINT",
        claim_type=ClaimType.CALL, mechanism=Mechanism.FLOW_PREDICTION,
        regime=regime, cost_fraction=cost, concurrent_mentions=mentions,
        rugged=rugged,
        max_feasible_multiple=multiple if multiple is not None else 1.0 + ret,
        price_at_observation=1.0, mfe=ret, mae=-0.1)
    event.returns = {horizon: ret}
    for value in ACHIEVABLE_LATENCIES_S:
        event.achievable[value] = AchievableEntry(
            latency_s=value,
            price_multiple=latency_mult ** (value / 1.0),
            executable_sol=size, feasible=feasible)
    return event


def _ledger(events, **kwargs):
    ledger = SourceEdgeLedger(**kwargs)
    ledger.extend(events)
    return ledger


def _key(regime="hot", horizon=30.0, source="chan-a"):
    return EdgeKey(source_id=source, mechanism=Mechanism.FLOW_PREDICTION,
                   claim_type=ClaimType.CALL, regime=regime,
                   horizon_s=horizon)


# --- provenance ----------------------------------------------------------

def test_prohibited_material_is_refused_with_the_reason_recorded():
    ledger = SourceEdgeLedger()
    ok, reason = ledger.add(_event(0, access=LawfulAccess.PROHIBITED))
    assert not ok
    assert "inadmissible provenance PROHIBITED" in reason
    assert ledger.events == []
    assert ledger.refusals[0]["source_id"] == "chan-a"


def test_unlabelled_provenance_fails_closed():
    ledger = SourceEdgeLedger()
    ok, reason = ledger.add(_event(0, access=LawfulAccess.UNKNOWN))
    assert not ok
    assert "UNKNOWN" in reason


def test_the_three_admissible_classes_are_accepted():
    ledger = SourceEdgeLedger()
    for access in (LawfulAccess.PUBLIC, LawfulAccess.AUTHORIZED,
                   LawfulAccess.VOLUNTEERED):
        assert ledger.add(_event(0, access=access))[0], access


def test_refusals_are_counted_per_source_rather_than_silently_dropped():
    ledger = SourceEdgeLedger()
    for index in range(5):
        ledger.add(_event(index, access=LawfulAccess.PROHIBITED,
                          source="leaky"))
    assert ledger.refusals_by_source()["leaky"] == 5
    assert ledger.summary()["refused"] == 5


def test_an_event_observed_before_it_was_published_is_refused():
    ledger = SourceEdgeLedger()
    ok, reason = ledger.add(_event(0, source_at=1_000.0, observed=900.0))
    assert not ok
    assert "negative observation lag" in reason


# --- the record ----------------------------------------------------------

def test_an_unmeasured_cost_yields_no_net_return_rather_than_a_free_trade():
    event = _event(0, ret=0.5, cost=None)
    assert event.returns[30.0] == 0.5
    assert event.net_return(30.0, 1.0) is None
    assert event.log_growth(30.0, 1.0) is None


def test_latency_is_charged_against_the_gross_move():
    fast = _event(0, ret=1.0, latency_mult=1.0, cost=0.0)
    slow = _event(1, ret=1.0, latency_mult=1.5, cost=0.0)
    assert fast.net_return(30.0, 1.0) == pytest.approx(1.0)
    # Entering 1.5x worse turns a double into a third of a double.
    assert slow.net_return(30.0, 1.0) == pytest.approx(2.0 / 1.5 - 1.0)


def test_an_infeasible_entry_has_no_return_at_that_latency():
    event = _event(0, feasible=False)
    assert event.net_return(30.0, 1.0) is None
    assert not event.executable(1.0)


def test_log_growth_floors_a_total_loss_instead_of_returning_minus_infinity():
    event = _event(0, ret=-1.0, latency_mult=1.0, cost=0.0)
    growth = event.log_growth(30.0, 1.0)
    assert growth is not None and math.isfinite(growth)
    assert growth == pytest.approx(-4.0)


def test_observation_lag_is_publication_to_us_not_the_other_way():
    event = _event(0, source_at=1_000.0, observed=1_007.0)
    assert event.observation_lag == 7.0


# --- sample-size discipline ---------------------------------------------

def test_a_thin_cell_is_data_blocked_rather_than_scored():
    ledger = _ledger([_event(index) for index in range(4)])
    edge = ledger.edges()[_key()]
    assert edge.status == "DATA_BLOCKED"
    assert edge.n == 4
    assert edge.ev_net is None and edge.p_rug is None
    assert "noise wearing a decimal point" in edge.detail


def test_a_full_cell_is_scored():
    ledger = _ledger([_event(index) for index in range(30)])
    edge = ledger.edges()[_key()]
    assert edge.status == "OK"
    assert edge.n == 30
    assert edge.ev_net is not None and edge.e_log_w is not None
    assert edge.e_log_w_ci is not None
    assert edge.first_seen_lag_p50 == pytest.approx(4.0)


def test_wilson_keeps_a_zero_count_honest():
    low, high = wilson_interval(0, 9)
    assert low == 0.0
    assert high > 0.25   # NOT "this source never rugs"
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_a_rug_rate_carries_its_interval():
    events = [_event(index, rugged=index < 3) for index in range(30)]
    edge = _ledger(events).edges()[_key()]
    assert edge.p_rug == pytest.approx(0.1)
    low, high = edge.p_rug_ci
    assert low < 0.1 < high


# --- the point of splitting the cell ------------------------------------

def test_the_same_source_scores_separately_per_regime():
    events = ([_event(index, ret=1.0, regime="hot") for index in range(20)]
              + [_event(50 + index, ret=-0.4, regime="calm")
                 for index in range(20)])
    edges = _ledger(events).edges()
    hot = edges[_key(regime="hot")]
    calm = edges[_key(regime="calm")]
    assert hot.status == calm.status == "OK"
    assert hot.ev_net > 0 > calm.ev_net
    # Averaged into one number this source looks mediocre; split, it is a
    # regime bet, which is a different instrument.


def test_regime_stability_refuses_to_generalise_from_one_tape():
    ledger = _ledger([_event(index, regime="hot") for index in range(20)])
    result = ledger.regime_stability("chan-a", Mechanism.FLOW_PREDICTION,
                                     ClaimType.CALL, 30.0)
    assert result["status"] == "DATA_BLOCKED"
    assert "one tape" in result["reason"]


def test_regime_stability_reports_the_spread_when_two_tapes_exist():
    events = ([_event(index, ret=1.0, regime="hot") for index in range(20)]
              + [_event(50 + index, ret=-0.4, regime="calm")
                 for index in range(20)])
    result = _ledger(events).regime_stability(
        "chan-a", Mechanism.FLOW_PREDICTION, ClaimType.CALL, 30.0)
    assert result["status"] == "OK"
    assert result["spread"] > 0
    assert result["positive_in_all"] is False


def test_the_same_source_scores_separately_per_horizon():
    events = []
    for index in range(20):
        event = _event(index, ret=0.8, horizon=30.0)
        event.returns[1800.0] = -0.6   # great for thirty seconds, ruinous held
        events.append(event)
    edges = _ledger(events).edges()
    assert edges[_key(horizon=30.0)].ev_net > 0
    assert edges[_key(horizon=1800.0)].ev_net < 0


# --- the derived metrics -------------------------------------------------

def test_latency_decay_is_reported_per_achievable_latency():
    events = [_event(index, ret=2.0, latency_mult=1.05, cost=0.0)
              for index in range(20)]
    edge = _ledger(events).edges()[_key()]
    curve = edge.latency_decay
    assert set(curve) == set(ACHIEVABLE_LATENCIES_S)
    assert curve[0.1] > curve[30.0]      # slower entry, worse fill
    values = [curve[value] for value in sorted(curve)]
    assert values == sorted(values, reverse=True)


def test_an_edge_that_only_survives_small_size_reports_that_capacity():
    events = []
    for index in range(24):
        big = index >= 12
        events.append(_event(index, ret=-0.5 if big else 1.0,
                             size=20.0 if big else 0.5))
    edge = _ledger(events).edges()[_key()]
    assert edge.capacity_sol is not None
    assert edge.capacity_sol < 5.0


def test_crowding_decay_is_negative_when_company_kills_the_edge():
    events = ([_event(index, ret=1.0, mentions=0) for index in range(12)]
              + [_event(50 + index, ret=-0.3, mentions=8)
                 for index in range(12)])
    edge = _ledger(events).edges()[_key()]
    assert edge.crowding_decay is not None and edge.crowding_decay < 0


def test_crowding_decay_is_unknown_without_both_conditions_observed():
    edge = _ledger([_event(index, mentions=0)
                    for index in range(20)]).edges()[_key()]
    assert edge.crowding_decay is None


def test_copyability_counts_only_the_entries_that_were_actually_feasible():
    events = [_event(index, feasible=index % 2 == 0) for index in range(20)]
    edge = _ledger(events).edges()[_key()]
    assert edge.p_executable == pytest.approx(0.5)
    assert edge.copyability == edge.p_executable


def test_a_distributor_is_detected_from_the_wallet_shape():
    events = []
    for index in range(20):
        event = _event(index)
        event.pre_post_accumulation_usd = 5_000.0
        event.post_sell_usd = 4_000.0 if index < 16 else 0.0
        events.append(event)
    edge = _ledger(events).edges()[_key()]
    assert edge.distribution_probability == pytest.approx(0.8)


def test_distribution_is_unknown_when_the_wallets_were_never_observed():
    edge = _ledger([_event(index) for index in range(20)]).edges()[_key()]
    assert edge.distribution_probability is None


def test_an_edge_that_stopped_working_is_marked_decayed():
    events = ([_event(index, ret=1.5) for index in range(12)]
              + [_event(50 + index, ret=-0.4) for index in range(12)])
    edge = _ledger(events).edges()[_key()]
    assert edge.decay_status == "DECAYED"


def test_a_steady_edge_is_not_marked_decayed():
    edge = _ledger([_event(index, ret=0.8)
                    for index in range(24)]).edges()[_key()]
    assert edge.decay_status in {"STABLE", "STRENGTHENING"}


# --- the summary and the adapter ----------------------------------------

def test_summary_reports_blocked_cells_rather_than_hiding_them():
    events = ([_event(index, regime="hot") for index in range(20)]
              + [_event(50 + index, regime="calm") for index in range(3)])
    summary = _ledger(events).summary()
    assert summary["cells"] == 2
    assert summary["cells_scored"] == 1
    assert summary["cells_data_blocked"] == 1


class _Post:
    source_id = "tg:alpha"
    token = "MINT"
    posted_at = 1_000.0
    observed_at = 1_003.0
    named_wallets = ["W1", "W2"]


class _Outcome:
    executable_returns = {30.0: 0.4, 300.0: None}
    rugged = False
    max_feasible_multiple = 1.9
    pre_post_accumulation_usd = 100.0
    post_sell_usd = 90.0
    flow_acceleration = 2.5


def test_the_adapter_carries_facts_across_without_inventing_any():
    event = from_source_post(_Post(), _Outcome(), event_id="e1",
                             mechanism=Mechanism.FLOW_PREDICTION,
                             claim_type=ClaimType.CALL, regime="hot")
    assert event.source_id == "tg:alpha"
    assert event.observation_lag == 3.0
    assert event.linked_wallets == ["W1", "W2"]
    assert event.returns[30.0] == 0.4
    assert event.returns[300.0] is None
    assert event.flow_prediction_correct is True
    # The old records never carried a cost or an achievable entry, so the new
    # one does not pretend to either.
    assert event.cost_fraction is None
    assert event.achievable == {}
    assert event.net_return(30.0, 1.0) is None


def test_the_adapter_defaults_to_public_but_records_it_explicitly():
    event = from_source_post(_Post(), _Outcome(), event_id="e1")
    assert event.lawful_access is LawfulAccess.PUBLIC
    assert event.admissible

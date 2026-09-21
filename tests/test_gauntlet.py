"""The gauntlet, and the six ways a mechanism looks like an edge and is not.

Each test constructs a mechanism that would pass a naive check -- positive mean
return, plenty of observations -- and fails for one specific reason. If any of
these starts passing, the gauntlet has stopped being a gauntlet.
"""

import math
import random

import pytest

from src.research.gauntlet import (
    COST_MULTIPLIERS, Gauntlet, LATENCY_GRID_S, MechanismScoreboard,
    Observation, Verdict, bootstrap_lower_bound, max_drawdown,
    probability_of_backtest_overfitting)


def _obs(index, *, mechanism="m", ret=0.10, regime=None, family=None,
         cost=0.01, decay=None, latency_decay=0.0, control=False,
         base_time=1_756_000_000.0):
    """One observation with the same return at every latency by default.

    `latency_decay` subtracts that much per log-step of delay, which is how a
    mechanism that dies on slippage is expressed.
    """
    regimes = ("hot", "calm", "risk_off", "euphoric")
    families = ("tg-ko", "tg-en", "x-kol", "chain")
    returns = {}
    for position, latency in enumerate(LATENCY_GRID_S):
        value = ret - latency_decay * position
        if decay is not None and index >= decay:
            value = -abs(ret)
        returns[latency] = value
    return Observation(
        mechanism=mechanism, timestamp=base_time + index * 600.0,
        regime=regime if regime is not None else regimes[index % 4],
        source_family=family if family is not None else families[index % 4],
        net_return_by_latency=returns, cost_fraction=cost, is_control=control)


def _gauntlet(**kwargs):
    kwargs.setdefault("min_observations", 50)
    kwargs.setdefault("bootstrap", 400)
    return Gauntlet(**kwargs)


# --- the statistics ------------------------------------------------------

def test_the_lower_bound_is_below_the_mean_and_deterministic():
    values = [0.1] * 40 + [-0.3] * 10
    mean = sum(values) / len(values)
    first = bootstrap_lower_bound(values, iterations=500)
    second = bootstrap_lower_bound(values, iterations=500)
    assert first == second, "a verdict must not be a die roll"
    assert first < mean


def test_the_lower_bound_is_negative_when_the_mean_is_barely_positive():
    """The whole reason the gauntlet uses a bound and not a point estimate."""
    rng = random.Random(7)
    values = [rng.gauss(0.01, 0.9) for _ in range(60)]
    assert sum(values) / len(values) > -0.2
    bound = bootstrap_lower_bound(values, iterations=800)
    assert bound is not None and bound < sum(values) / len(values)


def test_the_lower_bound_needs_two_observations():
    assert bootstrap_lower_bound([]) is None
    assert bootstrap_lower_bound([0.5]) is None


def test_drawdown_is_a_path_property():
    assert max_drawdown([1.0, -2.0, 1.0]) == pytest.approx(-2.0)
    assert max_drawdown([1.0, 1.0, 1.0]) == 0.0
    # Same multiset, different order, different drawdown -- which is why the
    # gauntlet computes it on the realised order.
    assert max_drawdown([-2.0, 1.0, 1.0]) == pytest.approx(-2.0)
    assert max_drawdown([1.0, 1.0, -2.0]) == pytest.approx(-2.0)


def test_pbo_is_high_when_the_winner_is_noise():
    rng = random.Random(11)
    matrix = [[rng.gauss(0, 1) for _ in range(200)] for _ in range(12)]
    pbo = probability_of_backtest_overfitting(matrix)
    assert pbo is not None
    assert pbo > 0.25, "pure noise should not look like a discovered edge"


def test_pbo_is_low_when_one_candidate_is_genuinely_better():
    rng = random.Random(13)
    matrix = [[rng.gauss(0, 0.2) for _ in range(200)] for _ in range(11)]
    matrix.append([rng.gauss(1.0, 0.2) for _ in range(200)])
    pbo = probability_of_backtest_overfitting(matrix)
    assert pbo is not None and pbo < 0.1


def test_pbo_refuses_a_sample_too_small_to_diagnose():
    assert probability_of_backtest_overfitting([[1.0] * 100]) is None
    assert probability_of_backtest_overfitting([[1.0] * 3, [2.0] * 3]) is None


# --- the observation record ---------------------------------------------

def test_an_infeasible_latency_has_no_return_rather_than_zero():
    observation = Observation(mechanism="m", timestamp=0.0,
                              net_return_by_latency={1.0: 0.2})
    assert observation.net_return(1.0) == pytest.approx(0.2)
    assert observation.net_return(0.1) is None
    assert observation.log_growth(0.1) is None


def test_the_cost_multiplier_charges_the_extra_units_only():
    observation = Observation(mechanism="m", timestamp=0.0, cost_fraction=0.05,
                              net_return_by_latency={1.0: 0.10})
    assert observation.net_return(1.0, cost_multiplier=1.0) == pytest.approx(0.10)
    assert observation.net_return(1.0, cost_multiplier=2.0) == pytest.approx(0.05)


def test_a_total_loss_is_floored_rather_than_infinite():
    observation = Observation(mechanism="m", timestamp=0.0,
                              net_return_by_latency={1.0: -1.0})
    value = observation.log_growth(1.0)
    assert value is not None and math.isfinite(value)


# --- the gates -----------------------------------------------------------

def test_a_thin_record_is_data_blocked_however_good_it_looks():
    rows = [_obs(index, ret=3.0) for index in range(20)]
    result = _gauntlet().evaluate("m", rows)
    assert result.verdict is Verdict.DATA_BLOCKED
    assert "theatre at this sample size" in result.reasons[0]


def test_a_marginal_edge_is_killed_by_the_lower_bound():
    rng = random.Random(5)
    rows = []
    for index in range(300):
        rows.append(_obs(index, ret=rng.gauss(0.01, 0.8)))
    result = _gauntlet().evaluate("m", rows)
    assert result.e_log_w_lower is not None
    assert result.e_log_w_lower < 0
    assert result.verdict is Verdict.KILL
    assert any("lower bound" in reason for reason in result.reasons)


def test_a_strong_consistent_edge_survives():
    rows = [_obs(index, ret=0.25, cost=0.01) for index in range(300)]
    result = _gauntlet().evaluate("m", rows)
    assert result.verdict is Verdict.SURVIVOR, result.reasons
    assert result.e_log_w_lower > 0
    assert result.latency_survival_s == LATENCY_GRID_S[-1]


def test_an_edge_that_dies_on_slippage_is_fragile_not_a_survivor():
    rows = [_obs(index, ret=0.30, latency_decay=0.10) for index in range(300)]
    result = _gauntlet().evaluate("m", rows)
    assert result.latency_survival_s is not None
    assert result.latency_survival_s < 3.0
    assert result.verdict is Verdict.FRAGILE
    assert any("no headroom" in reason for reason in result.reasons)


def test_an_edge_with_no_cost_headroom_is_fragile_not_a_survivor():
    """It clears its measured cost and dies at 1.5x.

    FRAGILE rather than KILL on purpose: it does work at the cost actually
    observed. But a cost estimate is itself uncertain, and an edge with no
    headroom is one bad week of contention from negative -- which is a thing
    to size for, not a thing to ignore.
    """
    rows = [_obs(index, ret=0.004, cost=0.05) for index in range(300)]
    result = _gauntlet().evaluate("m", rows)
    assert result.verdict is Verdict.FRAGILE
    assert result.cost_survival_multiple == 1.0
    assert any("1.5x execution cost" in reason for reason in result.reasons)


def test_an_edge_that_never_clears_its_own_cost_is_killed():
    rows = [_obs(index, ret=-0.02, cost=0.05) for index in range(300)]
    result = _gauntlet().evaluate("m", rows)
    assert result.verdict is Verdict.KILL
    assert result.cost_survival_multiple is None


def test_an_edge_carried_by_one_source_family_is_killed():
    """Positive overall, negative the moment one family is removed."""
    rows = []
    for index in range(300):
        if index % 4 == 0:
            rows.append(_obs(index, ret=3.0, family="tg-ko"))
        else:
            rows.append(_obs(index, ret=-0.10, family=f"other-{index % 3}"))
    result = _gauntlet().evaluate("m", rows)
    assert result.family_loo_min is not None and result.family_loo_min < 0
    assert result.verdict is Verdict.KILL
    assert any("that family, not the mechanism" in reason
               for reason in result.reasons)


def test_an_edge_that_stopped_working_is_killed():
    rows = [_obs(index, ret=0.4, decay=150) for index in range(300)]
    result = _gauntlet().evaluate("m", rows)
    assert result.decay_status == "DECAYED"
    assert result.verdict is Verdict.KILL


def test_a_single_regime_edge_is_fragile():
    rows = [_obs(index, ret=0.25, regime="euphoric") for index in range(300)]
    result = _gauntlet().evaluate("m", rows)
    assert len(result.regimes) == 1
    assert result.verdict is Verdict.FRAGILE
    assert any("regime" in reason for reason in result.reasons)


def test_a_selected_champion_is_killed_by_pbo():
    rows = [_obs(index, ret=0.25) for index in range(300)]
    result = _gauntlet().evaluate("m", rows, pbo=0.8)
    assert result.verdict is Verdict.KILL
    assert any("selected, not discovered" in reason
               for reason in result.reasons)


def test_a_declared_control_is_carried_rather_than_judged():
    rows = [_obs(index, ret=-0.05, control=True) for index in range(300)]
    result = _gauntlet().evaluate("random_pump_launch", rows)
    assert result.verdict is Verdict.CONTROL


def test_a_mechanism_whose_entries_were_never_feasible_is_data_blocked():
    rows = [Observation(mechanism="m", timestamp=float(index),
                        net_return_by_latency={}) for index in range(300)]
    result = _gauntlet().evaluate("m", rows)
    assert result.verdict is Verdict.DATA_BLOCKED
    assert "never feasible" in result.reasons[0]


# --- the scoreboard ------------------------------------------------------

def _corpus():
    rows = []
    rows += [_obs(index, mechanism="funder_first25", ret=0.25)
             for index in range(300)]
    rows += [_obs(index, mechanism="famous_caller_copy", ret=-0.08)
             for index in range(300)]
    rows += [_obs(index, mechanism="random_pump_launch", ret=-0.05,
                  control=True) for index in range(300)]
    return rows


def test_the_scoreboard_says_in_one_word_whether_there_is_an_edge():
    report = MechanismScoreboard(_gauntlet()).report(_corpus())
    assert report["has_edge"] is True
    assert report["survivors"] == 1
    top = report["rows"][0]
    assert top["mechanism"] == "funder_first25"
    assert top["verdict"] == "SURVIVOR"
    assert "+ through" in top["delay_survival"]


def test_the_scoreboard_says_so_when_nothing_survives():
    rows = [_obs(index, mechanism="famous_caller_copy", ret=-0.08)
            for index in range(300)]
    report = MechanismScoreboard(_gauntlet()).report(rows)
    assert report["has_edge"] is False
    assert "machinery, not an edge" in report["detail"]


def test_pbo_is_computed_across_candidates_and_excludes_the_control():
    gauntlet = _gauntlet()
    results = gauntlet.run(_corpus())
    pbos = {result.pbo for result in results.values()}
    assert len(pbos) == 1, "PBO is a property of the selection, not the row"
    assert results["random_pump_launch"].verdict is Verdict.CONTROL


def test_the_rendered_table_is_readable_over_ssh():
    board = MechanismScoreboard(_gauntlet())
    text = board.render(board.build(_corpus()))
    assert "mechanism" in text and "verdict" in text
    assert "SURVIVOR" in text and "CONTROL" in text
    widths = {len(line) for line in text.splitlines()[:2]}
    assert len(widths) == 1, "header and rule must line up"

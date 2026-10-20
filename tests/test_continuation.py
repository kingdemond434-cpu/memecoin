"""The conditional continuation probability, and what it refuses to say.

`continuation` used to be `max(p_5x, p_10x)` -- unconditional survival from
launch. At 8x that asks whether the position will reach 5x, which it settled
an hour ago, so the number that decides whether a runner is held through a
drawdown carried no information exactly where the account is made. These tests
pin the conditional replacement and, more importantly, every case where it
declines to answer: a ratio of two uncalibrated scores, or of two numbers
built from a handful of examples, has the shape of a conviction and none of
the content.
"""

import math
from types import SimpleNamespace

import pytest

from src.strategies.continuation import (
    DEFAULT_MIN_POSITIVES, ContinuationModel, position_multiple,
    read_position_continuation)
from src.strategies.exit_policy import (
    NEVER_SUPPRESSED, ExitPolicy, evaluate_exit)
from src.strategies.multihead_predictor import PredictionTarget, SURVIVAL_LEVELS


class FakePredictor:
    """A predictor that answers the two questions the model actually asks."""

    def __init__(self, *, calibrated=None, positives=200, trained=True):
        self._is_trained = trained
        self._calibrated = (set(PredictionTarget) if calibrated is None
                            else set(calibrated))
        self._positives = positives

    def is_calibrated(self, target):
        return self._is_trained and target in self._calibrated

    def head_positives(self, target):
        if isinstance(self._positives, dict):
            return self._positives.get(target)
        return self._positives


def power_law(alpha=1.23, k=0.0493):
    """A prediction whose survival curve is exactly k * m ** -alpha."""
    values = {target.value: min(1.0, k * level ** -alpha)
              for target, level in SURVIVAL_LEVELS}
    return SimpleNamespace(**values)


def flat(**overrides):
    values = {target.value: 0.5 for target, _ in SURVIVAL_LEVELS}
    values.update(overrides)
    return SimpleNamespace(**values)


# --- the question it now answers ------------------------------------------

def test_it_asks_about_doubling_from_here_not_about_reaching_five():
    """At 8x, p_5x is settled history. The conditional is not."""
    model = ContinuationModel()
    prediction = power_law()
    reading = model.evaluate(FakePredictor(), prediction, 8.0)
    assert reading.ok
    assert reading.target_multiple == 16.0
    assert reading.probability != pytest.approx(prediction.p_5x)


def test_the_reading_means_the_same_thing_at_every_multiple():
    """A pure power law has a doubling probability independent of where you are."""
    model = ContinuationModel()
    prediction = power_law(alpha=1.23)
    readings = [model.evaluate(FakePredictor(), prediction, m).probability
                for m in (2.5, 6.0, 12.0, 40.0)]
    for value in readings[1:]:
        assert value == pytest.approx(readings[0], rel=1e-6)
    # And it equals the analytic 2 ** -alpha.
    assert readings[0] == pytest.approx(2.0 ** -1.23, rel=1e-6)


def test_the_curve_is_anchored_at_one_so_early_positions_are_answerable():
    """Without the anchor there is no segment covering a position at 1.4x."""
    model = ContinuationModel()
    reading = model.evaluate(FakePredictor(), power_law(), 1.4)
    assert reading.ok
    assert reading.survival_from is not None


def test_interpolation_is_log_linear_not_straight_line():
    """A straight line between 10x and 20x overstates the middle badly."""
    model = ContinuationModel()
    curve = model.curve(FakePredictor(), power_law())
    survival, _ = model.survival_at(curve, 14.0)
    analytic = 0.0493 * 14.0 ** -1.23
    assert survival == pytest.approx(analytic, rel=1e-6)
    straight = None
    for (low_x, low_y), (high_x, high_y) in zip(curve, curve[1:]):
        if low_x <= 14.0 <= high_x:
            weight = (14.0 - low_x) / (high_x - low_x)
            straight = (1 - weight) * low_y + weight * high_y
    assert straight > survival, "the naive interpolation is the flattering one"


# --- what it refuses ------------------------------------------------------

def test_an_untrained_predictor_gets_no_reading():
    model = ContinuationModel()
    reading = model.evaluate(FakePredictor(trained=False), power_law(), 3.0)
    assert reading.status == "DATA_BLOCKED"
    assert not reading.calibrated


def test_uncalibrated_heads_are_not_probabilities():
    """A ratio of two gradient-boosting scores is not a conditional probability."""
    model = ContinuationModel()
    reading = model.evaluate(FakePredictor(calibrated=()), power_law(), 3.0)
    assert reading.status == "DATA_BLOCKED"
    assert "calibrated" in reading.detail


def test_a_head_below_the_positive_floor_does_not_join_the_curve():
    model = ContinuationModel()
    thin = FakePredictor(positives=DEFAULT_MIN_POSITIVES - 1)
    reading = model.evaluate(thin, power_law(), 3.0)
    assert reading.status == "DATA_BLOCKED"


def test_a_bundle_that_never_recorded_positives_certifies_nothing():
    """Unknown is not enough; it is exactly the state a stale bundle is in."""
    model = ContinuationModel()
    reading = model.evaluate(FakePredictor(positives=None), power_law(), 3.0)
    assert reading.status == "DATA_BLOCKED"


def test_the_curve_stops_at_the_first_unusable_rung_rather_than_bridging():
    """The rungs are nested, so a gap cannot be spanned by interpolation."""
    model = ContinuationModel()
    predictor = FakePredictor(positives={
        PredictionTarget.P_2X: 900, PredictionTarget.P_5X: 400,
        PredictionTarget.P_10X: 5,          # too thin
        PredictionTarget.P_20X: 900,        # plenty, but unreachable past 10x
        PredictionTarget.P_50X: 900,
    })
    curve = model.curve(predictor, power_law())
    assert [point[0] for point in curve] == [1.0, 2.0, 5.0]


def test_it_will_not_extrapolate_past_the_last_measured_rung():
    model = ContinuationModel()
    predictor = FakePredictor(positives={
        PredictionTarget.P_2X: 900, PredictionTarget.P_5X: 400})
    reading = model.evaluate(predictor, power_law(), 4.0)
    assert reading.status == "DATA_BLOCKED"
    assert "beyond the last measured rung" in reading.detail


def test_a_denominator_that_rounds_to_nothing_is_not_a_conviction():
    """S(100)/S(50) with both at 1e-9 reads 0.9 and means nothing."""
    model = ContinuationModel(min_conditioning_survival=1e-3)
    prediction = flat(p_2x=0.4, p_5x=0.2, p_10x=1e-5, p_20x=9e-6,
                      p_50x=8e-6, p_100x=7e-6, p_250x=6e-6, p_500x=5e-6)
    reading = model.evaluate(FakePredictor(), prediction, 12.0)
    assert reading.status == "DATA_BLOCKED"
    assert "conditioning floor" in reading.detail


def test_a_non_monotone_curve_cannot_produce_a_probability_above_one():
    """The predictor enforces monotonicity; this does not take its word."""
    model = ContinuationModel()
    prediction = flat(p_2x=0.10, p_5x=0.30, p_10x=0.60, p_20x=0.90,
                      p_50x=0.95, p_100x=0.99, p_250x=0.99, p_500x=0.99)
    reading = model.evaluate(FakePredictor(), prediction, 3.0)
    assert reading.ok
    assert 0.0 <= reading.probability <= 1.0


def test_a_position_at_zero_is_refused_rather_than_divided_by():
    model = ContinuationModel()
    assert model.evaluate(FakePredictor(), power_law(), 0.0).status == "DATA_BLOCKED"


def test_no_prediction_is_data_blocked():
    model = ContinuationModel()
    assert model.evaluate(FakePredictor(), None, 3.0).status == "DATA_BLOCKED"


# --- the position-shaped entry point --------------------------------------

def test_the_multiple_comes_from_whichever_field_carries_it():
    assert position_multiple({"current_multiple": 3.5}) == 3.5
    assert position_multiple(
        {"entry_price": 2.0, "current_price": 9.0}) == 4.5
    assert position_multiple({}) == 0.0
    assert position_multiple({"current_multiple": "junk"}) == 0.0


def test_the_reading_uses_the_refreshed_prediction_not_the_entry_one():
    model = ContinuationModel()
    position = {"current_multiple": 3.0, "prediction_object": power_law()}
    reading = read_position_continuation(model, FakePredictor(), position)
    assert reading.ok


def test_a_caller_with_no_configured_model_still_grants_nothing():
    """The default is safe because every gate in it is a refusal."""
    bare = SimpleNamespace(_is_trained=True)
    position = {"current_multiple": 3.0, "prediction_object": power_law()}
    reading = read_position_continuation(None, bare, position)
    assert reading.status == "DATA_BLOCKED"
    assert not reading.calibrated


# --- the hold time --------------------------------------------------------

def test_conviction_extends_only_the_time_stop():
    policy = ExitPolicy(max_hold_seconds=3600.0,
                        max_conviction_hold_seconds=259_200.0)
    # Ordinary: out on the clock at an hour.
    assert evaluate_exit(policy, 11.0, 11.0, 0.9, {"cost_recovery", "bank_5x",
                                                   "bank_10x"}, 3601.0) == (
        "time_stop", 1.0)
    # Under conviction: still running at eleven hours.
    assert evaluate_exit(policy, 11.0, 11.0, 0.9,
                         {"cost_recovery", "bank_5x", "bank_10x"},
                         39_600.0, conviction=True) is None


def test_the_conviction_ceiling_is_a_ceiling():
    policy = ExitPolicy(max_hold_seconds=3600.0,
                        max_conviction_hold_seconds=259_200.0)
    assert evaluate_exit(policy, 40.0, 40.0, 0.99,
                         {"cost_recovery", "bank_5x", "bank_10x"},
                         259_201.0, conviction=True) == ("time_stop", 1.0)


def test_conviction_never_touches_the_hard_stop():
    policy = ExitPolicy(max_hold_seconds=3600.0,
                        max_conviction_hold_seconds=259_200.0)
    assert evaluate_exit(policy, 0.4, 8.0, 0.99, set(), 60.0,
                         conviction=True) == ("hard_stop_loss", 1.0)


def test_zero_ceiling_leaves_the_ordinary_hold_exactly_as_it_was():
    """A desk with no continuation model gets today's behaviour, unchanged."""
    policy = ExitPolicy(max_hold_seconds=3600.0)
    assert evaluate_exit(policy, 11.0, 11.0, 0.9,
                         {"cost_recovery", "bank_5x", "bank_10x"},
                         3601.0, conviction=True) == ("time_stop", 1.0)


def test_the_stops_conviction_may_never_stand_down():
    assert "hard_stop_loss" in NEVER_SUPPRESSED
    assert "time_stop" in NEVER_SUPPRESSED
    assert "adaptive_profit_trailing_stop" not in NEVER_SUPPRESSED


def test_the_config_carries_both_hold_times():
    import yaml
    config = yaml.safe_load(open("config/chains.yaml"))["global"]
    assert config["max_hold_time_minutes"] > 60, "60 closed a runner at 11x"
    assert (config["max_conviction_hold_time_minutes"]
            > config["max_hold_time_minutes"])


# --- which ceiling is actually capping the book ---------------------------

def _engine(equity, *, max_position_usd=500.0, fraction=0.01):
    from src.strategies.multihead_predictor import ElogwEngine
    engine = ElogwEngine.__new__(ElogwEngine)
    engine.portfolio_value = float(equity)
    engine.max_position_usd = float(max_position_usd)
    engine.max_position_pct = 0.05
    engine.max_liquidity_fraction = float(fraction)
    engine.small_account_mode = False
    engine._ceiling_counts = {}
    engine._depth_bound_fractions = []
    return engine


def test_depth_binds_a_large_account_on_an_early_curve():
    """$100k of equity against a $5k pool: 1% of the pool is $50."""
    engine = _engine(100_000.0)
    name, cap = engine.binding_ceiling(5_000.0)
    assert name == engine.CEILING_POOL_DEPTH
    assert cap * engine.portfolio_value == pytest.approx(50.0)


def test_depth_does_not_bind_a_small_account_on_the_same_curve():
    """The same $5k pool is no constraint at all on $1k of equity."""
    engine = _engine(1_000.0, max_position_usd=50.0)
    name, _ = engine.binding_ceiling(5_000.0)
    assert name != engine.CEILING_POOL_DEPTH


def test_the_report_names_the_wall_when_depth_starts_binding_everything():
    engine = _engine(100_000.0)
    for _ in range(10):
        engine.exposure_cap(5_000.0)
    report = engine.capacity_report()
    assert report["depth_bound_share"] == 1.0
    assert "more capital will not raise the growth rate" in report["detail"]


def test_the_report_is_data_blocked_before_anything_is_sized():
    assert _engine(1_000.0).capacity_report()["status"] == "DATA_BLOCKED"


def test_the_depth_sample_is_bounded():
    engine = _engine(100_000.0)
    for _ in range(engine.CEILING_SAMPLE + 250):
        engine.exposure_cap(5_000.0)
    assert len(engine._depth_bound_fractions) == engine.CEILING_SAMPLE


def test_the_cap_is_still_the_smallest_ceiling():
    """The accounting must not change what the sizing actually does."""
    engine = _engine(100_000.0)
    for pool in (5_000.0, 50_000.0, 5_000_000.0):
        ceilings = engine.exposure_ceilings(pool)
        assert engine.exposure_cap(pool) == pytest.approx(min(ceilings.values()))


# --- the two paths this whole change exists for ---------------------------

def _runner_path():
    """A 100x over six hours, with a 25% shakeout partway up."""
    path = []
    for step in range(0, 6 * 3600, 60):
        frac = step / (6 * 3600)
        multiple = 1.0 + 99.0 * (frac ** 1.6)
        if 0.42 < frac < 0.52:
            multiple *= 0.75
        path.append((float(step), multiple))
    return path


def _round_trip_path():
    """The same 100x peak, then a collapse back to single digits."""
    path = []
    for step in range(0, 12 * 3600, 60):
        frac = step / (12 * 3600)
        if frac < 0.5:
            multiple = 1.0 + 99.0 * math.sin(min(1.0, frac * 2) * math.pi / 2) ** 2
        else:
            multiple = 100.0 * math.exp(-6 * (frac - 0.5))
        path.append((float(step), max(0.5, multiple)))
    return path


def _replay(path, policy, conviction):
    """Walk a price path through the real policy, as the desk walks it."""
    model = ContinuationModel()
    prediction = power_law()
    predictor = FakePredictor()
    stages, high_water = set(), 1.0
    for elapsed, multiple in path:
        high_water = max(high_water, multiple)
        reading = model.evaluate(predictor, prediction, multiple)
        decision = evaluate_exit(
            policy, multiple, high_water, float(reading.probability or 0.0),
            stages, elapsed, conviction=conviction)
        if decision and conviction and decision[0] not in NEVER_SUPPRESSED:
            decision = None
        if decision:
            if decision[1] >= 1.0:
                return decision[0], multiple
            stages.add({"profit_ratchet_cost_recovery": "cost_recovery",
                        "profit_ratchet_5x": "bank_5x",
                        "profit_ratchet_10x": "bank_10x"}[decision[0]])
    return "held_to_end", path[-1][1]


OLD_POLICY = ExitPolicy(max_hold_seconds=3600.0)
NEW_POLICY = ExitPolicy(max_hold_seconds=240 * 60.0,
                        max_conviction_hold_seconds=4320 * 60.0)


def test_one_hour_closed_a_hundred_x_at_under_seven():
    """The behaviour this replaces, pinned so the regression is visible."""
    reason, multiple = _replay(_runner_path(), OLD_POLICY, conviction=False)
    assert reason == "time_stop"
    assert multiple < 7.0


def test_under_conviction_the_same_runner_reaches_its_peak():
    reason, multiple = _replay(_runner_path(), NEW_POLICY, conviction=True)
    assert multiple > 95.0, f"exited at {multiple:.1f}x on {reason}"


def test_a_round_trip_is_banked_near_the_peak_not_ridden_to_the_floor():
    """Conviction widens the trail; it does not remove it.

    With the trail merely suppressed -- which is what the monster override
    did on its own -- this path exited at 5.0x from a 100x peak, because the
    ratchet is checked BEFORE the trail, returned every cycle, was nulled by
    the caller, and `stages_done` therefore never advanced past it. The trail
    was unreachable for the life of the position.
    """
    reason, multiple = _replay(_round_trip_path(), NEW_POLICY, conviction=True)
    assert reason == "conviction_trailing_stop"
    assert multiple > 40.0, f"gave back to {multiple:.1f}x"


def test_the_ratchet_is_skipped_inside_the_policy_not_outside_it():
    """Under conviction the trail must be REACHABLE, which means no early return."""
    policy = ExitPolicy(max_conviction_hold_seconds=259_200.0)
    # 12x, high water 100x: the 10x ratchet would fire first if it were still
    # checked, and the conviction trail (floor 45x) would never be reached.
    assert evaluate_exit(policy, 12.0, 100.0, 0.5, set(), 600.0,
                         conviction=True) == ("conviction_trailing_stop", 1.0)
    # Without conviction the ratchet is exactly as it was.
    assert evaluate_exit(policy, 12.0, 100.0, 0.5, set(), 600.0)[0] == (
        "profit_ratchet_cost_recovery")


# --- how far up the curve the model can see -------------------------------

def test_the_report_says_what_the_model_can_actually_answer_about():
    """The question after a training run is not 'did it train'.

    A corpus with 94 tenx launches and 28 twentyx ones trains every head, and
    the curve still stops at 10x -- so conviction is DATA_BLOCKED above 5x and
    never engages on the runners it exists for.
    """
    predictor = FakePredictor(positives={
        PredictionTarget.P_2X: 683, PredictionTarget.P_5X: 210,
        PredictionTarget.P_10X: 94, PredictionTarget.P_20X: 28,
        PredictionTarget.P_50X: 9})
    report = ContinuationModel(extrapolate_tail=False).report(predictor)
    assert report["usable_rungs"] == [2.0, 5.0, 10.0]
    assert report["answerable_up_to_multiple"] == 5.0
    # And the model agrees with its own report.
    blind = ContinuationModel(extrapolate_tail=False)
    assert blind.evaluate(predictor, power_law(), 4.0).ok
    assert blind.evaluate(predictor, power_law(), 8.0).status == "DATA_BLOCKED"
    # The fitted tail is what lifts that ceiling, and says it did.
    seeing = ContinuationModel().evaluate(predictor, power_law(), 8.0)
    assert seeing.ok and seeing.extrapolated


def test_the_report_names_the_reason_when_nothing_is_usable():
    report = ContinuationModel().report(FakePredictor(calibrated=()))
    assert report["status"] == "DATA_BLOCKED"
    assert "every runner exits on the ordinary trail" in report["detail"]


def test_the_report_lists_every_rung_with_its_own_evidence():
    report = ContinuationModel().report(FakePredictor(positives={
        PredictionTarget.P_2X: 683, PredictionTarget.P_5X: 4}))
    by_head = {row["head"]: row for row in report["rungs"]}
    assert by_head["p_2x"]["usable"] and by_head["p_2x"]["positives"] == 683
    assert not by_head["p_5x"]["usable"]
    assert by_head["p_10x"]["positives"] is None


def test_an_untrained_predictor_reports_data_blocked():
    assert ContinuationModel().report(
        FakePredictor(trained=False))["status"] == "DATA_BLOCKED"


# --- the fitted tail: seeing past the last rung the corpus supports -------

def real_corpus_positives(episodes=32_542, alpha=1.23, k=0.0493):
    """What a corpus of this size actually supports, rung by rung.

    At 32,542 episodes: 683 positives at 2x, 221 at 5x, 94 at 10x, 40 at 20x,
    13 at 50x, 5 at 100x. The last two fall below the positive floor, so the
    curve ends at 20x and the conditional continuation -- which asks about
    2m -- goes DATA_BLOCKED above 10x. That is where the tail begins.
    """
    return {target: int(episodes * min(1.0, k * level ** -alpha))
            for target, level in SURVIVAL_LEVELS}


def _real_predictor():
    return FakePredictor(positives=real_corpus_positives())


def test_the_corpus_runs_out_exactly_where_the_tail_starts():
    """The problem, stated as a measurement rather than a worry."""
    model = ContinuationModel(extrapolate_tail=False)
    report = model.report(_real_predictor())
    assert report["usable_rungs"] == [2.0, 5.0, 10.0, 20.0]
    assert report["answerable_up_to_multiple"] == 10.0


def test_the_fit_recovers_a_known_exponent():
    model = ContinuationModel()
    curve = [(1.0, 1.0)] + [(m, 0.0493 * m ** -1.23) for m in (2, 5, 10, 20)]
    alpha, residual = model.fit_tail(curve)
    assert alpha == pytest.approx(1.23, abs=1e-6)
    assert residual == pytest.approx(0.0, abs=1e-9)


def test_the_definitional_anchor_is_excluded_from_the_fit():
    """S(1)=1 is not a point on the power law; including it fits 2.02."""
    model = ContinuationModel()
    curve = [(m, 0.0493 * m ** -1.23) for m in (2, 5, 10, 20)]
    assert model.fit_tail(curve)[0] == pytest.approx(1.23, abs=1e-6)


def test_two_rungs_are_not_enough_to_claim_a_line():
    """Two points fit perfectly and cannot disagree with the model."""
    model = ContinuationModel()
    assert model.fit_tail([(1.0, 1.0), (2.0, 0.02), (5.0, 0.006)]) is None


def test_a_curve_that_is_not_a_power_law_is_refused():
    model = ContinuationModel()
    bent = [(1.0, 1.0), (2.0, 0.5), (5.0, 0.49), (10.0, 0.48), (20.0, 1e-6)]
    alpha, residual = model.fit_tail(bent)
    assert residual > model.max_tail_residual
    # And the refusal reaches the reading, not just the fit.
    value, _basis, extrapolated = model.survival(bent, 60.0)
    assert value is None and not extrapolated


def test_a_rising_curve_is_not_a_survival_curve():
    model = ContinuationModel()
    assert model.fit_tail([(2.0, 0.1), (5.0, 0.2), (10.0, 0.4)]) is None


def test_the_extrapolation_matches_the_analytic_tail():
    model = ContinuationModel()
    curve = [(1.0, 1.0)] + [(m, 0.0493 * m ** -1.23) for m in (2, 5, 10, 20)]
    for multiple in (40.0, 100.0, 160.0):
        value, _basis, extrapolated = model.survival(curve, multiple)
        assert extrapolated
        assert value == pytest.approx(0.0493 * multiple ** -1.23, rel=1e-9)


def test_it_will_not_reach_arbitrarily_far_past_the_last_rung():
    """The reach is now FITTED, and capped. Eight times the last rung was a
    constant applied identically to a tail fitted on seven tight rungs and one
    fitted on three scattered ones; those two fits do not deserve the same
    horizon. The constant survives only as the cap, which is what this
    perfect four-rung fit runs into."""
    model = ContinuationModel()
    curve = [(1.0, 1.0)] + [(m, 0.0493 * m ** -1.23) for m in (2, 5, 10, 20)]
    assert model.fitted_reach(curve) == pytest.approx(20.0 * model.tail_reach)
    assert model.survival(curve, 160.0)[0] is not None
    value, basis, _ = model.survival(curve, 200.0)
    assert value is None
    assert "entitled to reach" in basis


def test_a_scattered_fit_earns_a_shorter_reach_than_a_tight_one():
    """The whole point of fitting the reach rather than choosing it.

    Prediction uncertainty for a least-squares line grows with distance from
    the centre of its own data, so a fit whose rungs disagree with the power
    law is entitled to say less about what lies beyond them.
    """
    model = ContinuationModel()
    clean = [(1.0, 1.0)] + [(m, 0.0493 * m ** -1.23) for m in (2, 5, 10, 20)]
    scattered = [(1.0, 1.0), (2.0, 0.35), (5.0, 0.02), (10.0, 0.11), (20.0, 0.004)]
    assert model.fitted_reach(scattered) < model.fitted_reach(clean)


def test_a_fit_too_uncertain_to_extrapolate_is_refused_outright():
    """When the uncertainty exceeds the tolerance even at the centre of the
    data, there is no multiple this fit may speak about."""
    model = ContinuationModel(max_tail_log_se=0.01)
    scattered = [(1.0, 1.0), (2.0, 0.35), (5.0, 0.02), (10.0, 0.11), (20.0, 0.004)]
    assert model.fitted_reach(scattered) is None
    value, basis, extrapolated = model.survival(scattered, 100.0)
    assert value is None and not extrapolated
    assert "may not be extrapolated" in basis


def test_interpolation_inside_the_measured_range_is_untouched():
    model = ContinuationModel()
    curve = [(1.0, 1.0)] + [(m, 0.0493 * m ** -1.23) for m in (2, 5, 10, 20)]
    value, _basis, extrapolated = model.survival(curve, 15.0)
    assert not extrapolated, "a measured reading must never be labelled fitted"
    assert value == pytest.approx(0.0493 * 15.0 ** -1.23, rel=1e-9)


# --- what it unlocks ------------------------------------------------------

def _monster_state(model, multiple):
    """Conviction for a position evaluated FRESH at this multiple."""
    from src.strategies.monster import MonsterEvidence, MonsterStateMachine
    machine = MonsterStateMachine()
    reading = model.evaluate(_real_predictor(), power_law(), multiple)
    machine.update("t", MonsterEvidence(
        monster_probability=(reading.probability if reading.ok else None),
        monster_probability_calibrated=bool(reading.ok and reading.calibrated),
        independent_buyer_acceleration=0.4,
        smart_wallet_net_accumulation=0.3))
    return machine.overrides_ordinary_exit("t")


def test_without_the_tail_conviction_dies_where_the_tail_begins():
    blind = ContinuationModel(extrapolate_tail=False)
    assert _monster_state(blind, 9.0) is True
    for multiple in (12.0, 25.0, 60.0):
        assert _monster_state(blind, multiple) is False, multiple


def test_with_the_tail_conviction_survives_into_the_tail():
    seeing = ContinuationModel()
    for multiple in (9.0, 12.0, 25.0, 60.0):
        assert _monster_state(seeing, multiple) is True, multiple


def test_and_still_refuses_beyond_the_fitted_reach():
    """120x doubles to 240x, which is past 8x the last measured rung."""
    assert _monster_state(ContinuationModel(), 120.0) is False


def test_every_extrapolated_reading_says_so():
    """So the gauntlet can score decisions made on a fit separately from
    decisions made on a measurement."""
    reading = ContinuationModel().evaluate(_real_predictor(), power_law(), 25.0)
    assert reading.ok and reading.extrapolated
    assert reading.tail_alpha == pytest.approx(1.23, abs=0.05)
    assert reading.to_dict()["extrapolated"] is True


def test_a_measured_reading_is_not_labelled_extrapolated():
    reading = ContinuationModel().evaluate(_real_predictor(), power_law(), 4.0)
    assert reading.ok and not reading.extrapolated


def test_the_report_states_how_far_the_fit_reaches():
    report = ContinuationModel().report(_real_predictor(), power_law())
    assert report["highest_usable_multiple"] == 20.0
    assert report["highest_answerable_survival"] == pytest.approx(160.0)
    assert report["answerable_up_to_multiple"] == pytest.approx(80.0)
    assert report["tail_alpha"] == pytest.approx(1.23, abs=0.05)


def test_the_extrapolation_can_be_turned_off_entirely():
    off = ContinuationModel(extrapolate_tail=False)
    assert off.evaluate(_real_predictor(), power_law(), 25.0).status == (
        "DATA_BLOCKED")

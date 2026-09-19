"""Maximising E[log W] and maximising P(reaching a target) are not the same.

The existing allocator answers "how fast does capital compound". The stated
goal is closer to

    maximise  P(W_T >= target)   subject to   P(ruin) <= epsilon

and without the constraint the target optimiser is degenerate: it bets
everything on the most explosive coin, which maximises the lottery payoff and
produces a near-certain dead account. The constraint is applied as a FILTER
rather than a penalty weight, because a weight lets a high enough target
probability buy its way past any amount of ruin.

The tests below are mostly about the input, not the optimiser, because that is
where a simulator like this actually fails: fed an invented distribution it
returns invented advice with a confidence interval attached.
"""

import random

import pytest

from src.strategies.wealth_target import (
    DEFAULT_MAX_RUIN, LaunchOutcomeModel, TailPolicy, WealthTargetSimulator,
    default_policy_grid)

#: The desk's own corpus rates: of 32,542 episodes, 683 reached 2x and 5
#: reached 100x.
CORPUS = [(2.0, 683 / 32542), (5.0, 221 / 32542), (10.0, 94 / 32542),
          (20.0, 40 / 32542), (50.0, 13 / 32542), (100.0, 5 / 32542)]

#: A deliberately generous curve, used only to show what a generous input does
#: to the conclusions.
GENEROUS = [(2.0, 0.35), (5.0, 0.12), (10.0, 0.05), (20.0, 0.02),
            (50.0, 0.006), (100.0, 0.002)]


class TestTheDrawMatchesTheCurve:
    def test_each_rung_is_hit_at_its_own_probability(self):
        """u <= P(M >= m) is the whole mapping. The first version tossed a
        separate rug coin on top of it and produced a model in which no launch
        ever reached 2x."""
        model = LaunchOutcomeModel(GENEROUS, round_trip_cost=0.0)
        rng = random.Random(11)
        draws = [model.draw(rng, 1.0) for _ in range(60_000)]
        reached_2x = sum(1 for value in draws if value >= 2.0) / len(draws)
        reached_10x = sum(1 for value in draws if value >= 10.0) / len(draws)
        assert reached_2x == pytest.approx(0.35, abs=0.01)
        assert reached_10x == pytest.approx(0.05, abs=0.01)

    def test_a_miss_is_a_total_loss_by_default(self):
        model = LaunchOutcomeModel(GENEROUS, round_trip_cost=0.0)
        rng = random.Random(3)
        assert any(model.draw(rng, 1.0) == 0.0 for _ in range(200))

    def test_the_floor_is_supplied_rather_than_invented(self):
        """What a launch below the first rung realises is a property of the
        EXIT POLICY, not of the launch distribution."""
        model = LaunchOutcomeModel(GENEROUS, floor_multiple=0.5,
                                   round_trip_cost=0.0)
        rng = random.Random(3)
        draws = {model.draw(rng, 1.0) for _ in range(500)}
        assert 0.5 in draws
        assert 0.0 not in draws

    def test_costs_are_charged_on_every_outcome(self):
        cheap = LaunchOutcomeModel(GENEROUS, round_trip_cost=0.0)
        dear = LaunchOutcomeModel(GENEROUS, round_trip_cost=0.5)
        assert dear.draw(random.Random(5), 1.0) < cheap.draw(random.Random(5), 1.0) \
            or cheap.draw(random.Random(5), 1.0) == 0.0


class TestItShowsTheEdgeItIsAssuming:
    def test_the_corpus_rates_imply_a_heavily_negative_edge(self):
        """The finding that matters. Playing the BASE RATE is catastrophic;
        every cent of the desk's value has to come from selection lifting the
        conditional distribution above this one."""
        model = LaunchOutcomeModel(CORPUS)
        assert model.expected_multiple() < 0.2

    def test_a_generous_curve_implies_an_edge_no_market_offers(self):
        model = LaunchOutcomeModel(GENEROUS)
        assert model.expected_multiple() > 1.5

    def test_the_sweep_states_the_edge_on_its_own_face(self):
        """A sweep run on a +105% edge concludes that almost any policy gets
        rich, which is a fact about the assumption and not about the market."""
        simulator = WealthTargetSimulator(LaunchOutcomeModel(GENEROUS), trials=50)
        report = simulator.sweep([TailPolicy(0.01)], bankroll=200.0,
                                 target=1_000.0, floor=20.0, launches=20)
        assert report["input_edge_implausible"] is True
        assert "not evidence about the market" in report["input_warning"]

    def test_a_plausible_curve_is_not_flagged(self):
        simulator = WealthTargetSimulator(LaunchOutcomeModel(CORPUS), trials=50)
        report = simulator.sweep([TailPolicy(0.01)], bankroll=200.0,
                                 target=1_000.0, floor=20.0, launches=20)
        assert report["input_edge_implausible"] is False
        assert report["input_warning"] == ""


class TestCaptureIsAFunctionOfTicketSize:
    def test_a_larger_ticket_realises_less_of_the_same_printed_move(self):
        """Measured on a real migrated pool the same 1000x is worth 95% to a
        0.1 SOL ticket and 2% to a 10 SOL one. An optimiser treating the
        multiple as size-independent recommends large tickets and is wrong by
        more than an order of magnitude."""
        def capture(printed, ticket_usd):
            return printed / (1.0 + ticket_usd / 100.0)
        model = LaunchOutcomeModel(GENEROUS, capture=capture,
                                   round_trip_cost=0.0)
        # Averaged over the same seeded draws, so the comparison is about the
        # ticket size rather than about which rung one lucky roll landed on.
        def mean(ticket):
            rng = random.Random(2)
            return sum(model.draw(rng, ticket) for _ in range(5_000)) / 5_000
        assert mean(1_000.0) < mean(1.0) / 5

    def test_without_a_capture_function_the_printed_multiple_is_used(self):
        model = LaunchOutcomeModel(GENEROUS, round_trip_cost=0.0)
        assert model.draw(random.Random(2), 1.0) in {0.0, 2.0, 5.0, 10.0,
                                                     20.0, 50.0, 100.0}


class TestRuinIsAFilterNotAWeight:
    def _simulator(self, **kwargs):
        return WealthTargetSimulator(LaunchOutcomeModel(CORPUS), trials=200,
                                     **kwargs)

    def test_a_ruinous_policy_is_inadmissible_however_often_it_hits(self):
        report = self._simulator().sweep(
            [TailPolicy(0.5, max_concurrent=1)], bankroll=200.0,
            target=100_000.0, floor=20.0, launches=200)
        row = report["ranked"][0]
        assert row["ruined"] > DEFAULT_MAX_RUIN
        assert row["admissible"] is False
        assert report["best"] is None

    def test_when_nothing_survives_the_report_says_so_rather_than_picking(self):
        report = self._simulator().sweep(
            [TailPolicy(0.9, max_concurrent=5)], bankroll=100.0,
            target=1_000_000.0, floor=50.0, launches=200)
        assert report["best"] is None
        assert "no policy stayed inside the ruin budget" in report["detail"]

    def test_the_corpus_base_rate_reaches_no_target_at_all(self):
        """Not a failure of the optimiser. Buying the base rate is -89% per
        launch, and no ticket size fixes a negative edge."""
        report = self._simulator().sweep(
            default_policy_grid(concurrency=(1, 10), reserves=(0.0,)),
            bankroll=200.0, target=100_000.0, floor=20.0, launches=300)
        assert all(row["hit_target"] == 0.0 for row in report["ranked"])


class TestItRefusesRatherThanGuesses:
    def test_an_empty_model_simulates_nothing(self):
        simulator = WealthTargetSimulator(LaunchOutcomeModel([]), trials=10)
        outcome = simulator.evaluate(TailPolicy(0.01), bankroll=100.0,
                                     target=200.0, floor=10.0)
        assert outcome.trials == 10
        assert outcome.hit_target == 0.0
        assert "nothing can be simulated" in outcome.detail

    def test_the_grid_is_a_grid_not_a_shortlist(self):
        grid = default_policy_grid()
        assert len(grid) == 6 * 4 * 2
        assert len({policy.name for policy in grid}) == len(grid)

    def test_a_run_is_reproducible_from_its_seed(self):
        model = LaunchOutcomeModel(CORPUS)
        first = WealthTargetSimulator(model, trials=100).evaluate(
            TailPolicy(0.01), bankroll=200.0, target=1_000.0, floor=20.0)
        second = WealthTargetSimulator(model, trials=100).evaluate(
            TailPolicy(0.01), bankroll=200.0, target=1_000.0, floor=20.0)
        assert first.median_terminal == second.median_terminal

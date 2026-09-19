"""A wallet's PnL is not a reason to copy it, and the gauntlet has to say so.

Every case here is a wallet that looks good by the measure the desk used to
rank on -- the hand-weighted composite over early-entry quality, forward
return, consistency, independence and sample size -- and is uncopyable for a
reason that measure cannot see.

The fixtures construct the uncopyability deliberately, so a passing test is a
statement that the gauntlet found a known defect rather than that it produced
a number.
"""

import math
import random

import pytest

from src.research.benchmark_wallets import WalletDecision
from src.research.copyability import (
    COPYABILITY_SCHEMA_VERSION, CopyabilityBoard, CopyabilityCriteria,
    CopyabilityGauntlet, cumulative_drawdown, profit_factor,
    single_trade_profit_share)

DELAYS = (0.05, 0.10, 0.25, 0.50, 1.00)
MONTH = 30 * 86_400.0
BASE = 1_700_000_000.0


def _decision(wallet, index, *, wallet_multiple, decay_per_delay=0.0,
              month=0, liquidity=50_000.0, rugged=False, buyer_rank=3):
    """One resolved decision with a chosen follower penalty per delay.

    `decay_per_delay` is how much worse the follower's ENTRY price is at each
    successive delay, as a fraction. Zero means following is as good as being
    the wallet; large means the wallet's edge is its speed.
    """
    entry = 1.0
    exit_price = entry * wallet_multiple
    prices = {}
    for step, delay in enumerate(DELAYS):
        prices[delay] = entry * (1.0 + decay_per_delay * step)
    return WalletDecision(
        wallet=wallet, token=f"{wallet}-{index}",
        entered_at=BASE + month * MONTH + index * 3_600.0,
        buyer_rank=buyer_rank, entry_price=entry, price_at_delay=prices,
        exit_price=exit_price, exited_at=BASE + month * MONTH + index * 3_600.0 + 600,
        state={"liquidity_usd": liquidity, "rugged": rugged,
               "regime": "early_curve", "source_family": "chain"})


def _steady_winner(wallet="steady", n=60, seed=7):
    """A wallet that is genuinely followable: modest, repeatable, spread out."""
    rng = random.Random(seed)
    rows = []
    for index in range(n):
        # Mostly small wins, some small losses. Nothing heroic.
        multiple = 1.35 if rng.random() < 0.62 else 0.85
        rows.append(_decision(wallet, index, wallet_multiple=multiple,
                              decay_per_delay=0.002, month=index % 6))
    return rows


class TestTheGauntletFindsKnownDefects:
    def test_a_steady_followable_wallet_survives(self):
        verdict = CopyabilityGauntlet().evaluate("steady", _steady_winner())
        assert verdict.status == "OK"
        assert verdict.copyable, verdict.reasons
        assert verdict.follower_lower_bound > 0

    def test_a_wallet_whose_edge_is_speed_is_not_copyable(self):
        """Profitable at 50ms, gone by 1s. Common, and fatal to a follower."""
        rows = [_decision("fast", i, wallet_multiple=3.0,
                          decay_per_delay=0.30, month=i % 6) for i in range(60)]
        verdict = CopyabilityGauntlet().evaluate("fast", rows)
        assert verdict.edge_is_speed is True
        assert "edge_is_speed_not_information" in verdict.reasons
        assert not verdict.copyable

    def test_one_enormous_trade_is_not_an_edge(self):
        rows = [_decision("lottery", i, wallet_multiple=0.80, month=i % 6)
                for i in range(59)]
        rows.append(_decision("lottery", 59, wallet_multiple=400.0, month=5))
        verdict = CopyabilityGauntlet().evaluate("lottery", rows)
        assert verdict.single_trade_share == pytest.approx(1.0)
        assert "profit_concentrated_in_one_trade" in verdict.reasons
        assert not verdict.copyable

    def test_a_losing_follow_is_killed_not_merely_flagged(self):
        rows = [_decision("bagholder", i, wallet_multiple=0.5, month=i % 6)
                for i in range(60)]
        verdict = CopyabilityGauntlet().evaluate("bagholder", rows)
        assert verdict.verdict == "KILL"
        assert "follower_lower_bound_not_positive" in verdict.reasons

    def test_a_brilliant_fortnight_is_not_a_record(self):
        rows = [_decision("sprinter", i, wallet_multiple=1.4,
                          decay_per_delay=0.002, month=0) for i in range(60)]
        verdict = CopyabilityGauntlet().evaluate("sprinter", rows)
        assert verdict.months == 1
        assert "record_too_short_in_months" in verdict.reasons

    def test_rug_exposure_is_counted_from_the_record(self):
        rows = [_decision("ruggy", i, wallet_multiple=1.4 if i % 2 else 0.9,
                          decay_per_delay=0.002, month=i % 6,
                          rugged=bool(i % 2 == 0)) for i in range(60)]
        verdict = CopyabilityGauntlet().evaluate("ruggy", rows)
        assert verdict.rug_rate == pytest.approx(0.5)
        assert "rug_exposure_beyond_limit" in verdict.reasons


class TestItRefusesRatherThanGuesses:
    def test_too_few_decisions_is_blocked_not_scored(self):
        rows = _steady_winner(n=10)
        verdict = CopyabilityGauntlet().evaluate("thin", rows)
        assert verdict.verdict == "DATA_BLOCKED"
        assert "insufficient_resolved_decisions" in verdict.blocked
        assert verdict.follower_lower_bound is None

    def test_unresolved_decisions_are_not_counted_as_break_even(self):
        rows = _steady_winner(n=60)
        for row in rows[:40]:
            row.exit_price = None
        verdict = CopyabilityGauntlet().evaluate("half", rows)
        assert verdict.resolved == 20
        assert verdict.verdict == "DATA_BLOCKED"

    def test_a_delay_with_no_reachable_fill_is_absent_not_zero(self):
        rows = _steady_winner(n=60)
        for row in rows:
            row.price_at_delay.pop(0.05, None)
        verdict = CopyabilityGauntlet().evaluate("gappy", rows)
        assert verdict.latency_curve[0.05] is None
        assert verdict.latency_curve[1.00] is not None

    def test_a_missing_rug_column_blocks_rather_than_reading_as_clean(self):
        rows = _steady_winner(n=60)
        for row in rows:
            row.state.pop("rugged")
        verdict = CopyabilityGauntlet().evaluate("unknown", rows)
        assert verdict.rug_rate is None
        assert "rug_rate" in verdict.blocked

    def test_a_missing_liquidity_column_blocks(self):
        rows = _steady_winner(n=60)
        for row in rows:
            row.state.pop("liquidity_usd")
        verdict = CopyabilityGauntlet().evaluate("unknown", rows)
        assert verdict.median_entry_liquidity_usd is None
        assert "entry_liquidity" in verdict.blocked


class TestItScoresUsNotThem:
    def test_the_score_is_taken_at_the_desk_latency_not_the_best_one(self):
        """Picking the best of five delays is selection on the data."""
        rows = [_decision("fast", i, wallet_multiple=3.0,
                          decay_per_delay=0.30, month=i % 6) for i in range(60)]
        gauntlet = CopyabilityGauntlet(CopyabilityCriteria(desk_latency_s=1.0))
        verdict = gauntlet.evaluate("fast", rows)
        at_desk = verdict.latency_curve[1.00]
        assert verdict.follower_mean == pytest.approx(at_desk)
        assert verdict.latency_curve[0.05] > at_desk

    def test_the_wallets_own_return_is_carried_but_never_scored(self):
        rows = [_decision("fast", i, wallet_multiple=3.0,
                          decay_per_delay=0.30, month=i % 6) for i in range(60)]
        verdict = CopyabilityGauntlet().evaluate("fast", rows, wallet_own_mean=1.1)
        assert verdict.wallet_own_mean == 1.1
        # The gap is the whole point: the wallet tripled, we did not.
        assert verdict.follower_mean < verdict.wallet_own_mean

    def test_costs_are_charged_and_their_survival_is_measured(self):
        cheap = CopyabilityGauntlet(CopyabilityCriteria(cost_per_round_trip=0.01))
        dear = CopyabilityGauntlet(CopyabilityCriteria(cost_per_round_trip=0.20))
        rows = _steady_winner()
        assert (cheap.evaluate("steady", rows).follower_mean
                > dear.evaluate("steady", rows).follower_mean)
        assert cheap.evaluate("steady", rows).cost_survival_multiple is not None

    def test_a_wallet_that_only_works_at_zero_cost_does_not_survive_them(self):
        rows = [_decision("thin", i, wallet_multiple=1.03,
                          decay_per_delay=0.0, month=i % 6) for i in range(60)]
        gauntlet = CopyabilityGauntlet(
            CopyabilityCriteria(cost_per_round_trip=0.04))
        verdict = gauntlet.evaluate("thin", rows)
        assert verdict.cost_survival_multiple is None
        assert not verdict.copyable


class TestTheArithmetic:
    def test_profit_factor_is_none_when_nothing_ever_lost(self):
        assert profit_factor([0.1, 0.2, 0.3]) is None

    def test_profit_factor_divides_gross_gain_by_gross_loss(self):
        assert profit_factor([2.0, -1.0]) == pytest.approx(2.0)

    def test_single_trade_share_finds_the_concentration(self):
        assert single_trade_profit_share([0.1, 0.1, 9.8]) == pytest.approx(0.98)

    def test_drawdown_is_a_positive_magnitude(self):
        value = cumulative_drawdown([1.0, -2.0, 0.5])
        assert value == pytest.approx(2.0)

    def test_drawdown_is_a_path_property(self):
        """Same returns, different order, different drawdown.

        Two wins then two losses digs a hole twice as deep as the same four
        trades alternating, and a follower lives through the path, not the
        sum. Shuffling the record would destroy exactly this.
        """
        alternating = cumulative_drawdown([1.0, -1.0, 1.0, -1.0])
        clustered = cumulative_drawdown([1.0, 1.0, -1.0, -1.0])
        assert clustered == pytest.approx(2.0)
        assert alternating == pytest.approx(1.0)


class TestTheBoard:
    def test_survivors_outrank_the_blocked_whatever_their_headline(self):
        board = CopyabilityBoard()
        rows = board.build({
            "steady": _steady_winner("steady"),
            "thin": _steady_winner("thin", n=5),
            "bagholder": [_decision("bagholder", i, wallet_multiple=0.5,
                                    month=i % 6) for i in range(60)],
        })
        assert rows[0].wallet == "steady"
        assert rows[-1].wallet == "thin"

    def test_the_report_carries_its_own_selection_warning(self):
        report = CopyabilityBoard().report({"steady": _steady_winner()})
        assert "selected" in report["selection_warning"]
        assert report["schema"] == COPYABILITY_SCHEMA_VERSION

    def test_the_report_states_the_wallet_minus_follower_gap(self):
        rows = [_decision("fast", i, wallet_multiple=3.0,
                          decay_per_delay=0.30, month=i % 6) for i in range(60)]
        report = CopyabilityBoard().report({"fast": rows},
                                           own_means={"fast": math.log(3.0)})
        assert report["median_wallet_minus_follower_log_return"] > 0

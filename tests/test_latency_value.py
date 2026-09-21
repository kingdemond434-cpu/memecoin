"""Whether being fast is worth anything, asked of data already collected.

Everything this desk has done for speed rests on a belief that has never
been priced. The decision the answer informs is concrete: if 250ms costs a
tenth of a percent of entry, most of the fast path is a rounding error and
the effort belongs elsewhere; if it costs eight percent, latency IS the
strategy.

Delaying live decisions to find out is the wrong instrument. It costs money
once the desk is trading, and it cannot run at all before then -- so it
would produce its first reading exactly when it starts being expensive. The
snapshots already carry the curve reserves at every rung, and for a
constant-product curve the price IS the reserve ratio, so the answer is
arithmetic on numbers already recorded.
"""

from __future__ import annotations

import math
import unittest

from src.research.latency_value import (
    DELAY_LADDER_MS, MIN_SAMPLES, RAN_MULTIPLE, LatencyValueLedger,
    marginal_price)


class ThePriceIsArithmeticNotAModel(unittest.TestCase):

    def test_the_curve_prices_at_its_reserve_ratio(self):
        self.assertAlmostEqual(2.0, marginal_price(20.0, 10.0))

    def test_an_empty_curve_has_no_price_rather_than_zero(self):
        for sol, tokens in ((0, 10), (10, 0), (-1, 10), (None, 10), ("x", 1)):
            self.assertIsNone(marginal_price(sol, tokens))


class ACostNeedsABaseline(unittest.TestCase):

    def test_a_launch_with_no_t0_price_contributes_nothing(self):
        ledger = LatencyValueLedger()
        self.assertFalse(ledger.observe({50: 1.0, 100: 1.1}, 5.0))
        self.assertEqual(1, ledger.without_t0)
        self.assertEqual(0, ledger.launches)

    def test_the_protocol_invariant_rows_are_skipped(self):
        # The initialisation constants are identical at every offset, so
        # including them would report a cost of exactly zero for every delay
        # and drown the real measurements in launches never observed at all.
        ledger = LatencyValueLedger()
        invariant = {"status": "OK", "provenance": "INVARIANT",
                     "curve_sol_reserves": 30e9, "curve_token_reserves": 1.073e15}
        self.assertFalse(ledger.observe_snapshots(
            {0.0: invariant, 0.05: invariant, 0.1: invariant}, 5.0))
        self.assertEqual(0, ledger.launches)


class ItSeparatesTheLaunchesThatRan(unittest.TestCase):
    """The cost of being late lives entirely in the launches whose price
    moved. An average over the rest hides exactly the amount that matters."""

    def _ledger(self, ran_drift=0.10, dud_drift=0.0, count=MIN_SAMPLES * 2 + 50):
        ledger = LatencyValueLedger()
        for index in range(count):
            ran = index % 2 == 0
            drift = ran_drift if ran else dud_drift
            prices = {0: 1.0}
            for position, delay in enumerate(DELAY_LADDER_MS, start=1):
                # Price climbing with the delay: the launch is being bought.
                prices[delay] = 1.0 + drift * position / len(DELAY_LADDER_MS)
            ledger.observe(prices, 10.0 if ran else 1.0)
        return ledger

    def test_a_delay_costs_more_on_the_launches_that_ran(self):
        ledger = self._ledger()
        report = ledger.report()
        worst = report["delays"][-1]
        self.assertEqual("OK", worst["launches_that_ran"]["status"])
        self.assertGreater(worst["launches_that_ran"]["median_cost_pct"],
                           worst["all_launches"]["median_cost_pct"])

    def test_the_cost_grows_with_the_delay(self):
        ledger = self._ledger()
        costs = [ledger.cost_of(delay) for delay in DELAY_LADDER_MS]
        self.assertTrue(all(cost is not None for cost in costs))
        # More delay, more harm -- and harm is the NEGATIVE direction, so a
        # correctly signed ladder descends.
        self.assertEqual(costs, sorted(costs, reverse=True))

    def test_the_arithmetic_is_the_log_of_the_price_ratio(self):
        ledger = LatencyValueLedger()
        for _ in range(MIN_SAMPLES):
            ledger.observe({0: 1.0, 100: 1.10}, 5.0)
        self.assertAlmostEqual(-math.log(1.10), ledger.cost_of(100), places=9)

    def test_a_launch_whose_price_fell_is_kept_not_dropped(self):
        # Being slow sometimes HELPS: the price fell in the first hundred
        # milliseconds, so the late entry bought more. Dropping those
        # launches would manufacture the answer this module exists to
        # measure, so they are kept and show as a positive delta.
        ledger = LatencyValueLedger()
        for _ in range(MIN_SAMPLES):
            ledger.observe({0: 1.0, 100: 0.90}, 5.0)
        self.assertGreater(ledger.cost_of(100), 0.0)
        bucket = ledger.report()["delays"][DELAY_LADDER_MS.index(100)]
        # And it reads as a NEGATIVE cost, which is a saving.
        self.assertLess(bucket["launches_that_ran"]["median_cost_pct"], 0.0)


class ItRefusesToAnswerOnTooLittle(unittest.TestCase):

    def test_a_thin_bucket_is_data_blocked_not_a_noisy_number(self):
        # A noisy answer about whether to spend a month on latency is worse
        # than no answer, because it will be acted on.
        ledger = LatencyValueLedger()
        for _ in range(MIN_SAMPLES - 1):
            ledger.observe({0: 1.0, 100: 1.05}, 5.0)
        self.assertIsNone(ledger.cost_of(100))
        bucket = ledger.report()["delays"][DELAY_LADDER_MS.index(100)]
        self.assertEqual("DATA_BLOCKED", bucket["launches_that_ran"]["status"])

    def test_the_verdict_says_data_blocked_before_it_says_anything_else(self):
        ledger = LatencyValueLedger()
        self.assertIn("DATA_BLOCKED", ledger.verdict())

    def test_the_verdict_is_one_actionable_sentence_once_priced(self):
        ledger = LatencyValueLedger()
        for _ in range(MIN_SAMPLES):
            ledger.observe({0: 1.0, 1000: 1.20}, RAN_MULTIPLE * 5)
        verdict = ledger.verdict()
        self.assertIn("1000ms", verdict)
        self.assertIn("%", verdict)
        self.assertNotIn("DATA_BLOCKED", verdict)

    def test_unreadable_rungs_are_counted_not_hidden(self):
        # A bucket that is 90% unreadable is not a measurement, however
        # confident the other 10% looks.
        ledger = LatencyValueLedger()
        for _ in range(10):
            ledger.observe({0: 1.0}, 5.0)
        self.assertEqual(10, ledger.buckets[100].unreadable)


class ItReadsTheSnapshotsTheDeskAlreadyTakes(unittest.TestCase):

    def test_reserves_at_each_rung_become_a_priced_delay(self):
        ledger = LatencyValueLedger()

        def row(sol):
            return {"status": "OK", "curve_sol_reserves": sol,
                    "curve_token_reserves": 1.0e15}

        self.assertTrue(ledger.observe_snapshots(
            {0.0: row(30e9), 0.1: row(33e9), 0.25: row(36e9)}, 8.0))
        self.assertEqual(1, ledger.launches)
        self.assertEqual(1, ledger.launches_that_ran)
        self.assertEqual(1, len(ledger.buckets[100].log_deltas_that_ran))

    def test_a_blocked_row_is_not_a_price(self):
        ledger = LatencyValueLedger()
        self.assertFalse(ledger.observe_snapshots(
            {0.0: {"status": "DATA_BLOCKED"}}, 5.0))


if __name__ == "__main__":
    unittest.main()

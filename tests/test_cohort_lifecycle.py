"""The opening cohort's afterlife, and the actor graph's blind spot."""

from __future__ import annotations

import unittest

from src.strategies.actor_graph import Entry
from src.strategies.cohort_lifecycle import (
    CHASER_SKILL_CEILING, MIN_ABSORBER_INDEPENDENCE, cohort_retention,
    evaluate_cohorts, late_chaser, opening_cohort, post_sniper_absorption)


def _entries(count: int, start: float = 100.0, step: float = 0.2):
    return [Entry(token="T", wallet=f"w{index}", timestamp=start + index * step,
                  skill=0.8, capital_usd=100.0)
            for index in range(count)]


def _bought(wallets, at: float, units: float = 10.0):
    return [{"wallet": wallet, "timestamp": at, "units": units} for wallet in wallets]


class OpeningCohortCountsActorsNotFills(unittest.TestCase):

    def test_one_wallet_buying_eight_times_is_one_member(self):
        entries = [Entry("T", "whale", 100.0 + index * 0.01) for index in range(8)]
        entries += [Entry("T", f"w{index}", 101.0 + index) for index in range(3)]
        cohort = opening_cohort(entries, 10)
        self.assertEqual(["whale", "w0", "w1", "w2"], cohort)

    def test_the_cohort_is_ordered_by_time_not_by_input_order(self):
        entries = [Entry("T", "late", 200.0), Entry("T", "early", 100.0)]
        self.assertEqual(["early", "late"], opening_cohort(entries, 10))


class RetentionRefusesToGuess(unittest.TestCase):

    def test_a_cohort_nobody_measured_is_blocked_not_zero(self):
        reading = cohort_retention(_entries(10), [], 10, as_of=200.0)
        self.assertEqual("DATA_BLOCKED", reading.status)

    def test_thin_coverage_is_refused(self):
        entries = _entries(10)
        # Units for only three of ten members.
        flows = _bought(["w0", "w1", "w2"], at=100.0)
        reading = cohort_retention(entries, flows, 10, as_of=300.0)
        self.assertEqual("DATA_BLOCKED", reading.status)
        self.assertIn("below", reading.detail)

    def test_a_cohort_that_held_everything_reads_as_fully_retained(self):
        entries = _entries(10)
        flows = _bought([f"w{i}" for i in range(10)], at=100.0)
        reading = cohort_retention(entries, flows, 10, as_of=400.0)
        self.assertEqual("OK", reading.status)
        self.assertEqual(1.0, reading.retained[60.0])
        self.assertEqual([], reading.fully_exited)

    def test_selling_shows_up_at_the_mark_it_happened_in(self):
        entries = _entries(10, start=100.0, step=0.0)
        wallets = [f"w{i}" for i in range(10)]
        flows = _bought(wallets, at=100.0)
        # Half the cohort dumps everything at +5s: after the 3s mark, before 10s.
        flows += [{"wallet": wallet, "timestamp": 105.0, "units": -10.0}
                  for wallet in wallets[:5]]
        reading = cohort_retention(entries, flows, 10, as_of=400.0)
        self.assertEqual(1.0, reading.retained[3.0])
        self.assertAlmostEqual(0.5, reading.retained[10.0])
        self.assertEqual(5, len(reading.fully_exited))

    def test_a_mark_that_has_not_happened_yet_is_omitted_not_full(self):
        # The flattering error would be reporting a five-second-old cohort as
        # 100% retained at sixty seconds.
        entries = _entries(10, start=100.0, step=0.0)
        flows = _bought([f"w{i}" for i in range(10)], at=100.0)
        reading = cohort_retention(entries, flows, 10, as_of=105.0)
        self.assertEqual("OK", reading.status)
        self.assertIn(1.0, reading.retained)
        self.assertNotIn(60.0, reading.retained)


class AbsorptionSeparatesDemandFromInventoryShuffling(unittest.TestCase):

    def setUp(self):
        self.entries = _entries(25, start=100.0, step=0.01)
        self.cohort = [f"w{i}" for i in range(25)]
        self.flows = _bought(self.cohort, at=100.0)
        # The cohort distributes 100 units between t=200 and t=260.
        self.flows += [{"wallet": wallet, "timestamp": 210.0, "units": -4.0}
                       for wallet in self.cohort]

    def test_independent_buyers_absorbing_the_supply_reads_as_absorbed(self):
        flows = self.flows + [
            {"wallet": f"buyer{i}", "timestamp": 220.0, "units": 10.0}
            for i in range(9)]
        independence = {f"buyer{i}": 0.9 for i in range(9)}
        reading = post_sniper_absorption(
            self.entries, flows, independence, 25, (200.0, 260.0),
            marks=[(200.0, 1.0), (260.0, 1.05)])
        self.assertEqual("OK", reading.status)
        self.assertEqual("ABSORBED", reading.verdict)
        self.assertGreater(reading.absorption_ratio, 0.85)

    def test_related_buyers_are_capture_not_absorption(self):
        # The same units, bought by wallets NOT independent of the sellers.
        # On-chain this is identical; economically it is one actor moving
        # inventory between its own pockets, and it must not read as demand.
        flows = self.flows + [
            {"wallet": f"buyer{i}", "timestamp": 220.0, "units": 10.0}
            for i in range(9)]
        independence = {f"buyer{i}": 0.1 for i in range(9)}
        reading = post_sniper_absorption(
            self.entries, flows, independence, 25, (200.0, 260.0),
            marks=[(200.0, 1.0), (260.0, 1.05)])
        self.assertEqual("CAPTURED", reading.verdict)
        self.assertEqual(0.0, reading.independent_absorbed_units)
        self.assertGreater(reading.related_absorbed_units, 0)

    def test_an_unscored_buyer_is_not_assumed_independent(self):
        flows = self.flows + [{"wallet": "stranger", "timestamp": 220.0, "units": 90.0}]
        reading = post_sniper_absorption(
            self.entries, flows, {}, 25, (200.0, 260.0))
        self.assertEqual(0.0, reading.independent_absorbed_units)
        self.assertEqual(90.0, reading.related_absorbed_units)

    def test_supply_hitting_a_collapsing_price_is_a_failure(self):
        flows = self.flows + [
            {"wallet": "buyer0", "timestamp": 220.0, "units": 20.0}]
        reading = post_sniper_absorption(
            self.entries, flows, {"buyer0": 0.9}, 25, (200.0, 260.0),
            marks=[(200.0, 1.0), (260.0, 0.4)])
        self.assertEqual("FAILED", reading.verdict)

    def test_a_cohort_that_sold_nothing_has_no_absorption_to_measure(self):
        flows = _bought(self.cohort, at=100.0)
        reading = post_sniper_absorption(
            self.entries, flows, {}, 25, (200.0, 260.0))
        self.assertEqual("DATA_BLOCKED", reading.status)
        self.assertIsNone(reading.absorption_ratio)


class LateChasersAreExitEvidence(unittest.TestCase):

    def test_poor_wallets_arriving_while_good_ones_leave(self):
        arrivals = [Entry("T", f"chaser{i}", 300.0 + i, capital_usd=500.0)
                    for i in range(8)]
        skills = {f"chaser{i}": 0.1 for i in range(8)}
        skills.update({"pro1": 0.9, "pro2": 0.9})
        reading = late_chaser(arrivals, ["pro1", "pro2"], skills, (250.0, 400.0))
        self.assertEqual("OK", reading.status)
        self.assertEqual(1.0, reading.chaser_share)
        self.assertTrue(reading.is_distribution_pattern)

    def test_unknown_wallets_are_unknown_not_unskilled(self):
        arrivals = [Entry("T", f"w{i}", 300.0 + i) for i in range(8)]
        reading = late_chaser(arrivals, [], {}, (250.0, 400.0))
        self.assertEqual("DATA_BLOCKED", reading.status)
        self.assertIn("unknown, not unskilled", reading.detail)

    def test_skilled_arrivals_are_not_a_distribution_pattern(self):
        arrivals = [Entry("T", f"w{i}", 300.0 + i) for i in range(8)]
        skills = {f"w{i}": 0.9 for i in range(8)}
        reading = late_chaser(arrivals, [], skills, (250.0, 400.0))
        self.assertEqual(0.0, reading.chaser_share)
        self.assertFalse(reading.is_distribution_pattern)


class TheReportOnlyCarriesWhatWasMeasured(unittest.TestCase):

    def test_unmeasured_cohorts_contribute_no_features(self):
        report = evaluate_cohorts([], [], {}, {}, as_of=100.0)
        self.assertEqual("DATA_BLOCKED", report.status)
        self.assertEqual({}, report.features())

    def test_measured_cohorts_produce_named_features(self):
        entries = _entries(25, start=100.0, step=0.0)
        flows = _bought([f"w{i}" for i in range(25)], at=100.0)
        report = evaluate_cohorts(entries, flows, {}, {}, as_of=400.0)
        self.assertEqual("OK", report.status)
        features = report.features()
        self.assertIn("cohort10_retained_60s", features)
        self.assertIn("cohort25_retained_60s", features)
        # Only 25 wallets ever bought, so there is no first-70 cohort. It
        # must not appear at all -- not as a zero, and not as a copy of the
        # 25 under a different name.
        self.assertNotIn("cohort70_retained_60s", features)


if __name__ == "__main__":
    unittest.main()


class ACohortLargerThanTheLaunchIsNotMeasurable(unittest.TestCase):

    def test_a_short_launch_has_no_deep_cohort(self):
        entries = _entries(12, start=100.0, step=0.0)
        flows = _bought([f"w{i}" for i in range(12)], at=100.0)
        reading = cohort_retention(entries, flows, 70, as_of=400.0)
        self.assertEqual("DATA_BLOCKED", reading.status)
        self.assertIn("no first-70 cohort", reading.detail)

    def test_the_depth_it_does_have_is_still_measured(self):
        entries = _entries(12, start=100.0, step=0.0)
        flows = _bought([f"w{i}" for i in range(12)], at=100.0)
        self.assertEqual("OK", cohort_retention(entries, flows, 10, 400.0).status)

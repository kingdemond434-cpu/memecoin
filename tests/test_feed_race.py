"""Coverage outranks speed, and only first arrival mutates state."""

from __future__ import annotations

import unittest

from src.runtime.feed_race import MIN_EVENTS_FOR_VERDICT, SETTLE_SECONDS, FeedRace


class OnlyTheFirstArrivalActs(unittest.TestCase):

    def test_the_winner_returns_true_and_everyone_else_false(self):
        race = FeedRace(["a", "b", "c"])
        self.assertTrue(race.observe("a", "evt", at=100.0))
        self.assertFalse(race.observe("b", "evt", at=100.01))
        self.assertFalse(race.observe("c", "evt", at=100.02))
        self.assertEqual(2, race.duplicates)

    def test_a_late_duplicate_after_settling_still_never_re_fires(self):
        race = FeedRace(["a", "b"])
        race.observe("a", "evt", at=100.0)
        # Long after the event settled and was evicted from the open set.
        self.assertFalse(race.observe("b", "evt", at=100.0 + SETTLE_SECONDS * 10))

    def test_distinct_events_each_get_their_own_winner(self):
        race = FeedRace(["a", "b"])
        self.assertTrue(race.observe("a", "one", at=100.0))
        self.assertTrue(race.observe("b", "two", at=100.0))


class TheRaceMeasuresCoverageNotJustSpeed(unittest.TestCase):

    def _run(self, count=120):
        race = FeedRace(["fast_lossy", "slow_complete"])
        for index in range(count):
            base = 1000.0 + index * 100.0
            # fast_lossy is 10ms quicker but misses one event in three.
            if index % 3:
                race.observe("fast_lossy", f"e{index}", at=base)
                race.observe("slow_complete", f"e{index}", at=base + 0.010)
            else:
                race.observe("slow_complete", f"e{index}", at=base + 0.010)
        # Settle everything.
        race.observe("slow_complete", "flush", at=1000.0 + count * 100.0 + 1e6)
        return race

    def test_the_fast_feed_wins_most_races(self):
        race = self._run()
        fast = race.verdict("fast_lossy")
        self.assertEqual("OK", fast.status)
        self.assertGreater(fast.win_share, 0.9)
        self.assertAlmostEqual(10.0, fast.lead_p50_ms, delta=1.0)

    def test_and_still_loses_on_coverage(self):
        race = self._run()
        fast = race.verdict("fast_lossy")
        complete = race.verdict("slow_complete")
        self.assertLess(fast.coverage, 0.75)
        self.assertGreater(complete.coverage, 0.99)

    def test_best_feed_prefers_the_one_that_sees_everything(self):
        # The whole point: a feed that misses events is not slow on them,
        # it is blind, and a win-rate table cannot see that.
        self.assertEqual("slow_complete", self._run().best_feed())

    def test_unique_finds_are_credited_to_the_only_feed_that_had_them(self):
        race = self._run()
        self.assertGreater(race.verdict("slow_complete").unique, 0)
        self.assertEqual(0, race.verdict("fast_lossy").unique)


class ThinEvidenceYieldsNoVerdict(unittest.TestCase):

    def test_a_handful_of_events_is_not_a_measurement(self):
        race = FeedRace(["a", "b"])
        for index in range(MIN_EVENTS_FOR_VERDICT - 5):
            race.observe("a", f"e{index}", at=100.0 + index)
        verdict = race.verdict("a")
        self.assertEqual("DATA_BLOCKED", verdict.status)
        self.assertIsNone(verdict.win_share)

    def test_the_report_says_so_rather_than_showing_zeros(self):
        report = FeedRace(["a"]).report()
        self.assertEqual("DATA_BLOCKED", report["status"])
        self.assertIn("coverage outranks speed", report["note"])


if __name__ == "__main__":
    unittest.main()

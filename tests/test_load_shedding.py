"""What the desk drops when it cannot look at everything.

The old rule dropped whatever arrived once the queue was full. That is a
queue discipline, not a decision: during the exact minute when the best
launch of the day is most likely to appear -- a burst -- the desk's rule for
handling it was "were you 99th or 101st". A burst is not noise to be
survived; it is when the opportunities are.
"""

from __future__ import annotations

import unittest

from src.runtime.load_shedding import (
    MAX_SHED_QUANTILE, MIN_PRIORITIES, SHED_FLOOR, EconomicLoadShedder)


class NothingIsShedWhileThereIsRoom(unittest.TestCase):

    def test_an_empty_desk_admits_the_worst_launch_there_is(self):
        shedder = EconomicLoadShedder(capacity=100)
        decision = shedder.admit({"deployer_score": -1.0}, in_flight=0)
        self.assertTrue(decision.admitted)
        self.assertIsNone(decision.bar)

    def test_below_the_floor_nothing_is_filtered(self):
        shedder = EconomicLoadShedder(capacity=100)
        for _ in range(MIN_PRIORITIES * 4):
            shedder.admit({"named_by_source": True}, in_flight=0)
        in_flight = int(SHED_FLOOR * 100) - 1
        self.assertTrue(shedder.admit({}, in_flight=in_flight).admitted)

    def test_a_new_desk_does_not_start_by_refusing_launches(self):
        # There is no distribution to take a quantile of yet, and refusing on
        # a sample of four is a policy change wearing a capacity argument.
        shedder = EconomicLoadShedder(capacity=10)
        for _ in range(4):
            shedder.admit({"named_by_source": True}, in_flight=0)
        self.assertTrue(shedder.admit({}, in_flight=10 - 1).admitted)


class UnderPressureTheWorstGoFirst(unittest.TestCase):

    def _loaded(self, capacity=100):
        shedder = EconomicLoadShedder(capacity=capacity)
        # A realistic spread: most launches score the base rate, some are
        # named by a source, some come from a deployer the desk dislikes.
        for index in range(MIN_PRIORITIES * 4):
            if index % 4 == 0:
                shedder.admit({"named_by_source": True}, in_flight=0)
            elif index % 4 == 1:
                shedder.admit({"deployer_score": -0.8}, in_flight=0)
            else:
                shedder.admit({}, in_flight=0)
        return shedder

    def test_at_capacity_a_promising_launch_is_still_admitted(self):
        shedder = self._loaded()
        decision = shedder.admit(
            {"named_by_source": True, "named_actor": True,
             "venue_verified": True}, in_flight=99)
        self.assertTrue(decision.admitted, decision.reason)

    def test_at_capacity_a_poor_launch_is_declined(self):
        shedder = self._loaded()
        decision = shedder.admit({"deployer_score": -1.0}, in_flight=99)
        self.assertFalse(decision.admitted)
        self.assertIn("below", decision.reason)

    def test_the_bar_rises_with_pressure_rather_than_with_opinion(self):
        shedder = self._loaded()
        low = shedder._bar(SHED_FLOOR + 0.01)
        high = shedder._bar(1.0)
        self.assertIsNotNone(high)
        if low is not None:
            self.assertLessEqual(low, high)

    def test_a_genuinely_full_queue_admits_nothing_at_any_priority(self):
        # The slot does not exist. No priority conjures one.
        shedder = self._loaded()
        decision = shedder.admit(
            {"named_by_source": True, "named_actor": True}, in_flight=100)
        self.assertFalse(decision.admitted)
        self.assertIn("in use", decision.reason)


class IgnoranceIsTheBaseRateNotAPenalty(unittest.TestCase):

    def test_an_unknown_deployer_scores_the_same_as_no_information(self):
        # Most launches are from a deployer nobody has seen, and so are most
        # of the ones worth catching. Treating that as negative would shed
        # exactly the population the desk exists to find.
        shedder = EconomicLoadShedder(capacity=10)
        self.assertEqual(shedder.priority({}), shedder.priority({"deployer_score": None}))

    def test_a_known_bad_deployer_ranks_below_an_unknown_one(self):
        shedder = EconomicLoadShedder(capacity=10)
        self.assertLess(shedder.priority({"deployer_score": -1.0}),
                        shedder.priority({}))

    def test_being_named_before_launch_outranks_everything_derivable(self):
        # The only free signal not derivable from the launch transaction.
        shedder = EconomicLoadShedder(capacity=10)
        named = shedder.priority({"named_by_source": True})
        structural = shedder.priority({"venue_verified": True,
                                       "deployer_launches": 5,
                                       "funding_wallets": ["a"]})
        self.assertGreater(named, structural)


class EveryShedLaunchIsARecordedDecision(unittest.TestCase):

    def test_a_decline_carries_its_priority_and_the_bar_it_failed(self):
        shedder = EconomicLoadShedder(capacity=10)
        for _ in range(MIN_PRIORITIES * 2):
            shedder.admit({"named_by_source": True}, in_flight=0)
        decision = shedder.admit({"deployer_score": -1.0}, in_flight=9)
        record = decision.as_dict()
        self.assertFalse(record["admitted"])
        self.assertIsNotNone(record["priority"])
        self.assertIsNotNone(record["bar"])
        self.assertGreater(record["utilisation"], 0.0)

    def test_the_report_says_what_was_shed_and_how_good_it_was(self):
        # The only question that matters about this whole mechanism is
        # whether the launches it shed were the ones that went on to run.
        shedder = EconomicLoadShedder(capacity=10)
        for _ in range(MIN_PRIORITIES * 2):
            shedder.admit({"named_by_source": True}, in_flight=0)
        for _ in range(5):
            shedder.admit({"deployer_score": -1.0}, in_flight=9)
        report = shedder.report()
        self.assertGreater(report["shed"], 0)
        self.assertIsNotNone(report["median_shed_priority"])
        self.assertIsNotNone(report["median_priority"])
        self.assertLess(report["median_shed_priority"], report["median_priority"])


if __name__ == "__main__":
    unittest.main()

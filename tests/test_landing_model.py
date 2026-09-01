"""The two conditioners the model was built to accept and never got.

`Attempt` has carried `leader` and the rest since it was written, with a
docstring saying the models that read them did not exist yet and that the
fields were recorded because they cannot be recovered later. These are those
models, plus account-lock contention -- the term that decides launch sniping
and that no bid/congestion table can see.
"""

from __future__ import annotations

import unittest

from src.execution.landing_model import (
    CONTENTION_BUCKETS, Attempt, LandingModel, contention_bucket)


class ContentionIsBucketedCoarsely(unittest.TestCase):

    def test_unmeasured_contention_is_its_own_bucket(self):
        # None is "nobody counted", not "nobody was competing".
        self.assertEqual(-1, contention_bucket(None))

    def test_buckets_take_their_lower_bound(self):
        self.assertEqual(0, contention_bucket(0))
        self.assertEqual(1, contention_bucket(2))
        self.assertEqual(3, contention_bucket(7))
        self.assertEqual(20, contention_bucket(500))

    def test_nine_and_eleven_competitors_are_the_same_situation(self):
        self.assertEqual(contention_bucket(9), contention_bucket(11))

    def test_one_and_twenty_are_not(self):
        self.assertNotEqual(contention_bucket(1), contention_bucket(20))


class ContentionConditionsLandingSeparatelyFromCongestion(unittest.TestCase):

    def _model(self):
        model = LandingModel(min_bucket_attempts=20)
        # Same bid, same congestion, opposite contention -- and opposite
        # outcomes. A bid/congestion table averages these into one number.
        for index in range(60):
            model.record(Attempt(bid_lamports=10_000, landed=index < 54,
                                 congestion=0.5, account_contention=0))
            model.record(Attempt(bid_lamports=10_000, landed=index < 6,
                                 congestion=0.5, account_contention=30))
        return model

    def test_a_quiet_curve_and_a_contested_one_read_differently(self):
        model = self._model()
        quiet = model.contention_probability(10_000, writers=0)
        hot = model.contention_probability(10_000, writers=30)
        self.assertEqual("OK", quiet.status)
        self.assertEqual("OK", hot.status)
        self.assertGreater(quiet.probability, 0.8)
        self.assertLess(hot.probability, 0.2)

    def test_the_unconditioned_estimate_averages_them_away(self):
        # The reason contention is worth conditioning on at all.
        pooled = self._model().probability(10_000, congestion=0.5)
        self.assertEqual("OK", pooled.status)
        self.assertAlmostEqual(0.5, pooled.probability, delta=0.1)

    def test_a_thin_contention_cell_falls_back_rather_than_refusing(self):
        model = self._model()
        estimate = model.contention_probability(10_000, writers=5)
        self.assertEqual("OK", estimate.status)
        self.assertNotIn("contention", estimate.congestion or "")

    def test_unmeasured_contention_falls_back_too(self):
        estimate = self._model().contention_probability(10_000, writers=None)
        self.assertEqual("OK", estimate.status)


class LeaderEffectIsARatioNotARate(unittest.TestCase):

    def _model(self):
        model = LandingModel(min_bucket_attempts=20)
        for index in range(100):
            model.record(Attempt(bid_lamports=1_000, landed=index < 90,
                                 leader="generous"))
            model.record(Attempt(bid_lamports=1_000, landed=index < 30,
                                 leader="stingy"))
        return model

    def test_a_generous_validator_scores_above_the_fleet(self):
        self.assertGreater(self._model().leader_effect("generous"), 1.0)

    def test_a_stingy_one_scores_below(self):
        self.assertLess(self._model().leader_effect("stingy"), 1.0)

    def test_an_unseen_validator_has_no_effect_rather_than_a_neutral_one(self):
        # There are thousands of them; a rate from four attempts is not one.
        self.assertIsNone(self._model().leader_effect("never_seen"))

    def test_a_thinly_observed_validator_is_refused(self):
        model = self._model()
        for _ in range(3):
            model.record(Attempt(bid_lamports=1_000, landed=True, leader="rare"))
        self.assertIsNone(model.leader_effect("rare"))


class ContentionSurvivesTheRoundTrip(unittest.TestCase):

    def test_it_is_serialised_so_it_can_never_be_lost(self):
        attempt = Attempt(bid_lamports=5_000, landed=True, account_contention=17)
        restored = Attempt.from_dict(attempt.to_dict())
        self.assertEqual(17, restored.account_contention)

    def test_an_unmeasured_value_round_trips_as_unmeasured(self):
        restored = Attempt.from_dict(
            Attempt(bid_lamports=1, landed=False).to_dict())
        self.assertIsNone(restored.account_contention)


if __name__ == "__main__":
    unittest.main()

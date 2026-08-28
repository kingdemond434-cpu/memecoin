"""The feasible-multiple head is fitted in log space; both exits must invert it.

There are two prediction paths -- `predict` and `predict_batch` -- and they
duplicate the post-processing chain. Inverting the log fit in one and not the
other does not raise: it silently returns e^x as a ceiling on what every
survival bin may pay, so a head that meant 1.0 authorises 2.7 and a head that
meant 4.75 authorises 115. These tests hold the two paths to the same answer
and hold the inverse to its bounds.
"""

import math
import unittest

import numpy as np

from src.strategies.multihead_predictor import (
    LOG_SPACE_TARGETS, LOG_TARGET_FLOOR, SURVIVAL_LEVELS, PredictionTarget,
    _from_log_space,
)


class TheInverseIsBounded(unittest.TestCase):
    def test_it_round_trips_ordinary_multiples(self):
        for multiple in (0.5, 1.0, 2.0, 7.5, 100.0):
            self.assertAlmostEqual(_from_log_space(math.log(multiple)),
                                   multiple, places=6)

    def test_a_runaway_prediction_cannot_authorise_more_than_the_curve(self):
        """exp() is unbounded; this value caps claimed upside, so it must not be."""
        ceiling = float(SURVIVAL_LEVELS[-1][1])
        for wild in (20.0, 50.0, 1e6, 1e300):
            self.assertLessEqual(_from_log_space(wild), ceiling)

    def test_it_never_returns_a_non_finite_or_negative_cap(self):
        for broken in (float("nan"), float("inf"), float("-inf"), -1e9):
            value = _from_log_space(broken)
            self.assertTrue(np.isfinite(value))
            self.assertGreaterEqual(value, LOG_TARGET_FLOOR)

    def test_a_total_loss_stays_at_the_floor_not_at_zero(self):
        self.assertEqual(_from_log_space(math.log(LOG_TARGET_FLOOR)),
                         LOG_TARGET_FLOOR)


class BothPredictionPathsInvertIt(unittest.TestCase):
    """A grep-level guard, because the failure is silent and the paths drift."""

    def source(self):
        from src.strategies import multihead_predictor
        with open(multihead_predictor.__file__, encoding="utf-8") as handle:
            return handle.read()

    def test_the_inverse_is_applied_twice(self):
        # Once in predict, once in predict_batch. If a third path appears it
        # must invert too, and this count is what makes that visible.
        self.assertEqual(self.source().count("_from_log_space(val)"), 2)

    def test_the_fit_uses_log_of_the_target(self):
        self.assertIn("np.log(np.clip(y, LOG_TARGET_FLOOR, None))", self.source())

    def test_the_head_is_registered_as_log_space(self):
        self.assertIn(PredictionTarget.EXPECTED_FEASIBLE_MULTIPLE,
                      LOG_SPACE_TARGETS)


class AStaleArtifactIsRefused(unittest.TestCase):
    """A raw-space bundle in exponentiating code is the dangerous combination."""

    def test_load_rejects_a_target_space_mismatch(self):
        from src.strategies import multihead_predictor
        with open(multihead_predictor.__file__, encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn("target-space mismatch", source)
        self.assertIn('"log_space_targets": sorted(', source)

    def test_artifact_version_moved_past_the_raw_space_bundles(self):
        from src.strategies.multihead_predictor import MultiHeadPredictor
        self.assertGreaterEqual(MultiHeadPredictor.ARTIFACT_VERSION, 6)


if __name__ == "__main__":
    unittest.main()

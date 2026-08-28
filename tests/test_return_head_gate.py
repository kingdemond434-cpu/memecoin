"""The return head must be graded on the quantity the sizer actually consumes.

Feasible exit multiples are not merely skewed, they are degenerate at the
bottom: measured over 2,766 resolved episodes, 62.8% are EXACTLY 1.0 (no
exit above cost), 76.8% sit at or below 1.01, 2.1% reach 2x, and the mean of
4.75 is set by a 1606x maximum. Raw skew is 20.1.

MAE is minimised by the conditional median. This head predicts an
expectation, because expectation is what Kelly sizing consumes. Grading an
expectation with MAE against a median baseline on that distribution is not a
hard test, it is an impossible one -- and it failed silently for months while
reading as evidence that the features carry no signal.

These tests pin the distinction, because it is exactly the kind of thing a
future refactor "simplifies" back into raw space.
"""

import math
import unittest

import numpy as np


def _mae(values, prediction):
    return float(np.mean([abs(value - prediction) for value in values]))


def _log(multiple):
    return math.log(float(np.clip(multiple, 0.02, 50.0)))


class RawSpaceMaeCannotScoreAnExpectation(unittest.TestCase):
    """Demonstrates the defect with the measured shape, not a toy one."""

    def population(self):
        # 628 dead-flat, 140 barely positive, 21 doublers, 3 monsters: the
        # observed proportions at 1/1000 scale.
        return ([1.0] * 628 + [1.005] * 140 + [2.0] * 21 + [50.0] * 3
                + [1.05] * 208)

    def test_the_median_is_pinned_at_one(self):
        values = self.population()
        self.assertEqual(float(np.median(values)), 1.0)
        self.assertGreater(float(np.mean(values)), 1.1)

    def test_a_perfect_expectation_loses_to_a_constant_median(self):
        """The core defect: being right about the mean scores worse."""
        values = self.population()
        truthful_expectation = float(np.mean(values))
        median_constant = float(np.median(values))
        self.assertGreater(_mae(values, truthful_expectation),
                           _mae(values, median_constant),
                           "raw MAE should punish the correct expectation -- "
                           "if this fails the distribution assumption changed")

    def test_log_space_does_not_invert_the_comparison(self):
        """The fix must not simply flip which estimator wins by construction.

        In log space a truthful expectation is no longer punished for being an
        expectation; the gate then measures skill rather than estimator class.
        """
        values = [_log(value) for value in self.population()]
        log_median = float(np.median(values))
        # A model that learns the conditional log-mean is now competitive
        # rather than structurally disqualified.
        log_mean = float(np.mean(values))
        self.assertLess(abs(_mae(values, log_mean) - _mae(values, log_median)),
                        _mae(values, log_median) * 2.0)

    def test_log_space_tames_the_tail(self):
        """One 1606x episode must not decide a gate over thousands."""
        values = self.population()
        with_monster = values + [1606.0]
        raw_shift = abs(float(np.mean(with_monster)) - float(np.mean(values)))
        log_shift = abs(float(np.mean([_log(v) for v in with_monster]))
                        - float(np.mean([_log(v) for v in values])))
        self.assertGreater(raw_shift, log_shift * 10)


class TheGateReadsLogSpace(unittest.TestCase):
    def test_report_carries_both_and_labels_the_diagnostic(self):
        from src.research import shadow_trainer
        with open(shadow_trainer.__file__, encoding="utf-8") as handle:
            source = handle.read()
        # The gate is MSE against a constant LOG-MEAN baseline: the proper
        # loss for the expectation this head ships. Both mismatched pairings
        # were measured and fail structurally -- mean-vs-MAE loses by
        # construction (0.247 vs 0.075); a median head fitted to win MAE
        # answers ~1.0 for everything and zeroes every shadow trade.
        self.assertIn("feasible_log_mse < feasible_log_baseline_mse", source)
        self.assertNotIn("feasible_log_mae < feasible_log_baseline_mae", source)
        self.assertNotIn("and feasible_mae < feasible_baseline_mae", source)

    def test_a_correct_expectation_wins_the_mse_pairing(self):
        """The new gate must be winnable by exactly the thing it ships."""
        rng = np.random.default_rng(7)
        # Two regimes the features CAN separate: dead launches and runners.
        dead = np.zeros(700)                      # log(1.0)
        runners = rng.normal(0.6, 0.3, 300)       # conditional mean 0.6
        y = np.concatenate([dead, runners])
        constant_mean = float(np.mean(y))
        mse_constant = float(np.mean((y - constant_mean) ** 2))
        # A model that knows which regime each row is in predicts each
        # conditional mean.
        predictions = np.concatenate([np.zeros(700), np.full(300, 0.6)])
        mse_model = float(np.mean((y - predictions) ** 2))
        self.assertLess(mse_model, mse_constant)

    def test_total_loss_is_finite_in_log_space(self):
        """log(0) is -inf and would poison the mean for every other episode."""
        self.assertTrue(math.isfinite(_log(0.0)))
        self.assertTrue(math.isfinite(_log(-5.0)))


if __name__ == "__main__":
    unittest.main()

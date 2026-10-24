import unittest
import numpy as np
from src.research.market_robustness import cpcv_splits, reality_check


class RobustnessTests(unittest.TestCase):
    def test_actual_label_windows_and_embargo_never_overlap_test(self):
        starts = np.arange(30) * 10 + 100
        ends = starts + 19
        result = cpcv_splits(starts, ends, embargo_seconds=15)
        self.assertEqual(15, result["n_splits"])
        for split in result["splits"]:
            self.assertFalse(set(split["train"]) & set(split["test"]))
            for a in split["train"]:
                for b in split["test"]:
                    self.assertTrue(ends[a] < starts[b] or starts[a] > ends[b] + 15)
        self.assertFalse(result["forward_evidence"])

    def test_invalid_or_unbounded_splits_are_refused(self):
        with self.assertRaises(ValueError): cpcv_splits([1, 1, 2, 3], [2, 2, 3, 4], groups=4)
        with self.assertRaises(ValueError): cpcv_splits(range(100), range(1, 101), groups=20)

    def test_identical_strategy_and_control_cannot_pass(self):
        result = reality_check(np.zeros((40, 2)), np.zeros(40), declared_trials=2,
                               evidence_refs=["fixture"] * 40, costs_measured=True, iterations=100)
        self.assertEqual(1., result["p_value"])
        self.assertEqual("NO_ESTABLISHED_FAMILY_EDGE", result["status"])

    def test_positive_control_survives_without_granting_authority(self):
        result = reality_check(np.full((40, 2), .01), np.zeros(40), declared_trials=2,
                               evidence_refs=["synthetic-test-only"] * 40, costs_measured=True, iterations=100)
        self.assertLess(result["p_value"], .05)
        self.assertFalse(result["promotion_authority"])
        self.assertFalse(result["untouched_forward_evidence"])

    def test_omitted_trials_and_missing_costs_are_not_permission(self):
        data = np.ones((40, 2)) * .01
        self.assertEqual("DATA_BLOCKED", reality_check(data, np.zeros(40), declared_trials=3,
                         evidence_refs=["fixture"] * 40, costs_measured=True)["status"])
        with self.assertRaises(ValueError): reality_check(data, np.zeros(40), declared_trials=2,
                         evidence_refs=["fixture"] * 40, costs_measured=False)

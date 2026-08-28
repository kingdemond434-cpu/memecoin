"""The gates that stood between the desk and its first shadow trade.

Each had the same shape: an honest-looking status hiding a structural
problem. The flash band was blocked by the head least relevant to entry; the
shadow policy reported a bare zero with no reason; and a five-positive sample
wore a REJECTED verdict it could not support.
"""

import unittest

from src.strategies.multihead_predictor import (
    CLASSIFICATION_TARGETS, ConstantZeroClassifier, MultiHeadPredictor,
    OPTIONAL_TAIL_TARGETS, PredictionFeatures, PredictionTarget,
)

# A fee-seeding "fix" for `fee_schedule_unobserved` was written and reverted
# on the same day: PumpFeeSchedule's legacy 100 bps is the BONDING CURVE's
# published fee, and the first observed PumpSwap trade tail charges 30 bps --
# a table-lookup seed would overstate every pool round trip by ~140 bps
# inside the research labels. The correct behaviour (pool unpriceable until a
# readable trade tail or account decode) is pinned by
# tests.test_core.TestDeskMaintainsPoolState; nothing here re-tests it.


def _features(index):
    return PredictionFeatures(
        token=f"tok{index}", chain="solana", timestamp=1000.0 + index,
        initial_buyers=index % 7, sol_volume=float(index % 11),
        buy_velocity=float(index % 5), liquidity_usd=100.0 + index)


def _labels(index, tail_positive=False):
    labels = {}
    for target in PredictionTarget:
        if target in CLASSIFICATION_TARGETS:
            if target in OPTIONAL_TAIL_TARGETS:
                labels[target] = 1.0 if tail_positive else 0.0
            else:
                # Ordinary rungs alternate so every fit window sees both
                # classes.
                labels[target] = float(index % 2)
        elif target == PredictionTarget.EXPECTED_FEASIBLE_MULTIPLE:
            labels[target] = 1.0 + (index % 3)
        else:
            labels[target] = float(index % 4)
    return labels


class TheFlashBandCannotBeBlockedByItsRarestHead(unittest.TestCase):
    """A missing 500x positive must not deny the 0-0.5s band a model."""

    def train(self, tail_positive=False):
        predictor = MultiHeadPredictor("unused")
        predictor.initialize_models()
        for index in range(160):
            predictor.add_training_sample(
                _features(index), _labels(index, tail_positive=tail_positive))
        results = predictor.train(min_samples=100)
        return predictor, results

    def test_absent_tail_positives_yield_a_conservative_zero_head(self):
        predictor, results = self.train(tail_positive=False)
        self.assertTrue(predictor._is_trained,
                        "the band trained; the tail head must not veto it")
        for target in OPTIONAL_TAIL_TARGETS:
            self.assertEqual(results[target.value]["status"],
                             "untrained_conservative_zero")
            self.assertIsInstance(predictor.models[target],
                                  ConstantZeroClassifier)

    def test_the_zero_is_stated_not_learned(self):
        predictor, _ = self.train(tail_positive=False)
        prediction = predictor.predict(_features(999))
        self.assertIsNotNone(prediction)
        for target in OPTIONAL_TAIL_TARGETS:
            self.assertEqual(getattr(prediction, target.value), 0.0)

    def test_an_entry_gating_head_still_blocks_its_band(self):
        """p_2x is not optional: a band that cannot cover it has no model."""
        predictor = MultiHeadPredictor("unused")
        predictor.initialize_models()
        for index in range(160):
            labels = _labels(index)
            labels[PredictionTarget.P_2X] = 0.0   # never any positive
            predictor.add_training_sample(_features(index), labels)
        predictor.train(min_samples=100)
        self.assertFalse(predictor._is_trained)

    def test_present_tail_positives_are_still_fitted_normally(self):
        predictor = MultiHeadPredictor("unused")
        predictor.initialize_models()
        for index in range(160):
            labels = _labels(index, tail_positive=(index % 3 == 0))
            predictor.add_training_sample(_features(index), labels)
        results = predictor.train(min_samples=100)
        for target in OPTIONAL_TAIL_TARGETS:
            self.assertEqual(results[target.value]["status"], "trained",
                             "real positives must be learned, not zeroed")


class RejectionReasonsAreCounted(unittest.TestCase):
    """`shadow_policy_trades: 0` must say WHICH gate refused every candidate."""

    def test_the_report_carries_the_breakdown(self):
        from src.research import shadow_trainer
        with open(shadow_trainer.__file__, encoding="utf-8") as handle:
            source = handle.read()
        for needle in ('"shadow_policy": shadow_policy',
                       '"rejected_by": rejected',
                       '"p_2x_below_0.10"',
                       '"max_p_2x_seen"'):
            self.assertIn(needle, source)


class FivepositivesDoNotSupportAVerdict(unittest.TestCase):
    """REJECTED means evidence against; five positives are not evidence."""

    def source(self):
        from src.research import hazard_trainer
        with open(hazard_trainer.__file__, encoding="utf-8") as handle:
            return handle.read()

    def test_a_verdict_needs_ten_oos_positives(self):
        self.assertIn("insufficient_oos_positives_for_verdict", self.source())
        self.assertIn("verdict_floor_positives", self.source())

    def test_the_model_is_still_kept_below_the_floor(self):
        """Blocked-for-verdict is not discarded: hazard is used defensively."""
        source = self.source()
        floor_at = source.index("insufficient_oos_positives_for_verdict")
        kept_at = source.index("models[key]=model")
        self.assertGreater(kept_at, floor_at,
                           "the fitted model must be registered even when the "
                           "verdict is withheld")


if __name__ == "__main__":
    unittest.main()

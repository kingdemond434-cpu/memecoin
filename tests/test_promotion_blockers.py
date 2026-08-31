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


class RejectedMonstersAreAttributedByCause(unittest.TestCase):
    """A vetoed monster must name WHICH veto cause discarded it.

    All three observed monster discards carried one aggregate label, which
    left the veto untunable: no way to see whether authority checks or exit
    impact or developer flags were doing the discarding, or whether the same
    cause ever earned a save.
    """

    def test_reject_reason_reaches_the_costliest_screens(self):
        from src.research.launch_census import LaunchCensus
        census = LaunchCensus()
        census.see("MintA")
        census.reject("MintA", "safety_veto:mint_authority_active")
        census.resolve("MintA", peak_multiple=75.0)
        report = census.report()
        ranked = report["missed_monsters"]["costliest_screens"]
        reasons = [row["reason"] for row in ranked]
        self.assertIn("safety_veto:mint_authority_active", reasons)

    def test_screened_monsters_keep_their_existing_attribution(self):
        from src.research.launch_census import LaunchCensus
        census = LaunchCensus()
        census.see("MintB")
        census.screen("MintB", "veto_safety")
        census.resolve("MintB", peak_multiple=75.0)
        reasons = [row["reason"] for row in
                   census.report()["missed_monsters"]["costliest_screens"]]
        self.assertIn("veto_safety", reasons)


class JupiterTokenListSurvivesTheRetiredDomain(unittest.TestCase):
    """token.jup.ag/all no longer resolves at all -- DNS failure, not 404/410.

    The miner ran silently (0 records, state ERROR) rather than reporting a
    fixable route. The v2 tag endpoint is the live replacement and keys the
    mint as "id", not "address" -- the field this miner's whole output is
    keyed on, so getting it wrong reads as an empty list rather than a
    mapping bug.
    """

    def test_the_url_is_not_the_retired_domain(self):
        from src.research import solana_miners
        self.assertNotIn("token.jup.ag", solana_miners.JUPITER_TOKENS_URL)
        self.assertIn("api.jup.ag/tokens/v2", solana_miners.JUPITER_TOKENS_URL)

    def test_the_miner_reads_id_not_address(self):
        from src.research import solana_miners
        with open(solana_miners.__file__, encoding="utf-8") as handle:
            source = handle.read()
        # The old field name must not still be read as the mint.
        self.assertNotIn('item.get("address")', source)
        self.assertIn('item.get("id")', source)


class BackfillSurvivesTransientRpcExhaustion(unittest.TestCase):
    """A shared-pool rate limit mid-run must not discard already-decoded work.

    Measured 2026-08-29: a backfill sharing the free RPC pool with the live
    desk hit simultaneous cooldowns on every endpoint serving
    getSignaturesForAddress at page 17 (2,239 events already decoded) and
    raised past the fetch loop entirely -- the write-out step never ran, so
    hours of paging and every decoded event were discarded for a condition
    that clears itself in seconds.
    """

    def source(self):
        from tools import backfill_history
        with open(backfill_history.__file__, encoding="utf-8") as handle:
            return handle.read()

    def test_the_no_healthy_endpoint_error_is_caught_and_retried(self):
        source = self.source()
        self.assertIn('"No healthy RPC endpoint" not in str(exc)', source)
        self.assertIn("retrying in", source)

    def test_exhaustion_still_reaches_the_write_out(self):
        """The retry loop must end in a `break`, not a raise, so control
        falls through to run_backfill() instead of skipping it."""
        source = self.source()
        exhausted_at = source.index("exhausted = True")
        write_out_at = source.index("_group_into_launches(collector.events)")
        self.assertGreater(write_out_at, exhausted_at)

    def test_the_report_names_an_early_stop(self):
        self.assertIn('"stopped_early_rpc_exhausted": exhausted', self.source())

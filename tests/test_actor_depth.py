"""Three actor signals the desk had the events for and was not computing.

The independence graph asks whether a PAIR of wallets is the same actor,
from evidence inside one launch. These ask three questions it structurally
cannot: which wallets keep arriving together across thousands of launches,
what a wallet DOES when its address is brand new, and whether a wallet is
ahead of the first public mention more often than its own timing predicts.
"""

from __future__ import annotations

import time
import unittest

from src.strategies.actor_graph import Entry, IndependenceReport, aggregate_smart_flow
from src.strategies.pre_event_anomaly import PreEventAnomaly
from src.strategies.sniper_rings import (
    MIN_CO_OPENINGS, MIN_LAUNCHES_OBSERVED, MIN_LAUNCHES_PER_WALLET,
    SniperRingDetector)
from src.strategies.wallet_signature import (
    MIN_CLUSTER_WALLETS, MIN_ENTRIES, WalletSignatures)


class RingsAreSurprisalNotCounts(unittest.TestCase):
    """The pair that co-opens most is usually just the pair that opens most.

    Two prolific bots meet everywhere. What matters is co-occurrence far
    above what their individual rates predict.
    """

    def _detector(self):
        return SniperRingDetector(min_observed=200, min_launches=20,
                                  min_co_openings=8)

    def test_two_wallets_that_always_arrive_together_form_a_ring(self):
        detector = self._detector()
        for index in range(400):
            if index % 4 == 0:
                detector.observe_launch(["A", "B", f"noise{index}"])
            else:
                detector.observe_launch([f"noise{index}"])
        rings = detector.rings(recompute=True)
        self.assertEqual(1, len(rings))
        self.assertEqual({"A", "B"}, set(rings[0].members))

    def test_two_prolific_wallets_that_merely_open_everything_do_not(self):
        # Both open 100% of launches, so co-opening 100% is exactly what
        # independence predicts and carries no surprise at all.
        detector = self._detector()
        for index in range(400):
            detector.observe_launch(["Busy1", "Busy2", f"noise{index}"])
        self.assertEqual([], detector.rings(recompute=True))

    def test_a_thin_history_refuses_to_conclude(self):
        # Below the observation floor the base rates are themselves noise,
        # so every pair looks surprising.
        detector = self._detector()
        for index in range(50):
            detector.observe_launch(["A", "B"])
        self.assertEqual([], detector.rings(recompute=True))
        self.assertEqual("DATA_BLOCKED", detector.report()["status"])

    def test_rings_are_transitive(self):
        # If A is not independent of B and B is not independent of C, then
        # counting A, B and C as three independent buyers is wrong even
        # where the A-C pair alone is not significant.
        detector = self._detector()
        for index in range(400):
            if index % 4 == 0:
                detector.observe_launch(["A", "B", f"n{index}"])
            elif index % 4 == 1:
                detector.observe_launch(["B", "C", f"n{index}"])
            else:
                detector.observe_launch([f"n{index}"])
        rings = detector.rings(recompute=True)
        self.assertEqual(1, len(rings))
        self.assertEqual({"A", "B", "C"}, set(rings[0].members))

    def test_a_ring_collapses_the_independent_count(self):
        detector = self._detector()
        for index in range(400):
            if index % 4 == 0:
                detector.observe_launch(["A", "B", "C", f"n{index}"])
            else:
                detector.observe_launch([f"n{index}"])
        count, detail = detector.independent_count(["A", "B", "C", "Stranger"])
        self.assertEqual(2, count, detail)
        self.assertEqual(4, detail["raw"])
        self.assertTrue(detail["collapsed"])

    def test_an_enormous_component_is_reported_but_not_used(self):
        # A "ring" of four hundred addresses describes the market, not a
        # coordinated actor, and discounting on it would zero every launch.
        from src.strategies.sniper_rings import Ring

        self.assertFalse(Ring(frozenset(f"w{i}" for i in range(400)), 9, 40.0).usable)
        self.assertTrue(Ring(frozenset({"a", "b"}), 9, 40.0).usable)


class ARingDiscountsTheSizeItInflated(unittest.TestCase):
    """The desk's sizing rests on counting INDEPENDENT skilled buyers.

    A ring of twelve addresses running one strategy is one buyer wearing
    twelve hats, and counting it as twelve overcounts in the direction that
    says enter bigger.
    """

    def _entries(self, wallets):
        return [Entry(token="M", wallet=w, timestamp=float(index),
                      skill=0.8, capital_usd=100.0)
                for index, w in enumerate(wallets)]

    def _rings_over(self, ringed):
        detector = SniperRingDetector(min_observed=200, min_launches=20,
                                      min_co_openings=8)
        for index in range(400):
            if index % 4 == 0:
                detector.observe_launch(list(ringed) + [f"n{index}"])
            else:
                detector.observe_launch([f"n{index}"])
        detector.rings(recompute=True)
        return detector

    def test_smart_flow_falls_when_the_buyers_are_one_ring(self):
        report = IndependenceReport(status="OK", scores={}, observed_pairs=0)
        wallets = ["A", "B", "C", "D"]
        without = aggregate_smart_flow(self._entries(wallets), report)
        with_rings = aggregate_smart_flow(self._entries(wallets), report,
                                          rings=self._rings_over(wallets))
        self.assertLess(with_rings.evidence, without.evidence)
        self.assertEqual(4, with_rings.ring_collapsed)
        self.assertIn("sniper rings", with_rings.detail)

    def test_unrelated_buyers_are_untouched(self):
        report = IndependenceReport(status="OK", scores={}, observed_pairs=0)
        entries = self._entries(["X", "Y", "Z"])
        rings = self._rings_over(["A", "B"])
        self.assertEqual(aggregate_smart_flow(entries, report).evidence,
                         aggregate_smart_flow(entries, report, rings=rings).evidence)

    def test_a_missing_detector_changes_nothing(self):
        report = IndependenceReport(status="OK", scores={}, observed_pairs=0)
        entries = self._entries(["X", "Y"])
        self.assertEqual(aggregate_smart_flow(entries, report).evidence,
                         aggregate_smart_flow(entries, report, rings=None).evidence)


class ASignatureIsBehaviourNotIdentity(unittest.TestCase):

    def _seeded(self, wallets, base_age=0.5, size=0.2):
        signatures = WalletSignatures()
        now = 1_700_000_000.0
        for wallet in wallets:
            for index in range(MIN_ENTRIES + 2):
                signatures.observe_entry(
                    wallet, entry_age_s=base_age, size_sol=size,
                    deployer=f"d{index}", at=now + index * 3600)
                signatures.observe_exit(wallet, hold_s=90.0)
        return signatures

    def test_a_wallet_with_too_little_history_has_no_signature(self):
        signatures = WalletSignatures()
        signatures.observe_entry("New", entry_age_s=0.5, size_sol=0.2)
        result = signatures.match("New")
        self.assertEqual("DATA_BLOCKED", result["status"])
        self.assertIn("needed for a", result["reason"])

    def test_a_cluster_needs_enough_wallets_to_be_defined(self):
        signatures = self._seeded(["a", "b"])
        self.assertFalse(signatures.define_cluster("tiny", ["a", "b"]))

    def test_a_new_wallet_behaving_like_a_measured_cluster_matches(self):
        members = [f"m{i}" for i in range(MIN_CLUSTER_WALLETS)]
        signatures = self._seeded(members + ["twin", "other"])
        # `other` runs a visibly different program: late, large, patient.
        for index in range(MIN_ENTRIES + 2):
            signatures.observe_entry("other", entry_age_s=600.0, size_sol=40.0,
                                     at=1_700_000_000.0 + index * 3600)
            signatures.observe_exit("other", hold_s=90_000.0)
        self.assertTrue(signatures.define_cluster("fast", members,
                                                  forward_elogw=0.02))
        twin = signatures.match("twin")
        self.assertEqual("OK", twin["status"])
        self.assertTrue(twin["matched"], twin)
        self.assertLess(twin["distance"], signatures.match("other")["distance"])

    def test_a_match_never_claims_the_addresses_are_one_actor(self):
        members = [f"m{i}" for i in range(MIN_CLUSTER_WALLETS)]
        signatures = self._seeded(members + ["twin"])
        signatures.define_cluster("fast", members, forward_elogw=0.02)
        result = signatures.match("twin")
        self.assertIn("NOT a claim", result["means"])
        self.assertEqual("BEHAVIOURAL_PRIOR", result["provenance"])

    def test_a_wallet_is_not_matched_to_its_own_cluster(self):
        members = [f"m{i}" for i in range(MIN_CLUSTER_WALLETS)]
        signatures = self._seeded(members)
        signatures.define_cluster("fast", members)
        self.assertEqual("DATA_BLOCKED", signatures.match("m0")["status"])


class EarlinessIsMeasuredAgainstThePublicSignal(unittest.TestCase):

    def _anomaly(self):
        return PreEventAnomaly(min_comparable=25)

    def test_a_launch_with_no_observed_mention_is_not_comparable(self):
        # Source coverage is uneven, and a wallet that happens to trade the
        # launches nobody covers would look prescient on a naive count.
        anomaly = self._anomaly()
        for index in range(50):
            anomaly.observe_entry("W", f"tok{index}", 1.0)
        result = anomaly.score("W")
        self.assertEqual("DATA_BLOCKED", result["status"])
        self.assertEqual(0, result["comparable_launches"])

    def test_the_earliest_mention_is_the_one_that_counts(self):
        # A launch mentioned at t+4s and again at t+90s was public at t+4s.
        anomaly = self._anomaly()
        anomaly.note_public_mention("tok", 90.0)
        anomaly.note_public_mention("tok", 4.0)
        self.assertEqual(4.0, anomaly._first_mention_s["tok"])

    def test_a_wallet_consistently_ahead_of_the_public_signal_is_flagged(self):
        anomaly = self._anomaly()
        for index in range(120):
            token = f"tok{index}"
            anomaly.note_public_mention(token, 60.0)
            # Early on the ones that get mentioned, late on the ones that
            # do not -- so its own null rate is genuinely middling.
            anomaly.observe_entry("Ahead", token, 1.0)
        for index in range(120):
            anomaly.observe_entry("Ahead", f"quiet{index}", 300.0)
        result = anomaly.score("Ahead")
        self.assertEqual("OK", result["status"])
        self.assertTrue(result["anomalous"], result)
        self.assertGreater(result["median_lead_s"], 0)

    def test_a_bot_that_buys_everything_early_is_not_flagged(self):
        # Its null rate is near one: leading everything is exactly what its
        # own timing predicts, so there is no surprise to report.
        anomaly = self._anomaly()
        for index in range(120):
            token = f"tok{index}"
            anomaly.note_public_mention(token, 60.0)
            anomaly.observe_entry("Fast", token, 0.2)
        result = anomaly.score("Fast")
        self.assertEqual("OK", result["status"])
        self.assertFalse(result["anomalous"], result)

    def test_the_claim_is_about_timing_not_knowledge(self):
        anomaly = self._anomaly()
        for index in range(60):
            token = f"tok{index}"
            anomaly.note_public_mention(token, 60.0)
            anomaly.observe_entry("W", token, 1.0)
        for index in range(60):
            anomaly.observe_entry("W", f"quiet{index}", 300.0)
        result = anomaly.score("W")
        self.assertIn("not about knowledge", result["means"])
        self.assertEqual("PUBLIC_TIMING_ONLY", result["provenance"])
        self.assertIn("uneven", anomaly.report()["coverage_caveat"])


class TheDeskFeedsAllThree(unittest.IsolatedAsyncioTestCase):

    async def test_they_exist_and_report(self):
        from src.main import MemecoinQuantDesk

        desk = MemecoinQuantDesk("config/chains.yaml", dry_run_override=True,
                                 offline=True)
        await desk.initialize()
        report = desk.readiness()
        for key in ("sniper_rings", "wallet_signatures", "pre_event_anomaly"):
            self.assertIn(key, report)
            self.assertNotEqual("MISSING", report[key]["status"])


if __name__ == "__main__":
    unittest.main()


class TheSurprisalBoundHasOneImplementation(unittest.TestCase):
    """Two copies of one piece of arithmetic in one language is the shape
    that silently diverges when one is fixed -- and this one has edge cases
    that are easy to get wrong in the direction that manufactures findings."""

    def test_certainty_under_the_null_is_never_surprising(self):
        # Two wallets that both open EVERY launch co-open every launch, and
        # that is exactly what independence predicts. Getting this wrong
        # turns the two busiest bots on the chain into the most suspicious
        # pair on it, which is precisely backwards.
        from src.strategies.surprisal import binomial_surprisal

        self.assertEqual(0.0, binomial_surprisal(400, 400, 1.0))
        self.assertLess(binomial_surprisal(400, 400, 0.999999), 1.0)

    def test_observing_every_trial_does_not_raise(self):
        from src.strategies.surprisal import binomial_surprisal

        # 0*log(0) is zero by convention and a domain error in floating point.
        self.assertGreater(binomial_surprisal(100, 100, 0.01), 0.0)

    def test_below_the_null_rate_is_not_surprising(self):
        from src.strategies.surprisal import binomial_surprisal

        self.assertEqual(0.0, binomial_surprisal(5, 100, 0.5))

    def test_the_bound_understates_rather_than_overstates(self):
        # A detector that misses a real ring costs an opportunity; one that
        # invents rings discounts every launch the desk sees.
        from src.strategies.surprisal import binomial_surprisal

        modest = binomial_surprisal(20, 100, 0.1)
        extreme = binomial_surprisal(60, 100, 0.1)
        self.assertLess(modest, extreme)
        self.assertGreater(modest, 0.0)

    def test_the_poisson_bound_still_answers_its_own_question(self):
        from src.strategies.surprisal import poisson_surprisal

        # An exchange emitting two withdrawals a second produces a
        # five-in-ten-seconds group constantly.
        self.assertEqual(0.0, poisson_surprisal(5, 10.0, 2.0))
        self.assertGreater(poisson_surprisal(50, 1.0, 0.1), 0.0)

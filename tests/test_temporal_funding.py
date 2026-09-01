"""Coordination routed through an exchange, and the base rate that guards it."""

from __future__ import annotations

import unittest

from src.strategies.temporal_funding import (
    MIN_CLUSTER_SIZE, TemporalCluster, Withdrawal, find_clusters,
    independence_discounts, measure_source_rate)


def _batch(count: int, start: float, spacing: float, amount: float = 5.0,
           source: str = "binance_hot", prefix: str = "w", fresh: bool = True):
    return [Withdrawal(wallet=f"{prefix}{index}", source=source,
                       timestamp=start + index * spacing, amount_sol=amount,
                       wallet_first_seen=start - 3600.0 if fresh else 0.0)
            for index in range(count)]


class TheBaseRateDoesTheWork(unittest.TestCase):
    """Without the hot wallet's own emission rate, everyone is a conspiracy."""

    def test_a_busy_exchange_makes_a_tight_group_unremarkable(self):
        # Five withdrawals in ten seconds, from a wallet emitting two per
        # second. Utterly ordinary, and a naive detector flags it.
        found = find_clusters(_batch(5, 1000.0, 2.0),
                              source_rates={"binance_hot": 2.0})
        self.assertEqual([], [c for c in found if c.status == "OK"])

    def test_the_same_group_is_damning_from_a_quiet_source(self):
        found = find_clusters(_batch(5, 1000.0, 2.0),
                              source_rates={"binance_hot": 0.002})
        clusters = [c for c in found if c.status == "OK"]
        self.assertEqual(1, len(clusters))
        self.assertEqual(5, clusters[0].size)
        self.assertGreater(clusters[0].surprisal, 3.0)

    def test_no_measured_rate_means_no_cluster_at_all(self):
        found = find_clusters(_batch(6, 1000.0, 1.0), source_rates={})
        self.assertTrue(all(c.status == "DATA_BLOCKED" for c in found))
        self.assertIn("no measured emission rate", found[0].detail)

    def test_a_rate_from_too_few_observations_is_not_a_rate(self):
        self.assertIsNone(measure_source_rate(_batch(4, 0.0, 1.0),
                                              "binance_hot", 60.0))

    def test_a_rate_from_enough_observations_is_returned(self):
        rate = measure_source_rate(_batch(60, 0.0, 1.0), "binance_hot", 60.0)
        self.assertAlmostEqual(1.0, rate)


class ClustersAreEvidenceNotIdentity(unittest.TestCase):

    def test_two_wallets_are_never_a_cluster(self):
        found = find_clusters(_batch(2, 1000.0, 1.0),
                              source_rates={"binance_hot": 0.001})
        self.assertEqual([], [c for c in found if c.status == "OK"])
        self.assertGreaterEqual(MIN_CLUSTER_SIZE, 3)

    def test_the_discount_can_never_zero_a_wallet_s_independence(self):
        # Overwhelming evidence: fifty wallets in one second from a dead source.
        found = find_clusters(_batch(50, 1000.0, 0.02),
                              source_rates={"binance_hot": 1e-6})
        cluster = [c for c in found if c.status == "OK"][0]
        self.assertGreater(cluster.surprisal, 50.0)
        self.assertLess(cluster.discount, 1.0)
        discounts = independence_discounts([cluster])
        self.assertGreater(min(discounts.values()), 0.0)

    def test_matching_amounts_strengthen_the_reading(self):
        tight = find_clusters(_batch(6, 1000.0, 1.0, amount=5.0),
                              source_rates={"binance_hot": 0.001})
        varied = [Withdrawal(f"v{i}", "binance_hot", 1000.0 + i, 1.0 + i * 7.0)
                  for i in range(6)]
        loose = find_clusters(varied, source_rates={"binance_hot": 0.001})
        strong = [c for c in tight if c.status == "OK"][0]
        weak = [c for c in loose if c.status == "OK"][0]
        self.assertGreater(strong.amount_agreement, weak.amount_agreement)
        self.assertGreater(strong.discount, weak.discount)

    def test_a_cluster_that_never_touched_this_launch_is_not_this_launch_s_problem(self):
        found = find_clusters(_batch(6, 1000.0, 1.0),
                              target_buyers=["someone_else"],
                              source_rates={"binance_hot": 0.001})
        self.assertEqual([], [c for c in found if c.status == "OK"])

    def test_a_cluster_touching_the_launch_is_reported(self):
        found = find_clusters(_batch(6, 1000.0, 1.0),
                              target_buyers=["w3"],
                              source_rates={"binance_hot": 0.001})
        self.assertEqual(1, len([c for c in found if c.status == "OK"]))


class DiscountsCombineWithoutDoubleCounting(unittest.TestCase):

    def test_a_wallet_in_two_clusters_takes_the_stronger_not_the_product(self):
        weak = TemporalCluster(status="OK", wallets=["w1"], surprisal=4.0,
                               amount_agreement=1.0)
        strong = TemporalCluster(status="OK", wallets=["w1"], surprisal=40.0,
                                 amount_agreement=1.0)
        combined = independence_discounts([weak, strong])["w1"]
        alone = independence_discounts([strong])["w1"]
        self.assertAlmostEqual(alone, combined,
                               msg="two views of one coordination are one fact")

    def test_blocked_clusters_discount_nobody(self):
        blocked = TemporalCluster(status="DATA_BLOCKED", wallets=["w1"])
        self.assertEqual({}, independence_discounts([blocked]))


if __name__ == "__main__":
    unittest.main()

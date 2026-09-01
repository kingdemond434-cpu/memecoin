"""Following a profitable wallet is a different question from admiring one."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from src.research.benchmark_wallets import (
    FOLLOW_DELAYS_S, MIN_DECISIONS_FOR_VERDICT, BenchmarkCorpus, BenchmarkWallet,
    DiscoveryCriteria, WalletDecision, discover_candidates, load_roster)

ROSTER = Path(__file__).resolve().parents[1] / "config" / "benchmark_wallets.yaml"


def _decision(wallet, index, entry, exit_price, delay_prices, age=10.0):
    return WalletDecision(
        wallet=wallet, token=f"t{index}", entered_at=1000.0 + index,
        launch_age_s=age, buyer_rank=index % 20, entry_price=entry,
        price_at_delay=dict(delay_prices), exit_price=exit_price,
        exited_at=2000.0 + index)


class HeadlinePnlIsAClaimNotAMeasurement(unittest.TestCase):

    def test_disagreeing_trackers_are_surfaced_not_averaged(self):
        wallet = BenchmarkWallet(
            address="A", claimed_pnl_usd={"okx": 1_900_000, "pawscan": 2_370_000})
        self.assertTrue(wallet.claims_disagree)

    def test_a_single_source_cannot_disagree_with_itself(self):
        self.assertFalse(
            BenchmarkWallet(address="A", claimed_pnl_usd={"okx": 11e6}).claims_disagree)

    def test_claims_never_reach_the_verdict(self):
        corpus = BenchmarkCorpus()
        corpus.register(BenchmarkWallet(address="A",
                                        claimed_pnl_usd={"okx": 99_000_000}))
        # An enormous claimed PnL and no decisions is still no verdict.
        verdict = corpus.follow_verdict("A")
        self.assertEqual("DATA_BLOCKED", verdict.status)
        self.assertIsNone(verdict.followable)


class TheVerdictIsAboutFOLLOWINGNotAboutTheWallet(unittest.TestCase):

    def test_a_wallet_whose_edge_is_speed_is_not_followable(self):
        # The wallet doubles its money every time. By +250ms the move has
        # already happened, and a follower buys the top.
        corpus = BenchmarkCorpus(cost_per_round_trip=0.02)
        corpus.register(BenchmarkWallet(address="fast"))
        for index in range(60):
            corpus.record(_decision(
                "fast", index, entry=1.0, exit_price=2.0,
                delay_prices={0.05: 1.1, 0.10: 1.4, 0.25: 2.1,
                              0.50: 2.4, 1.00: 2.6}))
        verdict = corpus.follow_verdict("fast")
        self.assertEqual("OK", verdict.status)
        # It made money for itself...
        self.assertGreater(verdict.wallet_mean_log_return, 0.6)
        # ...and following it late loses.
        self.assertLess(verdict.mean_log_return[1.00], 0.0)
        self.assertEqual(0.05, verdict.best_delay)
        self.assertGreater(verdict.edge_decay, 0.5)

    def test_a_wallet_with_durable_edge_survives_the_delay(self):
        corpus = BenchmarkCorpus(cost_per_round_trip=0.02)
        corpus.register(BenchmarkWallet(address="durable"))
        for index in range(60):
            corpus.record(_decision(
                "durable", index, entry=1.0, exit_price=3.0,
                delay_prices={d: 1.02 for d in FOLLOW_DELAYS_S}))
        verdict = corpus.follow_verdict("durable")
        self.assertTrue(verdict.followable)
        self.assertLess(abs(verdict.edge_decay), 1e-9)

    def test_costs_are_charged_on_every_simulated_follow(self):
        gross = BenchmarkCorpus(cost_per_round_trip=0.0)
        dear = BenchmarkCorpus(cost_per_round_trip=0.2)
        for corpus in (gross, dear):
            corpus.register(BenchmarkWallet(address="w"))
            for index in range(40):
                corpus.record(_decision("w", index, 1.0, 1.05,
                                        {d: 1.0 for d in FOLLOW_DELAYS_S}))
        cheap = gross.follow_verdict("w").mean_log_return[0.05]
        costly = dear.follow_verdict("w").mean_log_return[0.05]
        self.assertGreater(cheap, 0.0)
        self.assertLess(costly, 0.0,
                        "an edge measured gross is an edge that does not exist")

    def test_a_thin_record_yields_no_verdict(self):
        corpus = BenchmarkCorpus()
        corpus.register(BenchmarkWallet(address="thin"))
        for index in range(MIN_DECISIONS_FOR_VERDICT - 1):
            corpus.record(_decision("thin", index, 1.0, 50.0,
                                    {d: 1.0 for d in FOLLOW_DELAYS_S}))
        verdict = corpus.follow_verdict("thin")
        self.assertEqual("DATA_BLOCKED", verdict.status)
        self.assertIsNone(verdict.followable)

    def test_unresolved_decisions_do_not_count_toward_the_bar(self):
        corpus = BenchmarkCorpus()
        corpus.register(BenchmarkWallet(address="open"))
        for index in range(80):
            corpus.record(WalletDecision(wallet="open", token=f"t{index}",
                                         entered_at=1000.0, entry_price=1.0))
        self.assertEqual("DATA_BLOCKED", corpus.follow_verdict("open").status)


class TheCorpusCarriesItsOwnSelectionWarning(unittest.TestCase):

    def test_every_report_states_the_bias(self):
        corpus = BenchmarkCorpus()
        corpus.register(BenchmarkWallet(address="A", label="euris"))
        report = corpus.report()
        self.assertIn("biased upward", report["selection_warning"])

    def test_the_shipped_roster_parses_and_records_claims_as_claims(self):
        corpus = load_roster(str(ROSTER))
        self.assertGreaterEqual(len(corpus.wallets), 5)
        disputed = [w for w in corpus.wallets.values() if w.claims_disagree]
        self.assertTrue(disputed, "the roster keeps a worked example of "
                                  "two trackers disagreeing")
        # And with no reconstructed decisions, it claims nothing.
        self.assertEqual("DATA_BLOCKED", corpus.report()["status"])

    def test_a_missing_roster_is_not_fatal(self):
        corpus = load_roster("/nonexistent/roster.yaml")
        self.assertEqual(0, len(corpus.wallets))


class DiscoveryPrefersBehaviourOverJackpots(unittest.TestCase):

    def test_one_enormous_winner_is_not_a_behaviour(self):
        rows = [_decision("lucky", 0, 1.0, 2000.0,
                          {d: 1.0 for d in FOLLOW_DELAYS_S})]
        self.assertEqual([], discover_candidates({"lucky": rows}))

    def test_many_modest_wins_are(self):
        rows = [_decision("grinder", i, 1.0, 1.4,
                          {d: 1.0 for d in FOLLOW_DELAYS_S}) for i in range(50)]
        found = discover_candidates({"grinder": rows})
        self.assertEqual(1, len(found))
        self.assertEqual("grinder", found[0][0])
        self.assertIn("losing wallets were equally visible", found[0][1]["basis"])

    def test_a_wallet_that_only_buys_old_tokens_is_not_a_sniper(self):
        rows = [_decision("latecomer", i, 1.0, 1.4,
                          {d: 1.0 for d in FOLLOW_DELAYS_S}, age=7200.0)
                for i in range(50)]
        self.assertEqual([], discover_candidates({"latecomer": rows}))

    def test_criteria_are_adjustable_without_editing_the_finder(self):
        rows = [_decision("g", i, 1.0, 1.4, {d: 1.0 for d in FOLLOW_DELAYS_S})
                for i in range(10)]
        self.assertEqual([], discover_candidates({"g": rows}))
        loose = DiscoveryCriteria(min_decisions=5)
        self.assertEqual(1, len(discover_candidates({"g": rows}, loose)))


class TheCorpusSurvivesARestart(unittest.TestCase):

    def test_round_trip_preserves_decisions_and_claims(self):
        path = Path(tempfile.mkdtemp()) / "bench.json"
        corpus = BenchmarkCorpus(str(path))
        corpus.register(BenchmarkWallet(address="A", label="euris",
                                        claimed_pnl_usd={"okx": 11e6}))
        for index in range(5):
            corpus.record(_decision("A", index, 1.0, 2.0,
                                    {d: 1.1 for d in FOLLOW_DELAYS_S}))
        self.assertTrue(corpus.save())

        restored = BenchmarkCorpus(str(path))
        self.assertTrue(restored.load())
        self.assertEqual(5, restored.size)
        self.assertEqual("euris", restored.wallets["A"].label)
        self.assertEqual(1.1, restored.decisions("A")[0].price_at_delay[0.05])


if __name__ == "__main__":
    unittest.main()

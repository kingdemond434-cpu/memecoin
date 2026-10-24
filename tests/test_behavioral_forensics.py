import json
import tempfile
import time
import unittest
from pathlib import Path

from src.research.behavioral_forensics import PublicWalletResearch, normalize_trade, paired_growth_test, structure
from src.research.prediction_markets import PaperBroker


def raw(**changes):
    row = {"proxy_wallet": "0x" + "a" * 40, "side": "BUY", "condition_id": "0x" + "b" * 64,
           "token_id": "123", "size": "10", "price": ".4", "timestamp": time.time() - 10,
           "outcome": "Yes", "transaction_hash": "0x" + "c" * 64}
    row.update(changes)
    return row


class StructureTests(unittest.TestCase):
    def test_public_pnl_is_never_invented_from_a_slice(self):
        result = structure([normalize_trade(raw())])
        self.assertIsNone(result["realized_after_cost_pnl"])
        self.assertFalse(result["promotion_authority"])
        self.assertIn("BUY_ONLY_IN_OBSERVED_SLICE", result["mechanisms"])

    def test_complementary_buys_do_not_establish_atomic_arbitrage(self):
        result = structure([normalize_trade(raw()), normalize_trade(raw(outcome="No", token_id="124", price=".5"))])
        pair = result["complementary_buys"][0]
        self.assertAlmostEqual(.9, pair["combined_average_buy_price"])
        self.assertIsNone(pair["after_fee_edge"])
        self.assertFalse(pair["simultaneous_execution_established"])

    def test_unrelated_conditions_are_not_a_hedge(self):
        result = structure([normalize_trade(raw()), normalize_trade(raw(outcome="No", condition_id="0x" + "d" * 64))])
        self.assertEqual([], result["complementary_buys"])

    def test_adverse_size_escalation_is_detected_without_asserting_private_strategy(self):
        rows = [normalize_trade(raw(timestamp=time.time() - 100 + i, price=str(.8 - i * .01), size=str(2 ** i))) for i in range(12)]
        report = structure(rows)
        self.assertIn("ADVERSE_ADDS_OBSERVED", report["mechanisms"])
        self.assertIn("SIZE_ESCALATION_OBSERVED", report["mechanisms"])

    def test_milliseconds_are_normalized_and_precision_is_retained(self):
        stamp = int(time.time() * 1000) - 10000
        trade = normalize_trade(raw(timestamp=stamp))
        self.assertAlmostEqual(stamp / 1000, trade["at"])
        self.assertEqual("milliseconds", trade["timestamp_precision"])

    def test_invalid_or_future_data_is_refused(self):
        for values in ({"size": "NaN"}, {"price": 2}, {"proxy_wallet": "bad"}, {"timestamp": time.time() + 1000}):
            with self.assertRaises(ValueError): normalize_trade(raw(**values))

    def test_after_cost_test_requires_controls_native_costs_and_evidence(self):
        with self.assertRaises(ValueError): paired_growth_test([{"decision_at": 10}], declared_trials=1)
        self.assertEqual("DATA_BLOCKED", paired_growth_test([], declared_trials=100)["status"])

    def test_daily_dependence_and_multiplicity_are_preserved(self):
        base = int(time.time() // 86400) * 86400 - 60 * 86400
        rows = [{"decision_at": base + day * 86400, "outcome_observed_at": base + day * 86400 + 100,
                 "net_return": .01, "control_net_return": 0, "costs_measured": True,
                 "paired_same_cohort": True, "exposure_semantics": "daily_portfolio", "evidence_ref": f"fixture:{day}"} for day in range(40)]
        self.assertEqual("DATA_BLOCKED", paired_growth_test(rows, declared_trials=100, iterations=100)["status"])
        result = paired_growth_test(rows, declared_trials=10, iterations=1000)
        self.assertEqual(.005, result["familywise_alpha"])
        self.assertGreater(result["lower_bound"], 0)
        self.assertFalse(result["promotion_authority"])
        del rows[10]
        self.assertEqual("DATA_BLOCKED", paired_growth_test(rows, declared_trials=1)["status"])


class PublicMinerTests(unittest.IsolatedAsyncioTestCase):
    async def test_identical_observed_wallet_inputs_reuse_structure_after_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            broker = PaperBroker(Path(temp) / "paper.sqlite")
            trade = raw()
            class Venue:
                async def get(self, base, path, params):
                    if path.endswith("leaderboard"):
                        return {"data": [{"user_id": trade["proxy_wallet"]}], "pagination": {"next_cursor": None}}
                    return {"data": [trade], "pagination": {"next_cursor": None}}
            try:
                await PublicWalletResearch(broker).cycle(Venue())
                second = PublicWalletResearch(broker)
                await second.cycle(Venue())
                row = second.report()["research_memory"]["families"][0]
                self.assertEqual(1, row["distinct_studies"])
                self.assertEqual(1, row["exact_reuse_hits"])
                self.assertIsNone(row["economic_roi"])
                report = json.loads(broker.db.execute("SELECT payload FROM public_forensics").fetchone()[0])
                self.assertEqual(trade["proxy_wallet"], report["wallet"])
            finally: broker.close()

    async def test_real_producer_to_consumer_path_with_restart_and_pagination(self):
        with tempfile.TemporaryDirectory() as temp:
            broker = PaperBroker(Path(temp) / "paper.sqlite")
            try:
                miner = PublicWalletResearch(broker)
                requests = []
                class Venue:
                    async def get(self, base, path, params):
                        requests.append((path, params))
                        if path.endswith("leaderboard"): return {"data": [{"user_id": "0x" + "a" * 40, "pnl": 100}], "pagination": {"next_cursor": None}}
                        return {"data": [raw()], "pagination": {"next_cursor": "older"}}
                venue = Venue()
                await miner.cycle(venue)
                first = miner.report()
                self.assertEqual(1, first["forensic_reports"])
                second = PublicWalletResearch(broker)
                await second.cycle(venue)
                self.assertTrue(any(p.get("cursor") == "older" for _, p in requests))
                report = json.loads(broker.db.execute("SELECT payload FROM public_forensics").fetchone()[0])
                self.assertFalse(report["history_complete"])
                self.assertFalse(report["promotion_authority"])
                self.assertTrue(report["reported_pnl_is_unverified"])
            finally: broker.close()

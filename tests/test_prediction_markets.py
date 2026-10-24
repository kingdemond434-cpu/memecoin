import asyncio
import json
import tempfile
import time
import unittest
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from src.research.prediction_markets import (
    Book, Market, PaperBroker, PredictionPaperRunner, PublicVenues, fee, levels, make_book, number)


class PaperExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "paper.sqlite"
        self.broker = PaperBroker(self.path)
        self.market = Market("polymarket", "m", "Question", "yes", "no", "open", "0.07", "polymarket_quadratic")
        self.now = time.time()
        self.book = make_book("polymarket", "yes", [[".4", "100"]], [[".5", "20"], [".6", "100"]], self.now)
        self.broker.observe(self.market, [self.book])

    def tearDown(self):
        self.broker.close()
        self.temp.cleanup()

    def order(self, **kwargs):
        args = dict(account="polymarket:yes_control_v1", order_id="one", side="BUY", quantity="3", limit=".6", submitted_at=self.now - 2, now=self.now)
        args.update(kwargs)
        return self.broker.submit(self.market, self.book, **args)

    def test_walks_depth_charges_fees_and_updates_cash_atomically(self):
        receipt = self.order()
        self.assertEqual("SIMULATED_FILLED", receipt["status"])
        self.assertEqual(Decimal("1.6"), Decimal(receipt["gross"]))
        self.assertEqual(Decimal(".05180"), Decimal(receipt["fee"]))
        cash = self.broker.db.execute("SELECT cash FROM accounts WHERE id='polymarket:yes_control_v1'").fetchone()[0]
        self.assertEqual(Decimal("998.34820"), Decimal(cash))

    def test_retry_is_idempotent_and_intent_collision_is_refused(self):
        first = self.order()
        self.assertEqual(first, self.order())
        with self.assertRaises(ValueError): self.order(quantity="4")
        self.assertEqual(1, self.broker.db.execute("SELECT count(*) FROM orders").fetchone()[0])

    def test_fok_failure_does_not_consume_depth_or_cash(self):
        self.assertEqual("REJECTED", self.order(quantity="30")["status"])
        self.assertEqual(0, self.broker.db.execute("SELECT count(*) FROM consumed").fetchone()[0])
        self.assertEqual("SIMULATED_FILLED", self.order(order_id="new")["status"])

    def test_fak_partial_fill(self):
        receipt = self.order(quantity="30", tif="FAK")
        self.assertEqual("SIMULATED_PARTIAL", receipt["status"])
        self.assertEqual(Decimal("12"), Decimal(receipt["filled"]))

    def test_depth_cannot_be_reused_by_same_account(self):
        self.order(quantity="12")
        self.assertEqual("REJECTED", self.order(order_id="two", quantity="1")["status"])

    def test_controls_are_independent_liquidity_simulations(self):
        self.order(quantity="12")
        self.assertEqual("SIMULATED_FILLED", self.order(account="polymarket:no_control_v1", order_id="other", quantity="12")["status"])

    def test_unknown_fees_block(self):
        self.market = replace(self.market, fee_rate=None)
        self.assertEqual("unknown fees", self.order()["reason"])

    def test_closed_market_blocks(self):
        self.market = replace(self.market, status="closed")
        self.assertEqual("market is closed", self.order()["reason"])

    def test_stale_book_blocks(self):
        self.assertIn("stale", self.order(now=self.now + 31)["reason"])

    def test_predecision_snapshot_blocks(self):
        self.assertIn("post-decision", self.order(submitted_at=self.now)["reason"])

    def test_unretained_or_mutated_book_blocks(self):
        self.book = replace(self.book, asks=((".01", "9999"),))
        self.assertEqual("unretained book", self.order()["reason"])

    def test_short_selling_is_refused(self):
        self.assertEqual("short selling prohibited", self.order(side="SELL")["reason"])

    def test_venue_account_mismatch(self):
        self.assertEqual("venue account mismatch", self.order(account="kalshi:yes_control_v1")["reason"])

    def test_market_asset_mismatch(self):
        self.market = replace(self.market, yes="another")
        self.assertEqual("market/asset mismatch", self.order()["reason"])

    def test_markets_exposure_capped(self):
        self.broker.max_market_cost = Decimal("1")
        self.assertEqual("market exposure cap", self.order()["reason"])

    def test_cash_is_enforced(self):
        with self.broker.db: self.broker.db.execute("UPDATE accounts SET cash='.1'")
        self.assertEqual("insufficient paper cash", self.order()["reason"])

    def test_sell_realizes_after_cost_loss(self):
        self.order()
        receipt = self.order(order_id="sell", side="SELL", quantity="3", limit=".1")
        self.assertEqual("SIMULATED_FILLED", receipt["status"])
        row = self.broker.db.execute("SELECT realized FROM accounts WHERE id='polymarket:yes_control_v1'").fetchone()
        self.assertLess(Decimal(row[0]), 0)

    def test_official_settlement_is_idempotent(self):
        self.order()
        self.assertEqual(0, self.broker.settle(replace(self.market, status="closed")))
        self.assertEqual(1, self.broker.settle(replace(self.market, status="closed", result="yes")))
        self.assertEqual(0, self.broker.settle(replace(self.market, status="closed", result="yes")))
        cash = self.broker.db.execute("SELECT cash FROM accounts WHERE id='polymarket:yes_control_v1'").fetchone()[0]
        self.assertEqual(Decimal("1001.34820"), Decimal(cash))

    def test_restart_preserves_balances_positions_receipts_and_spec(self):
        self.order()
        before = self.broker.report()
        self.broker.close()
        self.broker = PaperBroker(self.path)
        self.assertEqual(before, self.broker.report())

    def test_spec_change_requires_new_experiment(self):
        with self.assertRaisesRegex(ValueError, "specification changed"):
            PaperBroker(self.path, depth_fraction=".2")

    def test_complete_set_rolls_back_both_legs_on_missing_depth(self):
        runner = PredictionPaperRunner(self.broker)
        runner.decisions[self.market.key] = self.now - 3
        yes = make_book("polymarket", "yes", [[".1", "20"]], [[".2", "100"]], self.now)
        no = make_book("polymarket", "no", [[".1", "20"]], [[".2", ".1"]], self.now)
        runner.process(self.market, [yes, no])
        self.assertEqual(0, self.broker.db.execute("SELECT count(*) FROM positions WHERE account='polymarket:complete_set_v1'").fetchone()[0])
        self.assertEqual("1000", self.broker.db.execute("SELECT cash FROM accounts WHERE id='polymarket:complete_set_v1'").fetchone()[0])

    def test_complete_set_commits_both_legs(self):
        runner = PredictionPaperRunner(self.broker)
        runner.decisions[self.market.key] = self.now - 3
        books = [make_book("polymarket", asset, [[".1", "20"]], [[".2", "100"]], self.now) for asset in ("yes", "no")]
        runner.process(self.market, books)
        self.assertEqual(2, self.broker.db.execute("SELECT count(*) FROM positions WHERE account='polymarket:complete_set_v1'").fetchone()[0])

    def test_observation_does_not_increment_real_fills_or_promotion(self):
        self.order()
        self.assertEqual(0, self.broker.report()["real_fills"])
        self.assertEqual(0, self.broker.report()["qualifying_promotion_receipts"])


class AdapterTests(unittest.TestCase):
    def test_fee_rounding_at_both_venues(self):
        p = Market("polymarket", "m", "q", "1", "2", "open", ".07", "polymarket_quadratic")
        self.assertEqual(Decimal("1.75000"), fee(p, Decimal(100), Decimal(".5")))
        k = replace(p, venue="kalshi", fee_kind="kalshi_quadratic")
        self.assertEqual(Decimal(".02"), fee(k, Decimal(1), Decimal(".5")))

    def test_invalid_numeric_input_is_refused(self):
        for value in ("NaN", "Infinity", "-Infinity"):
            with self.assertRaises(ValueError): number(value)
        for price in ("0", "1", "1.1", "-.1"):
            with self.assertRaises(ValueError): levels([[price, "10"]])

    def test_crossed_book_refused(self):
        with self.assertRaises(ValueError): make_book("x", "a", [[".6", "1"]], [[".5", "1"]], time.time())

    def test_polymarket_uses_outcome_labels_not_assumed_array_order(self):
        m = PublicVenues.polymarket({"id": "m", "clobTokenIds": '["2","1"]', "outcomes": '["No","Yes"]', "feesEnabled": False})
        self.assertEqual("1", m.yes)
        self.assertEqual("2", m.no)
        self.assertEqual("0", m.fee_rate)

    def test_polymarket_price_alone_is_not_a_resolution(self):
        row = {"id": "m", "clobTokenIds": '["1","2"]', "outcomes": '["Yes","No"]', "closed": True, "outcomePrices": '["1","0"]'}
        self.assertEqual("", PublicVenues.polymarket(row).result)
        row["umaResolutionStatus"] = "resolved"
        self.assertEqual("yes", PublicVenues.polymarket(row).result)

    def test_unrecognized_fee_exponent_is_blocked(self):
        m = PublicVenues.polymarket({"id": "m", "clobTokenIds": '["1","2"]', "outcomes": '["Yes","No"]', "feesEnabled": True, "feeSchedule": {"rate": .07, "exponent": 2}})
        self.assertIsNone(m.fee_rate)


class PublicTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_kalshi_asks_are_complements_of_opposite_bids(self):
        v = PublicVenues(None)
        async def get(*args, **kwargs):
            return {"orderbook_fp": {"yes_dollars": [[".3", "10.5"]], "no_dollars": [[".4", "20.25"]]}}
        v.get = get
        m = Market("kalshi", "KXTEST-A", "q", "KXTEST-A:yes", "KXTEST-A:no", "open", ".07", "kalshi_quadratic")
        yes, no = await v.books(m)
        self.assertEqual((("0.6", "20.25"),), yes.asks)
        self.assertEqual((("0.7", "10.5"),), no.asks)

    async def test_unknown_kalshi_fee_type_does_not_default_to_zero(self):
        v = PublicVenues(None)
        async def get(*args, **kwargs): return {"series": {"fee_type": "unknown", "fee_multiplier": 1}, "event_fee_changes": [], "cursor": ""}
        v.get = get
        m = await v.kalshi({"ticker": "KXTEST-A", "event_ticker": "KXTEST-E", "status": "active"})
        self.assertIsNone(m.fee_rate)

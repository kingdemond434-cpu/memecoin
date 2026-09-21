"""What the competition pays, off a stream the desk already consumes.

The landing model conditions on bid and chooses one economically, and
contains almost no data -- because a bid model learns from ATTEMPTS and a
DRY_RUN desk makes none. That is a chicken and egg the desk cannot solve on
its own: the model cannot be good until it trades, and it should not trade
on a model that is not good.

Every transaction on the stream landed and carries the compute unit price
its sender chose. The desk was walking past that on every single one.
"""

from __future__ import annotations

import struct
import unittest

from src.execution.observed_bids import (
    COMPUTE_BUDGET_PROGRAM, MIN_SAMPLES, ComputeBudget, ObservedBidCorpus,
    decode_compute_budget)


def _price(micro_lamports):
    return (COMPUTE_BUDGET_PROGRAM,
            bytes([3]) + struct.pack("<Q", micro_lamports))


def _limit(units):
    return (COMPUTE_BUDGET_PROGRAM, bytes([2]) + struct.pack("<I", units))


class TheDecoderReadsTheDeclaredPriority(unittest.TestCase):

    def test_price_and_limit_are_both_read(self):
        budget = decode_compute_budget([_limit(400_000), _price(1_000)])
        self.assertEqual(1_000, budget.unit_price_micro_lamports)
        self.assertEqual(400_000, budget.unit_limit)
        self.assertTrue(budget.stated)

    def test_priority_is_price_times_limit_not_price_alone(self):
        # Paying 1,000 micro-lamports per unit on a 40,000-unit budget and on
        # a 400,000-unit budget are ten times apart in what the leader
        # receives, and the leader orders by the total.
        small = decode_compute_budget([_limit(40_000), _price(1_000)])
        large = decode_compute_budget([_limit(400_000), _price(1_000)])
        self.assertAlmostEqual(large.priority_lamports,
                               small.priority_lamports * 10)

    def test_a_missing_limit_uses_the_runtime_default(self):
        budget = decode_compute_budget([_price(1_000)])
        # 1,000 micro-lamports per unit over the default 200,000-unit budget
        # is 200,000,000 micro-lamports, which is 200 lamports.
        self.assertAlmostEqual(200.0, budget.priority_lamports)

    def test_a_transaction_with_no_budget_states_nothing(self):
        # Not a zero bid: it took the default, and mixing those into a bid
        # distribution drags every percentile toward a number nobody chose.
        budget = decode_compute_budget([("SomeOtherProgram", b"\x01\x02")])
        self.assertFalse(budget.stated)
        self.assertIsNone(budget.priority_lamports)

    def test_a_malformed_budget_instruction_does_not_lose_the_transaction(self):
        budget = decode_compute_budget([
            (COMPUTE_BUDGET_PROGRAM, bytes([3, 1, 2])),   # truncated price
            _price(500)])
        self.assertEqual(500, budget.unit_price_micro_lamports)

    def test_other_programs_are_ignored(self):
        budget = decode_compute_budget([
            ("11111111111111111111111111111111", bytes([3]) + b"\x00" * 8)])
        self.assertFalse(budget.stated)


class ItBucketsByLaunchAge(unittest.TestCase):

    def _corpus(self, count=MIN_SAMPLES, price=1_000, age=0.5):
        corpus = ObservedBidCorpus()
        for index in range(count):
            corpus.observe(ComputeBudget(price + index, 200_000), age)
        return corpus

    def test_the_first_second_is_its_own_bucket(self):
        corpus = self._corpus(age=0.5)
        self.assertIsNotNone(corpus.reference_bid(0.5))
        # And an old launch shares nothing with it.
        self.assertIsNone(corpus.reference_bid(3_600.0))

    def test_a_thin_bucket_reports_no_percentile(self):
        # A bid taken from nine observations is a number that will be acted
        # on and should not be.
        corpus = self._corpus(count=MIN_SAMPLES - 1)
        self.assertIsNone(corpus.reference_bid(0.5))
        bucket = corpus.report()["buckets"][0]
        self.assertEqual("DATA_BLOCKED", bucket["status"])

    def test_percentiles_are_ordered(self):
        corpus = self._corpus()
        self.assertLessEqual(corpus.reference_bid(0.5, 0.50),
                             corpus.reference_bid(0.5, 0.90))

    def test_a_launch_of_unknown_age_is_counted_not_bucketed(self):
        # Without an age it says nothing about the bid war at T0, which is
        # the only part of the distribution worth having.
        corpus = ObservedBidCorpus()
        self.assertFalse(corpus.observe(ComputeBudget(1_000, 200_000), None))
        self.assertEqual(1, corpus.without_age)

    def test_silent_transactions_are_counted_separately(self):
        corpus = ObservedBidCorpus()
        for _ in range(50):
            corpus.observe(ComputeBudget(), 0.5)
        bucket = corpus.report()["buckets"][0]
        self.assertEqual(50, bucket["silent"])
        self.assertEqual(0, bucket["samples"])
        self.assertEqual(1.0, bucket["silent_share"])


class ItSaysWhatItIsNot(unittest.TestCase):

    def test_the_report_states_the_censoring(self):
        # Every transaction here landed. A model fitted to this and read as
        # a landing probability would be confidently wrong in the direction
        # that says every bid works.
        report = ObservedBidCorpus().report()
        self.assertIn("not P(land|bid)", report["censoring"])
        self.assertIn("invisible", report["censoring"])

    def test_an_empty_corpus_is_data_blocked_not_zero(self):
        self.assertEqual("DATA_BLOCKED", ObservedBidCorpus().report()["status"])


class TheStreamFeedsIt(unittest.IsolatedAsyncioTestCase):

    async def test_a_trade_carrying_a_budget_reaches_the_corpus(self):
        import time as _time

        from src.main import MemecoinQuantDesk

        desk = MemecoinQuantDesk("config/chains.yaml", dry_run_override=True,
                                 offline=True)
        await desk.initialize()
        token = "So11111111111111111111111111111111111111112"
        now = _time.time()
        desk.launch_census.see(token, creator="D", at=now - 0.4)
        before = desk.observed_bids.observed
        await desk._on_pump_event({
            "type": "token_trade", "token": token, "slot": 1,
            "timestamp": now, "wallet": "W", "side": "buy",
            "priority_lamports": 200.0, "compute_unit_price": 1_000,
            "compute_unit_limit": 200_000})
        self.assertEqual(before + 1, desk.observed_bids.observed)


if __name__ == "__main__":
    unittest.main()

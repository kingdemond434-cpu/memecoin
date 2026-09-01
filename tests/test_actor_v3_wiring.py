"""The new intelligence must reach a decision, not sit beside one.

Every audit of this repo has found the same failure mode at least once: a
well-built module constructed at startup, reported in /status, and consulted
by nothing. These assert the wiring itself.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.runtime.source_intelligence import SourceIntelligence
from src.strategies.actor_graph import IndependenceReport
from src.strategies.temporal_funding import Withdrawal


class TemporalClustersActuallyDiscountIndependence(unittest.TestCase):

    def _desk(self, withdrawals, scores):
        return SimpleNamespace(
            exchange_withdrawals=withdrawals, exchange_rates={},
            temporal_clusters=[], temporal_discounts={},
            independence_report=IndependenceReport(status="OK", scores=dict(scores)))

    def test_a_coordinated_batch_lowers_every_member_s_score(self):
        # Sixty withdrawals over an hour establishes the base rate; six of
        # them land inside five seconds, which that rate cannot explain.
        withdrawals = [Withdrawal(f"noise{i}", "hot", 1000.0 + i * 60.0, 5.0)
                       for i in range(60)]
        withdrawals += [Withdrawal(f"w{i}", "hot", 5000.0 + i * 0.8, 5.0)
                        for i in range(6)]
        scores = {f"w{i}": 1.0 for i in range(6)}
        desk = self._desk(withdrawals, scores)
        SourceIntelligence._apply_temporal_clusters(desk)
        self.assertTrue(desk.temporal_discounts)
        for wallet in scores:
            self.assertLess(desk.independence_report.scores[wallet], 1.0)

    def test_no_measured_rate_changes_nothing(self):
        # Too few observations to establish a base rate: the module must not
        # invent one, so no wallet is touched.
        withdrawals = [Withdrawal(f"w{i}", "hot", 2000.0 + i * 0.5, 5.0)
                       for i in range(5)]
        scores = {f"w{i}": 1.0 for i in range(5)}
        desk = self._desk(withdrawals, scores)
        SourceIntelligence._apply_temporal_clusters(desk)
        self.assertEqual({}, desk.temporal_discounts)
        for wallet in scores:
            self.assertEqual(1.0, desk.independence_report.scores[wallet])

    def test_a_blocked_independence_report_is_left_alone(self):
        desk = self._desk([Withdrawal("w0", "hot", 1.0, 5.0)], {})
        desk.independence_report = IndependenceReport(status="DATA_BLOCKED")
        SourceIntelligence._apply_temporal_clusters(desk)
        self.assertEqual({}, desk.temporal_discounts)

    def test_discounting_can_only_ever_reduce_never_zero(self):
        withdrawals = [Withdrawal(f"n{i}", "hot", 1000.0 + i * 60.0, 5.0)
                       for i in range(60)]
        withdrawals += [Withdrawal(f"w{i}", "hot", 5000.0 + i * 0.3, 5.0)
                        for i in range(8)]
        scores = {f"w{i}": 0.4 for i in range(8)}
        desk = self._desk(withdrawals, scores)
        SourceIntelligence._apply_temporal_clusters(desk)
        for wallet in scores:
            self.assertLessEqual(desk.independence_report.scores[wallet], 0.4)
            self.assertGreater(desk.independence_report.scores[wallet], 0.0)


class TheDeskConstructsAndShowsTheNewIntelligence(unittest.TestCase):

    def test_wiring_constructs_every_new_component(self):
        import inspect

        from src.runtime import wiring

        source = inspect.getsource(wiring)
        for name in ("cohort_reports", "temporal_clusters", "temporal_discounts",
                     "benchmark_corpus", "exchange_withdrawals"):
            self.assertIn(name, source, f"{name} is not constructed at startup")

    def test_status_reports_all_three(self):
        import inspect

        from src.runtime import reporting

        source = inspect.getsource(reporting)
        for key in ('"cohorts"', '"temporal_funding"', '"benchmark_wallets"'):
            self.assertIn(key, source, f"{key} is not visible in /status")


if __name__ == "__main__":
    unittest.main()

import asyncio
import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aiohttp.test_utils import TestClient, TestServer

from src.collectors.event_source import SourceHealth, SourceState
from src.research.intelligence_coverage import IntelligenceCoverage
from src.research.launch_census import LaunchCensus
from src.research.research_memory import ResearchMemory
from src.runtime.reporting import ReportingSurface
from src.runtime.market_gateway import make_app, COVERAGE_TASK
from src.research.prediction_markets import PaperBroker


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.addCleanup(self.db.close)
        self.memory = ResearchMemory(self.db)
        self.args = ("wallet", {"venue": "polymarket", "wallet": "a"}, ["trade1"], "algorithm1")

    def test_exact_context_inputs_and_algorithm_are_required(self):
        self.memory.retain(*self.args, {"finding": 1}, seconds=.01, records=1)
        for args in (("wallet", {"venue": "kalshi", "wallet": "a"}, ["trade1"], "algorithm1"),
                     (*self.args[:2], ["trade2"], "algorithm1"), (*self.args[:3], "algorithm2")):
            self.assertIsNone(self.memory.recall(*args))
        self.assertEqual({"finding": 1}, self.memory.recall(*self.args))
        row = self.memory.report()["families"][0]
        self.assertEqual(1, row["exact_reuse_hits"])
        self.assertIsNone(row["economic_roi"])

    def test_conflicting_results_and_corruption_are_rejected(self):
        self.memory.retain(*self.args, {"finding": 1}, seconds=0, records=1)
        with self.assertRaises(ValueError):
            self.memory.retain(*self.args, {"finding": 2}, seconds=0, records=1)
        self.db.execute("UPDATE research_memory SET result='{}'")
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.memory.recall(*self.args)

    def test_caller_rollback_and_restart_preserve_transaction_semantics(self):
        with self.assertRaises(RuntimeError):
            with self.db:
                self.db.execute("INSERT INTO research_memory SELECT 'unused','x','{}','x','x','{}','x',0,0,0,0")
                self.memory.retain(*self.args, {}, seconds=0, records=0)
                raise RuntimeError("rollback")
        self.assertIsNone(self.memory.recall(*self.args))
        self.memory.retain(*self.args, {}, seconds=0, records=0)
        self.assertEqual({}, ResearchMemory(self.db).recall(*self.args))


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.addCleanup(self.db.close)
        self.coverage = IntelligenceCoverage(SimpleNamespace(db=self.db))
        self.source = self.coverage.sources[0]["id"]

    def snapshot(self, state="OK", **extra):
        return {"observed_at": time.time(), "sources": [{"source_id": self.source, "state": state,
                "last_poll_ok_at": time.time() - 1, "last_event_at": None, **extra}], "misses": {}}

    def test_configured_is_not_observed_and_stale_is_not_healthy(self):
        self.assertEqual(0, self.coverage.report()["observed_ok_sources"])
        self.coverage.ingest(self.snapshot())
        self.assertEqual(1, self.coverage.report()["observed_ok_sources"])
        self.assertEqual(0, self.coverage.report()["observed_sources_with_events"])
        with patch("src.research.intelligence_coverage.time.time", return_value=time.time() + 181):
            self.assertEqual(0, self.coverage.report()["observed_ok_sources"])
        self.assertIsNone(self.coverage.report()["global_launch_coverage_fraction"])

    def test_dynamic_sources_are_visible_and_source_recovery_preserves_history(self):
        self.source = "discovered:public-channel"
        self.coverage.ingest(self.snapshot("DEAD"))
        self.assertEqual(1, self.coverage.report()["debt_records"])
        self.coverage.ingest(self.snapshot())
        self.assertEqual(1, len(self.coverage.report()["dynamic_source_rows"]))
        jobs = self.coverage.propose_round()
        self.assertFalse(any(j["kind"] == "SOURCE_HEALTH" for j in jobs))
        self.assertGreaterEqual(self.coverage.report()["debt_records"], 1)

    def test_plans_rotate_exploration_and_survive_restart_without_authority(self):
        plans = [self.coverage.propose_round() for _ in range(10)]
        domains = {j["subject"] for plan in plans for j in plan if j["kind"] == "BLIND_SPOT_EXPLORATION"}
        self.assertEqual(set(self.coverage.scope["domains"]), domains)
        for plan in plans:
            self.assertLessEqual(len(plan), 8)
            self.assertEqual(2, sum(j["kind"] == "BLIND_SPOT_EXPLORATION" for j in plan))
            self.assertTrue(all(j["promotion_authority"] is False for j in plan))
        self.assertEqual(10, IntelligenceCoverage(SimpleNamespace(db=self.db)).report()["completed_planning_rounds"])

    def test_untrusted_fields_are_not_retained_or_allowed_to_assert_profit(self):
        snap = self.snapshot(secret="do-not-retain")
        snap["misses"] = {"review_candidates": [{"token": "token", "peak_multiple": 1.1,
                             "trade_authority": True, "executable_after_cost_profit": 500, "raw": "secret"}]}
        self.coverage.ingest(snap)
        payload = self.db.execute("SELECT payload FROM intelligence_snapshots").fetchone()[0]
        self.assertNotIn("secret", payload)
        candidate = json.loads(payload)["misses"]["review_candidates"][0]
        self.assertFalse(candidate["trade_authority"])
        self.assertIsNone(candidate["executable_after_cost_profit"])
        self.assertEqual(1, self.coverage.report()["debt_records"])

    def test_invalid_batches_are_atomic(self):
        for mutate in (lambda s: s.update(observed_at=float("nan")),
                       lambda s: s["sources"].append(s["sources"][0]),
                       lambda s: s["sources"][0].update(last_event_at=float("inf")),
                       lambda s: s.update(misses={"review_candidates": [{"token": "bad", "peak_multiple": -1}]})):
            snap = self.snapshot()
            mutate(snap)
            with self.assertRaises(ValueError): self.coverage.ingest(snap)
        self.assertEqual(0, self.db.execute("SELECT count(*) FROM intelligence_snapshots").fetchone()[0])

    def test_planning_seeks_a_bounded_indexed_window_of_large_history(self):
        with self.db:
            self.db.executemany("INSERT INTO intelligence_debt VALUES(?,?,?,?,?,?,?,?)",
                ((f"old-{i:06}", "HINDSIGHT_POSITIVE_REVIEW", f"token-{i}", "{}", i+1, i+1, 1, 1) for i in range(20000)))
        steps = 0
        def guard():
            nonlocal steps
            steps += 100
            return steps > 20000
        self.db.set_progress_handler(guard, 100)
        try:
            first = self.coverage.propose_round()
            self.assertLessEqual(steps, 20000)
        finally: self.db.set_progress_handler(None, 0)
        second = IntelligenceCoverage(SimpleNamespace(db=self.db)).propose_round()
        a = {j["debt_id"] for j in first if "debt_id" in j}
        b = {j["debt_id"] for j in second if "debt_id" in j}
        self.assertEqual(6, len(a))
        self.assertTrue(a.isdisjoint(b))

    def test_old_schema_migrates_without_losing_history(self):
        db = sqlite3.connect(":memory:")
        try:
            db.execute("CREATE TABLE intelligence_debt(id TEXT PRIMARY KEY,kind TEXT,subject TEXT,evidence TEXT,first_seen REAL,last_seen REAL,observations INTEGER)")
            db.execute("INSERT INTO intelligence_debt VALUES('historic','SOURCE_HEALTH','historic','{}',1,2,3)")
            report = IntelligenceCoverage(SimpleNamespace(db=db)).report()
            self.assertEqual(1, report["debt_records"])
            self.assertEqual(1, report["active_debt_records"])
        finally: db.close()


class OpportunityTests(unittest.TestCase):
    def test_small_positive_peaks_are_reviewed_without_inventing_executable_profit(self):
        census = LaunchCensus()
        for mint in ("small", "monster", "entered", "unknown", "loss"):
            census.see(mint, at=1)
        census.screen("small", "holder_concentration")
        census.resolve("small", peak_multiple=1.1)
        census.resolve("monster", peak_multiple=20)
        census.resolve("entered", peak_multiple=5)
        census.enter("entered")
        census.resolve("loss", peak_multiple=.9)
        before = census.state()
        report = census.positive_opportunity_review()
        self.assertEqual(before, census.state())
        self.assertEqual(2, report["retained_positive_not_entered"])
        self.assertEqual(1, report["unresolved_retained"])
        self.assertEqual(["monster", "small"], [r["token"] for r in report["review_candidates"]])
        self.assertTrue(all(r["executable_after_cost_profit"] is None for r in report["review_candidates"]))
        self.assertTrue(census.positive_opportunity_review(1)["detail_truncated"])
        with self.assertRaises(ValueError): census.positive_opportunity_review(0)


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_main_endpoint_to_consumer_retains_actual_health_and_misses(self):
        census = LaunchCensus()
        census.see("token", at=1)
        census.resolve("token", peak_multiple=1.2)
        surface = ReportingSurface()
        surface.launch_census = census
        surface.source_mesh = SimpleNamespace(sources=[SimpleNamespace(health=lambda: SourceHealth("public:test", SourceState.OK))])
        response = await surface._intelligence_coverage_endpoint(None)
        db = sqlite3.connect(":memory:")
        try:
            consumer = IntelligenceCoverage(SimpleNamespace(db=db))
            consumer.ingest(json.loads(response.text))
            report = consumer.report()
            self.assertTrue(report["fresh_main_snapshot"])
            self.assertEqual("public:test", report["dynamic_source_rows"][0]["source_id"])
            self.assertEqual(1, report["debt_records"])
        finally: db.close()

    async def test_health_detects_dead_coverage_worker(self):
        async def worker(*args): await asyncio.Event().wait()
        with tempfile.TemporaryDirectory() as temp, \
             patch("src.runtime.market_gateway.PredictionPaperRunner.run", worker), \
             patch("src.runtime.market_gateway.PublicWalletResearch.run", worker), \
             patch("src.runtime.market_gateway.IntelligenceCoverage.run", worker):
            broker = PaperBroker(Path(temp) / "paper.sqlite")
            app = make_app(broker, token="a" * 32)
            async with TestClient(TestServer(app)) as client:
                app[COVERAGE_TASK].cancel()
                await asyncio.gather(app[COVERAGE_TASK], return_exceptions=True)
                response = await client.get("/health", headers={"Host": "127.0.0.1:18082"})
                self.assertEqual(503, response.status)
                self.assertFalse((await response.json())["components"]["coverage"])

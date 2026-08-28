import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.research.descendant_generator import generate_descendants
from src.research.trade_evidence import TradeEvidenceLedger, evidence_packet
from src.strategies.memecoin_state import (
    DevEvent, DevWalletMonitor, HolderSnapshot, HolderTrajectoryMonitor,
    RotationTrade, SmartWalletRotationTracker, social_price_disagreement,
)
from src.strategies.profit_sweeper import ProfitIsolationPolicy
from src.strategies.risk_veto import RiskVeto


class TestHolderTrajectory(unittest.TestCase):
    def test_levels_and_trajectory_remain_distinct(self):
        monitor = HolderTrajectoryMonitor()
        self.assertEqual(monitor.state("mint")["status"], "DATA_BLOCKED")
        monitor.record("mint", HolderSnapshot(
            timestamp=10, top_10_pct=70, top_20_pct=85, unique_holders=20))
        first = monitor.state("mint", as_of=10)
        self.assertEqual(first["status"], "OK")
        self.assertEqual(first["trajectory_status"], "MEASURING")
        monitor.record("mint", HolderSnapshot(
            timestamp=20, top_10_pct=60, top_20_pct=80, unique_holders=35))
        state = monitor.state("mint", as_of=20)
        self.assertEqual(state["trajectory_status"], "OK")
        self.assertEqual(state["changes"]["top_10_pct"], -10)
        self.assertEqual(state["velocity_per_second"]["unique_holders"], 1.5)

    def test_unknown_owner_enrichment_is_not_zero(self):
        snapshot = HolderSnapshot.from_mapping({"top_10_pct": 40}, timestamp=1)
        self.assertIsNone(snapshot.dev_pct)
        self.assertIsNone(snapshot.cluster_pct)


class TestDeveloperAndRotation(unittest.TestCase):
    def test_only_registered_developer_trade_is_attributed(self):
        monitor = DevWalletMonitor()
        monitor.register("mint", "dev")
        self.assertFalse(monitor.record_trade(
            "mint", wallet="someone", side="sell", timestamp=1))
        self.assertTrue(monitor.record_trade(
            "mint", wallet="dev", side="sell", timestamp=2,
            supply_share_pct=3))
        monitor.record("mint", DevEvent(
            3, "authority_mutation", wallet="dev", severity="critical"))
        state = monitor.state("mint", as_of=3)
        self.assertEqual(state["recent_dev_sells"], 1)
        self.assertIn("authority_mutation", state["hard_vetoes"])

    def test_rotation_requires_quality_independence_and_size(self):
        tracker = SmartWalletRotationTracker()
        tracker.record(RotationTrade("a", "w0", "buy", 10, None, None, None))
        self.assertEqual(tracker.report(as_of=10)["status"], "DATA_BLOCKED")
        tracker.record(RotationTrade("a", "w1", "buy", 11, .8, .5, 100, "dogs"))
        tracker.record(RotationTrade("b", "w2", "sell", 12, .9, 1, 10, "cats"))
        report = tracker.report(as_of=12)
        self.assertEqual(report["status"], "OK")
        self.assertAlmostEqual(report["token_flow"]["a"], 40)
        self.assertEqual(report["leader"], "a")


class TestDisagreementAndVeto(unittest.TestCase):
    def test_social_price_disagreement_is_evidence_not_probability(self):
        result = social_price_disagreement(
            [{"timestamp": 1, "velocity": 1}, {"timestamp": 2, "velocity": 9}],
            [{"timestamp": 1, "price_multiple": 1},
             {"timestamp": 2, "price_multiple": 1.1}], as_of=2)
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["authority"], "research_feature_only")
        self.assertGreater(result["evidence_score"], 0)
        self.assertNotIn("probability", result)

    def test_known_authority_and_route_facts_veto_alpha(self):
        report = SimpleNamespace(
            data_status="OK", blocked_checks=[], can_mint=True, can_freeze=False,
            token_extensions=[], sell_route_feasible=False,
            checks={"sell_route": {"price_impact_pct": .01}},
            risk_level=SimpleNamespace(value="medium"), score=50)
        verdict = RiskVeto(require_complete_safety=False).evaluate(report)
        self.assertEqual(verdict.status, "VETO")
        self.assertIn("mint_authority_active", verdict.reasons)
        self.assertIn("sell_route_unavailable", verdict.reasons)

    def test_unknown_connected_owners_are_visible_but_configurable(self):
        report = SimpleNamespace(
            data_status="OK", blocked_checks=[], can_mint=False, can_freeze=False,
            token_extensions=[], sell_route_feasible=True,
            checks={"sell_route": {"price_impact_pct": .01}},
            risk_level=SimpleNamespace(value="safe"), score=100)
        permissive = RiskVeto(require_complete_safety=False).evaluate(report)
        strict = RiskVeto(require_complete_safety=True).evaluate(report)
        self.assertEqual(permissive.status, "CLEAR")
        self.assertIn("connected_holder_concentration", permissive.unmeasured)
        self.assertEqual(strict.status, "DATA_BLOCKED")

    def test_unavailable_critical_display_level_is_not_false_rug_evidence(self):
        report = SimpleNamespace(
            data_status="DATA_BLOCKED", blocked_checks=["mint_account"],
            can_mint=False, can_freeze=False, token_extensions=[],
            sell_route_feasible=None, checks={},
            risk_level=SimpleNamespace(value="critical"), score=0)
        verdict = RiskVeto(require_complete_safety=True).evaluate(report)
        self.assertEqual(verdict.status, "DATA_BLOCKED")
        self.assertNotIn("native_risk_level:critical", verdict.reasons)
        self.assertIn("mint_account", verdict.unmeasured)


class TestEvidenceAndIsolation(unittest.TestCase):
    def test_hash_chain_detects_rewrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.jsonl"
            ledger = TradeEvidenceLedger(path)
            ledger.record("candidate", {"mint": "a"}, timestamp=1)
            ledger.record("decision", {"decision": "reject"}, timestamp=2)
            self.assertEqual(TradeEvidenceLedger.verify(path), (True, "OK", 2))
            rows = path.read_text().splitlines()
            changed = json.loads(rows[0])
            changed["payload"]["mint"] = "b"
            rows[0] = json.dumps(changed)
            path.write_text("\n".join(rows) + "\n")
            ok, reason, verified = TradeEvidenceLedger.verify(path)
            self.assertFalse(ok)
            self.assertEqual(reason, "row hash mismatch")
            self.assertEqual(verified, 0)

    def test_evidence_packet_has_every_declared_section(self):
        packet = evidence_packet(
            mint="m", timestamp=1, bonding_curve={}, liquidity={}, sellability={},
            authorities={}, holder_distribution={}, wallet_clusters={}, dev_wallet={},
            smart_wallet_flow={}, social_velocity={}, entry_cost={}, exit_liquidity={},
            risk_vetoes=[], expected_edge=None, position_size=None, exit_plan={},
            decision="reject")
        self.assertEqual(packet["mint"], "m")
        self.assertIn("holder_distribution", packet)
        self.assertIn("exit_plan", packet)

    def test_profit_isolation_never_creates_a_transaction(self):
        policy = ProfitIsolationPolicy(working_capital_usd=10_000,
                                       sweep_trigger_usd=1_000)
        self.assertEqual(policy.plan(equity_usd=10_500)["status"], "HOLD")
        blocked = policy.plan(equity_usd=12_000)
        self.assertEqual(blocked["status"], "DATA_BLOCKED")
        planned = policy.plan(equity_usd=12_000, cold_destination="cold", dry_run=True)
        self.assertEqual(planned["status"], "PAPER_PLAN")
        self.assertFalse(planned["transaction_created"])


class TestDescendants(unittest.TestCase):
    def test_only_measured_failures_generate_descendants(self):
        leak = SimpleNamespace(value="execution_miss")
        finding = SimpleNamespace(
            leak=leak, forgone_log_growth=.2,
            evidence={"failure_reason": "landing_timeout"})
        report = generate_descendants([finding])
        self.assertEqual(report["status"], "OK")
        hypothesis = report["hypotheses"][0]
        self.assertEqual(hypothesis["validation"], "chronological_out_of_sample")
        self.assertEqual(hypothesis["authority"], "none_until_promoted")


if __name__ == "__main__":
    unittest.main()

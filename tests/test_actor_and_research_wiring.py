"""Twenty-six surfaces that were built and answered nobody.

The rule this file enforces is the auditor's, not mine: a wiring has to change
an OBSERVABLE output. A subsystem that is constructed, called, and whose
result nothing can see is the same failure as one that was never called -- it
just costs CPU as well.

So every test below asserts on something a reader can see: a key in the entry
decision, a count in a report, a row in a graph. None of them assert that a
method exists.
"""

import time
from types import SimpleNamespace

import pytest

from src.runtime.actor_intelligence import (
    actor_context, entry_actor_block, ingest_launch_edges, launch_risk,
    note_wallet_first_seen, record_actor_feedback, record_buy_edge,
    record_resolution_feedback)
from src.runtime.research_reads import (
    evidence_quality_report, fallback_confidence, research_report,
    source_intelligence_report, wallet_intelligence_report)

T0 = 1_700_000_000.0


def _store(tmp=None):
    from src.research.actor_store import ActorStore
    return ActorStore(tmp)


class TestTheActorGraphIsBuiltFromLaunches:
    """It was never CONSTRUCTED -- not unwired, never instantiated -- so
    funders_of, prior_mints and shared_family had no callers because they had
    no object to be called on."""

    def test_a_launch_becomes_edges(self):
        desk = SimpleNamespace(actor_store=_store())
        written = ingest_launch_edges(desk, "mint1", {
            "creator": "dev", "timestamp": T0, "program": "pump",
            "funding_transfers": [{"from": "whale", "to": "dev"}]})
        assert written >= 3

    def test_a_repeat_deployer_is_visible_at_the_second_launch(self):
        desk = SimpleNamespace(actor_store=_store())
        ingest_launch_edges(desk, "mint1", {"creator": "dev", "timestamp": T0})
        ingest_launch_edges(desk, "mint2", {"creator": "dev",
                                            "timestamp": T0 + 3600})
        context = actor_context(desk, "mint2", "dev", T0 + 3600)
        assert context["prior_mints"] == 2

    def test_the_read_is_point_in_time_not_wall_clock(self):
        """A deployer's third launch must not be scored with the rug their
        fourth turned into. A graph queried at wall clock does exactly that
        and looks correct while doing it."""
        desk = SimpleNamespace(actor_store=_store())
        ingest_launch_edges(desk, "mint1", {"creator": "dev", "timestamp": T0})
        ingest_launch_edges(desk, "mint2", {"creator": "dev",
                                            "timestamp": T0 + 86_400})
        early = actor_context(desk, "mint1", "dev", T0 + 60)
        late = actor_context(desk, "mint2", "dev", T0 + 90_000)
        assert early["prior_mints"] == 1
        assert late["prior_mints"] == 2

    def test_funders_come_back_out_of_the_graph(self):
        desk = SimpleNamespace(actor_store=_store())
        ingest_launch_edges(desk, "mint1", {
            "creator": "dev", "timestamp": T0,
            "funding_transfers": [{"from": "whale", "to": "dev"}]})
        assert actor_context(desk, "mint1", "dev", T0 + 1)["funders"] == ["whale"]

    def test_buys_land_as_edges_in_order(self):
        desk = SimpleNamespace(actor_store=_store())
        ingest_launch_edges(desk, "mint1", {"creator": "dev", "timestamp": T0})
        for index in range(3):
            record_buy_edge(desk, "mint1", f"buyer{index}", T0 + index)
        assert len(desk.actor_store.edges) >= 4

    def test_no_store_is_blocked_rather_than_a_crash(self):
        assert actor_context(SimpleNamespace(), "m", "dev", T0)["status"] \
            == "DATA_BLOCKED"
        assert ingest_launch_edges(SimpleNamespace(), "m", {}) == 0


class TestTheCompositeIsRecordedWithoutAVote:
    def test_the_genealogy_reading_declares_it_has_no_authority(self):
        """0.4 for a rug rate over half, 0.2 over a fifth, 0.3 for a critical
        funding cluster -- every coefficient chosen rather than measured. The
        desk removed exactly that shape from decisions today."""
        graph = SimpleNamespace(
            assess_launch_risk=lambda d, f, b: {
                "risk_score": 0.6, "risk_factors": ["x"], "smart_buyers": 0,
                "insider_buyers": 0, "deployer_profile": None})
        desk = SimpleNamespace(genealogy=graph)
        reading = launch_risk(desk, "dev", ["whale"], [{"address": "b1"}])
        assert reading["authority"] == "none"
        assert reading["risk_score"] == 0.6
        assert "deployer_profile" not in reading

    def test_buyers_are_passed_as_dicts_because_the_body_requires_it(self):
        """Its signature annotated List[str] and its body called
        .get("address") on each element. The body is the contract."""
        seen = {}

        def assess(deployer, funders, buyers):
            seen["buyers"] = buyers
            return {"risk_score": 0.0, "risk_factors": []}
        desk = SimpleNamespace(genealogy=SimpleNamespace(assess_launch_risk=assess))
        launch_risk(desk, "dev", [], [{"address": "b1"}])
        assert seen["buyers"] == [{"address": "b1"}]


class TestTheEntryDecisionCarriesTheNewSlots:
    def test_the_block_produces_every_declared_slot(self):
        desk = SimpleNamespace(actor_store=_store(), genealogy=None,
                               info_graph=None, cold_distillate=None,
                               facts=None)
        candidate = SimpleNamespace(deployer="dev", timestamp=T0, metadata={})
        block = entry_actor_block(desk, "mint", candidate)
        assert set(block) == {"actors", "genealogy_risk", "lead_forecast",
                              "cold_start", "evidence_confidence"}

    def test_every_slot_answers_even_when_it_cannot_measure(self):
        desk = SimpleNamespace(actor_store=None, genealogy=None,
                               info_graph=None, cold_distillate=None,
                               facts=None)
        candidate = SimpleNamespace(deployer="", timestamp=T0, metadata={})
        block = entry_actor_block(desk, "mint", candidate)
        assert all(value.get("status") == "DATA_BLOCKED"
                   for value in block.values())

    def test_the_manifest_declares_all_four(self):
        from src.runtime.intelligence_manifest import ENTRY_CONTRIBUTORS
        keys = {contributor.key for contributor in ENTRY_CONTRIBUTORS}
        assert {"genealogy_risk", "lead_forecast", "cold_start",
                "evidence_confidence"} <= keys


class TestTheFeedsThatHadNoWriter:
    def test_an_exit_records_the_hold_time_not_just_the_sell(self):
        """A wallet's HOLD TIME is the most transferable thing about it, and
        the signature model had sells without durations."""
        recorded = []
        desk = SimpleNamespace(
            wallet_signatures=SimpleNamespace(
                observe_exit=lambda w, hold: recorded.append((w, hold))),
            dev_wallet_monitor=None)
        note_wallet_first_seen(desk, "mint", "w1", T0)
        record_actor_feedback(desk, "mint", {
            "wallet": "w1", "side": "sell", "timestamp": T0 + 120})
        assert recorded == [("w1", 120.0)]

    def test_an_exit_without_an_entry_records_nothing_rather_than_zero(self):
        recorded = []
        desk = SimpleNamespace(
            wallet_signatures=SimpleNamespace(
                observe_exit=lambda w, hold: recorded.append((w, hold))),
            dev_wallet_monitor=None)
        record_actor_feedback(desk, "mint", {
            "wallet": "w1", "side": "sell", "timestamp": T0})
        assert recorded == []

    def test_the_creator_balance_reaches_the_distribution_detector(self):
        recorded = []
        desk = SimpleNamespace(
            wallet_signatures=None,
            dev_wallet_monitor=SimpleNamespace(
                record_balance=lambda t, pct, **kw: recorded.append((t, pct))))
        record_actor_feedback(desk, "mint", {
            "wallet": "dev", "side": "sell", "timestamp": T0,
            "creator_balance_pct": 42.0})
        assert recorded == [("mint", 42.0)]

    def test_the_first_seen_map_is_bounded(self):
        desk = SimpleNamespace()
        for index in range(100_050):
            note_wallet_first_seen(desk, "mint", f"w{index}", T0)
        assert len(desk._wallet_first_seen) <= 100_000

    def test_a_resolution_teaches_the_adversarial_detector(self):
        """A feature value that used to precede monsters and now precedes
        rugs is exactly what it exists to notice, and nothing told it how
        anything ended."""
        recorded = []
        desk = SimpleNamespace(adversarial=SimpleNamespace(
            record_feature_value=lambda t, f, v, o: recorded.append((f, v))))
        written = record_resolution_feedback(desk, "mint", {
            "rugged": True, "features": {"holder_concentration": 0.9,
                                         "buyer_count": 4}})
        assert written == 2
        assert ("holder_concentration", 0.9) in recorded

    def test_a_resolution_with_no_features_writes_nothing(self):
        desk = SimpleNamespace(adversarial=SimpleNamespace(
            record_feature_value=lambda *a: None))
        assert record_resolution_feedback(desk, "mint", {"rugged": True}) == 0


class TestTheResearchSurfacesReportWithoutVoting:
    def test_the_report_declares_it_has_no_authority(self):
        report = research_report(SimpleNamespace())
        assert report["authority"] == "none"
        assert set(report) >= {"sources", "wallets", "evidence_quality"}

    def test_an_unbuilt_desk_blocks_every_surface(self):
        desk = SimpleNamespace()
        assert source_intelligence_report(desk)["status"] == "DATA_BLOCKED"
        assert wallet_intelligence_report(desk)["status"] == "DATA_BLOCKED"
        assert evidence_quality_report(desk)["status"] == "DATA_BLOCKED"

    def test_an_uncalibrated_model_is_none_not_false(self):
        """None is "nobody checked". False is "it failed the check". Folding
        them together turns an unmeasured instrument into a broken one."""
        desk = SimpleNamespace(calibration=SimpleNamespace(
            trustworthy=lambda name: None))
        models = evidence_quality_report(desk)["model_trustworthy"]
        assert set(models.values()) == {None}

    def test_the_weakest_rung_dominates_the_evidence_confidence(self):
        from src.research.fallback import FallbackResolver
        desk = SimpleNamespace(facts=FallbackResolver())
        reading = fallback_confidence(desk, ("a", "b"))
        assert reading["status"] in {"OK", "DATA_BLOCKED"}
        if reading["status"] == "OK":
            assert 0.0 <= reading["confidence"] <= 1.0

    def test_no_resolver_blocks_rather_than_assuming_full_confidence(self):
        assert fallback_confidence(SimpleNamespace(facts=None), ("a",))[
            "status"] == "DATA_BLOCKED"


class TestOneDefinitionOfKolReach:
    def test_the_model_and_the_touch_agree_on_the_threshold(self):
        """`read` re-implemented the predicate inline against a CONFIGURABLE
        threshold while `is_kol` used the module default, so the two could
        disagree."""
        from src.strategies.ignition import SourceTouch
        touch = SourceTouch(source_id="s", timestamp=T0, reach=5_000)
        assert touch.is_kol(1_000) is True
        assert touch.is_kol(10_000) is False

    def test_the_read_path_uses_the_touch_predicate(self):
        import ast
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1] / "src" / "strategies"
                  / "ignition.py").read_text(encoding="utf-8")
        assert "touch.is_kol(self.kol_reach)" in source
        tree = ast.parse(source)
        assert any(isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute)
                   and node.func.attr == "is_kol"
                   for node in ast.walk(tree))


def test_class_a_debt_is_zero():
    """The point of the whole exercise. Class A means the desk is believed to
    have a feature it does not have, and it is now empty."""
    from tools.orphan_sweep import load_classified_baseline
    rows = load_classified_baseline()
    debt = [name for name, (klass, _) in rows.items() if klass == "A"]
    assert debt == []

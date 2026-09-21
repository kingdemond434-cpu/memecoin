"""One reason, not seven symptoms.

The promotion gate reporting

    monster_enrichment was not measured; required >= 2.0
    net_log_growth was not measured
    rug_loss_share was not measured; required <= 0.2

reads like three problems. It is one problem, three levels down: nothing was
entered, because every entry was DATA_BLOCKED on the prediction, because no
model artifact existed, because nothing ever ran the trainer. Three of those
levels are things nobody can act on and the fourth is a half-hour fix -- and
a report listing the symptoms sends an operator to tune thresholds.
"""

from __future__ import annotations

import unittest

from src.research.readiness import DEFAULT_CHAIN, diagnose


def _facts(**overrides):
    base = {
        "launches_seen": 6754,
        "resolved_episodes": 3000,
        "training_rounds": 4,
        "model_trained": True,
        "cost_model_ok": True,
        "entries": 12,
        "real_fills": 5,
        "dry_run": False,
    }
    base.update(overrides)
    return base


class ItNamesTheDeepestUnsatisfiedLink(unittest.TestCase):

    def test_an_empty_desk_blames_the_feed_not_the_gate(self):
        result = diagnose(_facts(launches_seen=0, resolved_episodes=0,
                                 training_rounds=0, model_trained=False,
                                 cost_model_ok=False, entries=0, real_fills=0))
        self.assertEqual("stream", result.blocked_at)
        self.assertIn("not delivering", result.reason)

    def test_a_streaming_desk_with_no_training_blames_the_trainer(self):
        # Not the model, not the gate: the trainer had no caller.
        result = diagnose(_facts(training_rounds=0, model_trained=False,
                                 entries=0, real_fills=0))
        self.assertEqual("training", result.blocked_at)
        self.assertIn("no caller", result.reason)
        self.assertEqual(["stream", "episodes"], result.satisfied)

    def test_training_that_ran_but_never_passed_blames_the_model(self):
        result = diagnose(_facts(model_trained=False, entries=0, real_fills=0))
        self.assertEqual("model", result.blocked_at)
        self.assertIn("trainer's own report", result.reason)

    def test_unpriceable_cost_blames_costing_not_the_entry_bar(self):
        # No expected value can be computed net of cost, so nothing clears
        # an entry bar -- and "nothing entered" is the symptom of that.
        result = diagnose(_facts(cost_model_ok=False, entries=0, real_fills=0))
        self.assertEqual("costing", result.blocked_at)
        self.assertIn("FeeConfig", result.reason)

    def test_no_entries_says_the_criteria_are_unmeasurable_not_failing(self):
        result = diagnose(_facts(entries=0, real_fills=0))
        self.assertEqual("entries", result.blocked_at)
        self.assertIn("unmeasurable by", result.reason)

    def test_dry_run_with_no_fills_is_expected_and_says_so(self):
        result = diagnose(_facts(real_fills=0, dry_run=True))
        self.assertEqual("fills", result.blocked_at)
        self.assertIn("expected and correct", result.reason)

    def test_a_complete_chain_is_not_blocked(self):
        result = diagnose(_facts())
        self.assertTrue(result.ready)
        self.assertIsNone(result.blocked_at)
        self.assertEqual([link.name for link in DEFAULT_CHAIN], result.satisfied)
        self.assertEqual("OK", result.to_dict()["status"])


class ItReportsOneReasonAndTheTrailToIt(unittest.TestCase):

    def test_the_links_above_the_blockage_are_pending_not_failing(self):
        result = diagnose(_facts(training_rounds=0, model_trained=False,
                                 entries=0, real_fills=0))
        self.assertIn("training", result.pending)
        self.assertIn("entries", result.pending)
        self.assertNotIn("stream", result.pending)

    def test_the_facts_it_judged_on_are_carried(self):
        # So a disagreement about the verdict is settled by reading the
        # numbers rather than by rerunning the desk.
        result = diagnose(_facts(entries=0, real_fills=0))
        self.assertEqual(6754, result.to_dict()["facts"]["launches_seen"])

    def test_a_misbehaving_fact_provider_blocks_rather_than_passes(self):
        # Failing open here would report a desk as ready because a lookup
        # raised, which is the one direction this must never fail.
        class _Explodes(dict):
            def get(self, key, default=None):
                raise RuntimeError("boom")

        result = diagnose(_Explodes())
        self.assertFalse(result.ready)
        self.assertEqual("stream", result.blocked_at)


class TheDeskAnswersFromItsOwnFacts(unittest.IsolatedAsyncioTestCase):

    async def test_a_fresh_offline_desk_is_blocked_at_the_stream(self):
        from src.main import MemecoinQuantDesk

        desk = MemecoinQuantDesk("config/chains.yaml", dry_run_override=True,
                                 offline=True)
        await desk.initialize()
        result = desk._promotion_readiness().to_dict()
        self.assertEqual("BLOCKED", result["status"])
        self.assertEqual("stream", result["blocked_at"])

    async def test_it_reaches_the_status_report(self):
        from src.main import MemecoinQuantDesk

        desk = MemecoinQuantDesk("config/chains.yaml", dry_run_override=True,
                                 offline=True)
        await desk.initialize()
        self.assertIn("promotion_blocked_at", desk.readiness())


if __name__ == "__main__":
    unittest.main()

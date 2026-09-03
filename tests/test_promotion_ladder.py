"""The ladder, made load-bearing.

`PromotionLedger` and `can_advance` had zero callers anywhere in the
codebase. The gate computed a verdict, printed it in /status, and gated
nothing: the only thing between the desk and trading on an untrained model
was an operator remembering not to flip a flag.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.research.promotion_gate import (
    DEFAULT_CRITERIA, STAGE_ORDER, Evidence, PromotionLedger, Stage)


class _Fixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ledger = PromotionLedger(Path(self._tmp.name) / "promotion.jsonl")

    def _backdate(self, days: float = 60.0) -> None:
        """Pretend the desk arrived at this rung `days` ago.

        Dwell time is now enforced, so a test that wants to exercise a
        HIGHER rung has to buy the lower one honestly -- and the only thing
        it cannot wait for is the clock.
        """
        if not self.ledger._stage_path.exists():
            return
        record = json.loads(self.ledger._stage_path.read_text())
        record["at"] = record["at"] - days * 86_400
        record["decisions_at_entry"] = 0
        self.ledger._stage_path.write_text(json.dumps(record))

    def _climb(self, target: Stage, limit: int = 12) -> None:
        """Climb to `target`, backdating each rung. Bounded, never a `while`.

        An unbounded loop here spun forever the moment dwell time became
        real: the condition it waited on was one the ladder would never
        satisfy inside a test.
        """
        for _ in range(limit):
            if self.ledger.current_stage() is target:
                return
            self._backdate()
            self.ledger.submit(self._passing(self.ledger.current_stage()))
        self.fail(f"could not reach {target} in {limit} submissions")

    def _passing(self, stage: Stage) -> Evidence:
        criteria = DEFAULT_CRITERIA[stage]
        return Evidence(
            stage=stage,
            decisions=criteria.min_decisions + 1,
            real_fills=criteria.min_real_fills + 1,
            launch_cohorts=criteria.min_launch_cohorts + 1,
            regimes_covered=criteria.min_regimes + 1,
            net_log_growth=0.05,
            rug_loss_share=0.01,
            monster_enrichment=criteria.min_monster_enrichment + 1.0,
            execution_success=0.95,
            catastrophic_failures=0,
            # Added when the two money stages started requiring them. A book
            # that clears every count and every mean can still be
            # indistinguishable from a book with no edge, so CANARY and LIVE
            # ask for the lower bound and for mechanisms that survived the
            # gauntlet.
            net_log_growth_lower_bound=0.01,
            gauntlet_survivors=criteria.min_gauntlet_survivors,
        )

    def _passing_without_the_lower_bound(self, stage: Stage) -> Evidence:
        evidence = self._passing(stage)
        evidence.net_log_growth_lower_bound = None
        return evidence


class ItStartsUnauthorisedAndStaysThatWay(_Fixture):

    def test_a_fresh_ledger_is_at_the_first_stage(self):
        self.assertIs(STAGE_ORDER[0], self.ledger.current_stage())

    def test_a_fresh_ledger_authorises_nothing(self):
        authorised, reason = self.ledger.authorises_live_capital()
        self.assertFalse(authorised)
        self.assertIn("canary", reason)

    def test_a_missing_stage_file_reads_as_the_first_stage_not_the_last(self):
        # The direction of this default is the entire point.
        self.assertFalse(self.ledger._stage_path.exists())
        self.assertIs(STAGE_ORDER[0], self.ledger.current_stage())

    def test_a_corrupt_stage_file_never_reads_as_authorisation(self):
        self.ledger._stage_path.parent.mkdir(parents=True, exist_ok=True)
        self.ledger._stage_path.write_text("{not json")
        self.assertIs(STAGE_ORDER[0], self.ledger.current_stage())
        self.assertFalse(self.ledger.authorises_live_capital()[0])

    def test_an_unknown_stage_name_is_refused(self):
        self.ledger._stage_path.parent.mkdir(parents=True, exist_ok=True)
        self.ledger._stage_path.write_text(json.dumps({"stage": "god_mode"}))
        self.assertIs(STAGE_ORDER[0], self.ledger.current_stage())


class ItAdvancesExactlyOneStagePerPass(_Fixture):

    def test_a_pass_advances_one_stage(self):
        start = self.ledger.current_stage()
        verdict = self.ledger.submit(self._passing(start))
        self.assertTrue(verdict.passed, verdict.failures)
        self.assertIs(STAGE_ORDER[1], self.ledger.current_stage())

    def test_overwhelming_evidence_still_advances_only_one(self):
        # Each stage buys evidence the previous one could not; skipping one
        # means promoting on evidence that was never gathered.
        start = self.ledger.current_stage()
        huge = self._passing(Stage.LIVE)
        huge.stage = start
        self.ledger.submit(huge)
        self.assertIs(STAGE_ORDER[1], self.ledger.current_stage())

    def test_a_failure_does_not_advance(self):
        start = self.ledger.current_stage()
        thin = Evidence(stage=start, decisions=1)
        verdict = self.ledger.submit(thin)
        self.assertFalse(verdict.passed)
        self.assertIs(start, self.ledger.current_stage())

    def test_the_whole_ladder_can_be_climbed(self):
        # HISTORICAL -> CHRONOLOGICAL_OOS -> FORWARD_SHADOW -> CANARY -> LIVE
        for expected in STAGE_ORDER[1:]:
            stage = self.ledger.current_stage()
            self._backdate()
            self.ledger.submit(self._passing(stage))
            self.assertIs(expected, self.ledger.current_stage())
        self.assertIs(Stage.LIVE, self.ledger.current_stage())

    def test_live_is_the_top_and_does_not_wrap(self):
        self._climb(Stage.LIVE)
        self._backdate()
        self.ledger.submit(self._passing(Stage.LIVE))
        self.assertIs(Stage.LIVE, self.ledger.current_stage())


class CanaryIsWhereMoneyStarts(_Fixture):


    def test_forward_shadow_still_authorises_nothing(self):
        self._climb(Stage.FORWARD_SHADOW)
        self.assertFalse(self.ledger.authorises_live_capital()[0])

    def test_canary_authorises_live_capital(self):
        self._climb(Stage.CANARY)
        authorised, reason = self.ledger.authorises_live_capital()
        self.assertTrue(authorised)
        self.assertIn("canary", reason)

    def test_the_stage_survives_a_restart(self):
        # A stage that resets on restart is a desk that silently returns to
        # trading without authorisation -- or forgets it earned the right.
        self._climb(Stage.CANARY)
        reopened = PromotionLedger(self.ledger.path)
        self.assertIs(Stage.CANARY, reopened.current_stage())
        self.assertTrue(reopened.authorises_live_capital()[0])

    def test_a_demotion_drops_exactly_one_stage(self):
        self._climb(Stage.CANARY)
        self.assertIs(Stage.FORWARD_SHADOW, self.ledger.demote("misbehaving"))
        self.assertFalse(self.ledger.authorises_live_capital()[0])

    def test_a_demotion_cannot_fall_off_the_bottom(self):
        for _ in range(5):
            self.ledger.demote("test")
        self.assertIs(STAGE_ORDER[0], self.ledger.current_stage())

    def test_every_verdict_is_kept_including_the_failures(self):
        self.ledger.submit(Evidence(stage=self.ledger.current_stage(), decisions=1))
        self.ledger.submit(self._passing(self.ledger.current_stage()))
        self.assertEqual(2, len(self.ledger.history()))


class TheExecutionPathRefusesWithoutAnEarnedStage(unittest.TestCase):
    """`dry_run` is an operator's intent. Intent is not evidence."""

    def _engine(self, dry_run: bool, ledger=None):
        from src.execution.jupiter_jito import ExecutionEngine

        engine = ExecutionEngine.__new__(ExecutionEngine)
        engine.dry_run = dry_run
        engine.promotion_ledger = ledger
        engine.simulation_reasons = {}
        return engine

    def test_no_ledger_means_not_authorised(self):
        # Fails CLOSED. A bug in the check must never be the reason money
        # moves.
        engine = self._engine(dry_run=False, ledger=None)
        self.assertFalse(engine.live_capital_authorised()[0])
        self.assertIn("no promotion ledger", engine._submission_blocked())

    def test_dry_run_blocks_even_with_an_authorised_ledger(self):
        class _Authorised:
            @staticmethod
            def authorises_live_capital():
                return True, "earned"

        engine = self._engine(dry_run=True, ledger=_Authorised())
        self.assertEqual("dry_run", engine._submission_blocked())

    def test_an_unearned_stage_blocks_even_with_dry_run_off(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = PromotionLedger(Path(directory) / "p.jsonl")
            engine = self._engine(dry_run=False, ledger=ledger)
            blocked = engine._submission_blocked()
            self.assertIsNotNone(blocked)
            self.assertIn("canary", blocked)

    def test_both_conditions_together_allow_submission(self):
        class _Authorised:
            @staticmethod
            def authorises_live_capital():
                return True, "earned"

        engine = self._engine(dry_run=False, ledger=_Authorised())
        self.assertIsNone(engine._submission_blocked())

    def test_a_ledger_that_raises_is_not_authorisation(self):
        class _Broken:
            @staticmethod
            def authorises_live_capital():
                raise RuntimeError("disk gone")

        engine = self._engine(dry_run=False, ledger=_Broken())
        self.assertFalse(engine.live_capital_authorised()[0])
        self.assertIn("unreadable", engine._submission_blocked())


class TheDeskClimbsItsOwnLadder(unittest.IsolatedAsyncioTestCase):

    async def test_the_ledger_is_attached_to_the_execution_engine(self):
        from src.main import MemecoinQuantDesk

        desk = MemecoinQuantDesk("config/chains.yaml", dry_run_override=True,
                                 offline=True)
        await desk.initialize()
        self.assertIsNotNone(desk.promotion_ledger)
        self.assertIs(desk.promotion_ledger,
                      desk.execution_engine.promotion_ledger)

    async def test_the_evidence_accumulator_tracks_the_earned_stage(self):
        # It was pinned to FORWARD_SHADOW by a constructor default that
        # nothing ever changed, so the ladder could not have been climbed
        # even in principle.
        from src.main import MemecoinQuantDesk

        desk = MemecoinQuantDesk("config/chains.yaml", dry_run_override=True,
                                 offline=True)
        await desk.initialize()
        self.assertIs(desk.promotion_ledger.current_stage(),
                      desk.forward_evidence.stage)

    async def test_the_report_says_whether_capital_is_authorised(self):
        from src.main import MemecoinQuantDesk

        desk = MemecoinQuantDesk("config/chains.yaml", dry_run_override=True,
                                 offline=True)
        await desk.initialize()
        promotion = desk.readiness()["promotion"]
        self.assertIn("earned_stage", promotion)
        self.assertFalse(promotion["authorises_live_capital"])


if __name__ == "__main__":
    unittest.main()


class AnArmedDeskStillBootsWithoutAKey(unittest.IsolatedAsyncioTestCase):
    """`dry_run: false` used to mean "require a private key at startup".

    That raises before the health server binds, so an armed desk left
    without a key would crash-loop -- losing exactly the unbackfillable
    forward evidence the ladder is waiting for. The key requirement follows
    the EARNED stage now, not the flag.
    """

    async def test_armed_and_unearned_boots_on_a_paper_wallet(self):
        import os

        from src.main import MemecoinQuantDesk

        previous = os.environ.pop("SOLANA_PRIVATE_KEY", None)
        self.addCleanup(
            lambda: os.environ.__setitem__("SOLANA_PRIVATE_KEY", previous)
            if previous is not None else None)
        desk = MemecoinQuantDesk("config/chains.yaml", offline=True)
        await desk.initialize()
        self.assertIsNotNone(desk.keypair)
        self.assertFalse(desk.execution_engine.live_capital_authorised()[0])

    async def test_the_shipped_config_is_armed_but_cannot_spend(self):
        # Armed means "will begin trading when it climbs to canary", not
        # "is trading". Both halves of that need to be true in the file
        # somebody actually deploys.
        import yaml

        from src.research.promotion_gate import PromotionLedger, Stage

        with open("config/chains.yaml", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        section = config.get("global") or config
        self.assertFalse(section.get("dry_run", True),
                         "config is not armed; the ladder governs nothing")
        with tempfile.TemporaryDirectory() as directory:
            fresh = PromotionLedger(Path(directory) / "p.jsonl")
            self.assertIs(Stage.HISTORICAL, fresh.current_stage())
            self.assertFalse(fresh.authorises_live_capital()[0])


class TheLadderCannotBeClimbedInOneAfternoon(_Fixture):
    """Measured on this repository: HISTORICAL to CANARY in three sweeps.

    The evidence pool is cumulative, so a desk with 70,000 banked shadow
    decisions satisfied every rung's volume gate simultaneously and reached
    authority to spend real money in about four minutes, having observed
    nothing forward at all. The docstring said each stage buys evidence the
    previous one could not; nothing enforced it.
    """

    def _banked(self, stage):
        # The real numbers from a desk that has been observing for weeks.
        return Evidence(
            stage=stage, decisions=70_612, real_fills=0,
            launch_cohorts=10_121, regimes_covered=4,
            net_log_growth=0.03, rug_loss_share=0.10,
            monster_enrichment=2.4, execution_success=0.9,
            catastrophic_failures=0)

    def test_banked_evidence_does_not_buy_the_forward_stage(self):
        for _ in range(6):
            stage = self.ledger.current_stage()
            self.ledger.submit(self._banked(stage))
            if self.ledger.current_stage() is stage:
                break
        self.assertIs(Stage.FORWARD_SHADOW, self.ledger.current_stage())
        self.assertFalse(self.ledger.authorises_live_capital()[0])

    def test_the_refusal_says_it_is_about_elapsed_observation(self):
        while self.ledger.current_stage() is not Stage.FORWARD_SHADOW:
            self.ledger.submit(self._banked(self.ledger.current_stage()))
        verdict = self.ledger.submit(self._banked(Stage.FORWARD_SHADOW))
        self.assertFalse(verdict.passed)
        joined = " ".join(verdict.failures)
        self.assertTrue("since entering" in joined or "days at" in joined,
                        joined)

    def test_fresh_decisions_at_this_stage_are_what_count(self):
        while self.ledger.current_stage() is not Stage.FORWARD_SHADOW:
            self.ledger.submit(self._banked(self.ledger.current_stage()))
        banked = self.ledger.decisions_at_entry()
        self.assertGreater(banked, 0, "arrival should snapshot the count")
        # Enough NEW decisions, but no elapsed time: still refused.
        evidence = self._banked(Stage.FORWARD_SHADOW)
        evidence.decisions = banked + 5_000
        verdict = self.ledger.submit(evidence)
        self.assertFalse(verdict.passed)
        self.assertIn("days at", " ".join(verdict.failures))

    def test_enough_time_and_fresh_evidence_does_advance(self):
        import json

        while self.ledger.current_stage() is not Stage.FORWARD_SHADOW:
            self.ledger.submit(self._banked(self.ledger.current_stage()))
        # Backdate the arrival past the dwell requirement.
        record = json.loads(self.ledger._stage_path.read_text())
        record["at"] = record["at"] - 30 * 86_400
        self.ledger._stage_path.write_text(json.dumps(record))
        evidence = self._banked(Stage.FORWARD_SHADOW)
        evidence.decisions = self.ledger.decisions_at_entry() + 5_000
        self.assertTrue(self.ledger.submit(evidence).passed)
        self.assertIs(Stage.CANARY, self.ledger.current_stage())

    def test_a_first_submission_cannot_pass_by_having_no_history(self):
        # No recorded arrival reads as arriving NOW -- the same direction as
        # a missing stage file reading as the first stage.
        while self.ledger.current_stage() is not Stage.FORWARD_SHADOW:
            self.ledger.submit(self._banked(self.ledger.current_stage()))
        self.ledger._stage_path.unlink()
        fresh = PromotionLedger(self.ledger.path)
        self.assertIs(STAGE_ORDER[0], fresh.current_stage())

    def test_the_historical_stages_are_not_slowed_down(self):
        # They are backtests and buy nothing by waiting.
        for stage in (Stage.HISTORICAL, Stage.CHRONOLOGICAL_OOS):
            self.assertEqual(0.0, DEFAULT_CRITERIA[stage].min_days_at_stage)
        for stage in (Stage.FORWARD_SHADOW, Stage.CANARY):
            self.assertGreater(DEFAULT_CRITERIA[stage].min_days_at_stage, 0.0)


class TheMoneyStagesAskForMoreThanAMean(_Fixture):
    """A point estimate is what happened. A lower bound is what to size on.

    Every volume gate in this ladder can be satisfied by a book whose mean log
    growth is +0.02 on a confidence bound of -0.31 -- that book has shown
    nothing, and before these two criteria existed it could have reached LIVE.
    """

    def test_canary_cannot_be_left_without_a_measured_lower_bound(self):
        self._climb(Stage.CANARY)
        self._backdate()
        stage = self.ledger.current_stage()
        self.ledger.submit(self._passing_without_the_lower_bound(stage))
        self.assertIs(Stage.CANARY, self.ledger.current_stage())
        verdict = self.ledger.status()
        self.assertIn("net_log_growth_lower_bound",
                      json.dumps(verdict, default=str))

    def test_a_negative_lower_bound_blocks_the_money_stages(self):
        self._climb(Stage.CANARY)
        self._backdate()
        stage = self.ledger.current_stage()
        evidence = self._passing(stage)
        evidence.net_log_growth_lower_bound = -0.31
        self.ledger.submit(evidence)
        self.assertIs(Stage.CANARY, self.ledger.current_stage())

    def test_live_wants_two_survivors_not_one(self):
        self._climb(Stage.CANARY)
        self._backdate()
        stage = self.ledger.current_stage()
        evidence = self._passing(stage)
        evidence.gauntlet_survivors = 1
        self.ledger.submit(evidence)
        # One survivor clears CANARY's bar, which asks for one.
        self.assertIs(Stage.LIVE, self.ledger.current_stage())
        # And LIVE itself asks for two: a desk whose whole book is one
        # mechanism has nothing on the day that mechanism decays.
        self.assertEqual(2, DEFAULT_CRITERIA[Stage.LIVE].min_gauntlet_survivors)

    def test_the_two_money_stages_both_demand_the_bound(self):
        for stage in (Stage.CANARY, Stage.LIVE):
            self.assertTrue(
                DEFAULT_CRITERIA[stage].require_positive_lower_bound, stage)
        for stage in (Stage.HISTORICAL, Stage.CHRONOLOGICAL_OOS,
                      Stage.FORWARD_SHADOW):
            self.assertFalse(
                DEFAULT_CRITERIA[stage].require_positive_lower_bound,
                f"{stage} spends nothing; demanding a bound there only "
                "delays the forward observation that produces one")

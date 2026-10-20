"""Years of the chain, compressed to something a decision can carry.

The extraction layer can pull the history. What it cannot do is hold it: a
4GB box running a live desk has no room for a million reconstructed
episodes, and a prior that has to be recomputed by scanning a warehouse is a
prior no T0 decision will ever consult.
"""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from src.research.cold_distillation import (
    COLD_DISTILLATION_SCHEMA_VERSION, MIN_LAUNCHES_FOR_RATE, RECONSTRUCTED,
    ColdDistillate, distil)


def _episode(creator="D1", at=1_700_000_000.0, peak=1.0, collapsed=False,
             rugged=None, funders=(), venue="pump"):
    outcome = {"peak_multiple": peak, "collapsed": collapsed, "rugged": rugged}
    return {
        "creator": creator, "created_at": at, "venue": venue,
        "funding_transfers": [{"from": wallet} for wallet in funders],
        "final_outcome": outcome,
    }


class ACollapseIsNotARug(unittest.TestCase):
    """The separation is the whole discipline, not fussiness.

    A reconstruction can see that a price fell to nothing. It cannot see who
    made it fall. Recording a collapse as a rug teaches a rug model to
    predict drawdowns instead of rugs -- a different and much easier thing
    that happens to look like success on reconstructed data.
    """

    def _distilled(self, **kwargs):
        return distil(_episode(at=1_700_000_000.0 + index, **kwargs)
                      for index in range(MIN_LAUNCHES_FOR_RATE * 2))

    def test_collapses_are_reported_and_rugs_are_not_inferred(self):
        distillate = self._distilled(collapsed=True)
        prior = distillate.deployer_prior("D1")
        self.assertEqual("OK", prior["status"])
        self.assertEqual(1.0, prior["collapse_rate"])
        self.assertIsNone(prior["rug_rate"])
        self.assertIn("a collapse is not a rug", prior["rug_rate_reason"])

    def test_a_labelled_rug_does_give_a_rug_rate(self):
        distillate = self._distilled(collapsed=True, rugged=True)
        prior = distillate.deployer_prior("D1")
        self.assertEqual(1.0, prior["rug_rate"])
        self.assertEqual(MIN_LAUNCHES_FOR_RATE * 2, prior["rug_rate_from"])

    def test_everything_is_stamped_reconstructed(self):
        # A prior that arrives without its provenance cannot be discounted
        # for the survivorship, latency and depth a reconstruction flatters
        # itself with.
        distillate = self._distilled()
        self.assertEqual(RECONSTRUCTED,
                         distillate.deployer_prior("D1")["provenance"])


class ARateNeedsEnoughLaunches(unittest.TestCase):

    def test_a_two_launch_deployer_has_a_record_but_no_rates(self):
        # Knowing a deployer has launched twice before is information. A rate
        # computed from two launches is not.
        distillate = distil([_episode(at=1.0 + index, collapsed=True)
                             for index in range(2)])
        prior = distillate.deployer_prior("D1")
        self.assertEqual("DATA_BLOCKED", prior["status"])
        self.assertEqual(2, prior["launches"])
        self.assertNotIn("collapse_rate", prior)

    def test_an_unknown_deployer_gets_nothing_rather_than_a_default(self):
        distillate = distil([_episode()])
        self.assertIsNone(distillate.deployer_prior("NeverSeen"))


class UnresolvedIsNotSurvived(unittest.TestCase):

    def test_a_launch_nobody_resolved_contributes_nothing(self):
        # Treating "we never found out" as "it survived" is how a rate built
        # from cold data comes out lower than the truth, in the direction
        # that makes every deployer look safer than they are.
        distillate = distil([{
            "creator": "D1", "created_at": 1.0,
            "final_outcome": {"peak_multiple": None, "collapsed": None,
                              "rugged": None},
        }])
        self.assertEqual(0, distillate.launches_distilled)
        self.assertEqual(1, distillate.skipped_unresolved)

    def test_a_launch_with_no_timestamp_contributes_nothing(self):
        distillate = distil([_episode(at=0)])
        self.assertEqual(0, distillate.launches_distilled)


class ItRefusesToAnswerAboutTheFuture(unittest.TestCase):

    def test_a_distillate_whose_horizon_is_later_declines(self):
        # Serving a deployer prior built from a launch the decision could not
        # have seen is lookahead -- the one contamination that cannot be
        # undone once it reaches a training set.
        now = time.time()
        distillate = distil([_episode(at=now, collapsed=True)
                             for _ in range(MIN_LAUNCHES_FOR_RATE)])
        self.assertIsNotNone(distillate.deployer_prior("D1", as_of=now + 1))
        self.assertIsNone(distillate.deployer_prior("D1", as_of=now - 3600))
        self.assertEqual(1, distillate.lookahead_refusals)

    def test_a_live_decision_is_not_penalised_by_the_check(self):
        distillate = distil([_episode(at=1_700_000_000.0 + index, collapsed=True)
                             for index in range(MIN_LAUNCHES_FOR_RATE)])
        self.assertIsNotNone(distillate.deployer_prior("D1", as_of=time.time()))
        self.assertEqual(0, distillate.lookahead_refusals)


class ItStaysSmallEnoughToLoad(unittest.TestCase):

    def test_the_single_launch_tail_is_dropped(self):
        # Most creators launch once, and a deployer with one launch is
        # indistinguishable from one with none for every purpose a decision
        # has. It is also most of the file.
        records = [_episode(creator=f"one-{index}", at=1.0 + index)
                   for index in range(500)]
        records += [_episode(creator="busy", at=1.0 + index)
                    for index in range(50)]
        distillate = distil(records, max_deployers=10)
        self.assertLessEqual(len(distillate.deployers), 10)
        self.assertIn("busy", distillate.deployers)

    def test_peak_multiples_do_not_grow_without_bound(self):
        distillate = distil([_episode(peak=float(index), at=1.0 + index)
                             for index in range(1000)])
        self.assertLessEqual(len(distillate.deployers["D1"].peak_multiples), 64)

    def test_the_artifact_stores_rows_not_repeated_keys(self):
        distillate = distil([_episode(at=1.0 + index, collapsed=True)
                             for index in range(MIN_LAUNCHES_FOR_RATE)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cold.json"
            self.assertTrue(distillate.save(path))
            payload = json.loads(path.read_text())
            self.assertIsInstance(payload["deployers"][0], list)


class ItSurvivesARestartAndRefusesAStrangeOne(unittest.TestCase):

    def test_a_saved_distillate_loads_back_identically(self):
        distillate = distil(
            [_episode(at=1.0 + index, collapsed=True, funders=("F1",))
             for index in range(MIN_LAUNCHES_FOR_RATE * 2)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cold.json"
            distillate.save(path)
            loaded = ColdDistillate.load(path)
        self.assertEqual(distillate.deployer_prior("D1"),
                         loaded.deployer_prior("D1"))
        self.assertEqual(distillate.funder_prior("F1"),
                         loaded.funder_prior("F1"))
        self.assertEqual(distillate.covers_until, loaded.covers_until)

    def test_a_distillate_from_another_schema_is_ignored_not_guessed_at(self):
        # Guessing at an unknown layout would put silently wrong priors in
        # front of decisions.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cold.json"
            path.write_text(json.dumps({"schema": "v99", "deployers": []}))
            self.assertIsNone(ColdDistillate.load(path))

    def test_a_missing_file_is_no_distillate_rather_than_an_error(self):
        self.assertIsNone(ColdDistillate.load(Path("/nonexistent/cold.json")))

    def test_the_report_says_how_much_of_it_is_usable(self):
        distillate = distil(
            [_episode(creator=f"d{index // 6}", at=1.0 + index, collapsed=True)
             for index in range(60)])
        report = distillate.report()
        self.assertEqual("OK", report["status"])
        self.assertEqual(RECONSTRUCTED, report["provenance"])
        self.assertEqual(COLD_DISTILLATION_SCHEMA_VERSION, report["schema"])
        self.assertGreater(report["deployers_with_a_rate"], 0)


if __name__ == "__main__":
    unittest.main()

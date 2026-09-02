"""The trainers had no caller, which is why nothing was ever promoted.

`shadow_trainer`, `hazard_trainer`, `exit_policy_trainer` and
`action_value_trainer` are four complete, strictly gated `__main__` scripts
with zero runtime callers between them. The desk observed launches, resolved
outcomes, wrote episodes to disk -- and never fitted anything to them. So
the report said `prediction: DATA_BLOCKED`, every age band said "no artifact
for this age band", and the native inference module said the return model
"has no artifact, so there is nothing to port".

That reads like a model that failed validation. It was a model that was
never asked.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

from src.runtime.training import (
    MAX_CONSECUTIVE_FAILURES, TrainingJob, TrainingSupervisor, _explain)


class _Fixture(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.storage = self.root / "episodes"
        self.models = self.root / "models"
        self.storage.mkdir()
        self.models.mkdir()

    def _episodes(self, count, gz=True, subdir="2026-09-01"):
        directory = self.storage / subdir
        directory.mkdir(exist_ok=True, parents=True)
        for index in range(count):
            suffix = ".json.gz" if gz else ".json"
            (directory / f"tok{index}{suffix}").write_text("{}")

    def _supervisor(self, jobs=(), **kwargs):
        return TrainingSupervisor(self.storage, self.models, jobs=jobs, **kwargs)


class ItCountsTheEvidenceInBothLayouts(_Fixture):
    """The desk writes two. A count that saw one would report a stalled
    corpus on a desk collecting perfectly well -- and the whole trigger is
    built on this number."""

    async def test_gzipped_dated_episodes_are_counted(self):
        self._episodes(7)
        self.assertEqual(7, self._supervisor().resolved_episodes())

    async def test_flat_reconstructed_episodes_are_counted(self):
        self._episodes(3, gz=False, subdir="reconstructed")
        self.assertEqual(3, self._supervisor().resolved_episodes())

    async def test_the_outcome_index_is_not_an_episode(self):
        (self.storage / "outcome_index.json").write_text("{}")
        self.assertEqual(0, self._supervisor().resolved_episodes())

    async def test_in_flight_episodes_are_not_counted(self):
        # `active/` holds episodes that have not resolved. Training on them
        # would be training on outcomes that have not happened.
        active = self.storage / "active"
        active.mkdir()
        (active / "tok.json.gz").write_text("{}")
        self.assertEqual(0, self._supervisor().resolved_episodes())


class ItTrainsOnEvidenceNotOnAClock(_Fixture):

    async def test_a_fresh_process_trains_on_whatever_already_exists(self):
        # A desk that collected for a week and then restarted should not
        # wait for another 250 launches to fit what it already has.
        self._episodes(10)
        supervisor = self._supervisor(new_episodes_required=250)
        self.assertTrue(supervisor.should_train())

    async def test_an_empty_corpus_trains_nothing(self):
        self.assertFalse(self._supervisor().should_train())
        self.assertIn("no resolved episodes", self._supervisor().pending_reason())

    async def test_a_quiet_night_does_not_retrain(self):
        # Refitting the same corpus produces the same artifact and burns CPU
        # the desk needs for launches.
        self._episodes(300)
        supervisor = self._supervisor(new_episodes_required=250, jobs=())
        supervisor.episodes_at_last_round = 300
        self.assertFalse(supervisor.should_train())
        self.assertIn("waiting for evidence", supervisor.pending_reason())

    async def test_enough_new_evidence_does(self):
        self._episodes(600)
        supervisor = self._supervisor(new_episodes_required=250)
        supervisor.episodes_at_last_round = 300
        self.assertTrue(supervisor.should_train())


class _Script:
    """A stand-in trainer, written to a real module and really executed."""

    @staticmethod
    def write(root: Path, name: str, body: str) -> str:
        package = root / "fake_trainers"
        package.mkdir(exist_ok=True)
        (package / "__init__.py").write_text("")
        (package / f"{name}.py").write_text(body)
        return f"fake_trainers.{name}"


class ItRunsTheTrainersInAnotherInterpreter(_Fixture):
    """A scikit-learn fit holds the GIL for seconds to minutes. On two vCPUs
    a thread doing that IS the decision loop not running."""

    def _job(self, name, body, report="report.json"):
        module = _Script.write(self.root, name, body)
        return TrainingJob(name, module, report)

    async def _run(self, job, **kwargs):
        supervisor = self._supervisor(jobs=(job,), **kwargs)
        # The child must import `fake_trainers`, which lives in the temp dir.
        import os
        previous = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = f"{self.root}{os.pathsep}{previous}"
        self.addCleanup(lambda: os.environ.__setitem__("PYTHONPATH", previous))
        return supervisor, await supervisor.run_round()

    async def test_a_passing_trainer_marks_an_artifact_written(self):
        body = f'''
import json, pathlib, sys
out = pathlib.Path(sys.argv[sys.argv.index("--model-dir") + 1])
out.mkdir(parents=True, exist_ok=True)
(out / "report.json").write_text(json.dumps({{"status": "PASSED", "oos_samples": 900}}))
'''
        supervisor, results = await self._run(self._job("passer", body))
        self.assertEqual("PASSED", results["passer"])
        self.assertTrue(supervisor.artifact_written)
        self.assertEqual(1, supervisor.jobs[0].passes)

    async def test_a_rejection_is_a_result_not_a_failure(self):
        # The gate saying no means the evidence does not support the model
        # yet, which is exactly what a gate is for.
        body = f'''
import json, pathlib, sys
out = pathlib.Path(sys.argv[sys.argv.index("--model-dir") + 1])
out.mkdir(parents=True, exist_ok=True)
(out / "report.json").write_text(json.dumps(
    {{"status": "REJECTED", "oos_samples": 12, "shadow_policy_trades": 0}}))
'''
        supervisor, results = await self._run(self._job("rejecter", body))
        self.assertEqual("REJECTED", results["rejecter"])
        self.assertFalse(supervisor.artifact_written)
        self.assertEqual(0, supervisor.jobs[0].failures)
        self.assertEqual(1, supervisor.jobs[0].rejections)
        # And it says WHY, which is the difference between DATA_BLOCKED and
        # something an operator can act on.
        self.assertIn("oos_samples=12/50", supervisor.jobs[0].last_detail)

    async def test_a_crashing_trainer_is_a_failure_with_its_stderr(self):
        body = 'raise SystemExit("boom")\n'
        supervisor, results = await self._run(self._job("crasher", body))
        self.assertEqual("FAILED", results["crasher"])
        self.assertEqual(1, supervisor.jobs[0].failures)
        self.assertIn("boom", supervisor.jobs[0].last_detail)

    async def test_a_hanging_trainer_is_killed_rather_than_left_running(self):
        # A hung fit holding a core on a two-vCPU box costs more than the
        # model it is not producing.
        body = "import time\ntime.sleep(60)\n"
        supervisor, results = await self._run(self._job("hanger", body),
                                              timeout_s=1.0)
        self.assertEqual("FAILED", results["hanger"])
        self.assertIn("timed out", supervisor.jobs[0].last_detail)

    async def test_a_trainer_that_keeps_dying_is_left_alone(self):
        # A trainer that dies at import will die at import every time, and
        # respawning it forever buries the reason.
        body = 'raise SystemExit("always")\n'
        job = self._job("always", body)
        supervisor = self._supervisor(jobs=(job,))
        import os
        previous = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = f"{self.root}{os.pathsep}{previous}"
        self.addCleanup(lambda: os.environ.__setitem__("PYTHONPATH", previous))
        for _ in range(MAX_CONSECUTIVE_FAILURES):
            await supervisor.run_round()
        self.assertTrue(supervisor.jobs[0].abandoned)
        results = await supervisor.run_round()
        self.assertEqual("ABANDONED", results["always"])

    async def test_a_pass_after_failures_clears_the_streak(self):
        body = f'''
import json, pathlib, sys
out = pathlib.Path(sys.argv[sys.argv.index("--model-dir") + 1])
out.mkdir(parents=True, exist_ok=True)
(out / "report.json").write_text(json.dumps({{"status": "PASSED"}}))
'''
        job = self._job("recovers", body)
        job.consecutive_failures = MAX_CONSECUTIVE_FAILURES - 1
        supervisor, _ = await self._run(job)
        self.assertEqual(0, supervisor.jobs[0].consecutive_failures)


class TheReportSaysWhyNothingIsTrained(_Fixture):

    async def test_it_names_the_gate_and_the_shortfall(self):
        # "DATA_BLOCKED" tells nobody anything. "12 out-of-sample rows, 50
        # required" tells them whether to wait a day or look for a bug.
        detail = _explain({"status": "REJECTED", "oos_samples": 12,
                           "shadow_policy_trades": 3,
                           "mean_brier_skill": -0.02})
        self.assertIn("oos_samples=12/50", detail)
        self.assertIn("shadow_policy_trades=3/10", detail)
        self.assertIn("mean_brier_skill=-0.02", detail)

    async def test_an_explicit_reason_wins_over_the_derived_one(self):
        self.assertEqual("not enough rows",
                         _explain({"status": "DATA_BLOCKED",
                                   "reason": "not enough rows"}))

    async def test_the_report_is_honest_before_the_first_round(self):
        report = self._supervisor().report()
        self.assertEqual("NEVER_RUN", report["status"])
        self.assertIsNone(report["episodes_at_last_round"])
        self.assertIn("never attempted", report["pending_reason"])
        for job in report["jobs"]:
            self.assertEqual("NEVER_RUN", job["status"])


if __name__ == "__main__":
    unittest.main()


class APassedArtifactReachesTheNativeEvaluator(_Fixture):
    """`export_hazard_model.py` had no caller either.

    The chain is trainer -> artifact -> export -> native inference, and it
    was broken in two places, not one: even a hazard model that trained and
    passed would never have reached the Rust evaluator, so that path was
    dead while looking finished.
    """

    def _jobs(self, export_body, trainer_status="PASSED"):
        trainer_body = f'''
import json, pathlib, sys
out = pathlib.Path(sys.argv[sys.argv.index("--model-dir") + 1])
out.mkdir(parents=True, exist_ok=True)
(out / "report.json").write_text(json.dumps({{"status": "{trainer_status}"}}))
'''
        module = _Script.write(self.root, "trainer_x", trainer_body)
        export = _Script.write(self.root, "exporter_x", export_body)
        return (TrainingJob("hazard", module, "report.json",
                            export_module=export),)

    async def _run(self, jobs):
        import os
        supervisor = self._supervisor(jobs=jobs)
        previous = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = f"{self.root}{os.pathsep}{previous}"
        self.addCleanup(lambda: os.environ.__setitem__("PYTHONPATH", previous))
        return supervisor, await supervisor.run_round()

    async def test_a_pass_triggers_the_export(self):
        body = f'''
import pathlib, sys
out = pathlib.Path(sys.argv[sys.argv.index("--model-dir") + 1])
(out / "hazard_native.json").write_text("{{}}")
print("exported 2 head(s)")
'''
        supervisor, _ = await self._run(self._jobs(body))
        self.assertEqual(1, supervisor.jobs[0].exports)
        self.assertIn("exported", supervisor.jobs[0].last_export_detail)
        self.assertTrue((self.models / "hazard_native.json").exists())

    async def test_a_rejection_does_not_export(self):
        body = 'import pathlib, sys\nprint("should not run")\n'
        supervisor, _ = await self._run(
            self._jobs(body, trainer_status="REJECTED"))
        self.assertEqual(0, supervisor.jobs[0].exports)

    async def test_a_broken_export_does_not_fail_the_training_round(self):
        # The model passed its gate and the Python path will use it. What is
        # lost is only the native evaluation, and reporting the round as
        # failed would hide a good model behind a broken export.
        body = 'raise SystemExit("no hazard artifact to export")\n'
        supervisor, results = await self._run(self._jobs(body))
        self.assertEqual("PASSED", results["hazard"])
        self.assertTrue(supervisor.artifact_written)
        self.assertEqual(0, supervisor.jobs[0].failures)
        self.assertEqual(1, supervisor.jobs[0].export_failures)
        self.assertIn("no hazard artifact", supervisor.jobs[0].last_export_detail)

    async def test_the_job_copy_does_not_drop_declarative_fields(self):
        """The copy listed four fields and silently dropped `export_module`.

        So the export never ran and the feature looked wired while doing
        nothing -- the exact failure this whole audit keeps finding. The copy
        now carries everything and forgets only what it must.
        """
        from src.runtime.training import DEFAULT_JOBS

        supervisor = self._supervisor()
        by_name = {job.name: job for job in supervisor.jobs}
        for original in DEFAULT_JOBS:
            copied = by_name[original.name]
            self.assertEqual(original.module, copied.module)
            self.assertEqual(original.report_file, copied.report_file)
            self.assertEqual(original.export_module, copied.export_module)
            self.assertEqual(original.extra, copied.extra)
            # ...and forgets what another supervisor did.
            self.assertEqual(0, copied.attempts)
            self.assertEqual("NEVER_RUN", copied.last_status)

    async def test_the_hazard_job_still_declares_its_export(self):
        # Naming it here so deleting the wiring fails a test rather than
        # quietly returning the native evaluator to being dead.
        supervisor = self._supervisor()
        hazard = next(job for job in supervisor.jobs if job.name == "hazard")
        self.assertEqual("tools.export_hazard_model", hazard.export_module)

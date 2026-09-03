"""Something that actually runs the trainers.

Every trainer in this repository is a complete, careful, well-gated
`__main__` script that nothing calls. `shadow_trainer`, `hazard_trainer`,
`exit_policy_trainer` and `action_value_trainer` have zero runtime callers
between them: the desk observes launches, resolves outcomes, writes episodes
to disk -- and then never fits anything to them. So the report says
`prediction: DATA_BLOCKED`, every age band says "no artifact for this age
band", and the native inference module says the return model "has no
artifact, so there is nothing to port".

That reads like a model that failed validation. It is not. It is a model
that was never asked. The hot-reload path already exists and works off the
artifact's mtime, the gates are already written and are strict, and the
episodes are already on disk. The one missing piece was a caller.

Four decisions that shape this, each of which could have gone the other way:

**A subprocess, not a thread.** A scikit-learn fit holds the GIL for seconds
to minutes. On a two-vCPU box a thread doing that IS the decision loop not
running, and the desk would go blind at exactly the moment it was getting
smarter. A separate interpreter cannot do that, and it can be nice'd below
the desk and killed if it hangs.

**Triggered by new evidence, not by a clock.** Refitting the same corpus
produces the same artifact and burns CPU the desk needs for launches.
Training runs when enough episodes have RESOLVED since the last attempt --
which on a quiet night is never, correctly.

**One at a time.** Four concurrent fits on two vCPUs is a box that has
stopped being a desk.

**A rejection is a result, not a failure to retry.** The gate saying no is
information: it means the evidence does not yet support the model. Retrying
immediately on the same data asks the same question and gets the same
answer, so the next attempt waits for the same new-evidence threshold as any
other.

What this deliberately does NOT do is lower a bar. Every gate stays exactly
where its author put it. If the model does not pass, the desk keeps
declining to trade on it, and the report says which gate it failed and by
how much -- which is the difference between "DATA_BLOCKED" and an actionable
sentence.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from src.runtime.training_lock import TrainingBusy, training_lock

logger = logging.getLogger(__name__)

TRAINING_SCHEMA_VERSION = "v1"

#: How many newly resolved episodes must land before another attempt. Not a
#: clock: a quiet night should not retrain, and a busy hour should.
DEFAULT_NEW_EPISODES = 250

#: How long a single fit may run before it is killed. Generous, because a
#: real fit over tens of thousands of episodes is slow -- and bounded,
#: because a hung trainer holding a core on a two-vCPU box is worse than no
#: model at all.
DEFAULT_TIMEOUT_S = 1800.0

#: Consecutive hard failures (crash, timeout, unreadable report) before a job
#: is left alone. A trainer that dies at import will die at import every
#: time, and respawning it forever buries the reason.
MAX_CONSECUTIVE_FAILURES = 3

#: How far below the desk the fit runs. The same value the offloaded miners
#: use: background work yields to decisions, never the other way round.
TRAINER_NICE = 10


@dataclass
class TrainingJob:
    """One trainer, its arguments, and where it writes its verdict."""

    name: str
    module: str
    report_file: str
    #: Extra CLI arguments beyond --storage and --model-dir.
    extra: Sequence[str] = ()
    #: Which standard arguments this module actually accepts. The trainers
    #: take both; the corpus jobs take neither, and handing a script an
    #: argument it has never seen turns a working job into a failed one --
    #: which is how a chain of subprocesses quietly becomes a chain of
    #: nothing.
    takes_storage: bool = True
    takes_model_dir: bool = True
    #: Seconds between attempts, on top of the new-evidence trigger. The
    #: trainers run every round; a six-hour history walk does not, because
    #: its cost is a sustained RPC crawl and its benefit accrues in days.
    min_interval_s: float = 0.0
    #: Run after a PASS, to carry the artifact the last step of the way.
    #: `export_hazard_model.py` had no caller either -- so even a hazard
    #: model that trained and passed would never have reached the Rust
    #: evaluator, and the native inference path would have stayed dead
    #: while looking finished. The chain is trainer -> artifact -> export
    #: -> native, and it was broken in two places, not one.
    export_module: str = ""
    #: Whether that export takes --storage. The trainers do; the exporter
    #: does not, and passing it an argument it has never seen turns a
    #: successful training round into a failed one.
    export_takes_storage: bool = False
    exports: int = 0
    export_failures: int = 0
    last_export_detail: str = ""
    attempts: int = 0
    passes: int = 0
    rejections: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    last_attempt_at: float = 0.0
    last_status: str = "NEVER_RUN"
    last_detail: str = ""
    last_duration_s: float = 0.0

    @property
    def abandoned(self) -> bool:
        return self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.last_status,
            "detail": self.last_detail,
            "attempts": self.attempts,
            "passes": self.passes,
            "rejections": self.rejections,
            "failures": self.failures,
            "abandoned": self.abandoned,
            "last_attempt_at": self.last_attempt_at or None,
            "last_duration_s": round(self.last_duration_s, 1) or None,
            "exports": self.exports,
            "export_failures": self.export_failures,
            "last_export_detail": self.last_export_detail or None,
        }


#: Everything a copied job must forget. Declarative fields are carried by
#: `replace`; anything that counts what a PREVIOUS supervisor did belongs
#: here, or two desks in one process would share a history.
_FRESH = {
    "attempts": 0, "passes": 0, "rejections": 0, "failures": 0,
    "consecutive_failures": 0, "last_attempt_at": 0.0,
    "last_status": "NEVER_RUN", "last_detail": "", "last_duration_s": 0.0,
    "exports": 0, "export_failures": 0, "last_export_detail": "",
}

#: The jobs, in the order they run. Hazard first: it is the only model whose
#: artifact the native fast path already evaluates, so a desk that can only
#: finish one fit before the next launch burst should finish that one.
DEFAULT_JOBS = (
    # The corpus first, then the models fitted to it. Both were orphans:
    # `tools/backfill_history.py` walks the Pump program backwards and
    # reconstructs episodes at zero cost, and `tools/distil_history.py`
    # compresses them into the priors a T0 decision can afford -- and
    # nothing called either, so the moat never grew and the distillate was
    # never built. A desk that has been running for a month had exactly the
    # history it happened to watch live.
    TrainingJob("backfill", "tools.backfill_history",
                # It writes episodes, not a report. There is no verdict to
                # read, so success is the exit code and the corpus count is
                # the measurement -- which the next round sees.
                "", takes_storage=False, takes_model_dir=False,
                min_interval_s=6 * 3600.0),
    TrainingJob("distil", "tools.distil_history", "",
                takes_storage=False, takes_model_dir=False,
                min_interval_s=6 * 3600.0),
    TrainingJob("hazard", "src.research.hazard_trainer",
                "last_hazard_training_report.json",
                export_module="tools.export_hazard_model"),
    TrainingJob("shadow", "src.research.shadow_trainer",
                "last_training_report.json"),
    TrainingJob("exit_policy", "src.research.exit_policy_trainer",
                "last_exit_policy_report.json"),
    TrainingJob("action_value", "src.research.action_value_trainer",
                "last_action_value_report.json"),
)


class TrainingSupervisor:
    """Runs the trainers the desk has been collecting evidence for."""

    def __init__(self, storage: Path, model_dir: Path, *,
                 new_episodes_required: int = DEFAULT_NEW_EPISODES,
                 timeout_s: float = DEFAULT_TIMEOUT_S,
                 jobs: Sequence[TrainingJob] = ()):
        self.storage = Path(storage)
        self.model_dir = Path(model_dir)
        self.new_episodes_required = int(new_episodes_required)
        self.timeout_s = float(timeout_s)
        # Fresh instances, so two supervisors in one process (tests) do not
        # share counters through the module-level defaults.
        #
        # `replace` rather than a hand-written constructor call. The
        # hand-written one listed four fields and silently dropped
        # `export_module`, so the hazard export -- the whole reason that
        # field exists -- never ran, and the feature looked wired while doing
        # nothing. Copying everything and resetting the counters by name
        # inverts which mistake is possible: a new declarative field is
        # carried automatically, and a new counter has to be added to
        # _FRESH deliberately.
        self.jobs: List[TrainingJob] = [
            replace(job, **_FRESH) for job in (jobs or DEFAULT_JOBS)]
        self.running = ""
        self.rounds = 0
        self.episodes_at_last_round = -1
        self.last_round_at = 0.0
        self.artifact_written = False
        self._pending_exports: List[TrainingJob] = []

    # --- when to train ---------------------------------------------------

    def resolved_episodes(self) -> int:
        """How much evidence exists. Cheap enough to ask every minute.

        Counts files rather than parsing them: the question is "has the
        corpus grown", and opening tens of thousands of JSON documents to
        answer it would cost more than the fit it is deciding whether to run.

        Both layouts, because the desk writes two. Live episodes land
        gzipped under a date directory; reconstructed ones land flat and
        uncompressed. A count that saw only one of them would report a
        stalled corpus on a desk that was collecting perfectly well -- and
        the whole trigger is built on this number.
        """
        total = 0
        try:
            for pattern in ("*.json.gz", "*.json"):
                total += sum(1 for path in self.storage.rglob(pattern)
                             if path.name != "outcome_index.json"
                             and "active" not in path.parts)
        except OSError:
            return 0
        return total

    def should_train(self) -> bool:
        episodes = self.resolved_episodes()
        if self.episodes_at_last_round < 0:
            # First check of the process. Train if there is anything at all
            # to train on: a desk that has been collecting for a week and
            # then restarts should not wait for another 250 launches to fit
            # what it already has.
            return episodes > 0
        return episodes - self.episodes_at_last_round >= self.new_episodes_required

    def pending_reason(self) -> str:
        """Why training has not run, in a sentence an operator can act on."""
        episodes = self.resolved_episodes()
        if self.episodes_at_last_round < 0:
            return (f"never attempted; {episodes} resolved episode(s) on disk"
                    if episodes else
                    "never attempted; no resolved episodes on disk yet")
        grown = episodes - self.episodes_at_last_round
        if grown < self.new_episodes_required:
            return (f"waiting for evidence: {grown} new episode(s) since the "
                    f"last round, {self.new_episodes_required} required")
        return "ready to train on the next round"

    # --- running ---------------------------------------------------------

    async def run_round(self) -> Dict[str, Any]:
        """Fit everything that is worth fitting, one at a time.

        Under the cross-process training lock. The desk trains on its own
        clock and a systemd timer trains every fifteen minutes; both invoke
        these same trainers over the same corpus, and two shadow passes at
        once is roughly twice the resident corpus on a 4 GB box. The kernel
        resolves that by killing something, and what it kills may be the desk,
        whose forward evidence cannot be backfilled.
        """
        try:
            with training_lock(self.storage.parent
                               if self.storage.name == "launch_episodes"
                               else self.storage,
                               owner="training_supervisor"):
                return await self._run_round_locked()
        except TrainingBusy as exc:
            logger.info("TRAINING skipped: %s", exc)
            return {"status": "SKIPPED_LOCKED", "detail": str(exc)}

    async def _run_round_locked(self) -> Dict[str, Any]:
        self.rounds += 1
        self.last_round_at = time.time()
        self.episodes_at_last_round = self.resolved_episodes()
        self.artifact_written = False
        self._pending_exports = []
        results: Dict[str, Any] = {}
        now = time.time()
        for job in self.jobs:
            if job.abandoned:
                results[job.name] = "ABANDONED"
                continue
            if (job.min_interval_s
                    and job.last_attempt_at
                    and now - job.last_attempt_at < job.min_interval_s):
                results[job.name] = "NOT_DUE"
                continue
            results[job.name] = await self._run_job(job)
        # After the fits, not between them: an export is cheap and the round
        # should not pay for it twice if two jobs pass.
        for job in self._pending_exports:
            await self._export(job)
        return results

    async def _export(self, job: TrainingJob) -> bool:
        """Carry a passed artifact the last step, into the native evaluator.

        Failing here is NOT a failed training round. The model passed its
        gate and the Python path will use it; what is lost is only the
        native evaluation of it, and reporting the whole round as failed
        would hide a good model behind a broken export.
        """
        command = [sys.executable, "-m", job.export_module,
                   "--model-dir", str(self.model_dir)]
        if job.export_takes_storage:
            command += ["--storage", str(self.storage)]
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(Path(__file__).resolve().parents[2]),
                preexec_fn=_lower_priority if os.name == "posix" else None)
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=min(300.0, self.timeout_s))
        except Exception as exc:
            job.export_failures += 1
            job.last_export_detail = f"{type(exc).__name__}: {exc}"
            logger.warning("EXPORT %s failed: %s", job.name, job.last_export_detail)
            return False
        detail = (stdout or stderr or b"").decode("utf-8", "replace").strip()[-300:]
        if process.returncode != 0:
            job.export_failures += 1
            job.last_export_detail = f"exit {process.returncode}: {detail}"
            logger.warning("EXPORT %s failed: %s", job.name, job.last_export_detail)
            return False
        job.exports += 1
        job.last_export_detail = detail or "exported"
        logger.info("EXPORT %s: %s", job.name, job.last_export_detail)
        return True

    async def _run_job(self, job: TrainingJob) -> str:
        job.attempts += 1
        job.last_attempt_at = time.time()
        self.running = job.name
        started = time.monotonic()
        command = [sys.executable, "-m", job.module]
        if job.takes_storage:
            command += ["--storage", str(self.storage)]
        if job.takes_model_dir:
            command += ["--model-dir", str(self.model_dir)]
        command += list(job.extra)
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(Path(__file__).resolve().parents[2]),
                preexec_fn=_lower_priority if os.name == "posix" else None)
        except Exception as exc:
            return self._failed(job, f"could not start: {exc}", started)
        try:
            _, stderr = await asyncio.wait_for(process.communicate(),
                                               timeout=self.timeout_s)
        except asyncio.TimeoutError:
            # Killed rather than left running. A hung fit holding a core on a
            # two-vCPU box costs more than the model it is not producing.
            try:
                process.kill()
                await process.wait()
            except Exception:  # pragma: no cover - teardown only
                pass
            return self._failed(
                job, f"timed out after {self.timeout_s:.0f}s", started)
        finally:
            self.running = ""
        job.last_duration_s = time.monotonic() - started
        if process.returncode != 0:
            tail = (stderr or b"").decode("utf-8", "replace").strip()[-400:]
            return self._failed(job, f"exit {process.returncode}: {tail}", started)
        return self._read_report(job)

    def _failed(self, job: TrainingJob, detail: str, started: float) -> str:
        job.failures += 1
        job.consecutive_failures += 1
        job.last_duration_s = time.monotonic() - started
        job.last_status = "FAILED"
        job.last_detail = detail
        logger.error("TRAINER %s failed: %s", job.name, detail)
        if job.abandoned:
            logger.error(
                "TRAINER %s has failed %d times in a row and will not be "
                "retried this run. The desk continues on whatever artifact it "
                "already had, which for an untrained head means it keeps "
                "declining to trade.", job.name, job.consecutive_failures)
        return "FAILED"

    def _read_report(self, job: TrainingJob) -> str:
        """The trainer's own verdict, in its own words."""
        if not job.report_file:
            # A job that writes data rather than a model has no gate to
            # report. Its exit code is the whole verdict, and pretending
            # otherwise would mean inventing a status.
            job.consecutive_failures = 0
            job.last_status = "COMPLETED"
            job.last_detail = "no gate; produced data rather than a model"
            return "COMPLETED"
        path = self.model_dir / job.report_file
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return self._failed(job, f"unreadable report: {exc}", time.monotonic())
        job.consecutive_failures = 0
        status = str(report.get("status", "") or "UNKNOWN")
        job.last_status = status
        job.last_detail = _explain(report)
        if status == "PASSED":
            job.passes += 1
            self.artifact_written = True
            logger.info("TRAINER %s PASSED: %s", job.name, job.last_detail)
            if job.export_module:
                self._pending_exports.append(job)
        elif status == "REJECTED":
            job.rejections += 1
            # Not an error. The gate saying no means the evidence does not
            # support the model yet, which is exactly what a gate is for.
            logger.info("TRAINER %s rejected by its own gate: %s",
                        job.name, job.last_detail)
        else:
            logger.info("TRAINER %s: %s -- %s", job.name, status, job.last_detail)
        return status

    def report(self) -> Dict[str, Any]:
        return {
            "schema": TRAINING_SCHEMA_VERSION,
            "status": "OK" if self.rounds else "NEVER_RUN",
            "rounds": self.rounds,
            "running": self.running or None,
            "resolved_episodes": self.resolved_episodes(),
            "episodes_at_last_round": (self.episodes_at_last_round
                                       if self.episodes_at_last_round >= 0 else None),
            "new_episodes_required": self.new_episodes_required,
            "pending_reason": self.pending_reason(),
            "last_round_at": self.last_round_at or None,
            "jobs": [job.as_dict() for job in self.jobs],
            "detail": ("every trainer in this repository was a __main__ with "
                       "no caller; the desk collected evidence and never fitted "
                       "anything to it, which is why the report said "
                       "DATA_BLOCKED rather than that a model had failed"),
        }


def _lower_priority() -> None:  # pragma: no cover - child process only
    """Run the fit below the desk. Background work yields, never the reverse."""
    try:
        os.nice(TRAINER_NICE)
    except (OSError, AttributeError):
        pass


def _explain(report: Dict[str, Any]) -> str:
    """Turn a trainer's report into the sentence an operator needs.

    "DATA_BLOCKED" tells nobody anything. "12 out-of-sample rows, 50
    required" tells them whether to wait a day or go looking for a bug.
    """
    status = str(report.get("status", "") or "")
    if report.get("reason"):
        return str(report["reason"])
    if report.get("detail"):
        return str(report["detail"])
    parts: List[str] = []
    for key, needed in (("oos_samples", 50), ("shadow_policy_trades", 10),
                        ("feasible_return_samples", 10), ("rows", None),
                        ("episodes", None), ("lifecycles", None)):
        value = report.get(key)
        if value is None:
            continue
        parts.append(f"{key}={value}" + (f"/{needed}" if needed else ""))
    for key in ("mean_brier_skill", "net_elogw_proxy", "feasible_log_mse",
                "feasible_log_baseline_mse"):
        value = report.get(key)
        if isinstance(value, (int, float)):
            parts.append(f"{key}={value:.4g}")
    return ", ".join(parts) or status or "no detail reported"


class DeskTraining:
    """The desk's side of training: when to run it, and what to do on a pass.

    A mixin for the same reason the other runtime services are: both methods
    read and write the desk's own subsystems -- the predictor, the sizing
    engine, the validation register -- and injecting all of them into a
    collaborator would be a behaviour-changing refactor dressed as a tidy-up.
    """

    async def _training_loop(self):
        """Fit the models the desk has been collecting evidence for.

        Its own loop on a long clock, because a training round is minutes of
        another interpreter's CPU and must never share a cadence with
        anything that decides. The round itself decides whether to run: it
        fires only when enough episodes have RESOLVED since the last one, so
        a quiet night trains nothing and a busy hour trains once.

        A pass activates the artifact through the SAME mtime reload the desk
        already used for a model dropped in by hand -- so a model that
        passes at 4am is live at 4am, not at the next restart.
        """
        interval = float(self.global_config.get("training_check_seconds", 300.0))
        while self._running:
            await asyncio.sleep(interval)
            if self.offline or not self.training.should_train():
                continue
            try:
                results = await self.training.run_round()
            except Exception as exc:
                logger.exception("Training round error: %s", exc)
                continue
            logger.info("TRAINING round %d: %s", self.training.rounds,
                        ", ".join(f"{name}={status}"
                                  for name, status in results.items()))
            if self.training.artifact_written:
                # Something passed its gate. Reload now rather than waiting
                # for the intelligence sweep: a model that has just earned
                # promotion is worth the one extra check.
                try:
                    self._reload_promoted_model()
                except Exception as exc:
                    logger.exception("Model activation after training: %s", exc)

    def _reload_promoted_model(self) -> bool:
        """Activate a newer artifact, from wherever it came from.

        Shared by the intelligence sweep and the training loop, because
        "a new artifact appeared on disk" is one event however it got there
        -- dropped in by an operator, or just fitted by this process.
        """
        latest_mtime = self._latest_model_mtime()
        if latest_mtime <= self._model_artifact_mtime:
            return False
        candidate = AgeBandedPredictor(
            os.getenv("MODEL_DIR", "models"),
            allow_pooled_fallback=bool(
                self.global_config.get("allow_pooled_model_fallback", True)))
        if not any(candidate.load_latest().values()):
            return False
        self.predictor = candidate
        self.elogw_engine.predictor = candidate
        self._model_artifact_mtime = latest_mtime
        self._register_model_validation(candidate.validation_report)
        logger.info("Activated chronologically validated shadow model %s",
                    candidate.model_version)
        return True

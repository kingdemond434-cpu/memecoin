"""The forward-shadow ledger: evidence that accumulates instead of being asserted.

The promotion gate has known for some time exactly what it wants before live
capital -- decisions, real fills, launch cohorts, regimes, net log growth,
rug-loss share, monster enrichment, execution success, zero catastrophic
failures -- and nothing was counting any of it. So every audit reported
"forward shadow proof: insufficient", which was true and unhelpful, because
"insufficient" and "not started" are the same sentence and only one of them
gets better by waiting.

This is the thing that makes the number go up. It records outcomes as the desk
produces them, persists across restarts, and reports the distance to the next
stage as a set of ratios rather than a verdict. Distance is the useful form: a
gate that says PASS/FAIL tells you nothing about whether you are a week away or
a year away, and that difference decides whether to keep running or to change
something.

Three properties are load-bearing.

Nothing here is inferred. Every counter is fed from an outcome the desk
actually observed, and a field nobody has fed stays None rather than zero,
because the gate treats unmeasured as failing and would otherwise be told a
requirement was met by the absence of evidence for it.

Cohorts and regimes are counted as SETS, not totals. Five thousand decisions
about the same launch in the same regime is one cohort and one regime, and
counting them as five thousand is how a system passes a diversity requirement
without having seen any diversity.

And it persists. A shadow run that resets on restart never reaches a five
thousand decision threshold, no matter how long it runs -- which is the exact
shape of a requirement that looks stringent and is in fact unreachable.
"""

import json
import logging
import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.research.gauntlet import bootstrap_lower_bound
from src.research.promotion_gate import (
    DEFAULT_CRITERIA, Evidence, PromotionCriteria, Stage, evaluate, next_stage,
)

logger = logging.getLogger(__name__)

FORWARD_EVIDENCE_SCHEMA_VERSION = "v1"


@dataclass
class Outcome:
    """One closed trade or one declined candidate, as the desk saw it."""

    token: str
    entered: bool
    regime: str = "unknown"
    realized_pnl_usd: float = 0.0
    equity_at_decision_usd: float = 0.0
    real_fill: bool = False
    rugged: bool = False
    max_multiple: Optional[float] = None
    execution_attempted: bool = False
    execution_succeeded: bool = False
    catastrophic: bool = False


class ForwardEvidence:
    """Accumulates what the promotion gate asks for, and says how far off it is."""

    #: A launch counts as a monster above this. Fixed here rather than passed
    #: in, because an enrichment ratio whose denominator can be retuned is a
    #: ratio that can be made to pass.
    MONSTER_MULTIPLE = 10.0

    #: Below this many entered trades no lower bound is reported at all. A
    #: percentile bootstrap resamples the sample it was given; on twenty
    #: launches, four of which carried the return, it reports the confidence
    #: of those four and calls it evidence. CANARY and LIVE require the bound
    #: to be positive, and an unmeasured bound FAILS, so a shortfall here
    #: holds promotion rather than granting it.
    MIN_BOOTSTRAP_SAMPLE = 100

    #: A gauntlet verdict older than this stops counting. Mechanisms decay --
    #: that is the gauntlet's own finding about this market -- so a survivor
    #: established in March is not evidence about capital deployed in
    #: September, and letting it stand would make the freshest requirement on
    #: the ladder the most stale number in it.
    #:
    #: Ten days against a WEEKLY timer, deliberately. Equal to the interval
    #: would expire the verdict at the moment the next run replaces it, so a
    #: box that was down for one Sunday, or a run that took an hour, would
    #: silently block promotion; three days of slack absorbs that without
    #: letting a fortnight-old verdict authorise capital. Changing
    #: `memecoin-shadow-gauntlet.timer` means changing this.
    GAUNTLET_MAX_AGE_S = 10 * 86_400.0

    #: What a total loss of the book contributes to the bootstrap sample.
    #: log(0) is -inf and would make every resampled mean -inf; dropping the
    #: observation instead would delete the worst outcome the desk has ever
    #: had from the only statistic that authorises capital, which is wrong in
    #: the flattering direction. Floored at -99.9% of book, which still
    #: dominates any bootstrap it appears in.
    RUIN_RATIO_FLOOR = 1e-3

    def __init__(self, path: Optional[Path] = None, stage: Stage = Stage.FORWARD_SHADOW):
        self.path = Path(path) if path else None
        self.stage = stage
        self.decisions = 0
        self.real_fills = 0
        self.entered = 0
        self.rug_losses_usd = 0.0
        self.total_losses_usd = 0.0
        self.net_log_growth = 0.0
        self.execution_attempts = 0
        self.execution_successes = 0
        self.catastrophic_failures = 0
        self.monsters = 0
        self.scored_launches = 0
        self._cohorts: Set[str] = set()
        self._regimes: Set[str] = set()
        #: Per-trade log growth against the book at the time, kept because a
        #: running sum cannot be resampled and the lower bound needs the
        #: sample, not the total.
        self._log_returns: List[float] = []
        #: Monsters among the launches we ENTERED, and how many entered
        #: launches were scored at all. Both None on a ledger written before
        #: these existed, which reads as unmeasured rather than as zero: a
        #: numerator restarted against an all-time denominator would report a
        #: real desk as having caught nothing.
        self.entered_scored: Optional[int] = 0
        self.entered_monsters: Optional[int] = 0
        self._gauntlet_survivors: Optional[int] = None
        self._gauntlet_at: Optional[float] = None
        self.started_at = time.time()
        if self.path:
            self.load()

    # -- accumulation ------------------------------------------------------

    def record(self, outcome: Outcome) -> None:
        """One outcome. Declines count too.

        A ledger fed only on entries measures the trades we took and says
        nothing about the ones we passed on, which is half of what a decision
        policy does and the half that hides its mistakes.
        """
        self.decisions += 1
        self._cohorts.add(outcome.token)
        # "unknown" is deliberately NOT a regime. Counting it would let a desk
        # that never measured the market satisfy a diversity requirement with
        # one bucket, which is the requirement passing on the absence of the
        # thing it exists to demand.
        if outcome.regime and outcome.regime != "unknown":
            self._regimes.add(outcome.regime)
        if outcome.execution_attempted:
            self.execution_attempts += 1
            self.execution_successes += int(outcome.execution_succeeded)
        if outcome.real_fill:
            self.real_fills += 1
        if outcome.catastrophic:
            self.catastrophic_failures += 1
        monster = (outcome.max_multiple is not None
                   and float(outcome.max_multiple) >= self.MONSTER_MULTIPLE)
        if outcome.max_multiple is not None:
            self.scored_launches += 1
            if monster:
                self.monsters += 1
        if not outcome.entered:
            return
        self.entered += 1
        if outcome.max_multiple is not None:
            if self.entered_scored is None:
                self.entered_scored = 0
                self.entered_monsters = 0
            self.entered_scored += 1
            self.entered_monsters += int(monster)
        pnl = float(outcome.realized_pnl_usd)
        if pnl < 0:
            self.total_losses_usd += -pnl
            if outcome.rugged:
                self.rug_losses_usd += -pnl
        equity = float(outcome.equity_at_decision_usd)
        if equity > 0:
            # Log growth against the book at the time, which is the only
            # version that composes: summing percentage returns over a
            # changing book overstates a winning run and understates a losing
            # one.
            ratio = 1.0 + pnl / equity
            if ratio > 0:
                self.net_log_growth += math.log(ratio)
                self._log_returns.append(math.log(ratio))
            else:
                # A trade that took the whole book is not a large negative
                # number, it is the end of the sequence. Recorded as
                # catastrophic rather than as -inf, which would make every
                # later average meaningless -- but it still enters the
                # bootstrap sample at the ruin floor, because a lower bound
                # computed with the ruins removed is not a lower bound.
                self.catastrophic_failures += 1
                self._log_returns.append(math.log(self.RUIN_RATIO_FLOOR))

    def record_gauntlet(self, report: Any, *, at: Optional[float] = None
                        ) -> Optional[int]:
        """Take a survivor count from a gauntlet run.

        Accepts either `MechanismScoreboard.report()` output or a bare count,
        because the caller that has the scoreboard and the caller that has a
        stored number are different callers and neither should have to
        reconstruct the other's shape.

        A run that produced no rows at all is not zero survivors -- it is no
        measurement -- and stays None, so the gate fails on "not measured"
        rather than on "measured zero". The distinction matters because only
        one of them is fixed by running the gauntlet.
        """
        count: Optional[int]
        if isinstance(report, dict):
            if not report.get("mechanisms"):
                return None
            count = report.get("survivors")
        elif isinstance(report, (int, float)) and not isinstance(report, bool):
            count = int(report)
        else:
            return None
        if count is None:
            return None
        self._gauntlet_survivors = max(0, int(count))
        self._gauntlet_at = float(at if at is not None else time.time())
        return self._gauntlet_survivors

    def load_gauntlet(self, path: Path) -> Optional[int]:
        """Read a gauntlet verdict written by a separate process.

        The verdict lives in its own file rather than in this ledger's state,
        because the desk owns `forward_evidence.json` and rewrites it whole on
        every save. A second process recording into that file would be
        clobbered by the next save -- or would clobber the desk's counters,
        depending on which wrote last -- and the failure would look like a
        gauntlet that runs and never counts.

        Loaded WITH the timestamp the run carried, so reading a stale verdict
        does not make it fresh.
        """
        path = Path(path)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("gauntlet verdict unreadable at %s: %s", path, exc)
            return None
        if not isinstance(payload, dict):
            return None
        at = payload.get("at")
        try:
            at = None if at is None else float(at)
        except (TypeError, ValueError):
            at = None
        if at is None:
            # A verdict with no timestamp cannot be aged, and an unageable
            # verdict is one that never goes stale. Refused.
            logger.warning("gauntlet verdict at %s carries no timestamp", path)
            return None
        return self.record_gauntlet(payload, at=at)

    @staticmethod
    def write_gauntlet(path: Path, report: Dict[str, Any],
                       *, at: Optional[float] = None) -> bool:
        """Persist a gauntlet run for `load_gauntlet`, atomically."""
        payload = {
            "schema": FORWARD_EVIDENCE_SCHEMA_VERSION,
            "mechanisms": report.get("mechanisms"),
            "survivors": report.get("survivors"),
            "has_edge": report.get("has_edge"),
            "coverage": report.get("coverage"),
            "at": float(at if at is not None else time.time()),
        }
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(
                "w", dir=str(path.parent), delete=False, encoding="utf-8")
            with handle:
                json.dump(payload, handle, default=str)
            os.replace(handle.name, path)
            return True
        except (OSError, ValueError) as exc:
            logger.warning("gauntlet verdict save failed: %s", exc)
            return False

    # -- reading -----------------------------------------------------------

    def lower_bound(self) -> Optional[float]:
        """Bootstrap lower confidence bound on mean per-trade log growth.

        None until there are enough trades for the resampling to mean
        anything. The gate reads None as unmeasured and therefore as failing,
        which is the correct direction: a desk that has not traded enough to
        bound its edge has not established one.
        """
        if len(self._log_returns) < self.MIN_BOOTSTRAP_SAMPLE:
            return None
        return bootstrap_lower_bound(self._log_returns)

    def gauntlet_age_s(self) -> Optional[float]:
        """How long ago the stored gauntlet verdict was produced."""
        if self._gauntlet_at is None:
            return None
        return max(0.0, time.time() - self._gauntlet_at)

    def gauntlet_survivors(self) -> Optional[int]:
        """The survivor count, or None once it has gone stale."""
        if self._gauntlet_survivors is None or self._gauntlet_at is None:
            return None
        if time.time() - self._gauntlet_at > self.GAUNTLET_MAX_AGE_S:
            return None
        return self._gauntlet_survivors

    def evidence(self) -> Evidence:
        """The gate's own shape. Unmeasured stays None."""
        return Evidence(
            stage=self.stage,
            decisions=self.decisions,
            real_fills=self.real_fills,
            launch_cohorts=len(self._cohorts),
            regimes_covered=len(self._regimes) or None,
            net_log_growth=self.net_log_growth if self.entered else None,
            rug_loss_share=(self.rug_losses_usd / self.total_losses_usd
                            if self.total_losses_usd > 0 else None),
            monster_enrichment=self._enrichment(),
            execution_success=(self.execution_successes / self.execution_attempts
                               if self.execution_attempts else None),
            catastrophic_failures=self.catastrophic_failures,
            net_log_growth_lower_bound=self.lower_bound(),
            gauntlet_survivors=self.gauntlet_survivors(),
        )

    def _enrichment(self) -> Optional[float]:
        """Monsters among what we ENTERED, over monsters among what we saw.

        The numerator counts monsters we actually bought. It used to reuse
        `self.monsters` -- every monster the desk SAW, entered or not -- over
        `self.entered`, which algebraically cancels to
        `scored_launches / entered`: a selectivity ratio wearing enrichment's
        name. A desk that entered one launch in ten reported 10x enrichment
        while holding nothing, and the CANARY bar of 2.0 was cleared by being
        picky rather than by being right. Measured on this repository
        2026-09-03: ten monsters seen, ten launches entered, no overlap
        whatsoever, reported as 10.0.

        None until both sides have been observed. An enrichment computed
        against a base rate of zero is division by an absence, and reporting
        it as infinite enrichment is the single most flattering number this
        module could produce.
        """
        if not self.scored_launches:
            return None
        if not self.entered_scored:
            # Either nothing entered has resolved yet, or this ledger predates
            # the counter. Both are "not measured", and the gate fails closed
            # on that rather than passing on a number nobody computed.
            return None
        base_rate = self.monsters / self.scored_launches
        if base_rate <= 0:
            return None
        entered_rate = self.entered_monsters / self.entered_scored
        return entered_rate / base_rate

    def distance(self, criteria: Optional[PromotionCriteria] = None) -> Dict[str, Any]:
        """How far from the next stage, as ratios rather than a verdict.

        A gate that says FAIL does not distinguish a week away from a year
        away, and that difference decides whether to keep running or change
        something.
        """
        criteria = criteria or DEFAULT_CRITERIA.get(self.stage)
        if criteria is None:
            return {"status": "DATA_BLOCKED", "detail": f"no criteria for {self.stage.value}"}
        evidence = self.evidence()
        progress: Dict[str, Any] = {}
        for field_name, required in (
            ("decisions", criteria.min_decisions),
            ("real_fills", criteria.min_real_fills),
            ("launch_cohorts", criteria.min_launch_cohorts),
            ("regimes_covered", criteria.min_regimes),
        ):
            if required <= 0:
                continue
            have = getattr(evidence, field_name) or 0
            progress[field_name] = {"have": have, "need": required,
                                    "fraction": min(1.0, have / required)}
        verdict = evaluate(criteria, evidence)
        return {
            "schema": FORWARD_EVIDENCE_SCHEMA_VERSION,
            "status": "OK",
            "stage": self.stage.value,
            "next_stage": (next_stage(self.stage).value
                           if next_stage(self.stage) else None),
            "progress": progress,
            # The narrowest of the counting requirements, which is what
            # actually governs the wait.
            "slowest": (min(progress, key=lambda key: progress[key]["fraction"])
                        if progress else None),
            "verdict": verdict.to_dict(),
            "running_days": (time.time() - self.started_at) / 86_400.0,
        }

    def report(self) -> Dict[str, Any]:
        evidence = self.evidence()
        return {
            "schema": FORWARD_EVIDENCE_SCHEMA_VERSION,
            "stage": self.stage.value,
            "status": "OK" if self.decisions else "DATA_BLOCKED",
            "evidence": evidence.to_dict(),
            "distance": self.distance(),
            "persisted_at": str(self.path) if self.path else None,
        }

    # -- persistence -------------------------------------------------------

    def state(self) -> Dict[str, Any]:
        return {
            "schema": FORWARD_EVIDENCE_SCHEMA_VERSION, "stage": self.stage.value,
            "decisions": self.decisions, "real_fills": self.real_fills,
            "entered": self.entered, "rug_losses_usd": self.rug_losses_usd,
            "total_losses_usd": self.total_losses_usd,
            "net_log_growth": self.net_log_growth,
            "execution_attempts": self.execution_attempts,
            "execution_successes": self.execution_successes,
            "catastrophic_failures": self.catastrophic_failures,
            "monsters": self.monsters, "scored_launches": self.scored_launches,
            "entered_scored": self.entered_scored,
            "entered_monsters": self.entered_monsters,
            "cohorts": sorted(self._cohorts), "regimes": sorted(self._regimes),
            "log_returns": list(self._log_returns),
            "gauntlet_survivors": self._gauntlet_survivors,
            "gauntlet_at": self._gauntlet_at,
            "started_at": self.started_at,
        }

    def save(self) -> bool:
        """Persist atomically. A half-written ledger is a reset ledger.

        Written to a temporary file in the same directory and renamed, so a
        crash mid-write leaves the previous state intact rather than a
        truncated file that parses to nothing and silently restarts the count.
        """
        if not self.path:
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(
                "w", dir=str(self.path.parent), delete=False, encoding="utf-8")
            with handle:
                json.dump(self.state(), handle)
            os.replace(handle.name, self.path)
            return True
        except (OSError, ValueError) as exc:
            logger.warning("forward evidence save failed: %s", exc)
            return False

    def load(self) -> bool:
        """Restore. A shadow run that resets on restart never reaches 5,000."""
        if not self.path or not self.path.exists():
            return False
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("forward evidence unreadable at %s: %s", self.path, exc)
            return False
        self.decisions = int(state.get("decisions", 0))
        self.real_fills = int(state.get("real_fills", 0))
        self.entered = int(state.get("entered", 0))
        self.rug_losses_usd = float(state.get("rug_losses_usd", 0.0))
        self.total_losses_usd = float(state.get("total_losses_usd", 0.0))
        self.net_log_growth = float(state.get("net_log_growth", 0.0))
        self.execution_attempts = int(state.get("execution_attempts", 0))
        self.execution_successes = int(state.get("execution_successes", 0))
        self.catastrophic_failures = int(state.get("catastrophic_failures", 0))
        self.monsters = int(state.get("monsters", 0))
        self.scored_launches = int(state.get("scored_launches", 0))
        # Absent on a ledger written before these existed. Restored as None,
        # which reads as unmeasured: zeroing them would divide a fresh
        # numerator by an all-time `entered` and report a working desk as
        # having caught no monsters at all.
        entered_scored = state.get("entered_scored")
        entered_monsters = state.get("entered_monsters")
        self.entered_scored = (None if entered_scored is None
                               else int(entered_scored))
        self.entered_monsters = (None if entered_monsters is None
                                 else int(entered_monsters))
        self._cohorts = set(state.get("cohorts") or ())
        self._regimes = set(state.get("regimes") or ())
        self._log_returns = [float(value)
                             for value in (state.get("log_returns") or ())]
        survivors = state.get("gauntlet_survivors")
        self._gauntlet_survivors = None if survivors is None else int(survivors)
        gauntlet_at = state.get("gauntlet_at")
        # Restored WITH its timestamp, never stamped fresh on load. Reading a
        # stale verdict back in and calling it current would make a restart
        # the way to refresh a gauntlet result without running one.
        self._gauntlet_at = None if gauntlet_at is None else float(gauntlet_at)
        self.started_at = float(state.get("started_at", time.time()))
        return True

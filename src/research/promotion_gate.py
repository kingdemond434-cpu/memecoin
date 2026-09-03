"""Thresholds declared before results are seen, and hashed so they cannot move.

Every trading system reaches the same moment: the forward numbers arrive, they
are almost good enough, and the bar quietly becomes what the numbers happen to
be. Nobody experiences that as dishonesty. It feels like judgement, the
adjustment is always defensible in isolation, and the result is a system
promoted on a standard that was written after the fact to fit it.

So the criteria are frozen. A `PromotionCriteria` is content-hashed at
declaration, the hash is recorded in every evaluation, and an evaluation is
only comparable to another if the hashes match. Changing a threshold does not
alter the old verdict -- it produces a new criteria set with a new hash, and
the record shows both. The point is not that thresholds can never change; it
is that a change is visible as a change rather than absorbed into a story
about what the numbers meant.

This module governs the path to forward fills. It does not enable live
capital, and nothing here can: the `ALLOW_LIVE_TRADING` acknowledgement and
`dry_run` remain outside it entirely, and a PASS is a statement that the
evidence cleared a pre-declared bar, not an instruction to trade.
"""

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

PROMOTION_GATE_SCHEMA_VERSION = "v1"


class Stage(Enum):
    """Each stage buys evidence the previous one could not."""

    HISTORICAL = "historical"
    CHRONOLOGICAL_OOS = "chronological_oos"
    FORWARD_SHADOW = "forward_shadow"
    CANARY = "canary"
    LIVE = "live"


STAGE_ORDER = [Stage.HISTORICAL, Stage.CHRONOLOGICAL_OOS, Stage.FORWARD_SHADOW,
               Stage.CANARY, Stage.LIVE]


@dataclass(frozen=True)
class PromotionCriteria:
    """What must be true to advance, fixed at declaration time."""

    stage: Stage
    #: How long the desk must sit AT this stage before it may leave it.
    #:
    #: Without this the ladder collapses. The evidence pool is cumulative,
    #: so a desk that has banked 70,000 shadow decisions satisfies every
    #: rung's volume gate at once and climbs from the bottom to CANARY in
    #: three consecutive sweeps -- about four minutes -- reaching authority
    #: to spend real money having observed nothing forward at all. Measured
    #: on this exact repository before this field existed.
    #:
    #: Zero for the two historical stages, which are backtests and buy
    #: nothing by waiting. Non-zero for FORWARD_SHADOW, whose entire purpose
    #: is forward observation, and for CANARY, whose purpose is to find out
    #: what real execution does.
    min_days_at_stage: float = 0.0
    #: Decisions taken SINCE ENTERING this stage. The lifetime count is
    #: already banked and proves nothing about the stage being left.
    min_decisions_at_stage: int = 0
    min_decisions: int = 0
    min_real_fills: int = 0
    min_launch_cohorts: int = 0
    min_regimes: int = 1
    min_net_log_growth: float = 0.0
    max_rug_loss_share: float = 1.0
    min_monster_enrichment: float = 1.0
    min_execution_success: float = 0.0
    max_catastrophic_failures: int = 0
    #: The point estimate is what happened; the lower bound is the worst the
    #: evidence is consistent with, and only the second is a reason to size a
    #: position. A book whose mean log growth is +0.02 on a bound of -0.31 has
    #: not shown anything, and every volume gate above can be satisfied by
    #: such a book. See `src/research/gauntlet.py`.
    require_positive_lower_bound: bool = False
    #: How many mechanisms must have passed the whole gauntlet -- lower bound,
    #: latency headroom, cost headroom, regime and source-family holdouts,
    #: decay, and probability of backtest overfitting. Zero survivors means the
    #: desk has machinery rather than an edge, which is a fine place to be and
    #: not a place to spend from.
    min_gauntlet_survivors: int = 0

    @property
    def fingerprint(self) -> str:
        """Content hash. Two criteria sets are the same bar only if this matches."""
        payload = json.dumps({**asdict(self), "stage": self.stage.value},
                             sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "stage": self.stage.value,
                "fingerprint": self.fingerprint}


#: The shipped bars. Deliberately conservative and deliberately in source
#: control, so a change to one is a reviewable diff rather than a runtime
#: argument nobody sees.
DEFAULT_CRITERIA: Dict[Stage, PromotionCriteria] = {
    # The bottom rung, which had NO criteria at all -- so a fresh desk sat at
    # HISTORICAL with nothing to pass, and the ladder was unclimbable from
    # its own first step. Deliberately the weakest bar on the ladder: it asks
    # only whether the thing works at all on history the desk did not choose,
    # and every rung above it is where the evidence that matters lives. It is
    # a bar rather than a formality because a model that cannot clear this
    # has no business being measured forward for a fortnight to find out.
    Stage.HISTORICAL: PromotionCriteria(
        stage=Stage.HISTORICAL,
        min_decisions=100, min_launch_cohorts=25, min_regimes=1,
        min_net_log_growth=0.0, max_rug_loss_share=0.30,
        min_monster_enrichment=1.2,
    ),
    Stage.CHRONOLOGICAL_OOS: PromotionCriteria(
        stage=Stage.CHRONOLOGICAL_OOS,
        min_decisions=500, min_launch_cohorts=100, min_regimes=2,
        min_net_log_growth=0.0, max_rug_loss_share=0.25,
        min_monster_enrichment=1.5,
    ),
    Stage.FORWARD_SHADOW: PromotionCriteria(
        stage=Stage.FORWARD_SHADOW,
        # The rung that has to cost real time. Its whole purpose is forward
        # observation across regimes, and a fortnight is the shortest window
        # in which "three regimes" means three genuinely different markets
        # rather than three labels applied inside one afternoon.
        min_days_at_stage=14.0, min_decisions_at_stage=2_000,
        min_decisions=5_000, min_launch_cohorts=1_000, min_regimes=3,
        min_net_log_growth=0.0, max_rug_loss_share=0.20,
        min_monster_enrichment=2.0, min_execution_success=0.0,
    ),
    Stage.CANARY: PromotionCriteria(
        stage=Stage.CANARY,
        # Real money, deliberately slowly. The fills are the point and they
        # cannot be hurried: this stage exists to learn what execution
        # actually does, and that is a thing only elapsed time reveals.
        min_days_at_stage=14.0, min_decisions_at_stage=2_000,
        min_decisions=5_000, min_real_fills=1_000, min_launch_cohorts=1_000,
        min_regimes=3, min_net_log_growth=0.0, max_rug_loss_share=0.15,
        min_monster_enrichment=2.0, min_execution_success=0.60,
        max_catastrophic_failures=0,
        # Leaving CANARY means scaling real money. At minimum the book's
        # lower bound must be positive and one mechanism must have survived
        # the gauntlet -- otherwise the desk would be scaling a point
        # estimate.
        require_positive_lower_bound=True, min_gauntlet_survivors=1,
    ),
    Stage.LIVE: PromotionCriteria(
        stage=Stage.LIVE,
        min_decisions=20_000, min_real_fills=5_000, min_launch_cohorts=3_000,
        min_regimes=4, min_net_log_growth=0.0, max_rug_loss_share=0.12,
        min_monster_enrichment=2.5, min_execution_success=0.70,
        max_catastrophic_failures=0,
        # Two, not one. A single survivor is a desk whose entire book is one
        # mechanism, and the day that mechanism decays -- and they all decay --
        # the desk has nothing. Two independent survivors is the smallest
        # number that is a book rather than a bet.
        require_positive_lower_bound=True, min_gauntlet_survivors=2,
    ),
}


@dataclass
class Evidence:
    """What actually happened. Every field Optional; unmeasured is not zero."""

    stage: Stage
    decisions: Optional[int] = None
    real_fills: Optional[int] = None
    launch_cohorts: Optional[int] = None
    regimes_covered: Optional[int] = None
    net_log_growth: Optional[float] = None
    rug_loss_share: Optional[float] = None
    monster_enrichment: Optional[float] = None
    execution_success: Optional[float] = None
    catastrophic_failures: Optional[int] = None
    #: Bootstrap lower confidence bound on mean net log growth.
    net_log_growth_lower_bound: Optional[float] = None
    #: Mechanisms that cleared `gauntlet.Gauntlet`.
    gauntlet_survivors: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "stage": self.stage.value}


@dataclass
class Verdict:
    passed: bool
    stage: Stage
    criteria_fingerprint: str
    failures: List[str] = field(default_factory=list)
    unmeasured: List[str] = field(default_factory=list)
    evaluated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": PROMOTION_GATE_SCHEMA_VERSION,
            "passed": self.passed, "stage": self.stage.value,
            "criteria_fingerprint": self.criteria_fingerprint,
            "failures": list(self.failures), "unmeasured": list(self.unmeasured),
            "evaluated_at": self.evaluated_at,
        }


def evaluate(criteria: PromotionCriteria, evidence: Evidence) -> Verdict:
    """Check evidence against a frozen bar.

    An unmeasured criterion FAILS. Treating "we did not measure the rug-loss
    share" as satisfying "rug-loss share below 15%" is how a gate becomes
    decorative, and it fails in the direction that promotes.
    """
    if evidence.stage is not criteria.stage:
        return Verdict(False, criteria.stage, criteria.fingerprint,
                       failures=[f"evidence is for {evidence.stage.value}, "
                                 f"criteria are for {criteria.stage.value}"])

    failures: List[str] = []
    unmeasured: List[str] = []

    def check_min(name: str, value, minimum, label: str) -> None:
        if minimum in (0, 0.0) and value is None:
            return  # nothing was required, so nothing had to be measured
        if value is None:
            unmeasured.append(name)
            failures.append(f"{name} was not measured; required {label} {minimum}")
        elif value < minimum:
            failures.append(f"{name} {value} below required {minimum}")

    def check_max(name: str, value, maximum, label: str) -> None:
        if value is None:
            unmeasured.append(name)
            failures.append(f"{name} was not measured; required {label} {maximum}")
        elif value > maximum:
            failures.append(f"{name} {value} above allowed {maximum}")

    check_min("decisions", evidence.decisions, criteria.min_decisions, ">=")
    check_min("real_fills", evidence.real_fills, criteria.min_real_fills, ">=")
    check_min("launch_cohorts", evidence.launch_cohorts, criteria.min_launch_cohorts, ">=")
    check_min("regimes_covered", evidence.regimes_covered, criteria.min_regimes, ">=")
    check_min("execution_success", evidence.execution_success,
              criteria.min_execution_success, ">=")
    check_min("monster_enrichment", evidence.monster_enrichment,
              criteria.min_monster_enrichment, ">=")

    # Growth must be measured and must clear the bar. A book with unknown
    # forward growth has not earned anything.
    if evidence.net_log_growth is None:
        unmeasured.append("net_log_growth")
        failures.append("net_log_growth was not measured")
    elif evidence.net_log_growth <= criteria.min_net_log_growth:
        failures.append(f"net_log_growth {evidence.net_log_growth} does not exceed "
                        f"{criteria.min_net_log_growth}")

    # The lower bound, checked separately from the point estimate, because a
    # book can clear every count and every mean while being indistinguishable
    # from a book with no edge at all.
    if criteria.require_positive_lower_bound:
        if evidence.net_log_growth_lower_bound is None:
            unmeasured.append("net_log_growth_lower_bound")
            failures.append(
                "net_log_growth_lower_bound was not measured; a point "
                "estimate alone cannot authorise capital")
        elif evidence.net_log_growth_lower_bound <= 0:
            failures.append(
                f"net_log_growth_lower_bound "
                f"{evidence.net_log_growth_lower_bound:+.4f} is not positive; "
                "the mean being positive is not evidence of an edge")

    check_min("gauntlet_survivors", evidence.gauntlet_survivors,
              criteria.min_gauntlet_survivors, ">=")

    check_max("rug_loss_share", evidence.rug_loss_share, criteria.max_rug_loss_share, "<=")
    check_max("catastrophic_failures", evidence.catastrophic_failures,
              criteria.max_catastrophic_failures, "<=")

    return Verdict(passed=not failures, stage=criteria.stage,
                   criteria_fingerprint=criteria.fingerprint,
                   failures=failures, unmeasured=sorted(set(unmeasured)))


def next_stage(current: Stage) -> Optional[Stage]:
    index = STAGE_ORDER.index(current)
    return STAGE_ORDER[index + 1] if index + 1 < len(STAGE_ORDER) else None


def can_advance(current: Stage, verdict: Verdict) -> bool:
    """A pass at one stage advances exactly one stage. Never two.

    Skipping is how a promising backtest reaches real money without ever
    having produced a real fill.
    """
    return verdict.passed and verdict.stage is current and next_stage(current) is not None


@dataclass
class PromotionLedger:
    """Every verdict, kept including the failures.

    The rejections are the record that matters: they are what makes a later
    threshold change visible as a change rather than absorbed into a story
    about what the numbers meant.
    """

    path: Path

    def record(self, verdict: Verdict, evidence: Evidence,
               criteria: PromotionCriteria) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"verdict": verdict.to_dict(), "evidence": evidence.to_dict(),
               "criteria": criteria.to_dict()}
        with self.path.open("a") as handle:
            handle.write(json.dumps(row, default=str) + "\n")

    def history(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        return rows

    def fingerprints_for(self, stage: Stage) -> List[str]:
        """Distinct criteria fingerprints ever used at a stage.

        More than one means the bar moved. That is not forbidden, but it must
        be visible, because a moved bar and a cleared bar look identical in a
        verdict alone.
        """
        seen: List[str] = []
        for row in self.history():
            verdict = row.get("verdict") or {}
            if verdict.get("stage") != stage.value:
                continue
            fingerprint = verdict.get("criteria_fingerprint")
            if fingerprint and fingerprint not in seen:
                seen.append(fingerprint)
        return seen

    def bar_moved(self, stage: Stage) -> bool:
        return len(self.fingerprints_for(stage)) > 1

    # --- the part that makes the ladder load-bearing ---------------------

    @property
    def _stage_path(self) -> Path:
        return self.path.with_name(self.path.stem + "_stage.json")

    def _stage_record(self) -> Dict[str, Any]:
        try:
            return json.loads(self._stage_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def entered_stage_at(self) -> float:
        """When the desk arrived at its current rung, or 0 if never recorded.

        Zero means "no record", and the caller treats that as having just
        arrived rather than as having been here for ever -- the direction
        that withholds promotion rather than granting it.
        """
        try:
            return float(self._stage_record().get("at", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def decisions_at_entry(self) -> int:
        try:
            return int(self._stage_record().get("decisions_at_entry", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def current_stage(self) -> Stage:
        """The stage this desk has EARNED, read from disk.

        On disk rather than in memory, because a stage that resets on
        restart is a desk that silently returns to trading without
        authorisation -- or, in the other direction, forgets that it earned
        the right to. Neither is survivable in something that decides on its
        own whether to spend money.
        """
        try:
            payload = json.loads(self._stage_path.read_text(encoding="utf-8"))
            return Stage(str(payload.get("stage", "")))
        except (OSError, ValueError, KeyError):
            # No record is the FIRST stage, never the last. A missing or
            # corrupt file must never read as authorisation.
            return STAGE_ORDER[0]

    def _write_stage(self, stage: Stage, reason: str,
                     decisions_at_entry: int = 0) -> None:
        payload = {"stage": stage.value, "reason": reason, "at": time.time(),
                   # The two numbers that make "time and evidence AT this
                   # stage" answerable after a restart. Without them the
                   # ledger could only ever ask about lifetime totals, which
                   # are already banked and prove nothing about the rung
                   # being left.
                   "decisions_at_entry": int(decisions_at_entry)}
        self._stage_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._stage_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.replace(self._stage_path)

    def submit(self, evidence: Evidence,
               criteria: Optional[PromotionCriteria] = None) -> Verdict:
        """Evaluate evidence at the CURRENT stage, record it, and advance on a pass.

        One stage per pass, never two: each stage buys evidence the previous
        one could not, and skipping one means promoting on evidence that was
        never gathered.
        """
        stage = self.current_stage()
        criteria = criteria or DEFAULT_CRITERIA.get(stage)
        if criteria is None:  # pragma: no cover - every stage has criteria
            return Verdict(False, stage, "", failures=["no criteria for stage"])
        verdict = evaluate(criteria, evidence)
        # Time and evidence AT THIS STAGE, which the criteria alone cannot
        # see: `evaluate` is handed a cumulative snapshot and has no idea
        # when the desk arrived. Checked here, where the arrival is known.
        for failure in self._stage_dwell_failures(criteria, evidence):
            verdict.failures.append(failure)
            verdict.passed = False
        self.record(verdict, evidence, criteria)
        if can_advance(stage, verdict):
            earned = next_stage(stage)
            self._write_stage(earned, f"passed {stage.value}",
                              decisions_at_entry=int(evidence.decisions or 0))
            logger.warning(
                "PROMOTION: %s -> %s on %d decisions, %d real fills, "
                "net log growth %s. The desk's trading authority has CHANGED.",
                stage.value, earned.value, evidence.decisions,
                evidence.real_fills, evidence.net_log_growth)
        return verdict

    def _stage_dwell_failures(self, criteria: PromotionCriteria,
                              evidence: Evidence) -> List[str]:
        """Why this rung may not be left yet, on time and fresh evidence.

        A first-ever submission has no recorded arrival. That reads as
        arriving NOW, so a desk cannot be promoted on its first sweep by
        having no history -- which is the same failure as reading a missing
        stage file as authorisation, one level in.
        """
        failures: List[str] = []
        if criteria.min_days_at_stage > 0:
            entered = self.entered_stage_at()
            if not entered:
                self._write_stage(criteria.stage, "arrival recorded",
                                  decisions_at_entry=int(evidence.decisions or 0))
                entered = time.time()
            days = (time.time() - entered) / 86_400.0
            if days < criteria.min_days_at_stage:
                failures.append(
                    f"{days:.2f} days at {criteria.stage.value}; "
                    f"{criteria.min_days_at_stage:g} required. The evidence "
                    "below is cumulative and was mostly banked at earlier "
                    "stages; this rung is bought with elapsed observation")
        if criteria.min_decisions_at_stage > 0:
            fresh = int(evidence.decisions or 0) - self.decisions_at_entry()
            if fresh < criteria.min_decisions_at_stage:
                failures.append(
                    f"{fresh} decisions since entering {criteria.stage.value}; "
                    f"{criteria.min_decisions_at_stage} required")
        return failures

    #: Requirements that arrive with VOLUME, and can therefore be projected
    #: from an observed rate. Everything else on the ladder either arrives
    #: with the market (regimes), or has to be earned by picking better
    #: (enrichment, growth, the lower bound, the gauntlet) -- and projecting
    #: a date for those would be forecasting the desk becoming good at its
    #: job, which is not a thing a rate can say.
    _COUNTING = ("decisions", "real_fills", "launch_cohorts")

    def eta(self, evidence: Evidence, observed_days: float,
            criteria: Optional[PromotionCriteria] = None) -> Dict[str, Any]:
        """How long until the next rung, or what is stopping it entirely.

        Two very different answers, kept apart on purpose. The counting
        requirements have a rate and therefore a date. The rest do not: a
        desk short of three regimes is waiting on the market, and a desk
        below 2.0 monster enrichment is waiting on itself. Reporting one
        number over both would put a confident date on the requirement least
        likely to be met by waiting.
        """
        stage = self.current_stage()
        criteria = criteria or DEFAULT_CRITERIA.get(stage)
        if criteria is None:
            return {"status": "DATA_BLOCKED", "detail": f"no criteria for {stage.value}"}
        target = next_stage(stage)
        if target is None:
            return {"status": "OK", "stage": stage.value, "next_stage": None,
                    "detail": "top of the ladder"}

        observed_days = max(0.0, float(observed_days))
        entered = self.entered_stage_at()
        days_at_stage = ((time.time() - entered) / 86_400.0) if entered else 0.0
        dwell_remaining = max(0.0, criteria.min_days_at_stage - days_at_stage)

        counting: Dict[str, Any] = {}
        projected = [dwell_remaining]
        unrated: List[str] = []
        for name, need in (
                ("decisions", criteria.min_decisions),
                ("real_fills", criteria.min_real_fills),
                ("launch_cohorts", criteria.min_launch_cohorts)):
            if need <= 0:
                continue
            have = int(getattr(evidence, name) or 0)
            per_day = (have / observed_days) if observed_days > 0 else 0.0
            short = max(0, need - have)
            if short == 0:
                days = 0.0
            elif per_day > 0:
                days = short / per_day
            else:
                # No observed rate. Not "arrives instantly", and not a number
                # to average into a date -- named instead, so the answer says
                # which count is not moving.
                days = None
                unrated.append(name)
            counting[name] = {"have": have, "need": need,
                              "per_day": round(per_day, 3),
                              "days_remaining": (None if days is None
                                                 else round(days, 1))}
            if days is not None:
                projected.append(days)

        if criteria.min_decisions_at_stage > 0:
            fresh = int(evidence.decisions or 0) - self.decisions_at_entry()
            short = max(0, criteria.min_decisions_at_stage - fresh)
            per_day = ((fresh / days_at_stage) if days_at_stage > 0 else 0.0)
            days = (0.0 if short == 0 else
                    (short / per_day) if per_day > 0 else None)
            counting["decisions_at_stage"] = {
                "have": fresh, "need": criteria.min_decisions_at_stage,
                "per_day": round(per_day, 3),
                "days_remaining": None if days is None else round(days, 1)}
            if days is None:
                unrated.append("decisions_at_stage")
            else:
                projected.append(days)

        verdict = evaluate(criteria, evidence)
        blocking = [failure for failure in verdict.failures
                    if not any(failure.startswith(name) for name in self._COUNTING)]

        days_needed = max(projected) if projected else 0.0
        slowest = max(
            (name for name, row in counting.items()
             if row["days_remaining"] is not None),
            key=lambda name: counting[name]["days_remaining"], default=None)
        if dwell_remaining >= days_needed and dwell_remaining > 0:
            slowest = "days_at_stage"
        elif days_needed <= 0:
            slowest = None

        return {
            "schema_version": PROMOTION_GATE_SCHEMA_VERSION,
            "status": "OK",
            "stage": stage.value, "next_stage": target.value,
            "eligible_now": verdict.passed and dwell_remaining <= 0,
            "days_at_stage": round(days_at_stage, 2),
            "dwell_days_remaining": round(dwell_remaining, 2),
            "observed_days": round(observed_days, 2),
            "counting": counting,
            "counting_without_a_rate": sorted(set(unrated)),
            "days_until_counts_are_met": (
                None if unrated else round(days_needed, 1)),
            "slowest_count": slowest,
            # What no amount of waiting fixes.
            "blocking_regardless_of_time": blocking,
            "detail": (
                "eligible now" if verdict.passed and dwell_remaining <= 0 else
                f"{len(blocking)} requirement(s) will not be met by waiting"
                if blocking else
                "only counts and elapsed time remain"),
        }

    def demote(self, reason: str) -> Stage:
        """Drop one stage. The only thing that ever lowers trading authority.

        Deliberately not automatic on a failed verdict: evidence windows are
        noisy and a desk that flaps between authorised and not would trade
        on the noise. This is for the cases that are not noise -- a
        catastrophic failure, or an operator deciding the thing is
        misbehaving -- and it is one stage, so recovering means passing the
        gate again rather than being handed back what was taken away.
        """
        stage = self.current_stage()
        index = STAGE_ORDER.index(stage)
        lower = STAGE_ORDER[max(0, index - 1)]
        self._write_stage(lower, f"demoted: {reason}")
        logger.error("DEMOTION: %s -> %s (%s)", stage.value, lower.value, reason)
        return lower

    def authorises_live_capital(self) -> Tuple[bool, str]:
        """Whether the EARNED stage permits spending real money, and why not.

        CANARY is the first stage that does. Everything below it is
        measurement, and the whole point of the ladder is that the
        measurement happens before the money rather than after it.
        """
        stage = self.current_stage()
        index = STAGE_ORDER.index(stage)
        canary = STAGE_ORDER.index(Stage.CANARY)
        if index >= canary:
            return True, f"stage {stage.value} authorises live capital"
        return False, (
            f"stage is {stage.value}; live capital requires "
            f"{Stage.CANARY.value}, which is reached by passing the "
            f"{stage.value} gate on measured evidence")

    def status(self) -> Dict[str, Any]:
        stage = self.current_stage()
        authorised, reason = self.authorises_live_capital()
        history = self.history()
        return {
            "schema_version": PROMOTION_GATE_SCHEMA_VERSION,
            "earned_stage": stage.value,
            "next_stage": (next_stage(stage).value
                           if next_stage(stage) else None),
            "authorises_live_capital": authorised,
            "authorisation_detail": reason,
            "verdicts_recorded": len(history),
            "bar_moved_at_this_stage": self.bar_moved(stage),
            "last_verdict": (history[-1]["verdict"] if history else None),
        }

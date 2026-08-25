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
from typing import Any, Dict, List, Optional, Sequence

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
    min_decisions: int = 0
    min_real_fills: int = 0
    min_launch_cohorts: int = 0
    min_regimes: int = 1
    min_net_log_growth: float = 0.0
    max_rug_loss_share: float = 1.0
    min_monster_enrichment: float = 1.0
    min_execution_success: float = 0.0
    max_catastrophic_failures: int = 0

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
    Stage.CHRONOLOGICAL_OOS: PromotionCriteria(
        stage=Stage.CHRONOLOGICAL_OOS,
        min_decisions=500, min_launch_cohorts=100, min_regimes=2,
        min_net_log_growth=0.0, max_rug_loss_share=0.25,
        min_monster_enrichment=1.5,
    ),
    Stage.FORWARD_SHADOW: PromotionCriteria(
        stage=Stage.FORWARD_SHADOW,
        min_decisions=5_000, min_launch_cohorts=1_000, min_regimes=3,
        min_net_log_growth=0.0, max_rug_loss_share=0.20,
        min_monster_enrichment=2.0, min_execution_success=0.0,
    ),
    Stage.CANARY: PromotionCriteria(
        stage=Stage.CANARY,
        min_decisions=5_000, min_real_fills=1_000, min_launch_cohorts=1_000,
        min_regimes=3, min_net_log_growth=0.0, max_rug_loss_share=0.15,
        min_monster_enrichment=2.0, min_execution_success=0.60,
        max_catastrophic_failures=0,
    ),
    Stage.LIVE: PromotionCriteria(
        stage=Stage.LIVE,
        min_decisions=20_000, min_real_fills=5_000, min_launch_cohorts=3_000,
        min_regimes=4, min_net_log_growth=0.0, max_rug_loss_share=0.12,
        min_monster_enrichment=2.5, min_execution_success=0.70,
        max_catastrophic_failures=0,
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

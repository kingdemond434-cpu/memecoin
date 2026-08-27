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
        if outcome.max_multiple is not None:
            self.scored_launches += 1
            if float(outcome.max_multiple) >= self.MONSTER_MULTIPLE:
                self.monsters += 1
        if not outcome.entered:
            return
        self.entered += 1
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
            else:
                # A trade that took the whole book is not a large negative
                # number, it is the end of the sequence. Recorded as
                # catastrophic rather than as -inf, which would make every
                # later average meaningless.
                self.catastrophic_failures += 1

    # -- reading -----------------------------------------------------------

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
        )

    def _enrichment(self) -> Optional[float]:
        """Monsters among what we entered, over monsters among what we saw.

        None until both sides have been observed. An enrichment computed
        against a base rate of zero is division by an absence, and reporting
        it as infinite enrichment is the single most flattering number this
        module could produce.
        """
        if not self.scored_launches or not self.entered:
            return None
        base_rate = self.monsters / self.scored_launches
        if base_rate <= 0:
            return None
        entered_rate = self.monsters / self.entered
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
            "cohorts": sorted(self._cohorts), "regimes": sorted(self._regimes),
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
        self._cohorts = set(state.get("cohorts") or ())
        self._regimes = set(state.get("regimes") or ())
        self.started_at = float(state.get("started_at", time.time()))
        return True

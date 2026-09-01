"""The sell must be ready before it is needed, and modes must not be blended.

Two findings from builders who ran real capital, both of which this desk had
only partially.

**An exit must never be slower than an entry.** A desk that constructs its
sell transaction when the exit signal fires is discovering program accounts
while the rug is happening. Every field of the sell -- accounts, program ids,
ATAs, the instruction layout -- is known the moment the buy fills; only the
blockhash and the reserves are not. So the template is built at fill time and
refreshed, and `time_to_exit_ready` is a measured invariant rather than an
aspiration: if it is not ~0 after a fill, that is a defect with a number
attached.

**One exit policy cannot serve two return distributions.** The same builders
kept rediscovering that most tokens are +30% and gone, while rare ones are
+300% on their way to +3000%, and that a policy tuned for either destroys the
other. Banking the fast ones early is right; banking the monster early is the
single most expensive mistake available. So the mode is CHOSEN, per position,
from evidence, and the choice is recorded -- rather than a compromise policy
that is wrong for both.

The mode is not a guess about the future. It is a reading of what the
position has already shown: whether independent demand absorbed the opening
cohort's supply, whether the skilled cohort is still holding, whether late
chasers are the marginal buyer. Those are the cohort readings, which is why
this module consumes them.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

EXIT_READINESS_SCHEMA_VERSION = "v1"

#: A sell template older than this is stale: the blockhash it was built
#: against may no longer be valid.
TEMPLATE_STALE_S = 45.0

#: Exit readiness worse than this after a fill is a defect, not a slow day.
MAX_ACCEPTABLE_READY_MS = 250.0

#: Minimum favourable excursion over adverse excursion for an entry state to
#: be worth repeating. Below this, the entry is paying more in drawdown than
#: it collects in upside even when it wins.
MIN_MFE_MAE_RATIO = 2.0

#: Observations before an entry state's excursion profile means anything.
MIN_EXCURSION_SAMPLES = 30


@dataclass
class SellTemplate:
    """Everything about the exit that is knowable at fill time."""

    token: str
    built_at: float
    accounts: Tuple[str, ...] = ()
    program_id: str = ""
    #: Refreshed separately and cheaply; the rest never changes.
    blockhash: str = ""
    blockhash_at: float = 0.0
    ready: bool = False
    detail: str = ""

    def stale(self, now: Optional[float] = None) -> bool:
        moment = time.time() if now is None else now
        return (moment - self.blockhash_at) > TEMPLATE_STALE_S

    def usable(self, now: Optional[float] = None) -> bool:
        return bool(self.ready and self.accounts and self.program_id
                    and not self.stale(now))


class ExitReadinessLedger:
    """Proves the sell was ready, per position, with a number.

    A ledger rather than a flag because "we build the sell early" is a claim,
    and the only version of it worth having is one that fails loudly when it
    stops being true.
    """

    def __init__(self):
        self.templates: Dict[str, SellTemplate] = {}
        self._ready_ms: List[float] = []
        self.late = 0
        self.missing = 0

    def on_fill(self, token: str, filled_at: float, template: SellTemplate) -> float:
        """Record how long after the fill the exit became executable."""
        self.templates[token] = template
        elapsed_ms = max(0.0, (template.built_at - filled_at) * 1000.0)
        self._ready_ms.append(elapsed_ms)
        self._ready_ms = self._ready_ms[-2048:]
        if elapsed_ms > MAX_ACCEPTABLE_READY_MS:
            self.late += 1
            logger.warning(
                "EXIT READINESS %s took %.0fms after fill (limit %.0fms); the "
                "sell is being built too late to be useful in a rug",
                token, elapsed_ms, MAX_ACCEPTABLE_READY_MS)
        return elapsed_ms

    def template_for(self, token: str, now: Optional[float] = None
                     ) -> Optional[SellTemplate]:
        template = self.templates.get(token)
        if template is None:
            self.missing += 1
            return None
        if not template.usable(now):
            return None
        return template

    def report(self) -> Dict[str, Any]:
        if not self._ready_ms:
            return {"status": "DATA_BLOCKED", "schema": EXIT_READINESS_SCHEMA_VERSION,
                    "detail": "no fills observed yet"}
        ordered = sorted(self._ready_ms)
        p50 = ordered[len(ordered) // 2]
        p95 = ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))]
        return {
            "status": "OK" if self.late == 0 else "DEGRADED",
            "schema": EXIT_READINESS_SCHEMA_VERSION,
            "fills": len(ordered),
            "ready_p50_ms": round(p50, 2),
            "ready_p95_ms": round(p95, 2),
            "late_fills": self.late,
            "templates_missing_at_exit": self.missing,
            "detail": ("an exit slower than its entry is a position that "
                       "discovers its own accounts during a rug"),
        }


@dataclass
class ExcursionProfile:
    """What an entry state's positions did at their best and worst."""

    status: str
    samples: int = 0
    mfe_median: Optional[float] = None
    mae_median: Optional[float] = None
    detail: str = ""

    @property
    def ratio(self) -> Optional[float]:
        if self.mfe_median is None or not self.mae_median:
            return None
        return float(self.mfe_median / abs(self.mae_median))

    @property
    def worth_repeating(self) -> Optional[bool]:
        ratio = self.ratio
        return None if ratio is None else ratio >= MIN_MFE_MAE_RATIO


class ExcursionLedger:
    """MFE and MAE per entry state, which win rate cannot see.

    An entry that wins 80% of the time by risking 40% drawdown to capture 10%
    is a worse entry than one that wins 45% while never drawing down more
    than 5%, and every win-rate table ranks them the other way round. The
    excursions are what a position actually put the desk through.

    Both are measured on EXECUTABLE marks -- the price a sale could have
    been filled at -- for the same reason the label fix mattered: the highest
    price a token printed is not a price anyone could have sold into.
    """

    def __init__(self):
        self._by_state: Dict[str, List[Tuple[float, float]]] = {}

    def record(self, state_key: str, mfe: float, mae: float) -> None:
        self._by_state.setdefault(str(state_key), []).append(
            (float(mfe), float(mae)))

    def profile(self, state_key: str) -> ExcursionProfile:
        rows = self._by_state.get(str(state_key), [])
        if len(rows) < MIN_EXCURSION_SAMPLES:
            return ExcursionProfile(
                status="DATA_BLOCKED", samples=len(rows),
                detail=(f"{len(rows)} positions in this entry state, below "
                        f"{MIN_EXCURSION_SAMPLES}"))
        favourable = sorted(row[0] for row in rows)
        adverse = sorted(row[1] for row in rows)
        return ExcursionProfile(
            status="OK", samples=len(rows),
            mfe_median=favourable[len(favourable) // 2],
            mae_median=adverse[len(adverse) // 2],
            detail=f"{len(rows)} positions")

    def report(self) -> Dict[str, Any]:
        profiles = {key: self.profile(key) for key in self._by_state}
        measured = {k: v for k, v in profiles.items() if v.status == "OK"}
        return {
            "status": "OK" if measured else "DATA_BLOCKED",
            "states": len(profiles),
            "states_measured": len(measured),
            "by_state": {
                key: {"samples": p.samples, "mfe": p.mfe_median,
                      "mae": p.mae_median, "ratio": p.ratio,
                      "worth_repeating": p.worth_repeating}
                for key, p in sorted(measured.items())},
        }


#: The two return distributions, kept apart.
MODE_RECYCLER = "FAST_RECYCLER"
MODE_MONSTER = "MONSTER_HOLD"
MODE_UNDECIDED = "UNDECIDED"


@dataclass
class ModeChoice:
    mode: str
    reasons: List[str] = field(default_factory=list)
    detail: str = ""


def choose_exit_mode(cohort_report: Optional[Any],
                     monster_probability: Optional[float] = None,
                     ) -> ModeChoice:
    """Which distribution this position is in, from what it has already shown.

    UNDECIDED is a real answer and the default. A position whose cohort
    readings are blocked has shown nothing yet, and forcing a mode from no
    evidence would either bank a monster early or hold a dud through zero --
    the two most expensive mistakes available, one of which this would make
    every time.

    Note the asymmetry, which is deliberate: MONSTER_HOLD requires positive
    evidence (independent absorption, a cohort still holding), while
    FAST_RECYCLER is chosen on evidence of deterioration OR simply on the
    absence of anything exceptional. Holding is the expensive default; taking
    profit is the cheap one.
    """
    reasons: List[str] = []
    if cohort_report is None or getattr(cohort_report, "status", "") != "OK":
        return ModeChoice(
            mode=MODE_UNDECIDED,
            detail="no cohort evidence yet; neither mode is chosen from nothing")

    absorption = getattr(cohort_report, "absorption", None)
    chasers = getattr(cohort_report, "chasers", None)
    verdict = getattr(absorption, "verdict", "DATA_BLOCKED") if absorption else "DATA_BLOCKED"

    if verdict == "ABSORBED":
        reasons.append("independent demand absorbed the opening cohort's supply")
    elif verdict == "CAPTURED":
        reasons.append("the buyers absorbing supply are related to the sellers; "
                       "that is inventory moving, not demand")
    elif verdict == "FAILED":
        reasons.append("the opening cohort's supply was not absorbed")

    retained = None
    retention = getattr(cohort_report, "retention", {}) or {}
    for depth in sorted(retention):
        reading = retention[depth]
        if getattr(reading, "status", "") == "OK" and reading.retained:
            latest = max(reading.retained)
            retained = reading.retained[latest]
            break
    if retained is not None and retained >= 0.7:
        reasons.append(f"the opening cohort still holds {retained:.0%}")
    elif retained is not None and retained < 0.4:
        reasons.append(f"the opening cohort has cut to {retained:.0%}")

    if chasers is not None and getattr(chasers, "is_distribution_pattern", False):
        reasons.append("late chasers are the marginal buyer while skilled "
                       "wallets exit")
        return ModeChoice(mode=MODE_RECYCLER, reasons=reasons,
                          detail="distribution pattern; bank into the chase")

    if verdict in {"FAILED", "CAPTURED"}:
        return ModeChoice(mode=MODE_RECYCLER, reasons=reasons,
                          detail="supply was not independently absorbed")

    strong = (verdict == "ABSORBED" and (retained is None or retained >= 0.5))
    if strong and (monster_probability is None or monster_probability >= 0.1):
        return ModeChoice(mode=MODE_MONSTER, reasons=reasons,
                          detail="absorbed supply with the cohort still in")

    return ModeChoice(mode=MODE_RECYCLER, reasons=reasons or ["nothing exceptional shown"],
                      detail="the expensive default is holding; this is not that case")

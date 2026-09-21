"""The gauntlet: what a mechanism has to survive before it is called an edge.

The promotion ladder already asks for volume -- decisions, fills, cohorts,
regimes -- and volume is the easy half. A mechanism can clear every count and
still be nothing: a point estimate that is positive because of four launches,
an edge that lives entirely in one source family, a return that evaporates when
entry slips from 100ms to 1s, a backtest champion selected out of thirty
candidates on the same data.

Each of those has a specific test, and this module is those tests. A mechanism
is a SURVIVOR only if it passes all of them; anything else is KILL or FRAGILE,
and DATA_BLOCKED when there is not enough evidence to say.

**The lower bound, not the point estimate.** Every gate here is evaluated on a
bootstrap lower confidence bound of mean log growth. A point estimate answers
"what happened"; the lower bound answers "what is the worst this is consistent
with", and only the second one is a reason to size a position. The bootstrap is
seeded, so a verdict is reproducible rather than a die roll.

**Selection is priced.** `probability_of_backtest_overfitting` runs CSCV: split
the record into blocks, take every balanced in-sample/out-of-sample partition,
find the in-sample winner, and ask how often it lands below the median
out-of-sample. Picking the best of thirty candidates on one dataset and
reporting its Sharpe is not evidence, and PBO is what says so numerically.

**Robustness is directional.** Latency survival asks how far entry can slip
before the lower bound crosses zero -- a mechanism profitable only at 100ms is
a mechanism this desk cannot trade. Cost stress multiplies execution cost until
it dies. Leave-one-regime-out and leave-one-source-family-out both ask whether
the edge is a real effect or a single lucky cohort wearing a mechanism's name.
Decay compares the older half of the record to the newer one, because an edge
that stopped working in April is worse than no edge at all.

Nothing here decides anything. It produces verdicts; the promotion gate reads
them, and the promotion gate is the only thing that can move capital.
"""

from __future__ import annotations

import itertools
import logging
import math
import random
import statistics
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import (Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple)

logger = logging.getLogger(__name__)

GAUNTLET_SCHEMA_VERSION = "v1"

#: Entry latencies every mechanism is scored at. The survival question is not
#: "is it profitable" but "how much delay does it tolerate", because the desk's
#: real latency is a distribution, not its best case.
LATENCY_GRID_S: Tuple[float, ...] = (0.1, 0.5, 1.0, 3.0, 10.0, 30.0)

#: Cost multipliers for the stress pass. 1.0 is what was measured; 2.0 is a bad
#: week of contention and a worse fill.
COST_MULTIPLIERS: Tuple[float, ...] = (1.0, 1.5, 2.0)

#: Below this many observations a mechanism is DATA_BLOCKED. Not a statistical
#: constant -- a floor below which every test in this module is theatre.
DEFAULT_MIN_OBSERVATIONS = 200

#: Bootstrap resamples. Enough that the 5th percentile is stable to about the
#: third decimal, cheap enough to run for every mechanism on every pass.
DEFAULT_BOOTSTRAP = 2_000

#: One-sided confidence level for the lower bound.
DEFAULT_ALPHA = 0.05

#: Seed, so a verdict is a property of the evidence rather than of the moment
#: it was computed.
DEFAULT_SEED = 20260903

#: CSCV blocks. Eight gives 70 balanced partitions -- enough resolution for a
#: PBO estimate, few enough to compute in milliseconds.
DEFAULT_CSCV_BLOCKS = 8

#: A mechanism whose selection is this likely to be overfitting is not an edge
#: however good its point estimate looks.
DEFAULT_MAX_PBO = 0.5

#: Log-return floor, so one total loss does not become negative infinity and
#: swallow every other observation in the mean.
LOG_FLOOR = -4.0


class Verdict(Enum):
    SURVIVOR = "SURVIVOR"
    #: Positive, but it failed at least one robustness test. Tradeable only
    #: with the specific weakness named and sized for.
    FRAGILE = "FRAGILE"
    KILL = "KILL"
    DATA_BLOCKED = "DATA_BLOCKED"
    #: A deliberate negative benchmark -- "every Pump launch" -- carried so the
    #: table has a floor to be measured against.
    CONTROL = "CONTROL"


@dataclass
class Observation:
    """One decision's forward outcome, at every latency it could have had.

    `net_return_by_latency` is net of execution cost. Missing latencies are
    absent rather than zero: an entry that was never feasible at 100ms did not
    return nothing at 100ms, it did not happen, and the difference decides
    whether a latency column is evidence or an artefact.
    """

    mechanism: str
    timestamp: float
    regime: str = "unknown"
    source_family: str = ""
    cohort: str = ""
    net_return_by_latency: Dict[float, Optional[float]] = field(
        default_factory=dict)
    cost_fraction: float = 0.0
    is_control: bool = False

    def net_return(self, latency_s: float, *, cost_multiplier: float = 1.0
                   ) -> Optional[float]:
        base = self.net_return_by_latency.get(latency_s)
        if base is None:
            return None
        if cost_multiplier == 1.0:
            return base
        # The stored return already carries one unit of cost; scaling means
        # charging the extra units on top.
        return base - self.cost_fraction * (cost_multiplier - 1.0)

    def log_growth(self, latency_s: float, *, cost_multiplier: float = 1.0
                   ) -> Optional[float]:
        net = self.net_return(latency_s, cost_multiplier=cost_multiplier)
        if net is None:
            return None
        return max(LOG_FLOOR, math.log(max(1e-9, 1.0 + net)))


def bootstrap_lower_bound(values: Sequence[float], *,
                          alpha: float = DEFAULT_ALPHA,
                          iterations: int = DEFAULT_BOOTSTRAP,
                          seed: int = DEFAULT_SEED) -> Optional[float]:
    """One-sided lower confidence bound on the mean, by percentile bootstrap.

    Deliberately not a t-interval. Log growth over launches is violently
    skewed -- most observations are small losses and a handful are the entire
    return -- and a t-interval on that distribution is wrong in the direction
    that flatters the mechanism.
    """
    sample = [float(value) for value in values if value is not None]
    if len(sample) < 2:
        return None
    rng = random.Random(seed)
    size = len(sample)
    means: List[float] = []
    for _ in range(max(1, int(iterations))):
        total = 0.0
        for _ in range(size):
            total += sample[rng.randrange(size)]
        means.append(total / size)
    means.sort()
    index = int(alpha * len(means))
    return means[min(index, len(means) - 1)]


def max_drawdown(values: Sequence[float]) -> float:
    """Deepest peak-to-trough of cumulative log growth, in log units.

    Computed on the realised order, because a drawdown is a path property and
    shuffling the record destroys precisely the thing being measured.
    """
    peak = 0.0
    running = 0.0
    worst = 0.0
    for value in values:
        running += float(value)
        peak = max(peak, running)
        worst = min(worst, running - peak)
    return worst


def probability_of_backtest_overfitting(
        matrix: Sequence[Sequence[float]], *,
        blocks: int = DEFAULT_CSCV_BLOCKS) -> Optional[float]:
    """CSCV: how often the in-sample winner is a below-median performer OOS.

    `matrix` is candidates x observations, aligned in time. The record is cut
    into `blocks` contiguous blocks; every balanced split of those blocks into
    in-sample and out-of-sample halves is evaluated; the candidate with the
    best in-sample mean is looked up out-of-sample; PBO is the fraction of
    splits where its out-of-sample rank falls below the median.

    Returns None rather than a number when there are too few candidates or too
    short a record -- PBO computed from two candidates is not a diagnosis.
    """
    candidates = [list(row) for row in matrix if row]
    if len(candidates) < 2:
        return None
    length = min(len(row) for row in candidates)
    if length < blocks * 2:
        return None
    edges = [round(index * length / blocks) for index in range(blocks + 1)]
    spans = [range(edges[index], edges[index + 1]) for index in range(blocks)]
    half = blocks // 2
    below = 0
    total = 0
    for chosen in itertools.combinations(range(blocks), half):
        inside = set(chosen)
        outside = [index for index in range(blocks) if index not in inside]
        in_index = [i for block in chosen for i in spans[block]]
        out_index = [i for block in outside for i in spans[block]]
        if not in_index or not out_index:
            continue
        in_means = [statistics.fmean(row[i] for i in in_index)
                    for row in candidates]
        out_means = [statistics.fmean(row[i] for i in out_index)
                     for row in candidates]
        best = max(range(len(candidates)), key=lambda idx: in_means[idx])
        rank = sorted(range(len(candidates)),
                      key=lambda idx: out_means[idx]).index(best)
        # rank is 0 = worst. Below median means the in-sample winner is in the
        # bottom half of the out-of-sample ordering.
        if rank < (len(candidates) - 1) / 2.0:
            below += 1
        total += 1
    return (below / total) if total else None


@dataclass
class GauntletResult:
    """Everything the gauntlet learned about one mechanism."""

    mechanism: str
    n: int = 0
    verdict: Verdict = Verdict.DATA_BLOCKED
    reasons: List[str] = field(default_factory=list)

    net_ev: Optional[float] = None
    e_log_w: Optional[float] = None
    e_log_w_lower: Optional[float] = None

    latency_curve: Dict[float, Optional[float]] = field(default_factory=dict)
    latency_survival_s: Optional[float] = None

    cost_curve: Dict[float, Optional[float]] = field(default_factory=dict)
    cost_survival_multiple: Optional[float] = None

    regimes: Dict[str, Optional[float]] = field(default_factory=dict)
    regime_holdout_min: Optional[float] = None

    source_families: Dict[str, Optional[float]] = field(default_factory=dict)
    family_loo_min: Optional[float] = None

    early_half: Optional[float] = None
    late_half: Optional[float] = None
    decay_status: str = "UNKNOWN"

    max_drawdown: Optional[float] = None
    pbo: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["verdict"] = self.verdict.value
        for key in ("latency_curve", "cost_curve"):
            data[key] = {str(k): v for k, v in getattr(self, key).items()}
        return data


class Gauntlet:
    """Runs every test, and refuses to call anything an edge that fails one."""

    def __init__(self, *, reference_latency_s: float = 1.0,
                 min_observations: int = DEFAULT_MIN_OBSERVATIONS,
                 alpha: float = DEFAULT_ALPHA,
                 bootstrap: int = DEFAULT_BOOTSTRAP,
                 seed: int = DEFAULT_SEED,
                 max_pbo: float = DEFAULT_MAX_PBO,
                 required_regimes: int = 3,
                 latencies: Sequence[float] = LATENCY_GRID_S,
                 cost_multipliers: Sequence[float] = COST_MULTIPLIERS):
        self.reference_latency_s = float(reference_latency_s)
        self.min_observations = int(min_observations)
        self.alpha = float(alpha)
        self.bootstrap = int(bootstrap)
        self.seed = int(seed)
        self.max_pbo = float(max_pbo)
        self.required_regimes = int(required_regimes)
        self.latencies = tuple(latencies)
        self.cost_multipliers = tuple(cost_multipliers)

    # -- helpers ---------------------------------------------------------

    def _growths(self, observations: Sequence[Observation], *,
                 latency: Optional[float] = None,
                 cost_multiplier: float = 1.0) -> List[float]:
        target = self.reference_latency_s if latency is None else latency
        values = [item.log_growth(target, cost_multiplier=cost_multiplier)
                  for item in observations]
        return [value for value in values if value is not None]

    def _lower(self, values: Sequence[float]) -> Optional[float]:
        return bootstrap_lower_bound(values, alpha=self.alpha,
                                     iterations=self.bootstrap, seed=self.seed)

    # -- the tests -------------------------------------------------------

    def latency_survival(self, observations: Sequence[Observation]
                         ) -> Tuple[Dict[float, Optional[float]],
                                    Optional[float]]:
        """Lower bound at each latency, and the last one still positive."""
        curve: Dict[float, Optional[float]] = {}
        survival: Optional[float] = None
        for latency in self.latencies:
            values = self._growths(observations, latency=latency)
            bound = self._lower(values) if len(values) >= 2 else None
            curve[latency] = bound
            if bound is not None and bound > 0:
                survival = latency
            elif bound is not None:
                # The curve is monotone in intent, not in fact; stopping at the
                # first failure reports the delay the edge actually tolerates
                # rather than the largest one it happened to pass.
                break
        return curve, survival

    def cost_stress(self, observations: Sequence[Observation]
                    ) -> Tuple[Dict[float, Optional[float]], Optional[float]]:
        curve: Dict[float, Optional[float]] = {}
        survival: Optional[float] = None
        for multiplier in self.cost_multipliers:
            values = self._growths(observations, cost_multiplier=multiplier)
            bound = self._lower(values) if len(values) >= 2 else None
            curve[multiplier] = bound
            if bound is not None and bound > 0:
                survival = multiplier
            elif bound is not None:
                break
        return curve, survival

    def by_regime(self, observations: Sequence[Observation]
                  ) -> Dict[str, Optional[float]]:
        grouped: Dict[str, List[Observation]] = {}
        for item in observations:
            grouped.setdefault(item.regime, []).append(item)
        return {regime: self._lower(self._growths(rows))
                for regime, rows in sorted(grouped.items())}

    def family_leave_one_out(self, observations: Sequence[Observation]
                             ) -> Dict[str, Optional[float]]:
        """Lower bound with each source family REMOVED.

        The question is not "how did this family do" but "does the edge
        survive without it". A mechanism that is positive only because one
        Telegram cluster carried it is a bet on that cluster, and the moment
        it decays the mechanism does too.
        """
        families = {item.source_family for item in observations
                    if item.source_family}
        if len(families) < 2:
            return {}
        result: Dict[str, Optional[float]] = {}
        for family in sorted(families):
            remainder = [item for item in observations
                         if item.source_family != family]
            values = self._growths(remainder)
            result[family] = self._lower(values) if len(values) >= 2 else None
        return result

    def decay(self, observations: Sequence[Observation]
              ) -> Tuple[Optional[float], Optional[float], str]:
        ordered = sorted(observations, key=lambda item: item.timestamp)
        half = len(ordered) // 2
        if half < 2:
            return None, None, "UNKNOWN"
        early = self._lower(self._growths(ordered[:half]))
        late = self._lower(self._growths(ordered[half:]))
        if early is None or late is None:
            return early, late, "UNKNOWN"
        if late <= 0 < early:
            status = "DECAYED"
        elif early > 0 and late < early * 0.5:
            status = "DECAYING"
        elif late > early:
            status = "STRENGTHENING"
        else:
            status = "STABLE"
        return early, late, status

    # -- the verdict -----------------------------------------------------

    def evaluate(self, mechanism: str, observations: Sequence[Observation], *,
                 pbo: Optional[float] = None) -> GauntletResult:
        rows = list(observations)
        result = GauntletResult(mechanism=mechanism, n=len(rows), pbo=pbo)
        if rows and all(item.is_control for item in rows):
            result.verdict = Verdict.CONTROL

        growths = self._growths(rows)
        nets = [item.net_return(self.reference_latency_s) for item in rows]
        nets = [value for value in nets if value is not None]
        if nets:
            result.net_ev = statistics.fmean(nets)
        if growths:
            result.e_log_w = statistics.fmean(growths)
            result.max_drawdown = max_drawdown(
                [item.log_growth(self.reference_latency_s) or 0.0
                 for item in sorted(rows, key=lambda row: row.timestamp)])
        result.e_log_w_lower = self._lower(growths) if len(growths) >= 2 else None

        result.latency_curve, result.latency_survival_s = (
            self.latency_survival(rows))
        result.cost_curve, result.cost_survival_multiple = self.cost_stress(rows)
        result.regimes = self.by_regime(rows)
        regime_values = [value for value in result.regimes.values()
                         if value is not None]
        result.regime_holdout_min = min(regime_values) if regime_values else None
        result.source_families = self.family_leave_one_out(rows)
        family_values = [value for value in result.source_families.values()
                         if value is not None]
        result.family_loo_min = min(family_values) if family_values else None
        result.early_half, result.late_half, result.decay_status = (
            self.decay(rows))

        if result.verdict is Verdict.CONTROL:
            result.reasons.append("declared control; carried as a floor")
            return result

        # -- gates, in the order a reader would want them --
        if len(rows) < self.min_observations:
            result.verdict = Verdict.DATA_BLOCKED
            result.reasons.append(
                f"{len(rows)} observations, {self.min_observations} required; "
                "every test below is theatre at this sample size")
            return result
        if result.e_log_w_lower is None:
            result.verdict = Verdict.DATA_BLOCKED
            result.reasons.append(
                "no measurable log growth at the reference latency; the "
                "entries were never feasible, which is not the same as flat")
            return result

        failures: List[str] = []
        fragilities: List[str] = []

        if result.e_log_w_lower <= 0:
            failures.append(
                f"lower bound on E[log W] is {result.e_log_w_lower:+.4f}; the "
                "point estimate being positive is not a reason to size")
        if result.latency_survival_s is None:
            failures.append(
                "does not survive even the fastest entry on the grid")
        elif result.latency_survival_s <= self.reference_latency_s:
            # Relative to the latency it is SCORED at, not a hardcoded second.
            # An edge whose last positive rung is the reference latency has
            # zero margin: the desk's real latency is a distribution with a
            # tail past its median, and a mechanism with no headroom is one
            # that is negative on half its fills.
            fragilities.append(
                f"survives only to {result.latency_survival_s:g}s, the same "
                "latency it is scored at; no headroom for the tail of the "
                "desk's own latency distribution")
        if result.cost_survival_multiple is None:
            failures.append("does not survive its own measured execution cost")
        elif result.cost_survival_multiple < 1.5:
            fragilities.append(
                "dies at 1.5x execution cost; one bad week of contention")
        if len(result.regimes) < self.required_regimes:
            fragilities.append(
                f"observed in {len(result.regimes)} regime(s), "
                f"{self.required_regimes} required to call it general")
        elif result.regime_holdout_min is not None and (
                result.regime_holdout_min <= 0):
            fragilities.append(
                "negative in at least one regime; this is a regime bet "
                "wearing a mechanism's name")
        if result.family_loo_min is not None and result.family_loo_min <= 0:
            failures.append(
                "removing one source family takes it negative; the edge is "
                "that family, not the mechanism")
        if result.decay_status == "DECAYED":
            failures.append("the later half of the record is not positive")
        elif result.decay_status == "DECAYING":
            fragilities.append("the later half is less than half the earlier")
        if pbo is not None and pbo > self.max_pbo:
            failures.append(
                f"probability of backtest overfitting {pbo:.2f} exceeds "
                f"{self.max_pbo:.2f}; this was selected, not discovered")

        result.reasons = failures + fragilities
        if failures:
            result.verdict = Verdict.KILL
        elif fragilities:
            result.verdict = Verdict.FRAGILE
        else:
            result.verdict = Verdict.SURVIVOR
            result.reasons.append(
                f"lower bound {result.e_log_w_lower:+.4f} through "
                f"{result.latency_survival_s:g}s and "
                f"{result.cost_survival_multiple:g}x cost across "
                f"{len(result.regimes)} regimes")
        return result

    def run(self, observations: Iterable[Observation]
            ) -> Dict[str, GauntletResult]:
        """Every mechanism, with PBO computed across them jointly.

        PBO is a property of the SELECTION, so it is computed once over the
        whole candidate set rather than per mechanism -- asking "was this one
        overfitted" in isolation is the question that cannot be answered.
        """
        grouped: Dict[str, List[Observation]] = {}
        for item in observations:
            grouped.setdefault(item.mechanism, []).append(item)
        pbo = self._joint_pbo(grouped)
        return {name: self.evaluate(name, rows, pbo=pbo)
                for name, rows in sorted(grouped.items())}

    def _joint_pbo(self, grouped: Dict[str, List[Observation]]
                   ) -> Optional[float]:
        """Align every non-control mechanism on a common time grid for CSCV."""
        series: List[List[float]] = []
        for name, rows in sorted(grouped.items()):
            if any(item.is_control for item in rows):
                continue
            ordered = sorted(rows, key=lambda item: item.timestamp)
            values = [item.log_growth(self.reference_latency_s)
                      for item in ordered]
            values = [value for value in values if value is not None]
            if values:
                series.append(values)
        if len(series) < 2:
            return None
        return probability_of_backtest_overfitting(series)


@dataclass
class ScoreboardRow:
    mechanism: str
    n_oos: int
    net_ev: Optional[float]
    e_log_w: Optional[float]
    e_log_w_lower: Optional[float]
    delay_survival: str
    regimes: int
    verdict: str
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MechanismScoreboard:
    """The table that decides whether the desk has an edge.

    Deliberately one table with one row per mechanism and a verdict column,
    because the alternative -- a dashboard of forty metrics -- is how a desk
    talks itself into a position. If no row says SURVIVOR, the desk does not
    have an edge yet, and the scoreboard says so in one word.
    """

    def __init__(self, gauntlet: Optional[Gauntlet] = None):
        self.gauntlet = gauntlet or Gauntlet()

    def build(self, observations: Iterable[Observation]
              ) -> List[ScoreboardRow]:
        results = self.gauntlet.run(observations)
        rows: List[ScoreboardRow] = []
        for name, result in results.items():
            if result.latency_survival_s is None:
                delay = ("n/a" if result.verdict is Verdict.CONTROL
                         else "dies immediately")
            else:
                delay = f"+ through {result.latency_survival_s:g}s"
            rows.append(ScoreboardRow(
                mechanism=name, n_oos=result.n, net_ev=result.net_ev,
                e_log_w=result.e_log_w,
                e_log_w_lower=result.e_log_w_lower,
                delay_survival=delay, regimes=len(result.regimes),
                verdict=result.verdict.value,
                note=result.reasons[0] if result.reasons else ""))
        rows.sort(key=lambda row: (row.verdict != Verdict.SURVIVOR.value,
                                   -(row.e_log_w_lower or -9e9)))
        return rows

    def report(self, observations: Iterable[Observation]) -> Dict[str, Any]:
        rows = self.build(observations)
        survivors = [row for row in rows
                     if row.verdict == Verdict.SURVIVOR.value]
        return {
            "schema": GAUNTLET_SCHEMA_VERSION,
            "rows": [row.to_dict() for row in rows],
            "mechanisms": len(rows),
            "survivors": len(survivors),
            "has_edge": bool(survivors),
            "detail": ("" if survivors else
                       "no mechanism survived the gauntlet; the desk has "
                       "machinery, not an edge"),
        }

    @staticmethod
    def render(rows: Sequence[ScoreboardRow]) -> str:
        """Plain text, because this is read in a terminal over ssh."""
        header = (f"{'mechanism':28s} {'N':>7s} {'net EV':>9s} "
                  f"{'E[logW]':>9s} {'LB':>9s} {'delay':>18s} "
                  f"{'reg':>4s}  verdict")
        lines = [header, "-" * len(header)]
        for row in rows:
            def _fmt(value: Optional[float]) -> str:
                return "     -   " if value is None else f"{value:+9.4f}"
            lines.append(
                f"{row.mechanism[:28]:28s} {row.n_oos:7d} "
                f"{_fmt(row.net_ev)} {_fmt(row.e_log_w)} "
                f"{_fmt(row.e_log_w_lower)} {row.delay_survival:>18s} "
                f"{row.regimes:4d}  {row.verdict}")
        return "\n".join(lines)

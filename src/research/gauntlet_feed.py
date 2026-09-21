"""What actually feeds the gauntlet, so that it stops being a scoreboard of nothing.

`gauntlet.py` decides whether a mechanism has an edge. `forward_evidence.py`
carries the survivor count into the promotion gate, and CANARY requires one
survivor while LIVE requires two. Between those two facts there was nothing:
no caller anywhere built a single `Observation`, so `gauntlet_survivors` was
None for the life of the desk, the gate failed on "not measured", and the two
strictest rungs on the ladder were unreachable by construction.

This is the missing half. It turns stored launch episodes into gauntlet
observations, which means answering three questions the gauntlet cannot answer
for itself:

**Which mechanism was this?** A mechanism is a rule that could have been
evaluated BEFORE entry -- "the deployer has a clean history", "flow in the
first ten seconds was organic". Rules are declared with the snapshot they read
and therefore with the earliest instant they could have fired, and an
observation never claims a latency earlier than the evidence its rule needed.
A rule reading the ten-second snapshot cannot pretend to a 100ms entry.

**What would it have returned at each latency?** Priced by `lifecycle_replay`
against the curve as it actually was: filled at the last mark at or before the
delay, sold only what the curve could absorb. A latency the launch never
priced is ABSENT, not zero -- an entry that was never feasible at 50ms did not
return nothing at 50ms, it did not happen, and the gauntlet's latency gate
reads those two very differently.

**Which regime was it in?** From the launch's own measured market features.
An episode whose market was never measured is dropped rather than labelled
"unknown", because the gauntlet requires three regimes to call a mechanism
general and a bucket named for the absence of measurement would satisfy that
requirement with nothing. Dropping is visible in the report; a fake regime
would not be.

The control arm is not optional. Every run includes `control_all_launches`,
which selects nothing: if it survives alongside a mechanism, the mechanism is
not what produced the return.
"""

from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Optional,
                    Sequence, Tuple)

from src.research.gauntlet import (Gauntlet, MechanismScoreboard,
                                   Observation, Verdict)
from src.research.lifecycle_replay import (
    DEFAULT_DELAYS_S, ExitRule, lifecycle_from_episode, replay_cell,
    trailing_stop)

logger = logging.getLogger(__name__)

GAUNTLET_FEED_SCHEMA_VERSION = "v1"

#: Snapshot keys by the offset they describe. Mirrors `SNAPSHOT_OFFSETS_S` in
#: `dataset_builder`, copied rather than imported so this module does not drag
#: the whole builder (and numpy, and the genealogy) into a replay. Kept honest
#: by `test_snapshot_offsets_match_the_dataset_builder`.
SNAPSHOT_KEY_BY_OFFSET: Dict[float, str] = {
    0.0: "t0", 0.01: "t10ms", 0.025: "t25ms", 0.05: "t50ms", 0.1: "t100ms",
    0.25: "t250ms", 0.5: "t500ms", 1.0: "t1s", 3.0: "t3s", 5.0: "t5s",
    10.0: "t10s", 30.0: "t30s", 60.0: "t1m", 180.0: "t3m", 300.0: "t5m",
    900.0: "t15m", 3600.0: "t1h",
}

#: The size every mechanism is priced at. One size, deliberately: the gauntlet
#: compares mechanisms, and a mechanism that looks better only because it was
#: replayed at a smaller size is a sizing result wearing a mechanism's name.
DEFAULT_SIZE_SOL = 0.25

#: Round trip cost charged in the replay, and therefore the cost fraction the
#: gauntlet re-charges when it doubles costs. Matches `replay_cell`'s own
#: default so the two never disagree about what has already been paid.
DEFAULT_ROUND_TRIP_COST = 0.02

#: SOL 24h move, in per cent, that separates the three trend buckets.
REGIME_TREND_BAND = 2.0

#: Launches per hour separating a quiet market from a busy one. Not a tuned
#: number -- it exists to split the corpus into buckets that are actually
#: different, and it is recorded in the report so a later reader can see what
#: the split was.
REGIME_RATE_BANDS: Tuple[float, float] = (20.0, 100.0)


# --- reading an episode ---------------------------------------------------

class EpisodeView:
    """One stored episode, with feature access that distinguishes missing.

    Every accessor returns None for "this was never measured" and never a
    default. The whole module rests on that distinction: a rule that reads a
    missing feature as False silently moves the launch into the control arm's
    complement and reports a selection that never happened.
    """

    __slots__ = ("raw", "_snapshots")

    def __init__(self, raw: Dict[str, Any]):
        self.raw = raw or {}
        self._snapshots = self.raw.get("snapshots") or {}

    @property
    def token(self) -> str:
        return str(self.raw.get("token", ""))

    @property
    def created_at(self) -> float:
        try:
            return float(self.raw.get("created_at", 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    def snapshot(self, offset_s: float) -> Optional[Dict[str, Any]]:
        key = SNAPSHOT_KEY_BY_OFFSET.get(float(offset_s))
        if key is None:
            return None
        row = self._snapshots.get(key)
        return row if isinstance(row, dict) else None

    def features(self, offset_s: float, group: str) -> Optional[Dict[str, Any]]:
        row = self.snapshot(offset_s)
        if row is None:
            return None
        block = row.get(group)
        return block if isinstance(block, dict) else None

    def feature(self, offset_s: float, group: str, key: str) -> Any:
        block = self.features(offset_s, group)
        return None if block is None else block.get(key)


def regime_of(view: EpisodeView, *, at_offset_s: float = 0.0) -> Optional[str]:
    """A regime label built only from measured market features, or None.

    Two axes, because one is not a regime: a market can be quiet and rising or
    frantic and falling, and a mechanism that works in one of those has not
    been shown to work in the other. Returns None when either axis is
    unmeasured, and the caller drops the observation.
    """
    market = view.features(at_offset_s, "market_features")
    if market is None:
        return None
    change = market.get("sol_change_24h")
    rate = market.get("meme_launch_rate_1h")
    if change is None or rate is None:
        return None
    try:
        change = float(change)
        rate = float(rate)
    except (TypeError, ValueError):
        return None
    trend = ("up" if change > REGIME_TREND_BAND else
             "down" if change < -REGIME_TREND_BAND else "flat")
    low, high = REGIME_RATE_BANDS
    activity = ("quiet" if rate < low else "busy" if rate < high else "frantic")
    return f"{trend}/{activity}"


# --- mechanisms -----------------------------------------------------------

#: True: this launch is one of the mechanism's. False: it is not. None: the
#: feature the rule needs was never measured on this launch, which is neither.
MechanismRule = Callable[[EpisodeView], Optional[bool]]


@dataclass(frozen=True)
class Mechanism:
    """A rule that could have been evaluated before entry, and when.

    `decision_offset_s` is load-bearing rather than descriptive. It is the
    earliest instant the rule's evidence existed, and no observation this
    mechanism produces will carry a latency below it. Without that, a rule
    reading the ten-second flow snapshot would be credited with the return of
    a 50ms entry, which is the single most flattering mistake available here.
    """

    name: str
    rule: MechanismRule
    decision_offset_s: float = 0.0
    source_family: str = ""
    is_control: bool = False


def _clean_deployer(view: EpisodeView) -> Optional[bool]:
    features = view.features(0.0, "deployer_features")
    if features is None:
        return None
    if features.get("has_profile") is not True:
        # A deployer with no profile is not a dirty deployer; it is a
        # deployer we have never seen. That belongs in the first-time arm,
        # not in this one's complement.
        return False
    prior = features.get("prior_launches")
    rug_rate = features.get("rug_rate")
    if prior is None or rug_rate is None:
        return None
    try:
        return int(prior) >= 1 and float(rug_rate) <= 0.2
    except (TypeError, ValueError):
        return None


def _first_time_deployer(view: EpisodeView) -> Optional[bool]:
    features = view.features(0.0, "deployer_features")
    if features is None:
        return None
    if features.get("has_profile") is False:
        return True
    prior = features.get("prior_launches")
    if prior is None:
        return None
    try:
        return int(prior) == 0
    except (TypeError, ValueError):
        return None


def _organic_flow_10s(view: EpisodeView) -> Optional[bool]:
    features = view.features(10.0, "flow_features")
    if features is None or features.get("status") != "OK":
        return None
    organic = features.get("organic_ratio")
    bundled = features.get("bundle_concentration")
    if organic is None or bundled is None:
        return None
    try:
        return float(organic) >= 0.8 and float(bundled) <= 0.2
    except (TypeError, ValueError):
        return None


def _smart_money_10s(view: EpisodeView) -> Optional[bool]:
    features = view.features(10.0, "wallet_features")
    if features is None:
        return None
    count = features.get("smart_buyer_count")
    if count is None:
        return None
    try:
        return int(count) >= 1
    except (TypeError, ValueError):
        return None


def _sybil_discounted_flow_10s(view: EpisodeView) -> Optional[bool]:
    features = view.features(10.0, "entity_graph_features")
    if features is None or features.get("actor_status") != "OK":
        return None
    adjusted = features.get("actor_adjusted_flow")
    if adjusted is None:
        return None
    try:
        return float(adjusted) > 0.0
    except (TypeError, ValueError):
        return None


def _every_launch(view: EpisodeView) -> Optional[bool]:
    return True


#: The control arm. Present in every run, and never removable through
#: configuration: a scoreboard without it cannot tell an edge from a market.
CONTROL = Mechanism(name="control_all_launches", rule=_every_launch,
                    decision_offset_s=0.0, is_control=True)

DEFAULT_MECHANISMS: Tuple[Mechanism, ...] = (
    Mechanism("clean_deployer_history", _clean_deployer, 0.0, "deployer"),
    Mechanism("first_time_deployer", _first_time_deployer, 0.0, "deployer"),
    Mechanism("organic_flow_10s", _organic_flow_10s, 10.0, "flow"),
    Mechanism("smart_money_first_10s", _smart_money_10s, 10.0, "wallet"),
    Mechanism("sybil_discounted_flow_10s", _sybil_discounted_flow_10s, 10.0,
              "actor_graph"),
)


# --- the feed -------------------------------------------------------------

@dataclass
class FeedCounters:
    episodes: int = 0
    no_lifecycle: int = 0
    no_regime: int = 0
    observations: int = 0
    by_mechanism: Dict[str, Dict[str, int]] = field(default_factory=dict)
    regimes: Dict[str, int] = field(default_factory=dict)

    def bucket(self, name: str) -> Dict[str, int]:
        return self.by_mechanism.setdefault(
            name, {"matched": 0, "unmatched": 0, "unmeasured": 0,
                   "priced_latencies": 0, "unpriced_latencies": 0})


class GauntletFeed:
    """Stored episodes in, gauntlet observations out."""

    def __init__(self, *,
                 mechanisms: Sequence[Mechanism] = DEFAULT_MECHANISMS,
                 delays: Sequence[float] = DEFAULT_DELAYS_S,
                 size_sol: float = DEFAULT_SIZE_SOL,
                 exit_rule: Optional[ExitRule] = None,
                 exit_rule_name: str = "trailing_0.7",
                 round_trip_cost: float = DEFAULT_ROUND_TRIP_COST,
                 require_measured_regime: bool = True):
        # The control is prepended rather than appended so it is the first row
        # a reader's eye lands on, and deduplicated so a caller who passes it
        # explicitly does not get two of them.
        named = {mechanism.name: mechanism for mechanism in mechanisms}
        named.pop(CONTROL.name, None)
        self.mechanisms: Tuple[Mechanism, ...] = (CONTROL,) + tuple(named.values())
        self.delays = tuple(sorted(float(delay) for delay in delays))
        self.size_sol = float(size_sol)
        self.exit_rule = exit_rule or trailing_stop(0.7)
        self.exit_rule_name = exit_rule_name
        self.round_trip_cost = float(round_trip_cost)
        self.require_measured_regime = bool(require_measured_regime)
        self.counters = FeedCounters()

    # -- pricing -----------------------------------------------------------

    def _returns_by_latency(self, lifecycle: Any, decision_offset_s: float
                            ) -> Dict[float, Optional[float]]:
        """Net return per SOL committed, keyed by HOW LATE the entry was.

        Latency here is lateness relative to the mechanism's own decision
        instant, not clock time since launch. The gauntlet's question is
        "does this edge survive being a second slow", and for a rule that
        reads the ten-second flow snapshot the honest version of that question
        is 10s versus 11s -- not 10s versus 1s, which no implementation of
        that rule could ever answer yes to. Keying by lateness also makes the
        columns comparable across mechanisms that fire at different instants,
        which is the entire point of putting them in one table.

        Absent rather than zero where the launch could not be priced. A
        mechanism whose 100ms column is full of zeros looks like one that
        survives being late; a mechanism whose 100ms column is absent looks
        like what it is.
        """
        priced: Dict[float, Optional[float]] = {}
        for lateness in self.delays:
            cell = replay_cell(lifecycle, decision_offset_s + lateness,
                               self.size_sol, self.exit_rule_name,
                               self.exit_rule, self.round_trip_cost)
            if not cell.ok or cell.filled_sol <= 0:
                priced[lateness] = None
                continue
            priced[lateness] = float(cell.net_sol / cell.filled_sol)
        return priced

    # -- building ----------------------------------------------------------

    def observations(self, episodes: Iterable[Dict[str, Any]]
                     ) -> List[Observation]:
        built: List[Observation] = []
        for raw in episodes:
            built.extend(self._for_episode(EpisodeView(raw)))
        return built

    def _for_episode(self, view: EpisodeView) -> List[Observation]:
        self.counters.episodes += 1
        lifecycle = lifecycle_from_episode(view.raw)
        if lifecycle is None:
            self.counters.no_lifecycle += 1
            return []
        regime = regime_of(view)
        if regime is None:
            self.counters.no_regime += 1
            if self.require_measured_regime:
                return []
            regime = "unknown"
        self.counters.regimes[regime] = self.counters.regimes.get(regime, 0) + 1

        # Priced once per distinct decision offset and shared, because the
        # replay is the expensive half and two mechanisms deciding at the same
        # instant see exactly the same prices.
        priced_by_offset: Dict[float, Dict[float, Optional[float]]] = {}
        built: List[Observation] = []
        for mechanism in self.mechanisms:
            bucket = self.counters.bucket(mechanism.name)
            try:
                selected = mechanism.rule(view)
            except Exception as exc:  # a bad rule must not take the run down
                logger.debug("mechanism %s raised on %s: %s",
                             mechanism.name, view.token, exc)
                selected = None
            if selected is None:
                bucket["unmeasured"] += 1
                continue
            if not selected:
                bucket["unmatched"] += 1
                continue
            bucket["matched"] += 1
            offset = float(mechanism.decision_offset_s)
            if offset not in priced_by_offset:
                priced_by_offset[offset] = self._returns_by_latency(
                    lifecycle, offset)
            returns = priced_by_offset[offset]
            bucket["priced_latencies"] += sum(
                1 for value in returns.values() if value is not None)
            bucket["unpriced_latencies"] += sum(
                1 for value in returns.values() if value is None)
            if not any(value is not None for value in returns.values()):
                # Selected, but nothing about it could be priced. Emitting a
                # row of all-None would inflate N without adding evidence.
                continue
            built.append(Observation(
                mechanism=mechanism.name,
                timestamp=view.created_at,
                regime=regime,
                source_family=mechanism.source_family or mechanism.name,
                cohort=view.token,
                net_return_by_latency=dict(returns),
                cost_fraction=self.round_trip_cost,
                is_control=mechanism.is_control,
            ))
        self.counters.observations += len(built)
        return built

    # -- reporting ---------------------------------------------------------

    def coverage(self) -> Dict[str, Any]:
        """What the corpus could and could not say, before any verdict."""
        return {
            "schema": GAUNTLET_FEED_SCHEMA_VERSION,
            "episodes": self.counters.episodes,
            "dropped_no_lifecycle": self.counters.no_lifecycle,
            "dropped_no_measured_regime": (
                self.counters.no_regime if self.require_measured_regime else 0),
            "unmeasured_regime": self.counters.no_regime,
            "observations": self.counters.observations,
            "regimes": dict(sorted(self.counters.regimes.items())),
            "mechanisms": {name: dict(bucket) for name, bucket
                           in sorted(self.counters.by_mechanism.items())},
            "delays_s": list(self.delays),
            "size_sol": self.size_sol,
            "exit_rule": self.exit_rule_name,
            "round_trip_cost": self.round_trip_cost,
            "regime_bands": {"sol_change_pct": REGIME_TREND_BAND,
                             "launch_rate_1h": list(REGIME_RATE_BANDS)},
        }

    def run(self, episodes: Iterable[Dict[str, Any]], *,
            gauntlet: Optional[Gauntlet] = None) -> Dict[str, Any]:
        """Build, score, and return a report `record_gauntlet` accepts.

        A run that produced no observations reports zero mechanisms, which
        `ForwardEvidence.record_gauntlet` reads as no measurement rather than
        as zero survivors -- so an empty corpus blocks promotion instead of
        appearing to have tested something and found nothing.
        """
        observations = self.observations(episodes)
        scoreboard = MechanismScoreboard(gauntlet)
        # Built once. `report()` would run the whole gauntlet a second time
        # over the same observations to produce the same rows, and on a
        # thirty-thousand launch corpus that is a doubled bootstrap for
        # nothing.
        rows = scoreboard.build(observations)
        survivors = [row for row in rows if row.verdict == Verdict.SURVIVOR.value]
        return {
            "schema": GAUNTLET_FEED_SCHEMA_VERSION,
            "rows": [row.to_dict() for row in rows],
            "mechanisms": len(rows),
            "survivors": len(survivors),
            "has_edge": bool(survivors),
            "detail": ("" if survivors else
                       "no mechanism survived the gauntlet; the desk has "
                       "machinery, not an edge"),
            "coverage": self.coverage(),
            "table": MechanismScoreboard.render(rows),
        }


# --- loading --------------------------------------------------------------

def iter_episodes(storage: Path, *, limit: Optional[int] = None
                  ) -> Iterator[Dict[str, Any]]:
    """Stored episodes, oldest first, skipping the unreadable.

    Chronological because the gauntlet's CSCV split is over blocks of time and
    a shuffled corpus turns that into an in-sample split wearing an
    out-of-sample name.
    """
    paths = sorted(Path(storage).glob("*/*.json.gz"))
    emitted = 0
    for path in paths:
        if limit is not None and emitted >= limit:
            return
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                episode = json.load(handle)
        except (OSError, json.JSONDecodeError, EOFError) as exc:
            logger.debug("unreadable episode %s: %s", path, exc)
            continue
        if not isinstance(episode, dict):
            continue
        outcome = episode.get("final_outcome") or {}
        if outcome.get("status") != "OK":
            # An unresolved launch has no forward return to test a mechanism
            # against. Excluded, and counted by the caller, not scored as a
            # loss.
            continue
        emitted += 1
        yield episode

"""Where a narrative is in its life, and whether it is about to catch.

Ranking sources answers "is this source worth following". It does not answer
the question the position actually turns on, which is about the CROWD rather
than about the source:

    P(mass independent buyer acceleration within T | propagation state)

Those are different. A source can be excellent -- early, accurate, profitable
to follow -- and have posted into a narrative that will never spread, and a
mediocre source can post the thing that ignites. What decides whether a
position compounds is not the quality of who spoke; it is whether the next
wave of independent buyers is coming.

So this models the narrative's own lifecycle:

    CHAIN_ONLY                nobody outside the chain has said anything
    EARLY_SOURCE              one obscure source, no propagation yet
    KOL_IGNITION              a source with real reach has picked it up
    MULTI_SOURCE_ACCELERATION independent sources are compounding on it
    MASS_FOMO                 independent buyer arrival is accelerating
    SATURATION                everyone who was going to hear has heard

The states are ordered but not a ratchet: a narrative that reaches
KOL_IGNITION and does not spread falls back, and pretending otherwise would
hold a position on a fire that went out.

The distinction that matters most is between MULTI_SOURCE_ACCELERATION and
SATURATION, because they look identical in raw volume and imply opposite
actions. Acceleration is more independent sources reaching more new buyers;
saturation is the same sources reaching the same audience, with buyer arrival
flattening while posting continues. Separating them is what the second
derivative of independent buyer arrival is for.

And it takes SOURCES as independent actors, not as posts: eight accounts run
by one operator posting eight times is one source shouting, and treating it
as eight is exactly the manufactured ignition this is supposed to detect.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

IGNITION_SCHEMA_VERSION = "v1"

# Reach above which a source counts as a KOL rather than as an early voice.
# Deliberately about audience rather than about accuracy: ignition is a
# question of how many people heard, and a small accurate source and a large
# inaccurate one play opposite roles in it.
DEFAULT_KOL_REACH = 10_000

# Independent sources that must be on a narrative before it is "multi-source".
# Two is a coincidence.
DEFAULT_MULTI_SOURCE = 3

# Window over which buyer arrival is differenced. Short enough that ignition
# is caught while it is happening, long enough that one busy slot is not an
# acceleration.
DEFAULT_WINDOW_S = 60.0


class NarrativeState(Enum):
    CHAIN_ONLY = "chain_only"
    EARLY_SOURCE = "early_source"
    KOL_IGNITION = "kol_ignition"
    MULTI_SOURCE_ACCELERATION = "multi_source_acceleration"
    MASS_FOMO = "mass_fomo"
    SATURATION = "saturation"

    @property
    def rank(self) -> int:
        return _STATE_RANK[self]


_STATE_RANK = {
    NarrativeState.CHAIN_ONLY: 0,
    NarrativeState.EARLY_SOURCE: 1,
    NarrativeState.KOL_IGNITION: 2,
    NarrativeState.MULTI_SOURCE_ACCELERATION: 3,
    NarrativeState.MASS_FOMO: 4,
    NarrativeState.SATURATION: 5,
}


class KolRole(Enum):
    """What a source DOES to a narrative, which decides the right action.

    A DISTRIBUTOR is not a source to ignore -- it is a source whose posts are
    excellent at predicting flow and terrible to hold alongside. The optimal
    action is to be in before its followers arrive and out as their flow
    peaks, which is the opposite of what "he posted, so hold" produces.
    """

    UNKNOWN = "unknown"
    ORIGINATOR = "originator"
    EARLY_REPEATER = "early_repeater"
    FOMO_TRIGGER = "fomo_trigger"
    DISTRIBUTOR = "distributor"
    LAGGING_REPEATER = "lagging_repeater"
    ANTI_SIGNAL = "anti_signal"


@dataclass
class SourceTouch:
    """One source arriving on one narrative."""

    source_id: str
    timestamp: float
    reach: Optional[int] = None
    # Independence of this source from the ones already on the narrative.
    # Eight accounts run by one operator are one source shouting.
    independence: float = 1.0
    role: KolRole = KolRole.UNKNOWN

    @property
    def is_kol(self) -> bool:
        return bool(self.reach is not None and self.reach >= DEFAULT_KOL_REACH)


@dataclass
class IgnitionReading:
    """Where the narrative is, and whether the next wave is coming."""

    status: str
    state: NarrativeState = NarrativeState.CHAIN_ONLY
    # P(mass independent buyer acceleration within the horizon). None when
    # unmeasured -- never zero, which would read as "we checked and it will
    # not spread".
    probability: Optional[float] = None
    horizon_seconds: float = 0.0
    independent_sources: float = 0.0
    kol_sources: int = 0
    buyer_rate: Optional[float] = None
    buyer_acceleration: Optional[float] = None
    roles: Dict[str, str] = field(default_factory=dict)
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "OK"

    @property
    def igniting(self) -> bool:
        """Before the crowd, with the crowd still coming."""
        return self.ok and self.state in (NarrativeState.KOL_IGNITION,
                                          NarrativeState.MULTI_SOURCE_ACCELERATION)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": IGNITION_SCHEMA_VERSION, "status": self.status,
            "state": self.state.value, "probability": self.probability,
            "horizon_seconds": self.horizon_seconds,
            "independent_sources": round(self.independent_sources, 3),
            "kol_sources": self.kol_sources,
            "buyer_rate": self.buyer_rate,
            "buyer_acceleration": self.buyer_acceleration,
            "igniting": self.igniting, "roles": dict(self.roles),
            "detail": self.detail,
        }


def classify_role(dna: Any, lead_rate: Optional[float] = None) -> KolRole:
    """What this source does to narratives, from its measured DNA.

    Checked in order of how strongly each shape overrides the others. A
    distributor that is also early is still a distributor: being early makes
    it better at predicting flow and no safer to hold alongside.
    """
    if dna is None or getattr(dna, "status", "") != "MEASURED":
        return KolRole.UNKNOWN
    if getattr(dna, "is_distributor", False):
        return KolRole.DISTRIBUTOR
    horizon_return = getattr(dna, "best_horizon_return", None)
    if horizon_return is not None and horizon_return < 0:
        # Following it loses money on its own posts. That is information --
        # about the other side of the trade.
        return KolRole.ANTI_SIGNAL
    reach = getattr(dna, "reach", None)
    flow = getattr(dna, "flow_prediction", None)
    if lead_rate is not None and lead_rate >= 0.6:
        return KolRole.ORIGINATOR
    if flow is not None and flow > 1.5 and reach is not None and reach >= DEFAULT_KOL_REACH:
        # Large audience whose posts are followed by a wave. That is the shape
        # that ignites, whether or not the source is early.
        return KolRole.FOMO_TRIGGER
    lag = getattr(dna, "median_observation_lag", None)
    if lag is not None and lag > 300:
        return KolRole.LAGGING_REPEATER
    return KolRole.EARLY_REPEATER


class IgnitionModel:
    """The narrative lifecycle, from source touches and buyer arrival."""

    def __init__(self, *, kol_reach: int = DEFAULT_KOL_REACH,
                 multi_source: int = DEFAULT_MULTI_SOURCE,
                 window_seconds: float = DEFAULT_WINDOW_S,
                 horizon_seconds: float = 300.0):
        self.kol_reach = int(kol_reach)
        self.multi_source = int(multi_source)
        self.window_seconds = float(window_seconds)
        self.horizon_seconds = float(horizon_seconds)

    def read(self, touches: Sequence[SourceTouch],
             buyer_times: Sequence[float],
             now: Optional[float] = None) -> IgnitionReading:
        """Where this narrative is, and whether the crowd is still coming.

        ``buyer_times`` are the arrival times of INDEPENDENT buyers -- the
        actor graph's compression has already been applied. Passing raw wallet
        counts here would let a Sybil manufacture the acceleration this is
        meant to detect.
        """
        now = time.time() if now is None else now
        roles = {touch.source_id: touch.role.value for touch in touches}
        independent = sum(max(0.0, min(1.0, touch.independence)) for touch in touches)
        kols = sum(1 for touch in touches
                   if touch.reach is not None and touch.reach >= self.kol_reach)

        rate, acceleration = self._arrival(buyer_times, now)

        if not touches:
            state = NarrativeState.CHAIN_ONLY
        elif independent < 2:
            state = (NarrativeState.KOL_IGNITION if kols
                     else NarrativeState.EARLY_SOURCE)
        elif independent < self.multi_source:
            state = NarrativeState.KOL_IGNITION
        else:
            # More sources and more buyers is acceleration; more sources and
            # FLAT buyers is saturation. In raw volume they are identical and
            # they imply opposite actions, which is the whole reason the
            # second derivative is here.
            if acceleration is None:
                state = NarrativeState.MULTI_SOURCE_ACCELERATION
            elif acceleration > 0:
                state = (NarrativeState.MASS_FOMO if rate and rate > 0
                         and independent >= self.multi_source * 2
                         else NarrativeState.MULTI_SOURCE_ACCELERATION)
            else:
                state = NarrativeState.SATURATION

        if not buyer_times and not touches:
            return IgnitionReading(
                status="DATA_BLOCKED", state=state, roles=roles,
                horizon_seconds=self.horizon_seconds,
                detail="no source touches and no buyer arrivals observed")

        return IgnitionReading(
            status="OK", state=state,
            probability=self._probability(state, independent, kols, acceleration),
            horizon_seconds=self.horizon_seconds,
            independent_sources=independent, kol_sources=kols,
            buyer_rate=rate, buyer_acceleration=acceleration, roles=roles,
            detail=(f"{len(touches)} touches ({independent:.1f} independent, "
                    f"{kols} with reach), buyer rate {rate if rate is not None else 'n/a'}"))

    def _arrival(self, buyer_times: Sequence[float],
                 now: float) -> Tuple[Optional[float], Optional[float]]:
        """Buyers per second in the last window, and how that changed.

        Two equal windows rather than a fitted curve: the question is whether
        arrival is rising, and a difference of two counts answers it without
        pretending to a precision the sample does not support.
        """
        if len(buyer_times) < 2:
            return None, None
        recent = sum(1 for stamp in buyer_times if now - stamp <= self.window_seconds)
        previous = sum(1 for stamp in buyer_times
                       if self.window_seconds < now - stamp <= 2 * self.window_seconds)
        rate = recent / self.window_seconds
        if recent + previous < 2:
            return rate, None
        return rate, float(recent - previous) / self.window_seconds

    def _probability(self, state: NarrativeState, independent: float, kols: int,
                     acceleration: Optional[float]) -> Optional[float]:
        """P(mass independent buyer acceleration within the horizon).

        A structural estimate, not a trained one, and it says so: it is a
        prior over the lifecycle that a forward-labelled model should replace.
        Returned as None where the state carries no information about what
        comes next, because a zero here would read as "we checked and it will
        not spread".
        """
        if state is NarrativeState.CHAIN_ONLY:
            return None
        if state is NarrativeState.SATURATION:
            # Everyone who was going to hear has heard. Not "it will fall",
            # which is a different model's question -- just that the wave this
            # measures is not coming.
            return 0.0
        base = {
            NarrativeState.EARLY_SOURCE: 0.10,
            NarrativeState.KOL_IGNITION: 0.30,
            NarrativeState.MULTI_SOURCE_ACCELERATION: 0.50,
            NarrativeState.MASS_FOMO: 0.65,
        }[state]
        # More independent voices and more reach raise it; the cap keeps a
        # structural prior from ever reading as a confident forecast.
        lift = min(0.25, 0.03 * max(0.0, independent - 1) + 0.05 * kols)
        if acceleration is not None and acceleration > 0:
            lift += 0.05
        return float(min(0.9, base + lift))


def touches_from_events(events: Sequence[Any], dnas: Optional[Dict[str, Any]] = None,
                        independence: Optional[Dict[str, float]] = None,
                        lead_rates: Optional[Dict[str, float]] = None,
                        ) -> List[SourceTouch]:
    """Source touches from the desk's own indexed events.

    One touch per SOURCE, at its first post: a source posting six times is one
    source that has arrived, and counting each post would make a single loud
    account look like a crowd.
    """
    known = dnas or {}
    weights = independence or {}
    leads = lead_rates or {}
    first: Dict[str, Any] = {}
    for event in events:
        source_id = str(getattr(event, "source_id", "") or "")
        if not source_id:
            continue
        stamp = float(getattr(event, "source_at", 0.0)
                      or getattr(event, "observed_at", 0.0) or 0.0)
        if source_id not in first or stamp < first[source_id][0]:
            first[source_id] = (stamp, event)
    touches: List[SourceTouch] = []
    for source_id, (stamp, event) in sorted(first.items(), key=lambda item: item[1][0]):
        dna = known.get(source_id)
        touches.append(SourceTouch(
            source_id=source_id, timestamp=stamp,
            reach=(int(getattr(dna, "reach", 0)) if getattr(dna, "reach", None)
                   else (int(getattr(event, "reach", 0)) or None)),
            independence=float(weights.get(source_id, 1.0)),
            role=classify_role(dna, leads.get(source_id))))
    return touches

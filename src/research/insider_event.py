"""One record for every claim, and one score per question it could answer.

The desk already holds almost all of this information. It holds it in eight
shapes: `SourcePost`, `PostOutcome`, Telegram observations, public-coordination
rows, `LeadEvent`, identity claims, funder observations, launch episodes. Each
was built for the module that needed it, each is correct, and no two of them
can be joined without a bespoke adapter. Cross-source hypothesis mining -- "do
Korean Telegram channels lead English callers on migration claims during hot
regimes?" -- is a query nobody can write, because there is no table.

`InsiderEvent` is that table. Every observation, from any source, becomes one
row with the same columns: who said it, when they said it, when WE saw it,
what they claimed, what it was linked to, what the market looked like at that
instant, what an entry would actually have got at each realistic latency, and
what happened afterwards.

**Provenance is a field, not a convention.** `lawful_access` is required and
the ledger fails closed on it: PUBLIC, AUTHORIZED and VOLUNTEERED are
admissible; PROHIBITED is refused with the reason recorded; UNKNOWN is refused
too, because "we did not write down where this came from" is not a licence.
Refusals are counted, so a source feeding inadmissible material is visible
rather than silently absent.

**One reputation number is the mistake this module exists to correct.** The
same Telegram caller can be excellent at predicting thirty seconds of retail
flow, terrible to hold for ten minutes, sharp on Pump.fun, useless after
migration, profitable during mania and negative in a quiet tape. Collapsing
that into "score: 0.72" destroys precisely the structure that makes it
tradeable. So edges are scored per cell:

    source x mechanism x claim_type x regime x horizon

and a cell with too few samples returns DATA_BLOCKED rather than a number
computed from four observations.

**Nothing here is an entry signal.** A `SourceEdge` is evidence about a
source, not authority over capital; the promotion ladder remains the only
thing that can spend money. This module's job is to make the evidence
computable and honest about its own sample size.
"""

from __future__ import annotations

import logging
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

INSIDER_EVENT_SCHEMA_VERSION = "v1"

#: Latencies an entry could realistically be attempted at, in seconds. These
#: are the columns of the decay curve; a claim that is only profitable at 0ms
#: is not a claim this desk can trade.
ACHIEVABLE_LATENCIES_S: Tuple[float, ...] = (0.1, 0.5, 1.0, 3.0, 10.0, 30.0)

#: Forward horizons scored per event.
HORIZONS_S: Tuple[float, ...] = (1.0, 5.0, 30.0, 60.0, 300.0, 1800.0, 3600.0)

#: Below this many events, a cell reports DATA_BLOCKED instead of a rate. Twelve
#: is not enough to trade on; it is enough that a Wilson interval is worth
#: printing and wide enough to be obviously uninformative.
DEFAULT_MIN_CELL_EVENTS = 12

#: Log-return floor. A total loss is -inf in log space, which would make one
#: rug swallow every other observation in the mean.
LOG_RETURN_FLOOR = -4.0


class LawfulAccess(Enum):
    """How the material was obtained. Required, and fails closed."""

    #: Posted publicly, or readable on-chain.
    PUBLIC = "PUBLIC"
    #: A community the desk is a member of, read within its terms.
    AUTHORIZED = "AUTHORIZED"
    #: Handed to the desk by someone who chose to.
    VOLUNTEERED = "VOLUNTEERED"
    #: Stolen credentials, intercepted private messages, bypassed access
    #: control, purchased confidential material. Never admissible.
    PROHIBITED = "PROHIBITED"
    #: Nobody recorded it. Treated as inadmissible: an unlabelled provenance
    #: is the shape a prohibited one takes when it is not written down.
    UNKNOWN = "UNKNOWN"


ADMISSIBLE_ACCESS = frozenset(
    {LawfulAccess.PUBLIC, LawfulAccess.AUTHORIZED, LawfulAccess.VOLUNTEERED})


class ClaimType(Enum):
    LISTING = "listing"
    LAUNCH = "launch"
    PRESALE = "presale"
    ENDORSEMENT = "endorsement"
    PARTNERSHIP = "partnership"
    INSIDER_BUY = "insider_buy"
    CALL = "call"
    RUG_WARNING = "rug_warning"
    MIGRATION = "migration"
    UNSPECIFIED = "unspecified"


class Mechanism(Enum):
    """How the edge is supposed to work, which decides how to score it.

    A source can be valuable for opposite reasons, and the two must not be
    averaged. FLOW_PREDICTION says the crowd arrives after the post -- worth
    front-running, worthless to hold. INFORMATION_LEAD says the claim was true
    before it was public. DISTRIBUTION says linked wallets are selling into
    the audience, which is a reason to short the follow-through rather than
    join it.
    """

    INFORMATION_LEAD = "information_lead"
    FLOW_PREDICTION = "flow_prediction"
    WALLET_COPY = "wallet_copy"
    NARRATIVE_IGNITION = "narrative_ignition"
    DISTRIBUTION = "distribution"
    UNCLASSIFIED = "unclassified"


@dataclass
class AchievableEntry:
    """What an entry attempted `latency_s` after observation would have got.

    `price_multiple` is relative to the price at observation, so 1.0 means the
    latency cost nothing. `executable_sol` is how much size the book would
    actually have absorbed there -- the field that turns a paper edge into a
    capacity number.
    """

    latency_s: float
    price_multiple: Optional[float] = None
    executable_sol: Optional[float] = None
    feasible: bool = False

    @property
    def slippage_bps(self) -> Optional[float]:
        if self.price_multiple is None:
            return None
        return (self.price_multiple - 1.0) * 10_000.0


@dataclass
class InsiderEvent:
    """One claim, from one source, with everything needed to score it.

    Every outcome field is Optional and defaults to None. Unmeasured is not
    zero: an event whose forward return was never observed must not be counted
    as an event that returned nothing, because that is the difference between
    a source with no record and a source with a bad one.
    """

    event_id: str
    source_id: str
    #: When the source published it.
    source_at: float
    #: When the desk observed it. Every executable claim is measured from here.
    observed_at: float
    lawful_access: LawfulAccess = LawfulAccess.UNKNOWN
    provenance: str = ""
    source_type: str = ""
    source_url: str = ""

    token: str = ""
    mint: str = ""
    claim: str = ""
    claim_type: ClaimType = ClaimType.UNSPECIFIED
    mechanism: Mechanism = Mechanism.UNCLASSIFIED
    language: str = ""
    regime: str = "unknown"

    linked_wallets: List[str] = field(default_factory=list)
    linked_people: List[str] = field(default_factory=list)
    linked_entities: List[str] = field(default_factory=list)
    linked_domains: List[str] = field(default_factory=list)
    linked_sources: List[str] = field(default_factory=list)

    content_hash: str = ""
    raw_evidence_ref: str = ""
    confidence_at_time: Optional[float] = None

    price_at_observation: Optional[float] = None
    liquidity_at_observation: Optional[float] = None
    market_cap_at_observation: Optional[float] = None
    launch_age_at_observation: Optional[float] = None

    achievable: Dict[float, AchievableEntry] = field(default_factory=dict)
    returns: Dict[float, Optional[float]] = field(default_factory=dict)

    mfe: Optional[float] = None
    mae: Optional[float] = None
    max_feasible_multiple: Optional[float] = None
    rugged: Optional[bool] = None
    claim_correct: Optional[bool] = None
    flow_prediction_correct: Optional[bool] = None

    #: Cost of the round trip as a fraction of notional -- fees, tip, slippage.
    cost_fraction: Optional[float] = None
    #: How many other events named this mint inside the crowding window.
    concurrent_mentions: int = 0
    #: Linked wallets bought before the post and sold after it.
    pre_post_accumulation_usd: Optional[float] = None
    post_sell_usd: Optional[float] = None

    ingested_at: float = field(default_factory=time.time)

    # -- derived ---------------------------------------------------------

    @property
    def observation_lag(self) -> float:
        """Publication to observation. The part local speed cannot recover."""
        return max(0.0, self.observed_at - self.source_at)

    @property
    def admissible(self) -> bool:
        return self.lawful_access in ADMISSIBLE_ACCESS

    def entry_at(self, latency_s: float) -> Optional[AchievableEntry]:
        return self.achievable.get(latency_s)

    def executable(self, latency_s: float) -> bool:
        entry = self.achievable.get(latency_s)
        return bool(entry and entry.feasible and entry.executable_sol)

    def net_return(self, horizon_s: float, latency_s: float
                   ) -> Optional[float]:
        """Forward return an entry at `latency_s` would have realised.

        Three things have to be known: the gross move to the horizon, what the
        latency cost to get in, and what the round trip cost. Missing any of
        them returns None -- assuming a zero cost is how a backtest becomes a
        story.
        """
        gross = self.returns.get(horizon_s)
        entry = self.achievable.get(latency_s)
        if gross is None or entry is None or not entry.feasible:
            return None
        if entry.price_multiple is None or entry.price_multiple <= 0:
            return None
        # Entering `price_multiple` worse scales the whole trajectory.
        realised = (1.0 + gross) / entry.price_multiple - 1.0
        if self.cost_fraction is None:
            return None
        return realised - float(self.cost_fraction)

    def log_growth(self, horizon_s: float, latency_s: float
                   ) -> Optional[float]:
        net = self.net_return(horizon_s, latency_s)
        if net is None:
            return None
        return max(LOG_RETURN_FLOOR, math.log(max(1e-9, 1.0 + net)))

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["lawful_access"] = self.lawful_access.value
        data["claim_type"] = self.claim_type.value
        data["mechanism"] = self.mechanism.value
        data["achievable"] = {
            str(key): asdict(value) for key, value in self.achievable.items()}
        data["returns"] = {str(key): value
                           for key, value in self.returns.items()}
        data["schema"] = INSIDER_EVENT_SCHEMA_VERSION
        return data


@dataclass(frozen=True)
class EdgeKey:
    """The cell an edge is measured in. Not a source. A source and a question."""

    source_id: str
    mechanism: Mechanism
    claim_type: ClaimType
    regime: str
    horizon_s: float

    def to_dict(self) -> Dict[str, Any]:
        return {"source_id": self.source_id,
                "mechanism": self.mechanism.value,
                "claim_type": self.claim_type.value,
                "regime": self.regime, "horizon_s": self.horizon_s}


def wilson_interval(successes: int, trials: int, z: float = 1.96
                    ) -> Tuple[float, float]:
    """A proportion's interval that stays sane at the edges.

    The textbook normal interval on 0/9 gives [0, 0], which reads as "this
    source never rugs" from nine observations. Wilson gives [0, 0.30], which
    is the truth.
    """
    if trials <= 0:
        return (0.0, 1.0)
    phat = successes / trials
    denom = 1.0 + z * z / trials
    centre = (phat + z * z / (2 * trials)) / denom
    spread = (z / denom) * math.sqrt(
        phat * (1 - phat) / trials + z * z / (4 * trials * trials))
    return (max(0.0, centre - spread), min(1.0, centre + spread))


@dataclass
class SourceEdge:
    """Everything the desk can say about one source answering one question."""

    key: EdgeKey
    n: int = 0
    status: str = "DATA_BLOCKED"
    detail: str = ""

    first_seen_lag_p05: Optional[float] = None
    first_seen_lag_p50: Optional[float] = None
    first_seen_lag_p95: Optional[float] = None

    ev_gross: Optional[float] = None
    ev_net: Optional[float] = None
    e_log_w: Optional[float] = None
    e_log_w_ci: Optional[Tuple[float, float]] = None

    hit_rate: Optional[float] = None
    p_2x: Optional[float] = None
    p_5x: Optional[float] = None
    p_10x: Optional[float] = None
    p_rug: Optional[float] = None
    p_rug_ci: Optional[Tuple[float, float]] = None

    mfe_median: Optional[float] = None
    mae_median: Optional[float] = None
    max_drawdown: Optional[float] = None

    latency_decay: Dict[float, Optional[float]] = field(default_factory=dict)
    capacity_sol: Optional[float] = None
    copyability: Optional[float] = None
    crowding_decay: Optional[float] = None
    distribution_probability: Optional[float] = None
    manipulation_probability: Optional[float] = None
    p_claim_correct: Optional[float] = None
    p_flow_arrives: Optional[float] = None
    p_executable: Optional[float] = None

    decay_status: str = "UNKNOWN"
    last_validated: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["key"] = self.key.to_dict()
        data["latency_decay"] = {str(k): v
                                 for k, v in self.latency_decay.items()}
        return data


def _median(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [float(value) for value in values if value is not None]
    return statistics.median(present) if present else None


def _rate(values: Sequence[Optional[bool]]) -> Tuple[Optional[float], int, int]:
    present = [bool(value) for value in values if value is not None]
    if not present:
        return None, 0, 0
    hits = sum(1 for value in present if value)
    return hits / len(present), hits, len(present)


class SourceEdgeLedger:
    """Ingests events, refuses inadmissible ones, and scores cells.

    Holds no authority: the output is evidence. The forward ledger and the
    promotion gate remain the only things that decide what gets capital.
    """

    def __init__(self, *, min_cell_events: int = DEFAULT_MIN_CELL_EVENTS,
                 reference_latency_s: float = 1.0,
                 crowding_threshold: int = 3):
        self.min_cell_events = int(min_cell_events)
        self.reference_latency_s = float(reference_latency_s)
        self.crowding_threshold = int(crowding_threshold)
        self.events: List[InsiderEvent] = []
        self.refusals: List[Dict[str, str]] = []
        self._refusal_counts: Dict[str, int] = {}

    # -- ingest ----------------------------------------------------------

    def add(self, event: InsiderEvent) -> Tuple[bool, str]:
        """Accept one event, or refuse it with a reason on the record."""
        if not event.admissible:
            reason = (f"inadmissible provenance {event.lawful_access.value}: "
                      "only PUBLIC, AUTHORIZED and VOLUNTEERED material is "
                      "scoreable")
            self.refusals.append({"event_id": event.event_id,
                                  "source_id": event.source_id,
                                  "reason": reason})
            self._refusal_counts[event.source_id] = (
                self._refusal_counts.get(event.source_id, 0) + 1)
            return False, reason
        if event.observed_at < event.source_at:
            # Observing before publication is a clock problem, and it inflates
            # every lead measurement this ledger exists to compute.
            reason = ("observed_at precedes source_at; a negative observation "
                      "lag would manufacture a lead that did not happen")
            self.refusals.append({"event_id": event.event_id,
                                  "source_id": event.source_id,
                                  "reason": reason})
            return False, reason
        self.events.append(event)
        return True, ""

    def extend(self, events: Iterable[InsiderEvent]) -> Dict[str, int]:
        accepted = refused = 0
        for event in events:
            ok, _ = self.add(event)
            accepted += int(ok)
            refused += int(not ok)
        return {"accepted": accepted, "refused": refused}

    def refusals_by_source(self) -> Dict[str, int]:
        return dict(self._refusal_counts)

    # -- cells -----------------------------------------------------------

    def cells(self) -> Dict[EdgeKey, List[InsiderEvent]]:
        grouped: Dict[EdgeKey, List[InsiderEvent]] = {}
        for event in self.events:
            for horizon in HORIZONS_S:
                if horizon not in event.returns:
                    continue
                key = EdgeKey(source_id=event.source_id,
                              mechanism=event.mechanism,
                              claim_type=event.claim_type,
                              regime=event.regime, horizon_s=horizon)
                grouped.setdefault(key, []).append(event)
        return grouped

    def score(self, key: EdgeKey, events: Sequence[InsiderEvent]) -> SourceEdge:
        edge = SourceEdge(key=key, n=len(events), last_validated=time.time())
        if len(events) < self.min_cell_events:
            edge.status = "DATA_BLOCKED"
            edge.detail = (f"{len(events)} events in this cell, "
                           f"{self.min_cell_events} required; a rate from this "
                           "many observations is noise wearing a decimal point")
            return edge

        lags = sorted(event.observation_lag for event in events)
        edge.first_seen_lag_p05 = lags[max(0, int(0.05 * (len(lags) - 1)))]
        edge.first_seen_lag_p50 = statistics.median(lags)
        edge.first_seen_lag_p95 = lags[min(len(lags) - 1,
                                           int(0.95 * (len(lags) - 1)))]

        horizon = key.horizon_s
        reference = self.reference_latency_s
        gross = [event.returns.get(horizon) for event in events]
        nets = [event.net_return(horizon, reference) for event in events]
        logs = [event.log_growth(horizon, reference) for event in events]
        present_net = [value for value in nets if value is not None]
        present_log = [value for value in logs if value is not None]

        edge.ev_gross = _median(gross)
        if present_net:
            edge.ev_net = statistics.fmean(present_net)
            edge.hit_rate = sum(1 for value in present_net
                                if value > 0) / len(present_net)
        if present_log:
            edge.e_log_w = statistics.fmean(present_log)
            if len(present_log) > 1:
                stderr = statistics.stdev(present_log) / math.sqrt(
                    len(present_log))
                edge.e_log_w_ci = (edge.e_log_w - 1.96 * stderr,
                                   edge.e_log_w + 1.96 * stderr)

        multiples = [event.max_feasible_multiple for event in events
                     if event.max_feasible_multiple is not None]
        if multiples:
            edge.p_2x = sum(1 for m in multiples if m >= 2.0) / len(multiples)
            edge.p_5x = sum(1 for m in multiples if m >= 5.0) / len(multiples)
            edge.p_10x = sum(1 for m in multiples if m >= 10.0) / len(multiples)

        rug_rate, rug_hits, rug_n = _rate([event.rugged for event in events])
        edge.p_rug = rug_rate
        if rug_n:
            edge.p_rug_ci = wilson_interval(rug_hits, rug_n)

        edge.mfe_median = _median([event.mfe for event in events])
        edge.mae_median = _median([event.mae for event in events])
        maes = [event.mae for event in events if event.mae is not None]
        edge.max_drawdown = min(maes) if maes else None

        for latency in ACHIEVABLE_LATENCIES_S:
            values = [event.net_return(horizon, latency) for event in events]
            values = [value for value in values if value is not None]
            edge.latency_decay[latency] = (
                statistics.fmean(values) if values else None)

        edge.p_executable = sum(
            1 for event in events
            if event.executable(reference)) / len(events)
        edge.copyability = edge.p_executable
        edge.capacity_sol = self._capacity(events, horizon)
        edge.crowding_decay = self._crowding_decay(events, horizon)

        claim_rate, _, _ = _rate([event.claim_correct for event in events])
        edge.p_claim_correct = claim_rate
        flow_rate, _, _ = _rate(
            [event.flow_prediction_correct for event in events])
        edge.p_flow_arrives = flow_rate

        edge.distribution_probability = self._distribution_probability(events)
        edge.manipulation_probability = edge.distribution_probability
        edge.decay_status = self._decay_status(events, horizon)

        edge.status = "OK"
        return edge

    def _capacity(self, events: Sequence[InsiderEvent], horizon: float
                  ) -> Optional[float]:
        """The largest size that still cleared, not the largest size seen.

        Capacity is the point of this metric, and it is not the mean executable
        size: a source whose edge survives at 0.5 SOL and dies at 5 has a
        capacity of 0.5, whatever the average book depth was.
        """
        sized: List[Tuple[float, float]] = []
        for event in events:
            entry = event.entry_at(self.reference_latency_s)
            net = event.net_return(horizon, self.reference_latency_s)
            if entry is None or entry.executable_sol is None or net is None:
                continue
            sized.append((float(entry.executable_sol), net))
        if len(sized) < self.min_cell_events:
            return None
        sized.sort(key=lambda item: item[0])
        best: Optional[float] = None
        # Walk size buckets upward; the capacity is the largest bucket whose
        # mean net return is still positive.
        buckets = max(2, min(5, len(sized) // 4))
        span = max(1, len(sized) // buckets)
        for index in range(0, len(sized), span):
            chunk = sized[index:index + span]
            if len(chunk) < 2:
                continue
            if statistics.fmean(value for _, value in chunk) > 0:
                best = chunk[-1][0]
        return best

    def _crowding_decay(self, events: Sequence[InsiderEvent], horizon: float
                        ) -> Optional[float]:
        """Net return when crowded, minus net return when not.

        Negative means the edge is competed away as soon as other sources name
        the same mint -- which is the normal case, and the reason a source that
        looks good in aggregate can be untradeable in practice.
        """
        quiet: List[float] = []
        crowded: List[float] = []
        for event in events:
            net = event.net_return(horizon, self.reference_latency_s)
            if net is None:
                continue
            (crowded if event.concurrent_mentions >= self.crowding_threshold
             else quiet).append(net)
        if len(quiet) < 3 or len(crowded) < 3:
            return None
        return statistics.fmean(crowded) - statistics.fmean(quiet)

    @staticmethod
    def _distribution_probability(events: Sequence[InsiderEvent]
                                  ) -> Optional[float]:
        """How often linked wallets bought before the post and sold after it.

        Not a moral judgement. It is the shape that makes a source excellent at
        predicting flow and ruinous to hold alongside, and the two need
        separating before either can be used.
        """
        judged = [event for event in events
                  if event.pre_post_accumulation_usd is not None
                  and event.post_sell_usd is not None]
        if not judged:
            return None
        hits = sum(1 for event in judged
                   if (event.pre_post_accumulation_usd or 0) > 0
                   and (event.post_sell_usd or 0) > 0)
        return hits / len(judged)

    def _decay_status(self, events: Sequence[InsiderEvent], horizon: float
                      ) -> str:
        """Is this edge still there, or is the cell living on old evidence?"""
        ordered = sorted(events, key=lambda event: event.observed_at)
        half = len(ordered) // 2
        if half < 4:
            return "UNKNOWN"
        def _mean(chunk: Sequence[InsiderEvent]) -> Optional[float]:
            values = [event.net_return(horizon, self.reference_latency_s)
                      for event in chunk]
            values = [value for value in values if value is not None]
            return statistics.fmean(values) if values else None
        early, late = _mean(ordered[:half]), _mean(ordered[half:])
        if early is None or late is None:
            return "UNKNOWN"
        if late <= 0 < early:
            return "DECAYED"
        if early > 0 and late < early * 0.5:
            return "DECAYING"
        if late > early:
            return "STRENGTHENING"
        return "STABLE"

    # -- reporting -------------------------------------------------------

    def edges(self) -> Dict[EdgeKey, SourceEdge]:
        return {key: self.score(key, events)
                for key, events in self.cells().items()}

    def scored_edges(self) -> List[SourceEdge]:
        """Only the cells that cleared the sample-size gate."""
        return [edge for edge in self.edges().values() if edge.status == "OK"]

    def regime_stability(self, source_id: str, mechanism: Mechanism,
                         claim_type: ClaimType, horizon_s: float
                         ) -> Dict[str, Any]:
        """Does this edge survive a change of tape?

        Reported as the spread of E[log W] across regimes. A source whose edge
        exists only in one regime is a regime bet wearing a source's name, and
        sizing it as though it generalises is how a good month becomes a bad
        quarter.
        """
        found: Dict[str, Optional[float]] = {}
        for key, edge in self.edges().items():
            if (key.source_id == source_id and key.mechanism is mechanism
                    and key.claim_type is claim_type
                    and key.horizon_s == horizon_s and edge.status == "OK"):
                found[key.regime] = edge.e_log_w
        present = [value for value in found.values() if value is not None]
        if len(present) < 2:
            return {"status": "DATA_BLOCKED",
                    "reason": (f"{len(present)} regimes scored; stability is "
                               "not observable from one tape"),
                    "regimes": found}
        return {"status": "OK", "regimes": found,
                "min": min(present), "max": max(present),
                "spread": max(present) - min(present),
                "positive_in_all": all(value > 0 for value in present)}

    def summary(self) -> Dict[str, Any]:
        edges = self.edges()
        scored = [edge for edge in edges.values() if edge.status == "OK"]
        return {
            "schema": INSIDER_EVENT_SCHEMA_VERSION,
            "events": len(self.events),
            "refused": len(self.refusals),
            "refusals_by_source": self.refusals_by_source(),
            "cells": len(edges),
            "cells_scored": len(scored),
            "cells_data_blocked": len(edges) - len(scored),
            "min_cell_events": self.min_cell_events,
        }


def from_source_post(post: Any, outcome: Any, *, event_id: str,
                     lawful_access: LawfulAccess = LawfulAccess.PUBLIC,
                     mechanism: Mechanism = Mechanism.UNCLASSIFIED,
                     claim_type: ClaimType = ClaimType.UNSPECIFIED,
                     regime: str = "unknown",
                     cost_fraction: Optional[float] = None) -> InsiderEvent:
    """Adapt the genealogy's existing records into the canonical shape.

    Written as an adapter rather than a migration so `source_genealogy` keeps
    working unchanged. Fields the old records never carried stay None; they do
    not acquire a value by passing through here.
    """
    event = InsiderEvent(
        event_id=event_id,
        source_id=getattr(post, "source_id", ""),
        source_at=float(getattr(post, "posted_at", 0.0) or 0.0),
        observed_at=float(getattr(post, "observed_at", 0.0) or 0.0),
        lawful_access=lawful_access,
        token=str(getattr(post, "token", "") or ""),
        mint=str(getattr(post, "token", "") or ""),
        claim_type=claim_type,
        mechanism=mechanism,
        regime=regime,
        linked_wallets=list(getattr(post, "named_wallets", []) or []),
        cost_fraction=cost_fraction)
    returns = getattr(outcome, "executable_returns", None) or {}
    event.returns = {float(key): value for key, value in returns.items()}
    event.rugged = getattr(outcome, "rugged", None)
    event.max_feasible_multiple = getattr(outcome, "max_feasible_multiple",
                                          None)
    event.pre_post_accumulation_usd = getattr(
        outcome, "pre_post_accumulation_usd", None)
    event.post_sell_usd = getattr(outcome, "post_sell_usd", None)
    acceleration = getattr(outcome, "flow_acceleration", None)
    if acceleration is not None:
        event.flow_prediction_correct = bool(acceleration > 1.0)
    return event

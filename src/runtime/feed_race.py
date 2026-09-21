"""Which feed saw it first, measured rather than believed.

Every provider claims to be fastest. The claim is untestable from outside and
usually true in the benchmark the provider chose, so the only way to know
which one reaches THIS box first is to run several and record who arrived
first for events all of them eventually delivered.

The landing router already does the equivalent for outbound transactions --
race the mechanisms, learn from what lands. This is the inbound half, and it
was the missing side: a desk can be microseconds fast at deciding and still
lose every launch because it heard about it 40ms late.

What it produces, per feed:

    win share       how often it was first, over events everyone saw
    lead p50/p95    how far ahead it was when it won
    coverage        the share of all known events it delivered AT ALL
    unique finds    events only this feed ever delivered

Coverage matters more than speed and is the statistic a naive race hides. A
feed that delivers 60% of launches 5ms sooner is worse than one that delivers
100% of them 5ms later, because the 40% it drops are not slow, they are
INVISIBLE -- and a win-rate table computed only over events both feeds
delivered will rank the fast, lossy feed first. So coverage is computed over
the union of everything any feed delivered, and unique finds are counted
separately: a feed nobody else duplicates is carrying its own weight even if
it never wins a race.

Only first arrival mutates state. The duplicates are not waste -- they are
the measurement, and they are also the redundancy that makes one feed's
outage a slower desk rather than a blind one.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

FEED_RACE_SCHEMA_VERSION = "v1"

#: Events kept for the race statistics. Bounded: this is a measurement of the
#: recent past, and an unbounded ledger of every launch ever seen would grow
#: without limit to answer a question about the last few hours.
DEFAULT_HISTORY = 4096

#: An event is only scored once every feed has had a fair chance to deliver
#: it. Scoring immediately would count every event as a unique find for
#: whichever feed happened to be first.
SETTLE_SECONDS = 5.0

#: Below this many settled events a feed has no measurable performance.
MIN_EVENTS_FOR_VERDICT = 50


@dataclass
class Arrival:
    feed: str
    at: float


@dataclass
class RacedEvent:
    key: str
    first_seen: float
    arrivals: Dict[str, float] = field(default_factory=dict)

    def settled(self, now: float) -> bool:
        return now - self.first_seen >= SETTLE_SECONDS

    @property
    def winner(self) -> Optional[str]:
        if not self.arrivals:
            return None
        return min(self.arrivals.items(), key=lambda item: item[1])[0]


@dataclass
class FeedVerdict:
    status: str
    feed: str = ""
    delivered: int = 0
    wins: int = 0
    unique: int = 0
    coverage: float = 0.0
    lead_p50_ms: Optional[float] = None
    lead_p95_ms: Optional[float] = None
    detail: str = ""

    @property
    def win_share(self) -> Optional[float]:
        if self.status != "OK" or self.delivered <= 0:
            return None
        return self.wins / self.delivered


def _percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return float(ordered[index])


class FeedRace:
    """Runs several feeds, uses the first arrival, and learns from the rest."""

    def __init__(self, feeds: Sequence[str] = (), history: int = DEFAULT_HISTORY):
        self.feeds: List[str] = list(feeds)
        self.history = int(history)
        self._open: Dict[str, RacedEvent] = {}
        self._settled: Deque[RacedEvent] = deque(maxlen=self.history)
        # Keys already handed downstream, so a duplicate never re-triggers a
        # decision. This is the property that makes racing safe at all.
        #
        # ONE structure, insertion-ordered. It used to be a bounded deque
        # beside a set, and a bounded deque evicts its own leftmost item on
        # append -- silently, without telling the set. So once at capacity
        # every arrival dropped the oldest key from the deque while leaving
        # it in the set for ever, and the explicit eviction that followed
        # then removed a NEWER key from both. The set grew without bound and
        # simultaneously lost keys it was still meant to be protecting, which
        # means a duplicate could re-trigger a decision: exactly the one
        # property this class exists to guarantee.
        self._delivered: "Dict[str, None]" = {}
        self._delivered_capacity = max(1, self.history * 2)
        self.duplicates = 0

    def register_feed(self, feed: str) -> None:
        if feed not in self.feeds:
            self.feeds.append(feed)

    def observe(self, feed: str, key: str, at: Optional[float] = None) -> bool:
        """Record an arrival. True only for the FIRST feed to deliver `key`.

        The return value is the whole contract: the caller acts on True and
        discards on False, so an event delivered by five feeds produces one
        decision and four measurements.
        """
        moment = time.time() if at is None else float(at)
        self.register_feed(feed)
        event = self._open.get(key)
        if event is None:
            if key in self._delivered:
                # Arrived after the event already settled and was evicted.
                self.duplicates += 1
                return False
            event = RacedEvent(key=key, first_seen=moment)
            self._open[key] = event
            event.arrivals[feed] = moment
            self._delivered[key] = None
            while len(self._delivered) > self._delivered_capacity:
                # Oldest first, and the eviction is the ONLY one: nothing
                # else can drop a key behind this loop's back.
                self._delivered.pop(next(iter(self._delivered)))
            self._sweep(moment)
            return True
        if feed not in event.arrivals:
            event.arrivals[feed] = moment
        self.duplicates += 1
        self._sweep(moment)
        return False

    def _sweep(self, now: float) -> None:
        ready = [key for key, event in self._open.items() if event.settled(now)]
        for key in ready:
            self._settled.append(self._open.pop(key))

    def verdict(self, feed: str) -> FeedVerdict:
        settled = list(self._settled)
        if len(settled) < MIN_EVENTS_FOR_VERDICT:
            return FeedVerdict(
                status="DATA_BLOCKED", feed=feed, delivered=len(settled),
                detail=(f"{len(settled)} settled events, below the "
                        f"{MIN_EVENTS_FOR_VERDICT} needed"))
        delivered = [e for e in settled if feed in e.arrivals]
        wins = [e for e in delivered if e.winner == feed]
        unique = [e for e in delivered if len(e.arrivals) == 1]
        leads = []
        for event in wins:
            others = [t for name, t in event.arrivals.items() if name != feed]
            if others:
                leads.append((min(others) - event.arrivals[feed]) * 1000.0)
        return FeedVerdict(
            status="OK", feed=feed, delivered=len(delivered), wins=len(wins),
            unique=len(unique), coverage=len(delivered) / len(settled),
            lead_p50_ms=_percentile(leads, 0.5), lead_p95_ms=_percentile(leads, 0.95),
            detail=(f"delivered {len(delivered)} of {len(settled)} settled events, "
                    f"first on {len(wins)}, alone on {len(unique)}"))

    def report(self) -> Dict[str, Any]:
        verdicts = {feed: self.verdict(feed) for feed in self.feeds}
        measured = [v for v in verdicts.values() if v.status == "OK"]
        return {
            "status": "OK" if measured else "DATA_BLOCKED",
            "schema": FEED_RACE_SCHEMA_VERSION,
            "feeds": len(self.feeds),
            "settled_events": len(self._settled),
            "duplicates_discarded": self.duplicates,
            # Stated because a race table read alone invites the wrong
            # conclusion: the fastest feed is not the one to keep if it is
            # also the one that drops events.
            "note": ("coverage outranks speed: a feed that misses events is "
                     "not slow on them, it is blind, and a win-rate computed "
                     "only over shared events cannot see that"),
            "by_feed": {
                feed: {"status": v.status, "coverage": round(v.coverage, 4),
                       "win_share": v.win_share, "wins": v.wins,
                       "unique_finds": v.unique,
                       "lead_p50_ms": v.lead_p50_ms, "lead_p95_ms": v.lead_p95_ms,
                       "detail": v.detail}
                for feed, v in sorted(verdicts.items())},
        }

    def best_feed(self) -> Optional[str]:
        """The feed to keep if only one could be kept. Coverage first.

        Coverage first and speed second, deliberately: being 8ms later on
        every event costs a little on each, and missing 5% of events costs
        everything on those.
        """
        ranked = [v for v in (self.verdict(f) for f in self.feeds)
                  if v.status == "OK"]
        if not ranked:
            return None
        ranked.sort(key=lambda v: (round(v.coverage, 3), v.win_share or 0.0),
                    reverse=True)
        return ranked[0].feed

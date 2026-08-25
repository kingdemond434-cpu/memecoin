"""One interface and one event shape, so adding a source is not a new system.

The engine should not care which network a message arrived on. It cares when
the thing happened, when we learned about it, which entity it concerns, and
what tokens or links it names. Everything else is adapter detail.

Two timestamps, always, and never collapsed into one. ``source_at`` is when
the source published; ``observed_at`` is when it reached us. Their difference
is the source's own delivery latency, which no amount of local speed
recovers, and it is the single most important number for deciding whether a
source is worth acting on. A pipeline carrying one timestamp cannot compute it
and will rank a slow accurate source above a fast one.

Source health is first-class rather than an afterthought. A dead feed that
reports nothing looks exactly like a quiet feed that reports nothing, and the
failure mode of a large source mesh is not one adapter crashing loudly -- it
is six of them going silent while the dashboard stays green. Every adapter is
therefore required to declare when it last successfully polled, separately
from when it last produced an event.
"""

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

EVENT_SCHEMA_VERSION = "v1"


class SourceClass(Enum):
    """What kind of place this came from. Drives expectations, not trust."""

    CHAIN = "chain"
    CHAT = "chat"
    SOCIAL = "social"
    VIDEO = "video"
    OFFICIAL = "official"
    NEWS = "news"
    FEED = "feed"
    CODE = "code"
    WEB = "web"


@dataclass(frozen=True)
class Event:
    """The canonical shape every adapter normalises into."""

    source_id: str
    source_class: SourceClass
    source_at: float
    observed_at: float
    text: str = ""
    language: str = ""
    entity_ids: Sequence[str] = ()
    token_addresses: Sequence[str] = ()
    urls: Sequence[str] = ()
    author_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def observation_lag(self) -> float:
        """Publication to observation. Clamped at zero, never negative.

        Sources disagree about clocks and some stamp optimistically. A negative
        lag is a clock artefact, and letting it through would make a source
        look like it reached us before it published -- which then ranks it
        first.
        """
        return max(0.0, self.observed_at - self.source_at)

    @property
    def content_hash(self) -> str:
        """Stable identity for deduplication across sources.

        Hashes content, not the source, so the same message reposted by four
        channels collapses to one event with four observations -- which is
        what lead-lag analysis needs and what naive per-source dedupe
        destroys.
        """
        payload = f"{self.text}|{'|'.join(sorted(self.token_addresses))}"
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["source_class"] = self.source_class.value
        data["observation_lag"] = self.observation_lag
        data["content_hash"] = self.content_hash
        data["schema_version"] = EVENT_SCHEMA_VERSION
        return data


class SourceState(Enum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    DEAD = "DEAD"
    NEVER_STARTED = "NEVER_STARTED"


@dataclass
class SourceHealth:
    """Whether an adapter is working, told apart from whether it is quiet."""

    source_id: str
    state: SourceState
    last_poll_ok_at: Optional[float] = None
    last_event_at: Optional[float] = None
    consecutive_failures: int = 0
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "state": self.state.value}


class EventSource(ABC):
    """One lawful public source.

    Adapters normalise and nothing else. No adapter decides whether an event
    matters, because a source that filters on its own idea of relevance
    silently becomes the model, and its filtering is invisible to everything
    that scores it.
    """

    #: Default cadence expectations. Overridden per source, because five
    #: minutes without a Telegram push connection and five minutes without a
    #: regional RSS story are not the same fact, and one universal clock calls
    #: the healthy feed dead or lets the dead one look healthy.
    dead_after_seconds: float = 900.0
    degraded_after_seconds: float = 300.0

    def __init__(self, source_id: str, source_class: SourceClass,
                 degraded_after_seconds: Optional[float] = None,
                 dead_after_seconds: Optional[float] = None):
        self.source_id = source_id
        self.source_class = source_class
        if degraded_after_seconds is not None:
            self.degraded_after_seconds = float(degraded_after_seconds)
        if dead_after_seconds is not None:
            self.dead_after_seconds = float(dead_after_seconds)
        self._last_poll_ok_at: Optional[float] = None
        self._last_event_at: Optional[float] = None
        self._consecutive_failures = 0
        self._timeouts = 0

    def note_timeout(self) -> None:
        """Record that a poll exceeded its budget.

        Counted with failures rather than separately: from the mesh's point of
        view a source that never answers and one that answers too late are the
        same problem.
        """
        self._timeouts += 1
        self._consecutive_failures += 1

    @abstractmethod
    async def poll(self) -> List[Event]:
        """Fetch and normalise. May return an empty list; that is not failure."""

    async def collect(self, now: Optional[float] = None) -> List[Event]:
        """Poll, recording health regardless of outcome.

        An adapter raising is recorded and swallowed. One dead source must not
        take down the mesh, and a mesh that stops on the first failure is a
        mesh whose coverage silently equals its least reliable member.
        """
        now = time.time() if now is None else now
        try:
            events = await self.poll()
        except Exception as exc:
            self._consecutive_failures += 1
            logger.warning("source %s poll failed (%d consecutive): %s",
                           self.source_id, self._consecutive_failures, exc)
            return []
        self._consecutive_failures = 0
        self._last_poll_ok_at = now
        if events:
            self._last_event_at = now
        return events

    def health(self, now: Optional[float] = None) -> SourceHealth:
        now = time.time() if now is None else now
        if self._last_poll_ok_at is None:
            return SourceHealth(
                self.source_id,
                SourceState.NEVER_STARTED if self._consecutive_failures == 0
                else SourceState.DEAD,
                consecutive_failures=self._consecutive_failures,
                detail="no successful poll yet")
        silence = now - self._last_poll_ok_at
        if silence >= self.dead_after_seconds:
            state = SourceState.DEAD
        elif silence >= self.degraded_after_seconds:
            state = SourceState.DEGRADED
        else:
            state = SourceState.OK
        return SourceHealth(
            self.source_id, state, last_poll_ok_at=self._last_poll_ok_at,
            last_event_at=self._last_event_at,
            consecutive_failures=self._consecutive_failures,
            # Stated explicitly, because a dead feed and a quiet one produce
            # the same event count and the difference is the whole point.
            detail=(f"{silence:.0f}s since last successful poll "
                    f"(degraded at {self.degraded_after_seconds:.0f}s, "
                    f"dead at {self.dead_after_seconds:.0f}s, "
                    f"{self._timeouts} timeouts)"))


class SourceMesh:
    """Every adapter, one event stream, one health surface.

    Sources are polled CONCURRENTLY into a bounded queue. Awaiting them one
    after another means a single slow endpoint delays every source behind it,
    which is exactly backwards for a system whose value is being first: a
    stalled regional RSS feed must have no effect on Telegram or chain
    latency. With hundreds of sources the serial version's worst case is the
    sum of every timeout.

    The queue is bounded and drops the OLDEST on overflow, for the same reason
    the archive writer does: a queue that blocks converts a slow source into a
    stalled mesh, and one that grows without bound converts it into an OOM.
    """

    def __init__(self, sources: Optional[Sequence[EventSource]] = None,
                 dedupe_window: float = 300.0, max_queue: int = 10_000,
                 poll_timeout: float = 5.0):
        self.sources: List[EventSource] = list(sources or ())
        self.dedupe_window = dedupe_window
        self.max_queue = max(1, max_queue)
        # A source that has not answered in this long is not worth waiting
        # for; it is worth marking unhealthy and moving on.
        self.poll_timeout = poll_timeout
        self.dropped = 0
        # content_hash -> (first_seen_at, [source_ids in arrival order]).
        # Kept rather than discarded, because who saw it FIRST is the signal.
        self._seen: Dict[str, Any] = {}

    def add(self, source: EventSource) -> None:
        self.sources.append(source)

    def _expire(self, now: float) -> None:
        stale = [key for key, (first_at, _) in self._seen.items()
                 if now - first_at > self.dedupe_window]
        for key in stale:
            self._seen.pop(key, None)

    async def _collect_one(self, source: "EventSource", now: float) -> List[Event]:
        """Poll one source under a timeout, never propagating its failure."""
        try:
            return await asyncio.wait_for(source.collect(now), timeout=self.poll_timeout)
        except asyncio.TimeoutError:
            logger.warning("source %s exceeded the %.1fs poll timeout",
                           source.source_id, self.poll_timeout)
            source.note_timeout()
            return []
        except Exception as exc:  # pragma: no cover - collect already guards
            logger.warning("source %s raised past its own guard: %s",
                           source.source_id, exc)
            return []

    async def collect(self, now: Optional[float] = None) -> List[Event]:
        """Poll every source concurrently, returning first-observations only.

        A repeat of content already seen is not returned as a new event, but
        its source IS recorded against the original. Dropping the repeat
        outright would throw away exactly the lead-lag evidence that says
        which source is upstream of which.

        Results are merged in completion order, so a source that answered in
        5ms is not held behind one that took 4 seconds -- and the arrival
        order recorded for lead-lag is the order things actually arrived.
        """
        now = time.time() if now is None else now
        self._expire(now)
        if not self.sources:
            return []

        batches = await asyncio.gather(
            *(self._collect_one(source, now) for source in self.sources))

        fresh: List[Event] = []
        for batch in batches:
            for event in batch:
                key = event.content_hash
                if key in self._seen:
                    self._seen[key][1].append(event.source_id)
                    continue
                if len(fresh) >= self.max_queue:
                    # Oldest first: the newest observation is the one a
                    # decision might still depend on.
                    fresh.pop(0)
                    self.dropped += 1
                self._seen[key] = (now, [event.source_id])
                fresh.append(event)
        return fresh

    def repeaters_of(self, content_hash: str) -> List[str]:
        """Sources that carried this content, in the order they were observed."""
        entry = self._seen.get(content_hash)
        return list(entry[1]) if entry else []

    def health(self, now: Optional[float] = None) -> Dict[str, Any]:
        now = time.time() if now is None else now
        reports = [source.health(now) for source in self.sources]
        by_state: Dict[str, int] = {}
        for report in reports:
            by_state[report.state.value] = by_state.get(report.state.value, 0) + 1
        unhealthy = [report.to_dict() for report in reports
                     if report.state is not SourceState.OK]
        return {
            "sources": len(reports),
            "by_state": by_state,
            # The failure mode of a large mesh is six adapters going silent
            # while the dashboard stays green, so the silent ones are named.
            "unhealthy": unhealthy,
            "dropped_events": self.dropped,
            "coverage": (by_state.get("OK", 0) / len(reports)) if reports else None,
        }

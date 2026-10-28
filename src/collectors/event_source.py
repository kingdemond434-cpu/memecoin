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
import heapq
import logging
import threading
import time
from collections import OrderedDict, deque
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
    #: How often this source is worth asking. A property of the source, not of
    #: the mesh: a chat channel that pushes and a daily regulatory feed do not
    #: belong on one shared cadence, and a single polling loop forces exactly
    #: that.
    poll_interval_seconds: float = 1.0
    #: How long this source may take to answer. A property of the source for
    #: the same reason cadence is: a feed served from the other side of the
    #: world is not slow because it is broken. Measured 2026-08-30, the
    #: Korean, Chinese and Japanese outlets all exceeded the mesh's 5s
    #: default on every poll and produced nothing, while answering fine to
    #: curl. None means "use the mesh default".
    poll_timeout_seconds: Optional[float] = None

    def __init__(self, source_id: str, source_class: SourceClass,
                 degraded_after_seconds: Optional[float] = None,
                 dead_after_seconds: Optional[float] = None,
                 poll_interval_seconds: Optional[float] = None,
                 poll_timeout_seconds: Optional[float] = None):
        self.source_id = source_id
        self.source_class = source_class
        # How often this source is worth asking, which is a property of the
        # source and not of the mesh. A chat channel that pushes and a daily
        # regulatory feed do not belong on one shared cadence, and putting
        # them on one is what a single polling loop forces.
        if poll_interval_seconds is not None:
            self.poll_interval_seconds = max(0.01, float(poll_interval_seconds))
        if poll_timeout_seconds is not None:
            self.poll_timeout_seconds = max(0.1, float(poll_timeout_seconds))
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
            # Keep the first failures loud, then log powers of two. A broken
            # endpoint polled forever must not turn the research log into a
            # denial of service or hide failures from healthy sources.
            failures = self._consecutive_failures
            emit_warning = failures <= 3 or failures & (failures - 1) == 0
            log = logger.warning if emit_warning else logger.debug
            log("source %s poll failed (%d consecutive): %s",
                self.source_id, failures, exc)
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

    def retry_delay(self, elapsed_seconds: float = 0.0,
                    max_backoff_seconds: float = 300.0) -> float:
        """Next cadence, with bounded exponential backoff after failures."""
        base = max(0.01, float(self.poll_interval_seconds))
        if self._consecutive_failures:
            # A 1s feed backs off 1,2,4,... seconds. A 5m repository
            # remains at its declared cadence and never retries faster merely
            # because the previous request failed.
            exponent = min(max(0, self._consecutive_failures - 1), 12)
            interval = min(max_backoff_seconds, max(base, 1.0) * (2 ** exponent))
        else:
            interval = base
        return max(0.0, interval - max(0.0, float(elapsed_seconds)))


class SourceMesh:
    """Every adapter, one event stream, one health surface.

    Sources are polled CONCURRENTLY into a bounded queue. Awaiting them one
    after another means a single slow endpoint delays every source behind it,
    which is exactly backwards for a system whose value is being first: a
    stalled regional RSS feed must have no effect on Telegram or chain
    latency. With hundreds of sources the serial version's worst case is the
    sum of every timeout.

    Live delivery rotates across nonempty sources. A burst from one producer
    cannot put every other source behind its entire backlog. Active memory is
    bounded. The volatile compatibility queue evicts on overflow; the durable
    journal instead retains every committed body until its canonical ACK.
    Durable delivery prefetches a bounded fair batch to amortize lease commits.
    """

    def __init__(self, sources: Optional[Sequence[EventSource]] = None,
                 dedupe_window: float = 300.0, max_queue: int = 10_000,
                 poll_timeout: float = 5.0, journal_path=None, journal_batch_size: int = 8):
        if type(journal_batch_size) is not int or not 1 <= journal_batch_size <= 32:
            raise ValueError('journal_batch_size must be between 1 and 32')
        self.journal_batch_size = journal_batch_size
        self._journal_prefetch = deque()
        self._journal_read_gate = asyncio.Lock()
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
        self._seen_expiry = []
        # One total memory budget, with no fixed source admission quota.
        # Each nonempty source gets one delivery before its next turn.
        self._pending_sources = OrderedDict()
        self._queue_lock = threading.RLock()
        self._pending_count = 0
        self._available = asyncio.Event()
        self._delivered_by_source = {}
        self._dropped_by_source = {}
        self._queued_durable = set()
        self.queued_durable_repeats_coalesced = 0
        self._producers: List[asyncio.Task] = []
        self._running = False
        self._owner_loop = None
        self.journal = None
        self._journal_leases = {}
        self._journal_write_gate = asyncio.Lock()
        self._journal_write_failures = 0
        self._journal_last_error = None
        if journal_path is not None:
            from src.collectors.source_journal import SourceJournal
            self.journal = SourceJournal(journal_path)

    @staticmethod
    async def _disk_call(call, *args, timing=None, phase=None):
        if timing is not None:
            call, args = timing.worker(phase, call, *args), ()
        pending = asyncio.create_task(asyncio.to_thread(call, *args))
        try:
            return await asyncio.shield(pending)
        except asyncio.CancelledError:
            # A cancelled await cannot cancel an OS-thread transaction. Drain
            # it before shutdown may close the database; unacknowledged rows replay.
            await pending
            raise

    async def publish_batch(self, events):
        if self.journal is None:
            for event in events:
                self._publish(event, time.time())
            return
        # Admission stays on the owner loop; SQLite waits never hold its lock.
        # Keep the admitted batch until committed. A failed write does not poll
        # the source again or allow the deduper to silently suppress the retry.
        with self._queue_lock:
            fresh = [event for event in events if self._admit(event, time.time())]
        for offset in range(0, len(fresh), 128):
            batch = fresh[offset:offset+128]
            while True:
                try:
                    async with self._journal_write_gate:
                        await self._disk_call(self.journal.append, batch, time.time())
                    self._journal_last_error = None
                    self._available.set()
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._journal_write_failures += 1
                    self._journal_last_error = type(exc).__name__
                    failures = self._journal_write_failures
                    if failures <= 3 or failures & (failures-1) == 0:
                        logger.warning('source journal intake deferred: %s', type(exc).__name__)
                    await asyncio.sleep(1)

    def delivery_receipt(self, event):
        leased = self._journal_leases.get(id(event))
        return dict(leased[1]) if leased is not None and leased[0] is event else None

    async def acknowledge_event(self, event, job, reference, *, timing=None):
        if self.journal is None:
            return
        _, receipt, generation = self._journal_leases[id(event)]
        await self._disk_call(self.journal.acknowledge, event, receipt, generation, job, reference, time.time(),
                              timing=timing, phase="journal_ack")
        self._journal_leases.pop(id(event), None)

    async def acknowledge_bound_event(self, event, job, lookup, *, timing=None):
        """Fresh binding and exact FULL ACK in one drained disk dispatch.

        Capture the lease on its owner loop; only committed database work runs
        in the worker. No owner index mutation precedes the durable ACK.
        """
        if self.journal is None:
            return
        _, receipt, generation = self._journal_leases[id(event)]
        def commit():
            if timing is None:
                reference = lookup(job)
            else:
                with timing.measure("canonical_ref_inline"):
                    reference = lookup(job)
            if not reference:
                raise ValueError('canonical narrative source reference unavailable')
            if timing is None:
                self.journal.acknowledge(event, receipt, generation, job, reference, time.time())
            else:
                with timing.measure("journal_ack_inline"):
                    self.journal.acknowledge(event, receipt, generation, job, reference, time.time())
        await self._disk_call(commit, timing=timing, phase="canonical_ack")
        self._journal_leases.pop(id(event), None)

    async def retry_event(self, event):
        if self.journal is None:
            return
        _, receipt, generation = self._journal_leases[id(event)]
        await self._disk_call(self.journal.retry, event, receipt, generation, time.time())
        self._journal_leases.pop(id(event), None)
        self._available.set()

    def close(self):
        if self.journal is not None:
            self.journal.close()
            self._journal_prefetch.clear()

    def add(self, source: EventSource) -> None:
        if any(existing.source_id == source.source_id for existing in self.sources):
            return
        self.sources.append(source)
        if self._running and self._owner_loop is not None:
            def start_added_source():
                name = "source:" + source.source_id
                if self._running and not any(task.get_name() == name for task in self._producers):
                    self._producers.append(asyncio.create_task(self._producer(source), name=name))
            self._owner_loop.call_soon_threadsafe(start_added_source)

    def _expire(self, now: float) -> None:
        # The live path must expire too. A heap visits only expired entries,
        # not every cached message on every incoming observation. Timestamps
        # may arrive out of order (batch/live clocks), so a FIFO is insufficient.
        with self._queue_lock:
            while self._seen_expiry and now-self._seen_expiry[0][0] > self.dedupe_window:
                first_at, key = heapq.heappop(self._seen_expiry)
                current = self._seen.get(key)
                if current is not None and current[0] == first_at:
                    del self._seen[key]

    async def _collect_one(self, source: "EventSource", now: float) -> List[Event]:
        """Poll one source under a timeout, never propagating its failure.

        The source's own budget wins where it declares one. A distant feed
        that needs eight seconds is not unhealthy, and holding every source
        to the nearest one's latency silently drops whole regions.
        """
        timeout = getattr(source, "poll_timeout_seconds", None) or self.poll_timeout
        try:
            return await asyncio.wait_for(source.collect(now), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("source %s exceeded the %.1fs poll timeout",
                           source.source_id, timeout)
            source.note_timeout()
            return []
        except Exception as exc:  # pragma: no cover - collect already guards
            logger.warning("source %s raised past its own guard: %s",
                           source.source_id, exc)
            return []

    def _admit(self, event: Event, now: float) -> bool:
        """First observation, or a repeat recorded against the original.

        Dropping a repeat outright would throw away exactly the lead-lag
        evidence that says which source is upstream of which, so the repeat's
        source is appended to the original's list and only the event itself
        is suppressed.
        """
        self._expire(now)
        metadata = getattr(event, "metadata", None) or {}
        if metadata.get("onion_observation_id") or metadata.get("onion_directory_observation_id"):
            # This source has a durable outbox and a canonical queue ACK.
            # Suppressing redelivery here could strand a snapshot that the
            # research engine never accepted. Its stable ID is deduped there.
            return True
        key = event.content_hash
        if key in self._seen:
            self._seen[key][1].append(event.source_id)
            return False
        self._seen[key] = (now, [event.source_id])
        heapq.heappush(self._seen_expiry, (now, key))
        return True

    async def _producer(self, source: "EventSource") -> None:
        """One source, polled on its own cadence, forever.

        This is the whole point of the change. Under `gather` a source that
        takes four seconds held every event behind it for four seconds,
        including the 5ms one from the chat channel that saw the launch
        first -- and the barrier applied on every cycle, not just slow ones.
        A producer per source means a slow source is slow by itself.
        """
        while self._running:
            started = time.time()
            events = await self._collect_one(source, started)
            await self.publish_batch(events)
            elapsed = time.time() - started
            await asyncio.sleep(source.retry_delay(elapsed))

    def _publish(self, event: Event, now: float) -> None:
        """Admit fairly; overflow cannot let a hot feed evict every rare feed."""
        if self.journal is not None:
            raise RuntimeError('durable source producers must await publish_batch')
        with self._queue_lock:
            self._publish_locked(event, now)

    def _publish_locked(self, event: Event, now: float) -> None:
        durable = self._durable_key(event)
        if durable is not None and durable in self._queued_durable:
            # One unconsumed copy suffices. This is queue residency, not a
            # delivery ACK or content deduplication: dequeue/eviction releases
            # it so a failed consumer can receive the durable outbox again.
            self.queued_durable_repeats_coalesced += 1
            return
        if not self._admit(event, now):
            return
        self._pending_sources.setdefault(event.source_id, deque()).append((event, now))
        self._pending_count += 1
        if durable is not None:
            self._queued_durable.add(durable)
        while self._pending_count > self.max_queue:
            # Ties evict the oldest queued observation. A source's fresh
            # burst displaces its own backlog before a lone rare observation.
            source = max(self._pending_sources, key=lambda key:
                (len(self._pending_sources[key]), -self._pending_sources[key][0][1]))
            bucket = self._pending_sources[source]
            removed, _ = bucket.popleft()
            if not bucket:
                del self._pending_sources[source]
            self._pending_count -= 1
            self._queued_durable.discard(self._durable_key(removed))
            self.dropped += 1
            self._dropped_by_source[source] = self._dropped_by_source.get(source, 0) + 1
        self._available.set()

    async def start(self) -> int:
        """Spawn one persistent producer per source. Returns how many started."""
        if self._running:
            return len(self._producers)
        self._owner_loop = asyncio.get_running_loop()
        self._running = True
        self._producers = [asyncio.create_task(self._producer(source), name="source:" + source.source_id)
                           for source in self.sources]
        return len(self._producers)

    async def stop(self) -> None:
        self._running = False
        for task in self._producers:
            task.cancel()
        for task in self._producers:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # pragma: no cover
                pass
        self._producers = []

    async def next_event(self) -> Event:
        """Await the next event from any source. The consumer's entry point."""
        if self.journal is not None:
            # One bounded buffer across concurrent waiters. Bodies are available
            # in memory at the batch's first-delivery clock; per-event canonical
            # persistence/ACK and owner-loop indexing retain their existing order.
            async with self._journal_read_gate:
                while True:
                    if self._journal_prefetch:
                        event, receipt, generation = self._journal_prefetch.popleft()
                        self._journal_leases[id(event)] = (event, receipt, generation)
                        return event
                    self._available.clear()
                    leased = await self._disk_call(
                        lambda: self.journal.lease_many(time.time(), limit=self.journal_batch_size))
                    if leased:
                        self._journal_prefetch.extend(leased)
                        continue
                    try:
                        await asyncio.wait_for(self._available.wait(), 1)
                    except asyncio.TimeoutError:
                        pass  # Expired leases/retry deadlines become eligible too.
        while True:
            try:
                return self._take_pending()
            except asyncio.QueueEmpty:
                # There is no await between checking and clearing: a producer
                # cannot publish in a lost-wakeup window on the owner loop.
                self._available.clear()
                await self._available.wait()

    def _take_pending(self) -> Event:
        with self._queue_lock:
            return self._take_pending_locked()

    def _take_pending_locked(self) -> Event:
        if not self._pending_sources:
            raise asyncio.QueueEmpty
        source, bucket = self._pending_sources.popitem(last=False)
        event, _ = bucket.popleft()
        if bucket:
            self._pending_sources[source] = bucket
        self._pending_count -= 1
        self._queued_durable.discard(self._durable_key(event))
        self._delivered_by_source[source] = self._delivered_by_source.get(source, 0) + 1
        return event

    @staticmethod
    def _durable_key(event):
        metadata = getattr(event, 'metadata', None) or {}
        identity = metadata.get('onion_observation_id') or metadata.get('onion_directory_observation_id')
        return (event.source_id,identity) if isinstance(identity,str) and identity else None

    def drain(self, limit: Optional[int] = None) -> List[Event]:
        """Everything available right now, without waiting for any source."""
        if self.journal is not None:
            raise RuntimeError('durable source consumers must await next_event and acknowledge')
        events: List[Event] = []
        while limit is None or len(events) < limit:
            try:
                events.append(self._take_pending())
            except asyncio.QueueEmpty:
                break
        return events

    @property
    def pending(self) -> int:
        if self.journal is not None:
            return self.journal.report(time.time())['pending_events']
        with self._queue_lock:
            return self._pending_count

    def _queue_report(self, now):
        # Diagnostic projections run in a worker while the owner loop mutates
        # queues. Copy only scalar counters under this short, I/O-free lock.
        if self.journal is not None:
            report = self.journal.report(now)
            with self._queue_lock:
                self._expire(now)
                return {**report, 'delivery_policy': 'DURABLE_ROUND_ROBIN_WITH_CANONICAL_ACK',
                    'dedupe_cached_contents': len(self._seen), 'dedupe_window_seconds': self.dedupe_window,
                    'dropped_events': 0, 'journal_write_failures': self._journal_write_failures,
                    'journal_last_error_type': self._journal_last_error,
                    'journal_batch_size': self.journal_batch_size,
                    'prefetched_events': len(self._journal_prefetch),
                    'coverage_scope': 'committed source observations only; disk failure backpressures polling'}
        with self._queue_lock:
            self._expire(now)
            return {
                'dedupe_cached_contents': len(self._seen),
                'dedupe_window_seconds': self.dedupe_window,
                'pending_events': self._pending_count,
                'delivery_policy': 'ROUND_ROBIN_NONEMPTY_SOURCES',
                'pending_by_source': {key: len(rows) for key, rows in self._pending_sources.items()},
                'oldest_pending_age_by_source': {key: max(0., now-rows[0][1])
                                                for key, rows in self._pending_sources.items()},
                'delivered_by_source': dict(self._delivered_by_source),
                'dropped_by_source': dict(self._dropped_by_source),
                'dropped_events': self.dropped,
                'pending_durable_events': len(self._queued_durable),
                'queued_durable_repeats_coalesced': self.queued_durable_repeats_coalesced,
            }

    async def collect(self, now: Optional[float] = None) -> List[Event]:
        """Poll every source once and return the first observations.

        The BATCH path, kept for backfill and for tests that want a single
        deterministic sweep. It still waits for every source, which is exactly
        why it is not the live path any more: a four-second source held a
        five-millisecond one behind it on every cycle.
        """
        now = time.time() if now is None else now
        self._expire(now)
        if not self.sources:
            return []

        batches = await asyncio.gather(
            *(self._collect_one(source, now) for source in self.sources))

        fresh: List[Event] = []
        with self._queue_lock:
            for batch in batches:
                for event in batch:
                    if not self._admit(event, now):
                        continue
                    if len(fresh) >= self.max_queue:
                        fresh.pop(0)
                        self.dropped += 1
                    fresh.append(event)
        return fresh

    def repeaters_of(self, content_hash: str) -> List[str]:
        """Sources that carried this content, in the order they were observed."""
        with self._queue_lock:
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
            "streaming": self._running,
            "producers": len(self._producers),
            **self._queue_report(now),
            "coverage": (by_state.get("OK", 0) / len(reports)) if reports else None,
        }

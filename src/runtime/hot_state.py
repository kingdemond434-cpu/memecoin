"""Distilled hot state for a small trading node, with a cold moat elsewhere.

The proprietary advantage is millions of point-in-time launch observations.
The machine that has to react in milliseconds is not the machine that should
hold them. Loading a wallet's fifty thousand historical transactions to decide
whether to buy is the wrong shape of computation twice over: it is slow, and
almost none of that history is what the decision actually consumes.

So the split is:

    huge cold moat  ->  offline distillation  ->  tiny hot state  ->  execution

Research turns two million wallet transactions into a few hundred bytes of
``CompactWalletDNA``. That record is what the live node holds, and the archive
it came from lives somewhere the trading loop never touches.

Three properties matter more than the data structures:

The hot path never blocks on persistence. Archive writes go through a bounded
queue that DROPS on overflow and counts the drops. A queue that blocks when
storage is slow converts a disk problem into a missed launch, and a queue that
grows without bound converts it into an OOM. Dropping and counting is the only
option that fails in a direction the node can survive, and the count is what
makes the loss visible rather than silent.

Caps are hard and eviction is economic. Two hundred thousand irrelevant wallets
becoming active must not expand memory; something has to leave. Plain LRU is
the wrong rule here, because a prolific rugger cluster that has been quiet for
an hour is worth far more resident than a random wallet touched a second ago.
Eviction ranks by P(needed soon) * information value, with recency as one input
rather than the whole rule.

Budgets are declared, not hoped for. `HotStateBudget` states the intended
footprint per component so that exceeding it is a visible condition rather than
something discovered when the kernel kills the process.
"""

import logging
import math
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

HOT_STATE_SCHEMA_VERSION = "v1"


@dataclass
class HotStateBudget:
    """Declared footprint for a 4 GB node. Exceeding a cap is a condition, not a surprise."""

    max_active_tokens: int = 4_000
    max_hot_wallets: int = 20_000
    max_hot_creators: int = 5_000
    max_event_age_seconds: float = 3_600.0
    max_archive_queue: int = 50_000
    max_local_archive_gb: float = 5.0

    def report(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompactWalletDNA:
    """Everything a live decision needs about one wallet, and nothing else.

    Deliberately fixed-shape and small. The archive keeps the transactions;
    this keeps the conclusions, because the conclusions are what the T0
    decision consumes and re-deriving them per launch is the computation that
    does not fit in the window.
    """

    wallet: str
    launches_seen: int = 0
    early_entry_rate: float = 0.0
    monster_rate: float = 0.0
    rug_exposure: float = 0.0
    median_entry_age_s: float = 0.0
    median_exit_age_s: float = 0.0
    pnl_quality: float = 0.0
    independence: float = 0.0
    cluster_id: str = ""
    creator_links: int = 0
    last_active: float = 0.0
    # Set when the distillation ran against too little history to conclude
    # anything. A caller must treat this as "unknown", never as "average".
    status: str = "OK"

    @property
    def information_value(self) -> float:
        """How much this record changes a decision when it is present.

        A wallet with a strong record in either direction is worth resident
        memory; one that has never distinguished itself is not, however
        recently it traded. Rug exposure counts for as much as monster rate --
        knowing which wallets show up before collapses is as valuable as
        knowing which show up before runs.
        """
        if self.status != "OK" or self.launches_seen <= 0:
            return 0.0
        distinctiveness = max(self.monster_rate, self.rug_exposure, abs(self.pnl_quality))
        # Confidence grows with observations but saturates: the hundredth
        # launch tells you much less about a wallet than the tenth.
        confidence = 1.0 - math.exp(-self.launches_seen / 20.0)
        return float(distinctiveness * confidence)


@dataclass(frozen=True)
class CompactCreatorDNA:
    """The creator/funder equivalent, priced the same way."""

    creator: str
    launches: int = 0
    rug_rate: float = 0.0
    monster_rate: float = 0.0
    median_dump_delay_s: float = 0.0
    migration_rate: float = 0.0
    funder_quality: float = 0.0
    cluster_id: str = ""
    last_active: float = 0.0
    status: str = "OK"

    @property
    def information_value(self) -> float:
        if self.status != "OK" or self.launches <= 0:
            return 0.0
        distinctiveness = max(self.rug_rate, self.monster_rate)
        confidence = 1.0 - math.exp(-self.launches / 5.0)
        return float(distinctiveness * confidence)


class EconomicCache:
    """Bounded cache that evicts by expected value, not by recency alone.

    Plain LRU retires whatever has been quiet longest, which on this workload
    means retiring the serial deployers and known rugger clusters that go quiet
    between launches and are worth the most when they come back. Ranking by
    P(needed soon) * information value keeps recency as one input instead of
    the whole rule.
    """

    def __init__(self, capacity: int, half_life_seconds: float = 1_800.0,
                 pin_predicate: Optional[Callable[[Any], bool]] = None):
        self.capacity = max(1, capacity)
        self.half_life = max(1.0, half_life_seconds)
        # Entities that are always worth holding regardless of score, e.g. an
        # actor currently connected to a live narrative.
        self.pin_predicate = pin_predicate
        self._items: Dict[str, Any] = {}
        self._touched: Dict[str, float] = {}
        self.evictions = 0

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def put(self, key: str, value: Any, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        self._items[key] = value
        self._touched[key] = now
        if len(self._items) > self.capacity:
            self._evict(now)

    def get(self, key: str, now: Optional[float] = None) -> Optional[Any]:
        value = self._items.get(key)
        if value is not None:
            self._touched[key] = time.time() if now is None else now
        return value

    def peek(self, key: str) -> Optional[Any]:
        """Read without counting as use, so inspection cannot skew eviction."""
        return self._items.get(key)

    def _recency_weight(self, key: str, now: float) -> float:
        age = max(0.0, now - self._touched.get(key, now))
        return float(math.exp(-age * math.log(2.0) / self.half_life))

    def score(self, key: str, now: Optional[float] = None) -> float:
        now = time.time() if now is None else now
        value = self._items.get(key)
        if value is None:
            return 0.0
        if self.pin_predicate is not None and self.pin_predicate(value):
            return float("inf")
        information = float(getattr(value, "information_value", 0.0) or 0.0)
        # Recency alone never keeps a worthless record: a wallet that has told
        # us nothing over many launches is evictable no matter how recently it
        # traded. The small floor keeps brand-new records alive long enough to
        # be measured rather than evicted before they can earn a score.
        return (information + 1e-6) * self._recency_weight(key, now)

    def _evict(self, now: float) -> None:
        while len(self._items) > self.capacity:
            ranked = sorted(self._items, key=lambda key: self.score(key, now))
            victim = ranked[0]
            if self.score(victim, now) == float("inf"):
                # Everything resident is pinned. Growing past the cap is not an
                # option on a fixed-memory node, so the cap is enforced and the
                # condition is reported rather than silently exceeded.
                logger.error("economic cache at capacity with every entry pinned; "
                             "evicting the least recently used pinned entry")
                victim = min(self._items, key=lambda key: self._touched.get(key, 0.0))
            self._items.pop(victim, None)
            self._touched.pop(victim, None)
            self.evictions += 1

    def stats(self) -> Dict[str, Any]:
        return {"size": len(self._items), "capacity": self.capacity,
                "evictions": self.evictions}


@dataclass
class ArchiveStats:
    queued: int = 0
    written: int = 0
    dropped: int = 0
    bytes_written: int = 0
    quota_stops: int = 0


class AsyncArchiveWriter:
    """Bounded, non-blocking hand-off from the hot path to cold storage.

    The trading loop calls ``submit`` and returns. It never waits for a write,
    never retries, and never grows a queue without bound: a queue that blocks
    when storage is slow turns a disk problem into a missed launch, and one
    that grows without bound turns it into an OOM. Overflow drops the OLDEST
    record, because on this workload the newest observation is the one a
    decision might still depend on.

    Drops are counted rather than logged per event. Silent loss is what makes a
    research lake quietly wrong; a counter makes the gap measurable.
    """

    def __init__(self, root: Path, budget: HotStateBudget,
                 serializer: Optional[Callable[[Dict[str, Any]], bytes]] = None):
        self.root = Path(root)
        self.budget = budget
        self.serializer = serializer or (lambda record: (repr(record) + "\n").encode())
        self._queue: Deque[Dict[str, Any]] = deque()
        self._lock = threading.Lock()
        self.stats = ArchiveStats()

    def submit(self, record: Dict[str, Any]) -> bool:
        """Enqueue without blocking. False means the record was dropped."""
        with self._lock:
            self.stats.queued += 1
            if len(self._queue) >= self.budget.max_archive_queue:
                self._queue.popleft()
                self.stats.dropped += 1
                self._queue.append(record)
                return False
            self._queue.append(record)
            return True

    def _local_bytes(self) -> int:
        if not self.root.exists():
            return 0
        return sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file())

    def drain(self, limit: Optional[int] = None) -> int:
        """Write queued records. Called by a background worker, never the hot path."""
        with self._lock:
            batch = []
            while self._queue and (limit is None or len(batch) < limit):
                batch.append(self._queue.popleft())
        if not batch:
            return 0

        quota_bytes = int(self.budget.max_local_archive_gb * (1024 ** 3))
        if self._local_bytes() >= quota_bytes:
            # The local spool is a buffer, not the archive. Refusing to grow it
            # past its quota keeps a stalled upload from filling the disk the
            # trading process needs.
            self.stats.quota_stops += 1
            self.stats.dropped += len(batch)
            logger.error("local archive quota of %.1f GB reached; dropped %d records",
                         self.budget.max_local_archive_gb, len(batch))
            return 0

        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"events-{int(time.time())}.log"
        payload = b"".join(self.serializer(record) for record in batch)
        with path.open("ab") as handle:
            handle.write(payload)
        self.stats.written += len(batch)
        self.stats.bytes_written += len(payload)
        return len(batch)

    def report(self) -> Dict[str, Any]:
        return {**asdict(self.stats), "pending": len(self._queue),
                "drop_rate": (self.stats.dropped / self.stats.queued
                              if self.stats.queued else 0.0)}


class HotState:
    """The whole resident footprint of the trading node, under one budget."""

    def __init__(self, budget: Optional[HotStateBudget] = None,
                 archive_root: Optional[Path] = None,
                 pinned_clusters: Optional[Iterable[str]] = None):
        self.budget = budget or HotStateBudget()
        pinned = set(pinned_clusters or ())

        def pinned_predicate(record: Any) -> bool:
            return bool(pinned) and getattr(record, "cluster_id", "") in pinned

        self.wallets = EconomicCache(self.budget.max_hot_wallets,
                                     pin_predicate=pinned_predicate)
        self.creators = EconomicCache(self.budget.max_hot_creators,
                                      pin_predicate=pinned_predicate)
        self.active_tokens: Dict[str, float] = {}
        self.archive = AsyncArchiveWriter(
            archive_root or Path("data/spool"), self.budget)

    def touch_token(self, token: str, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        self.active_tokens[token] = now
        self.expire_tokens(now)
        if len(self.active_tokens) > self.budget.max_active_tokens:
            # Hard cap, not a target. A burst of irrelevant launches must not
            # be able to expand the footprint of a fixed-memory node.
            oldest = sorted(self.active_tokens.items(), key=lambda item: item[1])
            for stale, _ in oldest[: len(self.active_tokens) - self.budget.max_active_tokens]:
                self.active_tokens.pop(stale, None)

    def expire_tokens(self, now: Optional[float] = None) -> int:
        now = time.time() if now is None else now
        cutoff = now - self.budget.max_event_age_seconds
        stale = [token for token, seen in self.active_tokens.items() if seen < cutoff]
        for token in stale:
            self.active_tokens.pop(token, None)
        return len(stale)

    def report(self) -> Dict[str, Any]:
        return {
            "budget": self.budget.report(),
            "active_tokens": len(self.active_tokens),
            "wallets": self.wallets.stats(),
            "creators": self.creators.stats(),
            "archive": self.archive.report(),
        }

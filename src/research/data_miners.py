"""Market and chain data mined continuously, each source on its own clock.

The desk already reads the chain: the program stream carries every Pump and
PumpSwap event as it happens, and that is the fastest and most trustworthy
data it will ever have. What it does not carry is CONTEXT -- what a token's
metadata says, how concentrated its holders are, whether anything outside our
own stream has noticed it, what the wider market is doing while this launch
happens. That context is what turns a price path into an explicable one, and
it is what a forward ledger needs to answer "why did this work" rather than
only "did it".

So this is a pool of miners, not one loop. Each declares:

* its own CADENCE, because a token list that changes hourly and a holder
  distribution that changes every block do not belong on one clock, and
  putting them on one either hammers the slow source until it blocks us or
  reads the fast one too late to matter;
* what it ENRICHES, so a miner whose output nothing consumes is visible as
  exactly that rather than as diligence;
* whether it needs a CREDENTIAL, by name, so a dark miner is distinguishable
  from a broken one.

Three rules hold throughout, and they are the same rules the source mesh
follows because they are the same problem.

**Every record carries provenance.** Which miner, which endpoint, when. A
number in the lake whose origin cannot be recovered is a number that cannot be
trusted later, and "later" is when the model is being trained on it.

**A miner that cannot answer contributes nothing and says so.** Never a zero,
never a stale value re-emitted as fresh. An unmeasured holder concentration is
not a safe one, and the direction that error runs is the expensive one.

**Rate limits are respected per miner.** A 429 backs off that miner alone.
One aggressive source must not take the pool down with it, and a pool that
retries into a limit is a pool that gets its IP blocked and then reports every
source as dead.
"""

from __future__ import annotations

import asyncio
from src.runtime.loop_local import loop_local_semaphore
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DATA_MINER_SCHEMA_VERSION = "v1"

# Cadence floors. A miner may declare slower; it may not declare faster than
# the class allows, because the fastest useful cadence is a property of what
# is being measured and not of how much we would like to know it.
CADENCE_REALTIME = 1.0
CADENCE_FAST = 15.0
CADENCE_MINUTE = 60.0
CADENCE_QUARTER = 900.0
CADENCE_HOURLY = 3_600.0
CADENCE_DAILY = 86_400.0

# How long a miner sits out after being rate limited, doubling each time. A
# pool that retries straight into a limit gets its address blocked and then
# reports every source as dead, which is the same outcome as not having them.
BACKOFF_BASE_S = 60.0
BACKOFF_MAX_S = 3_600.0


class Enriches(Enum):
    """What a miner's output is FOR. A miner enriching nothing is dead weight."""

    TOKEN_METADATA = "token_metadata"
    HOLDER_STRUCTURE = "holder_structure"
    MARKET_CONTEXT = "market_context"
    VENUE_LIQUIDITY = "venue_liquidity"
    WALLET_HISTORY = "wallet_history"
    NARRATIVE = "narrative"
    # What it costs and how likely it is to land RIGHT NOW. Distinct from
    # market context: the market can be calm while the chain is congested, and
    # a bid sized for the wrong one of those is a bid that misses or overpays.
    EXECUTION_CONDITIONS = "execution_conditions"
    # Whether the supply we priced is the supply that will exist: mint
    # authority, LP burn, vesting. A rug is usually a supply event first.
    SUPPLY_CONTROL = "supply_control"
    # Measured public attention, as opposed to a source having mentioned it.
    # A mention is a touch; attention is how many people went looking.
    SOCIAL_ATTENTION = "social_attention"


@dataclass
class MinerSpec:
    """One data source, declared rather than coded into a loop."""

    miner_id: str
    enriches: Enriches
    cadence_seconds: float
    # Named for the report; never fetched by the pool itself, which holds no
    # transport. The miner's own callable owns how it talks to this.
    endpoint: str = ""
    requires_env: Tuple[str, ...] = ()
    detail: str = ""
    # Records per pass beyond which the result is truncated. A source that
    # suddenly returns fifty thousand rows is backfilling or broken, and
    # handing all of them downstream is worse than dropping the tail.
    max_records: int = 500

    def missing_credentials(self) -> List[str]:
        """Which required variables are absent. Presence only; never a value."""
        return [name for name in self.requires_env if not os.getenv(name)]


@dataclass
class MinerResult:
    """One pass of one miner."""

    miner_id: str
    status: str
    records: List[Dict[str, Any]] = field(default_factory=list)
    fetched_at: float = 0.0
    latency_ms: int = 0
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "OK"


@dataclass
class MinerHealth:
    """What this miner has actually done, as opposed to been configured to do."""

    miner_id: str
    passes: int = 0
    records: int = 0
    failures: int = 0
    rate_limited: int = 0
    last_ok_at: float = 0.0
    last_error: str = ""
    backoff_until: float = 0.0
    # The DURATION, kept separately from the deadline. Deriving the next
    # backoff from `backoff_until - now` reads negative once the deadline has
    # passed, which is exactly when the next limit arrives -- so it collapsed
    # to the base every time and a source that rate limited us repeatedly was
    # retried at a constant interval for ever.
    backoff_seconds: float = 0.0

    def to_dict(self, now: Optional[float] = None) -> Dict[str, Any]:
        now = time.time() if now is None else now
        return {
            "miner_id": self.miner_id, "passes": self.passes,
            "records": self.records, "failures": self.failures,
            "rate_limited": self.rate_limited,
            "seconds_since_ok": (round(now - self.last_ok_at, 1)
                                 if self.last_ok_at else None),
            "backing_off_for": (round(max(0.0, self.backoff_until - now), 1)
                                if self.backoff_until > now else None),
            "backoff_seconds": self.backoff_seconds or None,
            "last_error": self.last_error,
        }


class RateLimited(RuntimeError):
    """The source asked us to slow down. Distinct from a failure, on purpose."""


class DataMinerPool:
    """Runs every declared miner on its own cadence, concurrently and bounded.

    Owns no transport and no schema. A miner is a spec plus an async callable
    returning records; what those records mean belongs to whatever consumes
    them, and a pool that understood its payloads would be a pool that has to
    change every time a source does.
    """

    def __init__(self, *, concurrency: int = 6,
                 on_records: Optional[Callable[[str, List[Dict[str, Any]]], None]] = None):
        self.concurrency = max(1, int(concurrency))
        self.on_records = on_records
        self._specs: Dict[str, MinerSpec] = {}
        self._callables: Dict[str, Callable[[], Awaitable[List[Dict[str, Any]]]]] = {}
        self._health: Dict[str, MinerHealth] = {}
        self._next_due: Dict[str, float] = {}
        # Loop-local: this pool is CONSTRUCTED on the main loop and RUN on
        # the offload thread's loop, which is precisely the split that made
        # a plain semaphore leak a waiter per call until the box died.
        self._semaphore = loop_local_semaphore(self.concurrency, "miners")
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self.total_records = 0

    # --- registration ----------------------------------------------------

    def register(self, spec: MinerSpec,
                 fetch: Callable[[], Awaitable[List[Dict[str, Any]]]]) -> bool:
        """Add a miner. Refuses one whose credentials are absent, by name."""
        missing = spec.missing_credentials()
        if missing:
            self._health[spec.miner_id] = MinerHealth(
                miner_id=spec.miner_id,
                last_error=f"missing credentials: {', '.join(missing)}")
            self._specs[spec.miner_id] = spec
            return False
        self._specs[spec.miner_id] = spec
        self._callables[spec.miner_id] = fetch
        self._health[spec.miner_id] = MinerHealth(miner_id=spec.miner_id)
        # Staggered rather than all due at once: forty miners firing in the
        # same tick is a thundering herd against forty hosts and a latency
        # spike on ours.
        self._next_due[spec.miner_id] = time.time() + (
            len(self._next_due) % max(1, int(spec.cadence_seconds)))
        return True

    # --- lifecycle -------------------------------------------------------

    async def start(self) -> int:
        self._running = True
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())
        return len(self._callables)

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def _loop(self) -> None:
        while self._running:
            try:
                await self.run_due()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("data miner pool error: %s", exc)
            # The tick is short and the cadences are long: what decides how
            # often a miner runs is its own declaration, not this sleep.
            await asyncio.sleep(1.0)

    async def run_due(self, now: Optional[float] = None) -> List[MinerResult]:
        """Run every miner whose cadence has come round. Concurrent, bounded."""
        now = time.time() if now is None else now
        due = [miner_id for miner_id, at in self._next_due.items()
               if at <= now and miner_id in self._callables
               and self._health[miner_id].backoff_until <= now]
        if not due:
            return []
        results = await asyncio.gather(
            *(self._run_one(miner_id, now) for miner_id in due),
            return_exceptions=True)
        out: List[MinerResult] = []
        for miner_id, result in zip(due, results):
            if isinstance(result, Exception):
                logger.debug("miner %s raised: %s", miner_id, result)
                continue
            out.append(result)
        return out

    async def _run_one(self, miner_id: str, now: float) -> MinerResult:
        spec = self._specs[miner_id]
        health = self._health[miner_id]
        self._next_due[miner_id] = now + spec.cadence_seconds
        started = time.time()
        async with self._semaphore:
            try:
                records = await self._callables[miner_id]()
            except RateLimited as exc:
                health.rate_limited += 1
                health.last_error = f"rate limited: {exc}"
                # Doubling, per miner. One aggressive source must not take the
                # pool down with it.
                health.backoff_seconds = min(
                    BACKOFF_MAX_S,
                    (health.backoff_seconds * 2) if health.backoff_seconds
                    else BACKOFF_BASE_S)
                health.backoff_until = now + health.backoff_seconds
                return MinerResult(miner_id=miner_id, status="RATE_LIMITED",
                                   fetched_at=now, detail=str(exc))
            except Exception as exc:
                health.failures += 1
                health.last_error = f"{type(exc).__name__}: {exc}"
                return MinerResult(miner_id=miner_id, status="DATA_BLOCKED",
                                   fetched_at=now, detail=health.last_error)

        kept = list(records or ())[:spec.max_records]
        stamped = [self._stamp(spec, record, now) for record in kept]
        health.passes += 1
        health.records += len(stamped)
        health.last_ok_at = now
        health.last_error = ""
        # A pass that worked clears the penalty. Keeping it would punish a
        # source for a limit it has already recovered from.
        health.backoff_until = 0.0
        health.backoff_seconds = 0.0
        self.total_records += len(stamped)
        if stamped and self.on_records is not None:
            try:
                self.on_records(miner_id, stamped)
            except Exception as exc:
                logger.warning("miner %s consumer raised: %s", miner_id, exc)
        return MinerResult(
            miner_id=miner_id, status="OK", records=stamped, fetched_at=now,
            latency_ms=int((time.time() - started) * 1000),
            detail=f"{len(stamped)} record(s)")

    @staticmethod
    def _stamp(spec: MinerSpec, record: Dict[str, Any], now: float) -> Dict[str, Any]:
        """Provenance on every record.

        A number in the lake whose origin cannot be recovered is a number that
        cannot be trusted later -- and later is when a model is trained on it.
        """
        return {
            **dict(record),
            "_miner": spec.miner_id,
            "_enriches": spec.enriches.value,
            "_endpoint": spec.endpoint,
            "_fetched_at": now,
        }

    # --- reporting -------------------------------------------------------

    def report(self, now: Optional[float] = None) -> Dict[str, Any]:
        """What is mining, what is dark, and why.

        A miner that has never returned a record is reported separately from
        one that is failing: the first is usually a wrong endpoint or an empty
        universe, the second is usually the network, and they need different
        fixes.
        """
        now = time.time() if now is None else now
        rows = []
        for miner_id, health in self._health.items():
            row = health.to_dict(now)
            spec = self._specs[miner_id]
            if miner_id not in self._callables:
                state = "DATA_BLOCKED"
            elif row["backing_off_for"] is not None:
                state = "RATE_LIMITED"
            elif row["failures"] > 0 and not row["passes"]:
                state = "ERROR"
            elif row["records"] > 0:
                state = "PRODUCING"
            elif row["passes"] > 0:
                state = "IDLE"
            else:
                state = "IDLE"
            row.update({
                "state": state,
                "cadence_seconds": spec.cadence_seconds,
                "enriches": spec.enriches.value,
                "endpoint": spec.endpoint,
                "last_success_seconds_ago": row["seconds_since_ok"],
                "records_total": row["records"],
            })
            rows.append(row)
        producing = [row for row in rows if row["records"] > 0]
        silent = [row for row in rows
                  if row["records"] == 0 and row["failures"] == 0
                  and row["last_error"] and "missing credentials" not in row["last_error"]]
        keyless = [row for row in rows
                   if "missing credentials" in (row["last_error"] or "")]
        by_enrichment: Dict[str, int] = {}
        for spec in self._specs.values():
            if self._health[spec.miner_id].records > 0:
                by_enrichment[spec.enriches.value] = (
                    by_enrichment.get(spec.enriches.value, 0) + 1)
        return {
            "schema": DATA_MINER_SCHEMA_VERSION,
            "status": "OK" if producing else "DATA_BLOCKED",
            "detail": ("" if producing else
                       "no miner has returned a record yet; the lake is being "
                       "fed by the chain stream alone"),
            "registered": len(self._specs),
            "runnable": len(self._callables),
            "producing": len(producing),
            "failing": len([row for row in rows if row["failures"] > 0]),
            "rate_limited": len([row for row in rows if row["rate_limited"] > 0]),
            "awaiting_credentials": [row["miner_id"] for row in keyless],
            "silent": [row["miner_id"] for row in silent],
            "total_records": self.total_records,
            "enrichments_covered": dict(sorted(by_enrichment.items())),
            "cadences": {spec.miner_id: spec.cadence_seconds
                         for spec in sorted(self._specs.values(),
                                            key=lambda item: item.miner_id)},
            "miners": sorted(rows, key=lambda row: row["miner_id"]),
        }

"""Continuously refreshed blockhash state.

Signing needs a recent blockhash. Fetching one per transaction put an RPC
round trip -- tens of milliseconds on a good day, and unbounded on a bad one --
inside the exact window the whole system is optimised for, and it did it at the
worst possible moment: after the decision, while the opportunity ages.

Caching it naively is worse than the round trip, because a stale blockhash is
a transaction the cluster silently refuses, and that looks exactly like a
transaction that lost a race. So this does not merely cache. It refreshes in
the background, and it REFUSES to serve a hash it cannot vouch for:

* older than ``max_age_s`` -- a hash we have not confirmed recently enough;
* within ``min_slots_remaining`` of its ``lastValidBlockHeight`` -- a hash the
  cluster is about to stop accepting, which is the case that produces a
  transaction rejected for a reason nobody attributes correctly.

A refusal is a DATA_BLOCKED state, and the caller falls back to a synchronous
fetch. Slow is a cost; silently unlanded is a loss.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

BLOCKHASH_SCHEMA_VERSION = "v1"

# A blockhash is accepted for roughly 150 slots. Refreshing every two seconds
# keeps the served hash a couple of seconds old at worst, which is far inside
# that window, and costs one cheap RPC call per interval rather than one per
# transaction -- and unlike the per-transaction call, none of them is in front
# of a decision.
DEFAULT_REFRESH_INTERVAL_S = 2.0

# Beyond this the cached hash is not trusted even if the height arithmetic
# says it should still be valid. The two checks fail in different ways -- a
# stalled refresher versus a hash near expiry -- and one is not a substitute
# for the other.
DEFAULT_MAX_AGE_S = 20.0

# Slots of validity that must remain. A transaction built now still has to be
# signed, submitted, and included; handing it a hash with four slots left is
# handing it a rejection.
DEFAULT_MIN_SLOTS_REMAINING = 30

# The cluster's blockhash validity window, used to infer the current block
# height from a fresh response's lastValidBlockHeight. Inferred rather than
# fetched, because a second RPC call to protect the first one from staleness
# reintroduces exactly the round trip this exists to remove.
BLOCKHASH_VALIDITY_SLOTS = 150


@dataclass(frozen=True)
class BlockhashState:
    """What the cache can currently vouch for, and why not when it cannot."""

    status: str
    blockhash: str = ""
    last_valid_block_height: int = 0
    fetched_at: float = 0.0
    slots_remaining: Optional[int] = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "OK"

    def age_s(self, now: Optional[float] = None) -> float:
        if self.fetched_at <= 0:
            return float("inf")
        return max(0.0, (now if now is not None else time.time()) - self.fetched_at)

    def to_dict(self, now: Optional[float] = None) -> Dict[str, Any]:
        return {"status": self.status, "blockhash": self.blockhash,
                "last_valid_block_height": self.last_valid_block_height,
                "age_s": round(self.age_s(now), 3),
                "slots_remaining": self.slots_remaining, "detail": self.detail}


class BlockhashCache:
    """Background-refreshed blockhash with an expiry gate.

    The refresher owns the RPC calls; readers only ever touch local state, so
    ``current()`` is a field read and cannot block, time out, or raise.
    """

    def __init__(self, rpc: Any, *,
                 refresh_interval_s: float = DEFAULT_REFRESH_INTERVAL_S,
                 max_age_s: float = DEFAULT_MAX_AGE_S,
                 min_slots_remaining: int = DEFAULT_MIN_SLOTS_REMAINING):
        self.rpc = rpc
        self.refresh_interval_s = max(0.1, float(refresh_interval_s))
        self.max_age_s = float(max_age_s)
        self.min_slots_remaining = int(min_slots_remaining)
        self._blockhash = ""
        self._last_valid_block_height = 0
        self._fetched_at = 0.0
        self._task: Optional[asyncio.Task] = None
        self._running = False
        # Counters, so a cache that has silently stopped refreshing is visible
        # rather than merely slow.
        self.refreshes = 0
        self.failures = 0
        self.served = 0
        self.refused = 0
        self.last_error = ""

    async def start(self) -> bool:
        """Fetch once, then keep it fresh. Returns whether the first fetch landed.

        The first fetch is awaited rather than left to the loop: starting a
        desk whose first trade pays the round trip anyway would defeat the
        point of having the cache at all.
        """
        self._running = True
        first = await self.refresh()
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())
        return first

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
                await asyncio.sleep(self.refresh_interval_s)
                await self.refresh()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("blockhash refresh loop error: %s", exc)

    async def refresh(self) -> bool:
        """One fetch. Returns whether the state was updated.

        A failed refresh leaves the PREVIOUS state in place rather than
        clearing it: a hash from four seconds ago is still good, and throwing
        it away on one failed call would turn a transient RPC hiccup into a
        hot-path round trip on every subsequent trade.
        """
        try:
            response = await self.rpc.request(
                "getLatestBlockhash", [{"commitment": "confirmed"}])
        except Exception as exc:
            self.failures += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False
        value = ((response or {}).get("value") or {})
        blockhash = str(value.get("blockhash") or "")
        if not blockhash:
            self.failures += 1
            self.last_error = "response carried no blockhash"
            return False
        self._blockhash = blockhash
        self._last_valid_block_height = int(value.get("lastValidBlockHeight") or 0)
        self._fetched_at = time.time()
        self.refreshes += 1
        self.last_error = ""
        return True

    def current(self, now: Optional[float] = None) -> BlockhashState:
        """The cached hash, or a DATA_BLOCKED state saying why it is not usable.

        Never blocks and never raises: this is called from the signing path,
        and a hot-path accessor that can await is the thing being removed.
        """
        now = time.time() if now is None else now
        if not self._blockhash:
            self.refused += 1
            return BlockhashState(status="DATA_BLOCKED",
                                  detail="no blockhash fetched yet")
        age = max(0.0, now - self._fetched_at)
        if age > self.max_age_s:
            self.refused += 1
            return BlockhashState(
                status="DATA_BLOCKED", blockhash=self._blockhash,
                last_valid_block_height=self._last_valid_block_height,
                fetched_at=self._fetched_at,
                detail=f"cached blockhash is {age:.1f}s old "
                       f"(max {self.max_age_s:.0f}s); refresher may have stalled")
        # Height at the moment of the fetch, inferred from the validity window
        # the cluster publishes. Slots elapse at roughly 400ms, so the age of
        # the cache converts directly into slots consumed.
        elapsed_slots = int(age / 0.4)
        remaining = BLOCKHASH_VALIDITY_SLOTS - elapsed_slots
        if remaining < self.min_slots_remaining:
            self.refused += 1
            return BlockhashState(
                status="DATA_BLOCKED", blockhash=self._blockhash,
                last_valid_block_height=self._last_valid_block_height,
                fetched_at=self._fetched_at, slots_remaining=remaining,
                detail=f"only {remaining} slots of validity remain "
                       f"(need {self.min_slots_remaining})")
        self.served += 1
        return BlockhashState(
            status="OK", blockhash=self._blockhash,
            last_valid_block_height=self._last_valid_block_height,
            fetched_at=self._fetched_at, slots_remaining=remaining)

    def report(self, now: Optional[float] = None) -> Dict[str, Any]:
        state = BlockhashState(
            status="OK" if self._blockhash else "DATA_BLOCKED",
            blockhash=self._blockhash,
            last_valid_block_height=self._last_valid_block_height,
            fetched_at=self._fetched_at)
        total = self.served + self.refused
        return {
            "schema": BLOCKHASH_SCHEMA_VERSION,
            "status": "OK" if self._blockhash and not self.last_error else "DATA_BLOCKED",
            "running": self._running,
            "refresh_interval_s": self.refresh_interval_s,
            "refreshes": self.refreshes, "failures": self.failures,
            "served_from_cache": self.served, "refused": self.refused,
            # The number that says whether this is actually doing its job. A
            # cache serving 40% of requests is a hot path still paying for the
            # other 60%.
            "cache_hit_rate": (self.served / total) if total else None,
            "last_error": self.last_error,
            "state": state.to_dict(now),
        }

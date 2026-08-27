"""Keeping slow, bulky work off the loop that has to decide in milliseconds.

The desk runs everything on one asyncio loop: the chain stream, the decision
path, the health server, the dashboard, and twenty-nine miners. Under the GIL
that is one thread, and asyncio gives no pre-emption -- a coroutine holds the
loop until it awaits. So when the Binance ticker miner calls `json.loads` on a
multi-megabyte response, the decision path is not slow, it is STOPPED, for
however long that parse takes. Nothing in the design says that is fine; it
simply was not measured, which is how a latency problem hides for months.

The expensive part of a miner is not the await on the socket -- that yields
properly -- it is the synchronous CPU after it: JSON parsing, HTML parsing,
building thousands of dicts. That work belongs on a different thread.

So miners run on their own event loop in their own thread, and hand results
back through a plain queue that a small drainer coroutine empties on the main
loop. Two properties follow, and both matter:

**Every mutation of desk state still happens on the main loop.** The miner
thread never touches the census, the models, or the position book. It parses,
it enqueues, it goes back to sleep. A design where miners wrote to desk state
from another thread would be faster and would eventually corrupt something
during a burst, which is a bad trade at any latency.

**Backpressure is bounded and visible.** The queue has a ceiling. When the
main loop cannot drain as fast as the miners produce, the OLDEST records are
dropped and counted, rather than the queue growing until the box runs out of
memory -- which is what happened to this node once already. A dropped record
is reported; an unbounded queue is not.

The GIL still exists, so this is not true parallelism. It is pre-emption: the
interpreter switches threads every few milliseconds, so a long parse becomes a
series of interruptible slices instead of one uninterruptible block. That
converts a multi-millisecond stall on the decision path into scheduler jitter,
which is exactly the trade worth making here.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

OFFLOAD_SCHEMA_VERSION = "v1"

#: Records queued between the miner thread and the main loop. Sized for a
#: burst of several miners finishing at once, not for an outage: if the main
#: loop stops draining, dropping is the correct behaviour and growing is not.
DEFAULT_QUEUE_DEPTH = 4_096


class OffloadedPool:
    """Runs a DataMinerPool on its own loop in its own thread.

    Owns no schema and no desk state. It takes a pool, starts it somewhere
    else, and hands whatever it produces back through a queue.
    """

    def __init__(self, pool: Any, *, sink: Callable[[str, List[Dict[str, Any]]], None],
                 queue_depth: int = DEFAULT_QUEUE_DEPTH,
                 drain_interval_s: float = 0.05,
                 name: str = "miners"):
        self.pool = pool
        self.sink = sink
        self.name = name
        self.drain_interval_s = float(drain_interval_s)
        self._queue: "queue.Queue[Tuple[str, List[Dict[str, Any]]]]" = queue.Queue(
            maxsize=int(queue_depth))
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._drainer: Optional[asyncio.Task] = None
        self._running = False
        self.enqueued = 0
        self.delivered = 0
        self.dropped = 0
        self.sink_errors = 0
        self.started_at = 0.0
        self.last_delivery_at = 0.0
        self.thread_error = ""

    # --- the miner side --------------------------------------------------

    def _publish(self, miner_id: str, records: List[Dict[str, Any]]) -> None:
        """Called ON THE MINER THREAD. Enqueues; never touches desk state."""
        try:
            self._queue.put_nowait((miner_id, records))
            self.enqueued += 1
        except queue.Full:
            # Oldest out, newest in. A full queue means the main loop is
            # behind, and in that state the freshest observation is the one
            # worth keeping -- a stale market reading helps nobody.
            try:
                self._queue.get_nowait()
                self.dropped += 1
                self._queue.put_nowait((miner_id, records))
                self.enqueued += 1
            except (queue.Empty, queue.Full):
                self.dropped += 1

    def _run_thread(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.pool.start())
            loop.run_forever()
        except Exception as exc:  # pragma: no cover - thread teardown
            self.thread_error = f"{type(exc).__name__}: {exc}"
            logger.error("offloaded pool %s died: %s", self.name, exc)
        finally:
            try:
                loop.run_until_complete(self.pool.stop())
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass

    # --- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self.started_at = time.time()
        # Rebound here rather than at construction so a caller that builds the
        # pool normally and offloads it later cannot leave the original
        # callback wired to the main loop.
        self.pool.on_records = self._publish
        self._thread = threading.Thread(target=self._run_thread, name=f"offload-{self.name}",
                                        daemon=True)
        self._thread.start()
        self._drainer = asyncio.create_task(self._drain_loop())

    async def stop(self) -> None:
        self._running = False
        if self._drainer is not None:
            self._drainer.cancel()
            try:
                await self._drainer
            except (asyncio.CancelledError, Exception):
                pass
            self._drainer = None
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    # --- the main-loop side ----------------------------------------------

    async def _drain_loop(self) -> None:
        while self._running:
            try:
                self.drain()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("offload drain error: %s", exc)
            await asyncio.sleep(self.drain_interval_s)

    def drain(self, budget: int = 64) -> int:
        """Move up to `budget` batches onto the main loop's state.

        Bounded per pass on purpose. Draining the whole queue in one go turns
        a miner burst into exactly the stall this module exists to prevent,
        just moved to a different line of code.
        """
        moved = 0
        while moved < budget:
            try:
                miner_id, records = self._queue.get_nowait()
            except queue.Empty:
                break
            moved += 1
            self.delivered += 1
            self.last_delivery_at = time.time()
            try:
                self.sink(miner_id, records)
            except Exception as exc:
                self.sink_errors += 1
                logger.warning("offload sink raised for %s: %s", miner_id, exc)
        return moved

    # --- reporting -------------------------------------------------------

    def report(self, now: Optional[float] = None) -> Dict[str, Any]:
        """Whether offloading is working, and whether it is losing anything.

        `dropped` is the line to watch. A non-zero drop count means the main
        loop could not keep up with the miners, which is a real loss of data
        and is reported as one rather than absorbed silently.
        """
        moment = time.time() if now is None else now
        alive = bool(self._thread is not None and self._thread.is_alive())
        depth = self._queue.qsize()
        if not self._running:
            status, detail = "OFF", "miners are running on the main loop"
        elif not alive:
            status, detail = "CRITICAL", (
                self.thread_error or "the miner thread is not alive; miners are dark")
        elif self.dropped:
            status, detail = "DEGRADED", (
                f"{self.dropped} batch(es) dropped; the main loop is not "
                "draining as fast as the miners produce")
        else:
            status, detail = "OK", ""
        return {
            "schema": OFFLOAD_SCHEMA_VERSION,
            "status": status, "detail": detail,
            "thread_alive": alive,
            "queue_depth": depth,
            "queue_capacity": self._queue.maxsize,
            "enqueued": self.enqueued,
            "delivered": self.delivered,
            "dropped": self.dropped,
            "sink_errors": self.sink_errors,
            "seconds_since_delivery": (round(moment - self.last_delivery_at, 1)
                                       if self.last_delivery_at else None),
        }


def install_fast_event_loop() -> str:
    """Swap CPython's selector loop for uvloop where it is available.

    uvloop is libuv underneath and is materially faster at exactly what this
    desk does most: many concurrent sockets, many small reads, timers. It is a
    drop-in policy change with no API difference, and it is the cheapest
    latency improvement available to any asyncio program.

    Returns what actually happened, as a string for the status page. Never
    raises: a desk that will not start because an optional accelerator is
    missing is worse than a slightly slower desk.
    """
    try:
        import uvloop
    except ImportError:
        return ("DEGRADED: uvloop not installed; running on CPython's selector "
                "loop. `pip install uvloop` is the cheapest latency win here")
    try:
        uvloop.install()
        return f"OK: uvloop {getattr(uvloop, '__version__', 'unknown')}"
    except Exception as exc:  # pragma: no cover - platform specific
        return f"DEGRADED: uvloop present but refused to install: {exc}"

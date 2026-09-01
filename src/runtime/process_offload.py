"""Miners in another PROCESS, because a thread still holds the GIL.

`OffloadedPool` moved the miners onto their own event loop in their own
thread, and its docstring is honest about the limit: the GIL still exists,
so this is pre-emption rather than parallelism. A long `json.loads` on a
multi-megabyte response becomes a series of interruptible slices instead of
one uninterruptible block -- better, and still competing for the same
interpreter lock as the decision path.

A separate process does not compete for that lock at all. The desk's hot
loop cannot be delayed by a miner parsing a large document, because the
miner is not in this interpreter.

The cost is real and worth stating: records must be pickled, sent through a
pipe and unpickled, which is far more expensive per record than handing a
dict to a callback. That trade is correct HERE and only here, because these
are research observations arriving a few times a second, not decisions
arriving in a burst. Nothing on the money path crosses this boundary.

What crosses is deliberately narrow: (miner_id, records) upward, and
nothing downward. The child cannot touch desk state -- not because it is
forbidden to, but because it has no reference to any, which is a much
stronger guarantee than a convention. The same isolation that removes the
GIL contention removes the shared-mutable-state class of bug with it.

The child is built from a FACTORY named by import path, not by pickling a
constructed pool: the pool holds aiohttp sessions, event-loop-bound
primitives and closures, none of which survive a fork/spawn intact, and
several of which would be silently broken rather than loudly unpicklable.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import multiprocessing as mp
import os
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

PROCESS_OFFLOAD_SCHEMA_VERSION = "v1"

#: Records buffered between the child and this process.
DEFAULT_QUEUE_DEPTH = 4_096

#: How long to wait for the child to exit on its own before killing it.
GRACEFUL_STOP_S = 5.0

#: Backoff between restarts of a child that keeps dying. Without this the
#: supervisor respawns on every drain tick -- fifty milliseconds apart -- and
#: a child that fails at import produces a traceback flood that buries the one
#: line saying why. Measured while building this: 443KB of identical
#: tracebacks in under thirty seconds.
RESTART_BACKOFF_S = (1.0, 2.0, 5.0, 15.0, 30.0, 60.0)

#: After this many consecutive failures with no successful record, the child
#: is left down and reported CRITICAL. A miner pool that cannot start will not
#: start on the hundredth try either, and the desk keeps running without it --
#: which is the whole point of it being a separate process.


def _child_main(factory_path: str, config: Dict[str, Any],
                out: "mp.Queue", stop: "mp.Event",
                affinity: Sequence[int]) -> None:  # pragma: no cover - child
    """Entry point in the child. Builds its own pool and runs it.

    Nothing from the parent's memory is used here beyond the plain
    dictionaries passed in, which is the property that makes the isolation
    real rather than nominal.
    """
    try:
        if affinity and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(int(cpu) for cpu in affinity))
    except (OSError, ValueError):
        pass
    # Background work should yield to the desk, never the other way round.
    try:
        os.nice(5)
    except (OSError, AttributeError):
        pass

    module_name, _, attribute = factory_path.rpartition(".")
    factory = getattr(importlib.import_module(module_name), attribute)
    pool = factory(config)

    def publish(miner_id: str, records: List[Dict[str, Any]]) -> None:
        try:
            out.put_nowait((miner_id, records))
        except Exception:
            # Full or closed. Dropping here is correct: the parent is behind
            # and the freshest observation is the one worth keeping.
            pass

    pool.on_records = publish

    async def run() -> None:
        await pool.start()
        try:
            while not stop.is_set():
                await asyncio.sleep(0.2)
        finally:
            await pool.stop()

    try:
        from src.runtime.offload import install_fast_event_loop

        install_fast_event_loop()
    except Exception:
        pass
    try:
        asyncio.run(run())
    except Exception as exc:
        try:
            out.put_nowait(("__error__", [{"error": f"{type(exc).__name__}: {exc}"}]))
        except Exception:
            pass


class ProcessOffloadedPool:
    """Runs a miner pool in a child process; delivers records on this loop.

    Interface-compatible with OffloadedPool on purpose, so which one is in
    use is a configuration decision rather than a code path.
    """

    def __init__(self, factory_path: str, config: Dict[str, Any], *,
                 sink: Callable[[str, List[Dict[str, Any]]], None],
                 queue_depth: int = DEFAULT_QUEUE_DEPTH,
                 drain_interval_s: float = 0.05,
                 affinity: Sequence[int] = (),
                 name: str = "miners"):
        self.factory_path = factory_path
        self.config = dict(config)
        self.sink = sink
        self.name = name
        self.drain_interval_s = float(drain_interval_s)
        self.queue_depth = int(queue_depth)
        self.affinity = tuple(int(cpu) for cpu in affinity)
        self._ctx = mp.get_context("spawn")
        self._queue: Optional[Any] = None
        self._stop: Optional[Any] = None
        self._process: Optional[Any] = None
        self._reader: Optional[threading.Thread] = None
        self._local: "queue.Queue" = queue.Queue(maxsize=self.queue_depth)
        self._drainer: Optional[asyncio.Task] = None
        self._running = False
        self.enqueued = 0
        self.delivered = 0
        # Per-CHILD, not per-pool. `delivered` is cumulative for the life of
        # the process, so forgiving failures on it meant a generation that
        # once worked absolved every generation after it: a child dying at
        # import was respawned for ever, because the reset ran on any tick
        # where the newest child happened to still be alive.
        self.delivered_this_generation = 0
        # Bumped on every spawn. A reader thread checks it and exits when its
        # own generation is superseded -- without this the old thread stayed
        # in its loop, resolved `self._queue` afresh each pass, and started
        # competing with the new reader for the NEW child's records, leaking
        # one thread per restart.
        self._generation = 0
        self.dropped = 0
        self.sink_errors = 0
        self.restarts = 0
        self.consecutive_failures = 0
        self._next_restart_at = 0.0
        self.gave_up = False
        self.started_at = 0.0
        self.last_delivery_at = 0.0
        self.child_error = ""

    # --- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self.started_at = time.time()
        self._spawn()
        self._drainer = asyncio.create_task(self._drain_loop())

    def _spawn(self) -> None:
        self._generation += 1
        generation = self._generation
        self.delivered_this_generation = 0
        child_queue = self._ctx.Queue(maxsize=self.queue_depth)
        self._queue = child_queue
        self._stop = self._ctx.Event()
        self._process = self._ctx.Process(
            target=_child_main,
            args=(self.factory_path, self.config, child_queue, self._stop,
                  self.affinity),
            name=f"offload-{self.name}", daemon=True)
        self._process.start()
        # The queue is passed to the thread rather than read off `self`, so a
        # reader is bound for life to the child it was started for.
        self._reader = threading.Thread(
            target=self._read_forever, args=(generation, child_queue),
            name=f"offload-reader-{self.name}-{generation}", daemon=True)
        self._reader.start()
        logger.info("PROCESS OFFLOAD %s running as pid %s%s", self.name,
                    self._process.pid,
                    f" pinned to CPU {list(self.affinity)}" if self.affinity else "")

    def _read_forever(self, generation: int, child_queue: Any) -> None:
        """Moves records off ONE child's pipe. Its own thread, because
        mp.Queue blocks.

        A blocking read on the main loop would reintroduce exactly the stall
        this class exists to remove, so the block happens on a thread that
        does nothing else and holds the GIL only long enough to hand the
        record on.

        Bound to its generation and to its own queue object. Reading
        `self._queue` each pass meant a reader outlived its child and then
        started taking records from the child that replaced it, so two
        threads split one stream and one thread leaked per restart.
        """
        while self._running and generation == self._generation:
            try:
                item = child_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            except (OSError, ValueError, EOFError):
                return
            if item is None:
                return
            miner_id, records = item
            if miner_id == "__error__":
                self.child_error = str((records or [{}])[0].get("error", ""))
                logger.error("PROCESS OFFLOAD %s child failed: %s",
                             self.name, self.child_error)
                continue
            self.enqueued += 1
            try:
                self._local.put_nowait(item)
            except queue.Full:
                try:
                    self._local.get_nowait()
                    self.dropped += 1
                    self._local.put_nowait(item)
                except (queue.Empty, queue.Full):
                    self.dropped += 1

    async def _drain_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.drain_interval_s)
            self.drain()
            self._supervise()

    def drain(self, budget: int = 256) -> int:
        """Deliver queued records on THIS loop. Bounded per pass.

        Bounded because an unbounded drain hands the loop to the sink for as
        long as the backlog takes, which is the stall this class exists to
        prevent -- just relocated.
        """
        delivered = 0
        while delivered < budget:
            try:
                miner_id, records = self._local.get_nowait()
            except queue.Empty:
                break
            try:
                self.sink(miner_id, records)
                self.delivered += 1
                self.delivered_this_generation += 1
                self.last_delivery_at = time.time()
            except Exception as exc:
                self.sink_errors += 1
                logger.warning("offload sink error for %s: %s", miner_id, exc)
            delivered += 1
        return delivered

    def _supervise(self) -> None:
        """Bring the child back if it died, with backoff. Never the desk.

        Backoff and a give-up point, because the two failure shapes need
        different answers. A child killed by the OOM killer or a transient
        fault should come straight back. A child that dies at import will die
        at import every time, and respawning it on every drain tick buries
        the reason under thousands of identical tracebacks.

        Giving up leaves the desk running without these miners, which is
        exactly what a separate process is for.
        """
        if not self._running or self._process is None or self.gave_up:
            return
        if self._process.is_alive():
            # THIS child having produced records is what makes it healthy.
            # Checking the pool's lifetime total instead let a long-lived
            # first generation absolve every crash after it, so the give-up
            # point was unreachable and a child dying at import was respawned
            # for ever.
            if self.delivered_this_generation:
                self.consecutive_failures = 0
            return
        now = time.time()
        if now < self._next_restart_at:
            return
        self.consecutive_failures += 1
        if self.consecutive_failures > len(RESTART_BACKOFF_S):
            self.gave_up = True
            logger.error(
                "PROCESS OFFLOAD %s failed %d times; leaving it down. The "
                "desk continues without these miners -- which is what a "
                "separate process is for. Last error: %s",
                self.name, self.consecutive_failures,
                self.child_error or f"exit code {self._process.exitcode}")
            return
        delay = RESTART_BACKOFF_S[min(self.consecutive_failures - 1,
                                      len(RESTART_BACKOFF_S) - 1)]
        self._next_restart_at = now + delay
        self.restarts += 1
        logger.error("PROCESS OFFLOAD %s child exited (code %s); restart %d "
                     "in %.0fs", self.name, self._process.exitcode,
                     self.restarts, delay)
        try:
            self._spawn()
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("PROCESS OFFLOAD %s could not restart: %s", self.name, exc)

    async def stop(self) -> None:
        self._running = False
        # Retires every reader, including any that has not noticed yet.
        self._generation += 1
        if self._drainer is not None:
            self._drainer.cancel()
            try:
                await self._drainer
            except asyncio.CancelledError:
                pass
            self._drainer = None
        if self._stop is not None:
            try:
                self._stop.set()
            except Exception:  # pragma: no cover - teardown
                pass
        if self._process is not None:
            self._process.join(timeout=GRACEFUL_STOP_S)
            if self._process.is_alive():
                logger.warning("PROCESS OFFLOAD %s did not exit; terminating",
                               self.name)
                self._process.terminate()
                self._process.join(timeout=GRACEFUL_STOP_S)
            self._process = None
        if self._queue is not None:
            try:
                self._queue.close()
            except Exception:  # pragma: no cover - teardown
                pass
            self._queue = None

    # --- reporting -------------------------------------------------------

    @property
    def alive(self) -> bool:
        return bool(self._process is not None and self._process.is_alive())

    def report(self) -> Dict[str, Any]:
        return {
            "schema": PROCESS_OFFLOAD_SCHEMA_VERSION,
            "status": ("OK" if self.alive else
                       "STOPPED" if not self._running else
                       "CRITICAL"),
            "isolation": "process",
            "detail": ("miners run in their own interpreter, so a large parse "
                       "cannot hold the GIL against the decision path"),
            "pid": getattr(self._process, "pid", None),
            "affinity": list(self.affinity),
            "alive": self.alive,
            "restarts": self.restarts,
            "consecutive_failures": self.consecutive_failures,
            "gave_up": self.gave_up,
            "queue_depth": self._local.qsize(),
            "queue_capacity": self.queue_depth,
            "enqueued": self.enqueued,
            "delivered": self.delivered,
            "delivered_this_generation": self.delivered_this_generation,
            "generation": self._generation,
            "dropped": self.dropped,
            "sink_errors": self.sink_errors,
            "child_error": self.child_error,
            "seconds_since_delivery": (
                round(time.time() - self.last_delivery_at, 2)
                if self.last_delivery_at else None),
        }

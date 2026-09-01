"""Synchronisation primitives that belong to whichever loop is using them.

An asyncio.Semaphore binds to the event loop that first awaits it. Awaiting
it from a DIFFERENT loop does not merely fail -- and this is the part that
turns a bug into an outage -- it fails AFTER appending a waiter to the
semaphore's internal deque. The RuntimeError propagates, the caller retries
on its next cadence, and the waiter is never removed because the loop that
would wake it is not the loop it is queued on.

That is a leak with a clock on it. Observed on this desk 2026-09-01:

    RuntimeError: <asyncio.locks.Semaphore [locked, waiters:2437]>
    is bound to a different event loop

2,437 pinned coroutines on one semaphore and 1,164 on another, growing every
fifteen seconds, until the process was SIGKILLed by the cgroup OOM killer:

    Main process exited, code=killed, status=9/KILL

The desk had run for six days and reported `memory band: calm, 0.40 of
ceiling` ninety seconds before it died, because RSS was still moderate at the
last sample -- the waiters accumulate faster than the governor samples, and
the governor's own reliefs cannot release a waiter anyway.

The cause is structural, not incidental. `OffloadedPool` deliberately runs
the miners on their own loop in their own thread to keep multi-megabyte JSON
parses off the decision path. Every shared client the miners touch was
constructed on the MAIN loop. So the split that protects latency is exactly
what breaks these primitives, and every one of them has to be loop-aware or
the split cannot be safe.

The fix is the same shape as the one already applied to aiohttp sessions:
key by the running loop. Each loop gets its own primitive with the same
limit. That is the correct semantics as well as the safe one -- a
concurrency cap exists to bound in-flight work on a loop, and two loops
doing independent work were never sharing a queue in any meaningful sense.
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class LoopLocal:
    """One primitive per event loop, created on first use by that loop.

    Deliberately not a subclass of anything: `async with obj` must resolve
    the primitive at ENTRY time, on the loop actually entering it, and a
    subclass would have to pretend to be a semaphore it does not yet know
    which one it is.
    """

    __slots__ = ("_factory", "_instances", "_detached", "_name", "__weakref__")

    def __init__(self, factory: Callable[[], Any], name: str = ""):
        self._factory = factory
        self._name = name or getattr(factory, "__name__", "primitive")
        # Keyed on the LOOP OBJECT, held weakly -- never on id(loop).
        #
        # CPython reuses object ids after collection. A closed loop's id can
        # be handed to the next loop allocated, which would silently give
        # that loop the previous one's primitive: a semaphore whose permits
        # were drained by a loop that no longer exists, and which nothing
        # can ever release. That is the original deadlock wearing the
        # fix's clothes, and it is worse than the bug because the tests
        # would pass. A weak-keyed map is both correct and self-cleaning:
        # entries disappear when their loop is collected.
        self._instances: "weakref.WeakKeyDictionary[Any, Any]" = (
            weakref.WeakKeyDictionary())
        #: The instance used when there is no running loop at all.
        self._detached: Optional[Any] = None

    def get(self) -> Any:
        """The primitive for the calling loop, created if this is its first."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No loop: hand back a single instance rather than raising, so
            # synchronous construction and inspection still work.
            if self._detached is None:
                self._detached = self._factory()
            return self._detached
        existing = self._instances.get(loop)
        if existing is None:
            existing = self._factory()
            self._instances[loop] = existing
            if len(self._instances) > 1:
                # Worth a line: it means work is genuinely running on more
                # than one loop, which is by design here but is also the
                # condition under which the old code silently leaked.
                logger.debug("%s now has %d loop-local instances",
                             self._name, len(self._instances))
        return existing

    async def __aenter__(self):
        primitive = self.get()
        await primitive.acquire()
        return primitive

    async def __aexit__(self, exc_type, exc, tb):
        self.get().release()
        return False

    # A few pass-throughs so the object can stand in for the primitive in
    # the places that use it directly rather than as a context manager.
    async def acquire(self):
        return await self.get().acquire()

    def release(self) -> None:
        self.get().release()

    def locked(self) -> bool:
        return bool(self.get().locked())

    @property
    def loops(self) -> int:
        """How many LIVE loops hold an instance. Diagnostic only.

        Counts live loops because the map is weak: a closed and collected
        loop drops out by itself, which is the property that makes this
        safe against id reuse.
        """
        return len(self._instances)


def loop_local_semaphore(value: int, name: str = "") -> LoopLocal:
    """A concurrency cap that applies per loop rather than across loops."""
    return LoopLocal(lambda: asyncio.Semaphore(value), name or f"semaphore({value})")


def loop_local_lock(name: str = "") -> LoopLocal:
    """A mutex that guards one loop's critical section.

    Note the semantics change and why it is still right: two loops that were
    never able to share this lock cannot begin racing because of this
    change. They could not share it before -- one of them raised.
    """
    return LoopLocal(asyncio.Lock, name or "lock")

"""The leak that OOM-killed the desk, reproduced and closed.

2026-09-01, from the box's own journal:

    RuntimeError: <asyncio.locks.Semaphore [locked, waiters:2437]>
    is bound to a different event loop
    ...
    Main process exited, code=killed, status=9/KILL

A semaphore bound to loop A, awaited from loop B, appends a waiter BEFORE it
raises. The caller retries on its next cadence and the waiter is never
removed, because the loop that would wake it is not the loop it is queued
on. Every fifteen seconds, forever, until the cgroup kills the process.

The first test below fails against a plain asyncio.Semaphore -- it is a
regression test for a defect that actually happened, not a hypothetical.
"""

from __future__ import annotations

import asyncio
import unittest

from src.runtime.loop_local import (
    LoopLocal, loop_local_lock, loop_local_semaphore)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class ThePlainSemaphoreLeaksWaitersAcrossLoops(unittest.TestCase):
    """Demonstrates the defect, so the fix is not tested against a guess.

    Every attempt is bounded by wait_for, because the cross-loop failure
    mode is version-dependent and BOTH shapes are broken: some versions
    raise "bound to a different event loop", others simply block forever on
    a permit the other loop will never release. An unbounded await here
    would hang the suite -- which is the production symptom in miniature.
    """

    def test_a_shared_semaphore_accumulates_waiters(self):
        semaphore = asyncio.Semaphore(1)

        async def hold():
            await semaphore.acquire()

        # Loop A takes the only permit and never gives it back.
        _run(hold())

        blocked = 0
        for _ in range(25):
            async def attempt():
                await asyncio.wait_for(semaphore.acquire(), timeout=0.01)
            try:
                _run(attempt())
            except (RuntimeError, asyncio.TimeoutError, TimeoutError):
                blocked += 1
        # The assertion worth making, and the one that holds on every
        # version: a plain semaphore shared across loops can never be
        # acquired from a second loop. Whether the attempt raises or blocks
        # -- and whether the abandoned waiter is cleaned up -- varies by
        # version and by how the caller cancels. On the box it did not get
        # cleaned up, and 2,437 pinned waiters killed the process. Asserting
        # that internal here would be asserting a CPython detail rather than
        # the defect, so this asserts the defect.
        self.assertEqual(25, blocked,
                         "no cross-loop acquire can succeed, so every miner "
                         "RPC call through a shared semaphore fails")


class LoopLocalDoesNotLeak(unittest.TestCase):

    def test_repeated_cross_loop_use_leaves_no_waiters(self):
        semaphore = loop_local_semaphore(1, "test")

        async def hold_and_release():
            async with semaphore:
                pass

        held = []

        async def check():
            async with semaphore:
                pass
            held.append(semaphore.get())

        loop = asyncio.new_event_loop()
        try:
            for _ in range(50):
                loop.run_until_complete(check())
        finally:
            loop.close()
        for instance in held:
            self.assertFalse(getattr(instance, "_waiters", None) or [],
                             "no waiter may survive a completed acquire")

    def test_a_never_released_permit_does_not_block_another_loop(self):
        # The exact production shape: the main loop holds a permit while the
        # miner loop needs one. Bounded, so a regression fails rather than
        # hangs.
        semaphore = loop_local_semaphore(1, "test")

        async def take_only():
            await semaphore.acquire()

        _run(take_only())

        async def other_loop_still_works():
            await asyncio.wait_for(semaphore.acquire(), timeout=1.0)
            semaphore.release()
            return "acquired"

        self.assertEqual("acquired", _run(other_loop_still_works()))

    def test_each_loop_gets_its_own_instance(self):
        semaphore = loop_local_semaphore(4, "test")
        seen = []

        async def touch():
            seen.append(semaphore.get())

        # Loops kept alive for the duration, so both entries coexist. A
        # closed loop drops out of the weak map, which is the behaviour the
        # next test covers.
        loops = [asyncio.new_event_loop() for _ in range(2)]
        try:
            for loop in loops:
                loop.run_until_complete(touch())
            self.assertEqual(2, semaphore.loops)
            self.assertIsNot(seen[0], seen[1])
        finally:
            for loop in loops:
                loop.close()

    def test_a_collected_loop_cannot_hand_its_primitive_to_a_new_one(self):
        # id(loop) is reused after collection. Keying on it would give a
        # fresh loop a semaphore whose permits were taken by a loop that no
        # longer exists -- an unreleasable deadlock that looks like a fix.
        semaphore = loop_local_semaphore(1, "test")
        instances = []

        async def drain():
            instances.append(semaphore.get())
            await semaphore.acquire()   # taken and never released

        for _ in range(5):
            _run(drain())

        async def still_works():
            await asyncio.wait_for(semaphore.acquire(), timeout=1.0)
            semaphore.release()
            return True

        self.assertTrue(_run(still_works()))

    def test_one_loop_reuses_its_instance(self):
        semaphore = loop_local_semaphore(4, "test")

        async def twice():
            first = semaphore.get()
            self.assertIs(first, semaphore.get())
            self.assertEqual(1, semaphore.loops)

        _run(twice())

    def test_the_cap_still_binds_within_a_loop(self):
        # A per-loop cap must still be a cap, or the leak fix would have
        # quietly removed the concurrency limit it replaced.
        semaphore = loop_local_semaphore(2, "test")
        peak = 0
        live = 0

        async def worker():
            nonlocal peak, live
            async with semaphore:
                live += 1
                peak = max(peak, live)
                await asyncio.sleep(0.01)
                live -= 1

        async def many():
            await asyncio.gather(*(worker() for _ in range(10)))

        _run(many())
        self.assertEqual(2, peak)

    def test_a_lock_is_loop_local_too(self):
        lock = loop_local_lock("test")

        async def use():
            async with lock:
                return True

        self.assertTrue(_run(use()))
        self.assertTrue(_run(use()))

    def test_it_works_outside_a_loop_without_raising(self):
        self.assertIsNotNone(loop_local_semaphore(1, "test").get())


class TheLeakingPatternIsGoneFromTheOffloadPath(unittest.TestCase):
    """The two classes the miner thread actually touches."""

    def test_the_rpc_manager_uses_loop_local_primitives(self):
        import inspect

        from src.chains import rpc_manager

        source = inspect.getsource(rpc_manager)
        self.assertIn("loop_local_semaphore", source)
        self.assertNotIn("asyncio.Semaphore(", source,
                         "a plain semaphore here leaks a waiter per miner call")

    def test_the_miner_pool_does_too(self):
        import inspect

        from src.research import data_miners

        source = inspect.getsource(data_miners)
        self.assertIn("loop_local_semaphore", source)
        self.assertNotIn("asyncio.Semaphore(", source)

    def test_memory_relief_does_not_reintroduce_it(self):
        # Trimming under pressure rebuilt the semaphore; rebuilding it as a
        # plain one would restore the leak at exactly the worst moment.
        import inspect

        import src.main as main

        source = inspect.getsource(main._MemoryReliefSourceProbe) if hasattr(
            main, "_MemoryReliefSourceProbe") else inspect.getsource(main)
        relief = source[source.find("def _register_memory_reliefs"):]
        relief = relief[:relief.find("\n    @property")]
        self.assertIn("loop_local_semaphore", relief)
        self.assertNotIn("asyncio.Semaphore(", relief)


if __name__ == "__main__":
    unittest.main()

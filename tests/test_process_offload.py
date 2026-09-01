"""Miners in another interpreter, so the GIL cannot reach the decision path.

The thread-based pool converts a long parse into interruptible slices; it
does not stop that parse competing for the interpreter lock. A separate
process does. These assert the isolation is real -- a different pid -- and
that the failure modes a child process introduces are handled.
"""

from __future__ import annotations

import asyncio
import os
import unittest

from src.runtime.process_offload import (
    RESTART_BACKOFF_S, ProcessOffloadedPool)

FACTORY = "tests.support_offload_factory.build_ticking_pool"


def _drive(pool, seconds: float, records: list):
    async def main():
        await pool.start()
        deadline = asyncio.get_running_loop().time() + seconds
        while asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.05)
            pool.drain()
            if records:
                break
        await pool.stop()

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()


class TheMinersRunInADifferentInterpreter(unittest.TestCase):

    def test_records_arrive_and_carry_a_different_pid(self):
        records = []
        pool = ProcessOffloadedPool(
            FACTORY, {"interval_s": 0.01, "label": "hello"},
            sink=lambda miner_id, rows: records.extend(rows), name="test")
        _drive(pool, 20.0, records)
        self.assertTrue(records, "the child produced nothing")
        self.assertNotEqual(os.getpid(), records[0]["pid"],
                            "a thread would share this pid; a process must not")
        self.assertEqual("hello", records[0]["label"],
                         "config must reach the child")

    def test_the_report_says_which_isolation_is_in_force(self):
        pool = ProcessOffloadedPool(FACTORY, {}, sink=lambda *_: None, name="t")
        report = pool.report()
        self.assertEqual("process", report["isolation"])
        self.assertIn("GIL", report["detail"])


class ChildFailuresAreHandled(unittest.TestCase):

    def test_a_child_that_cannot_build_its_pool_is_reported_not_silent(self):
        pool = ProcessOffloadedPool(
            "tests.support_offload_factory.build_exploding_pool", {},
            sink=lambda *_: None, name="boom")
        errors = []

        async def main():
            await pool.start()
            for _ in range(60):
                await asyncio.sleep(0.1)
                pool.drain()
                if pool.child_error:
                    errors.append(pool.child_error)
                    break
            await pool.stop()

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(main())
        finally:
            loop.close()
        self.assertTrue(errors, "a child that dies at startup must say so")
        self.assertIn("could not build", errors[0])

    def test_a_child_that_dies_is_restarted(self):
        # Mining is worth restarting; it is never worth the desk.
        pool = ProcessOffloadedPool(
            "tests.support_offload_factory.build_dying_pool", {},
            sink=lambda *_: None, name="dying")

        async def main():
            await pool.start()
            for _ in range(80):
                await asyncio.sleep(0.1)
                pool.drain()
                pool._supervise()
                if pool.restarts:
                    break
            await pool.stop()

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(main())
        finally:
            loop.close()
        self.assertGreaterEqual(pool.restarts, 1)


class BackpressureIsBoundedAndVisible(unittest.TestCase):

    def test_a_full_queue_drops_oldest_and_counts_it(self):
        pool = ProcessOffloadedPool(FACTORY, {}, sink=lambda *_: None,
                                    queue_depth=4, name="small")
        pool._running = True
        for index in range(20):
            try:
                pool._local.put_nowait(("m", [{"i": index}]))
            except Exception:
                pool._local.get_nowait()
                pool.dropped += 1
                pool._local.put_nowait(("m", [{"i": index}]))
        self.assertLessEqual(pool._local.qsize(), 4)
        self.assertGreater(pool.dropped, 0)

    def test_draining_is_bounded_per_pass(self):
        delivered = []
        pool = ProcessOffloadedPool(FACTORY, {},
                                    sink=lambda m, r: delivered.append(r),
                                    queue_depth=1024, name="budget")
        for index in range(100):
            pool._local.put_nowait(("m", [{"i": index}]))
        self.assertEqual(10, pool.drain(budget=10))
        self.assertEqual(10, len(delivered))


if __name__ == "__main__":
    unittest.main()


class ADyingChildIsNotRespawnedInATightLoop(unittest.TestCase):
    """A child that fails at import fails at import every time.

    Found while verifying this class: the supervisor respawned on every drain
    tick -- fifty milliseconds apart -- and a child dying at import produced
    443KB of identical tracebacks in under thirty seconds, burying the one
    line that said why.
    """

    def _dead_pool(self):
        pool = ProcessOffloadedPool(FACTORY, {}, sink=lambda *_: None, name="d")
        pool._running = True

        class _Corpse:
            exitcode = 1

            def is_alive(self):
                return False

        pool._process = _Corpse()
        pool._spawn = lambda: None      # do not actually fork in a unit test
        return pool

    def test_restarts_back_off_instead_of_firing_every_tick(self):
        pool = self._dead_pool()
        pool._supervise()
        self.assertEqual(1, pool.restarts)
        # Immediately again: the backoff must swallow it.
        for _ in range(50):
            pool._supervise()
        self.assertEqual(1, pool.restarts,
                         "a second restart inside the backoff window is the "
                         "traceback flood this guards against")

    def test_it_eventually_stops_trying_and_says_so(self):
        pool = self._dead_pool()
        for _ in range(len(RESTART_BACKOFF_S) + 2):
            pool._next_restart_at = 0.0     # pretend the backoff elapsed
            pool._supervise()
        self.assertTrue(pool.gave_up)
        self.assertEqual("CRITICAL", pool.report()["status"])

    def test_giving_up_never_stops_the_desk(self):
        # The whole point of a separate process: losing it costs research,
        # not the decision path.
        pool = self._dead_pool()
        pool.gave_up = True
        pool._supervise()               # must not raise
        self.assertEqual(0, pool.restarts)

    def test_a_child_that_delivers_records_is_forgiven_its_past(self):
        pool = ProcessOffloadedPool(FACTORY, {}, sink=lambda *_: None, name="ok")
        pool._running = True
        pool.consecutive_failures = 3
        pool.delivered_this_generation = 10

        class _Live:
            exitcode = None

            def is_alive(self):
                return True

        pool._process = _Live()
        pool._supervise()
        self.assertEqual(0, pool.consecutive_failures)

    def test_an_earlier_generation_does_not_absolve_the_current_one(self):
        """The pool's lifetime total is not evidence about THIS child.

        Forgiving on `delivered` meant a first generation that ran for an
        hour absolved every crash after it: the reset fired on any tick where
        the newest child was still alive, `consecutive_failures` never
        climbed, and the give-up point was unreachable. A child dying at
        import was then respawned for ever, burying the reason under
        identical tracebacks -- which is the failure this backoff exists to
        prevent, reintroduced by the forgiveness rule.
        """
        pool = ProcessOffloadedPool(FACTORY, {}, sink=lambda *_: None, name="ok")
        pool._running = True
        pool.consecutive_failures = 3
        pool.delivered = 10_000          # an earlier generation was healthy
        pool.delivered_this_generation = 0

        class _Live:
            exitcode = None

            def is_alive(self):
                return True

        pool._process = _Live()
        pool._supervise()
        self.assertEqual(3, pool.consecutive_failures)

    def test_a_reader_retires_when_its_child_is_replaced(self):
        """A reader is bound to the child it was started for.

        Reading `self._queue` afresh each pass meant the old thread outlived
        its child and then competed with the new reader for the NEW child's
        records -- one leaked thread per restart, and one stream split
        between two consumers.
        """
        import queue as queue_module
        import threading

        pool = ProcessOffloadedPool(FACTORY, {}, sink=lambda *_: None, name="ok")
        pool._running = True
        pool._generation = 1
        old_queue = queue_module.Queue()
        thread = threading.Thread(
            target=pool._read_forever, args=(1, old_queue), daemon=True)
        thread.start()
        # The child is replaced.
        pool._generation = 2
        thread.join(timeout=3.0)
        self.assertFalse(thread.is_alive(),
                         "the reader for a retired child never exited")

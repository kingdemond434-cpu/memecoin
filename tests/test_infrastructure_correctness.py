"""Four bookkeeping faults, each of which broke the thing it was written for.

None of these were features that failed to land. Each was a correct design
whose accounting quietly did the opposite: a duplicate guard that let
duplicates through, a distinctness rule the code never applied, a failure
counter that could not count to its own give-up point, and a per-loop cache
keyed on something that is reused.
"""

from __future__ import annotations

import asyncio
import gc
import unittest


class TheDuplicateGuardActuallyGuards(unittest.TestCase):
    """A bounded deque evicts its own leftmost item on append, silently.

    `_delivered` was a `deque(maxlen=N)` beside a `set`. Once at capacity
    every append dropped the oldest key from the deque WITHOUT removing it
    from the set, and the explicit eviction that followed then popped a
    newer key from both. So the set grew without bound while simultaneously
    losing keys it was still meant to protect -- and a key it had forgotten
    could be delivered a second time, which is the one property the whole
    class exists to provide.
    """

    def test_a_key_is_delivered_once_however_full_the_history_gets(self):
        from src.runtime.feed_race import FeedRace

        race = FeedRace(history=4)
        self.assertTrue(race.observe("a", "first"))
        # Push far past capacity so any silent eviction has happened.
        for index in range(200):
            race.observe("a", f"filler-{index}")
        # The original key must still be known, or racing is unsafe.
        self.assertFalse(race.observe("b", "first"),
                         "a key the race had already delivered came back as new")

    def test_the_history_stays_bounded(self):
        from src.runtime.feed_race import FeedRace

        race = FeedRace(history=4)
        for index in range(500):
            race.observe("a", f"k-{index}")
        self.assertLessEqual(len(race._delivered), race._delivered_capacity)

    def test_the_first_feed_still_wins_and_the_rest_are_measured(self):
        from src.runtime.feed_race import FeedRace

        race = FeedRace(history=8)
        self.assertTrue(race.observe("fast", "sig", at=1.0))
        self.assertFalse(race.observe("slow", "sig", at=1.05))
        self.assertEqual(1, race.duplicates)


class VerificationCountsDistinctTransactions(unittest.TestCase):
    """The docstring said distinct signatures; the code just incremented.

    One transaction redelivered by three racing feeds verified a launchpad
    on its own -- exactly the single lucky byte alignment the observation
    count exists to rule out.
    """

    def _registry(self):
        from src.chains.launchpads import LaunchpadRegistry
        return LaunchpadRegistry()

    def _event(self, registry, signature):
        from src.chains.launchpads import CanonicalLaunchEvent

        program = next(pid for pid, spec in registry.specs.items()
                       if not spec.trusted)
        return program, CanonicalLaunchEvent(
            venue=registry.specs[program].name, program_id=program,
            instruction="create", mint="Mint1", creator="Creator1",
            signature=signature, slot=1, observed_at=0.0)

    def test_the_same_transaction_three_times_verifies_nothing(self):
        from src.chains.launchpads import OBSERVATIONS_TO_VERIFY

        registry = self._registry()
        program, event = self._event(registry, "SameSignature")
        for _ in range(OBSERVATIONS_TO_VERIFY + 5):
            registry.observe(event)
        spec = registry.specs[program]
        self.assertEqual(1, spec.observations)
        self.assertFalse(spec.trusted)
        self.assertGreater(spec.duplicate_observations, 0)

    def test_distinct_transactions_do_verify(self):
        from src.chains.launchpads import OBSERVATIONS_TO_VERIFY

        registry = self._registry()
        program, _ = self._event(registry, "x")
        verified = False
        for index in range(OBSERVATIONS_TO_VERIFY):
            _, event = self._event(registry, f"Signature{index}")
            verified = registry.observe(event) or verified
        self.assertTrue(verified)
        self.assertTrue(registry.specs[program].trusted)

    def test_an_event_with_no_signature_is_not_an_observation(self):
        registry = self._registry()
        program, event = self._event(registry, "")
        self.assertFalse(registry.observe(event))
        self.assertEqual(0, registry.specs[program].observations)

    def test_the_signature_set_does_not_grow_without_bound(self):
        from src.chains.launchpads import OBSERVATIONS_TO_VERIFY

        registry = self._registry()
        # A venue whose decodes are always malformed never verifies, so its
        # spec would otherwise hold every signature the chain ever produced.
        program = next(pid for pid, spec in registry.specs.items()
                       if not spec.trusted)
        spec = registry.specs[program]
        spec.status = "UNVERIFIED"
        for index in range(5_000):
            _, event = self._event(registry, f"S{index}")
            registry.observe(event)
            spec.status = "UNVERIFIED"      # keep it unverified for the test
        self.assertLessEqual(len(spec.seen_signatures),
                             OBSERVATIONS_TO_VERIFY * 64 + 1)


class APerLoopCacheIsKeyedOnTheLoopNotItsAddress(unittest.TestCase):
    """CPython reuses ids. A new loop must not inherit a dead loop's session.

    The failure this prevents is not theoretical and not new: the same
    mistake was made in the HttpClient fix and caught by the loop-local
    semaphore's own test. Its symptom is an intermittent "Timeout context
    manager should be used inside a task" that nothing in the logs explains,
    because by then the loop it belonged to no longer exists.
    """

    def _manager(self):
        from src.chains.rpc_manager import RPCManager

        class _Config:
            name = "solana"
            rpc_endpoints = ["https://example.invalid"]

        return RPCManager(_Config())

    def test_two_loops_get_two_sessions(self):
        manager = self._manager()
        sessions = []

        async def take():
            sessions.append(manager._session_for_loop())

        for _ in range(2):
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(take())
            finally:
                loop.run_until_complete(asyncio.sleep(0))
                loop.close()
        self.assertIsNot(sessions[0], sessions[1])
        for session in sessions:
            asyncio.new_event_loop().run_until_complete(session.close())

    def test_a_collected_loop_leaves_nothing_behind_to_inherit(self):
        manager = self._manager()
        first = []

        async def take(into):
            into.append(manager._session_for_loop())

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(take(first))
            # Closed on its own loop, so nothing holds it open. A live
            # session keeps a reference to its loop, and a loop that is still
            # referenced is a loop whose entry is legitimately still there.
            loop.run_until_complete(first[0].close())
        finally:
            loop.close()
        del loop
        gc.collect()
        # The next loop to ask sweeps the closed one out. Weak keys alone
        # cannot do this: an aiohttp session references its own loop, so the
        # entry's value pins its key and nothing is ever collected.
        second = []
        live = asyncio.new_event_loop()
        try:
            live.run_until_complete(take(second))
            self.assertEqual(1, len(manager._sessions))
            # And what it got is its OWN session, not the dead loop's.
            self.assertIsNot(first[0], second[0])
            live.run_until_complete(second[0].close())
        finally:
            live.close()


if __name__ == "__main__":
    unittest.main()

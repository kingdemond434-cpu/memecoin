"""Who produces the next slot, which nothing in this desk has ever known.

`landing_model` has carried a `leader` field since it was written, keeps
per-leader accept counts, and documents leader-specific landing curves as
the thing it exists to learn. Nothing populated it. Every attempt the desk
recorded said `leader: ""`, so the per-leader counts were one bucket with
everything in it, and "which validators actually take our transactions" --
a question with a real, stable answer available for two free RPC calls --
had never been asked.
"""

from __future__ import annotations

import unittest

from src.execution.leader_schedule import (
    NODES_TTL_S, REFRESH_BELOW_SLOTS, LeaderNode, LeaderSchedule)


class _Rpc:
    def __init__(self, leaders=None, nodes=None, slot=1000, fail=()):
        self.leaders = leaders
        self.nodes = nodes
        self.slot = slot
        self.fail = set(fail)
        self.calls = []

    async def request(self, method, params):
        self.calls.append(method)
        if method in self.fail:
            raise RuntimeError(f"{method} unavailable")
        if method == "getSlot":
            return self.slot
        if method == "getSlotLeaders":
            return self.leaders
        if method == "getClusterNodes":
            return self.nodes
        return None


def _nodes(*identities, quic=True):
    return [{"pubkey": identity, "tpu": f"{identity}:8003",
             "tpuQuic": f"{identity}:8009" if quic else ""}
            for identity in identities]


class ItAnswersFromMemoryNotFromTheNetwork(unittest.IsolatedAsyncioTestCase):

    async def _ready(self, **kwargs):
        # Four consecutive slots per leader, which is how Solana schedules.
        leaders = []
        for identity in ("V1", "V2", "V3"):
            leaders.extend([identity] * 4)
        rpc = _Rpc(leaders=leaders, nodes=_nodes("V1", "V2", "V3"), **kwargs)
        schedule = LeaderSchedule(rpc, lookahead=12)
        self.assertTrue(await schedule.refresh())
        return schedule, rpc

    async def test_a_lookup_costs_no_request(self):
        schedule, rpc = await self._ready()
        before = len(rpc.calls)
        self.assertEqual("V1", schedule.leader_for(1000))
        self.assertEqual("V2", schedule.leader_for(1004))
        self.assertEqual("V3", schedule.leader_for(1008))
        self.assertEqual(before, len(rpc.calls))

    async def test_a_slot_outside_the_window_is_empty_not_a_guess(self):
        # A wrong leader puts one validator's accept rate into another's
        # bucket, and the landing model never unlearns it.
        schedule, _ = await self._ready()
        self.assertEqual("", schedule.leader_for(99_999))
        self.assertGreater(schedule.report()["lookup_miss_rate"], 0.0)

    async def test_the_node_for_a_slot_carries_its_addresses(self):
        schedule, _ = await self._ready()
        node = schedule.node_for(1004)
        self.assertEqual("V2", node.identity)
        self.assertEqual("V2:8009", node.tpu_quic)
        self.assertTrue(node.reachable)


class PrewarmingTargetsDistinctLeaders(unittest.IsolatedAsyncioTestCase):

    async def test_the_next_leaders_are_distinct_not_the_next_slots(self):
        # Solana gives each leader four consecutive slots, so the next eight
        # slots are usually two validators repeated. Warming a connection
        # twice to the same address buys nothing.
        leaders = []
        for identity in ("V1", "V2", "V3"):
            leaders.extend([identity] * 4)
        schedule = LeaderSchedule(_Rpc(leaders, _nodes("V1", "V2", "V3")),
                                  lookahead=12, prewarm_leaders=2)
        await schedule.refresh()
        upcoming = schedule.upcoming_leaders()
        self.assertEqual(["V1", "V2"], [node.identity for node in upcoming])

    async def test_a_leader_that_advertises_no_tpu_is_skipped(self):
        # Some validators deliberately do not advertise one. A route that
        # pretended otherwise would fail at send time with a worse error.
        leaders = ["V1"] * 4 + ["V2"] * 4
        nodes = _nodes("V1", quic=False)
        nodes[0]["tpu"] = ""
        nodes += _nodes("V2")
        schedule = LeaderSchedule(_Rpc(leaders, nodes), lookahead=8,
                                  prewarm_leaders=2)
        await schedule.refresh()
        self.assertEqual(["V2"],
                         [node.identity for node in schedule.upcoming_leaders()])


class ItRefreshesOnNeedNotOnAClock(unittest.IsolatedAsyncioTestCase):

    async def test_a_fresh_schedule_does_not_refresh(self):
        schedule = LeaderSchedule(_Rpc(["V1"] * 400, _nodes("V1")),
                                  lookahead=400)
        await schedule.refresh()
        self.assertFalse(schedule.needs_refresh())

    async def test_a_consumed_lookahead_does(self):
        schedule = LeaderSchedule(_Rpc(["V1"] * 400, _nodes("V1")),
                                  lookahead=400)
        await schedule.refresh(slot=1000)
        schedule.observe_slot(1000 + 400 - REFRESH_BELOW_SLOTS + 1)
        self.assertTrue(schedule.needs_refresh())

    async def test_an_empty_schedule_always_needs_one(self):
        self.assertTrue(LeaderSchedule(_Rpc()).needs_refresh())

    async def test_the_window_is_replaced_not_merged(self):
        # A schedule mixing two fetches across an epoch boundary would answer
        # confidently and wrongly for the slots either side of it.
        rpc = _Rpc(["V1"] * 4, _nodes("V1"))
        schedule = LeaderSchedule(rpc, lookahead=4)
        await schedule.refresh(slot=1000)
        rpc.leaders = ["V2"] * 4
        await schedule.refresh(slot=2000)
        self.assertEqual("", schedule.leader_for(1000))
        self.assertEqual("V2", schedule.leader_for(2000))


class AFailedFetchDegradesRatherThanLies(unittest.IsolatedAsyncioTestCase):

    async def test_a_failing_slot_leaders_call_is_recorded_not_swallowed(self):
        schedule = LeaderSchedule(_Rpc(fail={"getSlotLeaders"}))
        self.assertFalse(await schedule.refresh(slot=1000))
        self.assertEqual(1, schedule.refresh_failures)
        self.assertIn("getSlotLeaders", schedule.last_error)
        self.assertEqual("DATA_BLOCKED", schedule.report()["status"])

    async def test_a_failing_cluster_nodes_call_keeps_the_schedule(self):
        # The schedule is still good; a leader whose address is unknown is
        # simply one the desk cannot reach directly. Degrading is correct.
        schedule = LeaderSchedule(_Rpc(["V1"] * 4, fail={"getClusterNodes"}),
                                  lookahead=4)
        self.assertTrue(await schedule.refresh(slot=1000))
        self.assertEqual("V1", schedule.leader_for(1000))
        self.assertIsNone(schedule.node_for(1000))
        self.assertEqual(0, schedule.refresh_failures)

    async def test_an_empty_answer_is_a_failure_not_an_empty_schedule(self):
        schedule = LeaderSchedule(_Rpc(leaders=[]))
        self.assertFalse(await schedule.refresh(slot=1000))
        self.assertEqual(1, schedule.refresh_failures)


class ItFollowsTheChainForFree(unittest.IsolatedAsyncioTestCase):

    async def test_observing_a_slot_only_moves_forward(self):
        schedule = LeaderSchedule(_Rpc())
        schedule.observe_slot(500)
        schedule.observe_slot(100)
        self.assertEqual(500, schedule.current_slot)

    async def test_the_report_names_what_it_would_prewarm(self):
        leaders = ["V1"] * 4 + ["V2"] * 4
        schedule = LeaderSchedule(_Rpc(leaders, _nodes("V1", "V2")),
                                  lookahead=8, prewarm_leaders=2)
        await schedule.refresh(slot=1000)
        report = schedule.report()
        self.assertEqual("OK", report["status"])
        self.assertEqual(["V1", "V2"], report["prewarm_targets"])
        self.assertEqual(2, report["nodes_with_tpu_quic"])


class TheLandingModelFinallyGetsALeader(unittest.TestCase):

    def test_a_fill_without_a_leader_is_resolved_from_the_schedule(self):
        from src.execution.jupiter_jito import ExecutionEngine

        engine = ExecutionEngine.__new__(ExecutionEngine)

        class _Schedule:
            def leader_for(self, slot):
                return "V7" if slot == 42 else ""

        engine.leader_schedule = _Schedule()
        self.assertEqual("V7", engine._leader_for_slot(42))
        self.assertEqual("", engine._leader_for_slot(43))

    def test_no_schedule_means_no_leader_rather_than_an_error(self):
        from src.execution.jupiter_jito import ExecutionEngine

        engine = ExecutionEngine.__new__(ExecutionEngine)
        engine.leader_schedule = None
        self.assertEqual("", engine._leader_for_slot(42))


if __name__ == "__main__":
    unittest.main()

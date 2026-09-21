"""Who produces the next slot, and where their TPU listens.

The landing model has carried a `leader` field since it was written, keeps
per-leader accept counts, and documents leader-specific landing curves as the
thing it exists to learn. Nothing has ever populated that field. Every
attempt the desk has recorded says `leader: ""`, so the per-leader counts are
one bucket with every attempt in it, and "which validators actually take our
transactions" -- a question with a real and stable answer, and one of the few
edges available without spending anything -- has never been asked.

Two RPC calls answer it, both free and both cacheable for a long time:

    getSlotLeaders(start, limit)   slot -> validator identity
    getClusterNodes()             identity -> gossip, tpu, tpu_quic addresses

The schedule is fixed for an entire epoch, which is roughly two days, so this
is not a hot-path fetch dressed as one: it is a background refresh whose
answer is then a dictionary lookup at the moment a transaction is built.

What this deliberately does NOT do is send anything. A direct TPU submission
needs a QUIC client speaking Solana's ALPN with a certificate keyed to the
sender's identity, and it cannot be exercised without a live validator and a
funded identity to sign for. Shipping that unproven -- into the money path,
on a desk whose whole discipline is that unmeasured is not measured -- would
be the same mistake as every DATA_BLOCKED this codebase refuses to paper
over. So the addresses are discovered, prewarming is decided, the route is
declared, and it stays disabled with a stated reason until something can
prove it lands what the RPC route lands.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

LEADER_SCHEDULE_SCHEMA_VERSION = "v1"

#: Slots per epoch on mainnet. Used only to bound how far ahead a fetch is
#: allowed to ask; the authoritative answer always comes from the RPC.
SLOTS_PER_EPOCH = 432_000

#: How many upcoming slots to hold. Four hundred is a little over three
#: minutes at 400ms a slot -- long enough that a refresh failing once is
#: survivable, short enough that the answer is never stale by an epoch.
LOOKAHEAD_SLOTS = 400

#: Refresh when fewer than this many slots of schedule remain ahead of the
#: current one. Not a clock: a desk that has been idle should not refresh,
#: and a desk consuming slots quickly should.
REFRESH_BELOW_SLOTS = 120

#: How many upcoming leaders to prewarm a connection to. The current leader
#: and the next few: Solana gives each leader four consecutive slots, so
#: warming two distinct leaders covers roughly three seconds of production,
#: which is far longer than a connection takes to establish.
PREWARM_LEADERS = 2

#: A cluster-node list is large and changes slowly. Refetched on this
#: cadence rather than with the schedule, because they answer different
#: questions and one being stale does not make the other wrong.
NODES_TTL_S = 900.0


@dataclass
class LeaderNode:
    """One validator's identity and the addresses it listens on."""

    identity: str
    tpu: str = ""
    tpu_quic: str = ""
    gossip: str = ""

    @property
    def reachable(self) -> bool:
        """Whether this node published somewhere a transaction could go.

        A validator with no TPU address in gossip is not a bug in this code;
        some nodes deliberately do not advertise one, and a route that
        pretended otherwise would fail at send time with a worse error.
        """
        return bool(self.tpu_quic or self.tpu)


class LeaderSchedule:
    """Slot to leader to address, refreshed in the background.

    Every lookup is a dictionary hit. The two RPC calls behind it happen on
    a refresh that is never on the path of a decision -- which is the same
    rule the fee config, the portfolio state and the risk audit now follow,
    for the same reason.
    """

    def __init__(self, rpc: Any, *, lookahead: int = LOOKAHEAD_SLOTS,
                 prewarm_leaders: int = PREWARM_LEADERS):
        self.rpc = rpc
        self.lookahead = int(lookahead)
        self.prewarm_leaders = int(prewarm_leaders)
        self._leaders: Dict[int, str] = {}
        self._nodes: Dict[str, LeaderNode] = {}
        self._first_slot = 0
        self._nodes_fetched_at = 0.0
        self.current_slot = 0
        self.refreshes = 0
        self.refresh_failures = 0
        self.lookups = 0
        self.lookup_misses = 0
        self.last_error = ""

    # --- refresh ---------------------------------------------------------

    def needs_refresh(self, slot: Optional[int] = None) -> bool:
        slot = int(slot or self.current_slot or 0)
        if not self._leaders:
            return True
        remaining = (self._first_slot + len(self._leaders)) - slot
        return remaining < REFRESH_BELOW_SLOTS

    async def refresh(self, slot: Optional[int] = None) -> bool:
        """Pull the schedule ahead of `slot`, and the node list if stale."""
        start = int(slot or self.current_slot or 0)
        if start <= 0:
            try:
                start = int(await self.rpc.request("getSlot", [{"commitment": "processed"}]) or 0)
            except Exception as exc:
                self.refresh_failures += 1
                self.last_error = f"getSlot: {exc}"
                return False
        if start <= 0:
            self.refresh_failures += 1
            self.last_error = "getSlot returned nothing"
            return False
        self.current_slot = start
        try:
            leaders = await self.rpc.request(
                "getSlotLeaders", [start, min(self.lookahead, SLOTS_PER_EPOCH)])
        except Exception as exc:
            self.refresh_failures += 1
            self.last_error = f"getSlotLeaders: {exc}"
            return False
        if not leaders:
            self.refresh_failures += 1
            self.last_error = "getSlotLeaders returned nothing"
            return False
        # Replaced wholesale rather than merged. A schedule that mixes two
        # fetches across an epoch boundary would answer confidently and
        # wrongly for the slots either side of it.
        self._leaders = {start + offset: str(identity)
                         for offset, identity in enumerate(leaders)}
        self._first_slot = start
        self.refreshes += 1
        self.last_error = ""
        if time.time() - self._nodes_fetched_at > NODES_TTL_S:
            await self._refresh_nodes()
        return True

    async def _refresh_nodes(self) -> bool:
        try:
            nodes = await self.rpc.request("getClusterNodes", [])
        except Exception as exc:
            # Not a refresh failure: the schedule is still good, and a
            # leader whose address is unknown is a leader the desk simply
            # cannot reach directly. Degrading is the correct behaviour.
            self.last_error = f"getClusterNodes: {exc}"
            return False
        if not nodes:
            return False
        self._nodes = {}
        for node in nodes:
            identity = str((node or {}).get("pubkey", "") or "")
            if not identity:
                continue
            self._nodes[identity] = LeaderNode(
                identity=identity,
                tpu=str(node.get("tpu", "") or ""),
                tpu_quic=str(node.get("tpuQuic", "") or node.get("tpu_quic", "") or ""),
                gossip=str(node.get("gossip", "") or ""))
        self._nodes_fetched_at = time.time()
        return True

    # --- lookups ---------------------------------------------------------

    def leader_for(self, slot: int) -> str:
        """The validator identity producing `slot`, or "" if unknown.

        Empty, never a guess. The landing model buckets by this string, and
        a wrong leader is worse than no leader: it puts one validator's
        accept rate into another's bucket, permanently.
        """
        self.lookups += 1
        identity = self._leaders.get(int(slot), "")
        if not identity:
            self.lookup_misses += 1
        return identity

    def node_for(self, slot: int) -> Optional[LeaderNode]:
        identity = self.leader_for(slot)
        return self._nodes.get(identity) if identity else None

    def upcoming_leaders(self, slot: Optional[int] = None,
                         count: Optional[int] = None) -> List[LeaderNode]:
        """The next distinct leaders, in order, from `slot`.

        DISTINCT, because Solana gives each leader four consecutive slots and
        a list of the next eight slots is usually two validators repeated.
        Warming a connection twice to the same address buys nothing; warming
        the one after it buys the next three seconds.
        """
        start = int(slot or self.current_slot or 0)
        wanted = int(count or self.prewarm_leaders)
        seen: List[LeaderNode] = []
        identities: set = set()
        for offset in range(0, self.lookahead):
            identity = self._leaders.get(start + offset, "")
            if not identity or identity in identities:
                continue
            identities.add(identity)
            node = self._nodes.get(identity)
            if node is not None and node.reachable:
                seen.append(node)
            if len(seen) >= wanted:
                break
        return seen

    def observe_slot(self, slot: int) -> None:
        """Told by the stream which slot the chain is on. Costs nothing."""
        slot = int(slot or 0)
        if slot > self.current_slot:
            self.current_slot = slot

    def report(self) -> Dict[str, Any]:
        known = len(self._leaders)
        ahead = (self._first_slot + known) - self.current_slot if known else 0
        return {
            "schema": LEADER_SCHEDULE_SCHEMA_VERSION,
            "status": "OK" if known and self._nodes else "DATA_BLOCKED",
            "current_slot": self.current_slot or None,
            "slots_known": known,
            "slots_ahead": max(0, ahead),
            "nodes": len(self._nodes),
            "nodes_with_tpu_quic": sum(
                1 for node in self._nodes.values() if node.tpu_quic),
            "refreshes": self.refreshes,
            "refresh_failures": self.refresh_failures,
            "lookups": self.lookups,
            "lookup_miss_rate": (self.lookup_misses / self.lookups
                                 if self.lookups else None),
            "prewarm_targets": [node.identity for node in self.upcoming_leaders()],
            "last_error": self.last_error,
            "detail": ("slot to leader to address, refreshed in the "
                       "background; the landing model buckets accept rates on "
                       "this and has been writing every attempt into one "
                       "empty-string bucket"),
        }

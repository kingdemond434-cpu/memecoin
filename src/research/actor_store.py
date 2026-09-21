"""The actor graph, materialised, and answerable as of a date.

`actor_graph` scores actors in memory from whatever the desk happens to be
holding: wallet independence, smart flow, buyer DNA, swarm. All of it is
correct and all of it is transient. Nothing persists the graph, so "how many
launches had this deployer done BEFORE the one I am reconstructing" is a
question the desk cannot answer about 2024 while standing in 2026 -- and the
naive way to answer it, counting everything on disk, is the single most
effective way to manufacture insider-looking alpha out of leakage.

So this is a store, and the only way to read it is as of a time.

**Every edge carries when it was observed, and every query carries an as-of.**
A traversal at `as_of` cannot see an edge observed one second later. That is not
a convenience for backtests; it is the property that makes a reconstructed
feature the same number the desk would have computed live. A store without it
will happily tell you a deployer had forty prior launches on the day of their
first.

**Hubs are suppressed, not followed.** An exchange hot wallet funds hundreds of
thousands of addresses. Walking through one collapses the entire chain into a
single family and makes every wallet look related to every other -- which reads
downstream as "no independent buyers" on every launch simultaneously. A node
whose out-degree AS OF THE QUERY exceeds a threshold is a hub: it is recorded,
reported, and never traversed through. The degree is computed as-of for the same
reason everything else is, so a funder that became a hub in 2026 is still
walkable in a 2024 query.

**Traversal is bounded in hops and in nodes.** An unbounded walk over a chain
graph is unbounded work; the desk has 4 GB and a deadline. Both limits are
declared and both are reported when they bind, so a truncated answer is
distinguishable from a complete one.

The file format is append-only JSONL, one edge per line, because the corpus is
built incrementally from bulk history and an interrupted build should cost the
last line rather than the run.
"""

from __future__ import annotations

import bisect
import json
import logging
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import (Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple)

logger = logging.getLogger(__name__)

ACTOR_STORE_SCHEMA_VERSION = "v1"

#: Out-degree above which a node is a hub and is never traversed through.
#: Calibrated to be far above any plausible deployer or funding family and far
#: below an exchange: a real family funds tens of wallets, a hot wallet funds
#: six figures.
DEFAULT_HUB_DEGREE = 200

#: Hops a traversal may take. Four is enough to reach
#: exchange -> fresh -> funder -> wallet without becoming a graph scan.
DEFAULT_MAX_HOPS = 4

#: Nodes a single traversal may visit before it reports itself truncated.
DEFAULT_MAX_NODES = 5_000


class EdgeKind(Enum):
    #: funder -> wallet, a SOL transfer that created or topped up the wallet.
    FUNDED = "funded"
    #: deployer -> mint.
    DEPLOYED = "deployed"
    #: wallet -> mint, a buy.
    BOUGHT = "bought"
    #: wallet -> mint, a sell.
    SOLD = "sold"
    #: mint -> launchpad.
    LAUNCHED_ON = "launched_on"


@dataclass(frozen=True)
class Edge:
    """One observed relationship, with the moment it became observable."""

    source: str
    target: str
    kind: EdgeKind
    observed_at: float
    #: Ordinal among buyers of a mint, where known. `first_buyers` is the
    #: sequence the whole First25 fingerprint is built on, and it is not
    #: recoverable from an unordered edge set.
    rank: Optional[int] = None
    amount: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "kind": self.kind.value}

    @classmethod
    def from_dict(cls, row: Dict[str, Any]) -> "Edge":
        return cls(source=str(row["source"]), target=str(row["target"]),
                   kind=EdgeKind(row["kind"]),
                   observed_at=float(row["observed_at"]),
                   rank=row.get("rank"), amount=row.get("amount"))


@dataclass
class Traversal:
    """What a walk found, and whether it finished."""

    reached: Dict[str, int] = field(default_factory=dict)
    hubs_skipped: List[str] = field(default_factory=list)
    truncated: bool = False
    visited: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"reached": dict(self.reached),
                "hubs_skipped": list(self.hubs_skipped),
                "truncated": self.truncated, "visited": self.visited}


class ActorStore:
    """An append-only edge log with as-of indexes over it."""

    def __init__(self, path: Optional[Path] = None, *,
                 hub_degree: int = DEFAULT_HUB_DEGREE,
                 max_hops: int = DEFAULT_MAX_HOPS,
                 max_nodes: int = DEFAULT_MAX_NODES):
        self.path = Path(path) if path else None
        self.hub_degree = int(hub_degree)
        self.max_hops = int(max_hops)
        self.max_nodes = int(max_nodes)
        self.edges: List[Edge] = []
        #: (source, kind) -> [(observed_at, index)], kept sorted by time so an
        #: as-of cut is a bisect rather than a scan of the whole adjacency.
        self._out: Dict[Tuple[str, EdgeKind], List[Tuple[float, int]]] = (
            defaultdict(list))
        self._in: Dict[Tuple[str, EdgeKind], List[Tuple[float, int]]] = (
            defaultdict(list))
        self._appended = 0

    # -- building --------------------------------------------------------

    def add(self, edge: Edge) -> None:
        index = len(self.edges)
        self.edges.append(edge)
        for store, key in ((self._out, (edge.source, edge.kind)),
                           (self._in, (edge.target, edge.kind))):
            bucket = store[key]
            item = (edge.observed_at, index)
            # Insert in time order rather than appending: bulk history arrives
            # out of order (one partition at a time, and partitions are days),
            # and an unsorted bucket makes every as-of cut wrong rather than
            # slow.
            bisect.insort(bucket, item)

    def extend(self, edges: Iterable[Edge]) -> int:
        count = 0
        for edge in edges:
            self.add(edge)
            count += 1
        return count

    def append_to_log(self, edges: Sequence[Edge]) -> int:
        if self.path is None:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            for edge in edges:
                handle.write(json.dumps(edge.to_dict(),
                                        separators=(",", ":")) + "\n")
        self._appended += len(edges)
        return len(edges)

    def load(self) -> int:
        """Read the log back. A malformed line is skipped, not fatal.

        The log is built by long bulk-history runs; losing the whole corpus to
        one truncated final line is a worse failure than dropping that line and
        saying how many were dropped.
        """
        if self.path is None or not self.path.exists():
            return 0
        loaded = 0
        dropped = 0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                self.add(Edge.from_dict(json.loads(line)))
                loaded += 1
            except (ValueError, KeyError, TypeError):
                dropped += 1
        if dropped:
            logger.warning("actor store: dropped %d unreadable edge(s)",
                           dropped)
        return loaded

    # -- as-of primitives ------------------------------------------------

    @staticmethod
    def _cut(bucket: Sequence[Tuple[float, int]], as_of: float) -> int:
        """How many entries were observed at or before `as_of`."""
        return bisect.bisect_right(bucket, (float(as_of), len(bucket) + 1))

    def out_edges(self, node: str, kind: EdgeKind, as_of: float
                  ) -> List[Edge]:
        bucket = self._out.get((node, kind)) or []
        return [self.edges[index]
                for _, index in bucket[:self._cut(bucket, as_of)]]

    def in_edges(self, node: str, kind: EdgeKind, as_of: float) -> List[Edge]:
        bucket = self._in.get((node, kind)) or []
        return [self.edges[index]
                for _, index in bucket[:self._cut(bucket, as_of)]]

    def out_degree(self, node: str, kind: EdgeKind, as_of: float) -> int:
        bucket = self._out.get((node, kind)) or []
        return self._cut(bucket, as_of)

    def is_hub(self, node: str, kind: EdgeKind, as_of: float) -> bool:
        """Hub-ness is as-of too.

        A funder that became an exchange hot wallet in 2026 was an ordinary
        address in 2024, and refusing to walk it in a 2024 query would apply
        knowledge the desk did not have -- which is leakage in the direction
        nobody checks for, because it makes the answer more conservative
        rather than better.
        """
        return self.out_degree(node, kind, as_of) >= self.hub_degree

    # -- materialised features -------------------------------------------

    def prior_launches(self, deployer: str, as_of: float) -> int:
        """Launches this deployer had already done. The leakage canary."""
        return self.out_degree(deployer, EdgeKind.DEPLOYED, as_of)

    def prior_mints(self, deployer: str, as_of: float) -> List[str]:
        return [edge.target
                for edge in self.out_edges(deployer, EdgeKind.DEPLOYED, as_of)]

    def first_buyers(self, mint: str, as_of: float, depth: int = 25
                     ) -> List[str]:
        """The opening buyer sequence, in order, as of a time.

        Ordered by rank where the builder supplied one and by observation time
        otherwise, because the First25 fingerprint is a SEQUENCE and an
        unordered set of the same wallets is a different object.
        """
        edges = self.in_edges(mint, EdgeKind.BOUGHT, as_of)
        edges.sort(key=lambda edge: (edge.rank if edge.rank is not None
                                     else float("inf"), edge.observed_at))
        return [edge.source for edge in edges[:depth]]

    def funders_of(self, wallet: str, as_of: float) -> List[str]:
        return [edge.source
                for edge in self.in_edges(wallet, EdgeKind.FUNDED, as_of)]

    def family(self, wallet: str, as_of: float, *,
               max_hops: Optional[int] = None) -> Traversal:
        """Every wallet reachable through funding, hubs excluded.

        Walks both directions on FUNDED edges -- a sibling is reached by going
        up to the funder and back down -- and never through a hub, so two
        wallets that share only an exchange are not called family.
        """
        hops_limit = self.max_hops if max_hops is None else int(max_hops)
        result = Traversal(reached={wallet: 0})
        queue: deque = deque([(wallet, 0)])
        seen_hubs: Set[str] = set()
        while queue:
            node, hops = queue.popleft()
            result.visited += 1
            if result.visited > self.max_nodes:
                result.truncated = True
                break
            if hops >= hops_limit:
                continue
            neighbours: List[str] = []
            for edge in self.in_edges(node, EdgeKind.FUNDED, as_of):
                source = edge.source
                if self.is_hub(source, EdgeKind.FUNDED, as_of):
                    if source not in seen_hubs:
                        seen_hubs.add(source)
                        result.hubs_skipped.append(source)
                    continue
                neighbours.append(source)
            if not self.is_hub(node, EdgeKind.FUNDED, as_of):
                neighbours.extend(
                    edge.target
                    for edge in self.out_edges(node, EdgeKind.FUNDED, as_of))
            for neighbour in neighbours:
                if neighbour in result.reached:
                    continue
                result.reached[neighbour] = hops + 1
                queue.append((neighbour, hops + 1))
        return result

    def shared_family(self, wallets: Sequence[str], as_of: float
                      ) -> Dict[str, Any]:
        """How much of a buyer set collapses into one funding family.

        The number the independence machinery wants: ten wallets that are one
        family are one buyer, and a launch whose First25 is one family has no
        independent demand however many addresses appear in it.
        """
        unique = [wallet for wallet in dict.fromkeys(wallets) if wallet]
        if not unique:
            return {"status": "DATA_BLOCKED", "reason": "no wallets given"}
        assigned: Dict[str, int] = {}
        families: List[Set[str]] = []
        truncated = False
        for wallet in unique:
            if wallet in assigned:
                continue
            walk = self.family(wallet, as_of)
            truncated = truncated or walk.truncated
            group = {node for node in walk.reached if node in set(unique)}
            group.add(wallet)
            index = len(families)
            families.append(group)
            for member in group:
                assigned.setdefault(member, index)
        distinct = len({assigned[wallet] for wallet in unique})
        return {
            "status": "OK",
            "wallets": len(unique),
            "families": distinct,
            "independence": distinct / len(unique),
            "largest_family": max((len(group) for group in families),
                                  default=0),
            "truncated": truncated,
        }

    # -- construction from bulk history ----------------------------------

    def ingest_launch(self, *, mint: str, creator: str, created_at: float,
                      launchpad: str = "",
                      buyers: Sequence[Tuple[str, float]] = (),
                      funding: Sequence[Tuple[str, str, float]] = ()
                      ) -> int:
        """One reconstructed launch, as edges.

        `buyers` is (wallet, observed_at) in order; `funding` is
        (funder, wallet, observed_at). Rank is assigned from the order given,
        so the caller's ordering is the one preserved.
        """
        edges: List[Edge] = [Edge(source=creator, target=mint,
                                  kind=EdgeKind.DEPLOYED,
                                  observed_at=float(created_at))]
        if launchpad:
            edges.append(Edge(source=mint, target=launchpad,
                              kind=EdgeKind.LAUNCHED_ON,
                              observed_at=float(created_at)))
        for rank, (wallet, stamp) in enumerate(buyers):
            edges.append(Edge(source=wallet, target=mint,
                              kind=EdgeKind.BOUGHT,
                              observed_at=float(stamp), rank=rank))
        for funder, wallet, stamp in funding:
            edges.append(Edge(source=funder, target=wallet,
                              kind=EdgeKind.FUNDED, observed_at=float(stamp)))
        self.extend(edges)
        self.append_to_log(edges)
        return len(edges)

    def ingest_raw_launches(self, launches: Iterable[Any]) -> Dict[str, int]:
        """Bulk history's `RawLaunch` records, as edges.

        Reads defensively: a launch missing a creation time contributes
        nothing rather than an edge stamped zero, which would be visible to
        every as-of query ever made.
        """
        built = 0
        skipped = 0
        for launch in launches:
            mint = str(getattr(launch, "token", "") or "")
            created = getattr(launch, "created_at", None)
            if not mint or not created:
                skipped += 1
                continue
            buyers: List[Tuple[str, float]] = []
            for trade in (getattr(launch, "trades", None) or []):
                if not isinstance(trade, dict):
                    continue
                wallet = str(trade.get("signer", "") or "")
                stamp = trade.get("block_timestamp")
                if wallet and stamp:
                    buyers.append((wallet, float(stamp)))
            transfers: List[Tuple[str, str, float]] = []
            for row in (getattr(launch, "funding_transfers", None) or []):
                if not isinstance(row, dict):
                    continue
                source = str(row.get("source", "") or "")
                destination = str(row.get("destination", "") or "")
                stamp = row.get("block_timestamp")
                if source and destination and stamp:
                    transfers.append((source, destination, float(stamp)))
            built += self.ingest_launch(
                mint=mint, creator=str(getattr(launch, "creator", "") or ""),
                created_at=float(created), buyers=buyers, funding=transfers)
        return {"edges": built, "skipped": skipped}

    # -- reporting -------------------------------------------------------

    def report(self) -> Dict[str, Any]:
        by_kind: Dict[str, int] = defaultdict(int)
        for edge in self.edges:
            by_kind[edge.kind.value] += 1
        stamps = [edge.observed_at for edge in self.edges]
        return {
            "schema": ACTOR_STORE_SCHEMA_VERSION,
            "edges": len(self.edges),
            "by_kind": dict(sorted(by_kind.items())),
            "earliest": min(stamps) if stamps else None,
            "latest": max(stamps) if stamps else None,
            "appended": self._appended,
            "hub_degree": self.hub_degree,
            "max_hops": self.max_hops,
        }

"""Wallets that keep showing up together, across thousands of launches.

The independence graph already asks whether two wallets are the same actor:
shared funding, shared ancestry, one repeatedly following the other. That is
a question about a PAIR, and it is answered from evidence inside a launch.

This asks a different one. Some sets of wallets appear in the opening cohort
of launch after launch after launch, with no funding edge between them and
no follow relationship a pair test can see -- because they are not funding
each other, they are running from one operator's list, or one bot's config,
or one paid group's signal. On any single launch that looks like a
coincidence. Across four hundred launches it is not, and the arithmetic that
says so is a straightforward one: if wallet A opens 3% of launches and
wallet B opens 2%, they should co-open 0.06% of them, and if they co-open 4%
they are not two independent decisions.

Why it matters is not that they are cheating. It is that the desk's whole
sizing argument rests on counting INDEPENDENT skilled buyers, and a ring of
twelve addresses running one strategy is one buyer wearing twelve hats. A
First-25 that is really First-3 has been badly overcounted, and the
overcounting is in the direction that says enter bigger.

Two disciplines make this safe:

**Surprisal, not counts.** The pair that co-opens most is usually just the
pair that opens most -- two prolific bots meet everywhere. What matters is
co-occurrence far above what their individual rates predict, so every edge
is scored against the null of independence and the busy pair with no
relationship scores nothing.

**A ring is a claim about behaviour, never about identity.** This never says
two addresses are one person. It says their entries are not independent
evidence, which is the only thing the sizing needs and the only thing the
data can support.
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from src.strategies.surprisal import binomial_surprisal

logger = logging.getLogger(__name__)

SNIPER_RINGS_SCHEMA_VERSION = "v1"

#: Launches a wallet must have opened before its rate is worth anything.
#: A wallet seen twice has a co-occurrence rate of 0%, 50% or 100% and none
#: of those numbers mean anything.
MIN_LAUNCHES_PER_WALLET = 20

#: Launches the desk must have observed before ANY rate is computed. Below
#: this the base rates themselves are noise, so every pair looks surprising.
MIN_LAUNCHES_OBSERVED = 200

#: Co-openings a pair needs before it is scored at all. Cuts the quadratic
#: blow-up of pairs that met twice, which is almost all of them.
MIN_CO_OPENINGS = 8

#: Surprisal, in nats, above which a pair is treated as related. This is
#: -log(P(co-occurrence at least this often | independent)), so 12 nats is
#: about one in 160,000 -- comfortably past what thousands of pairs tested
#: at once will throw up by chance.
SURPRISAL_THRESHOLD = 12.0

#: Wallets tracked. The tail of one-launch addresses is most of the
#: population and none of the information.
MAX_WALLETS = 20_000

#: Ring members beyond which the set is reported but not used to discount.
#: A "ring" of four hundred addresses is a description of the market, not of
#: a coordinated actor, and discounting on it would zero every launch.
MAX_RING_SIZE = 40


@dataclass
class Ring:
    """A set of wallets whose entries are not independent evidence."""

    members: FrozenSet[str]
    co_openings: int
    surprisal: float
    first_seen: float = 0.0
    last_seen: float = 0.0

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def usable(self) -> bool:
        """Whether this ring may discount a cohort.

        A ring of four hundred addresses is a description of the market
        rather than of a coordinated actor, and discounting on it would zero
        every launch the desk ever sees.
        """
        return 2 <= self.size <= MAX_RING_SIZE

    def as_dict(self) -> Dict[str, Any]:
        return {
            "size": self.size,
            "members": sorted(self.members)[:12],
            "co_openings": self.co_openings,
            "surprisal_nats": round(self.surprisal, 2),
            "usable": self.usable,
            "first_seen": self.first_seen or None,
            "last_seen": self.last_seen or None,
        }


class SniperRingDetector:
    """Finds wallet sets that keep opening the same launches together."""

    def __init__(self, *, min_launches: int = MIN_LAUNCHES_PER_WALLET,
                 min_observed: int = MIN_LAUNCHES_OBSERVED,
                 min_co_openings: int = MIN_CO_OPENINGS,
                 surprisal_threshold: float = SURPRISAL_THRESHOLD,
                 max_wallets: int = MAX_WALLETS):
        self.min_launches = int(min_launches)
        self.min_observed = int(min_observed)
        self.min_co_openings = int(min_co_openings)
        self.surprisal_threshold = float(surprisal_threshold)
        self.max_wallets = int(max_wallets)
        self.launches_observed = 0
        self._opened: Dict[str, int] = defaultdict(int)
        self._pairs: Dict[Tuple[str, str], int] = defaultdict(int)
        self._first_seen: Dict[str, float] = {}
        self._last_seen: Dict[str, float] = {}
        self._rings: List[Ring] = []
        self._rings_computed_at = 0.0
        self.evicted = 0

    # --- accumulation ----------------------------------------------------

    def observe_launch(self, openers: Sequence[str],
                       at: Optional[float] = None) -> None:
        """One launch's opening cohort. Order does not matter here.

        The pair counting is quadratic in the cohort, which is why the
        cohort is the OPENING one -- the first 25 distinct wallets -- rather
        than everybody who ever bought. 25 wallets is 300 pairs; 2,000
        wallets would be two million, per launch, and the question being
        asked is about who arrived first anyway.
        """
        unique = sorted({str(wallet) for wallet in openers if wallet})
        if not unique:
            return
        moment = float(at or time.time())
        self.launches_observed += 1
        for wallet in unique:
            self._opened[wallet] += 1
            self._first_seen.setdefault(wallet, moment)
            self._last_seen[wallet] = moment
        for index, left in enumerate(unique):
            for right in unique[index + 1:]:
                self._pairs[(left, right)] += 1
        self._evict()

    def _evict(self) -> None:
        if len(self._opened) <= self.max_wallets:
            return
        # Keep the wallets that open often: a wallet seen once carries no
        # rate and is most of the population.
        keep = set(sorted(self._opened, key=self._opened.get,
                          reverse=True)[:self.max_wallets])
        for wallet in [w for w in self._opened if w not in keep]:
            del self._opened[wallet]
            self._first_seen.pop(wallet, None)
            self._last_seen.pop(wallet, None)
            self.evicted += 1
        self._pairs = defaultdict(
            int, {pair: count for pair, count in self._pairs.items()
                  if pair[0] in keep and pair[1] in keep})

    # --- detection -------------------------------------------------------

    def related_pairs(self) -> List[Tuple[Tuple[str, str], int, float]]:
        """Pairs whose co-openings are far above independence, with scores."""
        if self.launches_observed < self.min_observed:
            return []
        out: List[Tuple[Tuple[str, str], int, float]] = []
        for (left, right), together in self._pairs.items():
            if together < self.min_co_openings:
                continue
            left_count = self._opened.get(left, 0)
            right_count = self._opened.get(right, 0)
            if (left_count < self.min_launches or right_count < self.min_launches):
                continue
            # The null: two wallets choosing independently at their own
            # observed rates. The pair that co-opens most is usually just
            # the pair that opens most, and this is what tells them apart.
            expected = ((left_count / self.launches_observed)
                        * (right_count / self.launches_observed))
            surprisal = binomial_surprisal(together, self.launches_observed,
                                           expected)
            if surprisal >= self.surprisal_threshold:
                out.append(((left, right), together, surprisal))
        out.sort(key=lambda row: row[2], reverse=True)
        return out

    def rings(self, recompute: bool = False) -> List[Ring]:
        """Connected components of the related-pair graph.

        Transitive on purpose: if A is not independent of B, and B is not
        independent of C, then counting A, B and C as three independent
        buyers is wrong even when the A-C pair alone is not significant. The
        sizing question is "how many independent decisions are here", and
        that is a component count.
        """
        if self._rings and not recompute:
            return self._rings
        edges = self.related_pairs()
        adjacency: Dict[str, Set[str]] = defaultdict(set)
        strength: Dict[FrozenSet[str], Tuple[int, float]] = {}
        for (left, right), together, surprisal in edges:
            adjacency[left].add(right)
            adjacency[right].add(left)
            strength[frozenset((left, right))] = (together, surprisal)
        seen: Set[str] = set()
        rings: List[Ring] = []
        for start in adjacency:
            if start in seen:
                continue
            component: Set[str] = set()
            stack = [start]
            while stack:
                wallet = stack.pop()
                if wallet in component:
                    continue
                component.add(wallet)
                stack.extend(adjacency[wallet] - component)
            seen |= component
            if len(component) < 2:
                continue
            inner = [strength[frozenset(pair)]
                     for pair in strength
                     if set(pair) <= component]
            rings.append(Ring(
                members=frozenset(component),
                co_openings=max((row[0] for row in inner), default=0),
                surprisal=max((row[1] for row in inner), default=0.0),
                first_seen=min(self._first_seen.get(w, 0.0) for w in component),
                last_seen=max(self._last_seen.get(w, 0.0) for w in component)))
        rings.sort(key=lambda ring: ring.surprisal, reverse=True)
        self._rings = rings
        self._rings_computed_at = time.time()
        return rings

    def ring_for(self, wallet: str) -> Optional[Ring]:
        for ring in self.rings():
            if wallet in ring.members:
                return ring
        return None

    def independent_count(self, wallets: Iterable[str]) -> Tuple[int, Dict[str, Any]]:
        """How many INDEPENDENT decisions this set of wallets represents.

        A ring of twelve addresses running one strategy is one buyer wearing
        twelve hats, and the desk's sizing rests on counting independent
        skilled buyers -- so a First-25 that is really a First-3 has been
        overcounted, in the direction that says enter bigger.

        Returns the count and what was collapsed, because a number that
        shrinks a position needs to be able to say why.
        """
        members = [str(wallet) for wallet in wallets if wallet]
        if not members:
            return 0, {"status": "OK", "collapsed": [], "raw": 0}
        usable = [ring for ring in self.rings() if ring.usable]
        if not usable:
            return len(set(members)), {
                "status": "DATA_BLOCKED" if self.launches_observed < self.min_observed
                          else "OK",
                "reason": (f"{self.launches_observed} launches observed, "
                           f"{self.min_observed} needed before co-occurrence "
                           "rates mean anything")
                          if self.launches_observed < self.min_observed else "",
                "collapsed": [], "raw": len(set(members))}
        remaining = set(members)
        collapsed: List[Dict[str, Any]] = []
        count = 0
        for ring in usable:
            overlap = remaining & ring.members
            if len(overlap) > 1:
                collapsed.append({"collapsed_to_one": sorted(overlap),
                                  "surprisal_nats": round(ring.surprisal, 2)})
                count += 1
                remaining -= overlap
            elif overlap:
                # One member of a ring present is one independent decision,
                # exactly like any unrelated wallet.
                remaining -= overlap
                count += 1
        count += len(remaining)
        return count, {"status": "OK", "collapsed": collapsed,
                       "raw": len(set(members))}

    def report(self) -> Dict[str, Any]:
        rings = self.rings(recompute=True)
        usable = [ring for ring in rings if ring.usable]
        return {
            "schema": SNIPER_RINGS_SCHEMA_VERSION,
            "status": "OK" if self.launches_observed >= self.min_observed
                      else "DATA_BLOCKED",
            "launches_observed": self.launches_observed,
            "launches_needed": self.min_observed,
            "wallets_tracked": len(self._opened),
            "pairs_tracked": len(self._pairs),
            "evicted": self.evicted,
            "rings": len(rings),
            "usable_rings": len(usable),
            "largest_ring": max((ring.size for ring in rings), default=0),
            "top": [ring.as_dict() for ring in rings[:5]],
            "detail": ("wallet sets that open the same launches together far "
                       "more often than their individual rates predict; a "
                       "claim about whether their entries are independent "
                       "evidence, never about who they are"),
        }

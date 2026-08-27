"""Collapsing wallets into the actors that funded them.

Behavioural independence already asks whether twelve wallets buying together
are twelve opinions or one. It answers from repeated co-occurrence and
lead-lag, which is the right question asked of history -- and it is blind on
exactly the launch that matters most: the first time a freshly funded cluster
appears. Wallets created this morning and funded from one source have no
co-occurrence history at all, so every one of them reads as an unmeasured
independent participant, and a Sybil built for one launch is invisible
precisely when it is deployed.

Funding ancestry answers the same question from structure rather than history.
If five buyers trace back to one funder within a few hops, they are one actor's
capital however new they are, and the evidence they represent should be counted
once. That is not a heuristic overlay on independence -- it is the other half
of it, and the two belong in one number:

    effective independent actors = sum over clusters of (cluster independence)

where a cluster is a set of wallets sharing a funding ancestor within `k` hops
and its independence is bounded by the most independent member rather than
summed across members.

Two things this deliberately does NOT do.

It does not treat a shared funder as proof of one operator. Exchange hot
wallets, bridges and faucets fund thousands of unrelated people, so a funder
above `hub_threshold` distinct fundees is treated as infrastructure and confers
no kinship at all. Without that, every wallet withdrawing from one exchange
would collapse into a single actor and the metric would report the whole market
as one participant.

And it never raises independence. Ancestry can only ever say "these are more
related than they looked"; a wallet whose behaviour already marks it dependent
does not become independent because its funding is untraceable.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

FUNDER_ANCESTRY_SCHEMA_VERSION = "v1"

# How far back a funding chain is followed. Four hops covers the usual
# laundering depth for a launch-day cluster -- fund a parent, split to
# children, split again -- without walking so far that everything eventually
# shares an exchange ancestor and the whole market collapses to one actor.
DEFAULT_MAX_HOPS = 4

# A wallet that has funded more than this many distinct wallets is
# infrastructure, not an operator. Exchange withdrawal wallets fund tens of
# thousands; treating them as kinship would report every trader on one venue
# as the same actor.
DEFAULT_HUB_THRESHOLD = 50

# Kinship decays with distance: sharing a direct funder is strong evidence of
# one operator, sharing a great-great-grandparent is weak. The independence a
# cluster keeps rises back toward 1 as the shared ancestor gets further away.
DEFAULT_HOP_DECAY = 0.45


@dataclass
class FunderCluster:
    """Wallets that trace to one funding ancestor, and how strong that tie is."""

    ancestor: str
    wallets: Tuple[str, ...]
    # Hops from each wallet to the shared ancestor, worst case in the cluster.
    max_depth: int
    kinship: float
    detail: str = ""

    @property
    def size(self) -> int:
        return len(self.wallets)

    def to_dict(self) -> Dict[str, Any]:
        return {"ancestor": self.ancestor, "wallets": list(self.wallets),
                "size": self.size, "max_depth": self.max_depth,
                "kinship": round(self.kinship, 4), "detail": self.detail}


@dataclass
class AncestryReport:
    """What the funding graph says about a set of buyers."""

    status: str
    wallets: int = 0
    # Wallets whose funding could actually be traced. The rest are not
    # "independent" -- they are unmeasured, and are reported as such.
    traced: int = 0
    clusters: List[FunderCluster] = field(default_factory=list)
    # wallet -> the multiplier ancestry applies to its independence, <= 1.
    compression: Dict[str, float] = field(default_factory=dict)
    hub_funders: List[str] = field(default_factory=list)
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "OK"

    @property
    def effective_actors(self) -> Optional[float]:
        """How many independent actors these wallets actually represent."""
        if not self.compression:
            return None
        return float(sum(self.compression.values()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": FUNDER_ANCESTRY_SCHEMA_VERSION, "status": self.status,
            "wallets": self.wallets, "traced": self.traced,
            "clusters": [cluster.to_dict() for cluster in self.clusters],
            "clustered_wallets": sum(cluster.size for cluster in self.clusters
                                     if cluster.size > 1),
            "effective_actors": self.effective_actors,
            "compression_ratio": (self.effective_actors / self.wallets
                                  if self.wallets and self.effective_actors is not None
                                  else None),
            "hub_funders": list(self.hub_funders), "detail": self.detail,
        }


class FunderAncestry:
    """Funding ancestry over the genealogy graph, with hubs excluded.

    Reads the graph rather than owning one: the genealogy module already
    records who funded whom, and a second copy of that relation is a second
    thing to keep in step.
    """

    def __init__(self, genealogy: Any, *, max_hops: int = DEFAULT_MAX_HOPS,
                 hub_threshold: int = DEFAULT_HUB_THRESHOLD,
                 hop_decay: float = DEFAULT_HOP_DECAY):
        self.genealogy = genealogy
        self.max_hops = max(1, int(max_hops))
        self.hub_threshold = max(2, int(hub_threshold))
        self.hop_decay = float(hop_decay)
        self._fundee_counts: Dict[str, int] = {}
        self._hub_cache: Dict[str, bool] = {}

    # --- graph access ----------------------------------------------------

    def _funders_of(self, wallet: str) -> Set[str]:
        profile = getattr(self.genealogy, "wallets", {}).get(wallet)
        if profile is None:
            return set()
        return {str(item) for item in (getattr(profile, "funding_sources", None) or ())
                if item and item != wallet}

    def _count_fundees(self) -> Dict[str, int]:
        """How many distinct wallets each funder has funded, across the graph."""
        counts: Dict[str, int] = {}
        for address, profile in getattr(self.genealogy, "wallets", {}).items():
            for funder in (getattr(profile, "funding_sources", None) or ()):
                counts[str(funder)] = counts.get(str(funder), 0) + 1
        return counts

    def is_hub(self, wallet: str) -> bool:
        """Whether this funder is infrastructure rather than an operator."""
        if wallet not in self._hub_cache:
            if not self._fundee_counts:
                self._fundee_counts = self._count_fundees()
            self._hub_cache[wallet] = (
                self._fundee_counts.get(wallet, 0) > self.hub_threshold)
        return self._hub_cache[wallet]

    def refresh(self) -> None:
        """Recount fundees. Cheap, and the counts move as the graph grows."""
        self._fundee_counts = self._count_fundees()
        self._hub_cache.clear()

    def ancestors(self, wallet: str) -> Dict[str, int]:
        """Funding ancestors within ``max_hops``, mapped to their distance.

        Breadth-first, so a wallet reachable by two paths is recorded at the
        SHORTER one: kinship is about the closest tie, and taking the longer
        path would understate a cluster that also happens to be laundered.
        """
        found: Dict[str, int] = {}
        queue: deque = deque((funder, 1) for funder in self._funders_of(wallet))
        seen = {wallet}
        while queue:
            current, depth = queue.popleft()
            if current in seen or depth > self.max_hops:
                continue
            seen.add(current)
            if self.is_hub(current):
                # Infrastructure confers no kinship, and neither does anything
                # behind it: two people who both withdrew from one exchange are
                # not related through whoever funded the exchange.
                continue
            if current not in found or depth < found[current]:
                found[current] = depth
            if depth < self.max_hops:
                queue.extend((parent, depth + 1) for parent in self._funders_of(current))
        return found

    # --- compression -----------------------------------------------------

    def kinship_at(self, depth: int) -> float:
        """How strongly a shared ancestor at this distance implies one actor."""
        return float(self.hop_decay ** max(0, depth - 1))

    def analyse(self, wallets: Sequence[str]) -> AncestryReport:
        """Collapse these wallets into the actors that funded them."""
        unique = [wallet for index, wallet in enumerate(wallets)
                  if wallet and wallet not in wallets[:index]]
        if not unique:
            return AncestryReport(status="DATA_BLOCKED", detail="no wallets supplied")
        self.refresh()

        ancestry = {wallet: self.ancestors(wallet) for wallet in unique}
        traced = sum(1 for found in ancestry.values() if found)
        if not traced:
            # Not "all independent". Nothing was traceable, and reporting that
            # as full independence would let an untracked graph certify a
            # Sybil.
            return AncestryReport(
                status="DATA_BLOCKED", wallets=len(unique),
                detail="no funding ancestry recorded for any of these wallets")

        # ancestor -> {wallet: depth}
        shared: Dict[str, Dict[str, int]] = {}
        for wallet, found in ancestry.items():
            for ancestor, depth in found.items():
                shared.setdefault(ancestor, {})[wallet] = depth

        clusters: List[FunderCluster] = []
        # Strongest ties first, so a wallet is attributed to its closest
        # shared ancestor rather than to whichever was found first.
        candidates = sorted(
            ((ancestor, members) for ancestor, members in shared.items()
             if len(members) > 1),
            key=lambda item: (min(item[1].values()), -len(item[1])))
        claimed: Set[str] = set()
        for ancestor, members in candidates:
            free = {wallet: depth for wallet, depth in members.items()
                    if wallet not in claimed}
            if len(free) < 2:
                continue
            depth = max(free.values())
            clusters.append(FunderCluster(
                ancestor=ancestor, wallets=tuple(sorted(free)), max_depth=depth,
                kinship=self.kinship_at(depth),
                detail=f"{len(free)} wallets share this funder within {depth} hop(s)"))
            claimed.update(free)

        # Each clustered wallet keeps only the share of its own independence
        # that the cluster does not already account for. One wallet in a
        # cluster of five at full kinship contributes a fifth.
        compression: Dict[str, float] = {wallet: 1.0 for wallet in unique}
        for cluster in clusters:
            kept = (1.0 - cluster.kinship) + cluster.kinship / cluster.size
            for wallet in cluster.wallets:
                compression[wallet] = min(compression[wallet], kept)

        clustered = sum(cluster.size for cluster in clusters)
        return AncestryReport(
            status="OK", wallets=len(unique), traced=traced, clusters=clusters,
            compression=compression,
            hub_funders=sorted(name for name, count in self._fundee_counts.items()
                               if count > self.hub_threshold)[:50],
            detail=(f"{clustered} of {len(unique)} wallets collapse into "
                    f"{len(clusters)} funded cluster(s)"))

    # --- features for First25 -------------------------------------------

    def buyer_features(self, ordered_wallets: Sequence[str],
                       creator: str = "") -> List[Dict[str, Any]]:
        """Per-position funding features for an ordered buyer sequence.

        "Ten buyers" becomes an ordered sequence of economic actors rather
        than of wallets. Each position carries what its funding says about it
        RELATIVE TO THE BUYERS BEFORE IT, which is the thing a fingerprint
        can learn from and a per-wallet attribute cannot express.
        """
        self.refresh()
        ancestry = {wallet: self.ancestors(wallet) for wallet in ordered_wallets}
        creator_ancestry = self.ancestors(creator) if creator else {}
        features: List[Dict[str, Any]] = []
        for index, wallet in enumerate(ordered_wallets):
            found = ancestry.get(wallet, {})
            prior = ordered_wallets[:index]
            shares_with_prior = False
            nearest = None
            for earlier in prior:
                overlap = set(found) & set(ancestry.get(earlier, {}))
                if not overlap:
                    continue
                shares_with_prior = True
                depth = min(found[key] + ancestry[earlier][key] for key in overlap)
                nearest = depth if nearest is None else min(nearest, depth)
            creator_overlap = set(found) & set(creator_ancestry)
            features.append({
                "wallet": wallet,
                "traced": bool(found),
                "funder_count": len(found),
                "nearest_funder_depth": (min(found.values()) if found else None),
                "same_funder_as_prior_buyer": shares_with_prior,
                "nearest_common_funder_depth": nearest,
                # A buyer funded from the creator's own tree is the creator
                # buying his own launch, whatever address he used.
                "shares_funder_with_creator": bool(creator_overlap),
                "creator_funder_depth": (
                    min(found[key] + creator_ancestry[key] for key in creator_overlap)
                    if creator_overlap else None),
            })
        return features


def compress_independence(scores: Dict[str, float], report: AncestryReport,
                          ) -> Dict[str, float]:
    """Independence after funding ancestry, never above what behaviour said.

    Ancestry can only lower it. A wallet whose behaviour already marks it
    dependent does not become independent because its funding is untraceable,
    and a structurally-clean wallet does not earn independence it has not
    demonstrated.
    """
    if not report.ok:
        return dict(scores)
    return {wallet: min(float(value), float(report.compression.get(wallet, 1.0)))
            for wallet, value in scores.items()}

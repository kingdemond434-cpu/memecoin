"""Recognising a wallet by how it behaves, when its address is brand new.

Every wallet-quality signal this desk has is keyed on history: a score
needs outcomes, a rank needs a sample, `is_watched` needs a wallet to have
been watched. So a fresh address is unknown by construction -- and a fresh
address is exactly what a competent operator uses. Rotate the wallet, and
every reputation the desk built evaporates.

What does not evaporate is the strategy. An operator who buys in the first
900 milliseconds, sizes at a fifth of a SOL, takes profit inside two
minutes, never touches a launch whose deployer has prior rugs, and works
the 02:00-06:00 UTC window is running a program. The program survives the
rotation, because the program is the thing they built.

So this measures the program. Not identity -- a signature match is not a
claim that two addresses are one person, and this module refuses to make
that claim anywhere. It says: this new wallet's behaviour is drawn from the
same distribution as a cluster whose forward returns the desk has already
measured, and here is how confident that is and on how many observations.

The output feeds sizing the same way every other actor signal does: as a
PRIOR on an unknown wallet, worth less than a measured score and more than
nothing, and stamped so nothing downstream mistakes it for the real thing.

Deliberately simple arithmetic. A learned embedding over a few thousand
wallets would overfit the ones the desk happened to watch, and its failure
mode is a confident number nobody can interrogate. Standardised features
and a distance are inspectable, and every dimension here is something an
operator can be told in a sentence.
"""

from __future__ import annotations

import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

WALLET_SIGNATURE_SCHEMA_VERSION = "v1"

#: Entries a wallet needs before it has a signature at all. Below this the
#: features are one or two launches of noise, and a match against them would
#: be matching the noise.
MIN_ENTRIES = 8

#: Wallets a cluster needs before its centroid is worth matching against.
#: A "cluster" of two wallets is two wallets.
MIN_CLUSTER_WALLETS = 5

#: Standardised distance below which a wallet is called a match. Roughly:
#: within this many standard deviations, summed across the feature space, of
#: the cluster's centre.
MATCH_DISTANCE = 1.5

#: The features. Each is something an operator can be told in one sentence,
#: which is the property that makes the result interrogable when it is
#: wrong -- and it will sometimes be wrong.
FEATURES: Tuple[str, ...] = (
    "median_entry_age_s",       # how early into a launch they buy
    "median_size_sol",          # what they stake
    "median_hold_s",            # how long they stay
    "buy_share",                # buys as a share of their trades
    "launches_per_active_hour", # how hard they run
    "hour_concentration",       # how much of their activity sits in few hours
    "deployer_reuse_rate",      # how often they revisit a deployer
)


@dataclass
class WalletBehaviour:
    """Raw observations about one wallet, before standardisation."""

    wallet: str
    entry_ages_s: List[float] = field(default_factory=list)
    sizes_sol: List[float] = field(default_factory=list)
    holds_s: List[float] = field(default_factory=list)
    buys: int = 0
    sells: int = 0
    hours: Dict[int, int] = field(default_factory=dict)
    deployers: Dict[str, int] = field(default_factory=dict)
    first_seen: float = 0.0
    last_seen: float = 0.0

    @property
    def entries(self) -> int:
        return len(self.entry_ages_s)

    @property
    def measurable(self) -> bool:
        return self.entries >= MIN_ENTRIES

    def features(self) -> Optional[Dict[str, float]]:
        """The behaviour as numbers, or None when there is too little of it."""
        if not self.measurable:
            return None
        trades = self.buys + self.sells
        active_hours = max(1.0, (self.last_seen - self.first_seen) / 3600.0)
        hour_total = sum(self.hours.values()) or 1
        # How concentrated their activity is in time: 1.0 means everything
        # in one hour of the day, near 1/24 means uniform. A bot on a
        # schedule and a human awake in one timezone both show here, and
        # the desk does not need to know which.
        concentration = max(self.hours.values(), default=0) / hour_total
        revisits = sum(count - 1 for count in self.deployers.values() if count > 1)
        return {
            "median_entry_age_s": float(statistics.median(self.entry_ages_s)),
            "median_size_sol": float(statistics.median(self.sizes_sol))
                               if self.sizes_sol else 0.0,
            "median_hold_s": float(statistics.median(self.holds_s))
                             if self.holds_s else 0.0,
            "buy_share": float(self.buys / trades) if trades else 0.0,
            "launches_per_active_hour": float(self.entries / active_hours),
            "hour_concentration": float(concentration),
            "deployer_reuse_rate": float(revisits / self.entries),
        }


class WalletSignatures:
    """Behavioural signatures, and what a new wallet's resembles."""

    def __init__(self, *, min_entries: int = MIN_ENTRIES,
                 match_distance: float = MATCH_DISTANCE):
        self.min_entries = int(min_entries)
        self.match_distance = float(match_distance)
        self.behaviours: Dict[str, WalletBehaviour] = {}
        self.clusters: Dict[str, Dict[str, Any]] = {}
        self.matches_served = 0
        self.matches_found = 0
        self._scale: Optional[Dict[str, Tuple[float, float]]] = None

    # --- observation -----------------------------------------------------

    def observe_entry(self, wallet: str, *, entry_age_s: float,
                      size_sol: Optional[float] = None,
                      deployer: str = "", at: Optional[float] = None) -> None:
        """One wallet's first buy into one launch."""
        key = str(wallet or "")
        if not key:
            return
        moment = float(at or time.time())
        behaviour = self.behaviours.get(key)
        if behaviour is None:
            behaviour = WalletBehaviour(wallet=key, first_seen=moment)
            self.behaviours[key] = behaviour
        behaviour.entry_ages_s.append(max(0.0, float(entry_age_s)))
        if size_sol is not None:
            behaviour.sizes_sol.append(float(size_sol))
        behaviour.buys += 1
        hour = time.gmtime(moment).tm_hour
        behaviour.hours[hour] = behaviour.hours.get(hour, 0) + 1
        if deployer:
            behaviour.deployers[deployer] = behaviour.deployers.get(deployer, 0) + 1
        behaviour.last_seen = max(behaviour.last_seen, moment)
        # Bounded: a wallet with ten thousand entries has the same median as
        # one with five hundred.
        if len(behaviour.entry_ages_s) > 512:
            behaviour.entry_ages_s = behaviour.entry_ages_s[-512:]
            behaviour.sizes_sol = behaviour.sizes_sol[-512:]
        self._scale = None

    def observe_exit(self, wallet: str, hold_s: float) -> None:
        behaviour = self.behaviours.get(str(wallet or ""))
        if behaviour is None:
            return
        behaviour.sells += 1
        behaviour.holds_s.append(max(0.0, float(hold_s)))
        if len(behaviour.holds_s) > 512:
            behaviour.holds_s = behaviour.holds_s[-512:]
        self._scale = None

    # --- standardisation -------------------------------------------------

    def _scales(self) -> Dict[str, Tuple[float, float]]:
        """Mean and spread per feature, over every measurable wallet.

        Standardising against the POPULATION rather than against a fixed
        scale, because "buys early" only means anything relative to how
        early everyone else buys, and that changes with the market.
        """
        if self._scale is not None:
            return self._scale
        columns: Dict[str, List[float]] = {name: [] for name in FEATURES}
        for behaviour in self.behaviours.values():
            values = behaviour.features()
            if values is None:
                continue
            for name in FEATURES:
                columns[name].append(values[name])
        scale: Dict[str, Tuple[float, float]] = {}
        for name, values in columns.items():
            if len(values) < 2:
                scale[name] = (0.0, 1.0)
                continue
            mean = statistics.fmean(values)
            spread = statistics.pstdev(values) or 1.0
            scale[name] = (mean, spread)
        self._scale = scale
        return scale

    def _standardise(self, values: Dict[str, float]) -> List[float]:
        scale = self._scales()
        out = []
        for name in FEATURES:
            mean, spread = scale[name]
            out.append((values[name] - mean) / spread)
        return out

    # --- clusters --------------------------------------------------------

    def define_cluster(self, name: str, wallets: Sequence[str],
                       forward_elogw: Optional[float] = None) -> bool:
        """Name a set of wallets whose forward value the desk has measured.

        `forward_elogw` is the point of the exercise. A cluster is only worth
        matching against if following it was worth something, and the number
        that says so is E[dlogW] AFTER the desk's own delay, fees and
        crowding -- never a headline PnL, which says what THEY made and
        nothing about what we would.
        """
        measurable = [wallet for wallet in wallets
                      if (self.behaviours.get(wallet) is not None
                          and self.behaviours[wallet].measurable)]
        if len(measurable) < MIN_CLUSTER_WALLETS:
            return False
        vectors = [self._standardise(self.behaviours[wallet].features())
                   for wallet in measurable]
        centroid = [statistics.fmean(column) for column in zip(*vectors)]
        # The cluster's own spread, so a tight cluster demands a close match
        # and a loose one does not pretend to.
        spread = [max(0.25, statistics.pstdev(column)) if len(column) > 1 else 1.0
                  for column in zip(*vectors)]
        self.clusters[str(name)] = {
            "members": list(measurable),
            "centroid": centroid,
            "spread": spread,
            "forward_elogw": forward_elogw,
            "defined_at": time.time(),
        }
        return True

    def match(self, wallet: str) -> Dict[str, Any]:
        """Which measured cluster this wallet behaves like, if any.

        Returns DATA_BLOCKED rather than a guess when the wallet has too
        little history to have a signature -- which is most of them, and is
        the honest answer.
        """
        self.matches_served += 1
        behaviour = self.behaviours.get(str(wallet or ""))
        values = behaviour.features() if behaviour is not None else None
        if values is None:
            return {
                "status": "DATA_BLOCKED",
                "reason": (f"{behaviour.entries if behaviour else 0} entries "
                           f"observed, {self.min_entries} needed for a "
                           "signature"),
            }
        if not self.clusters:
            return {"status": "DATA_BLOCKED",
                    "reason": "no measured cluster to compare against"}
        vector = self._standardise(values)
        best_name, best_distance = "", math.inf
        for name, cluster in self.clusters.items():
            if wallet in cluster["members"]:
                continue  # matching a wallet to its own cluster proves nothing
            distance = math.sqrt(sum(
                ((value - centre) / spread) ** 2
                for value, centre, spread in zip(vector, cluster["centroid"],
                                                 cluster["spread"])))
            # Per-dimension, so adding a feature does not silently raise the
            # bar for every existing cluster.
            distance /= math.sqrt(len(FEATURES))
            if distance < best_distance:
                best_name, best_distance = name, distance
        if not best_name:
            return {"status": "DATA_BLOCKED",
                    "reason": "no cluster this wallet is not already in"}
        matched = best_distance <= self.match_distance
        if matched:
            self.matches_found += 1
        cluster = self.clusters[best_name]
        return {
            "status": "OK",
            "matched": matched,
            "cluster": best_name,
            "distance": round(best_distance, 3),
            "threshold": self.match_distance,
            "cluster_wallets": len(cluster["members"]),
            "cluster_forward_elogw": cluster["forward_elogw"],
            "entries_observed": behaviour.entries,
            # Said in the result, not just the docstring: this is a
            # behavioural resemblance, and a resemblance is not an identity.
            "means": ("this wallet's behaviour is drawn from the same "
                      "distribution as a cluster whose forward value was "
                      "measured; it is NOT a claim that they are the same "
                      "actor"),
            "provenance": "BEHAVIOURAL_PRIOR",
        }

    def report(self) -> Dict[str, Any]:
        measurable = [b for b in self.behaviours.values() if b.measurable]
        priced = [name for name, cluster in self.clusters.items()
                  if cluster.get("forward_elogw") is not None]
        return {
            "schema": WALLET_SIGNATURE_SCHEMA_VERSION,
            "status": "OK" if (measurable and self.clusters) else "DATA_BLOCKED",
            "wallets_observed": len(self.behaviours),
            "wallets_with_a_signature": len(measurable),
            "min_entries": self.min_entries,
            "clusters": len(self.clusters),
            "clusters_with_measured_value": len(priced),
            "matches_served": self.matches_served,
            "matches_found": self.matches_found,
            "features": list(FEATURES),
            "detail": ("what a wallet DOES, so a rotated address is not "
                       "automatically unknown; a behavioural prior on an "
                       "unknown wallet, worth less than a measured score and "
                       "more than nothing, and never a claim about identity"),
        }

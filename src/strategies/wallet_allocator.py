"""Wallets as alpha sources with their own capital budgets, not a fixed 1 SOL.

"Wallet A bought, buy 1 SOL" makes every tracked wallet equally right forever.
The wallets worth following are a portfolio: hundreds of strategies with
different measured value, different decay, and -- crucially -- different
amounts of evidence that they still work. A guru wallet that carried the book
for a month and quietly stopped working looks, to a fixed-size copier,
identical to one that never stopped.

So allocation is a function of AGREEMENT between two independent estimates:
what the historical gauntlet said following this wallet was worth, and what
following it has actually paid since. A wallet earns capital while those two
agree and loses it while they diverge, and the divergence is measured against
the historical LOWER BOUND rather than its mean -- the bound is what the
gauntlet was willing to stand behind, so falling below it is the wallet
failing its own test rather than ordinary noise.

Four properties that are deliberate rather than incidental:

**Nothing is allocated on history alone.** A wallet with a survivor verdict
and no live evidence gets a probation weight, and probation is small because
here, unlike watching, being wrong costs capital rather than a subscription.
The weight it can eventually reach is set by its measured value; the weight it
has now is set by how much live evidence agrees.

**Decay is faster than growth.** An edge that has stopped working costs money
every day it is still funded, and an edge that is working is still there next
week. The asymmetry is the whole risk posture in one constant.

**Clusters share a budget.** Ten addresses belonging to one operator are one
alpha source, and allocating to each of them separately is how a book ends up
with ten times its intended exposure to one person. A wallet's weight is
capped by what remains of its cluster's budget.

**A wallet with no verdict gets nothing.** Not a small default -- nothing.
Unmeasured is not a weak signal, it is an absence of one, and the difference
between those two is the difference between a book that concentrates on what
it has measured and one that spreads thinly across what it has not.
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import (Any, Callable, Deque, Dict, Iterable, List, Optional,
                    Sequence, Tuple)

logger = logging.getLogger(__name__)

WALLET_ALLOCATOR_SCHEMA_VERSION = "v1"

#: Weight a survivor wallet carries before any live evidence exists. Small on
#: purpose: it buys the observations that decide whether it deserves more.
PROBATION_WEIGHT = 0.05

#: Live followed outcomes below which a wallet is still on probation, however
#: good its history looked.
MIN_LIVE_SAMPLES = 10

#: Multiplicative step toward the earned weight per agreeing observation.
GROWTH_RATE = 0.05

#: Multiplicative step away from it per disagreeing observation. Four times
#: the growth rate: an edge that has stopped working costs money every day it
#: stays funded, and one that is working will still be there next week.
DECAY_RATE = 0.20

#: Live outcomes kept per wallet. A wallet that worked last year should not be
#: defended by last year's trades.
LIVE_HISTORY = 200

#: Share of the copy book any single cluster may hold.
DEFAULT_CLUSTER_CAP = 0.35

#: Lower bound at which a wallet reaches its full share of the book. Per
#: followed trade in log space; 0.10 is already a strong wallet.
FULL_WEIGHT_LOWER_BOUND = 0.10


@dataclass
class WalletBudget:
    """One wallet's standing in the copy book."""

    wallet: str
    cluster: str = ""
    #: What the historical gauntlet stood behind.
    historical_lower_bound: Optional[float] = None
    historical_verdict: str = ""
    #: The ceiling this wallet's history entitles it to.
    earned_weight: float = 0.0
    #: What it currently holds, after live agreement or divergence.
    weight: float = 0.0
    live_samples: int = 0
    live_mean: Optional[float] = None
    agreeing: int = 0
    diverging: int = 0
    last_update: float = 0.0
    status: str = "DATA_BLOCKED"
    detail: str = ""

    @property
    def on_probation(self) -> bool:
        return self.status == "OK" and self.live_samples < MIN_LIVE_SAMPLES

    def to_dict(self) -> Dict[str, Any]:
        return {"wallet": self.wallet, "cluster": self.cluster,
                "status": self.status, "verdict": self.historical_verdict,
                "historical_lower_bound": self.historical_lower_bound,
                "earned_weight": round(self.earned_weight, 5),
                "weight": round(self.weight, 5),
                "live_samples": self.live_samples,
                "live_mean": self.live_mean,
                "agreeing": self.agreeing, "diverging": self.diverging,
                "on_probation": self.on_probation, "detail": self.detail}


class WalletAllocator:
    """Holds a weight per wallet and moves it on evidence, not on opinion."""

    def __init__(self, *,
                 cluster_provider: Optional[Callable[[str], Optional[str]]] = None,
                 probation_weight: float = PROBATION_WEIGHT,
                 min_live_samples: int = MIN_LIVE_SAMPLES,
                 growth_rate: float = GROWTH_RATE,
                 decay_rate: float = DECAY_RATE,
                 cluster_cap: float = DEFAULT_CLUSTER_CAP,
                 full_weight_lower_bound: float = FULL_WEIGHT_LOWER_BOUND,
                 live_history: int = LIVE_HISTORY):
        self.cluster_provider = cluster_provider
        self.probation_weight = float(probation_weight)
        self.min_live_samples = int(min_live_samples)
        self.growth_rate = float(growth_rate)
        self.decay_rate = float(decay_rate)
        self.cluster_cap = float(cluster_cap)
        self.full_weight_lower_bound = float(full_weight_lower_bound)
        self._budgets: Dict[str, WalletBudget] = {}
        self._live: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=int(live_history)))

    # -- registration -----------------------------------------------------

    def _cluster_of(self, wallet: str) -> str:
        """Which alpha source this address belongs to.

        Its own address when nothing can tell. That is the permissive answer
        here and it is the right one: inventing a cluster would merge two
        genuinely independent wallets into one budget and halve the book's
        diversification on no evidence. The cap still bounds each of them.
        """
        provider = self.cluster_provider
        if not callable(provider):
            return wallet
        try:
            found = provider(wallet)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("cluster provider failed for %s: %s", wallet, exc)
            return wallet
        return str(found) if found else wallet

    def register(self, wallet: str, *, lower_bound: Optional[float],
                 verdict: str, refreeze: bool = False) -> WalletBudget:
        """Record what the gauntlet concluded about following this wallet.

        A verdict other than SURVIVOR, or a bound that is not positive, earns
        a ceiling of zero. It is registered anyway so the book can say it
        considered the wallet and declined, which is different from never
        having heard of it.

        The bound FREEZES once live evidence starts arriving, and that is the
        whole reason the agreement test means anything. Historical value and
        live performance have to be two estimates from disjoint data; letting
        a re-registration fold the live outcomes back into the bound would
        make the wallet its own benchmark, and a wallet compared against
        itself always agrees. ``refreeze`` exists for a deliberate
        re-baselining and is never the default.
        """
        existing = self._budgets.get(wallet)
        if existing is not None and existing.live_samples > 0 and not refreeze:
            return existing
        budget = existing or WalletBudget(wallet=wallet)
        budget.wallet = wallet
        budget.cluster = self._cluster_of(wallet)
        budget.historical_lower_bound = lower_bound
        budget.historical_verdict = str(verdict)
        budget.last_update = time.time()
        if verdict != "SURVIVOR" or lower_bound is None or lower_bound <= 0:
            budget.earned_weight = 0.0
            budget.weight = 0.0
            budget.status = "DATA_BLOCKED" if lower_bound is None else "OK"
            budget.detail = (f"verdict {verdict!r} earns no allocation"
                             if lower_bound is not None else
                             "no measured follow value")
        else:
            saturated = min(1.0, float(lower_bound) / self.full_weight_lower_bound)
            budget.earned_weight = float(saturated)
            budget.status = "OK"
            budget.detail = "earned on historical copyability"
            if budget.weight <= 0.0:
                # Entering the book: probation, never the earned ceiling. The
                # ceiling is what the history entitles it to; the weight is
                # what live evidence has so far agreed to.
                budget.weight = min(self.probation_weight, budget.earned_weight)
        self._budgets[wallet] = budget
        return budget

    # -- live evidence ----------------------------------------------------

    def record_live(self, wallet: str, log_return: float) -> Optional[WalletBudget]:
        """One followed outcome we actually took. Moves the weight.

        Agreement is measured against the historical LOWER BOUND, not the
        mean: the bound is what the gauntlet was prepared to stand behind, so
        a live result beneath it is the wallet failing its own test rather
        than ordinary dispersion around a point estimate.
        """
        budget = self._budgets.get(wallet)
        if budget is None or budget.status != "OK":
            return None
        values = self._live[wallet]
        values.append(float(log_return))
        budget.live_samples = len(values)
        budget.live_mean = float(sum(values) / len(values))
        budget.last_update = time.time()

        floor = budget.historical_lower_bound or 0.0
        if budget.live_mean >= floor:
            budget.agreeing += 1
            if budget.live_samples >= self.min_live_samples:
                budget.weight = min(
                    budget.earned_weight,
                    budget.weight + self.growth_rate * budget.earned_weight)
                budget.detail = "live performance agrees with its history"
            else:
                budget.detail = (
                    f"{budget.live_samples}/{self.min_live_samples} live "
                    "outcomes; still on probation")
        else:
            budget.diverging += 1
            budget.weight = max(0.0, budget.weight * (1.0 - self.decay_rate))
            budget.detail = (
                f"live mean {budget.live_mean:.4f} below the historical lower "
                f"bound {floor:.4f}; decaying")
        return budget

    # -- reading ----------------------------------------------------------

    def weight(self, wallet: str) -> float:
        """This wallet's share of the copy book, after its cluster's cap.

        Zero for a wallet nobody has measured. Not a small number -- zero.
        Unmeasured is an absence of signal, not a weak one.
        """
        budget = self._budgets.get(wallet)
        if budget is None or budget.status != "OK" or budget.weight <= 0:
            return 0.0
        return min(budget.weight, self._cluster_headroom(budget))

    def _cluster_headroom(self, budget: WalletBudget) -> float:
        """What is left of this wallet's cluster budget, excluding itself."""
        siblings = sum(
            other.weight for other in self._budgets.values()
            if other.cluster == budget.cluster and other.wallet != budget.wallet
            and other.status == "OK")
        return max(0.0, self.cluster_cap - siblings)

    def allocate(self, budget_usd: float,
                 wallets: Optional[Iterable[str]] = None) -> Dict[str, float]:
        """Split a copy budget across wallets in proportion to their weights.

        Returns an empty allocation rather than an even split when no wallet
        has a weight. "Nothing has earned capital" is an answer, and spreading
        the budget anyway to avoid returning nothing is how a book funds the
        wallets it has specifically failed to validate.
        """
        names = list(wallets) if wallets is not None else list(self._budgets)
        weights = {name: self.weight(name) for name in names}
        weights = {name: value for name, value in weights.items() if value > 0}
        total = sum(weights.values())
        if total <= 0 or budget_usd <= 0:
            return {}
        return {name: float(budget_usd * value / total)
                for name, value in weights.items()}

    def report(self) -> Dict[str, Any]:
        active = [b for b in self._budgets.values() if self.weight(b.wallet) > 0]
        probation = [b for b in active if b.on_probation]
        decaying = [b for b in self._budgets.values()
                    if b.diverging > b.agreeing and b.status == "OK"]
        by_cluster: Dict[str, float] = defaultdict(float)
        for budget in active:
            by_cluster[budget.cluster] += self.weight(budget.wallet)
        return {
            "schema": WALLET_ALLOCATOR_SCHEMA_VERSION,
            "status": "OK" if active else "DATA_BLOCKED",
            "registered": len(self._budgets),
            "funded": len(active),
            "on_probation": len(probation),
            "decaying": len(decaying),
            "clusters": dict(sorted(by_cluster.items())),
            "concentration": (max(by_cluster.values()) / sum(by_cluster.values())
                              if by_cluster else None),
            "by_wallet": [b.to_dict() for b in sorted(
                self._budgets.values(), key=lambda item: -item.weight)],
            "detail": ("" if active else
                       "no wallet has both a survivor verdict and a positive "
                       "weight; the copy book is funding nothing"),
        }

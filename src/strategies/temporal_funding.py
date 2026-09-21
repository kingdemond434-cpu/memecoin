"""Coordination that an exchange withdrawal was used to hide.

``FunderAncestry`` follows funding edges backwards and collapses wallets that
share an ancestor. It is defeated, cheaply and deliberately, by routing
through a centralised exchange: A and B both withdraw from Binance, the
on-chain graph shows a hot wallet that funds hundreds of thousands of
unrelated people, and the edge carries no information at all. Every
independence model that stops at the funding graph reads the pair as arm's
length, which is exactly what the person who routed through the exchange paid
the withdrawal fee for.

What survives the exchange is TIME and SHAPE. The hot wallet funds everyone,
but it does not fund everyone within the same eleven seconds, in similar
amounts, into wallets that were all created that morning and all bought the
same launch minutes later. That conjunction is rare by chance and cheap to
compute, and it is the residue the exchange could not launder.

So this scores a cluster hypothesis rather than an identity:

    P(these wallets are one economic actor | timing, amount, age, target)

and the output is EVIDENCE, expressed as an independence discount the actor
graph can apply. It is never an assertion about who anybody is. A cluster is
"these five wallets behaved as one" -- a claim about observable coordination,
falsifiable from public data -- and not "this is person X", which the data
cannot support and which nothing here attempts.

The base rate does the important work. A busy exchange hot wallet emits
thousands of withdrawals an hour, so "two withdrawals in the same minute" is
not evidence of anything; the module measures the hot wallet's OWN emission
rate and asks how surprising the observed tightness is against it. Without
that denominator every wallet on Solana is in a cluster with every other.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.strategies.surprisal import poisson_surprisal

logger = logging.getLogger(__name__)

TEMPORAL_FUNDING_SCHEMA_VERSION = "v1"

#: Withdrawals further apart than this are not the same batch, whatever else
#: they have in common. Wide enough for a human clicking through an exchange
#: UI five times, tight enough to exclude coincidence at ordinary rates.
DEFAULT_WINDOW_S = 120.0

#: Two amounts within this relative distance are "similar". Exchange
#: withdrawals of coordinated wallets are usually equal or round.
AMOUNT_TOLERANCE = 0.15

#: A cluster needs at least this many wallets. Two wallets withdrawing
#: together is a coincidence that happens constantly at exchange scale.
MIN_CLUSTER_SIZE = 3

#: Clusters below this surprisal are not reported. Natural log of the odds
#: against the timing arising from the hot wallet's own rate; 3.0 is roughly
#: 20:1 against.
MIN_SURPRISAL = 3.0

#: The strongest independence discount a temporal cluster may impose. Never
#: 1.0: this is circumstantial evidence about coordination, and a model that
#: can zero a wallet's independence on circumstantial evidence will eventually
#: zero an innocent one.
MAX_DISCOUNT = 0.7


@dataclass
class Withdrawal:
    """One funding event out of a known exchange or bridge hot wallet."""

    wallet: str
    source: str
    timestamp: float
    amount_sol: float
    #: When the receiving wallet was first seen on chain, if known.
    wallet_first_seen: Optional[float] = None


@dataclass
class TemporalCluster:
    status: str
    source: str = ""
    wallets: List[str] = field(default_factory=list)
    span_seconds: float = 0.0
    median_amount_sol: float = 0.0
    amount_agreement: float = 0.0
    fresh_wallets: int = 0
    surprisal: float = 0.0
    detail: str = ""

    @property
    def size(self) -> int:
        return len(self.wallets)

    @property
    def discount(self) -> float:
        """How much independence to remove from each member, in [0, MAX].

        Grows with surprisal and saturates. A cluster that is merely unlikely
        moves the number a little; one that is overwhelming still cannot take
        a wallet's independence to zero.
        """
        if self.status != "OK":
            return 0.0
        strength = 1.0 - math.exp(-max(0.0, self.surprisal - MIN_SURPRISAL) / 6.0)
        shaped = 0.5 + 0.5 * self.amount_agreement
        return float(min(MAX_DISCOUNT, MAX_DISCOUNT * strength * shaped))


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


def _amount_agreement(amounts: Sequence[float]) -> float:
    """Fraction of amounts within tolerance of the group's median."""
    if not amounts:
        return 0.0
    centre = _median(amounts)
    if centre <= 0:
        return 0.0
    close = sum(1 for value in amounts
                if abs(value - centre) / centre <= AMOUNT_TOLERANCE)
    return float(close / len(amounts))



def find_clusters(withdrawals: Sequence[Withdrawal],
                  target_buyers: Optional[Sequence[str]] = None,
                  window_s: float = DEFAULT_WINDOW_S,
                  source_rates: Optional[Dict[str, float]] = None,
                  ) -> List[TemporalCluster]:
    """Groups of wallets funded from one source, too tightly to be chance.

    `source_rates` gives each hot wallet's measured emission rate in
    withdrawals per second. Without it, no cluster can be scored: the rate is
    the denominator, and inventing one would turn every busy exchange into a
    conspiracy. A source with no measured rate is skipped and said so.
    """
    rates = dict(source_rates or {})
    by_source: Dict[str, List[Withdrawal]] = {}
    for item in withdrawals:
        by_source.setdefault(str(item.source), []).append(item)

    targets = set(target_buyers or ())
    clusters: List[TemporalCluster] = []
    for source, items in sorted(by_source.items()):
        rate = rates.get(source)
        if rate is None or rate <= 0:
            clusters.append(TemporalCluster(
                status="DATA_BLOCKED", source=source,
                detail=(f"no measured emission rate for {source}; without it "
                        "any group of withdrawals looks coordinated")))
            continue
        items.sort(key=lambda entry: float(entry.timestamp))
        # Sliding window over arrivals from this one source.
        start = 0
        for end in range(len(items)):
            while float(items[end].timestamp) - float(items[start].timestamp) > window_s:
                start += 1
            group = items[start:end + 1]
            if len(group) < MIN_CLUSTER_SIZE:
                continue
            wallets = [entry.wallet for entry in group]
            if targets and not any(wallet in targets for wallet in wallets):
                # Coordination that never touched the launch under decision is
                # somebody else's cluster. Real, and not this token's problem.
                continue
            span = float(group[-1].timestamp) - float(group[0].timestamp)
            surprisal = poisson_surprisal(len(group), max(span, 1e-6), rate)
            if surprisal < MIN_SURPRISAL:
                continue
            amounts = [float(entry.amount_sol) for entry in group]
            fresh = sum(1 for entry in group
                        if entry.wallet_first_seen is not None
                        and float(entry.timestamp) - float(entry.wallet_first_seen) <= 86400.0)
            clusters.append(TemporalCluster(
                status="OK", source=source, wallets=sorted(set(wallets)),
                span_seconds=span, median_amount_sol=_median(amounts),
                amount_agreement=_amount_agreement(amounts),
                fresh_wallets=fresh, surprisal=surprisal,
                detail=(f"{len(set(wallets))} wallets funded from {source} within "
                        f"{span:.1f}s against a measured {rate:.3g}/s rate")))
    return _deduplicate(clusters)


def _deduplicate(clusters: Sequence[TemporalCluster]) -> List[TemporalCluster]:
    """Keep only maximal wallet sets; the sliding window nests by construction.

    Advancing the window one arrival at a time emits every prefix of a batch
    -- {w0,w1,w2}, {w0..w3}, {w0..w4} -- and each is a different tuple, so
    keying on the exact set keeps all of them. Six coordinated wallets would
    then be reported as four overlapping clusters describing one event, and
    anything counting clusters would count it four times.

    A set contained by another is dropped in favour of its superset: the
    larger group is the same coordination, seen whole.
    """
    best: Dict[Tuple[str, ...], TemporalCluster] = {}
    blocked: List[TemporalCluster] = []
    for cluster in clusters:
        if cluster.status != "OK":
            blocked.append(cluster)
            continue
        key = tuple(cluster.wallets)
        current = best.get(key)
        if current is None or cluster.surprisal > current.surprisal:
            best[key] = cluster
    kept: List[TemporalCluster] = []
    for cluster in sorted(best.values(), key=lambda item: -len(item.wallets)):
        members = set(cluster.wallets)
        if any(members <= set(other.wallets) and other.source == cluster.source
               for other in kept):
            continue
        kept.append(cluster)
    kept.sort(key=lambda item: -item.surprisal)
    return kept + blocked


def independence_discounts(clusters: Sequence[TemporalCluster]) -> Dict[str, float]:
    """wallet -> multiplier in (0, 1] to apply to its independence score.

    A wallet in two clusters takes the stronger discount rather than the
    product: two pieces of circumstantial evidence about the same underlying
    coordination are not independent observations, and multiplying them would
    double-count one fact.
    """
    discounts: Dict[str, float] = {}
    for cluster in clusters:
        if cluster.status != "OK":
            continue
        multiplier = 1.0 - cluster.discount
        for wallet in cluster.wallets:
            discounts[wallet] = min(discounts.get(wallet, 1.0), multiplier)
    return discounts


def measure_source_rate(withdrawals: Sequence[Withdrawal], source: str,
                        span_s: float) -> Optional[float]:
    """The hot wallet's own emission rate, or None if too little was seen.

    None rather than a small number: a rate estimated from four observations
    over a minute is not a base rate, and using it would make every group
    surprising.
    """
    if span_s <= 0:
        return None
    seen = [item for item in withdrawals if str(item.source) == source]
    if len(seen) < 30:
        return None
    return float(len(seen) / span_s)

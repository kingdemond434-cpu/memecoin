"""Counting economic actors instead of counting wallets.

"Twelve wallets bought" is close to meaningless. Twelve wallets funded by one
address, entering in the same slot, in the order they always enter in, are one
actor buying twelve times -- which is the cheapest signal in the market to
manufacture and the one a naive flow model weights most heavily. Meanwhile
three wallets with no funding relationship, no history of co-occurrence and
genuinely good independent records are far stronger evidence than the twelve,
and a counter of buyers ranks them below.

Three pieces, each attacking one way the count lies:

``WalletIndependence`` learns, from observed history, how much of each wallet's
signal is its own. Two wallets that repeatedly enter the same tokens seconds
apart in a consistent order are one source of evidence, not two. Evidence is
then aggregated as sum(skill * capital * independence), which is bounded above
by the naive sum by construction -- independence can only ever discount.

``BuyerDNA`` fingerprints the first N buyers as an ORDERED sequence.
"bad wallet, then good wallet, then good independent wallet" and "ten linked
wallets, then retail" can share every aggregate statistic and mean opposite
things. Matching is nearest-neighbour against labelled history, and refuses to
answer at all when the corpus is too small: a 1-NN against three launches is
not a prior, it is a coincidence with a confidence interval.

``SwarmPredictor`` asks the forward question -- will more independent skilled
wallets arrive in the next few seconds -- rather than the backward one, which
is what copying a famous wallet after its fill amounts to.

Everything here is derived from public chain observations. Nothing infers
identity from anything other than transactions and funding that are already
public.
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ACTOR_GRAPH_SCHEMA_VERSION = "v1"

# Two entries closer together than this, repeatedly, is following rather than
# agreeing. Wide enough to catch a copy bot, tight enough that two people
# reacting to the same public event are not called one actor.
FOLLOW_WINDOW_SECONDS = 3.0
# Below this many shared launches there is no evidence about a pair either
# way, and a ratio computed from two observations is noise with a decimal point.
MIN_PAIR_OBSERVATIONS = 4


@dataclass
class Entry:
    """One wallet entering one token at one time."""

    token: str
    wallet: str
    timestamp: float
    skill: Optional[float] = None
    capital_usd: Optional[float] = None


@dataclass
class IndependenceReport:
    status: str
    scores: Dict[str, float] = field(default_factory=dict)
    followers: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)
    observed_pairs: int = 0
    detail: str = ""


class WalletIndependence:
    """How much of each wallet's signal is genuinely its own.

    Independence is learned from repeated co-occurrence and lead-lag, never
    asserted from cluster membership alone. A wallet with no history is not
    assumed independent: it is unmeasured, which is a different thing, and the
    aggregator is told which.
    """

    def __init__(self, follow_window: float = FOLLOW_WINDOW_SECONDS,
                 min_observations: int = MIN_PAIR_OBSERVATIONS):
        self.follow_window = follow_window
        self.min_observations = max(2, min_observations)
        self._entries: Dict[str, List[Entry]] = defaultdict(list)

    def record_entries(self, entries: Iterable[Entry]) -> None:
        for entry in entries:
            self._entries[entry.token].append(entry)

    def _pair_statistics(self) -> Tuple[Dict[Tuple[str, str], int], Dict[Tuple[str, str], int]]:
        """(shared launches, follow events) for each ordered wallet pair."""
        shared: Dict[Tuple[str, str], int] = defaultdict(int)
        follows: Dict[Tuple[str, str], int] = defaultdict(int)
        for entries in self._entries.values():
            ordered = sorted(entries, key=lambda item: item.timestamp)
            # One entry per wallet per token: a wallet buying twice is not two
            # pieces of evidence about anyone else.
            first: Dict[str, Entry] = {}
            for entry in ordered:
                first.setdefault(entry.wallet, entry)
            wallets = list(first)
            for follower in wallets:
                for leader in wallets:
                    if follower == leader:
                        continue
                    shared[(follower, leader)] += 1
                    delay = first[follower].timestamp - first[leader].timestamp
                    if 0 < delay <= self.follow_window:
                        follows[(follower, leader)] += 1
        return shared, follows

    def compute(self) -> IndependenceReport:
        shared, follows = self._pair_statistics()
        measurable = {pair: count for pair, count in shared.items()
                      if count >= self.min_observations}
        if not measurable:
            return IndependenceReport(
                status="DATA_BLOCKED", observed_pairs=0,
                detail=(f"no wallet pair shares {self.min_observations} launches; "
                        "a follow ratio from fewer is noise with a decimal point"),
            )

        followers: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        worst: Dict[str, float] = {}
        for (follower, leader), count in measurable.items():
            ratio = follows.get((follower, leader), 0) / count
            if ratio > 0:
                followers[follower].append((leader, ratio))
            worst[follower] = max(worst.get(follower, 0.0), ratio)

        scores = {wallet: float(np.clip(1.0 - ratio, 0.0, 1.0))
                  for wallet, ratio in worst.items()}
        for wallet, leaders in followers.items():
            leaders.sort(key=lambda item: item[1], reverse=True)
        return IndependenceReport(status="OK", scores=scores, followers=dict(followers),
                                  observed_pairs=len(measurable),
                                  detail=f"{len(measurable)} measurable pairs")

    def score_of(self, wallet: str, report: IndependenceReport) -> Optional[float]:
        """Independence for one wallet, or None when it was never measurable."""
        if report.status != "OK":
            return None
        return report.scores.get(wallet)


@dataclass
class SmartFlow:
    status: str
    evidence: float = 0.0
    naive_evidence: float = 0.0
    measured_wallets: int = 0
    unmeasured_wallets: int = 0
    # How many wallets funding ancestry lowered below what behaviour said.
    # Counted, because a smart-flow number that quietly halved deserves to say
    # what halved it.
    ancestry_compressed: int = 0
    detail: str = ""

    @property
    def discount(self) -> Optional[float]:
        """Share of the naive signal that survived collapsing wallets to actors."""
        if self.naive_evidence <= 0:
            return None
        return float(self.evidence / self.naive_evidence)


def aggregate_smart_flow(
    entries: Sequence[Entry],
    report: IndependenceReport,
    unmeasured_independence: float = 0.5,
    ancestry: Optional[Any] = None,
) -> SmartFlow:
    """sum(skill * capital * independence) over one token's skilled buyers.

    ``unmeasured_independence`` is the weight given to a wallet the graph has
    never seen enough of. It is deliberately not 1.0: a brand-new wallet
    appearing alongside a known cluster is exactly what a Sybil looks like, so
    treating unknown as fully independent would make the cheapest wallets the
    most persuasive. It is also not 0.0, which would make the metric blind to
    genuinely new participants.
    """
    scored = [entry for entry in entries
              if entry.skill is not None and entry.capital_usd is not None]
    if not scored:
        return SmartFlow(status="DATA_BLOCKED",
                         detail="no buyer carried both a skill score and a size")

    # Funding ancestry, when it has been traced, compresses wallets that share
    # a funder into the actor whose capital they are. This is the case
    # behavioural independence structurally cannot see: wallets funded and
    # deployed for ONE launch have no co-occurrence history, so every one of
    # them reads as an unmeasured independent participant -- and a Sybil built
    # for this launch is invisible at exactly the moment it is used.
    compression = (getattr(ancestry, "compression", None) or {}
                   if getattr(ancestry, "status", "") == "OK" else {})
    evidence = 0.0
    naive = 0.0
    measured = unmeasured = 0
    compressed = 0
    for entry in scored:
        weight = float(entry.skill) * float(entry.capital_usd)
        naive += weight
        independence = report.scores.get(entry.wallet) if report.status == "OK" else None
        if independence is None:
            unmeasured += 1
            independence = unmeasured_independence
        else:
            measured += 1
        ancestral = compression.get(entry.wallet)
        if ancestral is not None and ancestral < independence:
            # Ancestry only ever lowers it. A wallet whose funding is
            # untraceable does not earn independence it has not shown.
            compressed += 1
            independence = float(ancestral)
        evidence += weight * independence
    detail = f"{len(scored)} skilled buyers, {unmeasured} unmeasured"
    if compressed:
        detail += f", {compressed} compressed by funding ancestry"
    return SmartFlow(status="OK", evidence=float(evidence), naive_evidence=float(naive),
                     measured_wallets=measured, unmeasured_wallets=unmeasured,
                     ancestry_compressed=compressed, detail=detail)


@dataclass
class BuyerFingerprint:
    """The first N buyers as an ordered sequence of ECONOMIC ACTORS.

    Skill, independence, creator linkage and size say what each wallet was.
    The funding fields say what each wallet was RELATIVE TO THE ONES BEFORE
    IT -- whether it shares a funder with an earlier buyer, and how close that
    tie is. That is the difference between "ten buyers" and "one operator
    entering ten times", and it is not expressible as a per-wallet attribute.
    """

    token: str
    skills: List[float]
    independence: List[float]
    creator_linked: List[bool]
    sizes: List[float]
    # -1 marks unmeasured throughout, as everywhere else here.
    shares_funder_with_prior: List[float] = field(default_factory=list)
    common_funder_depth: List[float] = field(default_factory=list)
    creator_funded: List[float] = field(default_factory=list)

    @property
    def depth(self) -> int:
        return len(self.skills)

    def vector(self, depth: int) -> np.ndarray:
        """Fixed-length ordered encoding, padded with a neutral marker.

        Padding is -1 rather than 0 so "this launch had only four buyers" and
        "the fifth buyer scored zero" are different states. Collapsing them is
        how a thin launch starts matching a bad one.
        """
        def take(values, cast=float):
            padded = [cast(item) for item in values[:depth]]
            return padded + [-1.0] * (depth - len(padded))
        return np.asarray(
            take(self.skills) + take(self.independence)
            + take(self.creator_linked, lambda item: 1.0 if item else 0.0)
            + take(self.sizes) + take(self.shares_funder_with_prior)
            + take(self.common_funder_depth) + take(self.creator_funded), dtype=float)


def build_fingerprint(
    token: str,
    entries: Sequence[Entry],
    report: IndependenceReport,
    creator_linked: Optional[Dict[str, bool]] = None,
    depth: int = 25,
    funding_features: Optional[Sequence[Dict[str, Any]]] = None,
) -> BuyerFingerprint:
    """Order the first ``depth`` buyers and record what each one was.

    ``funding_features`` comes from FunderAncestry.buyer_features over the SAME
    ordered wallets. Absent, the funding columns read -1 everywhere, which is
    "unmeasured" rather than "no shared funders" -- the distinction matters,
    because an untracked graph must not certify a cluster as clean.
    """
    linked = creator_linked or {}
    ordered = sorted(entries, key=lambda item: item.timestamp)
    seen: set = set()
    unique: List[Entry] = []
    for entry in ordered:
        if entry.wallet in seen:
            continue
        seen.add(entry.wallet)
        unique.append(entry)
        if len(unique) >= depth:
            break
    by_wallet = {str(row.get("wallet", "")): row for row in (funding_features or ())}

    def funding(entry: Entry, key: str, missing: float = -1.0) -> float:
        row = by_wallet.get(entry.wallet)
        if row is None or not row.get("traced"):
            return missing
        value = row.get(key)
        if value is None:
            return missing
        return 1.0 if value is True else 0.0 if value is False else float(value)

    return BuyerFingerprint(
        token=token,
        skills=[float(entry.skill) if entry.skill is not None else -1.0 for entry in unique],
        independence=[report.scores.get(entry.wallet, -1.0) for entry in unique],
        creator_linked=[bool(linked.get(entry.wallet, False)) for entry in unique],
        sizes=[float(entry.capital_usd) if entry.capital_usd is not None else -1.0
               for entry in unique],
        shares_funder_with_prior=[funding(entry, "same_funder_as_prior_buyer")
                                  for entry in unique],
        common_funder_depth=[funding(entry, "nearest_common_funder_depth")
                             for entry in unique],
        creator_funded=[funding(entry, "shares_funder_with_creator") for entry in unique],
    )


@dataclass
class DNAMatch:
    status: str
    label: Optional[str] = None
    confidence: float = 0.0
    neighbours: List[Tuple[str, str, float]] = field(default_factory=list)
    detail: str = ""


class BuyerDNA:
    """Nearest-neighbour match of a launch's opening sequence against history.

    Refuses to answer on a thin corpus. A 1-NN against three labelled launches
    is not a prior; it is a coincidence with a confidence interval, and it will
    happily tell a caller that a launch "resembles 37 previous monsters" on the
    strength of one.
    """

    def __init__(self, depth: int = 25, min_corpus: int = 50, neighbours: int = 5):
        self.depth = depth
        self.min_corpus = max(2, min_corpus)
        self.neighbours = max(1, neighbours)
        self._corpus: List[Tuple[str, str, np.ndarray]] = []

    def add(self, fingerprint: BuyerFingerprint, label: str) -> None:
        self._corpus.append((fingerprint.token, str(label), fingerprint.vector(self.depth)))

    @property
    def size(self) -> int:
        return len(self._corpus)

    def match(self, fingerprint: BuyerFingerprint) -> DNAMatch:
        if self.size < self.min_corpus:
            return DNAMatch(status="DATA_BLOCKED",
                            detail=(f"corpus holds {self.size} launches, "
                                    f"below the {self.min_corpus} needed to be a prior"))
        if fingerprint.depth == 0:
            return DNAMatch(status="DATA_BLOCKED", detail="no buyers observed yet")

        query = fingerprint.vector(self.depth)
        scored = []
        for token, label, vector in self._corpus:
            # Only compare positions both launches actually observed. Padding
            # matched against real values would make short launches resemble
            # whatever the padding happens to look like.
            mask = (query >= 0) & (vector >= 0)
            if not mask.any():
                continue
            distance = float(np.linalg.norm((query - vector)[mask]) / math.sqrt(mask.sum()))
            scored.append((token, label, distance))
        if not scored:
            return DNAMatch(status="DATA_BLOCKED",
                            detail="no historical launch overlaps this one's observed depth")

        scored.sort(key=lambda item: item[2])
        top = scored[: self.neighbours]
        votes: Dict[str, float] = defaultdict(float)
        for _, label, distance in top:
            votes[label] += 1.0 / (1.0 + distance)
        label, weight = max(votes.items(), key=lambda item: item[1])
        total = sum(votes.values())
        return DNAMatch(status="OK", label=label,
                        confidence=float(weight / total) if total else 0.0,
                        neighbours=top,
                        detail=f"{len(top)} nearest of {self.size} labelled launches")


@dataclass
class SwarmReading:
    status: str
    evidence: float = 0.0
    independent_skilled_so_far: int = 0
    probability: Optional[float] = None
    detail: str = ""


class SwarmPredictor:
    """P(k or more further independent skilled wallets arrive within a horizon).

    The forward question. Copying a famous wallet after its fill is the
    backward one, and by the time it can be answered the information is in the
    price. As everywhere else here, an uncalibrated model yields evidence and
    explicitly no probability.
    """

    def __init__(self, skill_threshold: float = 0.6, independence_threshold: float = 0.5,
                 target_count: int = 3):
        self.skill_threshold = skill_threshold
        self.independence_threshold = independence_threshold
        self.target_count = max(1, target_count)
        self._model: Optional[Any] = None

    def load_model(self, model: Any) -> bool:
        if not hasattr(model, "predict_proba"):
            logger.warning("swarm model rejected: no predict_proba")
            return False
        self._model = model
        return True

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def evaluate(self, entries: Sequence[Entry], report: IndependenceReport,
                 as_of: float, window: float = 10.0) -> SwarmReading:
        recent = [entry for entry in entries
                  if 0 <= as_of - entry.timestamp <= window and entry.skill is not None]
        if not recent:
            return SwarmReading(status="DATA_BLOCKED",
                                detail="no scored entries inside the window")

        qualifying = [
            entry for entry in recent
            if float(entry.skill) >= self.skill_threshold
            and (report.scores.get(entry.wallet, 0.0) if report.status == "OK" else 0.0)
            >= self.independence_threshold
        ]
        count = len(qualifying)
        # Arrival rate of independent skilled wallets, which is the thing that
        # either is or is not accelerating.
        rate = count / max(window, 1e-9)
        acceleration = 0.0
        if count >= 2:
            spans = sorted(entry.timestamp for entry in qualifying)
            first_half = sum(1 for value in spans if value <= as_of - window / 2)
            second_half = count - first_half
            acceleration = float((second_half - first_half) / max(count, 1))
        evidence = float(np.clip(rate * (1.0 + max(0.0, acceleration)), 0.0, 1.0))

        if not self.is_trained:
            return SwarmReading(status="DATA_BLOCKED", evidence=evidence,
                                independent_skilled_so_far=count,
                                detail="no chronologically validated swarm model")
        vector = np.asarray([[rate, acceleration, float(count), float(len(recent))]])
        try:
            raw = self._model.predict_proba(vector)[0]
        except Exception as exc:  # pragma: no cover - defensive
            return SwarmReading(status="DATA_BLOCKED", evidence=evidence,
                                independent_skilled_so_far=count,
                                detail=f"model inference failed: {exc}")
        probability = float(raw[1]) if len(raw) > 1 else float(raw[0])
        return SwarmReading(status="OK", evidence=evidence,
                            independent_skilled_so_far=count, probability=probability,
                            detail=f"model over {len(recent)} scored entries")

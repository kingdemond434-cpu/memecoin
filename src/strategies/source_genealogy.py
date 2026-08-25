"""Mining public callers for what they actually predict, not what they claim.

A public group that posts "insider" calls is a signal whether or not the calls
are any good, and the two questions have to be kept apart:

    Does following this source make money?
    Does this source predict what the crowd is about to do?

A channel whose picks are terrible can still be extremely valuable if its posts
reliably precede retail buying by eight seconds. Equally, a channel with a
great-looking hit rate can be worthless if the move is already over by the time
its post is observable. Scoring sources on reputation, follower count, or their
own claims answers neither question.

The adversarial case is explicitly modelled rather than filtered out. The
common structure is:

    linked wallets accumulate -> group posts -> followers buy -> insiders sell

A source doing that has excellent FOMO-prediction value and negative hold
value, and a system with one "is this source good" number cannot express that.
So flow prediction and executable return are tracked separately, and
pre-post accumulation by wallets linked to the source is tracked as its own
signal -- it is the thing that distinguishes a caller from a distributor.

Source genealogy applies the same lead-lag reasoning one level up. If an
obscure regional account reliably precedes the large channel that everyone
watches, the obscure one is the asset and the large one is a repeater. Finding
that requires asking who posted first, recursively, rather than ranking by
audience.

All of this operates on lawfully public messages: their content, their
timestamps, and the on-chain activity around them. Nothing here needs, or has
any path to, private or access-controlled material.
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SOURCE_GENEALOGY_SCHEMA_VERSION = "v1"

# Executable-return horizons after a public post. If the edge is gone by the
# time we could have acted, the source has predictive interest and no trading
# value, and these separate the two.
RETURN_HORIZONS: Tuple[float, ...] = (0.1, 0.5, 1.0, 3.0, 10.0, 30.0)

# Below this many posts a source has not been measured. Ratios from a handful
# of calls are noise, and promoting a source on them is how a lucky channel
# gets capital.
MIN_POSTS_FOR_VERDICT = 20


@dataclass
class SourcePost:
    """One public message, already attributed and timestamped."""

    source_id: str
    token: str
    posted_at: float
    # When we actually saw it, which is what any executable claim must use.
    observed_at: float
    edited: bool = False
    deleted: bool = False
    named_wallets: List[str] = field(default_factory=list)

    @property
    def observation_lag(self) -> float:
        """Seconds between publication and our seeing it.

        The source's own delivery latency, which no amount of local speed
        recovers. A signal that reaches us after the move is not actionable
        however accurate it was.
        """
        return max(0.0, self.observed_at - self.posted_at)


@dataclass
class PostOutcome:
    """What happened around one post. Every field Optional: unmeasured is not zero."""

    post: SourcePost
    # Executable return at each horizon AFTER we observed the post.
    executable_returns: Dict[float, Optional[float]] = field(default_factory=dict)
    # Independent buyer arrivals in the window after the post, over the rate
    # before it. Above 1 means the crowd did arrive.
    flow_acceleration: Optional[float] = None
    # Notional bought by wallets linked to this source BEFORE the post.
    pre_post_accumulation_usd: Optional[float] = None
    # Notional sold by those wallets after the post.
    post_sell_usd: Optional[float] = None
    rugged: Optional[bool] = None
    max_feasible_multiple: Optional[float] = None


@dataclass
class SourceDNA:
    """What a source is worth, split by the question being asked."""

    source_id: str
    posts: int = 0
    status: str = "MEASURING"
    median_observation_lag: Optional[float] = None
    best_horizon: Optional[float] = None
    best_horizon_return: Optional[float] = None
    flow_prediction: Optional[float] = None
    distribution_score: Optional[float] = None
    rug_rate: Optional[float] = None
    monster_rate: Optional[float] = None
    edit_delete_rate: float = 0.0
    detail: str = ""

    @property
    def is_distributor(self) -> bool:
        """Linked wallets accumulate before the post and sell after it.

        Not a moral judgement and not a reason to ignore the source -- it is
        the specific shape that makes a source good at predicting flow and bad
        to hold alongside.
        """
        return bool(self.distribution_score is not None and self.distribution_score > 0.5)

    @property
    def tradeable_directly(self) -> bool:
        return bool(self.status == "MEASURED"
                    and self.best_horizon_return is not None
                    and self.best_horizon_return > 0)

    @property
    def useful_as_flow_signal(self) -> bool:
        return bool(self.status == "MEASURED"
                    and self.flow_prediction is not None
                    and self.flow_prediction > 1.2)


def _median(values: Sequence[float]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.median(clean)) if clean else None


def build_source_dna(source_id: str, outcomes: Sequence[PostOutcome],
                     min_posts: int = MIN_POSTS_FOR_VERDICT) -> SourceDNA:
    """Score one source on each question separately.

    Below ``min_posts`` the verdict is MEASURING and every derived rate is
    None. Reporting a rate from six calls as though it were an estimate is how
    a lucky channel gets capital.
    """
    dna = SourceDNA(source_id=source_id, posts=len(outcomes))
    if not outcomes:
        dna.status = "DATA_BLOCKED"
        dna.detail = "no observed posts"
        return dna

    dna.edit_delete_rate = float(
        sum(1 for o in outcomes if o.post.edited or o.post.deleted) / len(outcomes))
    dna.median_observation_lag = _median([o.post.observation_lag for o in outcomes])

    if len(outcomes) < min_posts:
        dna.status = "MEASURING"
        dna.detail = f"{len(outcomes)} posts, below the {min_posts} needed for a verdict"
        return dna

    dna.status = "MEASURED"
    # Executable return, per horizon, using only horizons that were measured.
    by_horizon: Dict[float, List[float]] = defaultdict(list)
    for outcome in outcomes:
        for horizon, value in outcome.executable_returns.items():
            if value is not None and math.isfinite(float(value)):
                by_horizon[float(horizon)].append(float(value))
    means = {horizon: float(np.mean(values))
             for horizon, values in by_horizon.items() if values}
    if means:
        dna.best_horizon = max(means, key=lambda h: means[h])
        dna.best_horizon_return = means[dna.best_horizon]

    dna.flow_prediction = _median([o.flow_acceleration for o in outcomes])

    # Distribution: the share of posts where linked wallets were already long
    # beforehand and sold afterwards. Either half alone is unremarkable.
    scored = [o for o in outcomes
              if o.pre_post_accumulation_usd is not None and o.post_sell_usd is not None]
    if scored:
        dna.distribution_score = float(
            sum(1 for o in scored
                if o.pre_post_accumulation_usd > 0 and o.post_sell_usd > 0) / len(scored))

    rugs = [o.rugged for o in outcomes if o.rugged is not None]
    if rugs:
        dna.rug_rate = float(sum(1 for value in rugs if value) / len(rugs))
    monsters = [o.max_feasible_multiple for o in outcomes
                if o.max_feasible_multiple is not None]
    if monsters:
        dna.monster_rate = float(sum(1 for value in monsters if value >= 10.0) / len(monsters))

    dna.detail = (f"{len(outcomes)} posts; "
                  f"best horizon {dna.best_horizon}s; "
                  f"flow prediction {dna.flow_prediction}")
    return dna


@dataclass
class LeadLag:
    leader: str
    follower: str
    shared_tokens: int
    median_lead_seconds: float
    lead_rate: float


class SourceGenealogy:
    """Who publishes first, recursively.

    The same lead-lag machinery the wallet graph uses, applied one level up. A
    source that is consistently first is the asset; one that is consistently
    second is a repeater, however large its audience. Ranking sources by
    audience finds the repeaters, which is why almost everyone watches them.
    """

    def __init__(self, min_shared_tokens: int = 5, max_lead_seconds: float = 600.0):
        self.min_shared_tokens = max(2, min_shared_tokens)
        # Beyond this the two posts are about the same token but not the same
        # information event, and calling that "leading" would credit a source
        # for a coincidence.
        self.max_lead_seconds = max_lead_seconds
        self._first_post: Dict[Tuple[str, str], float] = {}

    def record(self, post: SourcePost) -> None:
        key = (post.token, post.source_id)
        existing = self._first_post.get(key)
        if existing is None or post.posted_at < existing:
            self._first_post[key] = post.posted_at

    def _by_token(self) -> Dict[str, Dict[str, float]]:
        grouped: Dict[str, Dict[str, float]] = defaultdict(dict)
        for (token, source), at in self._first_post.items():
            grouped[token][source] = at
        return grouped

    def lead_lag(self) -> List[LeadLag]:
        shared: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        totals: Dict[Tuple[str, str], int] = defaultdict(int)
        for sources in self._by_token().values():
            for leader, leader_at in sources.items():
                for follower, follower_at in sources.items():
                    if leader == follower:
                        continue
                    totals[(leader, follower)] += 1
                    delta = follower_at - leader_at
                    if 0 < delta <= self.max_lead_seconds:
                        shared[(leader, follower)].append(delta)

        results: List[LeadLag] = []
        for pair, count in totals.items():
            if count < self.min_shared_tokens:
                continue
            leads = shared.get(pair, [])
            if not leads:
                continue
            results.append(LeadLag(
                leader=pair[0], follower=pair[1], shared_tokens=count,
                median_lead_seconds=float(np.median(leads)),
                lead_rate=float(len(leads) / count),
            ))
        results.sort(key=lambda item: (item.lead_rate, item.median_lead_seconds),
                     reverse=True)
        return results

    def upstream_of(self, source_id: str, min_lead_rate: float = 0.6) -> List[LeadLag]:
        """Sources that reliably publish before ``source_id``.

        The recursion target: apply this to whatever the answer is, and keep
        going until nothing publishes earlier. The last node is the earliest
        lawfully observable one, which is the one worth watching.
        """
        return [item for item in self.lead_lag()
                if item.follower == source_id and item.lead_rate >= min_lead_rate]


def source_value(dna: SourceDNA, latency_penalty_s: float = 1.0) -> Optional[float]:
    """One comparable number per source, or None when it has not been measured.

    Combines executable return with how long the source's own delivery lag
    eats into it. A source whose information is excellent but arrives thirty
    seconds late scores below one that is mediocre and instant, which is the
    correct ordering for a system that has to act.
    """
    if dna.status != "MEASURED" or dna.best_horizon_return is None:
        return None
    lag = dna.median_observation_lag if dna.median_observation_lag is not None else 0.0
    decay = math.exp(-lag / max(latency_penalty_s, 1e-9))
    return float(dna.best_horizon_return * decay)


def rank_sources(dnas: Iterable[SourceDNA]) -> Dict[str, Any]:
    """Split sources by what they are actually good for."""
    tradeable, flow_only, distributors, measuring = [], [], [], []
    for dna in dnas:
        if dna.status != "MEASURED":
            measuring.append(dna.source_id)
            continue
        if dna.is_distributor:
            distributors.append(dna.source_id)
        if dna.tradeable_directly:
            tradeable.append((dna.source_id, source_value(dna)))
        elif dna.useful_as_flow_signal:
            # Worthless to follow, valuable to anticipate.
            flow_only.append((dna.source_id, dna.flow_prediction))
    tradeable.sort(key=lambda item: (item[1] is not None, item[1]), reverse=True)
    flow_only.sort(key=lambda item: item[1], reverse=True)
    return {
        "tradeable": [{"source": s, "value": v} for s, v in tradeable],
        "flow_signal_only": [{"source": s, "flow_prediction": v} for s, v in flow_only],
        "distributors": sorted(distributors),
        "measuring": sorted(measuring),
    }

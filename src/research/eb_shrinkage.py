"""Empirical-Bayes shrinkage of small-sample hit rates toward their pool.

A wallet that hit 5x on 3 of 5 trades reads as a 60% 5x-hitter. Across the
whole population the 5x rate is a few percent, and five trades cannot tell
skill from luck at that base rate. Ranking on the raw ratio puts the luckiest
small samples at the top of every leaderboard, which is exactly the list the
desk follows. This is the quant desk's James-Stein / EB step (port ledger rows
3 and 30), done the standard way for rates: fit a Beta prior to the pool by
the method of moments, then use each member's posterior mean.

    shrunk = (hits + alpha) / (trials + alpha + beta)

With many trials the member's own rate dominates; with few, the pool's does.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

#: Below this many members the pool's variance is itself too noisy to split
#: into signal and noise, so the prior falls back to a weak one at the pool mean.
MIN_POOL = 8
#: Prior strength used when the pool cannot be fitted (pseudo-trials).
FALLBACK_STRENGTH = 10.0
#: Upper bound on prior strength, so a pool whose members look identical cannot
#: freeze every member at the mean regardless of its own record.
MAX_STRENGTH = 1_000.0


@dataclass(frozen=True)
class BetaPrior:
    alpha: float
    beta: float
    members: int
    fitted: bool

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def strength(self) -> float:
        return self.alpha + self.beta

    def shrink(self, hits: float, trials: float) -> float:
        if trials < 0 or hits < 0 or hits > trials:
            raise ValueError("hits must lie in [0, trials]")
        return (hits + self.alpha) / (trials + self.alpha + self.beta)

    def report(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta, "mean": self.mean,
                "strength": self.strength, "members": self.members, "fitted": self.fitted}


def fit_beta_prior(samples: Iterable[Tuple[float, float]]) -> BetaPrior:
    """Method-of-moments Beta prior from (hits, trials) pairs.

    Uses the trials-weighted pooled rate as the mean and removes the expected
    binomial noise from the observed spread of rates, so what is left is the
    real between-member variance. When nothing is left (all spread is noise),
    the prior is as strong as allowed: the pool explains every member.
    """
    rows = [(float(h), float(n)) for h, n in samples if n > 0]
    total = sum(n for _, n in rows)
    if not rows or total <= 0:
        return BetaPrior(1.0, 1.0, 0, False)
    mean = sum(h for h, _ in rows) / total
    # A pool with no hits (or all hits) still must not certify zero (or one).
    mean = min(max(mean, 0.5 / total), 1 - 0.5 / total)
    if len(rows) < MIN_POOL:
        return BetaPrior(mean * FALLBACK_STRENGTH, (1 - mean) * FALLBACK_STRENGTH, len(rows), False)
    rates = [h / n for h, n in rows]
    observed = sum((r - mean) ** 2 for r in rates) / (len(rates) - 1)
    noise = mean * (1 - mean) * sum(1 / n for _, n in rows) / len(rows)
    between = observed - noise
    if between <= 0:
        strength = MAX_STRENGTH
    else:
        strength = min(MAX_STRENGTH, max(mean * (1 - mean) / between - 1, 1e-3))
    return BetaPrior(mean * strength, (1 - mean) * strength, len(rows), True)

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
#: The prior may weigh at most this many times the pool's median record.
#: A small pool often shows no between-member spread by chance alone; without
#: this a fitted "all noise" prior would pin every wallet to the mean and the
#: ranking would stop ranking anyone. Capped, a long record still pulls away.
STRENGTH_PER_MEDIAN_TRIALS = 3.0


@dataclass(frozen=True)
class BetaPrior:
    alpha: float
    beta: float
    members: int
    fitted: bool
    capped: bool = False

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
                "strength": self.strength, "members": self.members, "fitted": self.fitted,
                "capped": self.capped}


def fit_beta_prior(samples: Iterable[Tuple[float, float]]) -> BetaPrior:
    """Method-of-moments Beta prior from (hits, trials) pairs.

    Uses the trials-weighted pooled rate as the mean and removes the expected
    binomial noise from the trials-weighted spread of rates, so what is left
    is the real between-member variance. When nothing is left (all spread is
    noise), the prior is as strong as allowed: the pool explains every member,
    up to a ceiling tied to the pool's median record length.
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
    # Trials-weighted moments (Kleinman 1973). With Var(r_i) = p(1-p)/n_i +
    # tau^2 (1 - 1/n_i), E[sum n_i (r_i - p)^2] = (K-1) p(1-p)
    # + tau^2 (N - K + 1 - sum n_i^2 / N). Weighting by n_i keeps one-trial
    # members, whose rate is 0 or 1, from swamping the spread estimate, and
    # the (1 - 1/n_i) factor is accounted for exactly. All-single-trial pools
    # cannot identify tau^2 (denominator 0) and are treated as no spread.
    k = len(rows)
    spread = sum(n * (h / n - mean) ** 2 for h, n in rows)
    denominator = total - k + 1 - sum(n * n for _, n in rows) / total
    between = ((spread - (k - 1) * mean * (1 - mean)) / denominator
               if denominator > 0 else 0.0)
    trials = sorted(n for _, n in rows)
    median = (trials[(len(trials) - 1) // 2] + trials[len(trials) // 2]) / 2
    ceiling = min(MAX_STRENGTH, STRENGTH_PER_MEDIAN_TRIALS * median)
    wanted = mean * (1 - mean) / between - 1 if between > 0 else float("inf")
    strength = min(ceiling, max(wanted, 1e-3))
    return BetaPrior(mean * strength, (1 - mean) * strength, len(rows), True,
                     capped=wanted > ceiling)

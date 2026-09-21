"""How surprised to be, in nats. One implementation, three callers.

Three modules independently ask the same shape of question -- "is this more
often than chance, given what chance actually is here?" -- and two of them
had grown their own copy of the binomial bound. Two copies of one piece of
arithmetic in one language is the shape that silently diverges when one is
fixed, and this one has edge cases that are easy to get wrong in exactly the
direction that manufactures findings.

Both bounds here are CHERNOFF bounds rather than exact tails. That is a
deliberate choice and it runs one way: a Chernoff bound understates
surprisal, so every module using it errs towards NOT flagging. A detector
that misses a real ring costs an opportunity; one that invents rings
discounts every launch the desk sees.
"""

from __future__ import annotations

import math

#: Guards against log(0). Small enough not to move any real answer.
_EPSILON = 1e-12


def binomial_surprisal(successes: int, trials: int, probability: float) -> float:
    """-ln P(at least `successes` successes | Binomial(trials, probability)).

    The degenerate cases are the whole reason this is a shared function.
    Two wallets that both open EVERY launch co-open every launch, and that
    is exactly what independence predicts -- so it must score zero, not
    infinity. Getting that wrong turns the two busiest bots on the chain
    into the most suspicious pair on it, which is precisely backwards.
    """
    if trials <= 0 or successes <= 0:
        return 0.0
    if probability >= 1.0 - _EPSILON:
        # Certain under the null. Nothing observed can be surprising.
        return 0.0
    probability = max(probability, _EPSILON)
    observed = successes / trials
    if observed <= probability:
        return 0.0
    # Kullback-Leibler divergence between the observed and null rates, times
    # the number of trials.
    divergence = observed * math.log(observed / probability)
    if observed < 1.0:
        # At observed == 1 the second term is 0*log(0), which is zero by
        # convention and a domain error in floating point.
        divergence += (1.0 - observed) * math.log(
            (1.0 - observed) / (1.0 - probability))
    return float(trials * divergence)


def poisson_surprisal(count: int, span_s: float, rate_per_s: float) -> float:
    """-ln P(at least `count` arrivals in `span_s` from a Poisson stream).

    For counting events in a window rather than successes in trials. An
    exchange hot wallet emitting two withdrawals a second produces a
    five-in-ten-seconds group constantly, and calling that a cluster would
    flag most of Solana; measured against the wallet's own rate that group
    is unremarkable and this returns ~0.
    """
    if count <= 0 or span_s <= 0 or rate_per_s <= 0:
        return 0.0
    expected = rate_per_s * span_s
    if expected <= 0 or count <= expected:
        return 0.0
    return float(count * (math.log(count / expected) - 1.0) + expected)

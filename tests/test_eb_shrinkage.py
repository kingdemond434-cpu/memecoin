"""Small lucky samples must not outrank long records (port ledger rows 3/30)."""
import asyncio
import random

import pytest

from src.research.eb_shrinkage import MAX_STRENGTH, fit_beta_prior
from src.strategies.wallet_intelligence import (
    WalletIntelligenceEngine, WalletRegime, WalletRegimePerformance)


def test_a_pool_of_pure_noise_shrinks_everyone_to_the_mean():
    rng = random.Random(7)
    rows = [(sum(rng.random() < 0.05 for _ in range(20)), 20) for _ in range(400)]
    prior = fit_beta_prior(rows)
    assert prior.fitted and abs(prior.mean - 0.05) < 0.01
    assert prior.strength > 100
    assert prior.shrink(3, 5) < 0.07          # 3 of 5 is luck at this base rate


def test_real_dispersion_keeps_long_records_close_to_their_own_rate():
    rng = random.Random(11)
    rows = []
    for _ in range(300):
        skill = rng.choice((0.02, 0.30))
        rows.append((sum(rng.random() < skill for _ in range(200)), 200))
    prior = fit_beta_prior(rows)
    assert prior.fitted and prior.strength < 20
    assert abs(prior.shrink(60, 200) - 0.30) < 0.03


def test_the_long_record_outranks_the_lucky_short_one():
    rng = random.Random(3)
    rows = [(sum(rng.random() < 0.05 for _ in range(30)), 30) for _ in range(200)]
    prior = fit_beta_prior(rows)
    lucky, proven = prior.shrink(3, 5), prior.shrink(60, 300)   # raw 0.60 vs 0.20
    assert proven > lucky


def test_tiny_or_degenerate_pools_fall_back_without_certifying_zero():
    assert not fit_beta_prior([]).fitted
    small = fit_beta_prior([(0, 10)] * 3)
    assert not small.fitted and 0 < small.mean < 0.1
    flat = fit_beta_prior([(0, 10)] * 50)
    assert flat.strength == MAX_STRENGTH and flat.shrink(0, 10) > 0


def test_invalid_counts_are_refused():
    with pytest.raises(ValueError):
        fit_beta_prior([(1, 10)] * 10).shrink(11, 10)


def _engine():
    engine = WalletIntelligenceEngine.__new__(WalletIntelligenceEngine)
    engine.regime_performances = {}
    return engine


def test_wallet_scoring_ranks_on_shrunk_rates():
    engine = _engine()
    regime = WalletRegime.QUIET_REGIME
    rng = random.Random(5)
    for index in range(200):
        n = 40
        hits = sum(rng.random() < 0.04 for _ in range(n))
        engine.regime_performances[f"w{index}"] = {regime: WalletRegimePerformance(
            wallet=f"w{index}", regime=regime, trades=n, win_rate_5x=hits / n)}
    lucky = WalletRegimePerformance(wallet="lucky", regime=regime, trades=5, win_rate_5x=0.6)
    proven = WalletRegimePerformance(wallet="proven", regime=regime, trades=300, win_rate_5x=0.2)
    engine.regime_performances["lucky"] = {regime: lucky}
    engine.regime_performances["proven"] = {regime: proven}
    engine._fit_hit_rate_priors()
    assert engine._shrunk(proven, "win_rate_5x") > engine._shrunk(lucky, "win_rate_5x")
    assert engine._shrunk(lucky, "win_rate_5x") < 0.2
    # Without a fitted prior the raw rate is used unchanged.
    assert _engine()._shrunk(lucky, "win_rate_5x") == 0.6

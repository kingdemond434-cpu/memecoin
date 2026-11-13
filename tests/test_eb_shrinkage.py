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
    assert prior.strength > 40                # capped at 3x the median record (60)
    assert prior.shrink(3, 5) < 0.11          # 3 of 5 is luck at this base rate


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
    assert flat.capped and flat.strength == 30 and flat.shrink(0, 10) > 0
    huge = fit_beta_prior([(10, 10_000)] * 50)
    assert huge.strength == MAX_STRENGTH


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


def test_single_trial_members_do_not_hide_real_spread():
    """Var(r) = p(1-p)/n + tau^2 (1 - 1/n); n=1 members carry no tau^2 at all."""
    rng = random.Random(13)
    rows = []
    for index in range(600):
        skill = rng.choice((0.05, 0.45))
        n = 1 if index % 2 else 40
        rows.append((sum(rng.random() < skill for _ in range(n)), n))
    prior = fit_beta_prior(rows)
    # True tau^2 = 0.04 -> strength ~ 0.25*0.75/0.04 - 1 ~= 3.7.
    assert prior.fitted and 2 < prior.strength < 6


def test_a_small_noise_pool_is_capped_so_long_records_still_pull_away():
    rows = [(1, 20)] * 12                       # zero spread: "all noise"
    prior = fit_beta_prior(rows)
    assert prior.capped and prior.strength == pytest.approx(3 * 20)
    assert prior.shrink(120, 400) > 0.25        # a long 0.30 record stays well above the 0.05 mean
    assert prior.report()["capped"] is True


def test_genealogy_smart_money_uses_the_shrunk_rate():
    from src.strategies.genealogy_graph import EntityType, GenealogyGraph, WalletProfile

    graph = GenealogyGraph.__new__(GenealogyGraph)
    graph.wallets = {}
    rng = random.Random(17)
    for index in range(200):
        n = 30
        hits = sum(rng.random() < 0.04 for _ in range(n))
        graph.wallets[f"w{index}"] = WalletProfile(
            address=f"w{index}", entity_type=EntityType.WALLET, first_seen=0, last_seen=0,
            win_rate_5x=hits / n, win_rate_trials=n)
    lucky = WalletProfile(address="lucky", entity_type=EntityType.WALLET, first_seen=0,
                          last_seen=0, win_rate_5x=0.6, win_rate_trials=5)
    proven = WalletProfile(address="proven", entity_type=EntityType.WALLET, first_seen=0,
                           last_seen=0, win_rate_5x=0.4, win_rate_trials=300)
    legacy = WalletProfile(address="legacy", entity_type=EntityType.WALLET, first_seen=0,
                           last_seen=0, win_rate_5x=0.6)
    graph.wallets.update(lucky=lucky, proven=proven, legacy=legacy)
    assert graph.shrunk_win_rate_5x(lucky) < 0.3 < graph.shrunk_win_rate_5x(proven)
    assert graph.shrunk_win_rate_5x(legacy) == 0.6      # no sample size recorded
    assert graph._calculate_trust_score(proven) > graph._calculate_trust_score(lucky)

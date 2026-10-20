"""The measured wallet value existed and nothing in production read it.

`wallet_value.py` was written to replace a hand-weighted composite -- early
entry 0.25, forward return 0.30, consistency 0.20, independence 0.15, sample
size 0.10, times (1 - rug_exposure/2), times (1 - crowding*0.3), "top" above
0.5 -- with the only question a copier has: what would following this wallet
have done to OUR capital, at fills we could have got.

It was built, tested, and wired for WRITING. `record_follow_outcome` was
called; `get_wallet_value` and `get_top_wallets_by_value` had no caller at
all. Every production consumer -- the training feature that marks a buyer
smart, the hazard signal that fires when a smart wallet exits, the social
discovery pass -- still ranked on the composite. The measurement was being
taken and thrown away.

This is the fourth "written, tested, never wired" defect found in this
codebase, so these tests assert the WIRING rather than the methods. Testing
`followable_wallets` in isolation would have passed the whole time the desk
was ignoring it.
"""

import ast
import inspect
from pathlib import Path

import pytest

from src.strategies.wallet_intelligence import WalletIntelligenceEngine
from src.strategies.wallet_value import FollowOutcome, WalletValueModel

ROOT = Path(__file__).resolve().parents[1] / "src"


class _Engine:
    """The read surface under test, without the network the real engine wants."""

    followable_wallets = WalletIntelligenceEngine.followable_wallets
    get_top_wallets = WalletIntelligenceEngine.get_top_wallets
    UNMEASURED_CONFIDENCE_CEILING = WalletIntelligenceEngine.UNMEASURED_CONFIDENCE_CEILING
    MEASURED_CONFIDENCE_FLOOR = WalletIntelligenceEngine.MEASURED_CONFIDENCE_FLOOR
    CONFIDENCE_SCALE = WalletIntelligenceEngine.CONFIDENCE_SCALE

    def __init__(self):
        self.wallet_value = WalletValueModel(min_samples=4, shrinkage_strength=0.0)
        self.wallet_scores = {}


def _measured(engine, wallet, multiple, n=12):
    for index in range(n):
        engine.wallet_value.record(FollowOutcome(
            wallet=wallet, token=f"{wallet}-{index}", observed_at=float(index),
            executable_multiple=multiple))


def _composite(engine, wallet, score):
    from src.strategies.wallet_intelligence import WalletScore
    engine.wallet_scores[wallet] = WalletScore(wallet, score, {}, 0, 0, 0, 0, 0, 0, 20)


class TestMeasurementOutranksTheFormula:
    def test_a_measured_wallet_outranks_a_perfectly_scored_unmeasured_one(self):
        engine = _Engine()
        _measured(engine, "measured", 1.20)
        _composite(engine, "formula", 1.0)
        confidences = engine.followable_wallets(limit=10)
        assert confidences["measured"] > confidences["formula"]

    def test_the_gap_is_structural_not_incidental(self):
        """A composite score cannot reach a measured wallet's floor, ever."""
        engine = _Engine()
        _measured(engine, "measured", 1.01)
        _composite(engine, "formula", 99.0)
        confidences = engine.followable_wallets(limit=10)
        assert confidences["formula"] <= _Engine.UNMEASURED_CONFIDENCE_CEILING
        assert confidences["measured"] >= _Engine.MEASURED_CONFIDENCE_FLOOR

    def test_a_wallet_that_loses_money_to_follow_is_not_listed_at_all(self):
        engine = _Engine()
        _measured(engine, "loser", 0.5)
        assert "loser" not in engine.followable_wallets(limit=10)

    def test_confidence_saturates_so_one_outlier_cannot_buy_certainty(self):
        engine = _Engine()
        _measured(engine, "good", 1.5)
        _measured(engine, "absurd", 50.0)
        confidences = engine.followable_wallets(limit=10)
        assert confidences["absurd"] < 1.0
        assert confidences["absurd"] - confidences["good"] < 0.2

    def test_training_features_refuse_the_formula_entirely(self):
        engine = _Engine()
        _composite(engine, "formula", 1.0)
        assert engine.followable_wallets(limit=10, include_unmeasured=False) == {}
        assert "formula" in engine.followable_wallets(limit=10)


def _calls_in(path: str, name: str) -> int:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    return sum(1 for node in ast.walk(tree)
               if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Attribute)
               and node.func.attr == name)


class TestProductionActuallyReadsIt:
    """The part that was missing. Each of these had zero callers."""

    CONSUMERS = ("research/dataset_builder.py", "strategies/rug_hazard.py",
                 "strategies/social_intelligence.py")

    @pytest.mark.parametrize("path", CONSUMERS)
    def test_the_consumer_asks_for_followable_wallets(self, path):
        assert _calls_in(path, "followable_wallets") >= 1

    @pytest.mark.parametrize("path", CONSUMERS)
    def test_the_consumer_no_longer_ranks_on_the_composite(self, path):
        assert _calls_in(path, "get_top_wallets") == 0

    def test_the_smart_buyer_feature_is_not_a_thresholded_formula(self):
        """Parsed, not grepped: the comment explaining the fix says the same
        words the fix removed, and a text search cannot tell them apart."""
        tree = ast.parse((ROOT / "research/dataset_builder.py").read_text(encoding="utf-8"))
        reads = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Attribute) and node.attr == "overall_score"]
        assert reads == []

    def test_the_engine_still_offers_the_composite_as_a_named_fallback(self):
        """Deleting it would hide that anything unmeasured is being used."""
        assert callable(WalletIntelligenceEngine.get_top_wallets)
        doc = inspect.getdoc(WalletIntelligenceEngine.followable_wallets) or ""
        assert "fallback" in doc

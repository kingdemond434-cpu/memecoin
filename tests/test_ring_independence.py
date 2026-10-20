"""Distinct wallets are not distinct deciders.

`SniperRingDetector` has been observing every launch's opening cohort since
it was built, learning which wallets keep arriving together -- and
`independent_count`, the method that turns that into an answer, had no caller
anywhere. It was learning and nobody asked it anything.

The model saw `organic_ratio` (distinct wallets) and `bundle_concentration`
(same-slot arrivals). Neither can see a ring that funds from one source and
enters over several slots, which is what a competent bundler looks like. So a
First-25 that was really a First-3 in twenty-five hats read as maximally
organic -- overcounted in the direction that says enter bigger, on exactly
the launches most likely to be a setup.
"""

import random

import pytest

from src.research.feature_engine import build_features
from src.strategies.multihead_predictor import MultiHeadPredictor, PredictionFeatures
from src.strategies.sniper_rings import SniperRingDetector


def _trained_detector(ring, *, launches=400, seed=7):
    """A detector with enough observations to have an opinion."""
    rng = random.Random(seed)
    detector = SniperRingDetector()
    for index in range(launches):
        crowd = [f"W{rng.randrange(4000)}" for _ in range(8)]
        detector.observe_launch(
            (list(ring) + crowd) if index % 3 == 0 else crowd, at=float(index))
    return detector


# --- the detector's own answer -------------------------------------------

def test_a_ring_collapses_to_one_decision():
    ring = [f"R{index}" for index in range(5)]
    detector = _trained_detector(ring)
    count, detail = detector.independent_count(ring + ["S1", "S2", "S3"])
    assert detail["status"] == "OK"
    assert count == 4, "five hats and three people is four deciders"
    assert set(detail["collapsed"][0]["collapsed_to_one"]) == set(ring)


def test_independent_wallets_are_not_collapsed():
    detector = _trained_detector([f"R{index}" for index in range(5)])
    count, _detail = detector.independent_count(["S1", "S2", "S3", "S4"])
    assert count == 4


def test_it_refuses_to_answer_before_it_has_seen_enough():
    """A co-occurrence rate over twelve launches is not a rate."""
    ring = [f"R{index}" for index in range(5)]
    detector = _trained_detector(ring, launches=12)
    count, detail = detector.independent_count(ring + ["S1", "S2"])
    assert detail["status"] == "DATA_BLOCKED"
    assert count == 7, "unmeasured means no compression, not full compression"


# --- it reaches the model ------------------------------------------------

def _features(wallet_features):
    return build_features({"token": "t", "chain": "solana", "created_at": 0.0},
                          {"wallet_features": wallet_features, "timestamp": 0.0})


def test_the_feature_reaches_the_array_the_model_trains_on():
    assert _features({"ring_compression": 0.5}).ring_compression == 0.5
    index = MultiHeadPredictor("models").feature_names.index("ring_compression")
    assert _features({"ring_compression": 0.5}).to_array()[index] == pytest.approx(0.5)


def test_unmeasured_reads_as_full_independence_not_as_suspicion():
    """Penalising a launch for a ring nobody detected would invent the ring."""
    assert _features({}).ring_compression == 1.0
    assert _features({"ring_compression": None}).ring_compression == 1.0


def test_the_array_and_the_name_list_agree():
    """A feature in one and not the other silently shifts every column."""
    features = PredictionFeatures("t", "solana", 0.0)
    assert len(features.to_array()) == len(
        MultiHeadPredictor("models").feature_names)


def test_it_is_clamped_to_a_ratio():
    assert _features({"ring_compression": 4.0}).to_array()[
        MultiHeadPredictor("models").feature_names.index("ring_compression")] == 1.0
    assert _features({"ring_compression": -2.0}).to_array()[
        MultiHeadPredictor("models").feature_names.index("ring_compression")] == 0.0


# --- the builder computes it ---------------------------------------------

def test_the_dataset_builder_asks_the_detector():
    from src.research.dataset_builder import PointInTimeDatasetBuilder
    builder = PointInTimeDatasetBuilder.__new__(PointInTimeDatasetBuilder)
    ring = [f"R{index}" for index in range(5)]
    builder.independence_provider = _trained_detector(ring).independent_count
    count, detail = builder._independent_buyers(ring + ["S1", "S2", "S3"])
    assert count == 4 and detail["status"] == "OK"


def test_no_detector_is_unmeasured_rather_than_an_error():
    from src.research.dataset_builder import PointInTimeDatasetBuilder
    builder = PointInTimeDatasetBuilder.__new__(PointInTimeDatasetBuilder)
    builder.independence_provider = None
    count, detail = builder._independent_buyers(["A", "B"])
    assert count is None and detail["status"] == "DATA_BLOCKED"


def test_a_raising_detector_does_not_break_feature_capture():
    from src.research.dataset_builder import PointInTimeDatasetBuilder
    builder = PointInTimeDatasetBuilder.__new__(PointInTimeDatasetBuilder)
    builder.independence_provider = lambda _w: (_ for _ in ()).throw(
        RuntimeError("boom"))
    count, detail = builder._independent_buyers(["A", "B"])
    assert count is None and detail["status"] == "DATA_BLOCKED"


def test_the_provider_is_wired_to_the_detector_that_is_being_fed():
    """It observes every launch; this is the first thing that reads it."""
    from pathlib import Path
    wiring = Path("src/runtime/wiring.py").read_text(encoding="utf-8")
    assert "sniper_rings.observe_launch" in Path(
        "src/runtime/source_intelligence.py").read_text(encoding="utf-8").replace(
        "rings.observe_launch", "sniper_rings.observe_launch")
    assert "independence_provider" in wiring
    assert "sniper_rings.independent_count" in wiring

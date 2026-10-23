import pytest

from src.research.calibration import ModelCalibration, binomial_interval


def test_unobserved_tail_is_not_reported_as_a_proven_zero_probability():
    interval = binomial_interval(0, 200)
    assert interval[0] == pytest.approx(0, abs=1e-12)
    assert 0.01 < interval[1] < 0.03
    assert binomial_interval(0, 0) is None


def test_thin_positive_tail_has_wide_uncertainty_even_with_low_average_error():
    book = ModelCalibration("p_100x", min_samples=20)
    for i in range(200):
        book.record(.005, i == 0, at=i + 1)
    report = book.report()
    assert report["positive_events"] == 1
    lower, upper = report["observed_rate_interval_95"]
    assert lower < .005 < upper
    assert upper > .02
    assert report["reliability"][0]["observed_rate_interval_95"] == [lower, upper]
    assert report["interval_assumption"] == "INDEPENDENT_BERNOULLI_DESCRIPTIVE_ONLY"


@pytest.mark.parametrize("successes,samples", [(2, 1), (-1, 3), (1, -2), (True, 1), (1, 3.0)])
def test_invalid_evidence_counts_are_rejected(successes, samples):
    with pytest.raises(ValueError):
        binomial_interval(successes, samples)

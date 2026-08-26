"""Whether a further age split is supported by the data.

The predictor is split into four age bands because a coin three hundred
milliseconds old and one ten minutes old are not the same problem. The
temptation is to keep going -- split "flash" at 200ms, split "early" at two
seconds -- because every extra band looks like more precision.

It usually is not. Splitting a band does two things at once: it makes each
sub-band's model see a more homogeneous world, and it halves the rows that
model is fitted on. Below a certain amount of evidence the second effect wins
outright, and the result is two confidently wrong models where there was one
honestly uncertain one. Worse, the split will LOOK justified, because any
partition of a finite sample shows some difference between its halves.

So a split has to be warranted, and this is what warrants it:

* both sides clear the training floor on their own, with no topping up from a
  neighbour -- a band fed from its neighbour is the pooled model wearing a
  band's name;
* the two sides' outcomes differ by more than sampling noise, on a target that
  matters, at a stated significance.

Neither condition alone is enough. A difference that is real but leaves one
side with forty rows cannot be modelled; a split with ten thousand rows a side
and no difference between them is two copies of the same model.

This answers the question. It does not perform the split: adding a band is an
edit to AGE_BANDS with this report recorded next to it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

BAND_SPLIT_SCHEMA_VERSION = "v1"

# Rows each side must hold on its own. Matches the trainer's per-band floor:
# a split that leaves a side unable to train is not a split, it is a deletion.
DEFAULT_MIN_SIDE_SAMPLES = 60

# Two-sided significance for the difference between the sides. 0.01 rather
# than the customary 0.05 because this test will be run against many candidate
# cuts, and at 0.05 one in twenty arbitrary cuts looks warranted.
DEFAULT_ALPHA = 0.01


def _normal_sf(value: float) -> float:
    """Upper tail of the standard normal. erfc, so no SciPy dependency."""
    return 0.5 * math.erfc(value / math.sqrt(2.0))


@dataclass
class SplitWarrant:
    """The verdict on one candidate cut, with the numbers behind it."""

    status: str
    band: str = ""
    cut_seconds: float = 0.0
    target: str = ""
    left_samples: int = 0
    right_samples: int = 0
    left_mean: float = 0.0
    right_mean: float = 0.0
    difference: float = 0.0
    z: float = 0.0
    p_value: float = 1.0
    detail: str = ""

    @property
    def warranted(self) -> bool:
        return self.status == "WARRANTED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": BAND_SPLIT_SCHEMA_VERSION, "status": self.status,
            "band": self.band, "cut_seconds": self.cut_seconds, "target": self.target,
            "left_samples": self.left_samples, "right_samples": self.right_samples,
            "left_mean": round(self.left_mean, 6), "right_mean": round(self.right_mean, 6),
            "difference": round(self.difference, 6), "z": round(self.z, 4),
            "p_value": round(self.p_value, 6), "warranted": self.warranted,
            "detail": self.detail,
        }


def split_warrant(left: Sequence[float], right: Sequence[float], *,
                  band: str = "", cut_seconds: float = 0.0, target: str = "",
                  min_side_samples: int = DEFAULT_MIN_SIDE_SAMPLES,
                  alpha: float = DEFAULT_ALPHA) -> SplitWarrant:
    """Is cutting this band at ``cut_seconds`` supported by these outcomes?

    Welch's t statistic, read against the normal -- the samples are large
    where the answer matters, and where they are not, the sample floor has
    already refused the split for the reason that actually applies.
    """
    n_left, n_right = len(left), len(right)
    base = dict(band=band, cut_seconds=cut_seconds, target=target,
                left_samples=n_left, right_samples=n_right)
    if n_left < min_side_samples or n_right < min_side_samples:
        return SplitWarrant(
            status="DATA_BLOCKED", **base,
            detail=(f"{n_left}/{n_right} rows either side; each needs "
                    f"{min_side_samples}. A side that cannot train on its own "
                    "would have to be fed from its neighbour, which is the "
                    "pooled model wearing a band's name."))

    mean_left = sum(left) / n_left
    mean_right = sum(right) / n_right
    var_left = sum((value - mean_left) ** 2 for value in left) / max(1, n_left - 1)
    var_right = sum((value - mean_right) ** 2 for value in right) / max(1, n_right - 1)
    standard_error = math.sqrt(var_left / n_left + var_right / n_right)
    if standard_error <= 0:
        # Zero variance on both sides. Identical constants are not a
        # difference; a genuine step between two constants is.
        difference = mean_right - mean_left
        status = "WARRANTED" if difference else "NOT_WARRANTED"
        return SplitWarrant(
            status=status, **base, left_mean=mean_left, right_mean=mean_right,
            difference=difference, z=float("inf") if difference else 0.0,
            p_value=0.0 if difference else 1.0,
            detail=("both sides are constant and differ" if difference
                    else "both sides are the same constant"))

    difference = mean_right - mean_left
    z = difference / standard_error
    p_value = 2.0 * _normal_sf(abs(z))
    if p_value > alpha:
        return SplitWarrant(
            status="NOT_WARRANTED", **base, left_mean=mean_left,
            right_mean=mean_right, difference=difference, z=z, p_value=p_value,
            detail=(f"p={p_value:.4f} > {alpha}: the two sides are not "
                    "distinguishable, so splitting produces two copies of one "
                    "model on half the rows each"))
    return SplitWarrant(
        status="WARRANTED", **base, left_mean=mean_left, right_mean=mean_right,
        difference=difference, z=z, p_value=p_value,
        detail=(f"p={p_value:.4f} <= {alpha} with {n_left}/{n_right} rows: the "
                "sides differ and both can be trained"))


def evaluate_cuts(rows: Sequence[Tuple[float, float]], *, band: str = "",
                  cuts: Sequence[float] = (), target: str = "",
                  min_side_samples: int = DEFAULT_MIN_SIDE_SAMPLES,
                  alpha: float = DEFAULT_ALPHA) -> Dict[str, Any]:
    """Test several candidate cuts of one band. ``rows`` is (age, outcome).

    The Bonferroni correction is applied and stated. Testing eight cuts at
    p<0.01 and reporting the best one is testing at p<0.08 and calling it
    0.01, which is how an arbitrary boundary acquires a significance figure.
    """
    candidates = [float(cut) for cut in cuts]
    corrected = alpha / max(1, len(candidates))
    warrants: List[SplitWarrant] = []
    for cut in candidates:
        left = [outcome for age, outcome in rows if age < cut]
        right = [outcome for age, outcome in rows if age >= cut]
        warrants.append(split_warrant(
            left, right, band=band, cut_seconds=cut, target=target,
            min_side_samples=min_side_samples, alpha=corrected))
    supported = [warrant for warrant in warrants if warrant.warranted]
    return {
        "schema": BAND_SPLIT_SCHEMA_VERSION,
        "band": band, "target": target, "rows": len(rows),
        "alpha": alpha, "corrected_alpha": corrected,
        "cuts_tested": len(candidates),
        # The recommendation is deliberately singular. A band that "could" be
        # split three ways at once is a band whose evidence has not been read.
        "status": "WARRANTED" if supported else "NOT_WARRANTED",
        "recommended_cut": (min(supported, key=lambda item: item.p_value).cut_seconds
                            if supported else None),
        "warrants": [warrant.to_dict() for warrant in warrants],
        "detail": ("" if supported else
                   "no candidate cut separates this band's outcomes; leave it "
                   "as one model until it does"),
    }

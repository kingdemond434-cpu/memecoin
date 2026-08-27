"""Are the probabilities true? Measured, per model, with sample gating.

Every sizing decision this desk makes is a function of stated probabilities:
P(rug in 30s), P(land | bid), P(monster). Kelly is exquisitely sensitive to
them -- a hazard stated at 0.10 that is really 0.25 does not make the book
slightly too large, it makes a positive-edge strategy negative. And nothing
in this system has ever checked whether those numbers are true.

Accuracy is not the question. A rug model that says 0.05 for every token and
is right 95% of the time is accurate and useless. The question is CALIBRATION:
of the tokens this model called 20% likely to rug, did about 20% of them rug?

Three outputs, and the third is the one that matters operationally.

**Reliability curve.** Predicted probability against observed frequency, in
bins. The shape says where the model is wrong, which is more useful than a
single score: a model that is well calibrated in the middle and wildly
over-confident in the tail is dangerous in exactly the region that decides
tail-preservation.

**Expected calibration error.** The bin-weighted average gap. One number to
watch move.

**Direction.** Over-confident or under-confident, stated separately, because
they are not symmetric here. A hazard model that UNDER-states risk holds
positions through rugs; one that OVER-states it exits monsters early. Both
cost money; the first can end the account, so it is reported first and
weighted separately.

Sample gating runs through all of it. An ECE computed from thirty
observations is noise, and a gate that accepts it would promote a model on
the strength of a coin flip. Below the floor every reading is DATA_BLOCKED --
never a comfortable default, because "not yet measured" and "measured and
fine" are the two states this must never confuse.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

CALIBRATION_SCHEMA_VERSION = "v1"

#: Reliability bins. Ten equal-width buckets over [0, 1].
DEFAULT_BINS = 10

#: Observations below which nothing is reported as measured. Chosen so a bin
#: at the tail can carry a handful of events rather than one.
DEFAULT_MIN_SAMPLES = 200

#: Per-bin floor. A bin thinner than this contributes to the ECE but is
#: flagged, because a 100% observed rate from two events is not evidence.
DEFAULT_MIN_BIN_SAMPLES = 10

#: ECE at or below which a model is treated as calibrated enough to size on.
DEFAULT_ECE_THRESHOLD = 0.05


class Provenance(Enum):
    """Where a prediction/outcome pair came from.

    Without this an "OK" model looks equally proven whether it was validated
    against real money or replayed over reconstructed history -- and those are
    not remotely the same evidence. A landing model reporting 900 clean
    observations while the signer has signed zero transactions is not lying;
    it is answering a question nobody asked precisely enough.
    """

    #: Real capital, real fills. The only kind that proves execution.
    FORWARD_REAL = "forward_real"
    #: Live decisions, paper fills. Proves the model, not the execution.
    SHADOW = "shadow"
    #: Replayed over historical point-in-time snapshots. Real market data,
    #: but our own latency and slippage are assumed rather than measured.
    RECONSTRUCTED = "reconstructed"
    #: Generated. Useful for wiring tests, worthless as evidence.
    SYNTHETIC = "synthetic"


#: How much each provenance counts toward a promotion decision. Reconstructed
#: evidence is real evidence about the market and no evidence at all about our
#: own execution, so it is admitted at a discount rather than excluded --
#: excluding it would throw away the fastest available path to a calibrated
#: model, and counting it in full would let a replay authorise live capital.
PROVENANCE_WEIGHT: Dict[str, float] = {
    Provenance.FORWARD_REAL.value: 1.0,
    Provenance.SHADOW.value: 0.5,
    Provenance.RECONSTRUCTED.value: 0.25,
    Provenance.SYNTHETIC.value: 0.0,
}


@dataclass
class Bin:
    """One reliability bucket."""

    lower: float
    upper: float
    count: int = 0
    predicted_sum: float = 0.0
    observed_sum: float = 0.0

    @property
    def predicted(self) -> Optional[float]:
        return (self.predicted_sum / self.count) if self.count else None

    @property
    def observed(self) -> Optional[float]:
        return (self.observed_sum / self.count) if self.count else None

    @property
    def gap(self) -> Optional[float]:
        if not self.count:
            return None
        return (self.observed_sum - self.predicted_sum) / self.count

    def to_dict(self, min_bin_samples: int) -> Dict[str, Any]:
        return {
            "range": [round(self.lower, 3), round(self.upper, 3)],
            "count": self.count,
            "predicted": (round(self.predicted, 4) if self.predicted is not None else None),
            "observed": (round(self.observed, 4) if self.observed is not None else None),
            "gap": (round(self.gap, 4) if self.gap is not None else None),
            # A bin this thin is shown but must not be reasoned from.
            "thin": self.count < min_bin_samples,
        }


class ModelCalibration:
    """Reliability for one model's one probability, over time."""

    def __init__(self, name: str, *, bins: int = DEFAULT_BINS,
                 min_samples: int = DEFAULT_MIN_SAMPLES,
                 min_bin_samples: int = DEFAULT_MIN_BIN_SAMPLES,
                 ece_threshold: float = DEFAULT_ECE_THRESHOLD):
        self.name = name
        self.min_samples = max(1, int(min_samples))
        self.min_bin_samples = max(1, int(min_bin_samples))
        self.ece_threshold = float(ece_threshold)
        width = 1.0 / max(1, int(bins))
        self.bins = [Bin(lower=index * width, upper=(index + 1) * width)
                     for index in range(int(bins))]
        self.count = 0
        self.brier_sum = 0.0
        self.first_at = 0.0
        self.last_at = 0.0
        #: Observations by where they came from. A model whose evidence is
        #: entirely reconstructed must not read as proven.
        self.by_provenance: Dict[str, int] = {}

    def record(self, probability: float, occurred: bool,
               at: Optional[float] = None,
               provenance: str = Provenance.SHADOW.value) -> bool:
        """One prediction and what actually happened.

        A probability outside [0, 1] is refused rather than clipped: a model
        emitting 1.4 has a defect, and silently clamping it produces a
        calibration report that says the model is fine.
        """
        value = float(probability)
        if not (0.0 <= value <= 1.0) or math.isnan(value):
            logger.debug("calibration %s refused probability %r", self.name, probability)
            return False
        now = float(at if at is not None else time.time())
        outcome = 1.0 if occurred else 0.0
        index = min(len(self.bins) - 1, int(value * len(self.bins)))
        bucket = self.bins[index]
        bucket.count += 1
        bucket.predicted_sum += value
        bucket.observed_sum += outcome
        self.count += 1
        self.brier_sum += (value - outcome) ** 2
        key = str(provenance or Provenance.SHADOW.value)
        self.by_provenance[key] = self.by_provenance.get(key, 0) + 1
        if not self.first_at:
            self.first_at = now
        self.last_at = now
        return True

    # --- the measurements ------------------------------------------------

    @property
    def brier(self) -> Optional[float]:
        return (self.brier_sum / self.count) if self.count else None

    def evidence_weight(self) -> float:
        """Provenance-weighted observation count.

        A thousand synthetic observations weigh nothing; a thousand replayed
        ones weigh two hundred and fifty. This is what a promotion gate should
        read instead of the raw count.
        """
        return sum(count * PROVENANCE_WEIGHT.get(name, 0.0)
                   for name, count in self.by_provenance.items())

    def dominant_provenance(self) -> str:
        if not self.by_provenance:
            return "none"
        return max(self.by_provenance.items(), key=lambda item: item[1])[0]

    def expected_calibration_error(self) -> Optional[float]:
        """Bin-weighted mean absolute gap. None below the sample floor."""
        if self.count < self.min_samples:
            return None
        total = 0.0
        for bucket in self.bins:
            if not bucket.count:
                continue
            total += (bucket.count / self.count) * abs(bucket.gap or 0.0)
        return total

    def direction(self) -> Optional[Dict[str, float]]:
        """Signed error, split by direction.

        Under-confidence and over-confidence are not interchangeable for a
        hazard: understating risk holds through rugs, overstating it exits
        monsters early. Reported apart so the asymmetry survives into whatever
        reads this.
        """
        if self.count < self.min_samples:
            return None
        understated = 0.0
        overstated = 0.0
        for bucket in self.bins:
            if not bucket.count:
                continue
            weight = bucket.count / self.count
            gap = bucket.gap or 0.0
            # gap > 0: it happened MORE often than predicted -- the model
            # understated the probability.
            if gap > 0:
                understated += weight * gap
            else:
                overstated += weight * -gap
        return {"understated": understated, "overstated": overstated}

    def report(self) -> Dict[str, Any]:
        ece = self.expected_calibration_error()
        direction = self.direction()
        if self.count < self.min_samples:
            status, detail = "DATA_BLOCKED", (
                f"{self.count} of {self.min_samples} observations; calibration "
                "is unmeasured and must not be read as acceptable")
        elif ece is not None and ece <= self.ece_threshold:
            status, detail = "OK", ""
        else:
            status, detail = "MISCALIBRATED", (
                f"expected calibration error {ece:.3f} exceeds "
                f"{self.ece_threshold:.3f}; sizing on these probabilities is "
                "sizing on the wrong number")
        return {
            "model": self.name,
            "status": status,
            "detail": detail,
            "observations": self.count,
            # Beside every number, where the number came from. An OK model
            # built on replay is a hypothesis; the same model built on real
            # fills is a result.
            "provenance": dict(sorted(self.by_provenance.items())),
            "dominant_provenance": self.dominant_provenance(),
            "evidence_weight": round(self.evidence_weight(), 1),
            "proven_on_real_fills": bool(
                self.by_provenance.get(Provenance.FORWARD_REAL.value, 0)),
            "min_samples": self.min_samples,
            "brier": (round(self.brier, 5) if self.brier is not None else None),
            "expected_calibration_error": (round(ece, 5) if ece is not None else None),
            "ece_threshold": self.ece_threshold,
            "understates_risk_by": (round(direction["understated"], 5)
                                    if direction else None),
            "overstates_risk_by": (round(direction["overstated"], 5)
                                   if direction else None),
            "reliability": [bucket.to_dict(self.min_bin_samples)
                            for bucket in self.bins],
            "observed_over": (round((self.last_at - self.first_at) / 3_600.0, 2)
                              if self.first_at and self.last_at > self.first_at
                              else None),
        }

    def state(self) -> Dict[str, Any]:
        return {
            "name": self.name, "count": self.count, "brier_sum": self.brier_sum,
            "first_at": self.first_at, "last_at": self.last_at,
            "by_provenance": dict(self.by_provenance),
            "bins": [{"lower": b.lower, "upper": b.upper, "count": b.count,
                      "predicted_sum": b.predicted_sum,
                      "observed_sum": b.observed_sum} for b in self.bins],
        }


class CalibrationBook:
    """Every model's calibration, persisted, with one honest summary."""

    def __init__(self, path: Optional[Path] = None, *,
                 min_samples: int = DEFAULT_MIN_SAMPLES,
                 ece_threshold: float = DEFAULT_ECE_THRESHOLD):
        self.path = Path(path) if path else None
        self.min_samples = int(min_samples)
        self.ece_threshold = float(ece_threshold)
        self.models: Dict[str, ModelCalibration] = {}

    def record(self, model: str, probability: float, occurred: bool,
               at: Optional[float] = None,
               provenance: str = Provenance.SHADOW.value) -> bool:
        if not model:
            return False
        book = self.models.get(model)
        if book is None:
            book = ModelCalibration(model, min_samples=self.min_samples,
                                    ece_threshold=self.ece_threshold)
            self.models[model] = book
        return book.record(probability, occurred, at, provenance)

    def trustworthy(self, model: str) -> Optional[bool]:
        """Can this model's probabilities be sized on?

        None means unmeasured. Callers must treat None as "no" for sizing and
        as "not yet" for reporting -- collapsing the two is how an unmeasured
        model gets promoted for having no evidence against it.
        """
        book = self.models.get(model)
        if book is None:
            return None
        ece = book.expected_calibration_error()
        if ece is None:
            return None
        return ece <= book.ece_threshold

    def report(self) -> Dict[str, Any]:
        rows = [book.report() for book in
                sorted(self.models.values(), key=lambda item: item.name)]
        measured = [row for row in rows if row["status"] != "DATA_BLOCKED"]
        bad = [row for row in rows if row["status"] == "MISCALIBRATED"]
        return {
            "schema": CALIBRATION_SCHEMA_VERSION,
            "status": ("DATA_BLOCKED" if not measured
                       else "MISCALIBRATED" if bad else "OK"),
            "detail": ("no model has reached its sample floor; every stated "
                       "probability in this desk is currently unverified"
                       if not measured else
                       f"{len(bad)} model(s) miscalibrated: "
                       f"{', '.join(row['model'] for row in bad)}"
                       if bad else ""),
            "models_tracked": len(rows),
            "models_measured": len(measured),
            "models_miscalibrated": len(bad),
            "models_proven_on_real_fills": len(
                [row for row in rows if row.get("proven_on_real_fills")]),
            "models": rows,
        }

    # --- persistence -----------------------------------------------------

    def save(self) -> bool:
        if self.path is None:
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(
                "w", dir=str(self.path.parent), delete=False)
            with handle:
                json.dump({"schema": CALIBRATION_SCHEMA_VERSION,
                           "models": [book.state() for book in self.models.values()]},
                          handle)
            os.replace(handle.name, self.path)
            return True
        except OSError as exc:
            logger.warning("calibration save failed: %s", exc)
            return False

    def load(self) -> bool:
        if self.path is None or not self.path.exists():
            return False
        try:
            state = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("calibration unreadable: %s", exc)
            return False
        for row in state.get("models") or []:
            name = row.get("name")
            if not name:
                continue
            book = ModelCalibration(name, bins=len(row.get("bins") or DEFAULT_BINS),
                                    min_samples=self.min_samples,
                                    ece_threshold=self.ece_threshold)
            book.count = int(row.get("count", 0))
            book.brier_sum = float(row.get("brier_sum", 0.0))
            book.first_at = float(row.get("first_at", 0.0))
            book.last_at = float(row.get("last_at", 0.0))
            book.by_provenance = dict(row.get("by_provenance") or {})
            for bucket, saved in zip(book.bins, row.get("bins") or []):
                bucket.count = int(saved.get("count", 0))
                bucket.predicted_sum = float(saved.get("predicted_sum", 0.0))
                bucket.observed_sum = float(saved.get("observed_sum", 0.0))
            self.models[name] = book
        return True

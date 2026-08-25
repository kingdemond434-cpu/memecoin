"""One brain per launch age, because a launch is a different object at each.

The predictor was trained on every snapshot horizon pooled together --
sub-second rows and hour-old rows in one bag -- and learned the average
launch. There is no such thing.

At 100ms almost nothing is observable. Who funded whom, how fast the first
buys arrived, whether the deployer bought his own mint. The holder
distribution does not exist yet; the social response has not happened; half
the columns are structurally missing rather than merely unknown. By five
minutes all of those exist, and the features that mattered at 100ms have been
overwritten by their own consequences: the funding pattern that predicted the
first wave is now visible in the holder concentration it produced, so a model
sees it twice and weights it as two independent pieces of evidence.

One model reconciling those regimes is not learning a launch. It is learning a
blend of four, and the blend is dominated by whichever horizon produced the
most rows -- which, before the sub-second rungs existed, was the late ones.
The decisions that matter most were being made by a model fitted mostly to
states that arrive long after the decision.

So each band gets its own artifact, trained only on rows from that band and
consulted only for decisions at that age. The alternative -- adding age as a
feature to one pooled model -- is also done, because it costs nothing and
helps within a band, but it does not fix this: age as a feature lets a model
shift its estimate, while what actually changes is what every other column
MEANS.

Two disciplines are load-bearing.

A band with no artifact answers DATA_BLOCKED. It does not borrow a neighbour's
model. Predicting a 100ms launch with a brain fitted to five-minute rows is
exactly the training-serving skew this separation exists to remove, and doing
it silently would leave the system looking trained while being wrong in the
one place it is most expensive to be wrong.

A pooled fallback is available but always LABELLED. Shadow evaluation needs to
keep running while the per-band artifacts accumulate, and a labelled fallback
is an honest bridge; an unlabelled one is a lie that promotion would read as
evidence. `band_status` travels with every prediction, and only OWN_BAND is
evidence about that band.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.strategies.multihead_predictor import (
    AGE_BANDS, MultiHeadPrediction, MultiHeadPredictor, PredictionFeatures, band_for,
)

logger = logging.getLogger(__name__)

AGE_BANDED_SCHEMA_VERSION = "v1"

OWN_BAND = "OWN_BAND"
POOLED_FALLBACK = "POOLED_FALLBACK"
BLOCKED = "DATA_BLOCKED"

BAND_NAMES = tuple(name for name, _, _ in AGE_BANDS)


def band_model_dir(root: str, band: str) -> str:
    """Where one band's artifacts live. Separate directories, not a shared one.

    `MultiHeadPredictor.load_latest` picks the newest bundle in its directory,
    so bands sharing a directory would silently load each other's models --
    the precise failure this module exists to prevent, reintroduced by a
    filename convention.
    """
    return os.path.join(root, "bands", band)


@dataclass
class BandReport:
    band: str
    trained: bool
    model_version: str = ""
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"band": self.band, "trained": self.trained,
                "model_version": self.model_version, "detail": self.detail}


class AgeBandedPredictor:
    """Routes each decision to the brain fitted to launches of that age."""

    def __init__(self, model_dir: str = "models", *,
                 allow_pooled_fallback: bool = True):
        self.model_dir = model_dir
        self.allow_pooled_fallback = bool(allow_pooled_fallback)
        self.bands: Dict[str, MultiHeadPredictor] = {}
        for band in BAND_NAMES:
            predictor = MultiHeadPredictor(band_model_dir(model_dir, band))
            predictor.initialize_models()
            self.bands[band] = predictor
        # The pooled model is the bridge, never the destination.
        self.pooled = MultiHeadPredictor(model_dir)
        self.pooled.initialize_models()

    # -- schema, shared by construction ------------------------------------

    @property
    def ARTIFACT_VERSION(self) -> int:
        return MultiHeadPredictor.ARTIFACT_VERSION

    @property
    def feature_names(self) -> List[str]:
        """One schema across every band.

        The bands differ in what they LEARNED, never in what they are shown.
        A per-band feature list would make the artifacts incomparable and would
        let the pooled bridge and a band disagree about column order, which is
        the kind of mismatch that produces confident nonsense rather than an
        error.
        """
        return list(self.pooled.feature_names)

    @property
    def model_version(self) -> str:
        trained = self.trained_bands
        if not trained:
            return self.pooled.model_version if self.pooled._is_trained else ""
        return "+".join(f"{band}:{self.bands[band].model_version}" for band in trained)

    @property
    def validation_report(self) -> Dict[str, Any]:
        """The band reports, keyed by band, plus the pooled bridge's own."""
        report: Dict[str, Any] = {
            band: predictor.validation_report
            for band, predictor in self.bands.items() if predictor._is_trained
        }
        if self.pooled._is_trained:
            report["pooled"] = self.pooled.validation_report
        return report

    # -- loading -----------------------------------------------------------

    def load_latest(self) -> Dict[str, bool]:
        """Load whatever exists. A band with nothing stays untrained, loudly."""
        loaded: Dict[str, bool] = {}
        for band, predictor in self.bands.items():
            try:
                loaded[band] = bool(predictor.load_latest())
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("age band %s failed to load: %s", band, exc)
                loaded[band] = False
        loaded["pooled"] = bool(self.pooled.load_latest())
        return loaded

    @property
    def trained_bands(self) -> List[str]:
        return [band for band, predictor in self.bands.items() if predictor._is_trained]

    @property
    def _is_trained(self) -> bool:
        """True when ANY brain can answer.

        Deliberately not "all bands": a desk with a flash model and no mature
        one can still trade sub-second launches, and blocking it entirely
        would throw away the band it does have evidence for.
        """
        return bool(self.trained_bands) or (
            self.allow_pooled_fallback and self.pooled._is_trained)

    # -- inference ---------------------------------------------------------

    def band_of(self, features: PredictionFeatures) -> str:
        return band_for(getattr(features, "time_since_launch", 0.0) or 0.0)

    def predict(self, features: PredictionFeatures) -> Optional[MultiHeadPrediction]:
        band = self.band_of(features)
        predictor = self.bands.get(band)
        if predictor is not None and predictor._is_trained:
            prediction = predictor.predict(features)
            if prediction is not None:
                prediction.age_band = band
                prediction.band_status = OWN_BAND
                return prediction
        if not self.allow_pooled_fallback or not self.pooled._is_trained:
            # No brain for this age, and no labelled bridge. Silence is the
            # only honest answer: a neighbour's model is not this band's.
            logger.debug("no model for age band %s and no pooled fallback", band)
            return None
        prediction = self.pooled.predict(features)
        if prediction is None:
            return None
        prediction.age_band = band
        prediction.band_status = POOLED_FALLBACK
        return prediction

    # -- reporting ---------------------------------------------------------

    def report(self) -> Dict[str, Any]:
        bands = {
            band: BandReport(
                band=band, trained=predictor._is_trained,
                model_version=predictor.model_version if predictor._is_trained else "",
                detail="" if predictor._is_trained else "no artifact for this age band",
            ).to_dict()
            for band, predictor in self.bands.items()
        }
        trained = self.trained_bands
        return {
            "schema": AGE_BANDED_SCHEMA_VERSION,
            "status": "OK" if len(trained) == len(BAND_NAMES) else (
                "PARTIAL" if trained else "DATA_BLOCKED"),
            "bands": bands,
            "trained_bands": trained,
            "pooled_fallback": {"allowed": self.allow_pooled_fallback,
                                "trained": self.pooled._is_trained},
            "boundaries_s": {name: [low, high if high != float("inf") else None]
                             for name, low, high in AGE_BANDS},
        }

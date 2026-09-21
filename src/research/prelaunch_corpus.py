"""Labelled rows for "is this entity about to launch", frozen before the answer.

The pre-launch model could not be trained because nothing ever wrote a
training row. `train_launch_predictor` takes a list of records with `features`
and `launched_within_1h`, and no code in this repository produced one, so the
model was permanently untrained and `predict_launch_probability` permanently
returned 0.0 -- a confident-looking zero that is really "nobody ever taught me
anything".

This is the missing producer. It is deliberately dull, because the only thing
that makes a pre-launch corpus worth anything is the discipline:

**The features are frozen before the label exists.** A row is opened the
moment the intent model flags an entity, carrying the feature snapshot taken
at that instant. The label is written later, by observing whether a launch
followed. Nothing measured after `observed_at` may enter `features`, and the
row carries `observed_at` so that invariant is checkable rather than asserted.

**Waiting is not a negative.** A row whose horizon has not elapsed is
unresolved, not a zero. Treating "no launch yet" as "no launch" would label
every recent observation false and teach the model that the present never
produces launches -- which is exactly backwards, since the present is the only
time the model is ever asked about.

**An entity that launched before we flagged it is not a positive.** It is
discarded. A row whose launch precedes its own observation is not a prediction,
it is a memory, and training on it teaches the model to recognise launches
that have already happened.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

PRELAUNCH_CORPUS_SCHEMA_VERSION = "v1"

#: The horizon the label asks about. One hour is what the model's own
#: `launch_probability_1h` predicts, so the corpus has to ask the same
#: question the model answers.
DEFAULT_HORIZON_S = 3600.0

#: Rows kept. Bounded so a desk running for months does not hold every
#: observation it ever made in memory.
DEFAULT_CAPACITY = 50_000

#: Below this many resolved rows, and below this many positives, training is
#: refused. A classifier fitted on four launches is a story about four
#: launches wearing a probability's clothes.
MIN_ROWS_TO_TRAIN = 200
MIN_POSITIVES_TO_TRAIN = 20


@dataclass
class PrelaunchRow:
    """One observation of an entity, and whether a launch followed it."""

    entity: str
    observed_at: float
    features: Dict[str, float] = field(default_factory=dict)
    intent_score: float = 0.0
    launch_probability_1h: float = 0.0
    #: Written only by `resolve`. None means the horizon has not elapsed.
    launched_within_1h: Optional[bool] = None
    launched_at: Optional[float] = None
    token: str = ""
    horizon_s: float = DEFAULT_HORIZON_S

    @property
    def resolved(self) -> bool:
        return self.launched_within_1h is not None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PrelaunchCorpus:
    """Opens a row when an entity is flagged, closes it when the horizon ends."""

    def __init__(self, path: Optional[str] = None, *,
                 horizon_s: float = DEFAULT_HORIZON_S,
                 capacity: int = DEFAULT_CAPACITY):
        self.path = Path(path) if path else None
        self.horizon_s = float(horizon_s)
        self._rows: Deque[PrelaunchRow] = deque(maxlen=int(capacity))
        #: Open rows by entity, so a launch can find what predicted it.
        self._open: Dict[str, List[PrelaunchRow]] = {}
        self.discarded_retrospective = 0

    # -- writing ----------------------------------------------------------

    def observe(self, entity: str, features: Dict[str, float], *,
                observed_at: Optional[float] = None,
                intent_score: float = 0.0,
                launch_probability_1h: float = 0.0) -> Optional[PrelaunchRow]:
        """Open a row. Everything in ``features`` must predate ``observed_at``."""
        if not entity or not features:
            return None
        moment = time.time() if observed_at is None else float(observed_at)
        existing = self._open.get(entity, [])
        # One open row per entity per horizon. Re-flagging the same entity
        # every scoring pass would otherwise write the same observation a
        # hundred times and let one entity dominate the training set.
        if any(moment - row.observed_at < self.horizon_s for row in existing):
            return None
        row = PrelaunchRow(
            entity=entity, observed_at=moment,
            features={str(k): float(v) for k, v in features.items()},
            intent_score=float(intent_score),
            launch_probability_1h=float(launch_probability_1h),
            horizon_s=self.horizon_s)
        self._rows.append(row)
        self._open.setdefault(entity, []).append(row)
        return row

    def record_launch(self, entity: str, token: str,
                      launched_at: Optional[float] = None) -> int:
        """An entity launched. Close every open row it can legitimately answer.

        Only rows observed BEFORE the launch and inside the horizon become
        positives. A row observed after the launch is discarded rather than
        labelled: it is a memory, not a prediction.
        """
        if not entity:
            return 0
        moment = time.time() if launched_at is None else float(launched_at)
        closed = 0
        for row in list(self._open.get(entity, ())):
            if row.resolved:
                continue
            if moment < row.observed_at:
                self.discarded_retrospective += 1
                self._drop(row)
                continue
            if moment - row.observed_at <= row.horizon_s:
                row.launched_within_1h = True
                row.launched_at = moment
                row.token = str(token)
                closed += 1
                self._forget(row)
        return closed

    def resolve(self, now: Optional[float] = None) -> int:
        """Close rows whose horizon has elapsed with no launch, as negatives."""
        moment = time.time() if now is None else float(now)
        closed = 0
        for entity in list(self._open):
            for row in list(self._open.get(entity, ())):
                if row.resolved:
                    self._forget(row)
                    continue
                if moment - row.observed_at > row.horizon_s:
                    row.launched_within_1h = False
                    closed += 1
                    self._forget(row)
        return closed

    def _forget(self, row: PrelaunchRow) -> None:
        open_rows = self._open.get(row.entity)
        if not open_rows:
            return
        self._open[row.entity] = [item for item in open_rows if item is not row]
        if not self._open[row.entity]:
            self._open.pop(row.entity, None)

    def _drop(self, row: PrelaunchRow) -> None:
        self._forget(row)
        try:
            self._rows.remove(row)
        except ValueError:
            pass

    # -- reading ----------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._rows)

    def resolved_rows(self) -> List[PrelaunchRow]:
        return [row for row in self._rows if row.resolved]

    def training_records(self) -> List[Dict[str, Any]]:
        """Exactly the shape `train_launch_predictor` consumes."""
        return [{"features": dict(row.features),
                 "launched_within_1h": bool(row.launched_within_1h),
                 "entity": row.entity, "observed_at": row.observed_at}
                for row in self.resolved_rows()]

    def trainable(self) -> bool:
        rows = self.resolved_rows()
        positives = sum(1 for row in rows if row.launched_within_1h)
        return (len(rows) >= MIN_ROWS_TO_TRAIN
                and positives >= MIN_POSITIVES_TO_TRAIN)

    def report(self) -> Dict[str, Any]:
        rows = self.resolved_rows()
        positives = sum(1 for row in rows if row.launched_within_1h)
        open_rows = sum(len(items) for items in self._open.values())
        return {
            "schema": PRELAUNCH_CORPUS_SCHEMA_VERSION,
            "status": "OK" if rows else "DATA_BLOCKED",
            "rows": self.size,
            "resolved": len(rows),
            "open": open_rows,
            "positives": positives,
            "base_rate": (positives / len(rows)) if rows else None,
            "discarded_retrospective": self.discarded_retrospective,
            "trainable": self.trainable(),
            "min_rows": MIN_ROWS_TO_TRAIN,
            "min_positives": MIN_POSITIVES_TO_TRAIN,
            "detail": ("" if rows else
                       "no row has resolved yet; the pre-launch model has "
                       "nothing to be trained on and predicts nothing"),
        }

    # -- persistence ------------------------------------------------------

    def save(self) -> bool:
        if self.path is None:
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"schema": PRELAUNCH_CORPUS_SCHEMA_VERSION,
                       "horizon_s": self.horizon_s,
                       "discarded_retrospective": self.discarded_retrospective,
                       "rows": [row.to_dict() for row in self._rows]}
            self.path.write_text(json.dumps(payload), encoding="utf-8")
            return True
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("prelaunch corpus not saved: %s", exc)
            return False

    def load(self) -> bool:
        if self.path is None or not self.path.exists():
            return False
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("prelaunch corpus not loaded: %s", exc)
            return False
        if payload.get("schema") != PRELAUNCH_CORPUS_SCHEMA_VERSION:
            return False
        self.horizon_s = float(payload.get("horizon_s", self.horizon_s))
        self.discarded_retrospective = int(
            payload.get("discarded_retrospective", 0))
        self._rows.clear()
        self._open.clear()
        for item in payload.get("rows", ()):
            row = PrelaunchRow(
                entity=str(item.get("entity", "")),
                observed_at=float(item.get("observed_at", 0.0)),
                features={str(k): float(v)
                          for k, v in (item.get("features") or {}).items()},
                intent_score=float(item.get("intent_score", 0.0)),
                launch_probability_1h=float(item.get("launch_probability_1h", 0.0)),
                launched_within_1h=item.get("launched_within_1h"),
                launched_at=item.get("launched_at"),
                token=str(item.get("token", "")),
                horizon_s=float(item.get("horizon_s", self.horizon_s)))
            self._rows.append(row)
            if not row.resolved:
                self._open.setdefault(row.entity, []).append(row)
        return True

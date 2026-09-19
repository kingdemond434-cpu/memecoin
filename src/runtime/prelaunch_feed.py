"""The pre-launch subsystem, connected at both ends.

`PrelaunchIntentModel` was constructed, started, and left almost entirely
dark. All seven of its public methods had no caller. Three of them are the
FEEDS -- `record_metadata_creation`, `record_social_creation`,
`record_infrastructure_interaction` -- and without them
`_consume_external` reported `DATA_BLOCKED: no registered source
observations` on every pass, forever. Three more are the READS, and the
seventh is `train_launch_predictor`, which nothing could call because nothing
produced a training row.

So the module ran a scoring loop over entities that had almost no signals,
scored them with a model that could never be trained, and published a
`launch_probability_1h` of 0.0 that looked like a measurement.

What is fed here is only what the desk genuinely observes:

**Metadata creation** is recorded from the creation event's own URI. The
metadata is uploaded before the mint, so its existence is a fact about the
deployer that predates the launch -- but the desk sees it AT the launch, so it
is recorded against the deployer as history for their NEXT launch, never as a
prediction of the one in hand.

**Infrastructure interaction** is recorded for the wallets that funded the
deployer. A funder that keeps appearing behind fresh deployers is the single
most transferable pre-launch fact on this chain, and it is observable from the
system transfers already extracted from the launch transaction.

**Social creation is deliberately NOT fed.** `_find_linked_social` already
states the reason and it has not changed: there is no verified public
wallet-to-social source, so any edge written here would be an identity claim
the desk cannot support. `record_social_creation` keeps its DATA_BLOCKED
status rather than being handed fabricated links, and that is the correct
outcome, not an omission.

Nothing here invents a signal the desk cannot see.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

#: Rows are resolved on this cadence. The horizon is an hour, so checking more
#: often than this buys nothing and checking less often lets negatives pile up
#: unlabelled.
PRELAUNCH_RESOLVE_INTERVAL_S = 300.0


class PrelaunchFeed:
    """Mixin: feeds the pre-launch model and reads its predictions back."""

    def _record_prelaunch_launch(self, token: str, event: Dict[str, Any]) -> None:
        """A launch happened. Feed the signals it reveals, and label the rows.

        Called from the creation handler. The ORDER matters: the corpus is
        told about the launch first, so rows observed before it become
        positives, and only then are new signals recorded -- otherwise this
        launch's own evidence would be folded into the features of the row
        that is supposed to have predicted it.
        """
        prelaunch = getattr(self, "prelaunch", None)
        if prelaunch is None:
            return
        creator = str(event.get("creator") or "")
        at = float(event.get("timestamp", time.time()) or time.time())

        corpus = getattr(self, "prelaunch_corpus", None)
        if corpus is not None and creator:
            corpus.record_launch(creator, token, at)

        if not creator:
            return
        uri = str(event.get("uri") or event.get("metadata_uri") or "")
        if uri:
            # History for this deployer's NEXT launch, never evidence about
            # the one in hand: the desk sees the URI at the mint, not before.
            prelaunch.record_metadata_creation(
                creator, {"token": token, "uri": uri, "observed_at_mint": True}, at)
        for funder in self.prelaunch_funders(event, creator):
            prelaunch.record_infrastructure_interaction(
                funder, {"funded": creator, "token": token}, at)

    @staticmethod
    def prelaunch_funders(event: Dict[str, Any], creator: str) -> List[str]:
        """Who funded this deployer, from the launch transaction itself.

        The system transfers are already extracted from the transaction the
        desk decoded, so this costs no RPC and invents nothing. An empty list
        when the transaction carried no transfers is the honest answer: a
        funder graph that guessed would make unrelated launches look like one
        operator's work, which is the precise error it exists to catch.
        """
        funders: List[str] = []
        for transfer in (event.get("funding_transfers") or ()):
            if not isinstance(transfer, dict):
                continue
            source = str(transfer.get("from") or "")
            target = str(transfer.get("to") or "")
            if source and source != creator and (not target or target == creator):
                funders.append(source)
        # Order-preserving dedupe: one funder appearing in three transfers is
        # one funder, and counting it three times is how a single wallet
        # becomes a "cluster".
        return list(dict.fromkeys(funders))

    def harvest_prelaunch_rows(self, now: Optional[float] = None) -> int:
        """Open a corpus row for every entity the model currently flags.

        The features are the model's own point-in-time snapshot, taken when it
        flagged the entity. Nothing observed after that instant may enter the
        row, which is what makes the eventual label a prediction rather than a
        description.
        """
        prelaunch = getattr(self, "prelaunch", None)
        corpus = getattr(self, "prelaunch_corpus", None)
        if prelaunch is None or corpus is None:
            return 0
        opened = 0
        for pending in prelaunch.get_imminent_launches(min_prob=0.0):
            features = pending.get("features") or {}
            if not features:
                continue
            row = corpus.observe(
                str(pending.get("entity", "")), features,
                observed_at=float(pending.get("detected_at", time.time())),
                intent_score=float(pending.get("intent_score", 0.0) or 0.0),
                launch_probability_1h=float(pending.get("launch_prob_1h", 0.0) or 0.0))
            opened += int(row is not None)
        corpus.resolve(now)
        return opened

    def train_prelaunch_predictor(self) -> Dict[str, Any]:
        """Train the launch predictor, or say exactly why it cannot be.

        Refused rather than attempted below the corpus minimums. A calibrated
        classifier fitted on a handful of launches produces a confident
        probability from almost no evidence, and that number would then flow
        into sizing as though it meant something.
        """
        prelaunch = getattr(self, "prelaunch", None)
        corpus = getattr(self, "prelaunch_corpus", None)
        if prelaunch is None or corpus is None:
            return {"status": "DATA_BLOCKED", "detail": "pre-launch not wired"}
        report = corpus.report()
        if not corpus.trainable():
            return {"status": "DATA_BLOCKED",
                    "detail": "corpus below the training minimums",
                    "corpus": report}
        records = corpus.training_records()
        prelaunch.train_launch_predictor(records)
        return {"status": "OK" if getattr(prelaunch, "_is_trained", False)
                else "DATA_BLOCKED",
                "trained_on": len(records), "corpus": report}

    def prelaunch_report(self) -> Dict[str, Any]:
        """Who is about to launch, and whether that claim means anything."""
        prelaunch = getattr(self, "prelaunch", None)
        if prelaunch is None:
            return {"status": "DATA_BLOCKED", "detail": "pre-launch not wired"}
        stats = prelaunch.get_stats()
        corpus = getattr(self, "prelaunch_corpus", None)
        trained = bool(stats.get("model_trained"))
        imminent = prelaunch.get_imminent_launches(min_prob=0.5)
        return {
            "status": "OK" if trained else "DATA_BLOCKED",
            **stats,
            "corpus": corpus.report() if corpus is not None else
                      {"status": "DATA_BLOCKED", "detail": "no corpus"},
            "imminent": [
                {"entity": item.get("entity"),
                 "intent_score": item.get("intent_score"),
                 # The learned probability where one exists, and the heuristic
                 # it falls back to otherwise -- named apart, because a
                 # heuristic wearing a model's field name is how an untrained
                 # 0.0 gets read as a measurement.
                 "learned_probability": (
                     prelaunch.predict_launch_probability(str(item.get("entity", "")))
                     if trained else None),
                 "heuristic_probability": item.get("launch_prob_1h")}
                for item in imminent[:20]],
            "top_entities": [
                {"entity": profile.entity, "intent_score": profile.intent_score,
                 "risk_level": profile.risk_level}
                for profile in prelaunch.get_top_entities(limit=10)],
            "detail": ("" if trained else
                       "the launch predictor is untrained; every learned "
                       "probability is withheld rather than reported as 0.0"),
        }

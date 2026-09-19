"""The pre-launch model had no feed, no training row, and no reader.

All seven of `PrelaunchIntentModel`'s public methods had no caller. Three of
them are the signal FEEDS, so `_consume_external` reported
`DATA_BLOCKED: no registered source observations` on every pass, forever.
Three are the READS. The seventh is `train_launch_predictor`, which nothing
could call because nothing in the repository produced a training record.

The result looked healthy: a scoring loop ran, entities were ranked, and
`predict_launch_probability` returned 0.0 -- a confident-looking number whose
real meaning was "nobody ever taught me anything".

The corpus tests below are all about one invariant, because it is the only
thing that makes a pre-launch corpus worth having: the features are frozen
before the label exists.
"""

import pytest

from src.research.prelaunch_corpus import (
    MIN_POSITIVES_TO_TRAIN, MIN_ROWS_TO_TRAIN, PrelaunchCorpus)
from src.runtime.prelaunch_feed import PrelaunchFeed

T0 = 1_700_000_000.0
FEATURES = {"funding_velocity": 2.0, "wallet_cluster_size": 4.0}


class TestTheLabelComesAfterTheFeatures:
    def test_a_launch_inside_the_horizon_is_a_positive(self):
        corpus = PrelaunchCorpus(horizon_s=3600.0)
        corpus.observe("dev", FEATURES, observed_at=T0)
        assert corpus.record_launch("dev", "mint", T0 + 600) == 1
        row = corpus.resolved_rows()[0]
        assert row.launched_within_1h is True
        assert row.token == "mint"

    def test_a_launch_after_the_horizon_does_not_relabel_a_closed_row(self):
        corpus = PrelaunchCorpus(horizon_s=3600.0)
        corpus.observe("dev", FEATURES, observed_at=T0)
        corpus.resolve(T0 + 7200)
        assert corpus.resolved_rows()[0].launched_within_1h is False
        assert corpus.record_launch("dev", "mint", T0 + 7300) == 0

    def test_an_unelapsed_row_is_unresolved_not_a_negative(self):
        """Treating "no launch yet" as "no launch" would teach the model that
        the present never produces launches -- and the present is the only
        time it is ever asked about."""
        corpus = PrelaunchCorpus(horizon_s=3600.0)
        corpus.observe("dev", FEATURES, observed_at=T0)
        corpus.resolve(T0 + 60)
        assert corpus.resolved_rows() == []
        assert corpus.report()["open"] == 1

    def test_a_row_observed_after_the_launch_is_discarded_not_counted(self):
        """That row is a memory, not a prediction."""
        corpus = PrelaunchCorpus(horizon_s=3600.0)
        corpus.observe("dev", FEATURES, observed_at=T0 + 600)
        assert corpus.record_launch("dev", "mint", T0) == 0
        assert corpus.discarded_retrospective == 1
        assert corpus.size == 0

    def test_features_are_stored_as_given_and_never_touched_again(self):
        corpus = PrelaunchCorpus(horizon_s=3600.0)
        source = dict(FEATURES)
        corpus.observe("dev", source, observed_at=T0)
        source["funding_velocity"] = 999.0
        corpus.record_launch("dev", "mint", T0 + 60)
        assert corpus.resolved_rows()[0].features["funding_velocity"] == 2.0


class TestOneEntityCannotFloodTheCorpus:
    def test_reflagging_inside_the_horizon_opens_no_second_row(self):
        corpus = PrelaunchCorpus(horizon_s=3600.0)
        for offset in range(0, 600, 60):
            corpus.observe("dev", FEATURES, observed_at=T0 + offset)
        assert corpus.size == 1

    def test_a_later_horizon_does_open_a_new_row(self):
        corpus = PrelaunchCorpus(horizon_s=3600.0)
        corpus.observe("dev", FEATURES, observed_at=T0)
        corpus.observe("dev", FEATURES, observed_at=T0 + 7200)
        assert corpus.size == 2

    def test_an_entity_with_no_features_is_not_recorded(self):
        assert PrelaunchCorpus().observe("dev", {}, observed_at=T0) is None


class TestTrainingIsRefusedOnThinEvidence:
    def _fill(self, corpus, positives, negatives):
        index = 0
        for _ in range(positives):
            at = T0 + index * 7200
            corpus.observe(f"dev{index}", FEATURES, observed_at=at)
            corpus.record_launch(f"dev{index}", "mint", at + 60)
            index += 1
        for _ in range(negatives):
            at = T0 + index * 7200
            corpus.observe(f"dev{index}", FEATURES, observed_at=at)
            corpus.resolve(at + 7200)
            index += 1
        return corpus

    def test_a_corpus_below_the_row_floor_is_not_trainable(self):
        corpus = self._fill(PrelaunchCorpus(), MIN_POSITIVES_TO_TRAIN, 10)
        assert not corpus.trainable()

    def test_a_corpus_with_rows_but_no_positives_is_not_trainable(self):
        """A classifier fitted on zero launches has learned that nothing ever
        launches, which is true of the sample and false of the world."""
        corpus = self._fill(PrelaunchCorpus(), 0, MIN_ROWS_TO_TRAIN + 10)
        assert corpus.report()["resolved"] >= MIN_ROWS_TO_TRAIN
        assert not corpus.trainable()

    def test_enough_of_both_is_trainable(self):
        corpus = self._fill(PrelaunchCorpus(), MIN_POSITIVES_TO_TRAIN + 5,
                            MIN_ROWS_TO_TRAIN)
        assert corpus.trainable()
        record = corpus.training_records()[0]
        assert set(record) >= {"features", "launched_within_1h"}

    def test_an_empty_corpus_reports_blocked_with_a_reason(self):
        report = PrelaunchCorpus().report()
        assert report["status"] == "DATA_BLOCKED"
        assert "nothing to be trained on" in report["detail"]
        assert report["base_rate"] is None


class TestTheDeskFeedsIt:
    """The half that was missing. Testing the corpus alone would have passed
    every day the desk was ignoring it."""

    def _desk(self, prelaunch=None, corpus=None):
        class Desk(PrelaunchFeed):
            pass
        desk = Desk()
        desk.prelaunch = prelaunch
        desk.prelaunch_corpus = corpus
        return desk

    def _model(self):
        class Model:
            def __init__(self):
                self.metadata = []
                self.infrastructure = []
                self.social = []
                self._is_trained = False
                self.trained_on = None

            def record_metadata_creation(self, entity, evidence, ts=None):
                self.metadata.append((entity, evidence))

            def record_infrastructure_interaction(self, entity, evidence, ts=None):
                self.infrastructure.append((entity, evidence))

            def record_social_creation(self, entity, evidence, ts=None):
                self.social.append((entity, evidence))

            def get_imminent_launches(self, min_prob=0.5):
                return [{"entity": "dev", "detected_at": T0,
                         "intent_score": 0.8, "launch_prob_1h": 0.6,
                         "features": dict(FEATURES)}]

            def get_top_entities(self, limit=20):
                return []

            def get_stats(self):
                return {"model_trained": self._is_trained, "tracked_entities": 1}

            def train_launch_predictor(self, records):
                self.trained_on = len(records)
                self._is_trained = True

            def predict_launch_probability(self, entity, horizon_hours=1):
                return 0.42
        return Model()

    def test_a_launch_feeds_metadata_creation_for_its_deployer(self):
        model = self._model()
        desk = self._desk(model, PrelaunchCorpus())
        desk._record_prelaunch_launch(
            "mint", {"creator": "dev", "uri": "ipfs://abc", "timestamp": T0})
        assert model.metadata == [("dev", {"token": "mint", "uri": "ipfs://abc",
                                           "observed_at_mint": True})]

    def test_funders_come_from_the_launch_transaction_itself(self):
        model = self._model()
        desk = self._desk(model, PrelaunchCorpus())
        desk._record_prelaunch_launch("mint", {
            "creator": "dev", "timestamp": T0,
            "funding_transfers": [{"from": "whale", "to": "dev"},
                                  {"from": "whale", "to": "dev"},
                                  {"from": "dev", "to": "other"}]})
        # Deduped, and the deployer funding someone else is not a funder of
        # the deployer.
        assert [entity for entity, _ in model.infrastructure] == ["whale"]

    def test_no_transfers_means_no_funders_rather_than_a_guess(self):
        assert PrelaunchFeed.prelaunch_funders({"creator": "dev"}, "dev") == []

    def test_social_creation_is_deliberately_never_fed(self):
        """There is no verified public wallet-to-social source, so an edge
        written here would be an identity claim the desk cannot support."""
        model = self._model()
        desk = self._desk(model, PrelaunchCorpus())
        desk._record_prelaunch_launch(
            "mint", {"creator": "dev", "uri": "ipfs://a", "timestamp": T0})
        assert model.social == []

    def test_the_launch_labels_the_row_before_it_feeds_new_signals(self):
        """Otherwise this launch's own evidence enters the features of the
        row that is supposed to have predicted it."""
        model = self._model()
        corpus = PrelaunchCorpus(horizon_s=3600.0)
        corpus.observe("dev", FEATURES, observed_at=T0)
        desk = self._desk(model, corpus)
        desk._record_prelaunch_launch(
            "mint", {"creator": "dev", "uri": "ipfs://a", "timestamp": T0 + 60})
        row = corpus.resolved_rows()[0]
        assert row.launched_within_1h is True
        assert row.features == FEATURES

    def test_harvest_opens_rows_from_the_models_own_snapshot(self):
        corpus = PrelaunchCorpus()
        desk = self._desk(self._model(), corpus)
        assert desk.harvest_prelaunch_rows(now=T0) == 1
        assert corpus.size == 1

    def test_training_is_refused_and_says_why_on_a_thin_corpus(self):
        model = self._model()
        desk = self._desk(model, PrelaunchCorpus())
        result = desk.train_prelaunch_predictor()
        assert result["status"] == "DATA_BLOCKED"
        assert model.trained_on is None

    def test_an_untrained_model_withholds_the_learned_probability(self):
        """Reporting 0.0 from an untrained model is how "nobody taught me
        anything" gets read as "very unlikely"."""
        desk = self._desk(self._model(), PrelaunchCorpus())
        report = desk.prelaunch_report()
        assert report["status"] == "DATA_BLOCKED"
        assert report["imminent"] == [] or all(
            item["learned_probability"] is None for item in report["imminent"])

    def test_a_trained_model_reports_the_learned_probability(self):
        model = self._model()
        model._is_trained = True
        desk = self._desk(model, PrelaunchCorpus())
        report = desk.prelaunch_report()
        assert report["status"] == "OK"
        assert report["imminent"][0]["learned_probability"] == 0.42
        assert report["imminent"][0]["heuristic_probability"] == 0.6

    def test_a_desk_without_the_subsystem_is_blocked_not_broken(self):
        assert self._desk()._record_prelaunch_launch("mint", {}) is None
        assert self._desk().prelaunch_report()["status"] == "DATA_BLOCKED"


def test_the_creation_handler_calls_the_feed():
    """The wiring, not the method. Three bugs in this codebase were methods
    that worked perfectly and were never called."""
    import ast
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / "src" / "main.py").read_text(
        encoding="utf-8")
    assert "_record_prelaunch_launch" in source
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and node.func.attr == "_record_prelaunch_launch"]
    assert len(calls) == 1


def test_the_training_loop_harvests_and_trains():
    import ast
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / "src" / "runtime"
              / "training.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    loop = next(node for node in ast.walk(tree)
                if isinstance(node, ast.AsyncFunctionDef)
                and node.name == "_training_loop")
    text = ast.unparse(loop)
    assert "harvest_prelaunch_rows" in text
    assert "train_prelaunch_predictor" in text

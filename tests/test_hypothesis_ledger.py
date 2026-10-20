"""Selection was being priced over a population nobody had enumerated.

`HypothesisRegistry` and `TrialLedger` both existed and neither was ever
constructed, so all eight of their public methods were dead: no hypothesis was
registered, no trial recorded, no multiplicity group declared, and nothing was
ever written to the graveyard.

That is worse than an unused class. The gauntlet's CSCV pass exists to price
SELECTION -- how many candidates were tried for the one that won -- and with
nothing counting the candidates, the probability of backtest overfitting was
computed against a denominator of one.
"""

import time

import pytest

from src.strategies.champion_challenger import (
    ChampionChallengerFramework, HypothesisSpec, ModelStatus, TrialResult)


def _spec(hid, family="tail", features=("a", "b", "c"), target="p_10x"):
    return HypothesisSpec(
        hypothesis_id=hid, mechanism="m", target=target, features=list(features),
        feature_hash="fh", model_type="gbdt", model_params={}, training_window="30d",
        threshold=0.5, sizing_rule={}, exit_rule={}, execution_policy={},
        fakeability={}, cost_model={}, falsifier="f", kill_thesis="k",
        source_provenance="s", trial_family=family, created_at=time.time())


def _framework():
    return ChampionChallengerFramework(state_path=None)


class TestEveryCandidateIsCounted:
    def test_submitting_registers_the_hypothesis(self):
        framework = _framework()
        framework.submit_hypothesis(_spec("h1"))
        assert framework.registry.get("h1") is not None
        assert [h.hypothesis_id for h in framework.registry.get_family("tail")] == ["h1"]

    def test_the_family_counts_every_member_not_just_the_winner(self):
        framework = _framework()
        for index in range(40):
            framework.submit_hypothesis(_spec(f"h{index}"))
        assert framework.get_stats()["family_members"]["tail"] == 40

    def test_similar_hypotheses_form_a_multiplicity_group(self):
        """Two hypotheses over the same features are one trial repeated."""
        framework = _framework()
        framework.submit_hypothesis(_spec("h1", features=("a", "b", "c")))
        framework.submit_hypothesis(_spec("h2", features=("a", "b", "c")))
        groups = framework.get_stats()["multiplicity_groups"]
        assert groups.get("tail", 0) >= 2

    def test_unrelated_features_do_not_form_a_group(self):
        framework = _framework()
        framework.submit_hypothesis(_spec("h1", features=("a", "b", "c")))
        framework.submit_hypothesis(_spec("h2", features=("x", "y", "z")))
        assert framework.get_stats()["multiplicity_groups"] == {}


class TestEveryTrialIsRecorded:
    def _result(self, hid, passed, elogw=0.01):
        return TrialResult(
            hypothesis_id=hid, stage="shadow", samples=500,
            metrics={"elogw": elogw}, oos_metrics={}, portfolio_impact=0.001,
            passed=passed, timestamp=time.time())

    def test_a_trial_reaches_the_ledger_with_its_family(self):
        framework = _framework()
        framework.submit_hypothesis(_spec("h1"))
        framework.record_trial_result(self._result("h1", True))
        trials = framework.trial_ledger.get_recent_trials(24)
        assert len(trials) == 1
        assert trials[0]["family"] == "tail"
        assert trials[0]["passed"] is True

    def test_the_pass_rate_is_the_denominator_selection_needs(self):
        framework = _framework()
        framework.submit_hypothesis(_spec("h1"))
        for index in range(39):
            framework.record_trial_result(self._result("h1", False))
        framework.record_trial_result(self._result("h1", True))
        stats = framework.get_stats()["families"]["tail"]
        assert stats["total"] == 40
        assert stats["passed"] == 1
        assert stats["pass_rate"] == pytest.approx(0.025)

    def test_an_empty_family_reports_a_count_rather_than_a_rate(self):
        framework = _framework()
        assert framework.trial_ledger.get_family_stats("nothing") == {"count": 0}

    def test_old_trials_fall_out_of_the_recent_window(self):
        framework = _framework()
        framework.submit_hypothesis(_spec("h1"))
        stale = self._result("h1", True)
        framework.record_trial_result(
            TrialResult(**{**stale.__dict__, "timestamp": time.time() - 90000}))
        assert framework.trial_ledger.get_recent_trials(24) == []


class TestWhatFailedIsEvidenceToo:
    def test_retiring_writes_to_the_graveyard(self):
        """Otherwise the same idea is rediscovered, retested, and counted as
        a fresh trial every time."""
        framework = _framework()
        framework.submit_hypothesis(_spec("h1"))
        framework._retire_hypothesis("h1", "decayed")
        assert framework.hypotheses["h1"].status == ModelStatus.RETIRED.value
        assert framework.registry.graveyard

    def test_the_graveyard_entry_carries_the_family_it_died_in(self):
        framework = _framework()
        framework.submit_hypothesis(_spec("h1"))
        framework.record_trial_result(TrialResult(
            hypothesis_id="h1", stage="shadow", samples=10, metrics={},
            oos_metrics={}, portfolio_impact=0.0, passed=False,
            timestamp=time.time()))
        framework._retire_hypothesis("h1", "failed")
        entry = framework.registry.graveyard[-1]
        assert entry["final_metrics"]["total"] == 1


def test_the_framework_constructs_both_ledgers():
    """The wiring, not the methods. Both classes worked perfectly and were
    never instantiated anywhere in the repository."""
    framework = _framework()
    assert framework.registry is not None
    assert framework.trial_ledger is not None

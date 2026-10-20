"""Optional replay configuration never invents proof or cheap resource costs."""
import json
import time
from types import SimpleNamespace

from src.runtime.wiring import MODEL_HYPOTHESIS_ID, SubsystemWiring
from src.strategies.champion_challenger import ChampionChallengerFramework
from test_scientific_core import spec


def test_missing_disabled_or_incomplete_replay_stays_data_blocked(tmp_path):
    for settings in ({}, {"research_replay": {"enabled": False}}, {"research_replay": {"enabled": True}}):
        desk = SimpleNamespace(global_config=settings, champion_challenger=ChampionChallengerFramework())
        assert SubsystemWiring._canonical_replay_options(desk) == {}
        assert desk.research_replay_configuration["status"] in {"DATA_BLOCKED", "DISABLED"}


def test_explicit_manifest_corpus_and_costs_create_bounded_canonical_handler(tmp_path):
    corpus = tmp_path / "episodes"
    corpus.mkdir()
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({"corpus_hash": "declared-corpus-hash", "train_end": 1,
        "validation_end": 2, "test_end": 3, "frozen_at": 0.5}))
    settings = {"enabled": True, "episode_path": str(corpus), "split_manifest_path": str(manifest),
        "work_bound": {"max_episodes": 1000, "max_family_variants": 16, "max_bootstrap_draws": 10_000_000},
        "resource_costs": {"inference": 0, "data": 0, "engineering": 0, "conversion": 0, "gauntlet": 3},
        "pass_resource_units": 4}
    desk = SimpleNamespace(global_config={"research_replay": settings}, champion_challenger=ChampionChallengerFramework())
    options = SubsystemWiring._canonical_replay_options(desk)
    assert options["gauntlet_handler"].episode_path == corpus
    assert options["gauntlet_work_bound"] == settings["work_bound"]
    assert options["gauntlet_costs"] == settings["resource_costs"]
    assert desk.research_replay_configuration["status"] == "CONFIGURED_NOT_EVALUATED"
    settings["resource_costs"].pop("data")
    assert SubsystemWiring._canonical_replay_options(desk) == {}
    assert desk.research_replay_configuration["status"] == "DATA_BLOCKED"


def test_passed_statistical_model_report_is_not_silently_canonical_when_legacy():
    framework = ChampionChallengerFramework()
    legacy = spec(MODEL_HYPOTHESIS_ID, exact=False)
    framework.submit_hypothesis(legacy)
    framework.freeze_hypothesis(MODEL_HYPOTHESIS_ID)
    desk = SimpleNamespace(champion_challenger=framework)
    report = {"status": "PASSED", "net_elogw_proxy": 0.1, "oos_samples": 1000,
              "mean_brier_skill": 0.2, "created_at": time.time()}
    assert SubsystemWiring._register_model_validation(desk, report) is False
    assert report["scientific_binding_status"] == "DATA_BLOCKED"
    assert framework.hypotheses[MODEL_HYPOTHESIS_ID].status == "data_blocked"
    assert not framework.shadow_models and not framework.is_live(MODEL_HYPOTHESIS_ID)

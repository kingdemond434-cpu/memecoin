"""Collapse independent breadth and duplicate work, never trial ancestry."""
import copy

import pytest

from src.research.research_contracts import canonical_candidate_identity
from src.strategies.champion_challenger import ChampionChallengerFramework
from test_scientific_core import spec, bound_trial
from test_candidate_replay import setup, replay_spec


def test_program_identity_ignores_ids_clocks_attribution_and_genealogy():
    first = spec().exact_candidate
    second = copy.deepcopy(first)
    second.update(candidate_id="rediscovered", created_at=first["created_at"] + 100,
                  agent_identity="another-model", parent_ids=["another-parent"],
                  source_lineage=["another-public-discovery"], trial_family="arbitrary-label")
    second["data_inputs"][0]["source"] = "another-public-attribution"
    assert canonical_candidate_identity(first) == canonical_candidate_identity(second)


def test_mapped_input_and_feature_aliases_do_not_manufacture_a_new_program():
    first = replay_spec().exact_candidate
    second = copy.deepcopy(first)
    second["data_inputs"][0]["name"] = "renamed_raw_input"
    second["feature_program"] = {"renamed_feature": {"op": "identity", "input": "renamed_raw_input"}}
    second["exact_entry"]["field"] = "renamed_feature"
    assert canonical_candidate_identity(first) == canonical_candidate_identity(second)


def test_numeric_neighbours_share_one_parameter_niche_but_keep_distinct_programs():
    first = spec().exact_candidate
    second = copy.deepcopy(first)
    second["exact_entry"]["value"] = 4
    one, two = canonical_candidate_identity(first), canonical_candidate_identity(second)
    assert one["program_signature"] != two["program_signature"]
    assert one["parameter_family_signature"] == two["parameter_family_signature"]


@pytest.mark.parametrize("field,value", [
    ("mechanism", "creator distribution rather than independent buyers"),
    ("chain", "ethereum"), ("protocol", "another-launchpad"),
    ("lifecycle_stage", "post-migration"),
])
def test_distinct_mechanism_chain_protocol_and_stage_never_collapse(field, value):
    first = spec().exact_candidate
    second = copy.deepcopy(first)
    second[field] = value
    one, two = canonical_candidate_identity(first), canonical_candidate_identity(second)
    assert one["program_signature"] != two["program_signature"]
    assert one["parameter_family_signature"] != two["parameter_family_signature"]


def test_different_rule_operator_or_feature_graph_does_not_share_parameter_niche():
    first = replay_spec().exact_candidate
    for mutation in ("rule", "mapping"):
        second = copy.deepcopy(first)
        if mutation == "rule":
            second["exact_entry"]["op"] = "lte"
        else:
            second["data_inputs"][0]["snapshot_key"] = "sell_count"
        assert canonical_candidate_identity(first)["parameter_family_signature"] != canonical_candidate_identity(second)["parameter_family_signature"]


def test_indices_dispositions_raw_trials_and_representatives_survive_restart(tmp_path):
    path = tmp_path / "science.json"
    framework = ChampionChallengerFramework(state_path=str(path))
    for hid, threshold in (("original", 3), ("duplicate", 3), ("neighbour", 4)):
        hypothesis = spec(hid)
        hypothesis.exact_candidate["exact_entry"]["value"] = threshold
        framework.submit_hypothesis(hypothesis)
        framework.freeze_hypothesis(hid)
        framework.record_trial_result(bound_trial(framework, hid))
    assert framework.candidate_niche("original")["disposition"] == "NOVEL_PROGRAM"
    assert framework.candidate_niche("duplicate")["disposition"] == "DUPLICATE"
    assert framework.candidate_niche("neighbour")["disposition"] == "PARAMETER_NEIGHBOUR"
    framework._retire_hypothesis("original", "genuine no-edge")
    restored = ChampionChallengerFramework(state_path=str(path))
    report = restored.get_stats()
    assert report["raw_exact_variants"] == 3
    assert report["effective_executable_programs"] == 2
    assert report["effective_independent_breadth"] == 1
    assert report["families"]["flow"]["total"] == 3
    assert restored.candidate_niche("neighbour")["representative"] == "original"
    assert set(restored.candidate_niche("duplicate")["program_members"]) == {"original", "duplicate"}
    assert restored.registry.graveyard[0]["reason"] == "genuine no-edge"


def test_duplicate_replay_work_is_reused_but_every_frozen_variant_still_enters_selection(tmp_path):
    framework, original, handler = setup(tmp_path)
    duplicate = replay_spec("duplicate")
    framework.submit_hypothesis(duplicate)
    framework.freeze_hypothesis("duplicate")
    result = handler(original.exact_candidate)
    receipt = result["receipt"]
    assert result["status"] == "HISTORICAL_SURVIVOR", result
    assert receipt["validation_work"] == {"raw_frozen_versions": 2,
        "unique_executable_programs": 1, "duplicate_replays_reused": 1}
    for partition in receipt["results"].values():
        assert len([name for name in partition if name != "control_all_launches"]) == 2
        assert all(partition[name]["pbo"] is not None for name in receipt["expected_variants"])

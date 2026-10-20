"""Training producers preserve PIT, multiplicity and frozen repair ancestry."""
import copy
import time
from types import SimpleNamespace

import pytest

from src.research.descendant_generator import diagnose_failure, generate_descendants
from src.research.machine_search import MachineSearch, SearchBudget, SearchDataError, SEARCH_METHODS, TARGET_SEMANTICS
from src.research.mechanism_labs import MechanismLaboratory, MECHANISM_LABS, STRATEGY_FAMILIES, MECHANISM_ALIASES, EXECUTION_FIELDS
from src.research.research_contracts import validate_exact_candidate
from src.strategies.champion_challenger import ChampionChallengerFramework
from test_scientific_core import exact_candidate, spec


def training_fixture(now=None, *, graph=False):
    now = now or time.time()
    names = MECHANISM_LABS["bonding_curve"]["observables"]
    contracts = {name: {"dtype": "float", "unit": "count/second" if i == 0 else "fraction",
        "source": "native-fixture", "point_in_time": True, "event_timestamp": "event_at",
        "availability_timestamp": "available_at", "snapshot_group": "curve",
        "snapshot_key": name} for i, name in enumerate(names)}
    contracts["mint_authority_active"] = {**contracts[names[0]], "dtype": "bool", "unit": "boolean",
        "snapshot_key": "mint_authority_active"}
    data = {"partition": "training", "target_semantics": TARGET_SEMANTICS, "rows": [
        {"row_id": str(i), "episode_id": "episode-" + str(i // 4),
         "event_at": now - 100 + i, "available_at": now - 99 + i,
         "decision_at": now - 98 + i, "outcome_at": now - 97 + i,
         "features": {names[0]: float(i + 1), names[1]: 0.1 + i / 32,
                      "mint_authority_active": False}, "target": 0.02 if i >= 8 else -0.01,
         "costs_observed": True, "route_observed": True, "fill_observed": True} for i in range(16)]}
    if graph:
        data["graph"] = {"edges": [{"from": a, "to": b, "source": "native-fixture", "partition": "training",
            "event_at": now - 20, "available_at": now - 19} for a, b in (("a", "b"), ("b", "c"), ("d", "e"))]}
    base = exact_candidate()
    execution = {key: copy.deepcopy(base[key]) for key in EXECUTION_FIELDS}
    execution["exit_rule"] = {"op": "timeout", "value": 60}
    native = {"chain": "solana", "protocol": "pump", "event_schema_version": "fixture-v1",
              "decoder_fixture_ref": "evidence:decoder", "buy_simulation_ref": "evidence:buy",
              "sell_simulation_ref": "evidence:sell"}
    return {"lab_id": "bonding_curve", "context": {"observable_contracts": contracts,
        "execution_contract": execution, "native_execution_contract": native, "training_data": data,
        "search_methods": list(SEARCH_METHODS), "search_budget": {"max_trials": 36}}}, now


def frozen_parent(framework):
    parent = spec("parent")
    framework.submit_hypothesis(parent)
    assert framework.freeze_hypothesis(parent.hypothesis_id)
    return parent


def test_all_catalog_recipes_and_miner_aliases_produce_concrete_questions_and_requests():
    service = MechanismLaboratory(ChampionChallengerFramework())
    assert len(MECHANISM_LABS) == 17 and len(STRATEGY_FAMILIES) == 24 and len(MECHANISM_ALIASES) == 11
    expanded = service.consume("DEEPENING_TASK", {}, ["evidence:seed"], time.time())
    assert {child["payload"]["lab_id"] for child in expanded["children"]} == set(MECHANISM_LABS)
    for family, lab_id in STRATEGY_FAMILIES.items():
        result = service.consume("MECHANISM_LAB", {"lab_id": lab_id, "strategy_family": family}, ["evidence:seed"], time.time())
        assert result["disposition"] == "DATA_REQUEST"
        assert len(result["detail"]["questions"]) == 5
        requested = {child["payload"]["variable"] for child in result["children"]}
        assert set(MECHANISM_LABS[lab_id]["observables"]) <= requested
        assert {"native_execution_contract", "exact_execution_policy", "chronological_training_observations"} <= requested
        assert all(child["payload"]["falsifier"] and child["payload"]["validation_plan"] for child in result["children"])
    for alias, lab_id in MECHANISM_ALIASES.items():
        result = service.consume("MECHANISM_HYPOTHESIS", {"mechanism": alias}, ["evidence:seed"], time.time())
        assert result["detail"]["lab_id"] == lab_id
    unknown = service.consume("MECHANISM_HYPOTHESIS", {"mechanism": "profitable magic"}, ["evidence:seed"], time.time())
    assert unknown["children"][0]["payload"]["variable"] == "economic_mechanism_definition"
    assert service.report()["measured_incremental_log_growth"] is None


def test_every_machine_path_runs_with_freeze_before_record_and_bounded_ancestry():
    payload, now = training_fixture(graph=True)
    context, frozen, records = payload["context"], {}, []
    def freeze(program):
        assert all(parent in frozen for parent in program["search_parents"])
        hid = "frozen-" + program["program_id"]
        frozen[hid] = copy.deepcopy(program)
        return hid
    def record(hid, result):
        assert hid in frozen and result["program_id"] == frozen[hid]["program_id"]
        assert "training_score" not in frozen[hid]
        records.append(result)
    result = MachineSearch(budget=SearchBudget(max_trials=36)).run(context["training_data"], context["observable_contracts"],
        now=now, freeze=freeze, record=record)
    assert set(result["methods"]) == set(SEARCH_METHODS)
    assert 0 < len(records) <= 36 and len(records) == len(frozen)
    assert {r["method"] for r in records} >= set(SEARCH_METHODS) - {"state_machine", "graph_search"}
    assert any(row["search_parents"] for row in records if row["method"] == "genetic_programming")
    assert result["methods"]["symbolic_regression"]["fits"]
    assert result["methods"]["bayesian_optimization"]["acquisitions"]
    assert "UNIDENTIFIED" in result["methods"]["causal_discovery"]["identification"]
    state = next(row for row in result["requests"] if row["variable"].startswith("prior_state_"))
    assert sum(row["count"] for row in state["construction"]["transition_counts"]) == 12
    graph = next(row for row in result["requests"] if row["variable"] == "actor_component_size")
    assert graph["construction"]["components"] == [["a", "b", "c"], ["d", "e"]]
    assert result["edge_evidence"] == "UNMEASURED"


def test_exact_candidates_all_trials_niches_and_retry_survive_restart(tmp_path):
    path = str(tmp_path / "desk.json")
    framework = ChampionChallengerFramework(state_path=path)
    payload, now = training_fixture()
    service = MechanismLaboratory(framework)
    result = service.consume("MACHINE_SEARCH", payload, ["evidence:seed"], now)
    assert result["disposition"] == "EXACT_CANDIDATE", result
    exact = [child for child in result["children"] if child["kind"] == "EXACT_CANDIDATE"]
    assert len(exact) == len(framework.trial_results) == len(framework.hypotheses)
    for child in exact:
        candidate = child["payload"]["candidate"]
        assert validate_exact_candidate(candidate).ok
        parent = framework.hypotheses[candidate["candidate_id"]]
        assert framework._exact_frozen_valid(parent)
        assert candidate["native_execution_contract"] == payload["context"]["native_execution_contract"]
        assert candidate["hard_vetoes"] == payload["context"]["execution_contract"]["hard_vetoes"]
        assert all(framework.hypotheses[p].frozen_hash == candidate["parent_frozen_hashes"][p] for p in parent.parent_ids)
    assert all(trial.stage == "TRAINING_SEARCH" and not trial.passed and trial.evidence_status == "BOUND"
               and trial.spec_hash and not trial.oos_metrics for trial in framework.trial_results)
    assert sum(framework.candidate_niche(hid)["contributes_independent_breadth"] for hid in framework.hypotheses) < len(exact)
    restarted = ChampionChallengerFramework(state_path=path)
    retry = MechanismLaboratory(restarted).consume("MACHINE_SEARCH", payload, ["evidence:seed"], now + 10)
    assert retry == result and len(restarted.trial_results) == len(exact)
    assert not restarted.champions and not restarted.shadow_models and not restarted.canary_models


@pytest.mark.parametrize("defect", ["heldout", "future", "mixed", "cost", "route", "fill", "type", "duplicate", "episode", "mapping", "authority", "native"])
def test_bad_evidence_cannot_create_or_score_a_candidate(defect):
    payload, now = training_fixture()
    context, framework = payload["context"], ChampionChallengerFramework()
    row = context["training_data"]["rows"][0]
    if defect == "heldout": row["extra"] = {"heldout_result": 1}
    if defect == "future": row["available_at"] = row["decision_at"] + 1
    if defect == "mixed": row["partition"] = "test"
    if defect in {"cost", "route", "fill"}: row[{"cost": "costs_observed", "route": "route_observed", "fill": "fill_observed"}[defect]] = False
    if defect == "type": row["features"]["curve_progress"] = "0.5"
    if defect == "duplicate": row["row_id"] = "1"
    if defect == "episode": del row["episode_id"]
    if defect == "mapping": context["observable_contracts"]["curve_progress"]["snapshot_key"] = "final_outcome"
    if defect == "authority": payload["live_authority"] = "LIVE"
    if defect == "native": context["native_execution_contract"]["chain"] = "ethereum"
    result = MechanismLaboratory(framework).consume("MACHINE_SEARCH", payload, ["evidence:seed"], now)
    assert result["disposition"] in {"BLOCKED", "DATA_REQUEST"}
    assert not framework.hypotheses and not framework.trial_results


def test_safe_missing_execution_no_guessed_fees_or_native_units():
    payload, now = training_fixture()
    del payload["context"]["execution_contract"]["fee_policy"]
    framework = ChampionChallengerFramework()
    result = MechanismLaboratory(framework).consume("MACHINE_SEARCH", payload, ["evidence:seed"], now)
    request = next(child["payload"] for child in result["children"] if child["payload"]["variable"] == "exact_execution_policy")
    assert "fee_policy" in request["required_fields"]
    assert not framework.hypotheses


def test_acquisition_requests_resume_original_recipe_and_merge_only_real_joined_features():
    payload, now = training_fixture()
    context = payload["context"]
    variable = "curve_net_buy_velocity"
    descriptor = context["observable_contracts"].pop(variable)
    points = []
    for row in context["training_data"]["rows"]:
        points.append({"variable": variable, "value": row["features"].pop(variable), "unit": descriptor["unit"],
            "source_id": descriptor["source"], "chain": "solana", "partition": "training",
            "row_id": row["row_id"], "episode_id": row["episode_id"], "event_at": row["event_at"],
            "available_at": row["available_at"], "observed_at": row["available_at"],
            "observable_contract": descriptor, "evidence_ref": "evidence:acquired"})
    service = MechanismLaboratory(ChampionChallengerFramework())
    missing = service.consume("MACHINE_SEARCH", payload, ["evidence:seed"], now)
    request = next(child["payload"] for child in missing["children"] if child["payload"]["variable"] == variable)
    assert request["resume_kind"] == "MACHINE_SEARCH" and request["partition"] == "training" and request["chain"] == "solana"
    resumed = {**request["resume_payload"], "observed_data": {"variable": variable, "rows": points, "partitions": ["training"]}}
    result = service.consume(request["resume_kind"], resumed, ["evidence:seed", "evidence:acquired"], now)
    assert result["disposition"] == "EXACT_CANDIDATE", result
    candidate = next(child["payload"]["candidate"] for child in result["children"] if child["kind"] == "EXACT_CANDIDATE")
    assert next(item for item in candidate["data_inputs"] if item["name"] == variable)["snapshot_key"] == variable
    for child in result["children"]:
        if child["kind"] == "DATA_REQUEST":
            assert child["payload"]["resume_kind"] == "MACHINE_SEARCH"
            assert variable in child["payload"]["resume_payload"]["context"]["observable_contracts"]
    for point in points: point.pop("observable_contract")
    resumed["observed_data"]["rows"] = points
    no_mapping = MechanismLaboratory(ChampionChallengerFramework()).consume("MACHINE_SEARCH", resumed, ["evidence:seed"], now)
    assert any(child["payload"]["variable"] == "typed_native_mapping:" + variable for child in no_mapping["children"])
    points[0]["observable_contract"] = descriptor
    for point in points[1:]: point["observable_contract"] = descriptor
    points[0]["episode_id"] = "different-episode"
    incomplete = MechanismLaboratory(ChampionChallengerFramework()).consume("MACHINE_SEARCH", resumed, ["evidence:seed"], now)
    assert incomplete["disposition"] == "DATA_REQUEST"
    assert any(child["payload"]["variable"] == variable for child in incomplete["children"])


def test_resumed_observations_reject_conflicts_and_test_partitions():
    payload, now = training_fixture()
    descriptor = payload["context"]["observable_contracts"]["curve_progress"]
    row = payload["context"]["training_data"]["rows"][0]
    point = {"variable": "curve_progress", "value": 100., "unit": descriptor["unit"], "source_id": descriptor["source"],
        "chain": "solana", "partition": "training", "row_id": row["row_id"], "episode_id": row["episode_id"],
        "event_at": row["event_at"], "available_at": row["available_at"], "observed_at": row["available_at"]}
    payload["observed_data"] = {"variable": "curve_progress", "rows": [point], "partitions": ["training"]}
    service = MechanismLaboratory(ChampionChallengerFramework())
    assert service.consume("MACHINE_SEARCH", payload, ["evidence:seed"], now)["disposition"] == "BLOCKED"
    point["value"] = row["features"]["curve_progress"]
    payload["observed_data"]["partitions"] = ["test"]
    assert service.consume("MACHINE_SEARCH", payload, ["evidence:seed"], now)["disposition"] == "BLOCKED"


def test_partial_checkpoint_failure_does_not_double_count_training_trial(tmp_path, monkeypatch):
    path = str(tmp_path / "desk.json")
    framework = ChampionChallengerFramework(state_path=path)
    payload, now = training_fixture()
    payload["context"]["search_methods"] = ["grammar"]
    payload["context"]["search_budget"]["max_trials"] = 2
    service = MechanismLaboratory(framework)
    def fail(): raise OSError("interrupted component checkpoint")
    monkeypatch.setattr(service, "_checkpoint", fail)
    with pytest.raises(OSError): service.consume("MACHINE_SEARCH", payload, ["evidence:seed"], now)
    assert len(framework.trial_results) == 1
    restarted = ChampionChallengerFramework(state_path=path)
    result = MechanismLaboratory(restarted).consume("MACHINE_SEARCH", payload, ["evidence:seed"], now + 5)
    assert result["disposition"] == "EXACT_CANDIDATE"
    assert len(restarted.trial_results) == 2 and len(restarted.hypotheses) == 2


def test_work_bounds_and_direct_search_heldout_rejected():
    for options in ({"max_trials": 97}, {"max_depth": 4}, {"max_features": 9}, {"max_graph_edges": 257}, {"max_rows": True}):
        with pytest.raises(ValueError): SearchBudget(**options)
    payload, now = training_fixture()
    context = payload["context"]
    context["training_data"]["metadata"] = {"heldout_returns": [1]}
    with pytest.raises(SearchDataError, match="contamination"):
        MachineSearch().run(context["training_data"], context["observable_contracts"], now=now,
            freeze=lambda _: pytest.fail("contaminated freeze"), record=lambda *_: None)


def test_failure_taxonomy_requires_attribution_and_observed_enabling_change():
    assert diagnose_failure({"price_return": -1}) is None
    assert diagnose_failure({"net_before_cost": 0.02, "net_after_cost": -0.01}) == "cost_death"
    framework = ChampionChallengerFramework()
    parent = frozen_parent(framework)
    service, now = MechanismLaboratory(framework), parent.frozen_at + 20
    payload = {"parent_id": parent.hypothesis_id, "parent_frozen_hash": parent.frozen_hash,
        "attribution": {"net_before_cost": 0.02, "net_after_cost": -0.01}}
    result = service.consume("FAILURE_DESCENDANT", payload, ["evidence:cost"], now)
    assert result["disposition"] == "NEGATIVE_KNOWLEDGE"
    assert result["children"][0]["payload"]["variable"] == "enabling_change_for_cost_death"
    payload["enabling_change"] = {"change_id": "measured-route-v2", "addresses_diagnosis": "cost_death",
        "evidence_refs": ["evidence:cost"], "observed_at": parent.frozen_at + 10,
        "description": "new native route reduces observed total costs"}
    payload["proposed_patch"] = {"fee_policy": {"max_fee_lamports": 4000}}
    result = service.consume("FAILURE_DESCENDANT", payload, ["evidence:cost"], now)
    assert result["disposition"] == "FAILURE_DESCENDANT", result
    candidate = result["children"][0]["payload"]["candidate"]
    assert candidate["parent_frozen_hashes"] == {parent.hypothesis_id: parent.frozen_hash}
    assert candidate["hard_vetoes"] == parent.exact_candidate["hard_vetoes"]
    assert validate_exact_candidate(candidate).ok and len(framework.hypotheses) == 1
    promoted_legacy = service.consume("MECHANISM_HYPOTHESIS", payload, ["evidence:cost"], now)
    assert promoted_legacy["children"][0]["payload"]["candidate"] == candidate
    for patch in ({"hard_vetoes": {"op": "constant", "value": False}}, {"chain": "ethereum"}):
        payload["proposed_patch"] = patch
        assert service.consume("FAILURE_DESCENDANT", payload, ["evidence:cost"], now)["disposition"] == "BLOCKED"
    payload["proposed_patch"] = {"fee_policy": {"max_fee_lamports": 4000}}
    payload["enabling_change"]["observed_at"] = parent.frozen_at - 1
    assert service.consume("FAILURE_DESCENDANT", payload, ["evidence:cost"], now)["disposition"] == "NEGATIVE_KNOWLEDGE"


def test_genuine_no_edge_is_negative_knowledge_and_survivor_ablation_preserves_safety():
    framework = ChampionChallengerFramework()
    parent = frozen_parent(framework)
    service, now = MechanismLaboratory(framework), parent.frozen_at + 20
    payload = {"parent_id": parent.hypothesis_id, "parent_frozen_hash": parent.frozen_hash,
        "attribution": {"canonical_controls_complete": True, "canonical_receipt_ref": "evidence:fail", "net_elogw_lower": -0.01}}
    result = service.consume("FAILURE_DESCENDANT", payload, ["evidence:fail"], now)
    assert result["disposition"] == "NEGATIVE_KNOWLEDGE" and not result["children"]
    payload.pop("attribution")
    payload["survival_evidence"] = {"verdict": "SURVIVOR", "spec_hash": parent.frozen_hash, "evidence_ref": "evidence:survivor"}
    result = service.consume("SURVIVOR_DESCENDANT", payload, ["evidence:survivor"], now)
    assert result["disposition"] == "SURVIVOR_DESCENDANT", result
    exact = [child for child in result["children"] if child["kind"] == "EXACT_CANDIDATE"]
    assert len(exact) == 1
    candidate = exact[0]["payload"]["candidate"]
    assert candidate["exact_entry"] == {"op": "constant", "value": True}
    assert candidate["hard_vetoes"] == parent.exact_candidate["hard_vetoes"]
    assert {child["kind"] for child in result["children"]} == {"EXACT_CANDIDATE", "DATA_REQUEST", "VALIDATION_ATTACK"}
    payload["parent_frozen_hash"] = "corrupted"
    assert service.consume("SURVIVOR_DESCENDANT", payload, ["evidence:survivor"], now)["disposition"] == "BLOCKED"


def test_legacy_descendant_proposals_remain_compatible_and_bounded():
    findings = [SimpleNamespace(leak="execution_miss", evidence={"rejection_reason": "no route"}, forgone_log_growth=.02)]
    result = generate_descendants(findings)
    assert result["status"] == "OK"
    assert result["hypotheses"][0]["diagnosis"] == "route_fill_death"
    assert result["hypotheses"][0]["resurrection"] == "BLOCKED_UNTIL_NAMED_ENABLING_CHANGE"
    consumed = MechanismLaboratory(ChampionChallengerFramework()).consume("MECHANISM_HYPOTHESIS",
        result["hypotheses"][0], ["evidence:legacy"], time.time())
    assert consumed["children"][0]["payload"]["variable"] == "frozen_parent_failure_attribution"
    assert generate_descendants(findings * 1025)["status"] == "DATA_BLOCKED"
    with pytest.raises(ValueError): generate_descendants(findings, top_n=100)

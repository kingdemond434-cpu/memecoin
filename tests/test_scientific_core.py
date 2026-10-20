"""Scientific identity, restart and selection controls across module boundaries."""
import asyncio
import copy
import json
import time

from src.research.gauntlet import Gauntlet, Observation, Verdict, probability_of_backtest_overfitting
from src.research.research_contracts import canonical_candidate_hash, validate_exact_candidate
from src.strategies.champion_challenger import ChampionChallengerFramework, HypothesisSpec, TrialResult


def exact_candidate(candidate_id="c1", parents=None):
    return {
        "schema_version": 1, "candidate_id": candidate_id, "agent_identity": "fixture-agent",
        "created_at": time.time(), "parent_ids": parents or [], "trial_family": "flow",
        "chain": "solana", "protocol": "pump", "lifecycle_stage": "curve",
        "mechanism": "independent buyers absorb distribution", "actor_incentive": "buy scarce curve supply",
        "data_inputs": [{"name": "independent_buyers", "source": "native-event",
                         "point_in_time": True, "event_timestamp": "event_at",
                         "availability_timestamp": "received_at"}],
        "feature_program": {"buyers": {"op": "identity", "input": "independent_buyers"}},
        "universe_rule": {"op": "constant", "value": True},
        "entry_horizon_seconds": 10, "max_decision_latency_ms": 250,
        "exact_entry": {"op": "gte", "field": "buyers", "value": 3},
        "hard_vetoes": {"op": "eq", "field": "mint_authority_active", "value": True},
        "size_rule": {"op": "fraction", "value": 0.001},
        "fee_policy": {"max_fee_lamports": 5000},
        "landing_assumptions": {"model": "measured", "fill_probability": 0.5},
        "exit_rule": {"op": "take_profit", "value": 1.5},
        "emergency_exit": {"op": "stop_loss", "value": 0.1},
        "holding_time_model": "timeout after 60 seconds", "expected_behavior": "positive net log growth",
        "failure_modes": ["crowded followers"], "falsifier": "nonpositive executable held-out return",
        "alternative_explanations": ["market beta"], "execution_assumptions": {"route": "curve-replay"},
        "novelty_signature": "independent-buyers:curve", "source_lineage": ["fixture-public-event"],
    }


def spec(hid="c1", *, exact=True, parents=None):
    return HypothesisSpec(
        hypothesis_id=hid, mechanism="flow", target="net_elogw", features=["buyers"],
        feature_hash="feature-v1", model_type="rule", model_params={}, training_window="chronological",
        threshold=3, sizing_rule={"fraction": 0.001}, exit_rule={"timeout": 60},
        execution_policy={"mode": "shadow"}, fakeability={}, cost_model={"fee_bps": 30},
        falsifier="nonpositive held-out return", kill_thesis="decay", source_provenance="fixture",
        trial_family="flow", created_at=time.time(), parent_ids=parents or [],
        agent_identity="fixture-agent", exact_candidate=exact_candidate(hid, parents) if exact else None)


def trial(hid="c1", **kwargs):
    return TrialResult(hypothesis_id=hid, stage="CHRONOLOGICAL_OOS", samples=500,
                       metrics={"elogw": 0.1}, oos_metrics={"elogw": 0.1},
                       portfolio_impact=0.1, passed=True, timestamp=time.time(), **kwargs)


def bound_trial(framework, hid="c1"):
    return trial(hid, spec_hash=framework.hypotheses[hid].frozen_hash)


def test_l0_refuses_prose_rules_future_inputs_and_tampered_seal():
    candidate = exact_candidate()
    assert validate_exact_candidate(candidate).ok
    candidate["candidate_hash"] = canonical_candidate_hash(candidate)
    candidate["exact_entry"] = "buy smart money"
    candidate["data_inputs"][0]["point_in_time"] = False
    result = validate_exact_candidate(candidate)
    assert not result.ok
    assert "candidate content hash mismatch" in result.reasons
    assert any("executable exact_entry" in reason for reason in result.reasons)
    assert any("PIT" in reason for reason in result.reasons)


def test_freezing_is_idempotent_and_nested_tampering_rejects_trials():
    framework = ChampionChallengerFramework()
    proposal = spec()
    framework.submit_hypothesis(proposal)
    assert framework.freeze_hypothesis("c1")
    frozen_at = proposal.frozen_at
    assert framework.freeze_hypothesis("c1")
    assert proposal.frozen_at == frozen_at
    assert framework.record_trial_result(bound_trial(framework))
    proposal.exact_candidate["size_rule"]["value"] = 0.5
    assert not framework.record_trial_result(trial())
    assert framework.trial_results[-1].evidence_status == "REJECTED_BINDING"
    asyncio.run(framework._evaluate_challengers())
    assert not framework.shadow_models


def test_trial_cannot_bind_another_version_or_predate_freeze():
    framework = ChampionChallengerFramework()
    framework.submit_hypothesis(spec())
    framework.freeze_hypothesis("c1")
    assert not framework.record_trial_result(trial(spec_hash="different"))
    old = trial()
    old.timestamp = framework.hypotheses["c1"].frozen_at - 1
    assert not framework.record_trial_result(old)


def test_exact_frozen_trial_enters_shadow_and_forward_results_require_its_seal():
    framework = ChampionChallengerFramework()
    framework.submit_hypothesis(spec())
    framework.freeze_hypothesis("c1")
    assert not framework.record_trial_result(trial())
    assert framework.record_trial_result(bound_trial(framework))
    asyncio.run(framework._evaluate_challengers())
    assert "c1" in framework.shadow_models
    assert not framework.record_forward_result("c1", {"elogw": 0.1, "pnl": 1})
    assert framework.record_forward_result("c1", {"elogw": 0.1, "pnl": 1,
        "spec_hash": framework.hypotheses["c1"].frozen_hash})
    assert len(framework.shadow_models["c1"]["forward_results"]) == 1


def test_legacy_trials_remain_counted_but_cannot_promote():
    framework = ChampionChallengerFramework()
    framework.submit_hypothesis(spec(exact=False))
    framework.record_trial_result(trial())
    asyncio.run(framework._evaluate_challengers())
    assert framework.trial_ledger.get_family_stats("flow")["total"] == 1
    assert framework.hypotheses["c1"].admission_status == "LEGACY_NONEXACT"
    assert not framework.shadow_models


def test_registry_trials_graveyard_and_genealogy_survive_restart(tmp_path):
    path = tmp_path / "science.json"
    first = ChampionChallengerFramework(state_path=str(path))
    for hid, parents in (("c1", []), ("c2", ["c1"])):
        first.submit_hypothesis(spec(hid, parents=parents))
        first.freeze_hypothesis(hid)
        first.record_trial_result(bound_trial(first, hid))
    first._retire_hypothesis("c1", "latency death")
    first._save_state()
    second = ChampionChallengerFramework(state_path=str(path))
    assert second.trial_ledger.get_family_stats("flow")["total"] == 2
    assert second.trial_ledger.multiplicity_groups["flow"] == ["c1", "c2"]
    assert second.registry.genealogy["c2"] == ["c1"]
    assert second.registry.graveyard[0]["reason"] == "latency death"
    assert second.registry.get("c1") is None
    assert second.registry.get("c2") is not None


def test_superseding_keeps_all_outcome_trials_without_inheriting_them(tmp_path):
    framework = ChampionChallengerFramework(state_path=str(tmp_path / "science.json"))
    framework.submit_hypothesis(spec())
    framework.freeze_hypothesis("c1")
    framework.record_trial_result(bound_trial(framework))
    replacement = spec()
    replacement.threshold = 4
    framework.submit_hypothesis(replacement)
    framework.freeze_hypothesis("c1")
    assert len(framework.trial_results) == 1
    assert framework.registry.graveyard[-1]["reason"] == "superseded"
    asyncio.run(framework._evaluate_challengers())
    assert not framework.shadow_models


def test_v1_restore_rebuilds_counts_without_inventing_bound_evidence(tmp_path):
    path = tmp_path / "legacy.json"
    first = ChampionChallengerFramework(state_path=str(path))
    first.submit_hypothesis(spec(exact=False))
    first.record_trial_result(trial())
    payload = json.loads(path.read_text())
    payload["schema_version"] = 1
    payload.pop("registry")
    payload.pop("trial_ledger")
    path.write_text(json.dumps(payload))
    second = ChampionChallengerFramework(state_path=str(path))
    assert second.registry.get("c1") is not None
    assert second.trial_ledger.get_family_stats("flow")["total"] == 1
    assert second.trial_results[0].evidence_status == "LEGACY_UNBOUND"


def test_research_checkpoints_deep_copy_and_restore_without_authority(tmp_path):
    path = tmp_path / "science.json"
    first = ChampionChallengerFramework(state_path=str(path))
    state = {"original_equity": 100, "loss_budget": 5, "episodes": ["launch-1"]}
    first.checkpoint_research_component("tail_sleeve", state)
    state["episodes"].append("uncheckpointed")
    loaded = first.get_research_component("tail_sleeve")
    loaded["loss_budget"] = 100
    second = ChampionChallengerFramework(state_path=str(path))
    assert second.get_research_component("tail_sleeve") == {
        "original_equity": 100, "loss_budget": 5, "episodes": ["launch-1"]}
    for invalid in ({"nested": {"allow_live_trading": True}}, {"loss": float("nan")}):
        try:
            first.checkpoint_research_component("tail_sleeve", invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid research authority/nonfinite checkpoint accepted")
    assert first.get_research_component("tail_sleeve")["loss_budget"] == 5


def observations(name, keys):
    return [Observation(mechanism=name, timestamp=float(key), cohort=f"launch-{key}",
                        regime=("a", "b", "c")[key % 3], source_family="public",
                        net_return_by_latency={0.1: 0.2, 1.0: 0.2, 3.0: 0.2}) for key in keys]


def test_pbo_refuses_ordinal_alignment_of_different_launches():
    gauntlet = Gauntlet(min_observations=16, bootstrap=10, latencies=(0.1, 1.0, 3.0))
    result = gauntlet.run(observations("a", range(32)) + observations("b", range(32, 64)))
    assert all(row.pbo is None and row.verdict is Verdict.DATA_BLOCKED for row in result.values())


def test_pbo_uses_shared_cohort_time_not_missing_row_position():
    gauntlet = Gauntlet(bootstrap=10)
    first = observations("a", range(40))
    second = observations("b", range(10, 50))
    for row in first + second:
        row.net_return_by_latency[1.0] = 0.1 + (row.timestamp % 7) / 100
    grouped = {"a": first, "b": list(reversed(second))}
    expected = probability_of_backtest_overfitting([
        [row.log_growth(1.0) for row in first[10:]],
        [row.log_growth(1.0) for row in second[:30]],
    ])
    assert gauntlet._joint_pbo(grouped) == expected


def test_missing_declared_variant_or_duplicate_cohort_blocks_selection():
    gauntlet = Gauntlet(min_observations=16, bootstrap=10)
    result = gauntlet.run(observations("a", range(32)), expected_mechanisms=["a", "unreported"])
    assert result["a"].verdict is Verdict.DATA_BLOCKED
    rows = observations("a", range(32)) + observations("b", range(32))
    rows.append(copy.deepcopy(rows[0]))
    assert all(row.verdict is Verdict.DATA_BLOCKED for row in gauntlet.run(rows).values())


def test_cscv_identical_variants_share_midrank_instead_of_arbitrary_name_penalty():
    row = [0.1 + index % 3 / 100 for index in range(32)]
    assert probability_of_backtest_overfitting([row, list(row)]) == 0.0

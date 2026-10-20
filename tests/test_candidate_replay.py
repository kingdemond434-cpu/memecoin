"""Exact programs reach existing native replay without invented evidence."""
import copy
import gzip
import json
import time

import pytest

from src.research.candidate_replay import (
    CandidateReplayHandler, ReplayBlocked, _ExactFeed, _digest, compile_candidate, compile_exit, evaluate_rule,
)
from src.research.gauntlet import Gauntlet, Verdict
from src.research.gauntlet_feed import EpisodeView
from src.research.lifecycle_replay import Mark
from src.strategies.champion_challenger import ChampionChallengerFramework
from test_scientific_core import spec


DELAYS = (0.1, 1.0, 3.0)


def replay_spec(hid="c1"):
    proposal = spec(hid)
    candidate = proposal.exact_candidate
    candidate["max_decision_latency_ms"] = 10_000
    candidate["execution_assumptions"] = {"decision_offset_seconds": 0.0, "native_unit": "SOL", "equity_native": 100}
    candidate["fee_policy"] = {"max_round_trip_cost_fraction": 0.2}
    candidate["exit_rule"] = {"op": "any", "args": [
        {"op": "take_profit", "value": 1.5}, {"op": "timeout", "value": 2.0}]}
    candidate["data_inputs"][0].update(snapshot_group="flow_features", snapshot_key="buyers",
        event_timestamp="event_at", availability_timestamp="available_at")
    candidate["data_inputs"].append({"name": "mint_authority_active", "source": "native-event", "point_in_time": True,
        "event_timestamp": "event_at", "availability_timestamp": "available_at",
        "snapshot_group": "token_features", "snapshot_key": "mint_authority_active"})
    return proposal


def episode(index):
    created = 100.0 + index * 100
    route = "pump_curve"
    return {"token": f"mint-{index}", "chain": "solana", "factory": "pump", "created_at": created,
        "snapshots": {"t0": {"event_at": created, "available_at": created,
            "flow_features": {"buyers": 3}, "token_features": {"mint_authority_active": False},
            "market_features": {"sol_change_24h": 5, "meme_launch_rate_1h": 10}}},
        "market_observations": [{"timestamp": created + step, "available_at": created + step,
            "price_multiple": 1.1 ** step, "executable_sol": 5, "feasible": True, "route": route}
            for step in (0, 0.1, 1, 2, 3, 4, 5, 6, 7, 8)],
        "replay_execution": {"native_unit": "SOL", "route": route, "fills_by_latency": {
            str(delay): {"evidence_kind": "native_simulation", "source": "fixture-native-replay",
                "filled": True, "route_feasible": True, "round_trip_cost_fraction": 0.01,
                "failed_cost_fraction": 0.001} for delay in DELAYS}},
        "final_outcome": {"status": "OK"}}


def setup(tmp_path, *, second_variant=False, mutate=None):
    framework = ChampionChallengerFramework()
    proposal = replay_spec()
    framework.submit_hypothesis(proposal)
    framework.freeze_hypothesis("c1")
    if second_variant:
        other = replay_spec("c2")
        other.threshold = 4
        other.exact_candidate["exact_entry"]["value"] = 2
        framework.submit_hypothesis(other)
        framework.freeze_hypothesis("c2")
    rows = [episode(index) for index in range(54)]
    if mutate:
        mutate(rows)
    for index, row in enumerate(rows):
        path = tmp_path / "episodes" / "fixture" / f"{index:03d}.json.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(row, handle)
    split = {"corpus_hash": _digest(rows), "train_end": 1850, "validation_end": 3650,
             "test_end": 5500, "frozen_at": proposal.frozen_at - 1}
    gauntlet = Gauntlet(min_observations=2, bootstrap=10, required_regimes=1,
                       latencies=DELAYS, cost_multipliers=(1.0, 1.5, 2.0))
    return framework, proposal, CandidateReplayHandler(framework, tmp_path / "episodes",
        chronological_split=split, gauntlet=gauntlet)


def test_bounded_interpreter_preserves_unknown_and_refuses_code():
    assert evaluate_rule({"op": "gte", "field": "buyers", "value": 3}, {}) is None
    assert evaluate_rule({"op": "all", "args": [{"op": "constant", "value": False},
        {"op": "gte", "field": "missing", "value": 1}]}, {}) is False
    with pytest.raises(ReplayBlocked):
        evaluate_rule({"op": "eval", "value": "__import__('os')"}, {})


def test_entry_reads_only_mapped_pit_snapshot_and_safety_fact_vetoes():
    compiled = compile_candidate(replay_spec().exact_candidate)
    row = episode(0)
    assert compiled.mechanism.rule(EpisodeView(row)) is True
    row["snapshots"]["t10s"] = {"flow_features": {"buyers": 999999}}
    assert compiled.mechanism.rule(EpisodeView(row)) is True
    row["snapshots"]["t0"]["token_features"]["mint_authority_active"] = True
    assert compiled.mechanism.rule(EpisodeView(row)) is False
    row["snapshots"]["t0"]["available_at"] += 0.001
    assert compiled.mechanism.rule(EpisodeView(row)) is None


def test_future_feature_mapping_and_foreign_native_units_are_refused():
    candidate = replay_spec().exact_candidate
    candidate["data_inputs"][0]["snapshot_key"] = "label_10x"
    with pytest.raises(ReplayBlocked, match="future/outcome"):
        compile_candidate(candidate)
    candidate = replay_spec().exact_candidate
    candidate["chain"] = "ethereum"
    with pytest.raises(ReplayBlocked, match="native replay adapter unavailable"):
        compile_candidate(candidate)


def test_exit_preserves_earliest_exact_take_profit_emergency_and_timeout():
    candidate = replay_spec().exact_candidate
    marks = [Mark(101, 1.2, 1), Mark(102, 0.8, 1), Mark(103, 2.0, 1)]
    assert compile_exit(candidate, entry_time=100)(marks, 1).timestamp == 102
    candidate["emergency_exit"] = {"op": "constant", "value": False}
    candidate["exit_rule"] = {"op": "take_profit", "value": 1.5}
    assert compile_exit(candidate, entry_time=100)(marks, 1).timestamp == 103
    assert compile_exit(candidate, entry_time=100)(marks[:1], 1) is None
    candidate["exit_rule"] = {"op": "timeout", "value": 2}
    assert compile_exit(candidate, entry_time=100)(marks, 1).timestamp == 102
    candidate["exit_rule"] = {"op": "gte", "field": "future_peak", "value": 2}
    with pytest.raises(ReplayBlocked, match="unsupported exact exit"):
        compile_exit(candidate, entry_time=100)


def test_native_feed_honors_fees_failed_fills_routes_depth_and_chain():
    candidate = replay_spec().exact_candidate
    compiled = compile_candidate(candidate)
    row = episode(0)
    execution = row["replay_execution"]
    execution["fills_by_latency"]["0.1"]["filled"] = False
    execution["fills_by_latency"]["1.0"]["route_feasible"] = False
    feed = _ExactFeed(compiled, execution, DELAYS)
    results = [observation for observation in feed.observations([row]) if observation.mechanism == "c1"]
    assert results[0].net_return_by_latency[0.1] == -0.001
    assert results[0].net_return_by_latency[1.0] == -0.001
    assert results[0].chain == "solana"
    assert results[0].cost_fraction == 0.01
    row["market_observations"][-1]["executable_sol"] = None
    with pytest.raises(ReplayBlocked, match="route/depth/PIT"):
        CandidateReplayHandler._execution(row, candidate)


def test_exact_chronological_native_adapter_uses_canonical_gauntlet_without_promotion(tmp_path):
    framework, proposal, handler = setup(tmp_path, second_variant=True)
    result = handler(proposal.exact_candidate)
    assert result["status"] == "HISTORICAL_SURVIVOR", result
    receipt = result["receipt"]
    assert receipt["scope"] == "L0-L4_ONLY"
    assert receipt["required_next_stage"] == "L5_UNTOUCHED_FORWARD"
    assert len(receipt["expected_variants"]) == 2
    assert all("control_all_launches" in rows for rows in receipt["results"].values())
    assert all(rows[receipt["expected_variants"][0]]["pbo"] is not None for rows in receipt["results"].values())
    assert not framework.trial_results and not framework.shadow_models and not framework.champions


@pytest.mark.parametrize("missing", ["replay_execution", "market_observations"])
def test_missing_execution_evidence_blocks_instead_of_using_defaults(tmp_path, missing):
    _, proposal, handler = setup(tmp_path, mutate=lambda rows: rows[0].pop(missing))
    result = handler(proposal.exact_candidate)
    assert result["status"] == "DATA_BLOCKED"
    assert result["disposition"] == "BLOCKED"


def test_legacy_family_member_and_unavailable_historical_variant_block_selection(tmp_path):
    framework, proposal, handler = setup(tmp_path)
    legacy = replay_spec("legacy")
    legacy.exact_candidate = None
    framework.submit_hypothesis(legacy)
    assert handler(proposal.exact_candidate)["status"] == "DATA_BLOCKED"
    framework.hypotheses.pop("legacy")
    framework.registry.hypotheses.pop("legacy")
    result = handler(proposal.exact_candidate)
    assert result["status"] == "DATA_BLOCKED"
    assert "history unavailable" in result["receipt"]["reason"]


def test_superseded_frozen_strategy_versions_are_counted_in_family(tmp_path):
    framework, proposal, handler = setup(tmp_path)
    replacement = replay_spec()
    replacement.threshold = 4
    replacement.exact_candidate["exact_entry"]["value"] = 2
    framework.submit_hypothesis(replacement)
    framework.freeze_hypothesis("c1")
    result = handler(replacement.exact_candidate)
    assert len(result["receipt"]["expected_variants"]) == 2


def test_split_corpus_mutation_and_insufficient_samples_never_certify(tmp_path):
    _, proposal, handler = setup(tmp_path)
    handler.split["corpus_hash"] = "different"
    assert handler(proposal.exact_candidate)["status"] == "DATA_BLOCKED"
    _, proposal, handler = setup(tmp_path)
    handler.gauntlet.min_observations = 200
    assert handler(proposal.exact_candidate)["status"] == "DATA_BLOCKED"


def test_bounded_work_and_unreadable_corpus_are_not_silent_omissions(tmp_path):
    _, proposal, handler = setup(tmp_path)
    handler.max_episodes = 2
    assert handler(proposal.exact_candidate)["status"] == "DATA_BLOCKED"
    handler.max_episodes = 100
    handler.max_bootstrap_draws = 1
    assert handler(proposal.exact_candidate)["status"] == "DATA_BLOCKED"
    handler.max_bootstrap_draws = 100000
    path = next((tmp_path / "episodes").glob("*/*.json.gz"))
    path.write_bytes(b"not a gzip episode")
    assert handler(proposal.exact_candidate)["status"] == "DATA_BLOCKED"


def configured_engine(tmp_path, *, configured=True):
    from src.research.frontier import FrontierIndex
    from src.research.trade_evidence import TradeEvidenceLedger
    from src.research.unknown_engine import UnknownOpportunityEngine
    _, proposal, example = setup(tmp_path / "corpus")
    framework = ChampionChallengerFramework(state_path=str(tmp_path / "framework.json"))
    handler = CandidateReplayHandler(framework, example.episode_path,
        chronological_split=example.split, gauntlet=example.gauntlet)
    costs = {"inference": 0, "data": 0, "engineering": 0, "conversion": 0, "gauntlet": 2}
    bounds = {"max_episodes": 2000, "max_family_variants": 32, "max_bootstrap_draws": 20_000_000}
    engine = UnknownOpportunityEngine(framework, FrontierIndex(tmp_path / "frontier.sqlite"),
        TradeEvidenceLedger(tmp_path / "evidence.jsonl"), **({"gauntlet_handler": handler,
        "gauntlet_costs": costs, "gauntlet_work_bound": bounds,
        "research_pass_resource_units": 4} if configured else {}))
    return engine, proposal.exact_candidate, handler, costs, bounds


def test_existing_engine_costed_jobs_dispatch_into_actual_canonical_replay(tmp_path):
    from test_unknown_engine import raw
    engine, candidate, _, _, _ = configured_engine(tmp_path)
    engine.ingest(raw(exact_candidate=candidate))
    assert engine.run_pass()["results"][0]["disposition"] == "EXACT_CANDIDATE"
    assert engine.frontier.report()["cost_blocked"] == 0
    result = engine.run_pass()
    assert result["results"][0]["disposition"] == "DEEPENING_TASK", result
    assert not engine.framework.trial_results and not engine.framework.champions


def test_new_explicit_budget_wakes_persisted_unknown_cost_work_without_dropping_it(tmp_path):
    from src.research.unknown_engine import UnknownOpportunityEngine
    from test_unknown_engine import raw
    engine, candidate, handler, costs, bounds = configured_engine(tmp_path, configured=False)
    engine.ingest(raw(exact_candidate=candidate))
    engine.run_pass()
    assert engine.frontier.report()["cost_blocked"] == 1
    pointer = next(iter(engine.framework.get_research_component("gauntlet_work")["jobs"].values()))
    job = engine.frontier.job(pointer["job_id"])
    engine.frontier.observe_coordinate(job["axes"], state="HIBERNATED_LOW_VOI",
        enabling_change="canonical-replay-cost-and-work-envelope-configured",
        hibernation_reason="replay resource cost was not yet declared", evidence_refs=[job["payload_ref"]])
    configured = UnknownOpportunityEngine(engine.framework, engine.frontier, engine.evidence,
        gauntlet_handler=handler, gauntlet_costs=costs, gauntlet_work_bound=bounds,
        research_pass_resource_units=4)
    assert configured.run_pass()["results"][0]["disposition"] == "DEEPENING_TASK"
    assert configured.frontier.report()["cost_blocked"] == 0
    assert configured.frontier.coordinate(job["axes"])["state"] == "DISCOVERED"


def test_parameter_neighbours_share_pending_family_sweep_and_duplicates_keep_ancestry(tmp_path):
    from test_unknown_engine import raw
    engine, candidate, _, _, _ = configured_engine(tmp_path)
    duplicate = copy.deepcopy(candidate)
    duplicate["candidate_id"] = "duplicate"
    neighbour = copy.deepcopy(candidate)
    neighbour["candidate_id"] = "neighbour"
    neighbour["exact_entry"]["value"] = 4
    for program in (candidate, duplicate, neighbour):
        engine.ingest(raw(exact_candidate=program))
    results = engine.run_pass()["results"]
    assert sum(row["disposition"] == "DUPLICATE" for row in results) == 1
    queued = [row for row in engine.frontier.report()["by_kind"] if row["kind"] == "GAUNTLET"]
    assert sum(row["count"] for row in queued) == 1
    assert len(engine.framework.hypotheses) == 3
    assert engine.framework.get_stats()["effective_independent_breadth"] == 1

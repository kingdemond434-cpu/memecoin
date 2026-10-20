"""Canonical retained observations drive real consumers across restart/failure."""
import asyncio
import copy
import time

import pytest

from src.research.frontier import AuctionInputs, FrontierIndex
from src.research.mechanism_labs import MechanismLaboratory
from src.research.population_hunter import PopulationHunter
from src.research.research_validation import ResearchValidationService
from src.research.research_workflows import ResearchWorkflows
from src.research.trade_evidence import TradeEvidenceLedger
from src.research.unknown_engine import UnknownOpportunityEngine
from src.strategies.champion_challenger import ChampionChallengerFramework
from test_candidate_replay import setup


def create(root, framework=None, handler=None):
    framework = framework or ChampionChallengerFramework(state_path=str(root / "framework.json"))
    options = {}
    if handler:
        options = {"gauntlet_handler": handler,
            "gauntlet_costs": {"inference": 0, "data": 0, "engineering": 0, "conversion": 0, "gauntlet": 2},
            "gauntlet_work_bound": {key: getattr(handler, key) for key in
                ("max_episodes", "max_family_variants", "max_bootstrap_draws")},
            "research_pass_resource_units": 8}
    engine = UnknownOpportunityEngine(framework, FrontierIndex(root / "frontier.sqlite"),
        TradeEvidenceLedger(root / "evidence.jsonl"), **options)
    validation = ResearchValidationService(framework, handler).configure_evidence(engine.evidence)
    engine.configure_workflows(ResearchWorkflows(engine, population=PopulationHunter(framework),
        laboratory=MechanismLaboratory(framework), validation=validation))
    return engine


@pytest.fixture
def desk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return create(tmp_path)


def point(value, at, **changes):
    return {"variable": "liquidity_usd", "value": value, "unit": "USD", "event_at": at,
        "available_at": at, "observed_at": at, "source_id": "native-fixture", "chain": "solana",
        "entity_id": "fixture-token", "partition": "training", "rights": "PUBLIC",
        "point_in_time": True, **changes}


def test_blocked_data_request_wakes_on_actual_new_series_and_survives_restart(desk, tmp_path):
    now = time.time()
    job = desk.enqueue_work("DATA_REQUEST", {"variable": "liquidity_usd", "min_observations": 2}, now=now)
    desk.run_pass(now=now)
    assert desk.frontier.job(job)["state"] == "HIBERNATED"
    restarted = create(tmp_path)
    restarted.enqueue_work("OBSERVABLE_SERIES", {"points": [point(10, now-2), point(20, now-1)]}, now=now)
    for _ in range(6):
        restarted.run_pass(now=now)
    assert restarted.frontier.job(job)["state"] == "ACKED"
    assert restarted.workflows.data.report()["variables"]["liquidity_usd"] == 2
    assert TradeEvidenceLedger.verify(restarted.evidence.path)[0]


def test_causal_window_carries_late_input_availability_and_partition_isolation(desk):
    now = time.time()
    desk.enqueue_work("OBSERVABLE_SERIES", {"points": [point(10, now-30, available_at=now-5, observed_at=now-5),
        point(20, now-20), point(40, now-10)]}, now=now)
    desk.run_pass(now=now)
    request = {"variable": "liquidity_usd", "transform": "change", "window": 3,
        "mechanism": "native depth change", "falsifier": "no chronological after-cost benefit"}
    result = desk.workflows.data.consume("REPRESENTATION_REQUEST", request, ["fixture-input"], now)
    output = result["children"][0]["payload"]["points"]
    assert all(row["available_at"] == now-5 and row["observed_at"] == now-5 for row in output)
    request["as_of"] = now-6
    result = desk.workflows.data.consume("REPRESENTATION_REQUEST", request, ["fixture-new-request"], now)
    assert [row["value"] for row in result["children"][0]["payload"]["points"]] == [20]
    desk.enqueue_work("OBSERVABLE_SERIES", {"points": [point(999, now-1, partition="test")]}, now=now)
    desk.run_pass(now=now)
    result = desk.workflows.data.consume("REPRESENTATION_REQUEST", {**request, "as_of": now}, ["fixture-third"], now)
    assert result["disposition"] == "INVALID" and "partition" in result["detail"]["reason"]


def test_registered_acquisition_and_paged_batch_reaches_canonical_series(desk):
    now = time.time()
    async def source():
        return [point(index, now-1000+index) for index in range(600)]
    desk.workflows.data.register_source("fixture-source", source, source_classes=["native_depth"])
    job = desk.enqueue_work("SOURCE_REQUEST", {"variable": "liquidity_usd", "source_classes": ["native_depth"]}, now=now)
    for _ in range(12):
        asyncio.run(desk.run_async_pass(now=now))
    assert desk.frontier.job(job)["state"] == "ACKED"
    assert desk.workflows.data.report()["variables"]["liquidity_usd"] == 600
    assert desk.workflows.data.report()["acquisitions"] == 1
    assert TradeEvidenceLedger.verify(desk.evidence.path)[0]


def test_no_consumer_never_acknowledges_unknown_work(desk):
    now = time.time()
    ref = desk._record("unknown_future_contract", {"value": 1}, now)
    job = desk.frontier.enqueue("FUTURE_UNIMPLEMENTED_KIND", ref, auction=desk._triage_budget(), now=now)
    desk.run_pass(now=now)
    assert desk.frontier.job(job)["state"] == "READY"


def test_receipt_write_failure_retains_parent_and_idempotent_children(desk, monkeypatch):
    now = time.time()
    job = desk.enqueue_work("OBSERVABLE_SERIES", {"points": [point(20, now-1)]}, now=now)
    original = desk.evidence.record_reference
    def fail_receipt(event, *args, **kwargs):
        return None if event == "unknown_conversion_receipt" else original(event, *args, **kwargs)
    monkeypatch.setattr(desk.evidence, "record_reference", fail_receipt)
    result = desk.run_pass(now=now)
    assert desk.frontier.job(job)["state"] == "RETRY"
    assert result["results"][0]["disposition"] == "BLOCKED"
    monkeypatch.setattr(desk.evidence, "record_reference", original)
    desk.run_pass(now=now+61)
    assert desk.frontier.job(job)["state"] == "ACKED"
    assert desk.workflows.data.report()["variables"]["liquidity_usd"] == 1


def test_already_frozen_training_candidate_reaches_real_historical_then_forward_queue(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    framework, spec, handler = setup(tmp_path)
    framework.state_path = tmp_path / "framework.json"
    desk = create(tmp_path, framework, handler)
    now = spec.frozen_at+10
    job = desk.enqueue_work("EXACT_CANDIDATE", {"candidate": copy.deepcopy(spec.exact_candidate)}, now=now)
    for _ in range(3):
        desk.run_pass(now=now)
    assert desk.frontier.job(job)["state"] == "ACKED"
    report = desk.workflows.validation.report()
    assert report["historical_receipts"] == 1 and report["forward_experiments"] == 1, (desk.last_pass, desk.frontier.report()["by_kind"])
    assert report["paper_outcomes"] == 0 and not framework.is_live(spec.hypothesis_id)
    assert any(row.stage == "CHRONOLOGICAL_OOS" and row.evidence_status == "BOUND" for row in framework.trial_results)
    assert any(row["kind"] == "SURVIVOR_DESCENDANT" for row in desk.frontier.report()["by_kind"])


def test_production_family_rotation_uses_real_frontier_axes_and_persists_all_families(desk):
    from src.research.mechanism_labs import STRATEGY_FAMILIES
    desk.workflows.seed_labs = True
    now = time.time()
    for _ in range(len(STRATEGY_FAMILIES)+1):
        desk.workflows.prepare(now)
    intake = desk.framework.get_research_component("research_family_intake")
    assert set(intake["seeded"]) == set(STRATEGY_FAMILIES)
    assert all(desk.frontier.job(job)["total_cost"] == 1 for job in intake["seeded"].values())
    report = desk.run_pass(now=now)
    assert report["processed"] > 0
    assert all(row["disposition"] != "BLOCKED" for row in report["results"])


def test_retained_unpriced_work_gets_consumer_envelope_without_reset(desk):
    now = time.time()
    ref = desk._record("legacy-request", {"variable": "unobserved"}, now)
    job = desk.frontier.enqueue("DATA_REQUEST", ref, now=now)
    assert desk.frontier.job(job)["total_cost"] is None
    desk.run_pass(now=now)
    state = desk.frontier.job(job)
    assert state["state"] == "HIBERNATED" and state["attempts"] == 1 and state["total_cost"] == 1


def test_observable_bound_refuses_tail_loss_and_corrupted_ref(desk):
    now = time.time()
    data = desk.workflows.data
    result = data.consume("OBSERVABLE_SERIES", {"points": [point(i, now-1) for i in range(data.max_points+1)]}, ["fixture-ref"], now)
    assert result["disposition"] == "INVALID" and not data.state["series"]
    desk.enqueue_work("OBSERVABLE_SERIES", {"points": [point(20, now-1)]}, now=now)
    desk.run_pass(now=now)
    ref = next(iter(data.state["series"]["liquidity_usd"].values()))["ref"]
    with desk.evidence.path.open("r+b") as handle:
        handle.seek(int(ref.split(":")[-1])+10)
        handle.write(b"CORRUPT")
    with pytest.raises((ValueError, KeyError)):
        data._read("liquidity_usd", {}, now)


def test_observation_batch_is_atomic_and_as_of_respects_actual_observation_clock(desk):
    now = time.time()
    data = desk.workflows.data
    result = data.consume("OBSERVABLE_SERIES", {"points": [point(10, now-10), point(20, now-10)]}, ["fixture-conflict"], now)
    assert result["disposition"] == "INVALID" and not data.state["series"]
    job = desk.enqueue_work("OBSERVABLE_SERIES", {"points": [point(20, now-10, observed_at=now-1)]}, now=now)
    desk.run_pass(now=now)
    assert desk.frontier.job(job)["state"] == "ACKED"
    assert not data._read("liquidity_usd", {"as_of": now-2}, now)
    assert len(data._read("liquidity_usd", {}, now)) == 1
    # Repeated acquisition retains one independent source observation.
    desk.enqueue_work("OBSERVABLE_SERIES", {"points": [point(20, now-10, observed_at=now-1)], "delivery": "retry"}, now=now)
    desk.run_pass(now=now)
    assert len(data._read("liquidity_usd", {}, now)) == 1


def test_acquisition_pages_all_registered_sources(desk):
    for index in range(6):
        desk.workflows.data.register_source("source-"+str(index), lambda: [], source_classes=["fixture-scope"])
    now = time.time()
    request = {"variable": "depth", "source_classes": ["fixture-scope"]}
    first = desk.workflows.data.consume("SOURCE_REQUEST", request, ["fixture-first"], now)
    continuation = next(child for child in first["children"] if child["kind"] == "SOURCE_REQUEST")
    second = desk.workflows.data.consume("SOURCE_REQUEST", continuation["payload"], ["fixture-next"], now)
    children = first["children"]+second["children"]
    assert {child["payload"]["source_id"] for child in children if child["kind"] == "SOURCE_ACQUISITION"} == {"source-"+str(i) for i in range(6)}
    assert not any(child["kind"] == "SOURCE_REQUEST" for child in second["children"])


def test_registered_capability_wakes_retained_source_request(desk):
    now = time.time()
    job = desk.enqueue_work("SOURCE_REQUEST", {"variable": "depth", "source_classes": ["native-fixture"]}, now=now)
    desk.run_pass(now=now)
    assert desk.frontier.job(job)["state"] == "HIBERNATED"
    desk.workflows.data.register_source("new-source", lambda: [], source_classes=["native-fixture"])
    desk.run_pass(now=now)
    assert desk.frontier.job(job)["state"] == "ACKED"


def test_miner_observation_timestamp_is_completion_not_request_start(monkeypatch):
    from src.research.data_miners import DataMinerPool, Enriches, MinerSpec
    clock = [100.]
    monkeypatch.setattr("src.research.data_miners.time.time", lambda: clock[0])
    pool = DataMinerPool()
    async def fetch():
        clock[0] = 110.
        return [{"price_usd": 2.}]
    pool.register(MinerSpec("fixture", Enriches.MARKET_CONTEXT, 60), fetch)
    result = asyncio.run(pool._run_one("fixture", 100.))
    assert result.fetched_at == 110 and result.records[0]["_fetched_at"] == 110
    assert result.records[0]["_request_started_at"] == 100


def test_gauntlet_ack_retry_reuses_canonical_receipt_and_descendant_identity(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    framework, spec, handler = setup(tmp_path)
    framework.state_path = tmp_path / "framework.json"
    desk = create(tmp_path, framework, handler)
    now = spec.frozen_at+10
    desk.enqueue_work("EXACT_CANDIDATE", {"candidate": copy.deepcopy(spec.exact_candidate)}, now=now)
    desk.run_pass(now=now)
    job = desk.frontier.pending_work(kinds=("GAUNTLET",))[0]
    acknowledge = desk.frontier.ack
    def interrupted(identity, *args, **kwargs):
        if identity == job["job_id"]:
            raise RuntimeError("injected interruption after durable children")
        return acknowledge(identity, *args, **kwargs)
    monkeypatch.setattr(desk.frontier, "ack", interrupted)
    desk.run_pass(now=now)
    assert desk.frontier.job(job["job_id"])["state"] == "RETRY"
    first_children = desk.frontier.pending_work(kinds=("SURVIVOR_DESCENDANT",))
    assert len(first_children) == 1
    monkeypatch.setattr(desk.frontier, "ack", acknowledge)
    leased = desk.frontier.lease_jobs(desk.worker, kinds=("GAUNTLET",), resource_budget=8, now=now+61)[0]
    payload = desk._load_refs({leased["payload_ref"]})[leased["payload_ref"]]
    desk._commit_work(leased, desk._dispatch_work(leased, payload, now+61), now+61)
    second_children = desk.frontier.pending_work(kinds=("SURVIVOR_DESCENDANT",))
    assert [child["job_id"] for child in second_children] == [child["job_id"] for child in first_children]
    assert desk.frontier.job(job["job_id"])["state"] == "ACKED"

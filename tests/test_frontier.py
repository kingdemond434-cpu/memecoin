"""The research index preserves unknowns, work and authority across restarts."""

from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest

from src.research.frontier import (
    AuctionInputs, CHAIN_METRICS, EXPLORATION_TAGS, FRONTIER_AXES,
    FrontierIndex, coordinate_id, stable_job_id,
)


def priced(value=1.0, cost=1.0, **kwargs):
    return AuctionInputs(
        probability_unique=0.5, expected_delta_logw=value, novelty=1,
        coverage_debt=2, urgency=1,
        costs={"inference": cost, "data": 0, "engineering": 0,
               "conversion": 0, "gauntlet": 0}, **kwargs)


def test_reference_index_uses_full_durability_wal_and_writer_can_commit_during_reader(tmp_path):
    path = tmp_path / "concurrent.sqlite"
    index = FrontierIndex(path)
    with index._db() as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
    reader = sqlite3.connect(path)
    reader.execute("BEGIN")
    reader.execute("SELECT * FROM jobs").fetchall()
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(index.enqueue, "RAW_EVIDENCE", "fixture:canonical-record", auction=priced())
        job = future.result(timeout=3)
        assert index.job(job)["state"] == "READY"
    finally:
        reader.rollback()
        reader.close()
        pool.shutdown(wait=True)
    assert path.with_name(path.name+"-wal").exists()
    index.close()
    assert not path.with_name(path.name+"-wal").exists()
    assert FrontierIndex(path).job(job)["payload_ref"] == "fixture:canonical-record"


def test_sparse_identity_covers_every_axis_and_unseen_is_not_inserted(tmp_path):
    index = FrontierIndex(tmp_path / "index.sqlite")
    a = {"chain": "new-chain", "protocol": "new-protocol"}
    assert coordinate_id(a) == coordinate_id(dict(reversed(list(a.items()))))
    unseen = index.coordinate(a)
    assert unseen["state"] == "UNSEEN"
    assert unseen["coverage_debt"] is None
    assert set(unseen["axes"]) == set(FRONTIER_AXES)
    assert len(FRONTIER_AXES) == 21
    assert index.report()["coordinates"] == 0
    for axis in FRONTIER_AXES:
        assert coordinate_id({axis: "a"}) != coordinate_id({axis: "b"})
    with pytest.raises(ValueError):
        coordinate_id({"typo_chain": "x"})


def test_frontier_debt_and_named_hibernation_survive_restart(tmp_path):
    path = tmp_path / "index.sqlite"
    index = FrontierIndex(path)
    axes = {"chain": "unlisted", "representation": "source-disagreement"}
    index.observe_coordinate(axes, state="HIBERNATED_LOW_VOI", evidence_refs=["lake:1"],
                             coverage_debt=4, expected_value=0.01,
                             enabling_change="licensed-history-arrives", hibernation_reason="no PIT history", now=100)
    restarted = FrontierIndex(path)
    assert restarted.coordinate(axes)["coverage_debt"] == 4
    with pytest.raises(ValueError, match="named enabling"):
        restarted.observe_coordinate(axes, evidence_refs=["lake:2"])
    with pytest.raises(ValueError, match="does not match"):
        restarted.wake_coordinate(axes, enabling_change="lower gates", evidence_refs=["lake:2"])
    restarted.wake_coordinate(axes, enabling_change="licensed-history-arrives", evidence_refs=["lake:2"], now=110)
    row = restarted.coordinate(axes)
    assert row["state"] == "DISCOVERED"
    assert row["coverage_debt"] == 4
    assert row["evidence_refs"] == ["lake:1", "lake:2"]


def test_dynamic_chain_is_discovered_without_static_chain_list(tmp_path):
    path = tmp_path / "index.sqlite"
    index = FrontierIndex(path)
    index.observe_discovery("chain", "never-listed-chain-abc", family="EVM",
                            evidence_refs=["lake:chain-observation"],
                            metrics={"launches_per_day": 250}, now=100)
    row = FrontierIndex(path).chain("never-listed-chain-abc")
    assert row["stage"] == "DISCOVERED"
    assert row["metrics"]["launches_per_day"] == 250
    assert set(row["metrics"]) == set(CHAIN_METRICS)
    assert row["metrics"]["executable_liquidity"] is None
    assert index.coordinate({"chain": "never-listed-chain-abc", "chain_family": "EVM"})["state"] == "DISCOVERED"
    with pytest.raises(ValueError, match="exactly one"):
        index.advance_chain("never-listed-chain-abc", "SIMULATED_EXECUTION", evidence_refs=["lake:2"])
    index.advance_chain("never-listed-chain-abc", "OBSERVATION_ONLY", evidence_refs=["lake:2"])
    assert index.chain("never-listed-chain-abc")["stage"] == "OBSERVATION_ONLY"
    index.register_chain("never-listed-chain-abc", "EVM", metrics={"wallet_activity": 42}, evidence_refs=["lake:3"])
    assert index.chain("never-listed-chain-abc")["stage"] == "OBSERVATION_ONLY"
    assert index.chain("never-listed-chain-abc")["metrics"]["launches_per_day"] == 250


def test_protocol_contract_unknown_layout_stays_unknown_and_versioned(tmp_path):
    index = FrontierIndex(tmp_path / "index.sqlite")
    index.observe_discovery("protocol", "unknown-factory", axes={"chain": "new-chain"},
                            evidence_refs=["lake:unknown-log"], contract={"unknown_event_rate": 1.0})
    row = index.protocol("unknown-factory", "new-chain", "1")
    assert row["contract"]["abi_ref"] is None
    assert row["contract"]["decoder_parity"] is None
    assert row["contract"]["execution_support"] is None
    with pytest.raises(ValueError, match="requires source"):
        index.register_protocol("unknown-factory", "new-chain", "1", contract={"status": "VERIFIED"}, evidence_refs=["lake:syntactic-decode"])
    assert index.protocol("unknown-factory", "new-chain", "1")["contract"]["status"] == "DISCOVERED"
    verified = {"verified_source": "official-repo:revision", "abi_ref": "lake:abi",
                "addresses": ["chain-qualified:address"], "verification_provenance": "lake:verification",
                "transaction_fixture_refs": ["fixture:real-native-tx"], "decoder_parity": True,
                "last_verified": 120, "status": "VERIFIED", "execution_support": "SHADOW_ONLY"}
    index.register_protocol("unknown-factory", "new-chain", "2", contract=verified, evidence_refs=["lake:fixture-parity"])
    assert index.protocol("unknown-factory", "new-chain", "1")["contract"]["abi_ref"] is None
    assert index.protocol("unknown-factory", "new-chain", "2")["contract"]["decoder_parity"] is True


@pytest.mark.parametrize("state", ["CANARY", "LIVE"])
def test_research_cannot_grant_live_authority(tmp_path, state):
    index = FrontierIndex(tmp_path / "index.sqlite")
    with pytest.raises(PermissionError):
        index.observe_coordinate({"chain": "x"}, state=state, evidence_refs=["lake:1"])
    index.register_chain("x", "EVM", evidence_refs=["lake:1"])
    with pytest.raises(PermissionError):
        index.advance_chain("x", state, evidence_refs=["lake:1"])
    with pytest.raises(PermissionError):
        index.register_protocol("p", "x", "1", contract={"status": state}, evidence_refs=["lake:1"])
    with pytest.raises(PermissionError):
        index.register_protocol("p", "x", "1", contract={"execution_support": True}, evidence_refs=["lake:1"])


def test_lease_restart_expiry_stale_worker_and_ack_are_atomic(tmp_path):
    path = tmp_path / "index.sqlite"
    index = FrontierIndex(path)
    job = index.enqueue("DATA_REQUEST", "lake:request-1", auction=priced(), now=100)
    assert job == stable_job_id("DATA_REQUEST", "lake:request-1")
    original = index.lease_jobs("worker", limit=1, lease_seconds=10, now=101)[0]
    assert set(original["axes"]) == set(FRONTIER_AXES)
    assert index.job(job)["axes"] == original["axes"]
    restarted = FrontierIndex(path)
    assert restarted.lease_jobs("other", now=105) == []
    redelivered = restarted.lease_jobs("worker", limit=1, now=112)[0]
    assert redelivered["attempts"] == 2
    assert redelivered["lease_token"] != original["lease_token"]
    with pytest.raises(PermissionError):
        restarted.ack(job, "worker", original["lease_token"], disposition="DATA_REQUEST", evidence_refs=["lake:old"], now=113)
    restarted.ack(job, "worker", redelivered["lease_token"], disposition="MECHANISM_HYPOTHESIS", evidence_refs=["lake:converted"], now=113)
    assert restarted.job(job)["state"] == "ACKED"
    assert restarted.job(job)["evidence_refs"] == ["lake:converted", "lake:request-1"]
    assert restarted.enqueue("DATA_REQUEST", "lake:request-1", auction=priced(value=999), now=114) == job
    assert restarted.job(job)["state"] == "ACKED"
    assert restarted.lease_jobs("third", now=1000) == []


def test_concurrent_workers_cannot_claim_the_same_work(tmp_path):
    path = tmp_path / "index.sqlite"
    index = FrontierIndex(path)
    for number in range(4):
        index.enqueue("research", f"lake:{number}", auction=priced(), now=100)
    with ThreadPoolExecutor(max_workers=2) as workers:
        batches = list(workers.map(lambda name: FrontierIndex(path).lease_jobs(name, limit=4, now=101), ["a", "b"]))
    all_ids = [row["job_id"] for batch in batches for row in batch]
    assert len(all_ids) == len(set(all_ids)) == 4


def test_retry_survives_restart_and_does_not_run_before_due(tmp_path):
    path = tmp_path / "index.sqlite"
    index = FrontierIndex(path)
    job = index.enqueue("source", "lake:outage", auction=priced(), now=100)
    leased = index.lease_jobs("worker", now=101)[0]
    index.retry(job, "worker", leased["lease_token"], reason="provider outage", delay_seconds=20, now=102)
    restarted = FrontierIndex(path)
    assert restarted.lease_jobs("worker", now=121) == []
    assert restarted.lease_jobs("worker", now=122)[0]["job_id"] == job
    assert restarted.job(job)["last_error"] == "provider outage"


def test_costed_auction_prefers_information_value_per_resource_and_fits_budget(tmp_path):
    index = FrontierIndex(tmp_path / "index.sqlite", exploration_floors={})
    cheap = index.enqueue("lab", "lake:cheap-decisive", auction=priced(value=2, cost=1), now=100)
    expensive = index.enqueue("lab", "lake:expensive", auction=priced(value=10, cost=100), now=100)
    batch = index.lease_jobs("worker", limit=8, resource_budget=1, now=101)
    assert [row["job_id"] for row in batch] == [cheap]
    assert index.job(expensive)["state"] == "READY"
    assert index.report(now=101)["pending"] == 2


def test_unknown_costs_are_visible_and_never_free(tmp_path):
    index = FrontierIndex(tmp_path / "index.sqlite")
    job = index.enqueue("new-chain", "lake:unknown-cost", exploration_tags=["unknown_chains"], now=100)
    assert index.lease_jobs("worker", now=101) == []
    assert index.job(job)["priority"] is None
    report = index.report(now=101)
    assert report["cost_blocked"] == 1
    assert report["cost_of_delay"] is None
    assert report["cost_of_delay_unmeasured_jobs"] == 1
    assert report["coverage_debt"] is None


def test_exploration_floors_rotate_across_restarts_without_erasing_low_voi(tmp_path):
    path = tmp_path / "index.sqlite"
    index = FrontierIndex(path)
    expected = set()
    for tag in EXPLORATION_TAGS:
        expected.add(index.enqueue("exploration", f"lake:{tag}", auction=priced(value=0), exploration_tags=[tag], now=100))
    # Every pass also has much higher value exploitation work available.
    for number in range(20):
        index.enqueue("exploit", f"lake:exploit-{number}", auction=priced(value=1000), now=99)
    selected = set()
    for number in range(len(EXPLORATION_TAGS)):
        restarted = FrontierIndex(path)
        lease = restarted.lease_jobs("worker", limit=1, now=101 + number)[0]
        selected.add(lease["job_id"])
        restarted.ack(lease["job_id"], "worker", lease["lease_token"], disposition="NEGATIVE_KNOWLEDGE", evidence_refs=["lake:experiment"], now=101.5 + number)
    assert selected == expected
    assert FrontierIndex(path).report(now=120)["queue_states"]["READY"] == 20


def test_hibernation_keeps_work_until_named_enabling_change(tmp_path):
    index = FrontierIndex(tmp_path / "index.sqlite")
    job = index.enqueue("representation", "lake:insufficient-history", auction=priced(), now=100)
    index.hibernate_job(job, enabling_change="new-history", reason="low expected information", now=101)
    assert index.lease_jobs("worker", now=1000) == []
    with pytest.raises(ValueError):
        index.wake_job(job, enabling_change="more compute", evidence_refs=["lake:2"])
    index.wake_job(job, enabling_change="new-history", evidence_refs=["lake:2"], now=1001)
    assert index.lease_jobs("worker", now=1002)[0]["job_id"] == job


def test_drain_oldest_and_cost_delay_metrics_retain_unknown_coverage(tmp_path):
    index = FrontierIndex(tmp_path / "index.sqlite")
    a = index.enqueue("conversion", "lake:a", auction=priced(delay_value_per_second=0.001), now=100)
    index.enqueue("conversion", "lake:b", now=105)
    lease = index.lease_jobs("worker", now=110)[0]
    assert lease["job_id"] == a
    index.ack(a, "worker", lease["lease_token"], disposition="EXACT_CANDIDATE", evidence_refs=["lake:exact"], now=111)
    report = index.report(now=120, window_seconds=30)
    assert report["arrivals"] == 2 and report["drained"] == 1
    assert report["growth_per_second"] == pytest.approx(1 / 30)
    assert report["drain_per_second"] == pytest.approx(1 / 30)
    assert report["oldest_pending_age_seconds"] == 15
    assert report["cost_of_delay"] is None
    assert report["cost_of_delay_unmeasured_jobs"] == 1
    assert report["status"] == "INDEX_ONLY_RUNTIME_UNMEASURED"


def test_failed_mutation_does_not_replace_existing_contract(tmp_path):
    index = FrontierIndex(tmp_path / "index.sqlite")
    index.register_chain("x", "EVM", evidence_refs=["lake:1"])
    with pytest.raises(ValueError):
        index.register_chain("x", "MOVE", evidence_refs=["lake:2"])
    assert index.chain("x")["family"] == "EVM"
    with pytest.raises(ValueError):
        index.enqueue("lab", "lake:nan", auction=AuctionInputs(expected_delta_logw=float("nan")))
    assert index.report()["pending"] == 0


def test_reobserving_registered_work_does_not_rewind_frontier_or_protocol_version(tmp_path):
    index = FrontierIndex(tmp_path / "index.sqlite")
    axes = {"chain": "x", "chain_family": "EVM"}
    index.observe_coordinate(axes, state="TESTED", evidence_refs=["lake:experiment"], coverage_debt=3)
    index.register_chain("x", "EVM", evidence_refs=["lake:new-sighting"])
    assert index.coordinate(axes)["state"] == "TESTED"
    assert index.coordinate(axes)["coverage_debt"] == 3
    index.register_protocol("factory", "x", "1", contract={"abi_ref": "lake:abi-v1"}, evidence_refs=["lake:1"])
    with pytest.raises(ValueError, match="new protocol version"):
        index.register_protocol("factory", "x", "1", contract={"abi_ref": "lake:changed-layout"}, evidence_refs=["lake:2"])
    assert index.protocol("factory", "x", "1")["contract"]["abi_ref"] == "lake:abi-v1"


def test_measured_delay_value_is_reported_without_filling_other_values(tmp_path):
    index = FrontierIndex(tmp_path / "index.sqlite")
    index.enqueue("lab", "lake:measured", auction=priced(delay_value_per_second=0.001), now=100)
    index.enqueue("lab", "lake:unmeasured", now=100)
    report = index.report(now=200)
    assert report["cost_of_delay"] == pytest.approx(0.1)
    assert report["cost_of_delay_measured_jobs"] == 1
    assert report["cost_of_delay_unmeasured_jobs"] == 1
    assert report["oldest_high_value_age_seconds"] == 100


def test_stable_id_collision_rolls_back_without_orphan_coordinate(tmp_path):
    index = FrontierIndex(tmp_path / "index.sqlite")
    index.enqueue("lab", "lake:one", job_id="fixed", axes={"chain": "x"})
    with pytest.raises(ValueError, match="collides"):
        index.enqueue("lab", "lake:two", job_id="fixed", axes={"chain": "y"})
    assert index.coordinate({"chain": "y"})["state"] == "UNSEEN"
    assert index.report()["coordinates"] == 1


def test_new_sightings_cannot_bypass_a_coordinates_low_voi_hibernation(tmp_path):
    index = FrontierIndex(tmp_path / "index.sqlite")
    axes = {"chain": "x", "protocol": "p"}
    old_job = index.enqueue("lab", "lake:old", axes=axes, auction=priced(), now=100)
    index.observe_coordinate(axes, state="HIBERNATED_LOW_VOI", evidence_refs=["lake:failure"],
                             enabling_change="new-independent-data", hibernation_reason="insufficient information", now=101)
    new_job = index.enqueue("lab", "lake:new-sighting", axes=axes, auction=priced(), now=102)
    assert index.job(new_job)["state"] == "HIBERNATED"
    assert index.job(old_job)["state"] == "READY"
    assert index.lease_jobs("worker", now=103) == []
    with pytest.raises(ValueError, match="wake the coordinate"):
        index.wake_job(new_job, enabling_change="new-independent-data", evidence_refs=["lake:new-data"], now=104)
    index.wake_coordinate(axes, enabling_change="new-independent-data", evidence_refs=["lake:new-data"], now=104)
    index.wake_job(new_job, enabling_change="new-independent-data", evidence_refs=["lake:new-data"], now=104)
    assert len(index.lease_jobs("worker", now=105)) == 2

"""Offline canonical producer -> validation consumer experiments.

Native execution rows are explicit synthetic controls, never production claims.
"""
import asyncio
import copy
from collections import deque
from types import SimpleNamespace

import pytest

from src.research.candidate_replay import _digest
from src.research.forward_evidence import ForwardEvidence
from src.research.research_validation import CHECKPOINT, ResearchValidationService
from src.research.trade_evidence import TradeEvidenceLedger
from src.strategies.champion_challenger import ChampionChallengerFramework
from test_candidate_replay import setup


@pytest.fixture
def desk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    framework, spec, handler = setup(tmp_path)
    monkeypatch.setattr("src.strategies.champion_challenger.time",
                        SimpleNamespace(time=lambda: spec.frozen_at + 100000, sleep=lambda seconds: None))
    framework.state_path = tmp_path / "framework.json"
    ledger = TradeEvidenceLedger(tmp_path / "evidence.jsonl")
    forward = ForwardEvidence(tmp_path / "forward.json")
    service = ResearchValidationService(framework, handler, forward).configure_evidence(ledger)
    return framework, spec, handler, service, ledger, forward


def register(desk):
    framework, spec, _, service, _, _ = desk
    now = spec.frozen_at + 10
    judged = service.consume("GAUNTLET", {"candidate_id": "c1"}, ["fixture-corpus"], now)
    assert judged["disposition"] == "DEEPENING_TASK", judged
    child = judged["children"][0]
    registered = service.consume(child["kind"], child["payload"], ["fixture-corpus"], now + 1)
    assert registered["detail"]["status"] == "REGISTERED_FORWARD", registered
    return child["payload"]["experiment_id"], now + 2


def cohort(desk, eid, now, cid="cohort-1"):
    _, spec, _, service, _, _ = desk
    payload = {"action": "cohort", "candidate_id": "c1", "spec_hash": spec.frozen_hash,
               "experiment_id": eid, "cohort_id": cid, "event_at": now,
               "available_at": now + .1, "decision_at": now + .2, "observed_at": now + .3}
    result = service.consume("FORWARD_SHADOW", payload, ["native-event:fixture:" + cid], now + .4)
    assert result["disposition"] == "DEEPENING_TASK", result
    return payload


def native_outcome(desk, eid, now, *, paper=True, cid="cohort-1", event_id="outcome-1", cost=.02):
    _, spec, _, _, ledger, _ = desk
    kind = "native_simulation" if paper else "observed_transaction"
    refs = []
    for phase, at, size in (("entry", now + 1, 1.), ("exit", now + 2, 1.2)):
        row = {"phase": phase, "origin": "canonical_native_execution", "candidate_id": "c1",
               "spec_hash": spec.frozen_hash, "cohort_id": cid, "evidence_kind": kind,
               "observed_at": at, "available_at": at + .01, "route": "pump_curve",
               "size_native": size, "costs_native": cost / 2, "filled": True}
        if paper:
            # Typed native producer fixtures for contract rejection tests. The
            # separate native_forward tests execute actual quoting and unsigned
            # simulation acquisition end to end.
            fee_amount = round((cost / 2 - .000005) * 1_000_000_000)
            input_amount = round(size * 1_000_000_000) + fee_amount if phase == "entry" else 1000000
            output_amount = 1200000 if phase == "entry" else round(size * 1_000_000_000) - fee_amount
            quote_ref = ledger.record_reference("research_forward_native_quote", {
                "phase": phase, "route": "pump_curve", "observed_at": at,
                "quote": {"data_status": "OK", "input_amount": input_amount,
                          "output_amount": output_amount, "fee_amount": fee_amount}}, timestamp=at)
            simulation_ref = ledger.record_reference("research_native_simulation_cost", {
                "origin": "canonical_native_simulation", "phase": phase, "observed_at": at,
                "route": "pump_curve", "input_amount": input_amount,
                "network_fee_native": .000005, "failed_cost_native": 0.,
                "simulation": {"err": None, "unitsConsumed": 50000}}, timestamp=at)
            row.update(quote_ref=quote_ref, simulation_ref=simulation_ref)
        else:
            row["signature"] = "fixture-transaction-signature:" + phase
        refs.append(ledger.record_reference("research_forward_native_execution", row, timestamp=at))
    execution = {"evidence_kind": kind, "native": True, "filled": True, "route": "pump_curve",
                 "entry_evidence_ref": refs[0], "exit_evidence_ref": refs[1],
                 "entry_cost_native": 1., "exit_proceeds_native": 1.2, "fees_native": cost,
                 "failed_cost_native": 0., "native_to_usd": 100.}
    return {"action": "outcome", "candidate_id": "c1", "spec_hash": spec.frozen_hash,
            "experiment_id": eid, "cohort_id": cid, "event_id": event_id, "partition": "forward",
            "scope": "observed_forward_shadow" if paper else "observed_forward", "entry_at": now + 1,
            "exit_at": now + 2, "observed_at": now + 2.1, "execution": execution,
            "all_costs_included": True, "realized_pnl_usd": (1.2 - 1. - cost) * 100.,
            "equity_at_decision_usd": 1000., "record_in_forward_evidence": True}, refs


def test_actual_chronological_replay_records_bound_trial_and_only_registers_next_work(desk):
    framework, spec, _, service, _, _ = desk
    result = service.consume("GAUNTLET", {"candidate_id": "c1"}, ["fixture-corpus"], spec.frozen_at + 1)
    trial = framework.trial_results[-1]
    assert trial.passed and trial.stage == "CHRONOLOGICAL_OOS" and trial.evidence_status == "BOUND"
    assert trial.spec_hash == spec.frozen_hash and trial.samples == 18
    assert result["detail"]["receipt"]["scope"] == "L0-L4_ONLY"
    assert len(result["detail"]["receipt"]["stages"]) == 5
    assert result["children"][0]["kind"] == "FORWARD_SHADOW"
    assert not framework.shadow_models and not framework.canary_models and not framework.champions
    assert "authority" not in str(framework.get_research_component(CHECKPOINT))


def test_untrusted_claim_and_missing_corpus_are_blocked_with_named_data_request(desk):
    _, spec, _, service, _, _ = desk
    service.gauntlet_handler = lambda candidate: {"status": "HISTORICAL_SURVIVOR"}
    result = service.consume("GAUNTLET", {"candidate_id": "c1", "receipt": {"status": "SURVIVOR"}},
                             [], spec.frozen_at + 1)
    assert result["disposition"] == "BLOCKED"
    assert result["children"][0]["kind"] == "DATA_REQUEST"
    assert result["detail"]["enabling_change"].startswith("research_validation:")
    registered = service.consume("FORWARD_SHADOW", {"action": "register", "candidate_id": "c1",
        "experiment_id": "forged", "historical_receipt_hash": "caller-says-pass"}, [], spec.frozen_at + 1)
    assert registered["disposition"] == "BLOCKED"


def test_rejected_canonical_trial_remains_immutable_and_does_not_register(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    def make_losing(rows):
        for row in rows:
            for mark in row["market_observations"]:
                mark["price_multiple"] = .9 ** (mark["timestamp"] - row["created_at"])
    framework, spec, handler = setup(tmp_path, mutate=make_losing)
    service = ResearchValidationService(framework, handler)
    result = service.consume("GAUNTLET", {"candidate_id": "c1"}, ["losing-corpus"], spec.frozen_at + 1)
    assert result["disposition"] == "NEGATIVE_KNOWLEDGE" and not result["children"]
    assert not framework.trial_results[-1].passed
    assert framework.trial_results[-1].metrics["elogw"] < 0
    assert service.consume("GAUNTLET", {"candidate_id": "c1"}, ["losing-corpus"], spec.frozen_at + 9) == result
    assert len(framework.trial_results) == 1


def test_operation_and_forward_registration_survive_framework_restart(desk):
    framework, _, handler, service, ledger, forward = desk
    eid, now = register(desk)
    cohort(desk, eid, now)
    restored = ChampionChallengerFramework(state_path=framework.state_path)
    handler.framework = restored
    service2 = ResearchValidationService(restored, handler, forward).configure_evidence(ledger)
    assert service2.report()["forward_experiments"] == 1
    assert service2.report()["forward_cohorts"] == 1
    result = service2.consume("GAUNTLET", {"candidate_id": "c1"}, ["fixture-corpus"], now + 10)
    assert result["detail"]["status"] == "HISTORICAL_SURVIVOR"
    assert len(restored.trial_results) == 1


@pytest.mark.parametrize("mutation", ["old", "future", "availability", "duplicate_time", "changed_hash"])
def test_forward_cohort_freeze_and_chronology_fail_closed(desk, mutation):
    _, spec, _, service, _, _ = desk
    eid, now = register(desk)
    payload = cohort(desk, eid, now)
    payload = {**payload, "cohort_id": "cohort-2"}
    if mutation == "old": payload.update(event_at=1, available_at=2, decision_at=3, observed_at=4)
    if mutation == "future": payload["observed_at"] = now + 100
    if mutation == "availability": payload["available_at"] = now + .25
    if mutation == "changed_hash": payload["spec_hash"] = "wrong"
    result = service.consume("FORWARD_SHADOW", payload, ["different-native-event"], now + 1)
    assert result["disposition"] == "BLOCKED", result
    assert service.report()["forward_cohorts"] == 1


def test_native_paper_is_operational_persistent_and_never_counted_as_real_fill(desk):
    framework, _, handler, service, ledger, forward = desk
    eid, now = register(desk)
    cohort(desk, eid, now)
    payload, refs = native_outcome(desk, eid, now)
    result = service.consume("FORWARD_SHADOW", payload, refs, now + 3)
    assert result["disposition"] == "DEEPENING_TASK", result
    assert result["detail"]["paper"] is True and result["detail"]["real_fill"] is False
    assert result["detail"]["pnl"] == pytest.approx(18)
    assert forward.decisions == 1 and forward.real_fills == 0
    assert not framework.shadow_models and not result["detail"]["framework_recorded"]
    restored = ChampionChallengerFramework(state_path=framework.state_path)
    handler.framework = restored
    forward2 = ForwardEvidence(forward.path)
    service2 = ResearchValidationService(restored, handler, forward2).configure_evidence(ledger)
    repeated = service2.consume("FORWARD_SHADOW", payload, refs, now + 30)
    assert repeated == result and forward2.decisions == 1
    # An existing canonical shadow bucket is required; reconciliation cannot
    # fabricate a stage. Paper can then be flushed once and only once.
    restored.shadow_models["c1"] = {"hypothesis": restored.hypotheses["c1"],
                                      "started_at": now, "forward_results": deque()}
    assert service2.reconcile_forward() == {"transferred": 1}
    assert service2.reconcile_forward() == {"transferred": 0}
    assert len(restored.shadow_models["c1"]["forward_results"]) == 1
    assert not restored.shadow_models["c1"]["forward_results"][0]["real_fill"]


def test_real_fill_is_bound_to_observed_native_signed_receipts(desk):
    framework, _, _, service, _, forward = desk
    eid, now = register(desk)
    cohort(desk, eid, now)
    framework.shadow_models["c1"] = {"hypothesis": framework.hypotheses["c1"],
                                      "started_at": now, "forward_results": deque()}
    payload, refs = native_outcome(desk, eid, now, paper=False)
    result = service.consume("FORWARD_SHADOW", payload, refs, now + 3)
    assert result["disposition"] == "DEEPENING_TASK", result
    assert result["detail"]["real_fill"] and result["detail"]["framework_recorded"]
    assert forward.real_fills == 1
    assert len(framework.shadow_models["c1"]["forward_results"]) == 1


@pytest.mark.parametrize("mutation", ["historical", "aggregator", "cashflow", "omitted_cost", "future",
    "unbound_ref", "paper_as_real", "signature_missing", "simulation_missing", "wrong_cohort"])
def test_fabricated_or_relabelled_forward_outcomes_never_count(desk, mutation):
    _, _, _, service, ledger, forward = desk
    eid, now = register(desk)
    cohort(desk, eid, now)
    payload, refs = native_outcome(desk, eid, now, paper=mutation != "signature_missing")
    if mutation == "historical": payload["partition"] = "test"
    if mutation == "aggregator": payload["execution"]["route"] = "jupiter"
    if mutation == "cashflow": payload["realized_pnl_usd"] = 99999
    if mutation == "omitted_cost": payload["execution"].pop("fees_native")
    if mutation == "future": payload["exit_at"] = now + 100
    if mutation == "unbound_ref": refs.pop()
    if mutation == "paper_as_real": payload["scope"] = "observed_forward"
    if mutation in {"signature_missing", "simulation_missing", "wrong_cohort"}:
        # Append a validly hashed but semantically invalid canonical row; the
        # consumer must inspect the producer payload, not just a string hash.
        import json
        row = json.loads(ledger.path.read_text().splitlines()[-1])["payload"]
        if mutation == "signature_missing": row.pop("signature")
        if mutation == "simulation_missing": row.pop("simulation_ref")
        if mutation == "wrong_cohort": row["cohort_id"] = "another-cohort"
        reference = ledger.record_reference("research_forward_native_execution", row)
        payload["execution"]["exit_evidence_ref"] = reference
        refs[-1] = reference
    result = service.consume("FORWARD_SHADOW", payload, refs, now + 3)
    assert result["disposition"] == "BLOCKED", result
    assert forward.decisions == 0 and service.report()["paper_outcomes"] == 0


def test_changed_frozen_candidate_cannot_inherit_registered_forward(desk):
    framework, _, _, service, _, forward = desk
    eid, now = register(desk)
    cohort(desk, eid, now)
    payload, refs = native_outcome(desk, eid, now)
    framework.hypotheses["c1"].exact_candidate["size_rule"]["value"] = .1
    result = service.consume("FORWARD_SHADOW", payload, refs, now + 3)
    assert result["disposition"] == "BLOCKED" and not forward.decisions


def test_actual_planted_null_leakage_delay_cost_use_same_feed_and_gauntlet(desk):
    framework, spec, handler, service, _, _ = desk
    payload = {"candidate_id": "c1", "corpus_hash": _digest(handler._episodes()),
               "attacks": ["null", "leakage", "delay", "cost"], "delay_seconds": 3., "cost_multiplier": 30.}
    result = service.consume("VALIDATION_ATTACK", payload, ["fixture-declared-corpus"], spec.frozen_at + 1)
    assert result["disposition"] == "NEGATIVE_KNOWLEDGE", result
    controls = result["detail"]["experiments"]
    assert controls["null"]["falsified"]  # fixture feature is constant: null survives
    assert controls["leakage"]["future_mapping_rejected"]
    assert controls["leakage"]["late_availability_unknown"]
    assert controls["cost"]["falsified"]
    assert "results" in controls["delay"]
    assert len(framework.trial_results) == 3
    assert all(trial.stage == "VALIDATION_ATTACK" and not trial.passed for trial in framework.trial_results)
    assert not framework.shadow_models and not framework.champions


def test_attack_missing_declared_corpus_is_data_request_not_completed_work(desk):
    framework, spec, _, service, _, _ = desk
    result = service.consume("VALIDATION_ATTACK", {"candidate_id": "c1", "attacks": ["null"]}, [], spec.frozen_at + 1)
    assert result["disposition"] == "BLOCKED" and result["children"][0]["kind"] == "DATA_REQUEST"
    assert not framework.trial_results and not service.report()["planted_attack_receipts"]


def test_integration_config_alone_is_not_native_observation_or_execution_proof(desk):
    _, spec, _, service, _, _ = desk
    registry = SimpleNamespace(get_rpc=lambda chain: object())
    service.configure_integrations(registry, {"base": {"enabled": True, "native": True}})
    result = service.consume("PROTOCOL_INTEGRATION", {"chain": "base", "protocol": "uniswap_v2",
        "capabilities": ["native_factory_decode"]}, ["config-row"], spec.frozen_at + 1)
    assert result["disposition"] == "BLOCKED"
    assert not service.report()["integration_receipts"]


def test_actual_native_observer_capabilities_are_verified_but_not_execution(desk, tmp_path):
    from test_evm_observation import FakeRPC, observer
    _, spec, _, service, _, _ = desk
    rpc = FakeRPC()
    adapter = observer(tmp_path, rpc)
    def sink(event):
        return desk[4].record_reference("unknown_native_chain_observation", event.to_event().to_dict())
    asyncio.run(adapter.poll_once(sink))
    registry = SimpleNamespace(get_rpc=lambda chain: rpc)
    service.configure_integrations(registry, {adapter.chain_name: adapter})
    payload = {"chain": adapter.chain_name, "protocol": "uniswap_v2",
               "capabilities": ["observation", "historical_backfill", "finality", "native_factory_decode"]}
    result = service.consume("PROTOCOL_INTEGRATION", payload, ["canonical-lake:fixture-event"], spec.frozen_at + 1)
    assert result["disposition"] == "DEEPENING_TASK", result
    assert result["detail"]["adapter"]["observed_events"] > 0
    assert result["detail"]["execution_eligible"] is False
    payload["capabilities"] = ["execution"]
    assert service.consume("PROTOCOL_INTEGRATION", payload, [], spec.frozen_at + 1)["disposition"] == "BLOCKED"


def test_paper_and_legacy_marks_never_increase_canary_or_champion_real_fill_credit(desk):
    framework, _, _, service, _, _ = desk
    eid, now = register(desk)
    cohort(desk, eid, now)
    payload, refs = native_outcome(desk, eid, now)
    result = service.consume("FORWARD_SHADOW", payload, refs, now + 3)["detail"]
    framework.canary_duration = 0
    framework.canary_models["c1"] = {"hypothesis": framework.hypotheses["c1"],
        "started_at": 1, "forward_results": deque(), "position_count": 0, "total_pnl": 0.}
    assert framework.record_forward_result("c1", result) is False
    assert framework.canary_models["c1"]["position_count"] == 0
    # Old checkpoints can contain untyped marks and historical counters. They
    # remain visible, but the consumer recomputes credit from actual typed rows.
    framework.canary_models["c1"]["forward_results"].extend([{"pnl": 1000, "elogw": 1}] * 25)
    framework.canary_models["c1"].update(position_count=25, total_pnl=25000)
    asyncio.run(framework._evaluate_canary_models())
    assert not framework.champions
    assert framework.hypotheses["c1"].status == "retired"


def test_paper_counts_only_existing_canonical_shadow_and_cannot_be_relabelled(desk):
    framework, _, _, service, _, _ = desk
    eid, now = register(desk)
    cohort(desk, eid, now)
    framework.shadow_models["c1"] = {"hypothesis": framework.hypotheses["c1"],
        "started_at": now, "forward_results": deque()}
    payload, refs = native_outcome(desk, eid, now)
    result = service.consume("FORWARD_SHADOW", payload, refs, now + 3)
    assert result["detail"]["framework_recorded"]
    row = framework.shadow_models["c1"]["forward_results"][0]
    assert framework._forward_credit(row, allow_paper=True)
    assert not framework._forward_credit(row)
    forged = {**row, "scope": "observed_forward", "real_fill": True}
    assert not framework._forward_credit(forged)


def test_nested_exact_candidate_and_untyped_forward_rejection_preserve_seal(desk):
    framework, spec, _, service, _, _ = desk
    result = service.consume("GAUNTLET", {"candidate": spec.exact_candidate}, [], spec.frozen_at + 1)
    assert result["disposition"] == "DEEPENING_TASK"
    changed = copy.deepcopy(spec.exact_candidate)
    changed["size_rule"]["value"] *= 2
    rejected = service.consume("GAUNTLET", {"candidate": changed}, [], spec.frozen_at + 2)
    assert rejected["disposition"] == "BLOCKED"
    assert len(framework.trial_results) == 1


def test_same_forward_cohort_cannot_be_counted_twice_under_new_event_id(desk):
    _, _, _, service, _, forward = desk
    eid, now = register(desk)
    cohort(desk, eid, now)
    payload, refs = native_outcome(desk, eid, now)
    assert service.consume("FORWARD_SHADOW", payload, refs, now + 3)["disposition"] == "DEEPENING_TASK"
    changed = {**payload, "event_id": "renamed-duplicate"}
    assert service.consume("FORWARD_SHADOW", changed, refs, now + 3)["disposition"] == "BLOCKED"
    assert service.report()["paper_outcomes"] == 1 and forward.decisions == 1


def test_checkpoint_retries_transient_windows_sharing_lock_with_unique_temp(tmp_path, monkeypatch):
    import os
    path = tmp_path / "checkpoint.json"
    framework = ChampionChallengerFramework(state_path=path)
    actual = os.replace
    calls = []
    def sharing_locked(source, destination):
        calls.append(str(source))
        if len(calls) <= 2:
            raise PermissionError("offline sharing-lock control")
        return actual(source, destination)
    monkeypatch.setattr("src.strategies.champion_challenger.os.replace", sharing_locked)
    framework.checkpoint_research_component("receipt", {"observations": 1})
    assert len(calls) == 3 and not list(tmp_path.glob("*.tmp"))
    assert ChampionChallengerFramework(state_path=path).get_research_component("receipt") == {"observations": 1}


def test_checkpoint_permanent_permission_failure_remains_visible_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.json"
    framework = ChampionChallengerFramework(state_path=path)
    framework.checkpoint_research_component("receipt", {"observations": 1})
    def forbidden(*args):
        raise PermissionError("offline permanent permission control")
    monkeypatch.setattr("src.strategies.champion_challenger.os.replace", forbidden)
    with pytest.raises(PermissionError):
        framework.checkpoint_research_component("receipt", {"observations": 2})
    assert not list(tmp_path.glob("*.tmp"))
    assert ChampionChallengerFramework(state_path=path).get_research_component("receipt") == {"observations": 1}

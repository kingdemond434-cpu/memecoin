"""Raw public evidence must reach durable, honest scientific dispositions."""
import time

import pytest

from src.research.frontier import FrontierIndex
from src.research.trade_evidence import TradeEvidenceLedger
from src.research.unknown_engine import UnknownOpportunityEngine
from src.strategies.champion_challenger import ChampionChallengerFramework
from test_scientific_core import exact_candidate


def engine(root):
    return UnknownOpportunityEngine(
        ChampionChallengerFramework(state_path=str(root / "champion.json")),
        FrontierIndex(root / "frontier.sqlite"), TradeEvidenceLedger(root / "evidence.jsonl"))


def raw(**changes):
    now = time.time()
    result = {"source_id": "public-fixture", "source_url": "https://example.org/research",
              "rights": "PUBLIC", "event_at": now - 2, "available_at": now - 1,
              "observed_at": now, "mechanism": "independent actor arrival",
              "candidate_features": ["independent_buyers"], "origin": "FIXTURE"}
    result.update(changes)
    return result


def test_public_signal_is_persisted_before_conversion_and_survives_restart(tmp_path):
    first = engine(tmp_path)
    payload = raw()
    job_id = first.ingest(payload)
    assert first.frontier.job(job_id)["state"] == "READY"
    second = engine(tmp_path)
    assert second.ingest(payload) == job_id
    receipt = second.run_pass()
    assert receipt["processed"] == 1
    assert receipt["results"][0]["disposition"] == "MECHANISM_HYPOTHESIS"
    assert second.frontier.job(job_id)["state"] == "ACKED"
    assert second.frontier.report()["pending"] == 1
    assert TradeEvidenceLedger.verify(tmp_path / "evidence.jsonl")[0]


@pytest.mark.parametrize("changes", [
    {"rights": "UNKNOWN"}, {"rights": "AUTHORIZED"},
    {"available_at": time.time() + 3600}, {"source_url": "file:///private"},
])
def test_unlawful_or_future_evidence_never_becomes_a_hypothesis(tmp_path, changes):
    service = engine(tmp_path)
    service.ingest(raw(**changes))
    result = service.run_pass()
    assert result["results"][0]["disposition"] == "BLOCKED"
    assert not service.framework.hypotheses


def test_complete_candidate_is_frozen_before_canonical_gauntlet_work(tmp_path):
    service = engine(tmp_path)
    candidate = exact_candidate("flow-rule")
    service.ingest(raw(exact_candidate=candidate))
    result = service.run_pass()
    assert result["results"][0]["disposition"] == "EXACT_CANDIDATE"
    spec = service.framework.hypotheses["flow-rule"]
    assert spec.frozen_hash and spec.frozen_at
    assert not service.framework.is_live("flow-rule")
    state = service.frontier.report()
    assert any(row["kind"] == "GAUNTLET" for row in state["by_kind"])
    # Unknown validation costs cannot silently turn into a free gauntlet.
    assert state["cost_blocked"] == 1


def test_vague_strategy_is_not_filled_in_with_invented_execution_rules(tmp_path):
    service = engine(tmp_path)
    service.ingest(raw(exact_candidate={"candidate_id": "buy-smart-money"}))
    assert service.run_pass()["results"][0]["disposition"] == "BLOCKED"
    assert not service.framework.hypotheses


def test_evidence_failure_cannot_ack_or_erase_useful_work(tmp_path, monkeypatch):
    service = engine(tmp_path)
    job_id = service.ingest(raw())
    monkeypatch.setattr(service.evidence, "record", lambda *args, **kwargs: None)
    result = service.run_pass()
    assert result["results"][0]["disposition"] == "BLOCKED"
    assert service.frontier.job(job_id)["state"] != "ACKED"


def test_secrets_are_refused_before_persistence(tmp_path):
    service = engine(tmp_path)
    with pytest.raises(ValueError, match="credentials"):
        service.ingest(raw(nested={"private_key": "not-a-real-key"}))
    assert not service.evidence.path.exists()


def test_future_chain_is_retained_but_not_registered(tmp_path):
    service = engine(tmp_path)
    service.ingest(raw(discovery_kind="chain", identity="future-chain",
                       available_at=time.time() + 1000))
    assert service.frontier.chain("future-chain") is None
    assert service.run_pass()["results"][0]["disposition"] == "BLOCKED"
    assert service.evidence.path.exists()


def test_dynamic_chain_identity_comes_from_evidence_not_a_static_list(tmp_path):
    service = engine(tmp_path)
    identity = "new-family-chain-from-public-fixture"
    service.ingest(raw(discovery_kind="chain", identity=identity,
                       chain_family="future-family", metrics={"launches_per_day": 12}))
    registry = service.frontier.chain(identity)
    assert registry["stage"] == "DISCOVERED"
    assert registry["metrics"]["launches_per_day"] == 12
    assert registry["metrics"]["route_quality"] is None

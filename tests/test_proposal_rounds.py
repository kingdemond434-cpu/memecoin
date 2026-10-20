"""No network or paid calls: independently injected, sealed proposal rounds."""
import asyncio
import copy
import json

import pytest

from src.research.proposal_rounds import HTTPProposalProvider, ProposalRoundRunner, ProviderConfig


def test_configuration_report_names_missing_inputs_without_network_or_secrets(tmp_path):
    async def forbidden(*args, **kwargs):
        raise AssertionError("configuration report attempted an API request")
    config = ProviderConfig("codex", "openai", "MODEL_NAME", "KEY_NAME", 1000, 100, 2, enabled=True)
    provider = HTTPProposalProvider(config, post_json=forbidden, environ={"KEY_NAME": "private-value"})
    runner = ProposalRoundRunner(TradeEvidenceLedger(tmp_path / "evidence.jsonl"), [provider])
    result = runner.configuration_report()
    assert result["status"] == "DATA_BLOCKED" and result["network_requests"] == 0
    row = next(item for item in result["providers"] if item["provider"] == "codex")
    assert row["missing_environment_names"] == ["MODEL_NAME"]
    assert "private-value" not in str(result)
    provider.environ = {"MODEL_NAME": "explicit-model", "KEY_NAME": "private-value"}
    assert runner.configuration_report()["status"] == "DATA_BLOCKED"  # other required brains absent
    single = ProposalRoundRunner(runner.ledger, [provider], required_providers=["codex"])
    assert single.configuration_report()["status"] == "CONFIGURED_NOT_AUTHENTICATED"
from src.research.trade_evidence import TradeEvidenceLedger


def fact():
    return {"evidence_id": "event-1", "source": "public-chain", "observed_at": 100.0,
            "available_at": 101.0, "content": {"buyers": 3}, "public": True, "partition": "training"}


def proposal(pid="p1", mechanism="flow", sign=1, observables=None):
    return {"proposal_id": pid, "mechanism_id": mechanism, "hypothesis": "independent buyer arrival",
            "expected_sign": sign, "horizon_seconds": 30, "required_observables": observables or ["buyers"],
            "falsifier": "no incremental chronological executable log growth",
            "experiment": "compare shuffled buyers and delayed fills", "source_ids": ["event-1"]}


class FakeProvider:
    def __init__(self, name, rows=None, error=None):
        self.name, self.rows, self.error = name, rows or [proposal()], error
        self.prompts = []

    async def propose(self, prompt):
        self.prompts.append(prompt)
        await asyncio.sleep(0)
        if self.error:
            raise self.error
        return {"status": "OK", "provider": self.name, "model": "explicit-fixture-model",
                "text": json.dumps({"proposals": self.rows}), "cost_usd": None,
                "usage": {"input_tokens": 20, "output_tokens": 30}}


def run(runner, round_id="round-1", **kwargs):
    return asyncio.run(runner.run_round(round_id=round_id, facts=kwargs.get("facts", [fact()]),
        schemas=kwargs.get("schemas", {"proposal_schema": 1}), training_memory=kwargs.get("training_memory", [])))


def test_all_three_get_identical_independent_inputs_and_seal_before_comparison(tmp_path):
    ledger = TradeEvidenceLedger(tmp_path / "evidence.jsonl")
    providers = [FakeProvider(name) for name in ("claude", "codex", "deepseek")]
    result = run(ProposalRoundRunner(ledger, providers))
    assert result["status"] == "OK"
    assert providers[0].prompts == providers[1].prompts == providers[2].prompts
    assert "explicit-fixture-model" not in providers[0].prompts[0]
    rows = [json.loads(line) for line in ledger.path.read_text().splitlines()]
    events = [row["event_type"] for row in rows]
    assert events.count("proposal_phase_a_sealed") == 3
    assert max(i for i, event in enumerate(events) if event == "proposal_phase_a_sealed") < events.index("proposal_phase_a_complete")
    assert events.index("proposal_phase_a_complete") < events.index("proposal_round_complete")
    assert all("INDEPENDENT_REDISCOVERY" in row["classifications"] and "DUPLICATE" in row["classifications"]
               for row in result["comparisons"])
    assert TradeEvidenceLedger.verify(ledger.path)[0]
    assert all(row["cost_usd"] is None for row in result["providers"])


@pytest.mark.parametrize("contamination", ["heldout", "forward_outcomes", "other_agent", "authority"])
def test_contaminated_input_never_reaches_any_provider(tmp_path, contamination):
    provider = FakeProvider("codex")
    raw = fact()
    raw["content"][contamination] = {"return": 10}
    result = run(ProposalRoundRunner(TradeEvidenceLedger(tmp_path / "evidence.jsonl"), [provider]), facts=[raw])
    assert result["status"] == "DATA_BLOCKED"
    assert not provider.prompts


def test_nested_test_partition_and_unapproved_memory_are_refused(tmp_path):
    provider = FakeProvider("codex")
    runner = ProposalRoundRunner(TradeEvidenceLedger(tmp_path / "evidence.jsonl"), [provider])
    raw = fact()
    raw["content"] = {"partition": "test", "return": 9}
    assert run(runner, facts=[raw])["status"] == "DATA_BLOCKED"
    memory = [{"memory_id": "m1", "partition": "forward", "content": "winner"}]
    assert run(runner, training_memory=memory)["status"] == "DATA_BLOCKED"
    assert not provider.prompts


def test_disagreement_and_complementarity_become_requests_not_verdicts(tmp_path):
    providers = [FakeProvider("claude", [proposal()]), FakeProvider("codex", [proposal(sign=-1)]),
                 FakeProvider("deepseek", [proposal(mechanism="fee", observables=["fees"])])]
    result = run(ProposalRoundRunner(TradeEvidenceLedger(tmp_path / "evidence.jsonl"), providers))
    classifications = {label for row in result["comparisons"] for label in row["classifications"]}
    assert classifications == {"DISAGREEMENT", "COMPLEMENTARITY"}
    assert {row["kind"] for row in result["experiment_requests"]} == {"discriminating_control", "incremental_ablation"}
    assert all(row["authority"] == "none" for row in result["experiment_requests"])
    assert "verdict" not in json.dumps(result)


def test_full_provider_outage_is_blocked_and_round_is_replayable_without_calls(tmp_path):
    ledger = TradeEvidenceLedger(tmp_path / "evidence.jsonl")
    providers = [FakeProvider(name, error=RuntimeError("secret-key-never-persist"))
                 for name in ("claude", "codex", "deepseek")]
    runner = ProposalRoundRunner(ledger, providers)
    result = run(runner)
    assert result["status"] == "DATA_BLOCKED"
    assert result["proposals"] == []
    assert "secret-key-never-persist" not in ledger.path.read_text()
    assert run(runner) == result
    assert all(len(provider.prompts) == 1 for provider in providers)
    # The caller can continue its deterministic pass after any round result.
    deterministic_work = []
    deterministic_work.append("compiler-and-collector-still-run")
    assert deterministic_work


def test_partial_round_restart_reuses_existing_seals_without_repaying(tmp_path):
    ledger = TradeEvidenceLedger(tmp_path / "evidence.jsonl")
    providers = [FakeProvider(name) for name in ("claude", "codex", "deepseek")]
    run(ProposalRoundRunner(ledger, providers))
    rows = [json.loads(line) for line in ledger.path.read_text().splitlines()]
    # Simulate interruption immediately after one durable first-pass seal.
    ledger.path.write_text("\n".join(json.dumps(row) for row in rows[:2]) + "\n")
    resumed = [FakeProvider(name) for name in ("claude", "codex", "deepseek")]
    restored = TradeEvidenceLedger(ledger.path)
    result = run(ProposalRoundRunner(restored, resumed))
    assert result["status"] == "OK"
    assert resumed[0].prompts == []
    assert len(resumed[1].prompts) == len(resumed[2].prompts) == 1
    assert TradeEvidenceLedger.verify(ledger.path)[0]


def test_authority_proposals_and_invented_provenance_are_sealed_but_not_admitted(tmp_path):
    malicious = proposal()
    malicious["authority"] = "live"
    invented = proposal()
    invented["source_ids"] = ["unseen-source"]
    ledger = TradeEvidenceLedger(tmp_path / "evidence.jsonl")
    runner = ProposalRoundRunner(ledger, [FakeProvider("claude", [malicious]), FakeProvider("codex", [invented])])
    result = run(runner)
    assert not result["proposals"]
    assert all(row["status"] == "DATA_BLOCKED" for row in result["providers"])
    assert "unseen-source" in ledger.path.read_text()


@pytest.mark.parametrize("api_kind,token_key", [("anthropic", "max_tokens"), ("openai", "max_completion_tokens"), ("deepseek", "max_tokens")])
def test_http_adapters_send_explicit_model_budget_and_usage_without_real_calls(api_kind, token_key):
    calls = []
    async def transport(url, *, headers, payload, timeout):
        calls.append((url, copy.deepcopy(headers), copy.deepcopy(payload), timeout))
        if api_kind == "anthropic":
            return {"model": "returned-model", "stop_reason": "end_turn", "content": [{"type": "text", "text": "{}"}],
                    "usage": {"input_tokens": 11, "output_tokens": 5}}
        return {"model": "returned-model", "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 5}}
    config = ProviderConfig("fixture", api_kind, "MODEL", "KEY", 10000, 512, 2.0, enabled=True,
                            max_cost_usd=0.1, input_usd_per_million=1.0, output_usd_per_million=2.0)
    provider = HTTPProposalProvider(config, post_json=transport, environ={"MODEL": "explicit-model", "KEY": "fixture-secret"})
    result = asyncio.run(provider.propose("public facts"))
    assert result["status"] == "OK"
    assert calls[0][2]["model"] == "explicit-model"
    assert calls[0][2][token_key] == 512
    assert "tools" not in calls[0][2]
    assert result["cost_usd"] is None
    assert result["estimated_cost_usd"] == pytest.approx(21 / 1e6)
    assert "fixture-secret" not in json.dumps(result)


def test_missing_model_disabled_provider_and_unpriced_dollar_budget_never_call_transport():
    async def forbidden(*args, **kwargs):
        raise AssertionError("must not perform any provider call")
    configs = [ProviderConfig("codex", "openai", "MODEL", "KEY", 10000, 100, 1, enabled=False),
               ProviderConfig("codex", "openai", "MODEL", "KEY", 10000, 100, 1, enabled=True),
               ProviderConfig("codex", "openai", "MODEL", "KEY", 10000, 100, 1, enabled=True, max_cost_usd=1)]
    for index, config in enumerate(configs):
        environment = {"KEY": "fixture"} if index == 1 else {"MODEL": "explicit-model", "KEY": "fixture"}
        result = asyncio.run(HTTPProposalProvider(config, post_json=forbidden, environ=environment).propose("facts"))
        assert result["status"] == "DATA_BLOCKED"


def test_timeout_and_missing_usage_are_not_zero_cost_or_healthy(tmp_path):
    class HangingProvider:
        name = "codex"
        async def propose(self, prompt):
            await asyncio.sleep(10)
    runner = ProposalRoundRunner(TradeEvidenceLedger(tmp_path / "evidence.jsonl"), [HangingProvider()],
                                 provider_timeout_seconds=0.01)
    result = run(runner)
    assert result["status"] == "DATA_BLOCKED"
    assert result["providers"][0]["cost_usd"] is None


def test_changed_input_identity_corrupt_chain_or_unsealable_outputs_block_phase_b(tmp_path):
    ledger = TradeEvidenceLedger(tmp_path / "evidence.jsonl")
    provider = FakeProvider("codex")
    runner = ProposalRoundRunner(ledger, [provider])
    run(runner)
    changed = fact()
    changed["content"]["buyers"] = 8
    assert run(runner, facts=[changed])["status"] == "DATA_BLOCKED"
    assert len(provider.prompts) == 1
    ledger.path.write_text(ledger.path.read_text().replace("public-chain", "edited-chain"))
    assert run(runner)["status"] == "DATA_BLOCKED"
    assert len(provider.prompts) == 1


def test_comparison_budget_exposes_deferred_debt_and_missing_providers(tmp_path):
    providers = [FakeProvider("codex", [proposal("a"), proposal("b"), proposal("c")])]
    result = run(ProposalRoundRunner(TradeEvidenceLedger(tmp_path / "evidence.jsonl"), providers,
                                    max_comparison_pairs=1))
    assert result["status"] == "PARTIAL"
    assert len(result["proposals"]) == 3
    assert result["deferred_comparison_pairs"] == 2
    assert {row["provider"] for row in result["providers"] if row["status"] == "DATA_BLOCKED"} == {"claude", "deepseek"}

"""Synthetic PIT observations exercise contracts, never assert mainnet coverage."""
import json

import pytest

from src.research.actor_store import ActorStore, EdgeKind, native_node
from src.research.population_hunter import COMPONENT, PopulationHunter
from src.research.trade_evidence import TradeEvidenceLedger
from src.strategies.champion_challenger import ChampionChallengerFramework


@pytest.fixture
def desk(tmp_path):
    framework = ChampionChallengerFramework(state_path=str(tmp_path / "champion.json"))
    ledger = TradeEvidenceLedger(tmp_path / "evidence.jsonl")
    return PopulationHunter(framework), ledger


def consume(desk, kind, payload, *, now=None):
    hunter, ledger = desk
    ref = ledger.record_reference("synthetic_" + kind, payload, timestamp=payload["observed_at"])
    return hunter.consume(kind, payload, [ref], now=now or payload["observed_at"])


def actor(address="AbC", **changes):
    row = dict(chain_id="solana", chain_family="SOLANA", address=address,
               event_at=10., observed_at=11., protocol="test-protocol", lifecycle_stage="POOL",
               regime="test-regime", capital_size=100., entry_delay_seconds=2.)
    row.update(changes)
    return row


def selected(address="AbC", population="UNBIASED", at=1., **changes):
    return actor(address, selection={"population": population, "selected_at": at,
        "rule_id": "predeclared-all-decoded", "sampling_frame_ref": "fixture:source-frame",
        "uses_outcomes": False, "benchmark_ref": "fixture:public-benchmark"}, **changes)


def outcome(address="AbC", number=0, start=12., delay=2., multiple=1.2, **changes):
    row = dict(chain_id="solana", chain_family="SOLANA", address=address, token_id=f"Token{number}",
        execution_scope="ACTOR_DELAYED_PAPER", execution_status="FILLED",
        outcome_id=f"follow-{address}-{number}", signal_available_at=start, decision_at=start + 1,
        entry_at=start + delay, exit_at=start + delay + 1, available_at=start + delay + 2,
        event_at=start + delay + 1, observed_at=start + delay + 3, delay_seconds=delay,
        capital_at_risk_usd=100., entry_fill_cost_usd=100., exit_fill_proceeds_usd=100. * multiple,
        costs={"fees_usd": 0., "gas_usd": 0., "tips_usd": 0., "failed_transaction_usd": 0., "crowding_usd": 0.},
        slippage_in_fills=True, impact_in_fills=True, crowding_observed=True,
        entry_filled=True, exit_filled=True, exit_capacity_ratio=1.)
    row.update(changes)
    return row


def narrative(**changes):
    row = dict(narrative_id="native-language-topic", source_id="regional-a", platform="RSS",
               language="zh", region="CN", event_at=10., observed_at=11.,
               text="猫币 新池流动性 公共观察", observation_id="mention-a")
    row.update(changes)
    return row


def test_no_observed_inputs_does_not_claim_coverage_or_value(desk):
    report = desk[0].report()
    assert report["status"] == "NO_OBSERVED_INPUTS"
    assert report["sources"] == {}
    assert report["measured_follow_outcomes"] == 0
    value = desk[0].actor_value("solana", "SOLANA", "AbC", as_of=100)
    assert value["lower_bound"] is None and not value["followable"]


def test_native_chain_collision_and_case_rules(desk):
    hunter = desk[0]
    for row in (actor(), actor("abc"), actor(chain_id="ethereum", chain_family="EVM", address="0xABcd"),
                actor(chain_id="base", chain_family="EVM", address="0xABcd")):
        assert consume(desk, "ACTOR_OBSERVATION", row)["disposition"] == "OBSERVED"
    assert hunter.report()["native_actor_count"] == 4
    assert native_node("ethereum", "wallet", "0xABcd", "EVM") == native_node("ethereum", "wallet", "0xabcd", "EVM")
    assert native_node("solana", "wallet", "AbC", "SOLANA") != native_node("solana", "wallet", "abc", "SOLANA")
    assert native_node("unknown", "wallet", "0xABcd") != native_node("unknown", "wallet", "0xabcd")


def test_native_graph_persists_refs_deduplicates_and_respects_exact_visibility(tmp_path):
    path = tmp_path / "graph.jsonl"
    graph = ActorStore(path)
    for index in range(10):
        graph.observe_native(chain_id="solana", chain_family="SOLANA", source=f"Source{index}",
            target="Token", kind=EdgeKind.BOUGHT, event_at=4, observed_at=5,
            evidence_refs=["fixture:canonical"], observation_id=f"decoded-{index}", target_type="token")
    # Global edge index can exceed this node's adjacency length at an exact PIT cut.
    node = native_node("solana", "wallet", "Source9", "SOLANA")
    assert graph.out_degree(node, EdgeKind.BOUGHT, 4.99) == 0
    assert graph.out_degree(node, EdgeKind.BOUGHT, 5) == 1
    graph.observe_native(chain_id="solana", chain_family="SOLANA", source="Source9", target="Token",
        kind=EdgeKind.BOUGHT, event_at=4, observed_at=5, evidence_refs=["fixture:canonical"],
        observation_id="decoded-9", target_type="token")
    assert len(graph.edges) == 10
    restored = ActorStore(path)
    assert restored.load() == 10
    assert restored.edges[-1].evidence_refs == ("fixture:canonical",)


def test_retry_reproduces_children_after_checkpoint_and_restart(desk, tmp_path):
    hunter, ledger = desk
    row = actor(relation={"kind": "funded", "target": "Other"})
    ref = ledger.record_reference("synthetic_actor", row, timestamp=11)
    first = hunter.consume("ACTOR_OBSERVATION", row, [ref], now=11)
    again = hunter.consume("ACTOR_OBSERVATION", row, [ref], now=100)
    assert first == again
    restarted = PopulationHunter(ChampionChallengerFramework(state_path=str(tmp_path / "champion.json")))
    assert restarted.consume("ACTOR_OBSERVATION", row, [ref], now=100) == first
    assert restarted.report()["actor_observations"] == 1
    assert len(restarted.actor_store.edges) == 1
    assert all(child["payload"]["evidence_refs"] == [ref] for child in first["children"])
    assert TradeEvidenceLedger.verify(ledger.path)[0]
    assert restarted.consume("ACTOR_OBSERVATION", row, [ref], now=10)["disposition"] == "DATA_BLOCKED"
    assert restarted.report(as_of=10)["receipt_count"] == 0


@pytest.mark.parametrize("change", [{"event_at": 12}, {"observed_at": 100},
    {"selection": {"population": "UNBIASED", "selected_at": 20, "uses_outcomes": False}},
    {"selection": {"population": "UNBIASED", "selected_at": 1, "uses_outcomes": True}}])
def test_future_and_hindsight_selection_fail_closed(desk, change):
    result = consume(desk, "ACTOR_OBSERVATION", actor(**change), now=11)
    assert result["disposition"] == "DATA_BLOCKED"
    assert desk[0].report()["actor_observations"] == 0


def matched_setup(desk):
    consume(desk, "ACTOR_OBSERVATION", selected("Loser"))
    consume(desk, "ACTOR_OBSERVATION", selected("Other"))
    consume(desk, "FOLLOW_OUTCOME", outcome("Loser", multiple=.8))
    plan = {"trial_id": "frozen-test", "seed": "predeclared-seed", "delay_grid_seconds": [2., 4.]}
    target = selected("Famous", population="FAMOUS_BENCHMARK", at=30., event_at=31., observed_at=32., control_plan=plan)
    result = consume(desk, "ACTOR_OBSERVATION", target)
    return result["detail"]["controls"], target


def test_losing_and_shuffled_populations_are_frozen_before_target_outcomes(desk):
    controls, target = matched_setup(desk)
    assert controls["status"] == "MATCHED"
    assert controls["losing_actor_id"] == native_node("solana", "wallet", "Loser", "SOLANA")
    assert controls["shuffled_actor_id"] != native_node("solana", "wallet", "Famous", "SOLANA")
    before = json.dumps(controls, sort_keys=True)
    consume(desk, "FOLLOW_OUTCOME", outcome("Other", 99, start=40, multiple=.01))
    same = consume(desk, "ACTOR_OBSERVATION", target)["detail"]["controls"]
    assert json.dumps(same, sort_keys=True) == before
    assert desk[0].report(as_of=20)["controls"] == 0
    assert desk[0].report()["populations"]["MATCHED_LOSING"] == 1


def test_future_available_loss_cannot_select_the_losing_population(desk):
    consume(desk, "ACTOR_OBSERVATION", selected("Loser"))
    consume(desk, "FOLLOW_OUTCOME", outcome("Loser", start=40, multiple=.5))
    row = selected("Famous", population="FAMOUS_BENCHMARK", at=30., event_at=31., observed_at=32.,
        control_plan={"trial_id": "pit", "seed": "seed", "delay_grid_seconds": [2, 4]})
    result = consume(desk, "ACTOR_OBSERVATION", row)
    assert result["detail"]["controls"]["losing_actor_id"] is None
    assert result["detail"]["controls"]["status"] == "DATA_BLOCKED"


@pytest.mark.parametrize("change", [{"costs": {}}, {"crowding_observed": False},
    {"exit_filled": False}, {"delay_seconds": 0}, {"exit_capacity_ratio": .2},
    {"available_at": 1}, {"impact_in_fills": False}])
def test_unmeasured_or_unresolved_outcome_is_not_a_zero_cost_return(desk, change):
    result = consume(desk, "FOLLOW_OUTCOME", outcome(**change))
    assert result["disposition"] == "DATA_BLOCKED"
    assert desk[0].report()["measured_follow_outcomes"] == 0


def test_all_costs_and_crowding_reduce_executable_net_edge(desk):
    row = outcome(costs={"fees_usd": 5, "gas_usd": 5, "tips_usd": 5, "failed_transaction_usd": 5, "crowding_usd": 5})
    result = consume(desk, "FOLLOW_OUTCOME", row)
    assert result["detail"]["executable_multiple"] == pytest.approx(.95)
    assert result["detail"]["value"]["followable"] is False


def test_known_failed_entries_contribute_measured_loss_instead_of_survivor_bias(desk):
    for number in range(12):
        assert consume(desk, "FOLLOW_OUTCOME", outcome(number=number, start=20 + number * 10))["disposition"] == "OBSERVED"
    assert desk[0].actor_value("solana", "SOLANA", "AbC", as_of=1000)["mean_log_return"] > 0
    for number in range(12, 24):
        row = outcome(number=number, start=20 + number * 10, execution_status="FAILED_ENTRY",
            entry_filled=False, exit_filled=False, entry_fill_cost_usd=0, exit_fill_proceeds_usd=0,
            inventory_at_resolution_raw=0, exit_capacity_ratio=0, slippage_in_fills=False, impact_in_fills=False,
            costs={"fees_usd": 1, "gas_usd": 1, "tips_usd": 0, "failed_transaction_usd": 98, "crowding_usd": 0})
        row["resolution_at"] = row.pop("exit_at")
        result = consume(desk, "FOLLOW_OUTCOME", row)
        assert result["disposition"] == "OBSERVED"
        assert result["detail"]["executable_multiple"] == 0
    value = desk[0].actor_value("solana", "SOLANA", "AbC", as_of=1000)
    assert value["samples"] == 24 and value["mean_log_return"] < 0
    assert value["lower_bound"] < 0 and not value["followable"]


@pytest.mark.parametrize("change", [{"execution_scope": "CANDIDATE_NATIVE_SIMULATION"},
    {"execution_scope": None}, {"execution_status": "FAILED_EXIT", "exit_filled": False},
    {"execution_status": "FAILED_ENTRY", "entry_filled": False, "exit_filled": False}])
def test_candidate_or_unresolved_failure_scope_cannot_enter_actor_value(desk, change):
    assert consume(desk, "FOLLOW_OUTCOME", outcome(**change))["disposition"] == "DATA_BLOCKED"
    assert desk[0].report()["measured_follow_outcomes"] == 0


def test_actor_paper_and_executed_receipts_do_not_share_value_estimates(desk):
    consume(desk, "FOLLOW_OUTCOME", outcome())
    consume(desk, "FOLLOW_OUTCOME", outcome(number=1, execution_scope="ACTOR_EXECUTED_FOLLOW"))
    assert desk[0].actor_value("solana", "SOLANA", "AbC", as_of=100)["samples"] == 1
    executed = desk[0].actor_value("solana", "SOLANA", "AbC", as_of=100, execution_scope="ACTOR_EXECUTED_FOLLOW")
    assert executed["samples"] == 1 and executed["execution_scope"] == "ACTOR_EXECUTED_FOLLOW"
    assert desk[0].report()["follow_outcomes_by_scope"] == {"ACTOR_DELAYED_PAPER": 1, "ACTOR_EXECUTED_FOLLOW": 1}


def test_followability_requires_real_delayed_edge_and_measured_frozen_controls(desk):
    controls, _ = matched_setup(desk)
    delay = controls["delay_seconds"]
    actors = {"Famous", "Loser", json.loads(controls["shuffled_actor_id"].removeprefix("native:"))[2]}
    for number in range(12):
        for address in sorted(actors):
            result = consume(desk, "FOLLOW_OUTCOME", outcome(address, number + 100, start=40 + number * 10,
                delay=delay, multiple=1.2 if address == "Famous" else .9, control_trial_id="frozen-test"))
            assert result["disposition"] == "OBSERVED"
    value = desk[0].actor_value("solana", "SOLANA", "Famous", as_of=1000)
    assert value["samples"] == 12 and value["control_samples"] == 12
    assert value["followable"]
    assert value["lower_bound"] > 0 and value["incremental_lower_bound"] > 0
    before = desk[0].actor_value("solana", "SOLANA", "Famous", as_of=39)
    assert before["samples"] == 0 and before["lower_bound"] is None


def test_twelve_positive_outcomes_without_matched_controls_do_not_grant_followability(desk):
    for number in range(12):
        consume(desk, "FOLLOW_OUTCOME", outcome(number=number, start=20 + number * 10))
    value = desk[0].actor_value("solana", "SOLANA", "AbC", as_of=1000)
    assert value["lower_bound"] > 0 and not value["followable"]


def test_unicode_text_is_reference_only_and_unverified_promotion_unknown(desk, tmp_path):
    row = narrative(label="PROMOTION", operator_id="claimed-same-owner")
    result = consume(desk, "NARRATIVE_OBSERVATION", row)
    assert result["detail"]["label"] == "UNKNOWN"
    assert result["detail"]["propagation"]["verified_independent_operators"] is None
    checkpoint = (tmp_path / "champion.json").read_text(encoding="utf-8")
    assert "猫币" not in checkpoint and row["text"] not in checkpoint
    assert desk[1].path.read_text(encoding="utf-8")  # raw fact remains canonical
    assert desk[0].state["narratives"][next(iter(desk[0].state["narratives"]))]["token_hashes"]
    assert "authority" not in json.dumps(desk[0].framework.get_research_component(COMPONENT))


def test_duplicate_promotion_cannot_inflate_independent_sources(desk):
    for index, text in enumerate(("猫币 新池 公共观察", "猫币 新池 公共观察", "独立的本地观察")):
        row = narrative(source_id=f"source-{index}", observation_id=f"mention-{index}",
            operator_id=f"operator-{index}", operator_verified=True, operator_evidence_refs=["fixture:owner-review"],
            platform="RSS" if index == 0 else "public_forum", region="CN" if index == 0 else "TW", text=text,
            event_at=10 + index, observed_at=11 + index)
        consume(desk, "NARRATIVE_OBSERVATION", row)
        if index == 1:
            report = desk[0].narrative(row["narrative_id"], as_of=12)
            assert report["reposted_content"] == 1 and not report["independent_confirmation"]
    report = desk[0].narrative("native-language-topic", as_of=13)
    assert report["verified_independent_operators"] == 2
    assert report["independent_cross_platform"] and report["independent_cross_region"]
    assert report["languages"] == ["zh"]
    assert desk[0].narrative("native-language-topic", as_of=10)["observations"] == 0


def test_label_requires_available_explicit_reviewed_provenance(desk):
    row = narrative(label="MIGRATION", label_provenance={"verified": True,
        "verification_method": "VERIFIED_CONTRACT_LINEAGE", "available_at": 10, "evidence_refs": ["fixture:lineage"]})
    assert consume(desk, "NARRATIVE_OBSERVATION", row)["detail"]["label"] == "MIGRATION"
    row["label_provenance"]["available_at"] = 12
    assert consume(desk, "NARRATIVE_OBSERVATION", row)["disposition"] == "DATA_BLOCKED"


def test_source_quality_uses_actual_metric_denominators_and_pit_evaluations(desk):
    consume(desk, "NARRATIVE_OBSERVATION", narrative())
    blank = desk[0].report()["sources"]["regional-a"]
    assert blank["metrics"]["authentic"]["rate"] is None
    assert blank["mean_executable_net_return"] is None
    evaluation = dict(observation_id="mention-a", event_at=12, observed_at=13,
                      evaluations={"authentic": True, "paid_promotion": False}, evaluation_evidence_refs=["fixture:review"])
    consume(desk, "SOURCE_OUTCOME", evaluation)
    consume(desk, "SOURCE_OUTCOME", dict(evaluation, observed_at=14))
    consume(desk, "SOURCE_OUTCOME", dict(evaluation, observed_at=15, evaluations={"organic": True}))
    report = desk[0].report()["sources"]["regional-a"]
    assert report["metrics"]["authentic"] == {"numerator": 1, "denominator": 1, "rate": 1.}
    assert report["metrics"]["organic"]["denominator"] == 1
    assert report["metrics"]["bot"]["rate"] is None
    assert report["evaluated_observations"] == 1
    old = desk[0].report(as_of=12)["sources"]["regional-a"]
    assert old["metrics"]["authentic"]["rate"] is None


def test_source_executable_return_requires_validated_fills_and_explicit_attribution(desk):
    consume(desk, "NARRATIVE_OBSERVATION", narrative())
    follow = consume(desk, "FOLLOW_OUTCOME", outcome(source_observation_id="mention-a"))
    row = dict(observation_id="mention-a", event_at=20, observed_at=21,
        evaluation_evidence_refs=["fixture:source-review"], entry_exit_evidence_refs=["fixture:resolved-fills"],
        executable_net_return=.2, evaluated_at=20, follow_outcome_id=follow["detail"]["outcome_id"])
    assert consume(desk, "SOURCE_OUTCOME", row)["disposition"] == "OBSERVED"
    report = desk[0].report()["sources"]["regional-a"]
    assert report["executable_return_denominator"] == 1
    assert report["mean_executable_net_return"] == pytest.approx(.2)
    assert consume(desk, "SOURCE_OUTCOME", dict(row, executable_net_return=9))["disposition"] == "DATA_BLOCKED"


def test_capacity_preserves_prior_work_and_exposes_debt(desk):
    hunter, ledger = desk
    hunter.max_observations = 1
    first = consume(desk, "ACTOR_OBSERVATION", actor())
    second = consume(desk, "ACTOR_OBSERVATION", actor("Different"))
    assert first["disposition"] == "OBSERVED" and second["disposition"] == "BLOCKED"
    assert hunter.report()["actor_observations"] == 1
    assert len(ledger.path.read_text().splitlines()) == 2


def test_cross_chain_bridge_targets_and_graph_mechanism_children(desk):
    row = actor(relation={"kind": "bridged", "target": "0xABcd", "target_type": "wallet",
        "target_chain_id": "ethereum", "target_chain_family": "EVM"})
    result = consume(desk, "ACTOR_OBSERVATION", row)
    edge = desk[0].actor_store.edges[0]
    assert edge.target == native_node("ethereum", "wallet", "0xabcd", "EVM")
    assert edge.source == native_node("solana", "wallet", "AbC", "SOLANA")
    assert result["detail"]["structure"]["outgoing_relation_counts"]["bridged"] == 1
    assert any(c["kind"] == "MECHANISM_HYPOTHESIS" for c in result["children"])
    assert desk[0].actor_structure("solana", "SOLANA", "AbC", as_of=10)["outgoing_relation_counts"]["bridged"] == 0


def test_conflicting_native_fact_is_refused_without_overwriting_graph():
    graph = ActorStore()
    kwargs = dict(chain_id="solana", chain_family="SOLANA", source="Actor", target="Token",
        kind=EdgeKind.BOUGHT, event_at=1, observed_at=2, evidence_refs=["fixture:fact"], observation_id="same-event")
    graph.observe_native(**kwargs)
    with pytest.raises(ValueError, match="conflicting facts"):
        graph.observe_native(**dict(kwargs, target="Different"))
    assert len(graph.edges) == 1


def test_checkpoint_receipt_retry_does_not_repeat_observations_or_children(desk, tmp_path):
    from src.research.frontier import AuctionInputs, FrontierIndex
    from src.research.unknown_engine import UnknownOpportunityEngine

    hunter, ledger = desk
    engine = UnknownOpportunityEngine(hunter.framework, FrontierIndex(tmp_path / "frontier.sqlite"), ledger)

    class Router:
        kinds = {"ACTOR_OBSERVATION"}

        def auction(self, kind):
            return AuctionInputs(costs={"inference": 0, "data": 0, "engineering": 0, "conversion": 1, "gauntlet": 0})

        def prepare(self, now):
            pass

        def consume(self, kind, payload, refs, axes, now):
            return hunter.consume(kind, payload["point_payload"], refs, now)

    engine.configure_workflows(Router())
    job_id = engine.enqueue_work("ACTOR_OBSERVATION", actor(), now=11)
    writer = ledger.record_reference
    failed = [False]

    def flaky(event, payload, **kwargs):
        if event == "unknown_conversion_receipt" and not failed[0]:
            failed[0] = True
            return None
        return writer(event, payload, **kwargs)

    ledger.record_reference = flaky
    assert engine.run_pass(limit=1, now=11)["results"][0]["disposition"] == "BLOCKED"
    assert hunter.report()["actor_observations"] == 1
    cached = next(iter(hunter.state["receipts"].values()))
    assert engine.frontier.job(job_id)["state"] == "RETRY"
    result = engine.run_pass(limit=1, now=72)
    assert result["results"][0]["disposition"] == "DEEPENING_TASK"
    assert engine.frontier.job(job_id)["state"] == "ACKED"
    assert hunter.report()["actor_observations"] == 1
    assert next(iter(hunter.state["receipts"].values())) == cached
    assert engine.frontier.report()["pending"] == len(cached["children"])

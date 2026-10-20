"""Adversarial shadow paths: no rare winner is required to remain solvent."""

import copy
from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from src.strategies.asymmetric_tail import AsymmetricTailSleeve, SLEEVE, TailSleevePolicy
from src.strategies.opportunity_allocator import Opportunity, OpportunityAllocator


NOW = datetime(2026, 1, 5, tzinfo=timezone.utc).timestamp()  # Monday


def ticket(token="a", *, capital=100.0, chain="solana", actor=None, narrative=None,
           repricer=True, **metadata):
    return Opportunity(token, 0.1, capital, 3600.0, 100_000.0,
        metadata={"chain": chain, "actor_id": actor or token,
                  "narrative_id": narrative or token, "safety_clear": True,
                  "exit_ready": True, **metadata},
        elogw_at=(lambda usd: usd * 0.001) if repricer else None)


def enter(sleeve, item=None, *, now=NOW):
    item = item or ticket()
    proposal = sleeve.propose([item], now=now)
    assert len(proposal.proposed) == 1, proposal.report()
    assert sleeve.record_shadow_entry(proposal.proposed[0], now=now)
    return proposal.proposed[0]


def test_no_monster_for_a_year_cannot_spend_more_than_the_declared_budget(tmp_path):
    """Every fill loses 100%, including fees. Exhaustion stops the sleeve."""
    sleeve = AsymmetricTailSleeve(10_000.0)
    admitted = []
    for day in range(365):
        at = NOW + day * 86400
        proposals = sleeve.propose([ticket(f"d{day}t{i}") for i in range(30)], now=at)
        for candidate in proposals.proposed:
            assert sleeve.record_shadow_entry(candidate, now=at)
            admitted.append(candidate.capital_usd)
            sleeve.record_shadow_exit(candidate.token, fraction=1.0, proceeds_usd=0.0, now=at)
        report = sleeve.report(now=at)
        assert report["daily_loss_usd"] <= 50.0 + 1e-8
        assert report["weekly_loss_usd"] <= 150.0 + 1e-8
        assert report["gross_loss_usd"] + report["open_risk_usd"] <= 300.0 + 1e-8
    report = sleeve.report(now=NOW + 365 * 86400)
    assert sum(admitted) == pytest.approx(300.0)
    assert report["status"] == "SHUTDOWN"
    assert report["max_drawdown"] == pytest.approx(0.03)
    assert report["time_between_monsters_seconds"] is None
    assert report["probability_of_ruin"] is None  # a single path is not a probability
    assert report["live_authority"] is False
    # Runtime artifact is explicitly a deterministic fixture stress, not a live claim.
    import json
    artifact = {"evidence_type": "deterministic_no_monster_shadow_stress",
                "days": 365, "terminal_portfolio_equity_usd": 9700.0, "report": report}
    path = tmp_path / "no-monster-stress.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert json.loads(path.read_text())["report"]["gross_loss_usd"] == pytest.approx(300)


def test_one_canonical_allocator_ranks_bounded_tickets_with_existing_positions():
    allocator = OpportunityAllocator()
    original_rank = allocator.rank
    allocator.rank = Mock(side_effect=original_rank)
    sleeve = AsymmetricTailSleeve(10_000, allocator=allocator)
    incumbent = replace(ticket("held"), is_open_position=True)
    proposal = sleeve.propose([ticket("new")], now=NOW, incumbents=[incumbent])
    allocator.rank.assert_called_once()
    assert allocator.rank.call_args.args[0][0] is incumbent
    assert proposal.proposed[0].capital_usd == 10
    assert proposal.proposed[0].sleeve == SLEEVE
    assert sleeve.positions == {}  # proposing cannot reserve or execute


def test_smaller_ticket_is_repriced_instead_of_scaling_an_old_elogw():
    repricer = Mock(return_value=0.024)
    item = replace(ticket(), elogw=123.0, elogw_at=repricer)
    proposal = AsymmetricTailSleeve(10_000).propose([item], now=NOW)
    assert proposal.proposed[0].capital_usd == 10.0
    assert proposal.proposed[0].elogw == 0.024
    repricer.assert_called_once_with(10.0)
    assert item.capital_usd == 100.0


def test_a_size_without_a_repricing_model_is_blocked():
    result = AsymmetricTailSleeve(10_000).propose([ticket(repricer=False)], now=NOW)
    assert result.proposed == []
    assert result.blocked["a"] == "DATA_BLOCKED_size_repricing"


def test_a_failed_repricing_provider_cannot_authorise_an_unpriced_ticket():
    item = replace(ticket(), elogw_at=Mock(side_effect=RuntimeError("provider unavailable")))
    result = AsymmetricTailSleeve(10_000).propose([item], now=NOW)
    assert result.proposed == []
    assert result.blocked["a"] == "DATA_BLOCKED_size_repricing"


@pytest.mark.parametrize("key,value", [("actor_id", ""), ("narrative_id", ""),
    ("chain", ""), ("safety_clear", None), ("exit_ready", False)])
def test_missing_independence_safety_or_exit_inputs_never_become_clear(key, value):
    result = AsymmetricTailSleeve(10_000).propose([ticket(**{key: value})], now=NOW)
    assert result.proposed == []
    assert result.blocked["a"].startswith("DATA_BLOCKED")


@pytest.mark.parametrize("dimension", ["actor", "narrative", "chain"])
def test_linked_positions_share_one_concentration_budget(dimension):
    policy = TailSleevePolicy(total_budget_fraction=.1, per_token_fraction=.01,
        daily_loss_fraction=.1, weekly_loss_fraction=.1,
        actor_fraction=.015, narrative_fraction=.015, chain_fraction=.015)
    sleeve = AsymmetricTailSleeve(10_000, policy=policy)
    items = []
    for i in range(10):
        kwargs = {"chain": f"chain-{i}", "actor": f"actor-{i}", "narrative": f"story-{i}"}
        kwargs[dimension] = "linked"
        items.append(ticket(str(i), **kwargs))
    proposal = sleeve.propose(items, now=NOW)
    assert sum(item.capital_usd for item in proposal.proposed) == pytest.approx(150)
    for item in proposal.proposed:
        assert sleeve.record_shadow_entry(item, now=NOW)


def test_stale_batch_is_rechecked_when_the_shadow_consumer_records_an_entry():
    sleeve = AsymmetricTailSleeve(10_000)
    first = sleeve.propose([ticket(str(i)) for i in range(5)], now=NOW)
    stale = sleeve.propose([ticket("stale")], now=NOW)
    for item in first.proposed:
        assert sleeve.record_shadow_entry(item, now=NOW)
    assert sleeve.record_shadow_entry(stale.proposed[0], now=NOW) is False


def test_crossing_a_day_or_week_never_forgets_open_loss_at_risk():
    sleeve = AsymmetricTailSleeve(10_000)
    for item in sleeve.propose([ticket(str(i)) for i in range(5)], now=NOW).proposed:
        assert sleeve.record_shadow_entry(item, now=NOW)
    assert sleeve.propose([ticket("next-day")], now=NOW + 86400).proposed == []
    assert sleeve.propose([ticket("next-week")], now=NOW + 7 * 86400).proposed == []


def test_restart_preserves_losses_open_positions_and_shutdown_without_renewing_budget():
    first = AsymmetricTailSleeve(10_000)
    enter(first, ticket("loser"))
    first.record_shadow_exit("loser", fraction=1, proceeds_usd=0, now=NOW + 1)
    enter(first, ticket("runner"), now=NOW + 2)
    first.shutdown("operator_research_pause")
    second = AsymmetricTailSleeve(10_000)
    second.restore(first.state())
    assert second.report(now=NOW + 3) == first.report(now=NOW + 3)
    assert second.propose([ticket("new")], now=NOW + 3).proposed == []
    changed = AsymmetricTailSleeve(10_000, policy=replace(first.policy, total_budget_fraction=.2))
    with pytest.raises(ValueError, match="mismatch"):
        changed.restore(first.state())


def test_corrupt_position_checkpoint_fails_before_changing_existing_state():
    sleeve = AsymmetricTailSleeve(10_000)
    enter(sleeve)
    state = sleeve.state()
    corrupt = copy.deepcopy(state)
    corrupt["positions"][0]["remaining_cost_usd"] = -1
    with pytest.raises(ValueError, match="invalid shadow position"):
        sleeve.restore(corrupt)
    assert sleeve.state() == state


def test_checkpoint_consumers_cannot_mutate_the_running_loss_budget():
    sleeve = AsymmetricTailSleeve(10_000)
    enter(sleeve)
    sleeve.record_shadow_exit("a", fraction=1, proceeds_usd=0, now=NOW)
    state = sleeve.state()
    state["losses"][0]["loss_usd"] = 0
    assert sleeve.gross_loss_usd == 10
    restored = AsymmetricTailSleeve(10_000)
    restored.restore(sleeve.state())
    checkpoint = sleeve.state()
    restored.restore(checkpoint)
    checkpoint["losses"][0]["loss_usd"] = 0
    assert restored.gross_loss_usd == 10


def test_partial_harvest_retains_runner_but_never_overrides_shutdown_or_safety():
    sleeve = AsymmetricTailSleeve(10_000)
    enter(sleeve)
    plan = sleeve.exit_plan("a", 3.0, now=NOW + 1)
    assert plan["action"] == "HARVEST" and plan["exit_fraction"] == .5
    sleeve.record_shadow_exit("a", fraction=.5, proceeds_usd=15.0, now=NOW + 1)
    assert sleeve.positions["a"].remaining_fraction >= sleeve.policy.runner_retention_fraction
    assert sleeve.exit_plan("a", 100.0, now=NOW + 2)["action"] == "HOLD"
    assert sleeve.exit_plan("a", 100.0, now=NOW + 2, known_safety_veto=True)["exit_fraction"] == 1
    assert sleeve.exit_plan("a", None, now=NOW + 2)["status"] == "DATA_BLOCKED"
    sleeve.shutdown("chain_outage")
    assert sleeve.exit_plan("a", None, now=NOW + 2)["exit_fraction"] == 1


def test_runner_cannot_hold_past_its_declared_time_budget():
    sleeve = AsymmetricTailSleeve(10_000)
    enter(sleeve)
    result = sleeve.exit_plan("a", 100, now=NOW + sleeve.policy.max_hold_seconds)
    assert result["action"] == "EXIT"
    assert result["reason"] == "holding_time_budget_exhausted"


def test_unmeasured_tail_performance_stays_unknown():
    report = AsymmetricTailSleeve(10_000).report(now=NOW)
    for name in ("realized_pnl_usd", "net_log_growth", "max_drawdown", "probability_of_ruin",
                 "time_between_monsters_seconds", "exit_reliability", "bet_return_correlation"):
        assert report[name] is None
    assert all(value is None for value in report["top_trade_return_share"].values())


def test_cash_beats_a_nonpositive_tail_candidate_and_time_cannot_be_rewound():
    sleeve = AsymmetricTailSleeve(10_000)
    assert sleeve.propose([replace(ticket(), elogw=-.1)], now=NOW).proposed == []
    enter(sleeve)
    with pytest.raises(ValueError, match="backwards"):
        sleeve.propose([ticket("old")], now=NOW - 1)

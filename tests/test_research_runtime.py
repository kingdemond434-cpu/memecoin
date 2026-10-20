from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.research.trade_evidence import TradeEvidenceLedger
from src.runtime.research_runtime import ResearchRuntime
from src.strategies.champion_challenger import ChampionChallengerFramework
from src.strategies.opportunity_allocator import Opportunity, OpportunityAllocator


class FixtureDesk(ResearchRuntime):
    def __init__(self, directory, *, equity=10_000, equity_status="OK", template=True):
        self.global_config = {}
        self.equity_status = equity_status
        self.wallet_equity_usd = equity
        self.champion_challenger = ChampionChallengerFramework(state_path=str(directory / "science.json"))
        self.opportunity_allocator = OpportunityAllocator()
        self.trade_evidence = TradeEvidenceLedger(directory / "evidence.jsonl")
        self.exit_readiness = SimpleNamespace(template_for=Mock(return_value=object() if template else None))
        self._ensure_asymmetric_tail()


def candidate(**metadata):
    return SimpleNamespace(address="token", chain="solana", deployer="public-deployer",
                           metadata={"narrative_id": "fixture-story", **metadata})


def opportunity():
    return Opportunity("token", .1, 100, 60, 100_000, elogw_at=lambda usd: usd * .001)


def clear_veto(**changes):
    return {**{"status": "CLEAR", "clear": True, "unmeasured": [], "reasons": []}, **changes}


def test_tail_budget_waits_for_measured_equity(tmp_path):
    desk = FixtureDesk(tmp_path, equity_status="DATA_BLOCKED")
    assert getattr(desk, "asymmetric_tail", None) is None
    assert desk._tail_shadow_candidate(candidate(), opportunity(), clear_veto())["status"] == "DATA_BLOCKED"


@pytest.mark.parametrize("template,veto", [
    (False, clear_veto()),
    (True, clear_veto(clear=False, status="VETO", reasons=["freeze_authority_active"])),
    (True, clear_veto(unmeasured=["sell_route"])),
])
def test_tail_proposes_only_with_known_exit_and_complete_safety(tmp_path, template, veto):
    desk = FixtureDesk(tmp_path, template=template)
    result = desk._tail_shadow_candidate(candidate(), opportunity(), veto)
    assert result["proposed"] == []
    assert desk._tail_pending == {} and desk.asymmetric_tail.positions == {}
    assert TradeEvidenceLedger.verify(desk.trade_evidence.path)[0]


def test_tail_proposal_has_no_reservation_until_successful_simulated_fill(tmp_path):
    desk = FixtureDesk(tmp_path)
    proposed = desk._tail_shadow_candidate(candidate(), opportunity(), clear_veto())
    assert proposed["proposed"] == ["token"]
    assert desk.asymmetric_tail.positions == {}
    assert "token" in desk._tail_pending
    desk._tail_shadow_fill("token", success=True, simulated=True)
    assert desk.asymmetric_tail.positions["token"].initial_cost_usd == 10
    saved = desk.champion_challenger.get_research_component("ASYMMETRIC_TAIL")
    assert saved["positions"][0]["token"] == "token"


@pytest.mark.parametrize("success,simulated", [(False, True), (True, False), (False, False)])
def test_failed_or_real_fill_never_enters_the_shadow_tail_ledger(tmp_path, success, simulated):
    desk = FixtureDesk(tmp_path)
    desk._tail_shadow_candidate(candidate(), opportunity(), clear_veto())
    desk._tail_shadow_fill("token", success=success, simulated=simulated)
    assert desk.asymmetric_tail.positions == {}
    assert desk.champion_challenger.get_research_component("ASYMMETRIC_TAIL")["positions"] == []


def test_canonical_checkpoint_restores_original_equity_and_spent_tail_loss_budget(tmp_path):
    first = FixtureDesk(tmp_path)
    first._tail_shadow_candidate(candidate(), opportunity(), clear_veto())
    first._tail_shadow_fill("token", success=True, simulated=True)
    first._tail_shadow_exit("token", fraction=1, proceeds_usd=0, allocated_cost_usd=10)
    assert first.asymmetric_tail.gross_loss_usd == 10
    # A changed main wallet balance must not mint a fresh research envelope.
    restarted = FixtureDesk(tmp_path, equity=50_000)
    assert restarted.asymmetric_tail.equity_usd == 10_000
    assert restarted.asymmetric_tail.budget_usd == 300
    assert restarted.asymmetric_tail.gross_loss_usd == 10
    assert restarted.asymmetric_tail.report(now=__import__("time").time())["remaining_lifetime_risk_usd"] == 290
    assert TradeEvidenceLedger.verify(restarted.trade_evidence.path)[0]


def test_tail_exit_attributes_only_its_share_of_a_canonical_simulated_fill(tmp_path):
    desk = FixtureDesk(tmp_path)
    desk._tail_shadow_candidate(candidate(), opportunity(), clear_veto())
    desk._tail_shadow_fill("token", success=True, simulated=True)
    # $10 sleeve ticket is a subset of a $100 canonical position.
    desk._tail_shadow_exit("token", fraction=.5, proceeds_usd=150, allocated_cost_usd=50)
    assert desk.asymmetric_tail.positions["token"].remaining_cost_usd == 5
    assert desk.asymmetric_tail.realized_pnl_usd == 10  # $15 proceeds minus $5 allocated cost


def test_stale_tail_proposal_cannot_write_an_accepted_fill_when_budget_is_spent(tmp_path):
    import json
    from dataclasses import replace
    desk = FixtureDesk(tmp_path)
    desk._tail_shadow_candidate(candidate(), opportunity(), clear_veto())
    # Five other simulated fills reserve the entire daily loss-at-risk budget
    # while the first candidate's execution result is still pending.
    for i in range(5):
        other = candidate()
        other.address = f"other-{i}"
        desk._tail_shadow_candidate(other, replace(opportunity(), token=other.address), clear_veto())
        desk._tail_shadow_fill(other.address, success=True, simulated=True)
    desk._tail_shadow_fill("token", success=True, simulated=True)
    assert "token" not in desk.asymmetric_tail.positions
    rows = [json.loads(line) for line in desk.trade_evidence.path.read_text().splitlines()]
    accepted = [row for row in rows if row["event_type"] == "asymmetric_tail_shadow_fill"]
    assert len(accepted) == 5
    assert all(row["payload"]["token"] != "token" for row in accepted)
    assert rows[-1]["event_type"] == "asymmetric_tail_shadow_fill_rejected"

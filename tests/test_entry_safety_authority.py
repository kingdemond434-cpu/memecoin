"""Exercise the real candidate decision, not only the veto's score function."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.main import MemecoinQuantDesk
from src.strategies.action_value import Action as ActionValue
from src.strategies.risk_veto import RiskVeto


def _risk(**changes):
    fields = dict(
        data_status="OK", blocked_checks=[], can_mint=False, can_freeze=False,
        token_extensions=[], sell_route_feasible=True,
        checks={"sell_route": {"price_impact_pct": 0.01}},
        risk_level=SimpleNamespace(value="safe"), score=100,
        connected_cluster_pct=10.0,
    )
    return SimpleNamespace(**{**fields, **changes})


def _desk(report, *, complete=True, sized_report=None):
    """All side effects are spies; the production candidate method still runs."""
    prediction = SimpleNamespace(p_50x=0.99, expected_hold_time=60)
    desk = SimpleNamespace(
        latency=SimpleNamespace(mark=Mock(), close=Mock()),
        elogw_engine=SimpleNamespace(
            open_positions={}, admissible=Mock(return_value=(True, {})),
            size_candidate=Mock(return_value={"position_size_sol": 0.1,
                "position_value_usd": 10.0, "elogw": 100.0}),
            portfolio_room=Mock(return_value=(True, {}))),
        rug_hazard=SimpleNamespace(get_hazard=Mock(return_value=None),
                                   register_token=Mock()),
        reentry_book=SimpleNamespace(admits=Mock(return_value=SimpleNamespace(
            admitted=True)), get=Mock(return_value=None)),
        dataset_builder=SimpleNamespace(start_episode=Mock(), record_risk_report=Mock()),
        _prelaunch_context=Mock(return_value=None), _risk_for_decision=Mock(return_value=report),
        public_coordination=SimpleNamespace(record_creator=Mock(), record_funding=Mock()),
        dev_wallet_monitor=SimpleNamespace(state=Mock(return_value={})),
        risk_veto=RiskVeto(require_complete_safety=complete),
        _record_blocked_decision=Mock(), _screen_entry=Mock(return_value=SimpleNamespace(rejected=False)),
        predictor=SimpleNamespace(_is_trained=True, predict=Mock(return_value=prediction)),
        _resolve_liquidity=AsyncMock(return_value=100_000.0),
        _build_prediction_features=AsyncMock(return_value={}),
        launch_census=SimpleNamespace(decision_ready=Mock()), dry_run=True,
        _ensure_portfolio_fresh=Mock(), _read_disagreement=Mock(return_value=None),
        sol_price_usd=100.0, wallet_equity_usd=10_000.0,
        _measured_depth_usd=Mock(return_value=100.0),
        _apply_screen_size=lambda info, screen: info,
        entry_coverage=SimpleNamespace(record=Mock(return_value=SimpleNamespace(to_dict=lambda: {}))),
        _score_entry=Mock(return_value=SimpleNamespace(
            status="OK", action=ActionValue.IGNORE, q=0.0, detail="cash wins")),
        contribution_ledger=SimpleNamespace(record_gate=Mock()),
        counterfactual_lab=SimpleNamespace(record_decision=Mock(return_value="decision-1")),
        _record_corpus_decision=Mock(), _record_trade_evidence_packet=Mock(),
        execution_engine=SimpleNamespace(execute_swap=AsyncMock()),
    )
    def intelligence(*args):
        verdict = desk.risk_veto.evaluate(
            sized_report or report, connected_holder_pct=10.0)
        return {"risk_veto": verdict.to_dict(), "actors": {}, "sources": {}}
    desk._entry_intelligence = intelligence
    return desk


async def _evaluate(desk):
    candidate = SimpleNamespace(chain="solana", address="token", deployer="dev",
        factory="pump", pair="curve", base_token="wsol", timestamp=100.0, metadata={})
    await MemecoinQuantDesk._evaluate_candidate(desk, candidate)


@pytest.mark.asyncio
@pytest.mark.parametrize("fact,reason", [
    ({"can_mint": True}, "mint_authority_active"),
    ({"can_freeze": True}, "freeze_authority_active"),
    ({"sell_route_feasible": False}, "sell_route_unavailable"),
    ({"token_extensions": ["permanent_delegate"]}, "dangerous_token_extension"),
])
async def test_known_fact_stops_a_high_alpha_candidate_before_prediction(fact, reason):
    desk = _desk(_risk(**fact))
    await _evaluate(desk)
    assert reason in desk._record_blocked_decision.call_args.args[1]
    desk.predictor.predict.assert_not_called()
    desk.elogw_engine.size_candidate.assert_not_called()
    desk.execution_engine.execute_swap.assert_not_awaited()


@pytest.mark.asyncio
async def test_required_unknown_safety_is_blocked_without_claiming_a_known_rug():
    desk = _desk(_risk(sell_route_feasible=None))
    await _evaluate(desk)
    assert desk._record_blocked_decision.call_args.args[1].startswith("DATA_BLOCKED_safety:")
    desk.predictor.predict.assert_not_called()
    desk.execution_engine.execute_swap.assert_not_awaited()


@pytest.mark.asyncio
async def test_new_veto_at_sizing_cannot_be_overridden_by_alpha():
    desk = _desk(_risk(), sized_report=_risk(can_freeze=True))
    await _evaluate(desk)
    desk.predictor.predict.assert_called_once()
    assert "freeze_authority_active" in desk._record_blocked_decision.call_args.args[1]
    desk._score_entry.assert_not_called()
    desk.execution_engine.execute_swap.assert_not_awaited()


@pytest.mark.asyncio
async def test_clear_candidate_records_the_actual_veto_snapshot_when_cash_wins():
    desk = _desk(_risk())
    await _evaluate(desk)
    desk._record_trade_evidence_packet.assert_called_once()
    packet = desk._record_trade_evidence_packet.call_args.kwargs
    assert packet["veto"]["status"] == "CLEAR"
    assert packet["decision"] == "action_value_ignore"
    desk.execution_engine.execute_swap.assert_not_awaited()

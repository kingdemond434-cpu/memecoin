import gzip
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.research.execution_measurement import (
    WSOL, analyze_fill, episode_measurements, normalize_attempt, reserve_reference,
)


def fill(side='buy', **changes):
    return {'fill_id': 'sig', 'source_ref': 'solana:sig#slot=4', 'market': 'token',
        'quote_mint': WSOL, 'unit': 'raw_quote_per_raw_base', 'side': side,
        'quantity': '10', 'fill_price': '102' if side == 'buy' else '98',
        'reference_at': 100., 'available_at': 100., 'decision_at': 98., 'arrival_at': 99.,
        'clock_basis': 'TEST_EXPLICIT_CLOCK', 'real_fill': True,
        'external_costs_complete': True, 'external_cost_quote_raw': '2', **changes}


def quote(at, price, **changes):
    return {**reserve_reference('token', price, 1, observed_at=at, source_ref='quote:'+str(at)), **changes}


def test_buy_tca_and_markout_close_arithmetically_after_explicit_costs():
    tape = [quote(98, 98), quote(99, 99), quote(100, 100), quote(105, 110)]
    r = analyze_fill(fill(), tape, as_of=105)
    assert r['decision_shortfall_quote_gross'] == '40'
    assert r['arrival_shortfall_quote_gross'] == '30'
    assert r['effective_reference_deviation_quote'] == '20'
    assert r['decision_shortfall_quote_after_external_cost'] == '42'
    assert r['horizons']['5']['reference_move_quote'] == '100'
    assert r['horizons']['5']['gross_markout_quote'] == '80'
    assert r['horizons']['5']['net_markout_quote'] == '78'
    assert r['horizons']['5']['realized_reference_spread_quote'] == '-80'
    assert r['horizons']['30']['status'] == 'NOT_MATURE'
    assert r['horizons']['30']['net_markout_quote'] is None


def test_sell_direction_and_negative_adverse_selection_are_signed_correctly():
    r = analyze_fill(fill('sell'), [quote(100, 100), quote(105, 110)], as_of=105)
    assert r['effective_reference_deviation_quote'] == '20'
    assert r['horizons']['5']['reference_move_quote'] == '-100'
    assert r['horizons']['5']['net_markout_quote'] == '-122'


@pytest.mark.parametrize('bad', [quote(105.1, 999), quote(100, 999, available_at=106),
    quote(105, 999, quote_mint='USDC'), quote(105, 999, unit='USD_per_token'),
    quote(105, 999, market='another-token'), quote(105, 999, source_ref='')])
def test_future_unavailable_foreign_or_unproven_quote_is_not_used(bad):
    r = analyze_fill(fill(), [quote(100, 100), bad], as_of=105)
    assert r['horizons']['5']['status'] == 'DATA_BLOCKED_REFERENCE_TAPE'
    assert r['horizons']['5']['net_markout_quote'] is None


def test_stale_or_changed_reference_kind_cannot_look_like_zero_cost():
    r = analyze_fill(fill(), [quote(97, 100), quote(105, 110)], as_of=105)
    assert r['effective_reference_deviation_quote'] is None
    r = analyze_fill(fill(), [quote(100, 100), quote(105, 110, kind='MID_QUOTE')], as_of=105)
    assert r['horizons']['5']['status'] == 'DATA_BLOCKED_REFERENCE_KIND_CHANGE'


def test_incomplete_fees_preserve_gross_measurement_and_refuse_net_claim():
    r = analyze_fill(fill(external_costs_complete=False), [quote(100, 100), quote(105, 110)], as_of=105)
    assert r['external_cost_quote_raw'] is None
    assert r['horizons']['5']['gross_markout_quote'] == '80'
    assert r['horizons']['5']['net_markout_quote'] is None
    assert r['horizons']['5']['status'] == 'GROSS_ONLY_COSTS_INCOMPLETE'


def real_attempt():
    return {'simulated': False, 'success': True, 'landed': True, 'filled': True, 'signature': 'sig',
        'measurement_context': {'amount_proof': 'confirmed_owner_balance_deltas', 'signature': 'sig',
            'input_mint': WSOL, 'output_mint': 'token', 'input_amount_raw': 1020, 'output_amount_raw': 10,
            'confirmation_observed_at': 100., 'arrival_at': 99., 'transaction_block_time': 98}}


def test_native_adapter_keeps_receipt_clock_units_and_missing_costs_explicit():
    attempt = real_attempt()
    row = normalize_attempt('token', attempt)
    assert row['fill_price'] == '102' and row['quantity'] == '10'
    assert row['clock_basis'] == 'CONFIRMATION_RECEIPT; NOT_EXECUTION_TIME'
    assert row['external_costs_complete'] is False
    attempt['measurement_context']['input_amount_raw'] = 0
    attempt['input_amount'] = 1000  # planned amount must never repair the missing fill
    assert normalize_attempt('token', attempt)['status'] == 'DATA_BLOCKED_FILL_AMOUNTS_OR_CLOCK'


def test_simulation_unfilled_and_identity_unknown_are_separate_from_actual_fills():
    attempt = real_attempt()
    attempt['simulated'] = True
    assert normalize_attempt('token', attempt)['real_fill'] is False
    del attempt['simulated']
    assert normalize_attempt('token', attempt)['status'] == 'DATA_BLOCKED_SIMULATION_IDENTITY'
    attempt.update(simulated=False, filled=False)
    assert normalize_attempt('token', attempt)['status'] == 'NOT_A_VERIFIED_FILL'


def test_episode_serialization_produces_measurement_and_audit_is_read_only(tmp_path):
    from src.research.dataset_builder import LaunchEpisode, PointInTimeDatasetBuilder
    from tools.execution_measurements import audit
    episode = LaunchEpisode('token', 'solana', 90., '', '', '', WSOL)
    episode.execution_attempts = [real_attempt()]
    episode.market_observations = [{'execution_reference': quote(100, 100)}, {'execution_reference': quote(105, 110)}]
    data = PointInTimeDatasetBuilder._episode_to_dict(episode)
    assert data['execution_measurement']['attempts'] == 1
    corpus = tmp_path/'corpus'
    corpus.mkdir()
    path = corpus/'token.json.gz'
    with gzip.open(path, 'wt') as handle:
        json.dump(data, handle)
    before = path.read_bytes()
    result = audit(corpus, tmp_path/'audit.jsonl', as_of=105.)
    assert result['counts']['verified_real_fill_attempts'] == 1
    assert path.read_bytes() == before
    with pytest.raises(FileExistsError):
        audit(corpus, tmp_path/'audit.jsonl', as_of=105.)


@pytest.mark.asyncio
async def test_fill_reconciliation_retains_actual_native_receipt_without_submitting():
    from src.execution.jupiter_jito import ExecutionEngine
    tx = {'slot': 4, 'blockTime': 98, 'transaction': {'message': {'accountKeys': ['owner']}},
          'meta': {'err': None, 'fee': 5000, 'preBalances': [9000], 'postBalances': [4000],
              'preTokenBalances': [{'mint': WSOL, 'owner': 'owner', 'uiTokenAmount': {'amount': '1020'}}],
              'postTokenBalances': [{'mint': 'token', 'owner': 'owner', 'uiTokenAmount': {'amount': '10'}}]}}
    desk = SimpleNamespace(confirmation_timeout=1, _await_landing=AsyncMock(return_value=True),
        rpc=SimpleNamespace(request=AsyncMock(return_value=tx)), tx_builder=SimpleNamespace(public_key='owner'),
        _token_balance_delta=ExecutionEngine._token_balance_delta,
        _token_balance_decrease=ExecutionEngine._token_balance_decrease,
        _native_balance_delta=ExecutionEngine._native_balance_delta)
    r = await ExecutionEngine._wait_for_fill(desk, 'sig', WSOL, 'token')
    ctx = r['measurement_context']
    assert ctx['input_amount_raw'] == 1020 and ctx['output_amount_raw'] == 10
    assert ctx['network_fee_lamports'] == 5000 and ctx['external_costs_complete'] is False
    assert ctx['balance_evidence']['account_keys'] == ['owner']
    assert desk.rpc.request.call_args.args[1][1]['maxSupportedTransactionVersion'] == 1

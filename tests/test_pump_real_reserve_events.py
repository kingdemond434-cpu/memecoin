"""Official-IDL event fields must reach depth checks without inventing data."""
import asyncio
import struct
from types import SimpleNamespace

import pytest
from src.chains.idl import load_idl
from src.chains.yellowstone_grpc import PumpFunMonitor,b58encode
from src.main import MemecoinQuantDesk


def payload(*, quote=None):
    values={'mint':bytes([7])*32,'user':bytes([8])*32,'creator':bytes([9])*32,
        'sol_amount':1000,'token_amount':2000,'is_buy':True,'timestamp':1700000000,
        'virtual_sol_reserves':31_000_000_000,'virtual_token_reserves':1_000_000_000_000,
        'real_sol_reserves':1_000_000_000,'real_token_reserves':700_000_000_000,
        'quote_mint':quote or bytes(32),'ix_name':'buy','shareholders':[(bytes([3])*32,10000)]}
    fields=next(row['type']['fields'] for row in load_idl('pump')['types'] if row['name']=='TradeEvent')
    out=bytes(PumpFunMonitor.TRADE_EVENT)
    for row in fields:
        name,kind=row['name'],row['type'];value=values.get(name)
        if kind=='pubkey':out+=value or bytes(32)
        elif kind in ('u64','i64'):out+=struct.pack('<Q' if kind=='u64' else '<q',value or 0)
        elif kind=='bool':out+=bytes([bool(value)])
        elif kind=='string':
            text=(value or '').encode();out+=struct.pack('<I',len(text))+text
        elif isinstance(kind,dict) and 'vec' in kind:
            out+=struct.pack('<I',len(value or []))
            for address,bps in value or []:out+=address+struct.pack('<H',bps)
        else:raise AssertionError('unreviewed IDL event fixture type')
    return out


def decoded(raw):
    return PumpFunMonitor.__new__(PumpFunMonitor)._decode_program_event(raw,'synthetic-idl-sig',1)


def test_official_idl_prefix_includes_real_reserves_creator_and_foreign_quote():
    event=decoded(payload())
    assert event['real_sol_reserves']==1_000_000_000
    assert event['real_token_reserves']==700_000_000_000
    assert event['creator']==b58encode(bytes([9])*32)
    assert event['real_reserve_data_status']=='OBSERVED_OFFICIAL_PUMP_TRADE'
    assert event['quote_mint']=='So11111111111111111111111111111111111111112'
    foreign=bytes([4])*32
    assert decoded(payload(quote=foreign))['quote_mint']==b58encode(foreign)


@pytest.mark.parametrize('length',[97,113,121,128])
def test_absent_or_partial_real_pair_remains_unknown(length):
    event=decoded(payload()[:length])
    assert event['real_sol_reserves'] is None and event['real_token_reserves'] is None
    assert event['real_reserve_data_status']=='UNMEASURED'


@pytest.mark.parametrize('mode',['observed','legacy','untagged','impossible','foreign'])
def test_decoded_reserves_reach_runtime_without_promoting_unmeasured_depth(mode):
    event=decoded(payload(quote=bytes([4])*32 if mode=='foreign' else None))
    if mode=='legacy':event=decoded(payload()[:113])
    if mode=='untagged':event.pop('real_reserve_data_status')
    if mode=='impossible':event['real_sol_reserves']=event['virtual_sol_reserves']+1
    event['curve_price_raw']=None  # price/peak accounting is outside this check
    token=event['token'];observations=[];noop=lambda *a,**k:None
    app=SimpleNamespace(_stream_events={},_latest_curve_state={},_curve_static={token:{'token_total_supply':10**15}},
        state_sequencer=SimpleNamespace(bump=noop),observed_holders=SimpleNamespace(record_trade=noop,snapshot=lambda _:None),
        execution_engine=None,_record_actor_entry=noop,rug_hazard=SimpleNamespace(record_observation=noop),
        request_redecision=noop,public_coordination=SimpleNamespace(record_trade=noop),
        dataset_builder=SimpleNamespace(record_market_observation=lambda token,row:observations.append(row)),
        wallet_intel=SimpleNamespace(record_live_trade=noop),_open_follow_candidate=noop,_record_followable_trade=noop)
    asyncio.run(MemecoinQuantDesk._on_pump_event(app,event))
    if mode=='foreign':
        assert not app._latest_curve_state
        assert app._stream_events['token_trade:observation_only_quote']==1
        return
    state=app._latest_curve_state[token]
    assert state.real_sol_reserves==(1_000_000_000 if mode=='observed' else 0)
    assert state.real_token_reserves==(700_000_000_000 if mode=='observed' else 0)
    assert observations[0]['real_reserve_data_status']==event.get('real_reserve_data_status','UNMEASURED')


def test_current_creation_supply_is_decoded_and_legacy_supply_stays_unknown():
    prefix=bytes(PumpFunMonitor.CREATE_EVENT)+struct.pack('<III',0,0,0)+bytes(128)+struct.pack('<q',1700000000)
    assert decoded(prefix)['token_total_supply'] is None
    raw=prefix+struct.pack('<QQQQ',10**15,30*10**9,700*10**12,10**15)
    assert decoded(raw)['token_total_supply']==10**15


def test_native_decoder_retains_same_new_fields_when_new_extension_is_available():
    fast=pytest.importorskip('solana_fastpath')
    # A stale installed wheel must not be passed off as validation of new
    # native fields. Native CI and release validation build this exact source.
    event=fast.decode_pump_event(payload())
    if 'real_sol_reserves' not in event:pytest.skip('installed native extension predates real-reserve fields')
    expected=decoded(payload())
    for field in ('real_sol_reserves','real_token_reserves','real_reserve_data_status','creator','quote_mint'):
        assert event[field]==expected[field]

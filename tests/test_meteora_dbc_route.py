"""Pinned SDK interface and synthetic bank controls, not live venue certification."""
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
from solders.pubkey import Pubkey

from src.chains import meteora_dbc_route as r
from src.chains.idl import load_idl
from src.chains.meteora_dbc import MeteoraDBCMonitor
from src.chains.pump_route import TOKEN_PROGRAM,TOKEN_2022_PROGRAM,WSOL_MINT,associated_token_address
from src.research import fork_simulation as f
from src.research.fork_probe import ForkProbe,reviewed_native_target
from tests.test_fork_simulation import wire


def encoded_account(kind,values):
    """Synthetic schema fixture, with independent published-offset checks below."""
    idl=load_idl(r.IDL);types={t['name']:t['type'] for t in idl['types']}
    def encode(schema,value=None):
        if schema=='pubkey':return bytes(Pubkey.from_string(value)) if value is not None else bytes(32)
        if isinstance(schema,str):return int(value or 0).to_bytes(int(schema[1:])//8,'little')
        if 'array' in schema:return b''.join(encode(schema['array'][0],value[i] if value else None) for i in range(schema['array'][1]))
        fields=types[schema['defined']['name']]['fields']
        return b''.join(encode(field['type'],(value or {}).get(field['name'])) for field in fields)
    disc=next(a['discriminator'] for a in idl['accounts'] if a['name']==kind)
    return bytes(disc)+encode({'defined':{'name':kind}},values)


def fixture(token2022=False):
    mint,config,creator=[str(Pubkey.new_unique()) for _ in range(3)]
    pool=r.pool_address(mint,config)
    state={'config':config,'creator':creator,'base_mint':mint,'pool_type':int(token2022),
        'base_vault':r.vault_address(pool,mint),'quote_vault':r.vault_address(pool,WSOL_MINT),
        'sqrt_price':2**64,'activation_point':123456}
    raw=encoded_account('VirtualPool',{'pool_state':state})
    config_raw=encoded_account('PoolConfig',{'quote_mint':WSOL_MINT,'token_type':int(token2022),
        'token_decimal':6,'version':1,'sqrt_start_price':2**64})
    accounts={pool:wire(raw,r.DBC_PROGRAM),config:wire(config_raw,r.DBC_PROGRAM)}
    authority=r.pda(b'pool_authority')
    for side,decimals,program in (('base',6,TOKEN_2022_PROGRAM if token2022 else TOKEN_PROGRAM),('quote',9,TOKEN_PROGRAM)):
        token=mint if side=='base' else WSOL_MINT
        data=bytearray(82);data[44:46]=bytes([decimals,1])
        if program==TOKEN_2022_PROGRAM:data+=bytes(83)+b'\x01'
        accounts[token]=wire(data,program)
        vault=bytearray(165);vault[:32]=bytes(Pubkey.from_string(token));vault[32:64]=bytes(Pubkey.from_string(authority))
        vault[64:72]=(10**9).to_bytes(8,'little');vault[108]=1
        if program==TOKEN_2022_PROGRAM:vault+=b'\x02'
        accounts[state[side+'_vault']]=wire(vault,program)
    return r.parse_pool(raw,pool),accounts


def test_public_captured_initializer_confirms_sdk_pool_and_vault_derivations():
    tx=json.loads((Path(__file__).parent/'fixtures/meteora_dbc_launches.json').read_text())['transactions'][0]
    from src.chains.yellowstone_grpc import transaction_parts,instruction_fields
    monitor=MeteoraDBCMonitor(SimpleNamespace(on=lambda *a:None),lambda event:None)
    keys,instructions,_,_=transaction_parts(tx)
    ix=next(ix for ix in instructions if instruction_fields(ix)[2][:8] in monitor.schemas)
    _,indices,raw=instruction_fields(ix);event=monitor.decode(keys,indices,raw)
    assert r.pool_address(event['token'],event['config'],event['quote_mint'])==event['pool']
    assert r.vault_address(event['pool'],event['token'])==event['base_vault']
    assert r.vault_address(event['pool'],event['quote_mint'])==event['quote_vault']
    assert event['quote_mint']!=WSOL_MINT  # not misrepresented as a runnable WSOL fixture


@pytest.mark.parametrize('token2022',[False,True])
def test_fixed_layout_and_exact_input_output_identity(token2022):
    state,accounts=fixture(token2022)
    raw=f.account_data(accounts[state['pool']])
    assert len(raw)==424 and len(f.account_data(accounts[state['config']]))==1048
    assert str(Pubkey.from_bytes(raw[72:104]))==state['config']
    assert str(Pubkey.from_bytes(raw[136:168]))==state['base_mint']
    assert state['activation_point']==123456 and state['sqrt_price']==2**64
    assert state['authority']=='FhVo3mqL8PW5pH5U2CN4XE33DokiyZnUwuGpH2hmHLuM'
    r.validate_frame(state,accounts,f.account_data)
    target={'token':state['base_mint'],'pool':state['pool'],'venue':'meteora_dbc','quote_budget':12345}
    buy,sell,program=f._route(target,accounts)
    assert program==(TOKEN_2022_PROGRAM if token2022 else TOKEN_PROGRAM)
    assert buy.data==bytes([248,198,158,145,225,117,135,200])+struct.pack('<QQ',12345,1)
    assert sell(987).data[8:]==struct.pack('<QQ',987,1)
    owner=str(f.virtual_owner().pubkey())
    base=associated_token_address(owner,state['base_mint'],program);quote=associated_token_address(owner,WSOL_MINT,TOKEN_PROGRAM)
    assert [a.pubkey for a in buy.accounts][3:5]==[quote,base]
    assert [a.pubkey for a in sell(987).accounts][3:5]==[base,quote]
    assert [i for i,a in enumerate(buy.accounts) if a.is_signer]==[9]
    assert [i for i,a in enumerate(buy.accounts) if a.is_writable]==[2,3,4,5,6]
    assert buy.accounts[12].pubkey==r.DBC_PROGRAM and not buy.accounts[12].is_writable
    assert buy.accounts[-1].pubkey==r.INSTRUCTIONS_SYSVAR and buy.expected_accounts==16


@pytest.mark.parametrize('bad',['short','long','discriminator','hook_layout','pool_flag','migrated','config_owner','quote',
    'quote_program','type_mismatch','pool_identity','vault_identity','vault_owner','vault_mint','vault_authority','mint_decimals','frozen_vault'])
def test_incomplete_foreign_or_unreviewed_bank_cannot_become_a_route(bad):
    state,accounts=fixture();pool=state['pool'];config=state['config']
    if bad in ('short','long','discriminator','hook_layout','pool_flag'):
        raw=bytearray(f.account_data(accounts[pool]))
        if bad=='short':raw.pop()
        elif bad=='long':raw.append(0)
        elif bad=='hook_layout':raw[:8]=bytes(next(a['discriminator'] for a in load_idl(r.IDL)['accounts'] if a['name']=='TransferHookPool'))
        elif bad=='pool_flag':raw[304]=255
        else:raw[0]^=1
        with pytest.raises(ValueError):r.parse_pool(raw,pool)
        return
    if bad=='migrated':state['is_migrated']=1
    elif bad=='config_owner':accounts[config]['owner']=TOKEN_PROGRAM
    elif bad in ('quote','quote_program','type_mismatch'):
        values={'quote_mint':WSOL_MINT,'token_decimal':6}
        values.update({'quote_mint':str(Pubkey.new_unique())} if bad=='quote' else {'quote_token_flag':1} if bad=='quote_program' else {'token_type':1})
        accounts[config]=wire(encoded_account('PoolConfig',values),r.DBC_PROGRAM)
    elif bad=='pool_identity':state['pool']=str(Pubkey.new_unique())
    elif bad=='vault_identity':state['base_vault']=str(Pubkey.new_unique())
    elif bad=='vault_owner':accounts[state['base_vault']]['owner']=TOKEN_2022_PROGRAM
    else:
        key=state['base_mint'] if bad=='mint_decimals' else state['base_vault'];raw=bytearray(f.account_data(accounts[key]))
        if bad=='mint_decimals':raw[44]^=1
        elif bad=='frozen_vault':raw[108]=2
        else:raw[0 if bad=='vault_mint' else 32]^=1
        accounts[key]=wire(raw)
    with pytest.raises(ValueError):r.validate_frame(state,accounts,f.account_data)


@pytest.mark.parametrize('extension',[1,14])
def test_transfer_fees_are_executed_in_vm_but_hook_accounts_stay_unknown(extension):
    state,accounts=fixture(True);mint=state['base_mint']
    raw=f.account_data(accounts[mint])+struct.pack('<HH',extension,108 if extension==1 else 64)+bytes(108 if extension==1 else 64)
    accounts[mint]=wire(raw,TOKEN_2022_PROGRAM)
    if extension==14:
        with pytest.raises(ValueError,match='transfer-hook'):r.validate_frame(state,accounts,f.account_data)
    else:r.validate_frame(state,accounts,f.account_data)


def test_native_intake_and_episode_views_preserve_dbc_identity_and_source_binding(tmp_path):
    state,_=fixture();event={'type':'token_launch','venue':'meteora_dbc','program':r.DBC_PROGRAM,
        'token':state['base_mint'],'pool':state['pool'],'quote_mint':WSOL_MINT}
    target=reviewed_native_target(event)
    assert target['venue']=='meteora_dbc'
    probe=ForkProbe(tmp_path/'probe.sqlite');ref='native-launch:'+'e'*64
    assert probe.enqueue_native_launch(event,source_ref=ref,available_at=100)['inserted']
    assert probe.report()['canonical_native_linked_targets']==1 and 'meteora_dbc' in probe.report()['supported_adapters']
    assert reviewed_native_target({**event,'program':'11111111111111111111111111111111'}) is None
    assert reviewed_native_target({**event,'quote_mint':str(Pubkey.new_unique())}) is None
    from src.runtime.ingestion import MinedRecordIngestion
    episode=SimpleNamespace(factory=r.DBC_PROGRAM,base_token=WSOL_MINT,pair=state['pool'])
    desk=SimpleNamespace(dataset_builder=SimpleNamespace(active_episodes={state['base_mint']:episode}))
    assert MinedRecordIngestion._native_fork_targets(desk)==[target]
    probe.close()


@pytest.mark.asyncio
async def test_acquire_rereads_config_vaults_and_instruction_dependencies_in_final_frame():
    state,accounts=fixture();calls=[]
    async def request(method,params):
        assert method=='getMultipleAccounts'
        keys,options=params;calls.append(set(keys));assert options['commitment']=='finalized'
        return {'context':{'slot':100+len(calls)},'value':[accounts.get(k) for k in keys]}
    target={'token':state['base_mint'],'pool':state['pool'],'venue':'meteora_dbc','quote_budget':10000}
    frame=await f.acquire(SimpleNamespace(request=request),target)
    assert len(calls)==4 and frame['slot']==104
    assert {state['config'],state['base_vault'],state['quote_vault']}<=calls[-1]
    assert r.INSTRUCTIONS_SYSVAR in calls[-1] and r.DBC_PROGRAM in calls[-1]
    assert frame['target']==target and frame['accounts'][state['config']]==accounts[state['config']]


@pytest.mark.asyncio
async def test_four_venue_service_includes_dbc_under_large_pump_burst(monkeypatch,tmp_path):
    from src.research import fork_probe as p
    async def unavailable(rpc,target):raise f.ForkUnavailable('synthetic bank unavailable')
    monkeypatch.setattr(p,'acquire',unavailable)
    probe=ForkProbe(tmp_path/'probe.sqlite')
    def target(venue):return {'token':str(Pubkey.new_unique()),'venue':venue,'quote_budget':1_000_000}
    initial=[target(v) for v in ('meteora_dbc','raydium_launchlab','pumpswap')]+[target('pump_curve') for _ in range(1100)]
    served=[]
    for turn in range(4):
        rows=await probe.fetch(None,lambda:initial if turn==0 else [target('pump_curve') for _ in range(20)])
        served.append(rows[0]['dispatch_venue']);probe.acknowledge(rows,'canonical:'+str(turn))
    assert set(served)=={'meteora_dbc','raydium_launchlab','pumpswap','pump_curve'}
    assert probe.report()['targets']==1163
    assert probe.report()['by_venue']['meteora_dbc']['canonical_dispositions']=={'FORK_ENVIRONMENT_UNKNOWN':1}
    probe.close()


def test_certified_source_recovery_includes_dbc_without_promoting_other_quotes(tmp_path):
    from src.research.trade_evidence import TradeEvidenceLedger
    from tools.backfill_native_fork_targets import backfill
    from tools.verify_evidence_backup import verify
    state,_=fixture();event={'type':'token_launch','venue':'meteora_dbc','program':r.DBC_PROGRAM,
        'token':state['base_mint'],'pool':state['pool'],'quote_mint':WSOL_MINT}
    ledger=tmp_path/'source.jsonl';certificate=tmp_path/'certificate.json'
    source=TradeEvidenceLedger(ledger);source.record('native_launch',event,timestamp=50)
    source.record('native_launch',{**event,'quote_mint':str(Pubkey.new_unique())},timestamp=60)
    certificate.write_text(json.dumps(verify(ledger)))
    before=ledger.read_bytes();database=tmp_path/'probe.sqlite'
    report=backfill(ledger,certificate,database=database,apply=True)
    assert report['reviewed_source_rows']==report['new_targets']==1 and report['other_or_unsupported']==1
    assert report['source_unchanged'] and ledger.read_bytes()==before
    probe=ForkProbe(database)
    assert probe.report()['by_venue']['meteora_dbc']['targets']==1
    assert probe.report()['canonical_dispositions']=={}
    probe.close()


@pytest.mark.parametrize('amount',[True,0,-1,2**64,1.5])
def test_no_invalid_probe_amount_or_minimum(amount):
    state,_=fixture()
    with pytest.raises(ValueError):r.build_swap(state,str(Pubkey.new_unique()),amount,1,buy=True)
    with pytest.raises(ValueError):r.build_swap(state,str(Pubkey.new_unique()),1,amount,buy=False)

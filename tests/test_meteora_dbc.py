import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.chains.idl import load_idl
from src.chains.meteora_dbc import DBC_PROGRAM, CREATE_NAMES, MeteoraDBCMonitor
from src.chains.yellowstone_grpc import b58encode, transaction_parts, instruction_fields, SolanaRpcProgramStream, create_combined_subscription

FIXTURE = Path(__file__).parent/'fixtures/meteora_dbc_launches.json'

def captured():
    return json.loads(FIXTURE.read_text())['transactions'][0]

def monitor():
    rows=[]
    return MeteoraDBCMonitor(SimpleNamespace(on=lambda *args: None), rows.append), rows

@pytest.mark.asyncio
async def test_public_dbc_capture_keeps_arbitrary_quote_and_creator():
    m, rows=monitor()
    await m._on_transaction(captured())
    assert len(rows)==1 and not m.rejected
    event=rows[0]
    assert event['token']=='ED8zZbr9kvKZf6SARrw3woiN1oh5L4HaVhqatBSYtata'
    assert event['creator']=='2MwTdN4qUPUWLAJyHJzrwK9c9htX35aVdc3W4F5G3ejY'
    assert event['quote_mint']=='METvsvVRapdj9cFLzq4Tr43xK4tAjQfwX76z3n6mWQL'
    assert event['timestamp']==captured()['blockTime']
    assert event['token_decimals'] is None and event['quote_decimals'] is None
    assert event['execution_support']=='DATA_BLOCKED_NATIVE_CURVE_EXECUTION'

@pytest.mark.asyncio
@pytest.mark.parametrize('name',CREATE_NAMES)
async def test_all_published_initializers_and_cpi_preserve_sponsoring_payer(name):
    tx=captured()
    keys,instructions,_,_=transaction_parts(tx)
    original=next(ix for ix in instructions if instruction_fields(ix)[2][:8] in monitor()[0].schemas)
    _,accounts,raw=instruction_fields(original)
    source=monitor()[0].schemas[raw[:8]]
    named={row['name']:keys[accounts[i]] for i,row in enumerate(source['accounts'])}
    named['payer']=named['quote_mint']
    schema=next(row for row in load_idl('meteora_dbc')['instructions'] if row['name']==name)
    allkeys=list(keys)
    selected=[]
    for row in schema['accounts']:
        value=row.get('address') or named.get(row['name']) or named['creator']
        if value not in allkeys:allkeys.append(value)
        selected.append(allkeys.index(value))
    ix={'programIdIndex':allkeys.index(DBC_PROGRAM),'accounts':selected,
        'data':b58encode(bytes(schema['discriminator'])+raw[8:])}
    tx['transaction']['message']['accountKeys']=allkeys
    tx['transaction']['message']['instructions']=[]
    tx['meta']['innerInstructions']=[{'index':0,'instructions':[ix]}]
    m,rows=monitor()
    await m._on_transaction(tx)
    assert len(rows)==1 and rows[0]['creator']!=rows[0]['payer']
    assert rows[0]['instruction']==name
    assert bool(rows[0]['transfer_hook_program'])==('transfer_hook' in name)

@pytest.mark.asyncio
@pytest.mark.parametrize('failure',['failed','meta','program','quote_program','fixed_account','truncated','tail','index'])
async def test_invalid_or_foreign_packets_cannot_become_dbc_launches(failure):
    tx=captured()
    keys,instructions,_,_=transaction_parts(tx)
    m,rows=monitor()
    ix=next(ix for ix in tx['transaction']['message']['instructions'] if instruction_fields(ix)[2][:8] in m.schemas)
    _,accounts,raw=instruction_fields(ix)
    schema=m.schemas[raw[:8]]
    positions={row['name']:i for i,row in enumerate(schema['accounts'])}
    if failure=='failed':tx['meta']['err']={'InstructionError':[0,'Custom']}
    elif failure=='meta':del tx['meta']
    elif failure in {'program','quote_program','fixed_account'}:
        field={'program':'program','quote_program':'token_quote_program','fixed_account':'pool_authority'}[failure]
        ix['accounts'][positions[field]]=accounts[positions['creator']]
    elif failure=='index':ix['accounts'][3]=-1
    else:ix['data']=b58encode(raw[:12] if failure=='truncated' else raw+b'x')
    await m._on_transaction(tx)
    assert not rows

def test_subscription_and_fallback_creation_filter_cover_every_dbc_initializer():
    request=create_combined_subscription()
    assert DBC_PROGRAM in request.transactions['memecoin_programs'].accounts_include
    for name in CREATE_NAMES:
        assert SolanaRpcProgramStream._looks_like_pool_creation(['Program log: Instruction: '+name])
    idl=Path('idl/meteora_dbc.json').read_bytes()
    assert hashlib.sha256(idl).hexdigest()=='63035b9a4a04b6a7f8fb9f81778275049c129a70d6f8bf6e08866516cf3da4cc'

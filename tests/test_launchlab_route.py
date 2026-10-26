"""Synthetic layout/SDK instruction contracts, not live execution proof."""
import hashlib
import struct
from types import SimpleNamespace

import pytest
from solders.pubkey import Pubkey

from src.chains import launchlab_route as r
from src.chains.idl import load_idl
from src.chains.pump_route import TOKEN_PROGRAM, TOKEN_2022_PROGRAM, WSOL_MINT, associated_token_address
from src.research import fork_simulation as f
from tests.test_fork_simulation import wire


def fixture():
    mint, creator, platform, config = [str(Pubkey.new_unique()) for _ in range(4)]
    pool = r.pool_address(mint)
    keys = [config, platform, mint, WSOL_MINT,
        r.pda(b'pool_vault', bytes(Pubkey.from_string(pool)), bytes(Pubkey.from_string(mint))),
        r.pda(b'pool_vault', bytes(Pubkey.from_string(pool)), bytes(Pubkey.from_string(WSOL_MINT))), creator]
    authority, bump = Pubkey.find_program_address([b'vault_auth_seed'], Pubkey.from_string(r.LAUNCHLAB_PROGRAM))
    raw = bytes([247,237,227,245,215,195,222,70])+struct.pack('<Q5B', 1,bump,0,6,9,1)
    raw += struct.pack('<10Q', *range(100,110))+bytes(40)
    raw += b''.join(bytes(Pubkey.from_string(k)) for k in keys)+bytes(64)
    state = r.parse_pool(raw, pool)
    accounts = {pool: wire(raw, r.LAUNCHLAB_PROGRAM)}
    for field, kind in (('global_config', 'GlobalConfig'), ('platform_config', 'PlatformConfig')):
        disc = next(a['discriminator'] for a in load_idl(r.IDL)['accounts'] if a['name'] == kind)
        accounts[state[field]] = wire(bytes(disc)+bytes(128), r.LAUNCHLAB_PROGRAM)
    for side, decimals in (('base', 6), ('quote',9)):
        mint_raw = bytearray(82);mint_raw[44:46] = bytes([decimals,1])
        accounts[state[side+'_mint']] = wire(mint_raw)
        vault = bytearray(165);vault[:32] = bytes(Pubkey.from_string(state[side+'_mint']))
        vault[32:64] = bytes(authority);vault[64:72] = (10**9).to_bytes(8,'little');vault[108] = 1
        accounts[state[side+'_vault']] = wire(vault)
    return state, accounts


@pytest.mark.parametrize('buy', [True,False])
def test_exact_input_matches_official_sdk_prefix_and_runtime_tail(buy):
    state, accounts = fixture()
    r.validate_frame(state, accounts, f.account_data)
    owner = str(Pubkey.new_unique())
    ix = r.build_swap(state, owner, 123, 7, buy=buy)
    name = 'buy_exact_in' if buy else 'sell_exact_in'
    assert ix.data == hashlib.sha256(('global:'+name).encode()).digest()[:8]+struct.pack('<3Q',123,7,0)
    authority = r.pda(b'vault_auth_seed')
    expected = [owner,authority,state['global_config'],state['platform_config'],state['pool'],
        associated_token_address(owner,state['base_mint'],TOKEN_PROGRAM),
        associated_token_address(owner,WSOL_MINT,TOKEN_PROGRAM),state['base_vault'],state['quote_vault'],
        state['base_mint'],WSOL_MINT,TOKEN_PROGRAM,TOKEN_PROGRAM,r.pda(b'__event_authority'),r.LAUNCHLAB_PROGRAM,
        '11111111111111111111111111111111',
        r.pda(bytes(Pubkey.from_string(state['platform_config'])), bytes(Pubkey.from_string(WSOL_MINT))),
        r.pda(bytes(Pubkey.from_string(state['creator'])), bytes(Pubkey.from_string(WSOL_MINT)))]
    assert [a.pubkey for a in ix.accounts] == expected
    assert [i for i,a in enumerate(ix.accounts) if a.is_signer] == [0]
    assert [i for i,a in enumerate(ix.accounts) if a.is_writable] == [0,4,5,6,7,8,16,17]
    assert ix.expected_accounts == 18 and ix.venue == 'raydium_launchlab'


@pytest.mark.parametrize('corruption', ['pool','bump','vault','discriminator','short','long','program_flag'])
def test_invalid_state_cannot_become_reviewed_route(corruption):
    state, accounts = fixture()
    raw = bytearray(f.account_data(accounts[state['pool']]))
    address = state['pool']
    if corruption == 'pool': address = str(Pubkey.new_unique())
    elif corruption == 'bump': raw[16] ^= 1
    elif corruption == 'vault': raw[269] ^= 1
    elif corruption == 'discriminator': raw[0] ^= 1
    elif corruption == 'short': raw.pop()
    elif corruption == 'long': raw.append(0)
    elif corruption == 'program_flag': raw[365] = 255
    with pytest.raises(ValueError): r.parse_pool(raw,address)


@pytest.mark.parametrize('corruption', ['migrated','quote','token2022','config_owner','mint','vault_mint','vault_authority'])
def test_unreviewed_or_wrong_bank_accounts_remain_unknown(corruption):
    state, accounts = fixture()
    if corruption == 'migrated': state['status'] = 2
    elif corruption == 'quote': state['quote_mint'] = str(Pubkey.new_unique())
    elif corruption == 'token2022':
        state['token_program_flag'] = 1
        state['base_token_program'] = TOKEN_2022_PROGRAM
    elif corruption == 'config_owner': accounts[state['platform_config']]['owner'] = TOKEN_PROGRAM
    else:
        key = state['base_mint' if corruption == 'mint' else 'base_vault']
        raw = bytearray(f.account_data(accounts[key]))
        raw[44 if corruption == 'mint' else (0 if corruption == 'vault_mint' else 32)] ^= 1
        accounts[key] = wire(raw)
    with pytest.raises(ValueError): r.validate_frame(state, accounts, f.account_data)


def test_fork_route_uses_exact_budget_and_actual_sell_amount():
    state, accounts = fixture()
    target = {'token':state['base_mint'],'pool':state['pool'],'venue':'raydium_launchlab','quote_budget':12345}
    buy, sell, program = f._route(target, accounts)
    assert program == TOKEN_PROGRAM and struct.unpack('<3Q',buy.data[8:]) == (12345,1,0)
    assert struct.unpack('<3Q',sell(987).data[8:]) == (987,1,0)
    accounts[state['pool']]['owner'] = TOKEN_2022_PROGRAM
    with pytest.raises(f.ForkUnavailable): f._route(target, accounts)


def test_episode_targets_preserve_other_venue_identity():
    from src.runtime.ingestion import MinedRecordIngestion
    episode = SimpleNamespace(factory=r.LAUNCHLAB_PROGRAM,base_token=WSOL_MINT,pair='observed_pool')
    value = SimpleNamespace(dataset_builder=SimpleNamespace(active_episodes={'observed_mint':episode}))
    targets = MinedRecordIngestion._native_fork_targets(value)
    assert targets == [{'token':'observed_mint','venue':'raydium_launchlab','pool':'observed_pool','quote_budget':1_000_000}]
    episode.base_token = str(Pubkey.new_unique())
    assert MinedRecordIngestion._native_fork_targets(value) == []


@pytest.mark.parametrize('amount', [True,0,-1,2**64,1.5])
def test_probe_never_builds_invalid_amounts(amount):
    state,_ = fixture()
    with pytest.raises(ValueError): r.build_swap(state,str(Pubkey.new_unique()),amount,1,buy=True)


@pytest.mark.parametrize('extension', [1,14])
def test_token2022_transfer_fees_reach_vm_hooks_remain_unknown(extension):
    state,accounts = fixture()
    raw = bytearray(f.account_data(accounts[state['pool']]))
    raw[365] = 1
    accounts[state['pool']] = wire(raw,r.LAUNCHLAB_PROGRAM)
    state = r.parse_pool(raw,state['pool'])
    mint = state['base_mint']
    data = f.account_data(accounts[mint])+bytes(83)+b'\x01'+struct.pack('<HH',extension,108 if extension==1 else 64)+bytes(108 if extension==1 else 64)
    accounts[mint] = wire(data,TOKEN_2022_PROGRAM)
    accounts[state['base_vault']] = wire(f.account_data(accounts[state['base_vault']])+b'\x02',TOKEN_2022_PROGRAM)
    if extension == 14:
        with pytest.raises(ValueError, match='transfer-hook'): r.validate_frame(state,accounts,f.account_data)
    else:
        r.validate_frame(state,accounts,f.account_data)
        buy,sell,program = f._route({'token':mint,'pool':state['pool'],'venue':'raydium_launchlab','quote_budget':1000},accounts)
        assert program == TOKEN_2022_PROGRAM and buy.accounts[11].pubkey == program
        assert buy.accounts[5].pubkey == associated_token_address(str(f.virtual_owner().pubkey()),mint,program)

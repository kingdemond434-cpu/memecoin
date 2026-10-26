"""Unsigned DBC local-bank probes, from the pinned MIT SDK IDL.

Only ordinary VirtualPool/PoolConfig accounts with SPL/Token-2022 base and
SPL WSOL quote are reviewed. Hook accounts and other quote assets stay unknown.
No curve formula, remote signer or network transaction transport lives here.
"""
from solders.pubkey import Pubkey

from src.chains.idl import AccountMeta, build_accounts, discriminator, load_idl
from src.chains.meteora_dbc import DBC_PROGRAM
from src.chains.pump_route import (PreparedInstruction, TOKEN_PROGRAM,
    TOKEN_2022_PROGRAM, WSOL_MINT, associated_token_address)

IDL='meteora_dbc'
INSTRUCTIONS_SYSVAR='Sysvar1nstructions1111111111111111111111111'
SIZES={'VirtualPool':424,'PoolConfig':1048}


class DBCSubsetUnavailable(ValueError):
    reason_code = 'DBC_POOL_MIGRATED'


def pda(*seeds):
    return str(Pubkey.find_program_address(list(seeds),Pubkey.from_string(DBC_PROGRAM))[0])


def pool_address(mint,config,quote=WSOL_MINT):
    first,second=sorted((bytes(Pubkey.from_string(mint)),bytes(Pubkey.from_string(quote))),reverse=True)
    return pda(b'pool',bytes(Pubkey.from_string(config)),first,second)


def vault_address(pool,mint):
    return pda(b'token_vault',bytes(Pubkey.from_string(mint)),bytes(Pubkey.from_string(pool)))


def decode_account(raw,kind):
    """Read the complete reviewed fixed layout, including explicit IDL padding."""
    idl=load_idl(IDL)
    expected=next(a['discriminator'] for a in idl['accounts'] if a['name']==kind)
    if (not isinstance(raw,(bytes,bytearray)) or len(raw)!=SIZES[kind]
            or raw[:8]!=bytes(expected)):
        raise ValueError('unreviewed DBC account layout')
    types={row['name']:row['type'] for row in idl['types']};offset=8
    def take(count):
        nonlocal offset
        if offset+count>len(raw):raise ValueError('truncated DBC account')
        value=bytes(raw[offset:offset+count]);offset+=count;return value
    def read(schema):
        if schema=='pubkey':return str(Pubkey.from_bytes(take(32)))
        if isinstance(schema,str) and schema in ('u8','u16','u32','u64','u128'):
            return int.from_bytes(take(int(schema[1:])//8),'little')
        if isinstance(schema,dict) and 'array' in schema:
            return [read(schema['array'][0]) for _ in range(schema['array'][1])]
        if isinstance(schema,dict) and 'defined' in schema:
            definition=types[schema['defined']['name']]
            if definition['kind']=='struct':
                return {field['name']:read(field['type']) for field in definition['fields']}
        raise ValueError('unreviewed DBC fixed field schema')
    value=read({'defined':{'name':kind}})
    if offset!=len(raw):raise ValueError('unreviewed DBC account tail')
    return value


def parse_pool(raw,address):
    state=decode_account(raw,'VirtualPool')['pool_state']
    Pubkey.from_string(address)
    if state['pool_type'] not in (0,1) or state['is_migrated'] not in (0,1):
        raise ValueError('unreviewed DBC pool flags')
    state.update(pool=address,authority=pda(b'pool_authority'),
        base_token_program=TOKEN_2022_PROGRAM if state['pool_type']==1 else TOKEN_PROGRAM)
    return state


def validate_frame(state,accounts,decode):
    config_value=accounts.get(state['config'])
    if not config_value or config_value['owner']!=DBC_PROGRAM:
        raise ValueError('DBC config owner mismatch')
    config=decode_account(decode(config_value),'PoolConfig')
    if state['is_migrated'] != 0:
        raise DBCSubsetUnavailable('DBC pool migrated; original curve is not the active trading venue')
    if (config['quote_mint']!=WSOL_MINT or config['quote_token_flag']!=0
            or config['token_type']!=state['pool_type']):
        raise ValueError('DBC unmigrated base/WSOL subset unavailable')
    if pool_address(state['base_mint'],state['config'])!=state['pool']:
        raise ValueError('DBC pool derivation mismatch')
    if state['base_mint']==WSOL_MINT:
        raise ValueError('DBC base and quote are identical')
    from src.detection.rug_detector import RugDetector
    for side,mint,program in (('base',state['base_mint'],state['base_token_program']),
                              ('quote',WSOL_MINT,TOKEN_PROGRAM)):
        address=state[side+'_vault']
        if address!=vault_address(state['pool'],mint):raise ValueError('DBC vault derivation mismatch')
        mint_value,vault=accounts.get(mint),accounts.get(address)
        if not mint_value or mint_value['owner']!=program or not vault or vault['owner']!=program:
            raise ValueError('DBC mint/vault program mismatch')
        mint_raw,vault_raw=decode(mint_value),decode(vault)
        parsed=RugDetector.parse_spl_mint(mint_raw,program)
        if 'transfer_hook' in parsed['extensions']:
            raise ValueError('DBC transfer-hook accounts unreviewed')
        if ((program==TOKEN_PROGRAM and (len(mint_raw)!=82 or len(vault_raw)!=165))
                or len(vault_raw)<165 or not parsed['initialized']
                or parsed['decimals']!=(config['token_decimal'] if side=='base' else 9)
                or vault_raw[:32]!=bytes(Pubkey.from_string(mint))
                or vault_raw[32:64]!=bytes(Pubkey.from_string(state['authority'])) or vault_raw[108]!=1):
            raise ValueError('DBC mint/vault identity or layout mismatch')
    # The program evaluates the recorded config/clock/curve, fees and activation.
    # We do not replace those checks with an estimated quote or patched balances.
    return config


def build_swap(state,owner,amount,minimum,*,buy):
    if type(buy) is not bool or any(type(n) is not int or not 0<n<2**64 for n in (amount,minimum)):
        raise ValueError('invalid DBC swap arguments')
    Pubkey.from_string(owner)
    base=associated_token_address(owner,state['base_mint'],state['base_token_program'])
    quote=associated_token_address(owner,WSOL_MINT,TOKEN_PROGRAM)
    supplied={'pool_authority':state['authority'],'config':state['config'],'pool':state['pool'],
        'base_vault':state['base_vault'],'quote_vault':state['quote_vault'],
        'base_mint':state['base_mint'],'quote_mint':WSOL_MINT,'payer':owner,
        'input_token_account':quote if buy else base,'output_token_account':base if buy else quote,
        'token_base_program':state['base_token_program'],'token_quote_program':TOKEN_PROGRAM,
        'referral_token_account':DBC_PROGRAM,'event_authority':pda(b'__event_authority'),'program':DBC_PROGRAM}
    accounts=build_accounts(IDL,'swap',supplied)
    optional=next(i for i,row in enumerate(load_idl(IDL)['instructions']) if row['name']=='swap')
    schema=load_idl(IDL)['instructions'][optional]
    accounts=[AccountMeta(meta.pubkey,False,False) if row.get('optional') and meta.pubkey==DBC_PROGRAM else meta
              for meta,row in zip(accounts,schema['accounts'])]
    # Always supply the real transaction instructions sysvar. The SDK requires
    # it for rate-limited/first-swap configs; no synthetic sysvar is fabricated.
    accounts.append(AccountMeta(INSTRUCTIONS_SYSVAR,False,False))
    fields=next(t['type']['fields'] for t in load_idl(IDL)['types'] if t['name']=='SwapParameters')
    if [(f['name'],f['type']) for f in fields]!=[('amount_in','u64'),('minimum_amount_out','u64')]:
        raise ValueError('DBC swap argument schema changed')
    data=discriminator(IDL,'swap')+amount.to_bytes(8,'little')+minimum.to_bytes(8,'little')
    return PreparedInstruction('OK',DBC_PROGRAM,accounts,data,expected_accounts=len(accounts),venue='meteora_dbc')

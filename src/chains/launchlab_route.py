"""Unsigned LaunchLab instructions for local sampled-bank research.

The pinned IDL supplies the 15-account prefix. Raydium's SDK additionally
requires system, platform claim vault and creator claim vault, even with no
share fee. No quote formula, transaction transport or real signer lives here.
"""
from solders.pubkey import Pubkey

from src.chains.idl import AccountMeta, build_accounts, encode_u64_args, load_idl
from src.chains.launchlab import LAUNCHLAB_PROGRAM
from src.chains.pump_route import PreparedInstruction, TOKEN_PROGRAM, TOKEN_2022_PROGRAM, WSOL_MINT, associated_token_address

IDL = 'raydium_launchlab'
POOL_SIZE = 429


def pda(*seeds):
    return str(Pubkey.find_program_address(list(seeds), Pubkey.from_string(LAUNCHLAB_PROGRAM))[0])


def pool_address(mint, quote=WSOL_MINT):
    return pda(b'pool', bytes(Pubkey.from_string(mint)), bytes(Pubkey.from_string(quote)))


def parse_pool(raw, address):
    """Read the complete reviewed fixed layout; never infer mint/vault identity."""
    if not isinstance(raw, (bytes, bytearray)):
        raise ValueError('LaunchLab pool bytes required')
    raw = bytes(raw)
    discriminator = next(a['discriminator'] for a in load_idl(IDL)['accounts'] if a['name'] == 'PoolState')
    if len(raw) != POOL_SIZE or raw[:8] != bytes(discriminator):
        raise ValueError('unreviewed LaunchLab pool layout')
    offset = 8
    def integer(width):
        nonlocal offset
        value = int.from_bytes(raw[offset:offset+width], 'little')
        offset += width
        return value
    state = {'pool': address, 'epoch': integer(8)}
    for name in ('auth_bump', 'status', 'base_decimals', 'quote_decimals', 'migrate_type'):
        state[name] = integer(1)
    for name in ('supply', 'total_base_sell', 'virtual_base', 'virtual_quote', 'real_base',
                 'real_quote', 'total_quote_fund_raising', 'quote_protocol_fee', 'platform_fee', 'migrate_fee'):
        state[name] = integer(8)
    state['vesting_schedule'] = {name: integer(8) for name in ('total_locked_amount',
        'cliff_period', 'unlock_period', 'start_time', 'allocated_share_amount')}
    for name in ('global_config', 'platform_config', 'base_mint', 'quote_mint', 'base_vault', 'quote_vault', 'creator'):
        state[name] = str(Pubkey.from_bytes(raw[offset:offset+32]))
        offset += 32
    state['token_program_flag'], state['amm_creator_fee_on'] = integer(1), integer(1)
    state['platform_vesting_share'] = integer(8)
    state['padding_hex'] = raw[offset:].hex()
    if (state['status'] not in (0, 1, 2) or state['migrate_type'] not in (0, 1)
            or state['token_program_flag'] > 3 or state['amm_creator_fee_on'] > 2
            or state['base_mint'] == state['quote_mint']
            or pool_address(state['base_mint'], state['quote_mint']) != address):
        raise ValueError('LaunchLab pool identity/flags mismatch')
    authority, bump = Pubkey.find_program_address([b'vault_auth_seed'], Pubkey.from_string(LAUNCHLAB_PROGRAM))
    if state['auth_bump'] != bump:
        raise ValueError('LaunchLab authority bump mismatch')
    state['authority'] = str(authority)
    state['base_token_program'] = TOKEN_2022_PROGRAM if state['token_program_flag'] & 1 else TOKEN_PROGRAM
    for field, mint in (('base_vault', state['base_mint']), ('quote_vault', state['quote_mint'])):
        if state[field] != pda(b'pool_vault', bytes(Pubkey.from_string(address)), bytes(Pubkey.from_string(mint))):
            raise ValueError('LaunchLab vault derivation mismatch')
    return state


def validate_frame(state, accounts, decode):
    """Verify SPL or Token-2022 base/WSOL, without guessed hook accounts."""
    if state['status'] != 0 or state['quote_mint'] != WSOL_MINT or state['token_program_flag'] not in (0, 1):
        raise ValueError('LaunchLab funding base/WSOL subset unavailable')
    for name, kind in (('global_config', 'GlobalConfig'), ('platform_config', 'PlatformConfig')):
        value = accounts.get(state[name])
        expected = next(a['discriminator'] for a in load_idl(IDL)['accounts'] if a['name'] == kind)
        if not value or value['owner'] != LAUNCHLAB_PROGRAM or decode(value)[:8] != bytes(expected):
            raise ValueError('LaunchLab configuration identity mismatch')
    for side in ('base', 'quote'):
        mint, vault = accounts.get(state[side+'_mint']), accounts.get(state[side+'_vault'])
        program = state['base_token_program'] if side == 'base' else TOKEN_PROGRAM
        if not mint or mint['owner'] != program or not vault or vault['owner'] != program:
            raise ValueError('LaunchLab mint/vault program mismatch')
        mint_raw, vault_raw = decode(mint), decode(vault)
        from src.detection.rug_detector import RugDetector
        parsed = RugDetector.parse_spl_mint(mint_raw, program)
        if 'transfer_hook' in parsed['extensions']:
            raise ValueError('LaunchLab transfer-hook remaining accounts unreviewed')
        if ((program == TOKEN_PROGRAM and (len(mint_raw) != 82 or len(vault_raw) != 165))
                or len(vault_raw) < 165 or not parsed['initialized'] or parsed['decimals'] != state[side+'_decimals']
                or vault_raw[:32] != bytes(Pubkey.from_string(state[side+'_mint']))
                or vault_raw[32:64] != bytes(Pubkey.from_string(state['authority'])) or vault_raw[108] != 1):
            raise ValueError('LaunchLab mint/vault layout mismatch')


def build_swap(state, owner, amount, minimum, *, buy):
    """Exact-input probe, with zero optional share fee and SDK remaining accounts.

    Probe minimum=1 tests transferability; it is NOT a live slippage policy.
    """
    if type(buy) is not bool or any(type(n) is not int or not 0 < n < 2**64 for n in (amount, minimum)):
        raise ValueError('invalid LaunchLab swap arguments')
    Pubkey.from_string(owner)
    name = 'buy_exact_in' if buy else 'sell_exact_in'
    supplied = {key: state[key] for key in ('authority', 'global_config', 'platform_config', 'base_vault', 'quote_vault')}
    supplied.update(payer=owner, pool_state=state['pool'], base_token_mint=state['base_mint'],
        quote_token_mint=state['quote_mint'], base_token_program=state['base_token_program'], quote_token_program=TOKEN_PROGRAM,
        user_base_token=associated_token_address(owner, state['base_mint'], state['base_token_program']),
        user_quote_token=associated_token_address(owner, state['quote_mint'], TOKEN_PROGRAM),
        event_authority=pda(b'__event_authority'), program=LAUNCHLAB_PROGRAM)
    accounts = build_accounts(IDL, name, supplied)
    # Official SDK instrument.ts/pda.ts at e2e8bcd; shareFeeReceiver omitted.
    accounts.extend((AccountMeta('11111111111111111111111111111111', False, False),
        AccountMeta(pda(bytes(Pubkey.from_string(state['platform_config'])), bytes(Pubkey.from_string(state['quote_mint']))), False, True),
        AccountMeta(pda(bytes(Pubkey.from_string(state['creator'])), bytes(Pubkey.from_string(state['quote_mint']))), False, True)))
    return PreparedInstruction('OK', LAUNCHLAB_PROGRAM, accounts,
        encode_u64_args(IDL, name, [amount, minimum, 0]), expected_accounts=len(accounts), venue='raydium_launchlab')

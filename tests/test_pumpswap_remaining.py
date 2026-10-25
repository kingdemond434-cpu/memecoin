"""Current runtime account contract, independently derived from official SDK."""
import pytest
from solders.pubkey import Pubkey

from src.chains.idl import PUMP_AMM_IDL, account_names
from src.chains.pump_route import (SYSTEM_PROGRAM, TOKEN_PROGRAM, TOKEN_2022_PROGRAM,
                                   WSOL_MINT, fee_recipients)
from src.chains.pumpswap_route import PUMPSWAP_PROGRAM, PoolState, PumpSwapRoute


def pda(seed, key, program=PUMPSWAP_PROGRAM):
    return str(Pubkey.find_program_address([seed, bytes(Pubkey.from_string(key))],
                                           Pubkey.from_string(program))[0])


def ata(owner, mint):
    from src.chains.pump_route import ASSOCIATED_TOKEN_PROGRAM
    return str(Pubkey.find_program_address([bytes(Pubkey.from_string(owner)),
        bytes(Pubkey.from_string(TOKEN_PROGRAM)), bytes(Pubkey.from_string(mint))],
        Pubkey.from_string(ASSOCIATED_TOKEN_PROGRAM))[0])


@pytest.mark.parametrize('side', ['buy', 'sell'])
@pytest.mark.parametrize('cashback', [False, True])
@pytest.mark.parametrize('creator_zero', [False, True])
@pytest.mark.parametrize('mayhem', [False, True])
def test_runtime_account_order_flags_and_conditional_offsets(side, cashback, creator_zero, mayhem):
    user, mint = TOKEN_PROGRAM, TOKEN_2022_PROGRAM
    pool = PoolState(status='OK', pool=WSOL_MINT, base_mint=mint,
        quote_mint=WSOL_MINT, pool_base_token_account=user,
        pool_quote_token_account=user, creator=user,
        coin_creator=SYSTEM_PROGRAM if creator_zero else user,
        is_cashback_coin=cashback, is_mayhem_mode=mayhem)
    instruction = getattr(PumpSwapRoute(), 'build_'+side)(pool, user, 1000, 100)
    assert instruction.ok, instruction.detail
    prefix = len(account_names(PUMP_AMM_IDL, side))
    tail = instruction.accounts[prefix:]
    expected = []
    if cashback:
        accumulator = pda(b'user_volume_accumulator', user)
        expected.append((ata(accumulator, WSOL_MINT), False, True))
        if side == 'sell':
            expected.append((accumulator, False, True))
    if not creator_zero:
        expected.append((pda(b'pool-v2', mint), False, False))
    # All current modes, including zero-fee mayhem, still require the pair.
    recipient = tail[-2].pubkey
    assert recipient in fee_recipients()['buyback']
    expected.extend([(recipient, False, False), (ata(recipient, WSOL_MINT), False, True)])
    assert [(m.pubkey, m.is_signer, m.is_writable) for m in tail] == expected
    assert instruction.expected_accounts == prefix + len(expected)


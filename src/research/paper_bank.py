"""Carry virtual trader accounts between observed local SVM banks.

Only the synthetic trader's allowlisted accounts persist. All market accounts,
program images and the clock come from the new public frame. This exogenous
market overlay omits our hypothetical impact on the subsequent market path.
It is a prospective infrastructure control, never a confirmed trade or alpha.
"""
from __future__ import annotations

import base64
import copy
import math

from src.research import fork_simulation as f

SCHEMA = 'prospective-local-bank-inventory-v1'
SCOPE = 'PROSPECTIVE_LOCAL_SVM_EXOGENOUS_MARKET_CONTROL'
LIMITATIONS = ['hypothetical entry impact is not propagated into subsequent public market states',
    'no landing, failed-submission, priority-fee or tip distribution',
    'public virtual trader has no on-chain history; targeted freezes and off-chain reactions unmodelled',
    'exact validator parity unverified; failed local exits are not proven honeypots',
    'locked rent conservatively expensed; no assumed account closure or rent refund']


def _frame_hash(frame):
    return f.digest({key: value for key, value in frame.items() if key != '_paper_request'})


def inventory_keys(frame, token_program):
    """No market, mint, vault or executable account may enter the overlay."""
    from solders.pubkey import Pubkey
    owner = str(f.virtual_owner().pubkey())
    mint = frame['target']['token']
    keys = {owner, f.associated_token_address(owner, mint, token_program),
            f.associated_token_address(owner, f.WSOL_MINT, f.TOKEN_PROGRAM)}
    # User-specific volume/cashback state; no global accumulator or creator vault.
    for program in (f.PUMP_PROGRAM, f.PUMPSWAP_PROGRAM):
        address = str(Pubkey.find_program_address(
            [b'user_volume_accumulator', bytes(Pubkey.from_string(owner))], Pubkey.from_string(program))[0])
        for key in (address, f.associated_token_address(address, f.WSOL_MINT, f.TOKEN_PROGRAM)):
            if key in frame['accounts']:
                keys.add(key)
    return keys


def _snapshot(vm, keys):
    from solders.pubkey import Pubkey
    rows = {}
    for key in sorted(keys):
        value = vm.get_account(Pubkey.from_string(key))
        rows[key] = None if value is None else {
            'data': [base64.b64encode(bytes(value.data)).decode(), 'base64'],
            'owner': str(value.owner), 'lamports': value.lamports,
            'executable': value.executable, 'rentEpoch': value.rent_epoch}
    return rows


def _seal(state):
    state['seal'] = f.digest({key: value for key, value in state.items() if key != 'seal'})
    return state


def _liquid(vm):
    return vm.get_balance(f.virtual_owner().pubkey()) + f._tokens(vm, f.WSOL_MINT, f.TOKEN_PROGRAM)


def capture_entry(vm, frame, result, token_program):
    keys = inventory_keys(frame, token_program)
    if any(frame['accounts'].get(key) is not None for key in keys):
        raise f.ForkUnavailable('virtual inventory collides with public account state')
    at = frame['observed_at']
    if type(at) not in (float, int) or not math.isfinite(at) or at <= 0:
        raise f.ForkUnavailable('paper entry observation clock invalid')
    amount = f._tokens(vm, frame['target']['token'], token_program)
    state = _seal({'schema': SCHEMA, 'target': copy.deepcopy(frame['target']),
        'token_program': token_program, 'entry_slot': frame['slot'], 'last_slot': frame['slot'],
        'entry_observed_at': at, 'last_observed_at': at,
        'entry_frame_sha256': _frame_hash(frame), 'last_frame_sha256': _frame_hash(frame),
        'virtual_funding_lamports': result['synthetic_funding_lamports'],
        'entry_token_units': amount, 'token_units': amount,
        'vm_fees_lamports': sum(result.get(phase, {}).get('vm_fee_lamports',0) for phase in ('setup','buy')),
        'accounts': _snapshot(vm, keys), 'exit_attempts': 0,
        'entry_liquid_lamports': _liquid(vm)})
    opened = result.get('buy',{}).get('succeeded') is True and amount > 0
    return {**result, 'status': 'PAPER_BANK_OPENED' if opened else 'PAPER_BANK_ENTRY_FAILED', 'scope': SCOPE, 'inventory': state,
        'real_fill': False, 'promotion_eligible': False, 'sell_simulation_pass': None,
        'conditional_net_liquid_pnl_lamports': None if opened else _liquid(vm)-state['virtual_funding_lamports'],
        'limitations': LIMITATIONS}


def execute_exit(vm, frame, inventory, sell, token_program):
    from solders.account import Account
    from solders.pubkey import Pubkey
    state = copy.deepcopy(inventory)
    if (state.get('schema') != SCHEMA or state.get('seal') != f.digest(
            {key: value for key, value in state.items() if key != 'seal'})
            or state.get('target') != frame['target'] or state.get('token_program') != token_program):
        raise f.ForkUnavailable('paper inventory identity/seal mismatch')
    if (type(frame.get('observed_at')) not in (float, int) or not math.isfinite(frame['observed_at'])
            or frame['slot'] <= state['last_slot'] or frame['observed_at'] <= state['last_observed_at']):
        raise f.ForkUnavailable('paper exit requires a strictly later observed bank')
    keys = inventory_keys(frame, token_program)
    if set(state['accounts']) != keys or any(frame['accounts'].get(key) is not None for key in keys):
        raise f.ForkUnavailable('paper inventory account allowlist/public collision changed')
    for key, value in state['accounts'].items():
        if value is not None:
            raw = f.account_data(value)
            if value['executable']:
                raise f.ForkUnavailable('paper inventory cannot overlay executable code')
            vm.set_account(Pubkey.from_string(key), Account(value['lamports'], raw,
                Pubkey.from_string(value['owner']), False, value['rentEpoch']))
    amount = f._tokens(vm, frame['target']['token'], token_program)
    if type(state['token_units']) is not int or not amount == state['token_units'] > 0:
        raise f.ForkUnavailable('paper inventory balance differs from retained actual units')
    receipt = f._receipt(f._transact(vm, [f._instruction(sell(amount))]))
    remaining = f._tokens(vm, frame['target']['token'], token_program)
    after = _liquid(vm)
    state.update(accounts=_snapshot(vm, keys), token_units=remaining,
        last_slot=frame['slot'], last_observed_at=frame['observed_at'],
        last_frame_sha256=_frame_hash(frame), exit_attempts=state['exit_attempts']+1,
        vm_fees_lamports=state['vm_fees_lamports']+receipt['vm_fee_lamports'])
    closed = receipt['succeeded'] and remaining == 0
    # A failed local transaction's actual fee debit persists into the next try.
    return {'schema': SCHEMA, 'status': 'PAPER_BANK_CLOSED' if closed else 'PAPER_BANK_EXIT_UNRESOLVED',
        'scope': SCOPE, 'inventory': _seal(state), 'exit': receipt,
        'sell_input_token_units': amount, 'remaining_token_units': remaining,
        'slot': frame['slot'], 'observed_at': frame['observed_at'],
        'entry_frame_sha256': state['entry_frame_sha256'], 'exit_frame_sha256': _frame_hash(frame),
        'input_sha256': f.digest(frame), 'conditional_net_liquid_pnl_lamports':
            after-state['virtual_funding_lamports'] if closed else None,
        'vm_fees_lamports': state['vm_fees_lamports'], 'fees_already_in_liquid_balance_change': True,
        'real_fill': False, 'real_transaction_submitted': False, 'promotion_eligible': False,
        'sell_simulation_pass': None, 'limitations': LIMITATIONS,
        'implementation': f.implementation_manifest()}

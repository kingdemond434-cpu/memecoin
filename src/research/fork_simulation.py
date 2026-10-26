"""Read-only chain snapshots and local, sequential SVM round trips.

This module has no network transaction submission or wallet signer. A pass is
an observation in a sampled local bank, never a promise about a future sell.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import struct
import sys
import time

from src.chains.pump_route import (ASSOCIATED_TOKEN_PROGRAM, NativePumpRoute,
    PUMP_PROGRAM, PumpRouteConfig, TOKEN_2022_PROGRAM, TOKEN_PROGRAM, WSOL_MINT,
    associated_token_address)
from src.chains.pumpswap_route import (PUMPSWAP_PROGRAM, PumpSwapRoute,
    PumpSwapRouteConfig, parse_pool)
from src.chains.rpc_manager import RPCError
from src.chains import launchlab_route, meteora_dbc_route

SCHEMA = 'native-fork-roundtrip-v1'
CLOCK = 'SysvarC1ock11111111111111111111111111111111'
UPGRADEABLE = 'BPFLoaderUpgradeab1e11111111111111111111111'
IMMUTABLE = 'BPFLoader2111111111111111111111111111111111'
NATIVE = 'NativeLoader1111111111111111111111111111111'
SYSTEM = '11111111111111111111111111111111'
MAX_FRAME_BYTES = 24 * 1024 * 1024  # parser/process envelope, not a token universe cap


class ForkUnavailable(ValueError):
    def __init__(self, message, *, reason_code='FORK_ENVIRONMENT_UNVERIFIED'):
        if reason_code not in {'FORK_ENVIRONMENT_UNVERIFIED', 'DBC_POOL_MIGRATED'}:
            raise ValueError('unreviewed fork availability code')
        self.reason_code = reason_code
        super().__init__(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def implementation_manifest():
    """Bind route/decoder/IDL inputs as well as this worker's own source.

    An unchanged worker file can load a corrected route builder. Its single
    file hash alone cannot identify the implementation that produced a probe.
    This is local reproducibility metadata, not external software attestation.
    """
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    paths = ('src/research/fork_simulation.py', 'src/research/paper_bank.py', 'src/chains/pump_route.py',
        'src/chains/pumpswap_route.py', 'src/chains/pump_curve.py', 'src/chains/idl.py',
        'idl/pump.json', 'idl/pump_amm.json', 'idl/pump_fees.json', 'idl/FEE_RECIPIENTS.md',
        'src/chains/launchlab_route.py', 'src/chains/launchlab.py', 'idl/raydium_launchlab.json',
        'src/detection/rug_detector.py','src/chains/meteora_dbc_route.py',
        'src/chains/meteora_dbc.py','idl/meteora_dbc.json')
    return {'schema': 'local-fork-source-manifest-v1',
        'python_version': '.'.join(map(str, sys.version_info[:3])),
        'source_sha256': {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in paths}}


def virtual_owner():
    # A PUBLIC, deterministic test seed. Never fund this address on chain.
    from solders.keypair import Keypair
    return Keypair.from_seed(hashlib.sha256(SCHEMA.encode()).digest())


def account_data(value):
    if not isinstance(value, dict) or not isinstance(value.get('data'), list) or value['data'][1:] != ['base64']:
        raise ForkUnavailable('missing/unencoded account')
    try:
        raw = base64.b64decode(value['data'][0], validate=True)
    except (ValueError, TypeError) as exc:
        raise ForkUnavailable('invalid account base64') from exc
    if len(raw) > MAX_FRAME_BYTES:
        raise ForkUnavailable('account exceeds process envelope')
    if type(value.get('lamports')) is not int or not 0 <= value['lamports'] < 2**64 or type(value.get('executable')) is not bool:
        raise ForkUnavailable('invalid account metadata')
    from solders.pubkey import Pubkey
    Pubkey.from_string(value['owner'])
    return raw


async def read_frame(rpc, addresses, *, batch=100):
    """Adaptive provider batches, but NEVER combine different bank slots."""
    keys = sorted(set(addresses))
    if not keys:
        raise ForkUnavailable('empty account frame')
    async def fetch_group(group):
        try:
            result = await asyncio.wait_for(rpc.request('getMultipleAccounts', [group,
                {'encoding': 'base64', 'commitment': 'finalized'}]), timeout=12)
        except RPCError as exc:
            error = exc.error if isinstance(exc.error, dict) else {}
            match = re.search(r'\bmaximum\s+of\s+(\d+)\s+accounts?\b', str(error.get('message', '')), re.I)
            limit = int(match[1]) if match and error.get('code') == -32602 else 0
            if not 0 < limit < len(group):
                raise
            # Concurrent groups improve the chance of reading one bank. The
            # shared RPC manager still owns concurrency and provider budgets.
            return await collect([group[i:i + limit] for i in range(0, len(group), limit)])
        return [(group, result)]

    async def collect(groups):
        children = await asyncio.gather(*(fetch_group(group) for group in groups), return_exceptions=True)
        for child in children:
            if isinstance(child, BaseException):
                raise child
        return [part for child in children for part in child]

    parts = await collect([keys[i:i + batch] for i in range(0, len(keys), batch)])
    rows, slots = {}, set()
    for group, result in parts:
        values = result.get('value') if isinstance(result, dict) else None
        slot = (result.get('context') or {}).get('slot') if isinstance(result, dict) else None
        if type(slot) is not int or slot <= 0 or not isinstance(values, list) or len(values) != len(group):
            raise ForkUnavailable('malformed account frame')
        slots.add(slot)
        if len(slots) != 1:
            raise ForkUnavailable('account groups came from different banks')
        for key, value in zip(group, values):
            if value is not None:
                account_data(value)
            rows[key] = value
    if len(json.dumps(rows)) > MAX_FRAME_BYTES * 2:
        raise ForkUnavailable('frame exceeds process envelope')
    return {'slot': slots.pop(), 'accounts': rows}


def _curve_address(mint):
    from solders.pubkey import Pubkey
    return str(Pubkey.find_program_address([b'bonding-curve', bytes(Pubkey.from_string(mint))],
                                          Pubkey.from_string(PUMP_PROGRAM))[0])


def _route(target, accounts):
    """Build only our reviewed native routes, with the virtual owner."""
    from src.chains.pump_curve import parse_bonding_curve
    mint, venue = target['token'], target['venue']
    mint_value = accounts.get(mint)
    raw_mint = account_data(mint_value)
    from src.research.pool_attribution import mint_decimals
    if mint_decimals(mint_value) is None:
        raise ForkUnavailable('unverified mint owner/layout')
    owner = str(virtual_owner().pubkey())
    if venue == 'pump_curve':
        curve = accounts.get(_curve_address(mint))
        if not curve or curve['owner'] != PUMP_PROGRAM:
            raise ForkUnavailable('curve owner mismatch')
        raw = account_data(curve)
        state = parse_bonding_curve(raw)
        # Pump defines absent trailing extension fields as zero/default. Its
        # on-chain default quote mint means native SOL; the v2 instruction
        # still receives WSOL as its interface mint, but transfers lamports.
        from solders.pubkey import Pubkey
        mayhem = raw[81] if len(raw) > 81 else 0
        cashback = raw[82] if len(raw) > 82 else 0
        if mayhem not in (0, 1) or cashback not in (0, 1) or not state.tradeable:
            raise ForkUnavailable('curve flags/creator/tradeability unavailable')
        creator = state.creator or SYSTEM  # documented default for pre-creator-field curves
        if 83 < len(raw) < 115 and any(raw[83:]):
            raise ForkUnavailable('partial non-default curve quote mint')
        quote_mint = str(Pubkey.from_bytes(raw[83:115])) if len(raw) >= 115 else SYSTEM
        if quote_mint not in {SYSTEM, WSOL_MINT}:
            raise ForkUnavailable('curve quote asset is not native SOL')
        if state.real_token_reserves <= 0:
            raise ForkUnavailable('curve real inventory unavailable')
        route = NativePumpRoute(PumpRouteConfig(base_token_program=mint_value['owner'], mayhem=bool(mayhem)))
        requested = min(state.real_token_reserves,
            target['quote_budget'] * state.virtual_token_reserves // (state.virtual_sol_reserves + target['quote_budget'])) * 9 // 10
        buy = route.build_buy(mint, creator, owner, requested, target['quote_budget'])
        sell = lambda amount: route.build_sell(mint, creator, owner, amount, 1)
    elif venue == 'pumpswap':
        pool_value = accounts.get(target['pool'])
        if not pool_value or pool_value['owner'] != PUMPSWAP_PROGRAM:
            raise ForkUnavailable('pool owner mismatch')
        pool = parse_pool(account_data(pool_value), target['pool'])
        quote_value = accounts.get(WSOL_MINT)
        if not pool.ok or pool.base_mint != mint or pool.quote_mint != WSOL_MINT or not quote_value or quote_value['owner'] != TOKEN_PROGRAM:
            raise ForkUnavailable('pool/mint/quote identity unsupported')
        base, quote = (accounts.get(key) for key in (pool.pool_base_token_account, pool.pool_quote_token_account))
        if not base or base['owner'] != mint_value['owner'] or not quote or quote['owner'] != TOKEN_PROGRAM:
            raise ForkUnavailable('pool token account owner mismatch')
        base_data, quote_data = account_data(base), account_data(quote)
        from solders.pubkey import Pubkey
        if (len(base_data) < 165 or len(quote_data) < 165 or base_data[:32] != bytes(Pubkey.from_string(mint))
                or quote_data[:32] != bytes(Pubkey.from_string(WSOL_MINT))
                or base_data[32:64] != bytes(Pubkey.from_string(pool.pool))
                or quote_data[32:64] != bytes(Pubkey.from_string(pool.pool))):
            raise ForkUnavailable('pool reserve mint/authority mismatch')
        b, q = int.from_bytes(base_data[64:72], 'little'), int.from_bytes(quote_data[64:72], 'little')
        if b <= 0 or q + pool.virtual_quote_reserves <= 0:
            raise ForkUnavailable('empty pool reserves')
        requested = target['quote_budget'] * b // (q + pool.virtual_quote_reserves + target['quote_budget']) * 9 // 10
        route = PumpSwapRoute(PumpSwapRouteConfig(base_token_program=mint_value['owner']))
        buy = route.build_buy(pool, owner, requested, target['quote_budget'])
        sell = lambda amount: route.build_sell(pool, owner, amount, 1)
    elif venue == 'raydium_launchlab':
        value = accounts.get(target['pool'])
        if not value or value['owner'] != launchlab_route.LAUNCHLAB_PROGRAM:
            raise ForkUnavailable('LaunchLab pool owner mismatch')
        try:
            state = launchlab_route.parse_pool(account_data(value), target['pool'])
            if state['base_mint'] != mint:
                raise ValueError('LaunchLab target mint mismatch')
            launchlab_route.validate_frame(state, accounts, account_data)
            buy = launchlab_route.build_swap(state, owner, target['quote_budget'], 1, buy=True)
            sell = lambda amount: launchlab_route.build_swap(state, owner, amount, 1, buy=False)
        except ValueError as exc:
            raise ForkUnavailable(str(exc)) from exc
    elif venue == 'meteora_dbc':
        value=accounts.get(target['pool'])
        if not value or value['owner']!=meteora_dbc_route.DBC_PROGRAM:
            raise ForkUnavailable('DBC pool owner mismatch')
        try:
            state=meteora_dbc_route.parse_pool(account_data(value),target['pool'])
            if state['base_mint']!=mint:raise ValueError('DBC target mint mismatch')
            meteora_dbc_route.validate_frame(state,accounts,account_data)
            buy=meteora_dbc_route.build_swap(state,owner,target['quote_budget'],1,buy=True)
            sell=lambda amount:meteora_dbc_route.build_swap(state,owner,amount,1,buy=False)
        except meteora_dbc_route.DBCSubsetUnavailable as exc:
            raise ForkUnavailable(str(exc), reason_code=exc.reason_code) from exc
        except ValueError as exc:
            raise ForkUnavailable(str(exc)) from exc
    else:
        raise ForkUnavailable('no reviewed native round-trip adapter for this venue')
    if not buy.ok or not sell(1).ok:
        raise ForkUnavailable('native instruction construction unavailable')
    return buy, sell, mint_value['owner']


async def acquire(rpc, target):
    from src.chains.pump_route import TOKEN_PROGRAM
    mint = target['token']
    if type(target.get('quote_budget')) is not int or not 0 < target['quote_budget'] < 2**63:
        raise ForkUnavailable('invalid local probe budget')
    if target['venue'] not in {'pump_curve', 'pumpswap', 'raydium_launchlab','meteora_dbc'}:
        raise ForkUnavailable('no reviewed native round-trip adapter for this venue')
    bootstrap_keys = [mint, WSOL_MINT, CLOCK]
    bootstrap_keys.append(_curve_address(mint) if target['venue'] == 'pump_curve' else target['pool'])
    bootstrap = await read_frame(rpc, bootstrap_keys)
    if target['venue'] == 'pumpswap':
        pool = parse_pool(account_data(bootstrap['accounts'][target['pool']]), target['pool'])
        if not pool.ok:
            raise ForkUnavailable('pool layout unavailable')
        bootstrap_keys.extend([pool.pool_base_token_account, pool.pool_quote_token_account])
        bootstrap = await read_frame(rpc, bootstrap_keys)
    elif target['venue'] == 'raydium_launchlab':
        value = bootstrap['accounts'].get(target['pool'])
        if not value or value['owner'] != launchlab_route.LAUNCHLAB_PROGRAM:
            raise ForkUnavailable('LaunchLab pool owner mismatch')
        try:
            state = launchlab_route.parse_pool(account_data(value), target['pool'])
        except ValueError as exc:
            raise ForkUnavailable(str(exc)) from exc
        bootstrap_keys.extend(state[key] for key in ('base_vault', 'quote_vault', 'global_config', 'platform_config'))
        bootstrap = await read_frame(rpc, bootstrap_keys)
    elif target['venue']=='meteora_dbc':
        value=bootstrap['accounts'].get(target['pool'])
        if not value or value['owner']!=meteora_dbc_route.DBC_PROGRAM:
            raise ForkUnavailable('DBC pool owner mismatch')
        try:state=meteora_dbc_route.parse_pool(account_data(value),target['pool'])
        except ValueError as exc:raise ForkUnavailable(str(exc)) from exc
        bootstrap_keys.extend(state[key] for key in ('base_vault','quote_vault','config'))
        bootstrap=await read_frame(rpc,bootstrap_keys)
    buy, sell, token_program = _route(target, bootstrap['accounts'])
    keys = set(bootstrap_keys) | {str(virtual_owner().pubkey())}
    for instruction in (buy, sell(1)):
        keys.add(instruction.program_id)
        keys.update(meta.pubkey for meta in instruction.accounts)
    keys.update([ASSOCIATED_TOKEN_PROGRAM, token_program, TOKEN_PROGRAM])
    # Discover loader dependencies, then reread EVERYTHING in the final bank.
    discovered = await read_frame(rpc, keys)
    for value in discovered['accounts'].values():
        if value and value['executable'] and value['owner'] == UPGRADEABLE:
            raw = account_data(value)
            if len(raw) != 36 or int.from_bytes(raw[:4], 'little') != 2:
                raise ForkUnavailable('upgradeable program pointer layout unsupported')
            from solders.pubkey import Pubkey
            keys.add(str(Pubkey.from_bytes(raw[4:36])))
    frame = await read_frame(rpc, keys)
    # Route identities may change between discovery and the final frame.
    final_buy, final_sell, _ = _route(target, frame['accounts'])
    required = {m.pubkey for ix in (final_buy, final_sell(1)) for m in ix.accounts}
    if not required <= set(frame['accounts']):
        raise ForkUnavailable('route dependencies changed while taking snapshot')
    frame.update(schema=SCHEMA, target=target, observed_at=time.time())
    return frame


def _program_images(frame):
    from solders.pubkey import Pubkey
    images = []
    for key, value in frame['accounts'].items():
        if not value or not value['executable']:
            continue
        owner, raw = value['owner'], account_data(value)
        if owner == NATIVE:
            if key != SYSTEM:
                raise ForkUnavailable('unreviewed native builtin')
            continue
        if owner == UPGRADEABLE:
            if len(raw) != 36 or int.from_bytes(raw[:4], 'little') != 2:
                raise ForkUnavailable('unsupported program pointer')
            address = str(Pubkey.from_bytes(raw[4:36]))
            backing = frame['accounts'].get(address)
            data = account_data(backing)
            if backing['owner'] != UPGRADEABLE or backing['executable'] or len(data) < 45 or int.from_bytes(data[:4], 'little') != 3:
                raise ForkUnavailable('program data owner/layout mismatch')
            if int.from_bytes(data[4:12], 'little') > frame['slot'] or data[12] not in (0, 1):
                raise ForkUnavailable('program data version is future/invalid')
            raw = data[45:]
        elif owner != IMMUTABLE:
            raise ForkUnavailable('unsupported program loader')
        if not raw.startswith(b'\x7fELF'):
            raise ForkUnavailable('program image is not ELF')
        images.append((key, owner, raw))
    return images


def _instruction(prepared):
    from solders.instruction import AccountMeta, Instruction
    from solders.pubkey import Pubkey
    if not prepared.ok:
        raise ForkUnavailable('native compiler refused instruction')
    return Instruction(Pubkey.from_string(prepared.program_id), bytes(prepared.data),
        [AccountMeta(Pubkey.from_string(m.pubkey), m.is_signer, m.is_writable) for m in prepared.accounts])


def _ata_instruction(mint, token_program):
    from solders.instruction import AccountMeta, Instruction
    from solders.pubkey import Pubkey
    user = str(virtual_owner().pubkey())
    address = associated_token_address(user, mint, token_program)
    return Instruction(Pubkey.from_string(ASSOCIATED_TOKEN_PROGRAM), b'\x01',
        [AccountMeta(Pubkey.from_string(key), signer, writable) for key, signer, writable in
         [(user, True, True), (address, False, True), (user, False, False),
          (mint, False, False), (SYSTEM, False, False), (token_program, False, False)]])


def _transact(vm, instructions):
    from solders.compute_budget import set_compute_unit_limit
    from solders.message import Message
    from solders.transaction import VersionedTransaction
    owner = virtual_owner()
    message = Message.new_with_blockhash([set_compute_unit_limit(800_000), *instructions],
                                         owner.pubkey(), vm.latest_blockhash())
    if message.header.num_required_signatures != 1 or message.account_keys[0] != owner.pubkey():
        raise ForkUnavailable('fork instruction requested a non-virtual signer')
    # This send_transaction belongs to the LOCAL LiteSVM, never an RPC client.
    return vm.send_transaction(VersionedTransaction(message, [owner]))


def _receipt(result):
    from solders.transaction_metadata import FailedTransactionMetadata
    failed = isinstance(result, FailedTransactionMetadata)
    meta = result.meta() if failed else result
    logs = meta.logs()
    return {'succeeded': not failed, 'error': str(result.err()) if failed else None,
            'compute_units': meta.compute_units_consumed(), 'vm_fee_lamports': meta.fee(),
            'logs': logs, 'logs_sha256': digest(logs)}


def _tokens(vm, mint, token_program):
    from solders.pubkey import Pubkey
    user = str(virtual_owner().pubkey())
    address = associated_token_address(user, mint, token_program)
    value = vm.get_account(Pubkey.from_string(address))
    if value is None:
        return 0
    raw = bytes(value.data)
    if (str(value.owner) != token_program or len(raw) < 165 or raw[:32] != bytes(Pubkey.from_string(mint))
            or raw[32:64] != bytes(virtual_owner().pubkey()) or raw[108] not in (1, 2)):
        raise ForkUnavailable('virtual token account identity/layout mismatch')
    return int.from_bytes(raw[64:72], 'little')


def execute_frame(frame):
    """Execute real program instructions against the post-buy LOCAL bank."""
    from solders.account import Account
    from solders.clock import Clock
    from solders.litesvm import LiteSVM
    from solders.pubkey import Pubkey
    from solders.system_program import transfer
    from solders.instruction import AccountMeta, Instruction
    if frame.get('schema') != SCHEMA or type(frame.get('slot')) is not int or frame['slot'] <= 0:
        raise ForkUnavailable('unrecognized fork frame')
    if len(json.dumps(frame)) > MAX_FRAME_BYTES * 2:
        raise ForkUnavailable('worker input exceeds envelope')
    target, accounts = frame['target'], frame['accounts']
    paper = frame.get('_paper_request')
    if paper is not None and (not isinstance(paper, dict) or paper.get('phase') not in {'entry', 'exit'}):
        raise ForkUnavailable('unrecognized paper operation')
    buy, sell, token_program = _route(target, accounts)
    owner = virtual_owner().pubkey()
    if accounts.get(str(owner)) is not None:
        raise ForkUnavailable('public virtual owner already exists on chain')
    for mint, program in ((target['token'], token_program), (WSOL_MINT, TOKEN_PROGRAM)):
        address = associated_token_address(str(owner), mint, program)
        if address not in accounts or accounts[address] is not None:
            raise ForkUnavailable('virtual ATA missing from snapshot or already exists')
    # Verify code and clock BEFORE creating a bank or executing any SBF.
    images = _program_images(frame)
    required_programs = {buy.program_id, ASSOCIATED_TOKEN_PROGRAM, token_program, TOKEN_PROGRAM}
    if not required_programs <= {row[0] for row in images}:
        raise ForkUnavailable('required on-chain executable code is missing')
    clock_value = accounts.get(CLOCK)
    raw_clock = account_data(clock_value)
    if len(raw_clock) != 40:
        raise ForkUnavailable('clock sysvar layout unsupported')
    slot, epoch_start, epoch, leader_epoch, unix = struct.unpack('<QqQQq', raw_clock)
    if slot != frame['slot'] or unix <= 0:
        raise ForkUnavailable('clock does not match sampled bank')
    vm = LiteSVM().with_mainnet_features().with_transaction_history(0).with_log_bytes_limit(32_768)
    vm.warp_to_slot(frame['slot'])
    vm.set_clock(Clock(slot, epoch_start, epoch, leader_epoch, unix))
    for key, value in accounts.items():
        if value is not None and not value['executable'] and key != CLOCK:
            vm.set_account(Pubkey.from_string(key), Account(value['lamports'], account_data(value),
                Pubkey.from_string(value['owner']), False, value.get('rentEpoch', 2**64 - 1)))
    for key, loader, raw in images:
        vm.add_program_with_loader(Pubkey.from_string(key), raw, Pubkey.from_string(loader))
    if paper is not None and paper['phase'] == 'exit':
        from src.research.paper_bank import execute_exit
        return execute_exit(vm, frame, paper['inventory'], sell, token_program)
    # Sole synthetic state change: virtual payer SOL. Pool, mints and holders
    # are copied, never topped up or patched to make a failed sell pass.
    virtual_funding = max(2_000_000_000, target['quote_budget'] + 100_000_000)
    vm.airdrop(owner, virtual_funding)
    setup = [_ata_instruction(target['token'], token_program), _ata_instruction(WSOL_MINT, TOKEN_PROGRAM)]
    quote_address = Pubkey.from_string(associated_token_address(str(owner), WSOL_MINT, TOKEN_PROGRAM))
    native_sol = target['venue'] == 'pump_curve'
    if not native_sol:
        setup += [transfer({'from_pubkey': owner, 'to_pubkey': quote_address, 'lamports': target['quote_budget']}),
                  Instruction(Pubkey.from_string(TOKEN_PROGRAM), b'\x11', [AccountMeta(quote_address, False, True)])]
    setup_receipt = _receipt(_transact(vm, setup))
    from importlib.metadata import version
    from pathlib import Path
    implementation = implementation_manifest()
    result = {'schema': SCHEMA, 'target': target, 'slot': frame['slot'], 'input_sha256': digest(frame),
        'solders_version': version('solders'),
        'worker_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'implementation': implementation, 'implementation_sha256': digest(implementation),
        'observed_at': frame['observed_at'], 'virtual_owner': str(owner),
        'program_hashes': {key: hashlib.sha256(raw).hexdigest() for key, _, raw in images},
        'scope': 'LOCAL_SVM_AT_SAMPLED_BANK_NOT_FUTURE_SELL_GUARANTEE',
        'runtime_parity': 'bundled mainnet feature set and reconstructed loaders; exact validator parity unverified',
        'quote_balance_measurement': ('native payer lamport change plus VM transaction fee; may include rent refunds'
                                      if native_sol else 'WSOL token-account amount change'),
        'synthetic_funding_lamports': virtual_funding, 'quote_budget_lamports': target['quote_budget'],
        'real_transaction_submitted': False, 'sell_simulation_pass': None, 'setup': setup_receipt}
    if not setup_receipt['succeeded']:
        if paper is not None:
            from src.research.paper_bank import capture_entry
            return capture_entry(vm, frame, result, token_program)
        return {**result, 'status': 'FORK_SETUP_FAILED_UNKNOWN'}
    initial = _tokens(vm, target['token'], token_program)
    quote_before_buy = vm.get_balance(owner) if native_sol else _tokens(vm, WSOL_MINT, TOKEN_PROGRAM)
    buy_receipt = _receipt(_transact(vm, [_instruction(buy)]))
    result['buy'] = buy_receipt
    if not buy_receipt['succeeded']:
        if paper is not None:
            from src.research.paper_bank import capture_entry
            return capture_entry(vm, frame, result, token_program)
        return {**result, 'status': 'FORK_BUY_FAILED_UNKNOWN'}
    received = _tokens(vm, target['token'], token_program) - initial
    quote_after_buy = vm.get_balance(owner) if native_sol else _tokens(vm, WSOL_MINT, TOKEN_PROGRAM)
    result.update(quote_balance_after_setup_units=quote_before_buy, quote_balance_after_buy_units=quote_after_buy)
    if received <= 0:
        if paper is not None:
            from src.research.paper_bank import capture_entry
            return capture_entry(vm, frame, result, token_program)
        return {**result, 'status': 'NO_RECEIVED_TOKEN_BALANCE_UNKNOWN'}
    if paper is not None:
        from src.research.paper_bank import capture_entry
        return capture_entry(vm, frame, result, token_program)
    # Critical: sell ACTUAL received units in this same modified VM.
    before_quote = vm.get_balance(owner) if native_sol else _tokens(vm, WSOL_MINT, TOKEN_PROGRAM)
    sell_receipt = _receipt(_transact(vm, [_instruction(sell(received))]))
    result.update(sell=sell_receipt, bought_token_units=received, sell_input_token_units=received)
    if not sell_receipt['succeeded']:
        # A failing sampled VM is NOT proof that the real token is a honeypot.
        return {**result, 'status': 'FORK_SELL_FAILED', 'sell_simulation_pass': False}
    remaining = _tokens(vm, target['token'], token_program)
    after_quote = vm.get_balance(owner) if native_sol else _tokens(vm, WSOL_MINT, TOKEN_PROGRAM)
    returned = after_quote - before_quote + (sell_receipt['vm_fee_lamports'] if native_sol else 0)
    result['quote_balance_after_sell_units'] = after_quote
    if remaining != initial or returned <= 0:
        return {**result, 'status': 'SELL_BALANCE_CHANGE_INCOMPLETE_UNKNOWN',
                'remaining_token_units': remaining, 'returned_quote_units': returned}
    return {**result, 'status': 'ROUNDTRIP_SUCCEEDED_IN_FORK', 'sell_simulation_pass': True,
            'remaining_token_units': remaining, 'returned_quote_units': returned,
            'cost_adjusted_profit': None, 'real_fill': False}


async def run_isolated(frame, *, timeout=15, paper_request=None):
    if paper_request is not None:
        frame = {**frame, '_paper_request': paper_request}
    # No API keys, wallet environment or network transport enters the VM worker.
    env = {key: os.environ[key] for key in ('SYSTEMROOT', 'WINDIR') if key in os.environ}
    env['PYTHONIOENCODING'] = 'utf-8'
    process = await asyncio.create_subprocess_exec(sys.executable, '-m',
        'src.research.fork_simulation', '--worker', stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, env=env,
        creationflags=0x08000000 if sys.platform == 'win32' else 0)
    try:
        output, _ = await asyncio.wait_for(process.communicate(json.dumps(frame).encode()), timeout)
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise
    if process.returncode or len(output) > 256_000:
        raise ForkUnavailable('local VM worker failed or exceeded result envelope')
    result = json.loads(output)
    if result.get('input_sha256') != digest(frame):
        raise ForkUnavailable('VM result does not bind to supplied snapshot')
    return result


if __name__ == '__main__':
    if sys.argv[1:] != ['--worker']:
        raise SystemExit('Only the isolated, stdin-fed VM worker is supported')
    try:
        raw = sys.stdin.buffer.read(MAX_FRAME_BYTES * 2 + 1)
        if len(raw) > MAX_FRAME_BYTES * 2:
            raise ForkUnavailable('worker input exceeds envelope')
        frame = json.loads(raw)
        result = execute_frame(frame)
        result['input_sha256'] = digest(frame)
    except Exception as exc:
        result = {'status': 'FORK_ENVIRONMENT_UNKNOWN', 'sell_simulation_pass': None,
                  'input_sha256': digest(frame) if 'frame' in locals() else None,
                  'reason': type(exc).__name__ + ': ' + str(exc)[:300], 'real_transaction_submitted': False}
    print(json.dumps(result, allow_nan=False), flush=True)

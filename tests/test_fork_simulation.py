"""VM machinery fixtures are SYNTHETIC; they do not certify a mainnet DEX."""
import base64
import copy
import json
import struct
from types import SimpleNamespace

import pytest
from solders.litesvm import LiteSVM
from solders.pubkey import Pubkey

from src.chains.idl import AccountMeta
from src.chains.pump_route import PreparedInstruction, TOKEN_PROGRAM, WSOL_MINT, ASSOCIATED_TOKEN_PROGRAM, associated_token_address
from src.chains.rpc_manager import RPCError
from src.research import fork_simulation as f
from src.research.fork_probe import ForkProbe


def test_manifest_binds_route_builders_decoder_and_vendored_contracts():
    from pathlib import Path
    import hashlib
    manifest = f.implementation_manifest()
    root = Path(f.__file__).resolve().parents[2]
    for name in ('src/chains/pumpswap_route.py', 'src/chains/pump_route.py',
                 'src/chains/pump_curve.py', 'src/chains/idl.py', 'idl/pump_amm.json', 'idl/FEE_RECIPIENTS.md'):
        assert manifest['source_sha256'][name] == hashlib.sha256((root/name).read_bytes()).hexdigest()


def wire(raw=b'', owner=TOKEN_PROGRAM, lamports=3_000_000, executable=False):
    return {'data': [base64.b64encode(raw).decode(), 'base64'], 'owner': owner,
            'lamports': lamports, 'executable': executable, 'rentEpoch': 2**64 - 1}


@pytest.mark.asyncio
async def test_account_limit_split_is_explicit_and_bank_must_match():
    keys = [str(Pubkey.new_unique()) for _ in range(5)]
    calls = []
    async def request(method, params):
        assert method == 'getMultipleAccounts' and params[1]['commitment'] == 'finalized'
        calls.append(params[0])
        if len(params[0]) > 2:
            raise RPCError({'code': -32602, 'message': 'maximum of 2 accounts'})
        return {'context': {'slot': 77}, 'value': [None] * len(params[0])}
    frame = await f.read_frame(SimpleNamespace(request=request), keys)
    assert set(frame['accounts']) == set(keys) and frame['slot'] == 77
    assert list(map(len, calls)) == [5, 2, 2, 1]
    async def inconsistent(method, params):
        return {'context': {'slot': 77 if params[0][0] == sorted(keys)[0] else 78},
                'value': [None] * len(params[0])}
    with pytest.raises(f.ForkUnavailable, match='different banks'):
        await f.read_frame(SimpleNamespace(request=inconsistent), keys, batch=2)


@pytest.mark.asyncio
async def test_rate_limit_is_not_fanned_out():
    calls = []
    async def request(method, params):
        calls.append(params)
        raise RPCError({'code': 429, 'message': 'maximum of 2 accounts'})
    with pytest.raises(RPCError):
        await f.read_frame(SimpleNamespace(request=request), [str(Pubkey.new_unique()) for _ in range(5)])
    assert len(calls) == 1


def synthetic_frame():
    """Copy actual bundled SPL/ATA SBF; create explicitly synthetic balances."""
    vm = LiteSVM()
    user = f.virtual_owner().pubkey()
    mint, treasury, quote_treasury = [str(Pubkey.new_unique()) for _ in range(3)]
    accounts = {str(user): None}
    for program in (TOKEN_PROGRAM, ASSOCIATED_TOKEN_PROGRAM):
        account = vm.get_account(Pubkey.from_string(program))
        accounts[program] = wire(bytes(account.data), str(account.owner), account.lamports, True)
        if str(account.owner) == f.UPGRADEABLE:
            backing_key = str(Pubkey.from_bytes(bytes(account.data)[4:36]))
            backing = vm.get_account(Pubkey.from_string(backing_key))
            accounts[backing_key] = wire(bytes(backing.data), str(backing.owner), backing.lamports)
    for key, decimals in ((mint, 6), (WSOL_MINT, 9)):
        raw = bytearray(82)
        raw[36:44] = (10**12).to_bytes(8, 'little')
        raw[44:46] = bytes([decimals, 1])
        accounts[key] = wire(raw)
        accounts[associated_token_address(str(user), key, TOKEN_PROGRAM)] = None
    def token(mint_key, amount, native=False):
        raw = bytearray(165)
        raw[:32], raw[32:64] = bytes(Pubkey.from_string(mint_key)), bytes(user)
        raw[64:72], raw[108] = amount.to_bytes(8, 'little'), 1
        if native:
            raw[109:113], raw[113:121] = (1).to_bytes(4, 'little'), (3_000_000).to_bytes(8, 'little')
        return wire(raw, lamports=3_000_000 + (amount if native else 0))
    accounts[treasury] = token(mint, 10**9)
    accounts[quote_treasury] = token(WSOL_MINT, 10**9, True)
    accounts[f.CLOCK] = wire(struct.pack('<QqQQq', 500_000_000, 1000, 0, 0, 1000),
                             owner='Sysvar1111111111111111111111111111111111111')
    frame = {'schema': f.SCHEMA, 'slot': 500_000_000, 'observed_at': 1000., 'accounts': accounts,
             'target': {'token': mint, 'venue': 'pump_curve', 'quote_budget': 1_000_000}}
    return frame, treasury, quote_treasury


def transfer(source, destination, amount):
    return PreparedInstruction('OK', TOKEN_PROGRAM,
        [AccountMeta(source, False, True), AccountMeta(destination, False, True),
         AccountMeta(str(f.virtual_owner().pubkey()), True, False)],
        bytes([3]) + int(amount).to_bytes(8, 'little'))


@pytest.mark.parametrize('broken_sell', [False, True])
@pytest.mark.parametrize('native_sol', [False, True])
def test_real_spl_vm_sells_actual_post_buy_balance(monkeypatch, broken_sell, native_sol):
    frame, treasury, quote_treasury = synthetic_frame()
    if not native_sol:
        frame['target']['venue'] = 'pumpswap'
    mint = frame['target']['token']
    user = str(f.virtual_owner().pubkey())
    ata = associated_token_address(user, mint, TOKEN_PROGRAM)
    requested = []
    def route(target, accounts):
        def sell(amount):
            requested.append(amount)
            return transfer(ata, treasury, amount + (1 if broken_sell else 0))
        return transfer(treasury, ata, 123), sell, TOKEN_PROGRAM
    monkeypatch.setattr(f, '_route', route)
    original = f._transact
    stages = []
    def transact(vm, instructions):
        stages.append(len(instructions))
        if len(stages) == 3:
            # Synthetic exchange repays quote in the same atomic transaction.
            quote_ata = associated_token_address(user, WSOL_MINT, TOKEN_PROGRAM)
            if native_sol:
                # Close an owned native account to return real lamports in
                # this fixture; the receipt explicitly includes rent refunds.
                repay = PreparedInstruction('OK', TOKEN_PROGRAM,
                    [AccountMeta(quote_treasury, False, True), AccountMeta(user, False, True),
                     AccountMeta(user, True, False)], b'\x09')
            else:
                repay = transfer(quote_treasury, quote_ata, 50_000)
            instructions = [*instructions, f._instruction(repay)]
        return original(vm, instructions)
    monkeypatch.setattr(f, '_transact', transact)
    result = f.execute_frame(frame)
    assert result['buy']['succeeded']
    assert result['bought_token_units'] == result['sell_input_token_units'] == 123
    assert requested == [123]  # no assumed quote amount or prefunded base balance
    assert result['sell_simulation_pass'] is (not broken_sell)
    assert result['real_transaction_submitted'] is False
    if broken_sell:
        assert result['status'] == 'FORK_SELL_FAILED'
    else:
        assert result['status'] == 'ROUNDTRIP_SUCCEEDED_IN_FORK'
        assert result['remaining_token_units'] == 0
        assert result['returned_quote_units'] == (1_003_000_000 if native_sol else 50_000)
        assert result['cost_adjusted_profit'] is None


def test_program_loader_binding_and_future_upgrade_rejected():
    frame, _, _ = synthetic_frame()
    assert len(f._program_images(frame)) == 2
    program = frame['accounts'][TOKEN_PROGRAM]
    pointer = str(Pubkey.from_bytes(f.account_data(program)[4:36]))
    backing = frame['accounts'][pointer]
    raw = bytearray(f.account_data(backing))
    raw[4:12] = (500_000_001).to_bytes(8, 'little')
    backing['data'][0] = base64.b64encode(raw).decode()
    with pytest.raises(f.ForkUnavailable, match='future'):
        f._program_images(frame)


@pytest.mark.asyncio
async def test_archival_retry_canonical_ack_and_pit(monkeypatch, tmp_path):
    from src.research import fork_probe as p
    frame, _, _ = synthetic_frame()
    async def acquire(rpc, target):
        return frame
    async def run(snapshot):
        assert snapshot == frame
        return {'status': 'ROUNDTRIP_SUCCEEDED_IN_FORK', 'sell_simulation_pass': True,
                'observed_at': frame['observed_at']}
    monkeypatch.setattr(p, 'acquire', acquire)
    monkeypatch.setattr(p, 'run_isolated', run)
    probe = ForkProbe(tmp_path/'fork.sqlite')
    target = frame['target']
    rows = await probe.fetch(None, lambda: [target])
    assert probe.latest(target['token'], as_of=10**12) is None
    assert probe._read_frame(rows[0]['input_sha256']) == frame
    assert probe.db.execute('SELECT count(*) FROM blobs').fetchone()[0] >= 1
    probe.close()
    probe = ForkProbe(tmp_path/'fork.sqlite')
    assert await probe.fetch(None, lambda: []) == rows
    altered = copy.deepcopy(rows)
    altered[0]['sell_simulation_pass'] = False
    with pytest.raises(ValueError):
        probe.acknowledge(altered, 'canonical:x')
    before_ack = p.time.time() - 1
    probe.acknowledge(rows, 'canonical:x')
    assert probe.latest(target['token'], as_of=before_ack) is None
    assert probe.latest(target['token'], as_of=10**12)['source_ref'] == 'canonical:x'
    probe.db.execute("UPDATE frames SET body=? WHERE id=?", (b'corrupt', rows[0]['input_sha256']))
    with pytest.raises(Exception):
        probe._read_frame(rows[0]['input_sha256'])
    probe.close()


@pytest.mark.asyncio
async def test_worker_environment_failure_is_unknown():
    result = await f.run_isolated({'schema': f.SCHEMA, 'slot': 1})
    assert result['status'] == 'FORK_ENVIRONMENT_UNKNOWN'
    assert result['sell_simulation_pass'] is None and result['real_transaction_submitted'] is False


def test_curve_adapter_uses_observed_quote_mint_flags_and_owner():
    from src.chains.pump_curve import BONDING_CURVE_DISCRIMINATOR
    mint, creator = str(Pubkey.new_unique()), str(Pubkey.new_unique())
    raw_mint = bytearray(82)
    raw_mint[44:46] = b'\x06\x01'
    curve = (BONDING_CURVE_DISCRIMINATOR + struct.pack('<QQQQQ', 10**15, 30*10**9, 8*10**14, 10**9, 10**15)
             + b'\x00' + bytes(Pubkey.from_string(creator)) + b'\x00\x00' + bytes(Pubkey.from_string(WSOL_MINT)))
    key = f._curve_address(mint)
    accounts = {mint: wire(raw_mint), key: wire(curve, f.PUMP_PROGRAM)}
    target = {'token': mint, 'venue': 'pump_curve', 'quote_budget': 1_000_000}
    buy, sell, program = f._route(target, accounts)
    assert buy.ok and program == TOKEN_PROGRAM and sell(123).ok
    assert int.from_bytes(sell(123).data[8:16], 'little') == 123
    # Per Pump's published compatibility rule, missing trailing fields are
    # zero/default; default quote mint means SOL, not an unsupported asset.
    for valid in (curve[:49], curve[:81], curve[:82], curve[:83] + bytes(32), curve):
        accounts[key] = wire(valid, f.PUMP_PROGRAM)
        assert f._route(target, accounts)[0].ok
    for altered in (curve[:83] + bytes(Pubkey.new_unique()), curve[:81] + b'\x02' + curve[82:]):
        accounts[key] = wire(altered, f.PUMP_PROGRAM)
        with pytest.raises(f.ForkUnavailable):
            f._route(target, accounts)
    accounts[key] = wire(curve, TOKEN_PROGRAM)
    with pytest.raises(f.ForkUnavailable, match='owner'):
        f._route(target, accounts)


def test_pumpswap_adapter_binds_reserve_mints_and_authorities():
    from src.chains.pumpswap_route import _POOL_LAYOUT, POOL_DISCRIMINATOR, PUMPSWAP_PROGRAM, derive_pool
    mint, creator, base_reserve, quote_reserve = [str(Pubkey.new_unique()) for _ in range(4)]
    pool = derive_pool(1, creator, mint, WSOL_MINT)
    values = {'index': 1, 'creator': creator, 'base_mint': mint, 'quote_mint': WSOL_MINT,
              'pool_base_token_account': base_reserve, 'pool_quote_token_account': quote_reserve,
              'coin_creator': creator}
    raw_pool = bytearray(POOL_DISCRIMINATOR)
    for name, width, kind in _POOL_LAYOUT:
        raw_pool.extend(bytes(Pubkey.from_string(values.get(name, creator))) if kind == 'pubkey'
                        else int(values.get(name, 0)).to_bytes(width, 'little'))
    raw_mint = bytearray(82)
    raw_mint[44:46] = b'\x06\x01'
    def reserve(token, amount):
        raw = bytearray(165)
        raw[:32], raw[32:64] = bytes(Pubkey.from_string(token)), bytes(Pubkey.from_string(pool))
        raw[64:72], raw[108] = amount.to_bytes(8, 'little'), 1
        return wire(raw)
    accounts = {mint: wire(raw_mint), WSOL_MINT: wire(raw_mint), pool: wire(raw_pool, PUMPSWAP_PROGRAM),
                base_reserve: reserve(mint, 10**12), quote_reserve: reserve(WSOL_MINT, 30*10**9)}
    target = {'token': mint, 'venue': 'pumpswap', 'pool': pool, 'quote_budget': 1_000_000}
    buy, sell, _ = f._route(target, accounts)
    assert buy.ok and sell(123).ok
    raw = bytearray(f.account_data(accounts[base_reserve]))
    raw[32:64] = bytes(Pubkey.new_unique())
    accounts[base_reserve] = wire(raw)
    with pytest.raises(f.ForkUnavailable, match='authority'):
        f._route(target, accounts)


@pytest.mark.asyncio
async def test_queue_has_no_top_n_and_fresh_targets_bypass_backlog_without_starving_old(monkeypatch, tmp_path):
    from src.research import fork_probe as p
    async def unavailable(rpc, target):
        raise f.ForkUnavailable('synthetic unavailable fixture')
    monkeypatch.setattr(p, 'acquire', unavailable)
    probe = ForkProbe(tmp_path/'fork.sqlite')
    targets = [{'token': str(Pubkey.new_unique()), 'venue': 'pump_curve', 'quote_budget': 1_000_000} for _ in range(1100)]
    rows = await probe.fetch(None, lambda: targets)
    probe.acknowledge(rows, 'canonical:first')
    assert probe.report()['targets'] == 1100
    new = {'token': str(Pubkey.new_unique()), 'venue': 'pump_curve', 'quote_budget': 1_000_000}
    rows2 = await probe.fetch(None, lambda: [new])
    assert rows2[0]['token'] == new['token']
    assert rows2[0]['dispatch_lane'] == 'FRESH_UNTRIED'
    assert rows2[0]['sell_simulation_pass'] is None
    probe.acknowledge(rows2, 'canonical:new')
    rows3 = await probe.fetch(None, lambda: [])
    assert rows3[0]['token'] == targets[0]['token']
    assert rows3[0]['dispatch_lane'] == 'OLDEST_UNTRIED'
    assert probe.report()['targets'] == 1101
    assert probe.report()['by_venue']['pump_curve']['untried_targets'] == 1098
    probe.close()


@pytest.mark.asyncio
async def test_due_retries_keep_their_share_under_fresh_arrivals_across_restart(monkeypatch,tmp_path):
    from src.research import fork_probe as p
    async def unavailable(rpc,target):raise f.ForkUnavailable('synthetic unavailable fixture')
    monkeypatch.setattr(p,'acquire',unavailable)
    path=tmp_path/'fork.sqlite'
    probe=ForkProbe(path)
    targets=[{'token':str(Pubkey.new_unique()),'venue':'pump_curve','quote_budget':1_000_000} for _ in range(5)]
    first=await probe.fetch(None,lambda:targets[:3])
    assert first[0]['token']==targets[2]['token']
    probe.acknowledge(first,'canonical:first')
    with probe.db:probe.db.execute('UPDATE jobs SET due=0 WHERE attempts>0')
    fresh=await probe.fetch(None,lambda:[targets[3]])
    assert fresh[0]['token']==targets[3]['token']
    probe.acknowledge(fresh,'canonical:fresh')
    oldest=await probe.fetch(None,lambda:[])
    assert oldest[0]['token']==targets[0]['token']
    probe.acknowledge(oldest,'canonical:oldest')
    probe.close()
    probe=ForkProbe(path)
    retry=await probe.fetch(None,lambda:[targets[4]])
    assert retry[0]['token']==targets[2]['token'] and retry[0]['dispatch_lane']=='DUE_RETRY'
    probe.acknowledge(retry,'canonical:retry')
    next_fresh=await probe.fetch(None,lambda:[])
    assert next_fresh[0]['token']==targets[4]['token']
    assert probe.report()['targets']==5
    probe.close()


@pytest.mark.asyncio
async def test_legacy_queue_migration_preserves_jobs_and_does_not_invent_arrival_times(monkeypatch,tmp_path):
    import json
    import sqlite3
    from src.research import fork_probe as p
    async def unavailable(rpc,target):raise f.ForkUnavailable('synthetic unavailable fixture')
    monkeypatch.setattr(p,'acquire',unavailable)
    path=tmp_path/'fork.sqlite'
    targets=[{'token':str(Pubkey.new_unique()),'venue':'pump_curve','quote_budget':1_000_000} for _ in range(2)]
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE jobs(id TEXT PRIMARY KEY,body TEXT NOT NULL,due REAL NOT NULL,attempts INTEGER NOT NULL)')
        for target in targets:
            db.execute('INSERT INTO jobs VALUES(?,?,0,0)',(p.digest(target),json.dumps(target)))
    probe=ForkProbe(path)
    rows=await probe.fetch(None,lambda:[])
    assert rows[0]['token']==targets[1]['token']
    assert rows[0]['target_first_queued_at'] is None and rows[0]['queue_age_at_attempt_seconds'] is None
    assert probe.report()['targets']==2
    cursor=probe.db.execute('SELECT cursor FROM fork_dispatch').fetchone()[0]
    probe.close()
    probe=ForkProbe(path)
    assert await probe.fetch(None,lambda:[])==rows  # Canonical ACK still fences new work.
    assert probe.db.execute('SELECT cursor FROM fork_dispatch').fetchone()[0]==cursor
    probe.close()


@pytest.mark.asyncio
async def test_venue_report_counts_only_acknowledged_dispositions(monkeypatch,tmp_path):
    from src.research import fork_probe as p
    async def unavailable(rpc,target):raise f.ForkUnavailable('synthetic unavailable fixture')
    monkeypatch.setattr(p,'acquire',unavailable)
    probe=ForkProbe(tmp_path/'fork.sqlite')
    targets=[{'token':str(Pubkey.new_unique()),'venue':venue,'quote_budget':1_000_000}
             for venue in ('pump_curve','pumpswap','raydium_launchlab')]
    rows=await probe.fetch(None,lambda:targets)
    before=probe.report()['by_venue']
    assert len(before)==3 and not any(row['canonical_dispositions'] for row in before.values())
    assert before['raydium_launchlab']['attempts']==1 and before['pump_curve']['untried_targets']==1
    probe.acknowledge(rows,'canonical:venue-unknown')
    assert probe.report()['by_venue']['raydium_launchlab']['canonical_dispositions']=={'FORK_ENVIRONMENT_UNKNOWN':1}
    assert probe.report()['by_venue']['pump_curve']['canonical_dispositions']=={}
    probe.close()


@pytest.mark.asyncio
async def test_arrival_priority_survives_wall_clock_correction_and_labels_queue_age(monkeypatch,tmp_path):
    from src.research import fork_probe as p
    tick=[100.]
    monkeypatch.setattr(p.time,'time',lambda:tick[0])
    async def unavailable(rpc,target):raise f.ForkUnavailable('synthetic unavailable fixture')
    monkeypatch.setattr(p,'acquire',unavailable)
    probe=ForkProbe(tmp_path/'fork.sqlite')
    targets=[{'token':str(Pubkey.new_unique()),'venue':'pump_curve','quote_budget':1_000_000} for _ in range(3)]
    rows=await probe.fetch(None,lambda:targets[:2]);probe.acknowledge(rows,'canonical:first')
    with probe.db:probe.db.execute('UPDATE jobs SET due=0 WHERE attempts=0')
    tick[0]=50.
    fresh=await probe.fetch(None,lambda:[targets[2]])
    assert fresh[0]['token']==targets[2]['token']
    probe.acknowledge(fresh,'canonical:fresh')
    old=await probe.fetch(None,lambda:[])
    assert old[0]['token']==targets[0]['token']
    assert old[0]['queue_clock_regressed'] is True and old[0]['queue_age_at_attempt_seconds']==0
    probe.close()


@pytest.mark.asyncio
async def test_empty_untried_lanes_share_capacity_with_due_retries(monkeypatch,tmp_path):
    from src.research import fork_probe as p
    async def unavailable(rpc,target):raise f.ForkUnavailable('synthetic unavailable fixture')
    monkeypatch.setattr(p,'acquire',unavailable)
    probe=ForkProbe(tmp_path/'fork.sqlite')
    target={'token':str(Pubkey.new_unique()),'venue':'pump_curve','quote_budget':1_000_000}
    first=await probe.fetch(None,lambda:[target]);probe.acknowledge(first,'canonical:first')
    for index in range(4):
        with probe.db:probe.db.execute('UPDATE jobs SET due=0')
        rows=await probe.fetch(None,lambda:[])
        assert rows[0]['token']==target['token'] and rows[0]['dispatch_lane']=='DUE_RETRY'
        probe.acknowledge(rows,f'canonical:retry:{index}')
    assert probe.report()['attempts']==5
    probe.close()


def test_runtime_retains_result_before_acknowledging():
    from src.runtime.research_runtime import ResearchRuntime
    called = []
    def record(kind, payload, now):
        called.append('retain')
        return 'canonical:fork'
    def ack(records, ref):
        assert called == ['retain'] and ref == 'canonical:fork'
        called.append('ack')
    bot = SimpleNamespace(unknown_engine=SimpleNamespace(_record=record),
                          fork_probe=SimpleNamespace(acknowledge=ack))
    ref = ResearchRuntime._retain_mined_batch(bot, 'chain:local_fork_roundtrip', [{'token': 'synthetic'}], 1000)
    assert ref == 'canonical:fork' and called == ['retain', 'ack']


@pytest.mark.asyncio
async def test_real_miner_transport_stamps_do_not_block_canonical_ack(monkeypatch, tmp_path):
    from src.research import fork_probe as p
    from src.research.data_miners import DataMinerPool
    async def unavailable(rpc, target):
        raise f.ForkUnavailable('synthetic provider diagnostic')
    monkeypatch.setattr(p, 'acquire', unavailable)
    probe = ForkProbe(tmp_path/'fork.sqlite')
    delivered = []
    def retain(miner, rows, now):
        assert miner == p.MINER and rows[0]['_miner'] == p.MINER
        assert rows[0]['_request_started_at'] <= rows[0]['_fetched_at']
        probe.acknowledge(rows, 'canonical:stamped')
        return 'canonical:stamped'
    pool = DataMinerPool(retain_batch=retain, on_records=lambda miner, rows: delivered.extend(rows))
    target = {'token': str(Pubkey.new_unique()), 'venue': 'pump_curve', 'quote_budget': 1_000_000}
    probe.register(pool, None, lambda: [target])
    result = await pool._run_one(p.MINER, p.time.time())
    assert result.status == 'OK' and len(delivered) == 1
    assert probe.report()['awaiting_canonical_ack'] == 0
    assert probe.latest(target['token'], as_of=10**12)['source_ref'] == 'canonical:stamped'
    # Only the declared transport envelope is ignored. New/tampered body
    # fields and another miner's identity still fail closed.
    for altered in ({**delivered[0], 'invented': True}, {**delivered[0], '_miner': 'untrusted'}):
        with pytest.raises(ValueError):
            probe.acknowledge([altered], 'canonical:forged')
    probe.close()

"""Real local SPL execution on SYNTHETIC markets; no mainnet profitability claim."""
import copy
import json
import struct

import pytest
from solders.pubkey import Pubkey
from solders.system_program import transfer as sol_transfer

from src.chains.pump_route import TOKEN_PROGRAM, WSOL_MINT, associated_token_address
from src.research import fork_simulation as f, fork_probe as probe_module, fork_paper
from test_fork_simulation import synthetic_frame, transfer, wire


def market(monkeypatch):
    frame, treasury, quote_treasury = synthetic_frame()
    owner = str(f.virtual_owner().pubkey())
    ata = associated_token_address(owner, frame['target']['token'], TOKEN_PROGRAM)
    cash = associated_token_address(owner, WSOL_MINT, TOKEN_PROGRAM)
    price = {'repayment': 50_000, 'broken': False}
    def route(target, accounts):
        return transfer(treasury, ata, 123), lambda amount: transfer(
            ata, treasury, amount + int(price['broken'])), TOKEN_PROGRAM
    monkeypatch.setattr(f, '_route', route)
    original = f._transact
    def transact(vm, instructions):
        if len(instructions)==1 and instructions[0].program_id==Pubkey.from_string(TOKEN_PROGRAM):
            if str(instructions[0].accounts[0].pubkey)==treasury:
                instructions = [*instructions, sol_transfer({'from_pubkey': f.virtual_owner().pubkey(),
                    'to_pubkey': Pubkey.from_string(quote_treasury),'lamports':700_000})]
            else:
                instructions = [*instructions, f._instruction(transfer(quote_treasury,cash,price['repayment']))]
        return original(vm,instructions)
    monkeypatch.setattr(f, '_transact', transact)
    return frame, price


def later(frame, *, at=1061., slot=500_000_001):
    result = copy.deepcopy(frame)
    result.update(slot=slot,observed_at=at)
    result['accounts'][f.CLOCK]=wire(struct.pack('<QqQQq',slot,1000,0,0,int(at)),
                                    owner=frame['accounts'][f.CLOCK]['owner'])
    return result


def operation(frame,phase,inventory=None):
    return f.execute_frame({**frame,'_paper_request':{'phase':phase,**({'inventory':inventory} if inventory else {})}})


def test_later_bank_uses_retained_actual_units_and_changes_public_market_proceeds(monkeypatch):
    frame,price=market(monkeypatch)
    entry=operation(frame,'entry')
    assert entry['status']=='PAPER_BANK_OPENED'
    held=json.loads(json.dumps(entry['inventory']))  # exact disk round trip
    assert held['entry_token_units']==123 and held['virtual_funding_lamports']==2_000_000_000
    low=operation(later(frame),'exit',held)
    assert low['status']=='PAPER_BANK_CLOSED' and low['sell_input_token_units']==123
    assert low['remaining_token_units']==0
    price['repayment']=250_000
    high=operation(later(frame),'exit',held)
    assert high['conditional_net_liquid_pnl_lamports']-low['conditional_net_liquid_pnl_lamports']==200_000
    assert low['conditional_net_liquid_pnl_lamports']<0  # includes entry, setup rent and all VM fees
    assert low['vm_fees_lamports']==held['vm_fees_lamports']+low['exit']['vm_fee_lamports']
    assert not low['real_fill'] and not low['promotion_eligible']
    assert low['entry_frame_sha256']==f.digest(frame) and low['exit_frame_sha256']==f.digest(later(frame))
    assert held==entry['inventory']  # exit never mutates the archived entry state


def test_failed_exit_retains_inventory_and_actual_fee_debit_for_later_retry(monkeypatch):
    frame,price=market(monkeypatch)
    entry=operation(frame,'entry');price['broken']=True
    failed=operation(later(frame),'exit',entry['inventory'])
    assert failed['status']=='PAPER_BANK_EXIT_UNRESOLVED'
    assert failed['remaining_token_units']==123 and failed['conditional_net_liquid_pnl_lamports'] is None
    owner=str(f.virtual_owner().pubkey())
    debit=entry['inventory']['accounts'][owner]['lamports']-failed['inventory']['accounts'][owner]['lamports']
    assert debit==failed['exit']['vm_fee_lamports']>0
    price['broken']=False
    final=operation(later(frame,at=1122.,slot=500_000_002),'exit',failed['inventory'])
    direct=operation(later(frame,at=1122.,slot=500_000_002),'exit',entry['inventory'])
    assert final['status']=='PAPER_BANK_CLOSED'
    assert direct['conditional_net_liquid_pnl_lamports']-final['conditional_net_liquid_pnl_lamports']==debit


@pytest.mark.parametrize('mutation', ['same_bank','same_clock','altered_seal','market_overlay','wrong_token','public_collision'])
def test_untrusted_or_nonprospective_inventory_cannot_execute(monkeypatch,mutation):
    frame,_=market(monkeypatch);entry=operation(frame,'entry')['inventory'];fresh=later(frame)
    if mutation=='same_bank':fresh['slot']=frame['slot'];fresh['accounts'][f.CLOCK]=frame['accounts'][f.CLOCK]
    if mutation=='same_clock':fresh['observed_at']=frame['observed_at']
    if mutation=='altered_seal':entry['token_units']+=1
    if mutation=='market_overlay':
        entry['accounts'][frame['target']['token']]=frame['accounts'][frame['target']['token']]
        entry['seal']=f.digest({k:v for k,v in entry.items() if k!='seal'})
    if mutation=='wrong_token':entry['target']['token']=str(Pubkey.new_unique())
    if mutation=='public_collision':fresh['accounts'][str(f.virtual_owner().pubkey())]=wire(owner=f.SYSTEM)
    with pytest.raises(f.ForkUnavailable):operation(fresh,'exit',entry)


@pytest.mark.asyncio
async def test_pipeline_atomically_retains_inventory_restarts_and_requires_canonical_ack(monkeypatch,tmp_path):
    frame,_=market(monkeypatch);clock=[1000.]
    monkeypatch.setattr(probe_module.time,'time',lambda:clock[0])
    async def acquire(rpc,target):return copy.deepcopy(frame)
    async def roundtrip(snapshot):return {'status':'SYNTHETIC_TEST_ROUNDTRIP_PLACEHOLDER'}
    async def execute(snapshot,*,paper_request):return operation(snapshot,paper_request['phase'],paper_request.get('inventory'))
    monkeypatch.setattr(probe_module,'acquire',acquire)
    monkeypatch.setattr(probe_module,'run_isolated',roundtrip)
    monkeypatch.setattr(fork_paper,'run_isolated',execute)
    path=tmp_path/'probe.sqlite';probe=probe_module.ForkProbe(path,paper_enabled=True)
    opened=await probe.fetch(None,lambda:[frame['target']])
    assert opened[0]['paper_control']['execution']['status']=='PAPER_BANK_OPENED'
    assert probe.report()['prospective_local_bank']['canonical_open_cohorts']==0
    assert probe.report()['prospective_local_bank']['pending_canonical_ack']==1
    probe.close();clock[0]=1061.;frame=later(frame)
    probe=probe_module.ForkProbe(path,paper_enabled=True)
    assert await probe.fetch(None,lambda:[])==opened  # durable redelivery, no repeated buy
    probe.acknowledge(opened,'canonical:paper-entry')
    assert probe.report()['prospective_local_bank']['canonical_open_cohorts']==1
    closed=await probe.fetch(None,lambda:[])
    control=closed[0]['paper_control']
    assert control['previous_canonical_ref']=='canonical:paper-entry'
    assert control['execution']['status']=='PAPER_BANK_CLOSED'
    assert probe.report()['prospective_local_bank']['canonical_closed_cohorts']==0
    probe.acknowledge(closed,'canonical:paper-exit')
    assert probe.report()['prospective_local_bank']['canonical_closed_cohorts']==1
    assert probe.report()['prospective_local_bank']['real_fills']==0
    probe.close()


@pytest.mark.asyncio
async def test_preexisting_targets_not_retroactively_enrolled(monkeypatch,tmp_path):
    frame,_=market(monkeypatch);clock=[900.]
    monkeypatch.setattr(probe_module.time,'time',lambda:clock[0])
    probe=probe_module.ForkProbe(tmp_path/'probe.sqlite')
    with probe.lock,probe.db:probe._enqueue_targets_locked([frame['target']],clock[0])
    probe.close();clock[0]=1000.
    probe=probe_module.ForkProbe(tmp_path/'probe.sqlite',paper_enabled=True)
    with probe.lock,probe.db:job=probe._reserve_job(clock[0])
    assert await probe.paper_controls.prepare(frame,job) is None
    probe.close()


def test_failed_entry_is_a_terminal_costed_trial_not_a_free_replenished_retry(monkeypatch):
    frame,_=market(monkeypatch)
    original=f._route
    def bad_route(target,accounts):
        buy,sell,program=original(target,accounts)
        return transfer(buy.accounts[0].pubkey,buy.accounts[1].pubkey,10**15),sell,program
    monkeypatch.setattr(f,'_route',bad_route)
    result=operation(frame,'entry')
    assert result['status']=='PAPER_BANK_ENTRY_FAILED'
    assert result['inventory']['token_units']==0
    assert result['conditional_net_liquid_pnl_lamports']<0
    assert result['buy']['vm_fee_lamports']>0


@pytest.mark.asyncio
async def test_inventory_and_outbox_rollback_together_and_corruption_cannot_resume(monkeypatch,tmp_path):
    import sqlite3
    frame,_=market(monkeypatch);clock=[1000.]
    monkeypatch.setattr(probe_module.time,'time',lambda:clock[0])
    async def acquire(rpc,target):return copy.deepcopy(frame)
    async def roundtrip(snapshot):return {'status':'SYNTHETIC_TEST_ROUNDTRIP_PLACEHOLDER'}
    async def execute(snapshot,*,paper_request):return operation(snapshot,paper_request['phase'],paper_request.get('inventory'))
    monkeypatch.setattr(probe_module,'acquire',acquire);monkeypatch.setattr(probe_module,'run_isolated',roundtrip)
    monkeypatch.setattr(fork_paper,'run_isolated',execute)
    probe=probe_module.ForkProbe(tmp_path/'probe.sqlite',paper_enabled=True)
    probe.db.execute("CREATE TRIGGER fail_inventory BEFORE INSERT ON paper_bank_inventory BEGIN SELECT RAISE(ABORT,'synthetic disk failure'); END")
    with pytest.raises(sqlite3.IntegrityError):await probe.fetch(None,lambda:[frame['target']])
    assert probe.db.execute('SELECT count(*) FROM results').fetchone()[0]==0
    assert probe.db.execute('SELECT count(*) FROM paper_bank_inventory').fetchone()[0]==0
    probe.db.execute('DROP TRIGGER fail_inventory');probe.db.execute('UPDATE jobs SET due=1000');probe.db.commit()
    rows=await probe.fetch(None,lambda:[]);probe.acknowledge(rows,'canonical:original')
    state=json.loads(probe.db.execute('SELECT body FROM paper_bank_inventory').fetchone()[0])
    state['virtual_funding_lamports']+=1
    probe.db.execute('UPDATE paper_bank_inventory SET body=?',(json.dumps(state),));probe.db.commit()
    clock[0]=1061.;frame=later(frame)
    next_rows=await probe.fetch(None,lambda:[])
    assert next_rows[0]['paper_control']['status']=='PAPER_CONTROL_UNAVAILABLE'
    assert next_rows[0]['paper_control']['error_type']=='ForkUnavailable'
    assert next_rows[0]['status']=='SYNTHETIC_TEST_ROUNDTRIP_PLACEHOLDER'
    probe.close()

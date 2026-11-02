import hashlib,json
from pathlib import Path
import pytest
import ops.canonical_recovery_tick as module
from ops.canonical_fragment_progress import record_fragment_step
from ops.incremental_backup import RESERVE
from src.research.trade_evidence import TradeEvidenceLedger,GENESIS
from test_canonical_raw_chunk import Timer

BASE=dict(offset=0,sequence=0,last_hash=GENESIS)
EVIDENCE='a'*64
CPU=lambda:dict(status='VERIFIED',cpu_rate=200)


def fixture(tmp_path):
    root=tmp_path/'staging';root.mkdir();src=tmp_path/'source';src.mkdir();source=src/'canonical.jsonl'
    writer=TradeEvidenceLedger(source);writer.record('fixture',{'first':True});first=source.read_bytes()
    for i in range(7):writer.record('fixture',{'n':i})
    raw=root/'seed.raw.pending';raw.write_bytes(first);extent=source.stat().st_size
    receipt=dict(status='SINGLE_RAW_CANONICAL_RANGE_READBACK_VERIFIED',pending_path=str(raw),offset=0,
        end_offset=len(first),bytes=len(first),captured_extent=extent,sha256=hashlib.sha256(first).hexdigest())
    path=root/'history.fragment-progress.json';timer=Timer()
    r=record_fragment_step(path,anchor=BASE,anchor_evidence_sha256=EVIDENCE,target_extent=extent,receipts=[receipt],
        expected_progress_sha256=None,continue_check=lambda:True,resource_check=CPU,free_bytes=lambda:RESERVE+4000000,
        clock=timer.clock,sleep=timer.sleep)
    return source,path,root/'work.fragment-tick.json',r['progress_sha256'],extent


def run(p,**kw):
    source,path,tick,pin,extent=p;timer=kw.pop('timer',Timer())
    return module.run_fragment_tick(source,path,kw.pop('tick',tick),anchor=BASE,anchor_evidence_sha256=EVIDENCE,
        target_extent=extent,expected_progress_sha256=kw.pop('pin',pin),continue_check=kw.pop('continue_check',lambda:True),
        max_records=kw.pop('max_records',1),resource_check=kw.pop('resource_check',CPU),
        free_bytes=kw.pop('free_bytes',lambda:RESERVE+4000000),clock=timer.clock,sleep=timer.sleep,**kw)


def test_four_real_steps_checkpoint_exact_pins_and_copy_budget(tmp_path):
    p=fixture(tmp_path);source=p[0].read_bytes();r=run(p)
    assert r['status']=='STEP_BUDGET_REACHED' and r['checkpointed_steps']==4
    assert r['last_checkpointed_cursor']['sequence']==5 and 0<r['source_copy_bytes_confirmed']<1048576
    assert hashlib.sha256(p[1].read_bytes()).hexdigest()==r['last_checkpointed_progress_sha256']
    journal=json.loads(p[2].read_bytes());assert len(journal['steps'])==4 and 'in_flight' not in journal
    assert journal['steps'][0]['prior_progress_sha256']==p[3]
    for a,b in zip(journal['steps'],journal['steps'][1:]):assert a['progress_sha256']==b['prior_progress_sha256']
    assert hashlib.sha256(p[2].read_bytes()).hexdigest()==r['tick_sha256']
    assert p[0].read_bytes()==source and not r['recovery_verified'] and not r['deployment_backup_accepted']


def test_source_byte_budget_stops_without_second_call_or_data_loss(tmp_path):
    p=fixture(tmp_path);saved=p[1].read_bytes();r=run(p,max_new_bytes=1)
    assert r['status']=='SOURCE_COPY_BUDGET_REACHED' and r['source_copy_bytes_confirmed']==1
    assert r['last_checkpointed_progress_sha256']==p[3] and p[1].read_bytes()==saved
    assert r['checkpointed_steps']==1 and len(list(p[1].parent.glob('step-*.raw.pending')))==1


def test_existing_journal_and_tick_lock_preserved(tmp_path):
    p=fixture(tmp_path);p[2].write_bytes(b'prior');saved=p[1].read_bytes()
    with pytest.raises(FileExistsError):run(p)
    assert p[2].read_bytes()==b'prior' and p[1].read_bytes()==saved
    lock=p[1].parent/'canonical-tick.lock';lock.write_bytes(b'other writer')
    with pytest.raises(FileExistsError):run(p,tick=p[2].with_name('new.fragment-tick.json'))
    assert lock.read_bytes()==b'other writer'


def test_admission_revoked_between_steps_retains_completed_pin(tmp_path,monkeypatch):
    p=fixture(tmp_path);original=module.advance_fragment_step;calls=[]
    def advance(*a,**kw):
        result=original(*a,**kw);calls.append(1);return result
    monkeypatch.setattr(module,'advance_fragment_step',advance)
    r=run(p,continue_check=lambda:not calls)
    assert r['status']=='ADMISSION_STOPPED' and r['checkpointed_steps']==1 and len(calls)==1
    assert hashlib.sha256(p[1].read_bytes()).hexdigest()==r['last_checkpointed_progress_sha256']


def test_checkpoint_failure_keeps_inflight_marker_and_does_not_guess_new_pin(tmp_path,monkeypatch):
    p=fixture(tmp_path);original=module._publish
    def publish(path,value,expected):
        if value.get('state')=='STEP_CHECKPOINTED':raise OSError('fixture checkpoint failure')
        return original(path,value,expected)
    monkeypatch.setattr(module,'_publish',publish)
    r=run(p);journal=json.loads(p[2].read_bytes())
    assert r['status']=='RECONCILIATION_REQUIRED' and r['last_checkpointed_progress_sha256']==p[3]
    assert r['checkpointed_steps']==0 and r['unresolved_step_copy_upper_bound']==1048576
    assert journal['steps']==[] and journal['in_flight']['progress_sha256']==p[3]
    assert hashlib.sha256(p[1].read_bytes()).hexdigest()!=p[3]
    assert not (p[1].parent/'canonical-tick.lock').exists()


def test_exception_after_real_step_does_not_auto_retry_or_roll_back(tmp_path,monkeypatch):
    p=fixture(tmp_path);original=module.advance_fragment_step;calls=[]
    def advance(*a,**kw):
        original(*a,**kw);calls.append(1);raise OSError('fixture lost result after publication')
    monkeypatch.setattr(module,'advance_fragment_step',advance)
    r=run(p)
    assert r['status']=='RECONCILIATION_REQUIRED' and len(calls)==1 and r['checkpointed_steps']==0
    assert hashlib.sha256(p[1].read_bytes()).hexdigest()!=p[3]
    assert 'in_flight' in json.loads(p[2].read_bytes())


@pytest.mark.parametrize('policy',[{'max_steps':True},{'max_steps':5},{'max_new_bytes':2097153},
    {'max_new_bytes':0},{'max_records':257},{'deadline_seconds':61},{'resource_check':lambda:dict(status='VERIFIED',cpu_rate=500)},
    {'free_bytes':lambda:RESERVE},{'continue_check':lambda:False},{'pin':'f'*64}])
def test_policy_and_resource_gates_precede_work(tmp_path,policy):
    p=fixture(tmp_path);saved=p[1].read_bytes()
    with pytest.raises((ValueError,RuntimeError)):run(p,**policy)
    assert p[1].read_bytes()==saved and not p[2].exists()


def test_capture_tip_and_explicit_link_are_not_silently_bypassed(tmp_path):
    p=fixture(tmp_path);r=run(p,max_records=256)
    assert r['status']=='CAPTURE_TIP_HISTORY_ONLY' and r['last_checkpointed_cursor']['sequence']==8
    with pytest.raises(ValueError,match='Exact sibling rollover link'):
        run(p,pin=r['last_checkpointed_progress_sha256'],tick=p[2].with_name('next.fragment-tick.json'),
            rollover_link=p[2].with_name('wrong.rollover.json'))


from test_canonical_fragment_rollover import full


def test_full_history_limit_stops_before_any_step_or_rollover(tmp_path,full,monkeypatch):
    p=fixture(tmp_path);raw,state,pin=full;p[0].write_bytes(raw);p[1].write_bytes(state)
    (p[1].parent/'input.raw.pending').write_bytes(raw)
    prepared=(p[0],p[1],p[2],pin,len(raw))
    def no_step(*a,**k):raise AssertionError('full history must stop before work')
    monkeypatch.setattr(module,'advance_fragment_step',no_step)
    r=run(prepared)
    assert r['status']=='HISTORY_LIMIT_REACHED' and r['checkpointed_steps']==0
    assert r['source_copy_bytes_confirmed']==0 and p[1].read_bytes()==state
    assert not list(p[1].parent.glob('*.rollover.json'))


def test_deadline_after_step_keeps_checkpoint_and_stops_next_call(tmp_path,monkeypatch):
    p=fixture(tmp_path);timer=Timer();original=module.advance_fragment_step;calls=[]
    def advance(*a,**kw):
        result=original(*a,**kw);calls.append(1);timer.sleep(61);return result
    monkeypatch.setattr(module,'advance_fragment_step',advance)
    r=run(p,timer=timer)
    assert r['status']=='ADMISSION_STOPPED' and r['checkpointed_steps']==1 and len(calls)==1
    assert r['elapsed_seconds']>=61 and r['unresolved_step_copy_upper_bound']==0
    assert hashlib.sha256(p[1].read_bytes()).hexdigest()==r['last_checkpointed_progress_sha256']
    assert 'in_flight' not in json.loads(p[2].read_bytes())

import hashlib,json
from pathlib import Path
import pytest
import ops.canonical_recovery_step as module
from ops.canonical_recovery_step import advance_fragment_step
from ops.canonical_fragment_progress import record_fragment_step
from ops.incremental_backup import RESERVE
from src.research.trade_evidence import TradeEvidenceLedger,GENESIS
from test_canonical_raw_chunk import Timer

CPU=lambda:dict(status='VERIFIED',cpu_rate=200)
BASE=dict(offset=0,sequence=0,last_hash=GENESIS)
EVIDENCE='a'*64


def fixture(tmp_path,large=0):
    src=tmp_path/'source';root=tmp_path/'staging';src.mkdir();root.mkdir()
    source=src/'canonical.jsonl';w=TradeEvidenceLedger(source);w.record('fixture',{'seed':True})
    first=source.read_bytes()
    if large:w.record('fixture',{'text':'x'*large})
    else:
        w.record('fixture',{'n':2});w.record('fixture',{'n':3})
    raw=root/'seed.raw.pending';raw.write_bytes(first);extent=source.stat().st_size
    receipt=dict(status='SINGLE_RAW_CANONICAL_RANGE_READBACK_VERIFIED',offset=0,end_offset=len(first),
        bytes=len(first),captured_extent=extent,sha256=hashlib.sha256(first).hexdigest(),pending_path=str(raw))
    progress=root/'recovery.fragment-progress.json';timer=Timer()
    seed=record_fragment_step(progress,anchor=BASE,anchor_evidence_sha256=EVIDENCE,target_extent=extent,
        receipts=[receipt],expected_progress_sha256=None,continue_check=lambda:True,resource_check=CPU,
        free_bytes=lambda:RESERVE+4000000,clock=timer.clock,sleep=timer.sleep)
    return source,progress,seed['progress_sha256'],extent


def run(source,path,pin,extent,**kw):
    timer=Timer()
    return advance_fragment_step(source,path,anchor=BASE,anchor_evidence_sha256=EVIDENCE,target_extent=extent,
        expected_progress_sha256=pin,continue_check=kw.pop('continue_check',lambda:True),resource_check=kw.pop('resource_check',CPU),
        free_bytes=kw.pop('free_bytes',lambda:RESERVE+4000000),clock=timer.clock,sleep=timer.sleep,**kw)


def test_one_window_advances_pinned_history_without_source_modification(tmp_path):
    source,path,pin,extent=fixture(tmp_path);original=source.read_bytes()
    r=run(source,path,pin,extent)
    assert r['status']=='RECOVERY_STEP_RECORDED' and r['next_cursor']['sequence']==3
    assert 0<r['new_source_bytes_copied']<=1048576 and r['retained_inputs']==1
    assert r['progress_sha256']==hashlib.sha256(path.read_bytes()).hexdigest() and source.read_bytes()==original
    assert not r['recovery_verified'] and not r['deployment_backup_accepted']
    assert not (path.parent/'canonical-step.lock').exists()
    tip=run(source,path,r['progress_sha256'],extent)
    assert tip['status']=='CAPTURE_TIP_REACHED_HISTORY_ONLY' and tip['new_source_bytes_copied']==0


def test_two_calls_preserve_first_incomplete_window_without_recopy(tmp_path):
    source,path,pin,extent=fixture(tmp_path,1100000);saved=path.read_bytes()
    first=run(source,path,pin,extent)
    assert first['status']=='RECOVERY_WINDOW_RETAINED' and first['new_source_bytes_copied']==1048576
    assert first['progress_sha256']==pin and path.read_bytes()==saved
    pending=list(path.parent.glob('step-*.raw.pending'));assert len(pending)==1;raw=pending[0].read_bytes()
    second=run(source,path,pin,extent)
    assert second['status']=='RECOVERY_STEP_RECORDED' and second['next_cursor']['sequence']==2
    assert second['new_source_bytes_copied']<1048576 and second['retained_inputs']==2
    assert pending[0].read_bytes()==raw


def fail_after_raw_publication(monkeypatch):
    original=module._publish
    def publish(path,value,expected):
        if value['inputs']:raise OSError('fixture receipt publication failure')
        return original(path,value,expected)
    monkeypatch.setattr(module,'_publish',publish)
    return original


def test_interrupted_receipt_publication_adopts_exact_source_matched_orphan(tmp_path,monkeypatch):
    source,path,pin,extent=fixture(tmp_path);saved=path.read_bytes();original=fail_after_raw_publication(monkeypatch)
    with pytest.raises(OSError):run(source,path,pin,extent)
    pending=list(path.parent.glob('step-*.raw.pending'));assert len(pending)==1;body=pending[0].read_bytes()
    assert path.read_bytes()==saved
    monkeypatch.setattr(module,'_publish',original)
    def no_copy(*a,**k):raise AssertionError('retained orphan must not be recopied')
    monkeypatch.setattr(module,'stage_raw_chunk',no_copy)
    r=run(source,path,pin,extent)
    assert r['status']=='RECOVERY_STEP_RECORDED' and r['adopted_existing'] and r['new_source_bytes_copied']==0
    assert pending[0].read_bytes()==body


def test_mismatched_orphan_is_preserved_and_rejected(tmp_path,monkeypatch):
    source,path,pin,extent=fixture(tmp_path);saved=path.read_bytes();original=fail_after_raw_publication(monkeypatch)
    with pytest.raises(OSError):run(source,path,pin,extent)
    pending=list(path.parent.glob('step-*.raw.pending'))[0];pending.write_bytes(b'x'*pending.stat().st_size)
    body=pending.read_bytes();monkeypatch.setattr(module,'_publish',original)
    with pytest.raises(ValueError):run(source,path,pin,extent)
    assert pending.read_bytes()==body and path.read_bytes()==saved


def test_history_publication_failure_reuses_retained_receipts(tmp_path,monkeypatch):
    source,path,pin,extent=fixture(tmp_path);saved=path.read_bytes();original=module.record_fragment_step
    def fail(*a,**k):raise OSError('fixture history error')
    monkeypatch.setattr(module,'record_fragment_step',fail)
    with pytest.raises(OSError):run(source,path,pin,extent)
    assert path.read_bytes()==saved
    monkeypatch.setattr(module,'record_fragment_step',original)
    def no_copy(*a,**k):raise AssertionError('retained candidate must not be recopied')
    monkeypatch.setattr(module,'stage_raw_chunk',no_copy)
    r=run(source,path,pin,extent)
    assert r['status']=='RECOVERY_STEP_RECORDED' and r['new_source_bytes_copied']==0


def test_stale_pin_stops_before_attempt_or_copy(tmp_path):
    source,path,pin,extent=fixture(tmp_path);saved=path.read_bytes()
    with pytest.raises(ValueError):run(source,path,'f'*64,extent)
    assert path.read_bytes()==saved and list(path.parent.glob('attempt-*'))==[]
    assert list(path.parent.glob('step-*.raw.pending'))==[]


def test_existing_driver_lock_preserved(tmp_path):
    source,path,pin,extent=fixture(tmp_path);lock=path.parent/'canonical-step.lock';lock.write_bytes(b'prior writer')
    with pytest.raises(FileExistsError):run(source,path,pin,extent)
    assert lock.read_bytes()==b'prior writer' and list(path.parent.glob('step-*.raw.pending'))==[]


def test_two_window_limit_never_skips_or_copies_a_third(tmp_path):
    source,path,pin,extent=fixture(tmp_path,2200000);saved=path.read_bytes()
    first=run(source,path,pin,extent);second=run(source,path,pin,extent);third=run(source,path,pin,extent)
    assert all(r['status']=='RECOVERY_WINDOW_RETAINED' for r in (first,second,third))
    assert second['bounded_row_window_exhausted'] and third['bounded_row_window_exhausted']
    assert third['new_source_bytes_copied']==0 and len(list(path.parent.glob('step-*.raw.pending')))==2
    assert path.read_bytes()==saved and third['progress_sha256']==pin


@pytest.mark.parametrize('policy',[{'window_bytes':True},{'window_bytes':1048577},{'max_records':257},{'deadline_seconds':61}])
def test_policy_rejection_does_not_copy(tmp_path,policy):
    source,path,pin,extent=fixture(tmp_path)
    with pytest.raises(ValueError):run(source,path,pin,extent,**policy)
    assert list(path.parent.glob('step-*.raw.pending'))==[]


def test_revoked_admission_preserves_history_and_source(tmp_path):
    source,path,pin,extent=fixture(tmp_path);saved=path.read_bytes();body=source.read_bytes()
    with pytest.raises(RuntimeError):run(source,path,pin,extent,continue_check=lambda:False)
    assert path.read_bytes()==saved and source.read_bytes()==body

@pytest.mark.parametrize('gate',['cpu','reserve'])
def test_driver_resource_proof_prevents_copy(tmp_path,gate):
    source,path,pin,extent=fixture(tmp_path);saved=path.read_bytes();kw={}
    if gate=='cpu':kw['resource_check']=lambda:dict(status='UNKNOWN')
    else:kw['free_bytes']=lambda:RESERVE
    with pytest.raises(RuntimeError):run(source,path,pin,extent,**kw)
    assert path.read_bytes()==saved and list(path.parent.glob('step-*.raw.pending'))==[]


def test_captured_source_truncation_stops_before_staging(tmp_path):
    source,path,pin,extent=fixture(tmp_path);saved=path.read_bytes();source.write_bytes(b'truncated')
    with pytest.raises(ValueError):run(source,path,pin,extent)
    assert path.read_bytes()==saved and list(path.parent.glob('step-*.raw.pending'))==[]


def test_progress_changed_during_copy_is_not_overwritten(tmp_path,monkeypatch):
    source,path,pin,extent=fixture(tmp_path);original=module.stage_raw_chunk
    def changed(*a,**k):
        result=original(*a,**k);path.write_bytes(b'concurrent change');return result
    monkeypatch.setattr(module,'stage_raw_chunk',changed)
    with pytest.raises(ValueError):run(source,path,pin,extent)
    assert path.read_bytes()==b'concurrent change' and len(list(path.parent.glob('step-*.raw.pending')))==1

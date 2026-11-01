import hashlib,json
import pytest
from ops.canonical_recovery_progress import verify_next_recovery_segment
from src.research.trade_evidence import TradeEvidenceLedger


def fixture(tmp_path):
    p=tmp_path/'source.jsonl';w=TradeEvidenceLedger(p);w.record('fixture',{'n':0})
    a=dict(offset=p.stat().st_size,sequence=1,last_hash=w.last_hash);anchor=dict(a);receipts=[]
    for n in (1,2):
        w.record('fixture',{'n':n});end=dict(offset=p.stat().st_size,sequence=n+1,last_hash=w.last_hash)
        body=p.read_bytes()[a['offset']:end['offset']];s=tmp_path/f'{n}.segment.pending';s.write_bytes(body)
        receipts.append(dict(status='SINGLE_CANONICAL_SEGMENT_READBACK_VERIFIED',pending_published=True,
            deployment_backup_accepted=False,start_cursor=a,next_cursor=end,bytes=len(body),
            sha256=hashlib.sha256(body).hexdigest(),pending_path=str(s)));a=end
    return anchor,receipts,tmp_path/'test.recovery-progress.json'


def run(args,**kw):
    return verify_next_recovery_segment(*args,continue_check=lambda:True,
        resource_check=lambda:dict(status='VERIFIED',cpu_rate=200),**kw)


def test_one_real_segment_per_call_and_no_revalidation_claim(tmp_path):
    args=fixture(tmp_path);one=run(args);assert one['checked']==1
    two=run(args);assert two['checked']==2 and two['all_segment_receipts_recorded']
    before=args[2].read_bytes()
    done=run(args,verifier=lambda *a,**k:pytest.fail('unexpected repeat read'))
    assert args[2].read_bytes()==before and not done['previous_results_revalidated']
    assert not done['recovery_verified'] and not done['deployment_backup_accepted']


def test_manifest_binding_and_history_corruption_preserve_progress(tmp_path):
    args=fixture(tmp_path);run(args);old=args[2].read_bytes()
    args[1][1]['sha256']='e'*64
    with pytest.raises(ValueError):run(args)
    assert args[2].read_bytes()==old
    args[1][1]['sha256']=hashlib.sha256(__import__('pathlib').Path(args[1][1]['pending_path']).read_bytes()).hexdigest()
    state=json.loads(old);state['checked'][0]['sha256']='f'*64;args[2].write_text(json.dumps(state))
    with pytest.raises(ValueError):run(args)


def test_existing_lock_preserved_and_bad_segment_does_not_advance(tmp_path):
    args=fixture(tmp_path);lock=tmp_path/'canonical-segment.lock';lock.write_text('other owner')
    with pytest.raises(FileExistsError):run(args)
    assert lock.read_text()=='other owner';lock.unlink()
    __import__('pathlib').Path(args[1][0]['pending_path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError):run(args)
    assert not args[2].exists() and not lock.exists()


def test_failed_publication_preserves_previous_progress(tmp_path,monkeypatch):
    import ops.canonical_recovery_progress as module
    args=fixture(tmp_path);run(args);old=args[2].read_bytes()
    def fail(*a):raise OSError('fixture publication failure')
    monkeypatch.setattr(module.os,'replace',fail)
    with pytest.raises(OSError):run(args)
    assert args[2].read_bytes()==old and not list(tmp_path.glob('recovery-progress-*.tmp'))
    assert not (tmp_path/'canonical-segment.lock').exists()


def test_forged_verifier_and_duplicate_state_are_rejected(tmp_path):
    args=fixture(tmp_path)
    with pytest.raises(ValueError):run(args,verifier=lambda *a,**kw:dict(status='PASS'))
    args[2].write_text('{"version":1,"version":1}')
    with pytest.raises(ValueError):run(args)


def test_admission_rejected_before_progress(tmp_path):
    args=fixture(tmp_path)
    with pytest.raises(RuntimeError):verify_next_recovery_segment(*args,continue_check=lambda:False,resource_check=lambda:dict(status='VERIFIED',cpu_rate=200))
    assert not args[2].exists()


def test_boolean_cursor_in_historical_receipt_rejected(tmp_path):
    args=fixture(tmp_path);run(args);state=json.loads(args[2].read_text())
    state['checked'][0]['start_cursor']['sequence']=True
    args[2].write_text(json.dumps(state))
    with pytest.raises(ValueError):run(args)

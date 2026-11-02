import hashlib,json,os
from pathlib import Path
import pytest
import ops.canonical_fragment_progress as module
from ops.canonical_fragment_progress import record_fragment_step
from ops.incremental_backup import RESERVE
from src.research.trade_evidence import TradeEvidenceLedger,GENESIS,_canonical
from test_canonical_raw_chunk import Timer

BASE=dict(offset=0,sequence=0,last_hash=GENESIS)
EVIDENCE='a'*64
CPU=lambda:dict(status='VERIFIED',cpu_rate=200)


def fixture(tmp_path,raw=None):
    if raw is None:
        p=tmp_path/'canonical.jsonl';w=TradeEvidenceLedger(p)
        for n in range(3):w.record('fixture',{'n':n})
        raw=p.read_bytes()
    p=tmp_path/'input.raw.pending';p.write_bytes(raw)
    r=dict(status='SINGLE_RAW_CANONICAL_RANGE_READBACK_VERIFIED',offset=0,end_offset=len(raw),bytes=len(raw),
           captured_extent=len(raw),sha256=hashlib.sha256(raw).hexdigest(),pending_path=str(p),
           ignored_source_text='untrusted instructions and unrelated data')
    return raw,r,tmp_path/'recovery.fragment-progress.json'


def step(path,r,pin=None,**kw):
    timer=Timer()
    return record_fragment_step(path,anchor=kw.pop('anchor',BASE),anchor_evidence_sha256=kw.pop('evidence',EVIDENCE),
       target_extent=kw.pop('extent',r['captured_extent']),receipts=[r],expected_progress_sha256=pin,
       continue_check=kw.pop('continue_check',lambda:True),resource_check=kw.pop('resource_check',CPU),
       free_bytes=kw.pop('free_bytes',lambda:RESERVE+1000000),clock=timer.clock,sleep=timer.sleep,**kw)


def test_two_steps_pin_parent_and_reconstruct_exact_intervals_without_duplicate_prefix(tmp_path):
    raw,r,path=fixture(tmp_path);first=step(path,r,max_records=1);saved=path.read_bytes()
    assert first['status']=='FRAGMENT_STEP_RECORDED' and first['next_cursor']['sequence']==1
    assert first['progress_sha256']==hashlib.sha256(saved).hexdigest()
    second=step(path,r,first['progress_sha256'],max_records=2);state=json.loads(path.read_bytes())
    assert second['steps']==2 and second['next_cursor']['sequence']==3
    assert second['verification']['bytes_before_cursor']==first['next_cursor']['offset']
    rebuilt=b''.join(raw[s['start_cursor']['offset']:s['next_cursor']['offset']] for s in state['steps'])
    assert rebuilt==raw and 'ignored_source_text' not in path.read_text()
    assert not second['previous_results_revalidated'] and not second['recovery_verified'] and not second['deployment_backup_accepted']
    assert state['steps'][1]['start_cursor']==state['steps'][0]['next_cursor']
    assert list(tmp_path.glob('*.partial'))==[] and not (tmp_path/'canonical-segment.lock').exists()


@pytest.mark.parametrize('pin',[None,'f'*64])
def test_existing_history_requires_exact_pin_without_reset(tmp_path,pin):
    _,r,path=fixture(tmp_path);first=step(path,r,max_records=1);saved=path.read_bytes()
    with pytest.raises((ValueError,FileExistsError)):step(path,r,pin)
    assert path.read_bytes()==saved


@pytest.mark.parametrize('kind',['gap','count','digest','traversal','anchor','extent','extra','bool_version'])
def test_pinned_but_invalid_history_is_not_accepted(tmp_path,kind):
    _,r,path=fixture(tmp_path);step(path,r,max_records=1);state=json.loads(path.read_bytes())
    if kind=='gap':state['steps'][0]['start_cursor']['offset']=1
    elif kind=='count':state['steps'][0]['verified_records']=True
    elif kind=='digest':state['steps'][0]['verified_sha256']='bad'
    elif kind=='traversal':state['steps'][0]['inputs'][0]['name']='../input.raw.pending'
    elif kind=='anchor':state['anchor']['last_hash']='b'*64
    elif kind=='extent':state['target_extent']+=1
    elif kind=='extra':state['deployment_backup_accepted']=True
    else:state['version']=True
    path.write_bytes(json.dumps(state).encode());saved=path.read_bytes();pin=hashlib.sha256(saved).hexdigest()
    with pytest.raises(ValueError):step(path,r,pin)
    assert path.read_bytes()==saved


def test_incomplete_input_never_publishes_history(tmp_path):
    _,r,path=fixture(tmp_path,b'partial')
    result=step(path,r)
    assert result['status']=='FRAGMENT_STEP_NOT_RECORDED' and not result['progress_changed']
    assert result['next_cursor']==BASE and not path.exists()


def test_late_bad_row_preserves_prior_history(tmp_path):
    raw,r,path=fixture(tmp_path);lines=raw.splitlines(keepends=True)
    bad=json.loads(lines[-1]);bad['payload']={'changed':True};lines[-1]=_canonical(bad)+b'\n'
    body=b''.join(lines);Path(r['pending_path']).write_bytes(body)
    r.update(bytes=len(body),end_offset=len(body),captured_extent=len(body),sha256=hashlib.sha256(body).hexdigest())
    first=step(path,r,max_records=1);saved=path.read_bytes()
    result=step(path,r,first['progress_sha256'],max_records=2)
    assert result['status']=='FRAGMENT_STEP_NOT_RECORDED' and result['next_cursor']==first['next_cursor']
    assert result['verification']['rows_checked_before_rejection']==1 and path.read_bytes()==saved


def test_atomic_replacement_failure_preserves_previous_progress(tmp_path,monkeypatch):
    _,r,path=fixture(tmp_path);first=step(path,r,max_records=1);saved=path.read_bytes()
    def fail(*a,**k):raise OSError('fixture publication error')
    monkeypatch.setattr(module.os,'replace',fail)
    with pytest.raises(OSError):step(path,r,first['progress_sha256'],max_records=1)
    assert path.read_bytes()==saved and list(tmp_path.glob('*.partial'))==[]
    assert not (tmp_path/'canonical-segment.lock').exists()


def test_first_publication_never_overwrites_competing_file(tmp_path,monkeypatch):
    _,r,path=fixture(tmp_path);original=module.os.link
    def collide(source,target):
        Path(target).write_bytes(b'other writer')
        return original(source,target)
    monkeypatch.setattr(module.os,'link',collide)
    with pytest.raises(FileExistsError):step(path,r)
    assert path.read_bytes()==b'other writer' and list(tmp_path.glob('*.partial'))==[]


def test_existing_shared_lock_is_preserved(tmp_path):
    _,r,path=fixture(tmp_path);lock=tmp_path/'canonical-segment.lock';lock.write_bytes(b'prior writer')
    with pytest.raises(FileExistsError):step(path,r)
    assert lock.read_bytes()==b'prior writer' and not path.exists()


@pytest.mark.parametrize('gate',['cpu','reserve','admission'])
def test_resource_gates_prevent_publication(tmp_path,gate):
    _,r,path=fixture(tmp_path);kw={}
    if gate=='cpu':kw['resource_check']=lambda:dict(status='UNKNOWN')
    elif gate=='reserve':kw['free_bytes']=lambda:RESERVE
    else:kw['continue_check']=lambda:False
    with pytest.raises(RuntimeError):step(path,r,**kw)
    assert not path.exists() and not (tmp_path/'canonical-segment.lock').exists()


def test_duplicate_and_oversized_metadata_rejected(tmp_path):
    _,r,path=fixture(tmp_path)
    for body in (b'{"version":1,"version":1}',b'x'*65537):
        path.write_bytes(body);pin=hashlib.sha256(body).hexdigest()
        with pytest.raises(ValueError):step(path,r,pin)
        assert path.read_bytes()==body

def test_progress_changed_after_verification_is_not_overwritten(tmp_path):
    _,r,path=fixture(tmp_path);first=step(path,r,max_records=1);changed=[]
    def admitted():
        if not changed and list(tmp_path.glob('fragment-progress-*.partial')):
            path.write_bytes(b'concurrent change');changed.append(True)
        return True
    with pytest.raises(ValueError):step(path,r,first['progress_sha256'],max_records=1,continue_check=admitted)
    assert changed and path.read_bytes()==b'concurrent change' and list(tmp_path.glob('*.partial'))==[]


def test_revoked_publication_admission_preserves_prior_history(tmp_path):
    _,r,path=fixture(tmp_path);first=step(path,r,max_records=1);saved=path.read_bytes()
    with pytest.raises(RuntimeError):
        step(path,r,first['progress_sha256'],max_records=1,
             continue_check=lambda:not list(tmp_path.glob('fragment-progress-*.partial')))
    assert path.read_bytes()==saved and list(tmp_path.glob('*.partial'))==[]


def test_raw_path_outside_private_history_directory_rejected(tmp_path):
    _,r,path=fixture(tmp_path);other=tmp_path/'other';other.mkdir()
    with pytest.raises(ValueError):step(other/path.name,r)
    assert list(other.iterdir())==[]


def test_history_limit_retains_all_thirty_two_verified_steps(tmp_path):
    p=tmp_path/'long.jsonl';w=TradeEvidenceLedger(p)
    for n in range(33):w.record('fixture',{'n':n})
    _,r,path=fixture(tmp_path,p.read_bytes());pin=None
    for count in range(1,33):
        result=step(path,r,pin,max_records=1);pin=result['progress_sha256']
        assert result['steps']==count and result['next_cursor']['sequence']==count
    saved=path.read_bytes()
    with pytest.raises(ValueError):step(path,r,pin,max_records=1)
    assert path.read_bytes()==saved and len(json.loads(saved)['steps'])==32

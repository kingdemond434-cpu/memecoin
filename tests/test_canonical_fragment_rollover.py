import hashlib,json
from pathlib import Path
import pytest
import ops.canonical_fragment_rollover as module
import ops.canonical_recovery_step as driver
from ops.canonical_fragment_progress import record_fragment_step
from ops.incremental_backup import RESERVE
from src.research.trade_evidence import TradeEvidenceLedger,GENESIS
from test_canonical_raw_chunk import Timer

CPU=lambda:dict(status='VERIFIED',cpu_rate=200)
BASE=dict(offset=0,sequence=0,last_hash=GENESIS)
EVIDENCE='a'*64


@pytest.fixture(scope='module')
def full(tmp_path_factory):
    root=tmp_path_factory.mktemp('rollover-seed');source=root/'source.jsonl'
    writer=TradeEvidenceLedger(source)
    for i in range(40):writer.record('fixture',{'n':i})
    raw=source.read_bytes();pending=root/'input.raw.pending';pending.write_bytes(raw)
    receipt=dict(status='SINGLE_RAW_CANONICAL_RANGE_READBACK_VERIFIED',pending_path=str(pending),
        offset=0,end_offset=len(raw),bytes=len(raw),captured_extent=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    parent=root/'parent.fragment-progress.json';pin=None;timer=Timer()
    for i in range(32):
        r=record_fragment_step(parent,anchor=BASE,anchor_evidence_sha256=EVIDENCE,target_extent=len(raw),
            receipts=[receipt],expected_progress_sha256=pin,continue_check=lambda:True,max_records=1,
            resource_check=CPU,free_bytes=lambda:RESERVE+4000000,clock=timer.clock,sleep=timer.sleep)
        pin=r['progress_sha256']
    return raw,parent.read_bytes(),pin


@pytest.fixture
def prepared(tmp_path,full):
    raw,state,pin=full;root=tmp_path/'staging';root.mkdir();src=tmp_path/'source';src.mkdir()
    source=src/'canonical.jsonl';source.write_bytes(raw);(root/'input.raw.pending').write_bytes(raw)
    parent=root/'parent.fragment-progress.json';parent.write_bytes(state)
    return source,parent,root/'child.fragment-progress.json',pin,len(raw)


def rollover(p,**kw):
    source,parent,child,pin,extent=p
    return module.rollover_fragment_history(parent,kw.pop('child',child),parent_anchor=BASE,
        parent_anchor_evidence_sha256=EVIDENCE,target_extent=extent,expected_parent_sha256=kw.pop('pin',pin),
        continue_check=kw.pop('continue_check',lambda:True),resource_check=kw.pop('resource_check',CPU),
        free_bytes=kw.pop('free_bytes',lambda:RESERVE+4000000),**kw)


def advance(p,r,**kw):
    timer=Timer()
    return driver.advance_fragment_step(p[0],p[2],anchor=r['anchor'],anchor_evidence_sha256=r['anchor_evidence_sha256'],
        target_extent=p[4],expected_progress_sha256=r['progress_sha256'],rollover_link=r['rollover_link'],
        continue_check=lambda:True,resource_check=CPU,free_bytes=lambda:RESERVE+4000000,
        max_records=1,clock=timer.clock,sleep=timer.sleep,**kw)


def test_linked_rollover_and_real_consumer_preserve_parent_and_reuse_raw(prepared,monkeypatch):
    source,parent,child,pin,extent=prepared;saved=parent.read_bytes();raw=(parent.parent/'input.raw.pending').read_bytes()
    r=rollover(prepared)
    assert r['status']=='FRAGMENT_HISTORY_ROLLOVER_LINKED' and r['anchor']['sequence']==32
    assert r['child_steps']==0 and r['new_verified_records']==0
    assert not r['recovery_verified'] and not r['recursive_ancestry_verified']
    assert json.loads(child.read_bytes())['steps']==[]
    def no_copy(*a,**k):raise AssertionError('retained suffix must be reused')
    monkeypatch.setattr(driver,'stage_raw_chunk',no_copy)
    result=advance(prepared,r)
    assert result['next_cursor']['sequence']==33 and result['new_source_bytes_copied']==0
    assert result['rollover_link_checked'] and result['reused_prior_inputs']==1
    assert parent.read_bytes()==saved and (parent.parent/'input.raw.pending').read_bytes()==raw
    advanced=child.read_bytes()
    with pytest.raises(FileExistsError):rollover(prepared)
    assert child.read_bytes()==advanced


def test_exact_empty_child_retry_is_idempotent(prepared):
    a=rollover(prepared);child=prepared[2];saved=child.read_bytes();link=Path(a['rollover_link']).read_bytes()
    b=rollover(prepared)
    assert a==b and child.read_bytes()==saved and Path(a['rollover_link']).read_bytes()==link
    assert not (child.parent/'canonical-step.lock').exists() and not (child.parent/'canonical-segment.lock').exists()


def test_interrupted_child_publication_retains_link_and_retries_exactly(prepared,monkeypatch):
    original=module._publish_exact
    def publish(path,payload,guard):
        if path==prepared[2]:raise OSError('fixture interruption')
        return original(path,payload,guard)
    monkeypatch.setattr(module,'_publish_exact',publish)
    with pytest.raises(OSError):rollover(prepared)
    link=prepared[2].with_name(prepared[2].name+'.rollover.json');saved=link.read_bytes()
    assert not prepared[2].exists()
    monkeypatch.setattr(module,'_publish_exact',original)
    r=rollover(prepared)
    assert link.read_bytes()==saved and prepared[2].exists() and r['child_steps']==0


def test_nonfull_parent_refused_without_new_artifact(prepared):
    source,parent,child,pin,extent=prepared;state=json.loads(parent.read_bytes());state['steps'].pop()
    parent.write_text(json.dumps(state));pin=hashlib.sha256(parent.read_bytes()).hexdigest();saved=parent.read_bytes()
    with pytest.raises(ValueError,match='Full nonterminal'):rollover(prepared,pin=pin)
    assert parent.read_bytes()==saved and not child.exists() and not list(parent.parent.glob('*.rollover.json'))


@pytest.mark.parametrize('kind',['link','parent','raw'])
def test_changed_rollover_evidence_fails_closed(prepared,monkeypatch,kind):
    r=rollover(prepared);child=prepared[2];saved=child.read_bytes()
    changed={'link':Path(r['rollover_link']),'parent':prepared[1],'raw':child.parent/'input.raw.pending'}[kind]
    changed.write_bytes(changed.read_bytes()+b' ');altered=changed.read_bytes()
    def no_copy(*a,**k):raise AssertionError('bad evidence must not cause recopy')
    monkeypatch.setattr(driver,'stage_raw_chunk',no_copy)
    if kind=='raw':
        result=advance(prepared,r);assert result['status']=='RECOVERY_STEP_NOT_RECORDED'
    else:
        with pytest.raises(ValueError):advance(prepared,r)
    assert child.read_bytes()==saved and changed.read_bytes()==altered


@pytest.mark.parametrize('kind',['alias','outside','collision','step_lock','segment_lock','pin','cpu','reserve','deadline','admission'])
def test_rollover_gates_preserve_existing_state(prepared,kind):
    source,parent,child,pin,extent=prepared;saved=parent.read_bytes();kw={};prior=None
    if kind=='alias':kw['child']=parent
    elif kind=='outside':kw['child']=source.parent/'escape.fragment-progress.json'
    elif kind=='collision':child.write_bytes(b'prior');prior=child
    elif kind.endswith('_lock'):
        prior=parent.parent/('canonical-step.lock' if kind=='step_lock' else 'canonical-segment.lock');prior.write_bytes(b'prior')
    elif kind=='pin':kw['pin']='f'*64
    elif kind=='cpu':kw['resource_check']=lambda:dict(status='VERIFIED',cpu_rate=500)
    elif kind=='reserve':kw['free_bytes']=lambda:RESERVE
    elif kind=='deadline':kw['deadline_seconds']=31
    elif kind=='admission':kw['continue_check']=lambda:False
    with pytest.raises((ValueError,FileExistsError,RuntimeError)):rollover(prepared,**kw)
    assert parent.read_bytes()==saved
    if prior:assert prior.read_bytes()==b'prior'
    assert not list(parent.parent.glob('*.rollover.json'))


def test_parent_change_between_link_and_child_preserves_both_evidence_and_change(prepared,monkeypatch):
    original=module._publish_exact;parent=prepared[1];saved=parent.read_bytes()
    def publish(path,payload,guard):
        original(path,payload,guard)
        if path.name.endswith('.rollover.json'):parent.write_bytes(saved+b' ')
    monkeypatch.setattr(module,'_publish_exact',publish)
    with pytest.raises(ValueError,match='Parent changed'):rollover(prepared)
    assert parent.read_bytes()==saved+b' ' and not prepared[2].exists()
    assert len(list(parent.parent.glob('*.rollover.json')))==1


def test_fsync_failure_does_not_publish_or_remove_prior_history(prepared,monkeypatch):
    saved=prepared[1].read_bytes()
    def fail(*a,**k):raise OSError('fixture fsync failure')
    monkeypatch.setattr(module.os,'fsync',fail)
    with pytest.raises(OSError):rollover(prepared)
    assert prepared[1].read_bytes()==saved and not prepared[2].exists()
    assert not list(prepared[1].parent.glob('*.partial'))


def test_rollover_never_opens_historical_raw_bytes(prepared,monkeypatch):
    original=Path.open
    def opening(path,*a,**k):
        if path.name.endswith('.raw.pending'):raise AssertionError('metadata-only rollover')
        return original(path,*a,**k)
    monkeypatch.setattr(Path,'open',opening)
    r=rollover(prepared)
    assert r['new_verified_records']==0 and not r['historical_raw_bytes_revalidated']


def test_revoked_admission_after_link_retains_exact_retry_evidence(prepared):
    child=prepared[2];link=child.with_name(child.name+'.rollover.json')
    with pytest.raises(RuntimeError):rollover(prepared,continue_check=lambda:not link.exists())
    saved=link.read_bytes();assert not child.exists()
    r=rollover(prepared)
    assert child.exists() and link.read_bytes()==saved and r['child_steps']==0


def test_explicit_link_is_checked_even_after_child_reaches_captured_tip(prepared):
    r=rollover(prepared)
    for i in range(8):
        result=advance(prepared,r);r['progress_sha256']=result['progress_sha256']
    assert result['next_cursor']['sequence']==40
    link=Path(r['rollover_link']);link.write_bytes(link.read_bytes()+b' ')
    saved=prepared[2].read_bytes()
    with pytest.raises(ValueError,match='link pin mismatch'):advance(prepared,r)
    assert prepared[2].read_bytes()==saved

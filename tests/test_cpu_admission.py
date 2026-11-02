import pickle
import pytest
from ops.cpu_admission import CpuAdmission

class Clock:
    def __init__(self):self.now=0
    def __call__(self):return self.now


def proof(**updates):
    row=dict(status='VERIFIED',cpu_rate=200,kernel_flags=5,native_threads=1,total_configured_percent=10)
    row.update(updates);return row


def test_initial_check_cache_hits_and_exact_expiry_refresh():
    c=Clock();calls=[]
    def check():calls.append(c.now);return proof()
    a=CpuAdmission(clock=c,pid=lambda:8,resource_check=check)
    assert a();c.now=.5;assert a();c.now=.999;assert a();c.now=1;assert a()
    assert calls==[0,1] and a.snapshot()['cache_hits']==2
    snap=a.snapshot();snap['last_proof']['cpu_rate']=999
    assert a.snapshot()['last_proof']['cpu_rate']==200
    assert not snap['limits_changed'] and not snap['permissions_granted']

@pytest.mark.parametrize('kwargs',[dict(status='UNVERIFIED'),dict(cpu_rate=300),dict(cpu_rate=100),dict(kernel_flags=1),dict(native_threads=2),dict(native_threads=True),dict(total_configured_percent=11),dict(total_configured_percent=float('nan'))])
def test_bad_proof_latches_closed_without_stale_positive_reuse(kwargs):
    c=Clock();rows=iter([proof(),proof(**kwargs),proof()]);calls=[]
    def check():calls.append(1);return next(rows)
    a=CpuAdmission(clock=c,pid=lambda:8,resource_check=check);assert a();c.now=1;assert not a()
    c.now=2;assert not a();assert len(calls)==2 and a.snapshot()['last_proof'] is None


def test_resource_error_is_private_and_latched():
    def fail():raise PermissionError('private file or API details')
    a=CpuAdmission(resource_check=fail)
    assert not a() and not a()
    assert a.snapshot()['proof_checks']==1 and 'private' not in repr(a.snapshot())


def test_context_change_cannot_reuse_parent_proof():
    c=Clock();identity=[8];a=CpuAdmission(clock=c,pid=lambda:identity[0],resource_check=proof)
    assert a();identity[0]=9;assert not a()
    identity[0]=8;assert not a();assert a.snapshot()['denial_reason']=='process_changed'


def test_context_change_during_check_is_rejected():
    c=Clock();identity=[8]
    def check():identity[0]=9;return proof()
    a=CpuAdmission(clock=c,pid=lambda:identity[0],resource_check=check)
    assert not a() and a.snapshot()['denial_reason']=='process_changed'

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-1,True])
def test_bad_clock_invalidates_positive_proof(bad):
    c=Clock();a=CpuAdmission(clock=c,pid=lambda:8,resource_check=proof);assert a();c.now=bad
    assert not a() and a.snapshot()['last_proof'] is None


def test_regressing_clock_invalidates_cache():
    c=Clock();c.now=2;a=CpuAdmission(clock=c,pid=lambda:8,resource_check=proof)
    assert a();c.now=1;assert not a();assert a.snapshot()['denial_reason']=='clock_regressed'


def test_slow_check_does_not_extend_evidence_lifetime():
    c=Clock()
    def check():c.now+=1;return proof()
    a=CpuAdmission(clock=c,pid=lambda:8,resource_check=check)
    assert not a() and a.snapshot()['denial_reason']=='proof_expired_during_check'


def test_evidence_age_starts_before_native_check():
    c=Clock();calls=[]
    def check():calls.append(c.now);c.now+=.2;return proof()
    a=CpuAdmission(clock=c,pid=lambda:8,resource_check=check)
    assert a();c.now=.9;assert a();c.now=1;assert a();assert calls==[0,1]

@pytest.mark.parametrize('bad',[0,-1,True,1.001,float('nan'),float('inf')])
def test_no_unbounded_evidence_lifetime(bad):
    with pytest.raises(ValueError):CpuAdmission(max_age_seconds=bad)


def test_private_proof_fields_not_retained():
    a=CpuAdmission(resource_check=lambda:proof(secret='private'))
    assert a() and 'private' not in repr(a.snapshot())


def test_evidence_cannot_be_serialized_for_another_operation():
    a=CpuAdmission(resource_check=proof);assert a()
    with pytest.raises(TypeError,match='cannot be serialized'):pickle.dumps(a)


def test_boolean_pid_cannot_alias_integer_owner():
    c=Clock();identity=[1]
    a=CpuAdmission(clock=c,pid=lambda:identity[0],resource_check=proof);assert a()
    identity[0]=True;assert not a()


def test_regression_during_native_check_is_rejected():
    c=Clock();c.now=2
    def check():c.now=1.9;return proof()
    a=CpuAdmission(clock=c,pid=lambda:8,resource_check=check)
    assert not a() and a.snapshot()['denial_reason']=='clock_regressed'


def test_staged_verifier_stops_when_refreshed_proof_is_rejected(tmp_path):
    import hashlib,sqlite3
    from ops.staged_integrity import verify_staged_sqlite
    from ops.incremental_backup import BackupStopped
    path=tmp_path/'fixture.pending';db=sqlite3.connect(path)
    try:
        db.execute('CREATE TABLE evidence(value TEXT)');db.execute("INSERT INTO evidence VALUES('retained')");db.commit()
    finally:db.close()
    original=path.read_bytes();clock=Clock();rows=iter([proof(),proof(status='UNVERIFIED')])
    def advance():clock.now+=.2;return clock.now
    a=CpuAdmission(max_age_seconds=.5,clock=advance,pid=lambda:8,resource_check=lambda:next(rows))
    receipt=dict(status='SINGLE_SQLITE_COPY_READBACK_VERIFIED_INTEGRITY_PENDING',source_write_requested=False,deployment_backup_accepted=False,bytes=len(original),sha256=hashlib.sha256(original).hexdigest(),pending_path=str(path))
    with pytest.raises(BackupStopped,match='host_gate'):verify_staged_sqlite(path,receipt,continue_check=a)
    assert path.read_bytes()==original and a.snapshot()['denial_reason']=='resource_proof_rejected'

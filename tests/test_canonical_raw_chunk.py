import hashlib
import pytest
from ops.canonical_raw_chunk import stage_raw_chunk
from ops.incremental_backup import BackupStopped, RESERVE

CPU=lambda:dict(status='VERIFIED',cpu_rate=200)

class Timer:
    def __init__(self): self.t=0
    def clock(self): return self.t
    def sleep(self,n): self.t+=n

def setup(tmp_path,body=b'first\n'+b'x'*1100000+b'\nlast\n'):
    source_dir=tmp_path/'source'; source_dir.mkdir(); source=source_dir/'canonical.jsonl'; source.write_bytes(body)
    dest=tmp_path/'separate'; dest.mkdir()
    return source,dest

def stage(source,dest,offset=0,**kw):
    timer=Timer()
    return stage_raw_chunk(source,dest,offset=offset,extent=source.stat().st_size,
        continue_check=kw.pop('continue_check',lambda:True),resource_check=kw.pop('resource_check',CPU),
        free_bytes=kw.pop('free_bytes',lambda:RESERVE+2000000),clock=timer.clock,sleep=timer.sleep,**kw)

def test_oversized_row_roundtrip_without_skipping_or_false_chain_authority(tmp_path):
    source,dest=setup(tmp_path); original=source.read_bytes(); parts=[]; offset=0
    while offset<len(original):
        target=dest/(str(offset)+'.raw.pending'); result=stage(source,target,offset)
        parts.append(target.read_bytes()); assert len(parts[-1])<=1048576
        assert result['offset']==offset and result['sha256']==hashlib.sha256(parts[-1]).hexdigest()
        assert result['logical_bytes_charged']==3*len(parts[-1])
        for field in ('chain_verified','record_boundary_verified','prefix_anchor_authenticated',
                      'cross_store_checkpoint_verified','recovery_verified','deployment_backup_accepted'):
            assert result[field] is False
        offset=result['end_offset']
    assert b''.join(parts)==original and source.read_bytes()==original
    assert list(dest.glob('*.partial'))==[] and not (dest/'canonical-segment.lock').exists()

@pytest.mark.parametrize('policy',[{'offset':True},{'offset':-1},{'extent':False},
    {'max_bytes':1048577},{'rate_bytes_per_second':0},{'deadline_seconds':float('nan')}])
def test_invalid_policy_no_files(tmp_path,policy):
    source,dest=setup(tmp_path,b'abc')
    args=dict(offset=0,extent=3,continue_check=lambda:True,resource_check=CPU); args.update(policy)
    with pytest.raises(ValueError): stage_raw_chunk(source,dest/'a.raw.pending',**args)
    assert list(dest.iterdir())==[]

@pytest.mark.parametrize('reason',['cpu','reserve','admission','deadline'])
def test_resource_gates_never_publish(tmp_path,reason):
    source,dest=setup(tmp_path); target=dest/'a.raw.pending'; args={}
    if reason=='cpu': args['resource_check']=lambda:dict(status='UNKNOWN')
    if reason=='reserve': args['free_bytes']=lambda:RESERVE
    if reason=='admission': args['continue_check']=lambda:False
    if reason=='deadline': args['deadline_seconds']=.001
    with pytest.raises((RuntimeError,BackupStopped)): stage(source,target,**args)
    assert list(dest.iterdir())==[]

def test_existing_destination_and_shared_writer_preserved(tmp_path):
    source,dest=setup(tmp_path,b'abc'); target=dest/'a.raw.pending'; target.write_bytes(b'prior')
    with pytest.raises(FileExistsError): stage(source,target)
    assert target.read_bytes()==b'prior'
    lock=dest/'canonical-segment.lock'; lock.write_bytes(b'prior writer')
    with pytest.raises(FileExistsError): stage(source,dest/'b.raw.pending')
    assert lock.read_bytes()==b'prior writer'

def test_truncation_during_copy_blocks_and_removes_only_owned_partial(tmp_path):
    source,dest=setup(tmp_path); calls=0
    def admitted():
        nonlocal calls
        calls+=1
        if calls==4: source.write_bytes(b'gone')
        return True
    with pytest.raises(BackupStopped): stage(source,dest/'a.raw.pending',continue_check=admitted)
    assert list(dest.iterdir())==[]

def test_captured_extent_excludes_later_append(tmp_path):
    source,dest=setup(tmp_path,b'abc'); source.write_bytes(b'abcdef'); timer=Timer()
    result=stage_raw_chunk(source,dest/'a.raw.pending',offset=1,extent=3,
        continue_check=lambda:True,resource_check=CPU,free_bytes=lambda:RESERVE+100,
        clock=timer.clock,sleep=timer.sleep)
    assert (dest/'a.raw.pending').read_bytes()==b'bc' and result['end_offset']==3


def test_readback_corruption_never_publishes(tmp_path):
    source,dest=setup(tmp_path,b'abcdef'); changed=False
    def admitted():
        nonlocal changed
        partial=list(dest.glob('*.partial'))
        if partial and partial[0].stat().st_size==6 and not changed:
            partial[0].write_bytes(b'broken'); changed=True
        return True
    with pytest.raises(BackupStopped): stage(source,dest/'a.raw.pending',continue_check=admitted)
    assert changed and list(dest.iterdir())==[] and source.read_bytes()==b'abcdef'

def test_source_directory_staging_rejected(tmp_path):
    source,dest=setup(tmp_path,b'abc'); nested=source.parent/'nested'; nested.mkdir()
    with pytest.raises(ValueError): stage(source,nested/'a.raw.pending')
    assert list(nested.iterdir())==[]

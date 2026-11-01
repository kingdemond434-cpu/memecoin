import pytest
from ops.canonical_segment import stage_chain_segment
from ops.incremental_backup import BackupStopped,RESERVE
from src.research.trade_evidence import TradeEvidenceLedger,GENESIS,_verified_row


def fixture(tmp_path):
    source=tmp_path/'source';source.mkdir();staging=tmp_path/'staging';staging.mkdir()
    path=source/'canonical.jsonl';writer=TradeEvidenceLedger(path);writer.record('fixture',{'n':1})
    prefix=path.read_bytes();cursor=dict(offset=len(prefix),sequence=1,last_hash=writer.last_hash)
    writer.record('fixture',{'n':2});writer.record('fixture',{'n':3})
    args=dict(source=path,pending=staging/'next.segment.pending',cursor=cursor,
        continue_check=lambda:True,resource_check=lambda:dict(status='VERIFIED',cpu_rate=200),
        free_bytes=lambda:RESERVE+4*1048576,rate_bytes_per_second=1048576)
    return args,prefix


def test_segment_readback_and_fixture_reconstruction_are_real(tmp_path):
    args,prefix=fixture(tmp_path);original=args['source'].read_bytes()
    result=stage_chain_segment(**args)
    assert result['pending_published'] and not result['deployment_backup_accepted']
    assert not result['full_prefix_chain_verified'] and not result['recovery_verified']
    recovered=prefix+args['pending'].read_bytes();assert recovered==original
    previous=GENESIS
    for seq,line in enumerate(recovered.splitlines(),1):_,previous=_verified_row(line,previous,seq)
    assert args['source'].read_bytes()==original and result['next_cursor']['sequence']==3


def test_existing_target_and_writer_lock_preserved(tmp_path):
    args,_=fixture(tmp_path);args['pending'].write_bytes(b'existing fixture')
    with pytest.raises(FileExistsError):stage_chain_segment(**args)
    assert args['pending'].read_bytes()==b'existing fixture'
    args['pending']=args['pending'].with_name('other.segment.pending')
    lock=args['pending'].parent/'canonical-segment.lock';lock.write_text('existing owner')
    with pytest.raises(FileExistsError):stage_chain_segment(**args)
    assert lock.read_text()=='existing owner'


def test_changed_range_after_verification_never_publishes(tmp_path,monkeypatch):
    import ops.canonical_segment as module
    args,_=fixture(tmp_path);original=module.verify_chain_window
    def change(*a,**kw):
        result=original(*a,**kw);body=bytearray(args['source'].read_bytes());body[-5]^=1
        args['source'].write_bytes(body);return result
    monkeypatch.setattr(module,'verify_chain_window',change)
    with pytest.raises(BackupStopped):stage_chain_segment(**args)
    assert not args['pending'].exists() and not list(args['pending'].parent.glob('*.partial'))


def test_incomplete_range_does_not_publish_empty_copy(tmp_path):
    args,_=fixture(tmp_path);body=args['source'].read_bytes()
    args['source'].write_bytes(body[:args['cursor']['offset']]+b'{"unfinished":')
    result=stage_chain_segment(**args)
    assert result['status']=='CHAIN_SEGMENT_NOT_STAGED' and not args['pending'].exists()
    assert result['verification']['next_cursor']==args['cursor']


def test_disk_reserve_rejection_preserves_source_and_no_copy(tmp_path):
    args,_=fixture(tmp_path);body=args['source'].read_bytes();args['free_bytes']=lambda:RESERVE
    result=stage_chain_segment(**args)
    assert not result['pending_published'] and args['source'].read_bytes()==body
    assert not args['pending'].exists()


def test_late_publication_collision_never_replaces_other_copy(tmp_path,monkeypatch):
    import ops.canonical_segment as module
    args,_=fixture(tmp_path);original=module.os.link
    def collide(source,destination):
        args['pending'].write_bytes(b'other publisher fixture')
        return original(source,destination)
    monkeypatch.setattr(module.os,'link',collide)
    with pytest.raises(FileExistsError):stage_chain_segment(**args)
    assert args['pending'].read_bytes()==b'other publisher fixture'
    assert not list(args['pending'].parent.glob('*.partial'))


def test_staged_corruption_before_readback_is_rejected(tmp_path,monkeypatch):
    import ops.canonical_segment as module
    args,_=fixture(tmp_path);body=args['source'].read_bytes();original=module.os.fsync
    def corrupt(fd):
        module.os.lseek(fd,0,module.os.SEEK_SET);module.os.write(fd,b'!')
        return original(fd)
    monkeypatch.setattr(module.os,'fsync',corrupt)
    with pytest.raises(BackupStopped):stage_chain_segment(**args)
    assert not args['pending'].exists() and args['source'].read_bytes()==body
    assert not list(args['pending'].parent.glob('*.partial'))

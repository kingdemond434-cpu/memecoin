import json,sqlite3
import pytest
from ops.backup_set import stage_next_store
from ops.incremental_backup import DATABASES

@pytest.fixture(autouse=True)
def deterministic_disk_space(monkeypatch):
    # The real backup path still executes; only host free-space input is modeled.
    from types import SimpleNamespace
    import ops.incremental_backup as backup
    monkeypatch.setattr(backup.shutil,"disk_usage",
        lambda _path:SimpleNamespace(free=backup.RESERVE+1024*1024))



def setup(tmp_path):
    source=tmp_path/'source';source.mkdir()
    for name in DATABASES:
        with sqlite3.connect(source/name) as db:
            db.execute('CREATE TABLE evidence(id INTEGER PRIMARY KEY,value TEXT)')
            db.execute("INSERT INTO evidence VALUES(1,'retained')")
    return dict(sources={n:source/n for n in DATABASES},staging_root=tmp_path/'staging',
                max_store_bytes=16384,continue_check=lambda:True,
                resource_check=lambda:dict(status='VERIFIED',cpu_rate=200))


def test_resume_one_real_sqlite_store_per_call_and_recover_fixture(tmp_path):
    args=setup(tmp_path)
    first=stage_next_store(**args);assert first['staged']==1
    state=(args['staging_root']/'backup-set.json').read_bytes()
    row=json.loads(state);name=next(iter(row['stores']))
    pending=args['staging_root']/(name+'.pending');before=pending.read_bytes()
    assert stage_next_store(**args)['staged']==2
    assert pending.read_bytes()==before
    for expected in range(3,11):assert stage_next_store(**args)['staged']==expected
    final=stage_next_store(**args)
    assert final['all_stores_staged'] and not final['deployment_backup_accepted']
    assert not final['recovery_verified'] and not final['cross_store_checkpoint_verified']
    with sqlite3.connect(pending) as db:
        assert db.execute('SELECT value FROM evidence').fetchone()==('retained',)
        assert db.execute('PRAGMA quick_check').fetchone()==('ok',)


def test_lock_blocks_overlapping_writer(tmp_path):
    args=setup(tmp_path);root=args['staging_root'];root.mkdir()
    (root/'backup-set.lock').write_text('existing owner')
    with pytest.raises(FileExistsError):stage_next_store(**args)
    assert (root/'backup-set.lock').read_text()=='existing owner'
    assert not list(root.glob('*.pending'))


def test_checkpoint_failure_preserves_copy_and_never_replaces_it(tmp_path,monkeypatch):
    import ops.backup_set as module
    args=setup(tmp_path)
    def fail(*a):raise OSError('fixture publication failure')
    monkeypatch.setattr(module.os,'replace',fail)
    with pytest.raises(OSError):stage_next_store(**args)
    copies=list(args['staging_root'].glob('*.pending'));assert len(copies)==1
    original=copies[0].read_bytes()
    with pytest.raises(FileExistsError):stage_next_store(**args)
    assert copies[0].read_bytes()==original


def test_admission_failure_does_not_create_staging(tmp_path):
    args=setup(tmp_path);args['continue_check']=lambda:False
    with pytest.raises(ValueError):stage_next_store(**args)
    assert not args['staging_root'].exists()


def test_source_inventory_change_rejected_on_resume(tmp_path):
    args=setup(tmp_path);stage_next_store(**args)
    args['sources'][next(iter(DATABASES))]=tmp_path/'other'/'source_intake.sqlite'
    with pytest.raises(ValueError):stage_next_store(**args)



def test_changed_prior_copy_blocks_resume_without_erasing_checkpoint(tmp_path):
    args=setup(tmp_path);stage_next_store(**args)
    checkpoint=args['staging_root']/'backup-set.json';original=checkpoint.read_bytes()
    pending=next(args['staging_root'].glob('*.pending'));pending.write_bytes(b'truncated fixture')
    with pytest.raises(ValueError):stage_next_store(**args)
    assert checkpoint.read_bytes()==original and pending.read_bytes()==b'truncated fixture'


def test_normalized_staging_cannot_alias_source_directory(tmp_path):
    args=setup(tmp_path);args['staging_root']=tmp_path/'alias'/'..'/'source'
    with pytest.raises(ValueError):stage_next_store(**args)


def test_insufficient_disk_reserve_still_blocks_real_copy(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import ops.incremental_backup as backup
    args=setup(tmp_path)
    originals={name:path.read_bytes() for name,path in args['sources'].items()}
    monkeypatch.setattr(backup.shutil,'disk_usage',lambda _path:SimpleNamespace(free=backup.RESERVE))
    with pytest.raises(backup.BackupStopped,match='disk_reserve'):stage_next_store(**args)
    assert not list(args['staging_root'].glob('*.pending'))
    assert not (args['staging_root']/'backup-set.json').exists()
    assert all(path.read_bytes()==originals[name] for name,path in args['sources'].items())

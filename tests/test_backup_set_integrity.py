import json,sqlite3
import pytest
from ops.backup_set import stage_next_store
from ops.backup_set_integrity import verify_next_staged_store
from ops.incremental_backup import DATABASES,BackupStopped


@pytest.fixture
def staged(tmp_path):
    source=tmp_path/'source';source.mkdir()
    for name in DATABASES:
        db=sqlite3.connect(source/name)
        db.execute('CREATE TABLE evidence(value INTEGER)');db.execute('INSERT INTO evidence VALUES(7)')
        db.commit();db.close()
    cpu=lambda:dict(status='VERIFIED',cpu_rate=200)
    staging=tmp_path/'staging'
    for _ in range(10):stage_next_store(sources={n:source/n for n in DATABASES},staging_root=staging,
        max_store_bytes=16384,continue_check=lambda:True,resource_check=cpu)
    return dict(staging_root=staging,continue_check=lambda:True,resource_check=cpu)


def test_one_real_quick_check_per_call_never_certifies_set(staged):
    for expected in range(1,11):assert verify_next_staged_store(**staged)['checked']==expected
    root=staged['staging_root'];before=(root/'backup-integrity.json').read_bytes()
    result=verify_next_staged_store(**staged)
    assert result['all_quick_check_receipts_recorded']
    assert not any(result[k] for k in ('previous_results_revalidated','sqlite_full_integrity_verified',
        'canonical_prefix_verified','cross_store_checkpoint_verified','recovery_verified','deployment_backup_accepted'))
    assert (root/'backup-integrity.json').read_bytes()==before


def test_changed_staging_checkpoint_cannot_reuse_results(staged):
    verify_next_staged_store(**staged);path=staged['staging_root']/'backup-set.json'
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='binding'):verify_next_staged_store(**staged)


def test_corrupt_copy_preserved_and_not_recorded_as_pass(staged):
    root=staged['staging_root'];name=sorted(DATABASES)[0];pending=root/(name+'.pending')
    body=bytearray(pending.read_bytes());body[:16]=b'corrupt fixture!';pending.write_bytes(body)
    with pytest.raises(BackupStopped):verify_next_staged_store(**staged)
    assert pending.read_bytes()==body and not (root/'backup-integrity.json').exists()


def test_existing_writer_lock_is_preserved(staged):
    lock=staged['staging_root']/'backup-set.lock';lock.write_text('existing owner')
    with pytest.raises(FileExistsError):verify_next_staged_store(**staged)
    assert lock.read_text()=='existing owner'


def test_incomplete_inventory_does_not_open_copies(staged):
    path=staged['staging_root']/'backup-set.json';state=json.loads(path.read_bytes())
    del state['stores'][next(iter(state['stores']))];path.write_text(json.dumps(state))
    staged['verifier']=lambda *a,**k:pytest.fail('must reject before database check')
    with pytest.raises(ValueError):verify_next_staged_store(**staged)



def test_large_pending_copy_does_not_starve_eligible_copies(staged):
    from ops.staged_integrity import MAX_BYTES
    root=staged['staging_root'];path=root/'backup-set.json';state=json.loads(path.read_bytes())
    large=sorted(DATABASES)[0]
    # Synthetic oversized receipt only; it must never be opened or accepted.
    state['stores'][large]['bytes']=MAX_BYTES+1
    path.write_text(json.dumps(state))
    for expected in range(1,10):
        result=verify_next_staged_store(**staged)
        assert result['checked']==expected and result['blocked_by_copy_budget']==[large]
    final=verify_next_staged_store(**staged)
    assert final['checked']==9 and not final['all_quick_check_receipts_recorded']
    assert not final['deployment_backup_accepted']

import hashlib
import json
import sqlite3

import pytest

from tools.consolidate_backup import DATABASES, consolidate


@pytest.fixture(autouse=True)
def controlled_fixture_headroom(monkeypatch):
    from types import SimpleNamespace
    import tools.consolidate_backup as module
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=100*1024**3))


def fixture(tmp_path):
    folders = [tmp_path/f'report-deployment-backup-20261028-0{i}0000' for i in (1,2,3)]
    receipts = []
    for folder in folders:
        folder.mkdir()
        for name in DATABASES:
            db = sqlite3.connect(folder/name)
            db.execute('CREATE TABLE evidence(payload BLOB)')
            db.execute('INSERT INTO evidence VALUES (?)', (b'public fixture'*1000,))
            db.commit(); db.close()
        for name in ('canonical-certificate.json', 'integrity.json'):
            (folder/name).write_text(json.dumps({'hash_chain_verified':True}))
        (folder/'trade_evidence.jsonl').write_bytes(b'canonical fixture preserved')
        files = {p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in folder.glob('*.sqlite')}
        receipt = tmp_path/(folder.name+'.json')
        receipt.write_text(json.dumps({'status':'PASS','backup':str(folder),'sqlite':dict.fromkeys(DATABASES,True),'files':files}))
        receipts.append(receipt)
    return folders, receipts


def run(tmp_path, folders, receipts, **kwargs):
    args = dict(backups_root=tmp_path, verification_receipt=receipts[0], protected_receipts=receipts[1:], retirement_receipt=tmp_path/'retired.json')
    args.update(kwargs)
    return consolidate(folders[0]/'phantom_perps.sqlite', **args)


def test_real_sqlite_consolidation_preserves_newest_and_canonical(tmp_path):
    folders, receipts = fixture(tmp_path)
    before = {str(p):p.read_bytes() for folder in folders for p in folder.iterdir()}
    result = run(tmp_path, folders, receipts)
    source = folders[0]/'phantom_perps.sqlite'
    assert result['status'] == 'ORIGINAL_RETIRED_ARCHIVE_RESTORE_VERIFIED'
    assert result['net_backup_bytes_saved'] > 0 and not source.exists()
    assert source.with_suffix('.sqlite.gz').exists()
    for path, body in before.items():
        if path != str(source):
            from pathlib import Path
            assert Path(path).read_bytes() == body
    assert not list(folders[0].glob('*.restore'))
    assert json.loads((tmp_path/'retired.json').read_text())['sqlite_quick_check'] == 'PASS'


def test_protected_newest_cannot_be_retired(tmp_path):
    folders, receipts = fixture(tmp_path)
    with pytest.raises(ValueError):
        consolidate(folders[2]/'phantom_perps.sqlite', backups_root=tmp_path, verification_receipt=receipts[2], protected_receipts=receipts[1:], retirement_receipt=tmp_path/'retired.json')
    assert (folders[2]/'phantom_perps.sqlite').exists()


def test_wrong_hash_prevents_retirement(tmp_path):
    folders, receipts = fixture(tmp_path)
    prior = json.loads(receipts[0].read_text()); prior['files']['phantom_perps.sqlite']['sha256']='0'*64
    receipts[0].write_text(json.dumps(prior))
    with pytest.raises(ValueError):run(tmp_path, folders, receipts)
    assert (folders[0]/'phantom_perps.sqlite').exists()
    assert not (tmp_path/'retired.json').exists()


def test_missing_protected_copy_prevents_retirement(tmp_path):
    folders, receipts = fixture(tmp_path)
    (folders[2]/'source_intake.sqlite').unlink()
    with pytest.raises(ValueError, match='missing'):run(tmp_path, folders, receipts)
    assert (folders[0]/'phantom_perps.sqlite').exists()


def test_canonical_file_never_eligible(tmp_path):
    folders, receipts = fixture(tmp_path)
    with pytest.raises(ValueError, match='SQLite'):
        consolidate(folders[0]/'trade_evidence.jsonl', backups_root=tmp_path, verification_receipt=receipts[0], protected_receipts=receipts[1:], retirement_receipt=tmp_path/'retired.json')
    assert (folders[0]/'trade_evidence.jsonl').read_bytes() == b'canonical fixture preserved'


def test_receipt_write_failure_preserves_original(tmp_path, monkeypatch):
    import tools.consolidate_backup as module
    folders, receipts = fixture(tmp_path)
    def fail(*args, **kwargs):raise OSError('isolated receipt disk failure')
    monkeypatch.setattr(module, '_save', fail)
    with pytest.raises(OSError):run(tmp_path, folders, receipts)
    assert (folders[0]/'phantom_perps.sqlite').exists()
    assert not list(folders[0].glob('*.restore'))


def test_outside_root_rejected(tmp_path):
    folders, receipts = fixture(tmp_path)
    with pytest.raises(ValueError, match='root'):
        run(tmp_path, folders, receipts, backups_root=folders[0])
    assert (folders[0]/'phantom_perps.sqlite').exists()



def test_low_disk_reserve_blocks_before_archive_or_retirement(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import tools.consolidate_backup as module
    folders, receipts = fixture(tmp_path)
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda path: SimpleNamespace(free=12*1024**3))
    with pytest.raises(ValueError, match='headroom'):
        run(tmp_path, folders, receipts)
    source = folders[0]/'phantom_perps.sqlite'
    assert source.exists() and not source.with_suffix('.sqlite.gz').exists()
    assert not (tmp_path/'retired.json').exists()

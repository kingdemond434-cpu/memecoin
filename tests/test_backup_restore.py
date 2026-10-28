import gzip
import hashlib
import sqlite3

import pytest

from tools.restore_backup import restore_archive


def test_isolated_database_restore(tmp_path):
    original = tmp_path/'sealed.sqlite'
    db = sqlite3.connect(original)
    db.execute('CREATE TABLE evidence(id INTEGER PRIMARY KEY, payload TEXT)')
    db.executemany('INSERT INTO evidence(payload) VALUES (?)', [('public fixture',)]*100)
    db.commit(); db.close()
    body = original.read_bytes(); archive = tmp_path/'archive.gz'
    archive.write_bytes(gzip.compress(body)); target = tmp_path/'restored.sqlite'
    result = restore_archive(archive, target, expected_sha256=hashlib.sha256(body).hexdigest(), expected_bytes=len(body), sqlite_check=True)
    assert result['sqlite_quick_check'] == 'PASS'
    assert target.read_bytes() == body and original.read_bytes() == body
    db = sqlite3.connect(target)
    assert db.execute('SELECT count(*) FROM evidence').fetchone() == (100,)
    db.close()


@pytest.mark.parametrize('size, digest', [(2, None), (100, None), (7, '0'*64)])
def test_incorrect_size_or_digest_never_publishes(tmp_path, size, digest):
    body = b'fixture'; archive = tmp_path/'archive.gz'; archive.write_bytes(gzip.compress(body))
    target = tmp_path/'restore'
    with pytest.raises(ValueError):
        restore_archive(archive, target, expected_sha256=digest or hashlib.sha256(body).hexdigest(), expected_bytes=size)
    assert not target.exists() and not list(tmp_path.glob('*.partial'))
    assert gzip.decompress(archive.read_bytes()) == body


def test_non_database_rejected_when_sqlite_requested(tmp_path):
    body = b'not a database'; archive = tmp_path/'archive.gz'; archive.write_bytes(gzip.compress(body))
    target = tmp_path/'restore'
    with pytest.raises(sqlite3.DatabaseError):
        restore_archive(archive, target, expected_sha256=hashlib.sha256(body).hexdigest(), expected_bytes=len(body), sqlite_check=True)
    assert not target.exists() and not list(tmp_path.glob('*.partial'))


def test_restore_does_not_replace_existing_file(tmp_path):
    body = b'fixture'; archive = tmp_path/'archive.gz'; archive.write_bytes(gzip.compress(body))
    target = tmp_path/'restore'; target.write_bytes(b'current state')
    with pytest.raises(ValueError):
        restore_archive(archive, target, expected_sha256=hashlib.sha256(body).hexdigest(), expected_bytes=len(body))
    assert target.read_bytes() == b'current state'


def test_racing_publication_preserved(tmp_path, monkeypatch):
    import tools.restore_backup as module
    archive = tmp_path/'archive.gz'; archive.write_bytes(gzip.compress(b'fixture'))
    target = tmp_path/'restore'
    def race(partial, destination):
        destination.write_bytes(b'other state')
        raise FileExistsError('fixture')
    monkeypatch.setattr(module.os, 'link', race)
    with pytest.raises(FileExistsError):
        restore_archive(archive, target, expected_sha256=hashlib.sha256(b'fixture').hexdigest(), expected_bytes=7)
    assert target.read_bytes() == b'other state' and not list(tmp_path.glob('*.partial'))


def test_readback_detects_written_file_corruption(tmp_path, monkeypatch):
    import tools.restore_backup as module
    archive = tmp_path/'archive.gz'; archive.write_bytes(gzip.compress(b'fixture'))
    target = tmp_path/'restore'; real_fsync = module.os.fsync
    def damage(fd):
        real_fsync(fd)
        partial = next(tmp_path.glob('*.partial'))
        partial.write_bytes(b'damaged')
    monkeypatch.setattr(module.os, 'fsync', damage)
    with pytest.raises(ValueError, match='read-back'):
        restore_archive(archive, target, expected_sha256=hashlib.sha256(b'fixture').hexdigest(), expected_bytes=7)
    assert not target.exists() and not list(tmp_path.glob('*.partial'))


def test_truncated_archive_cleanup(tmp_path):
    archive = tmp_path/'archive.gz'; archive.write_bytes(gzip.compress(b'fixture')[:-5])
    target = tmp_path/'restore'
    with pytest.raises((EOFError, OSError)):
        restore_archive(archive, target, expected_sha256=hashlib.sha256(b'fixture').hexdigest(), expected_bytes=7)
    assert not target.exists() and not list(tmp_path.glob('*.partial'))

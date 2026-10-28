import gzip
import hashlib

import pytest

from tools.compress_backup import compress_backup, verify_archive


def test_complete_sqlite_backup_roundtrip(tmp_path):
    import sqlite3
    source = tmp_path/'sealed.sqlite'
    db = sqlite3.connect(source)
    db.execute('CREATE TABLE payload(value BLOB)')
    db.executemany('INSERT INTO payload VALUES (?)', [(b'public fixture'*900,), (bytes(range(256))*5000,)])
    db.commit()
    db.close()
    original = source.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    archive = tmp_path/'sealed.sqlite.gz'
    result = compress_backup(source, archive, expected_sha256=digest)
    assert result['status'] == 'PASS' and result['space_reclaimed_bytes'] == 0
    assert source.read_bytes() == original
    assert verify_archive(archive, digest) == len(original)
    restored = tmp_path/'restored.sqlite'
    restored.write_bytes(gzip.decompress(archive.read_bytes()))
    db = sqlite3.connect(restored)
    assert db.execute('PRAGMA quick_check').fetchall() == [('ok',)]
    assert db.execute('SELECT count(*) FROM payload').fetchone() == (2,)
    db.close()


def test_wrong_receipt_never_publishes_or_changes_source(tmp_path):
    source = tmp_path/'source'; source.write_bytes(b'recovery evidence')
    target = tmp_path/'archive.gz'
    with pytest.raises(ValueError, match='prior backup receipt'):
        compress_backup(source, target, expected_sha256='0'*64)
    assert source.read_bytes() == b'recovery evidence'
    assert not target.exists() and not list(tmp_path.glob('*.partial'))


def test_existing_destination_preserved(tmp_path):
    source = tmp_path/'source'; source.write_bytes(b'recovery evidence')
    target = tmp_path/'archive.gz'; target.write_bytes(b'previous archive')
    with pytest.raises(ValueError, match='distinct'):
        compress_backup(source, target, expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    assert target.read_bytes() == b'previous archive'


def test_corrupted_archive_rejected(tmp_path):
    archive = tmp_path/'damaged.gz'
    archive.write_bytes(gzip.compress(b'fixture')[:-4])
    with pytest.raises((EOFError, OSError)):
        verify_archive(archive, hashlib.sha256(b'fixture').hexdigest())


def test_exclusive_publication_race(tmp_path, monkeypatch):
    import tools.compress_backup as module
    source = tmp_path/'source'; source.write_bytes(b'evidence')
    target = tmp_path/'archive.gz'
    def race(partial, destination):
        destination.write_bytes(b'other publisher')
        raise FileExistsError('fixture')
    monkeypatch.setattr(module.os, 'link', race)
    with pytest.raises(FileExistsError):
        compress_backup(source, target, expected_sha256=hashlib.sha256(b'evidence').hexdigest())
    assert target.read_bytes() == b'other publisher'
    assert source.read_bytes() == b'evidence' and not list(tmp_path.glob('*.partial'))


def test_source_mutation_rejected_before_publication(tmp_path, monkeypatch):
    import tools.compress_backup as module
    source = tmp_path/'source'; source.write_bytes(b'evidence')
    target = tmp_path/'archive.gz'
    real_fsync = module.os.fsync
    def mutate(fd):
        real_fsync(fd)
        source.write_bytes(b'changed evidence')
    monkeypatch.setattr(module.os, 'fsync', mutate)
    with pytest.raises(ValueError, match='source changed'):
        compress_backup(source, target, expected_sha256=hashlib.sha256(b'evidence').hexdigest())
    assert not target.exists() and not list(tmp_path.glob('*.partial'))


def test_failed_write_keeps_original(tmp_path, monkeypatch):
    import tools.compress_backup as module
    source = tmp_path/'source'; source.write_bytes(b'evidence')
    target = tmp_path/'archive.gz'
    def full_disk(fd):
        raise OSError('isolated disk-full fixture')
    monkeypatch.setattr(module.os, 'fsync', full_disk)
    with pytest.raises(OSError):
        compress_backup(source, target, expected_sha256=hashlib.sha256(b'evidence').hexdigest())
    assert source.read_bytes() == b'evidence'
    assert not target.exists() and not list(tmp_path.glob('*.partial'))

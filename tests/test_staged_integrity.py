import hashlib
import sqlite3

import pytest
from ops.incremental_backup import BackupStopped, stage_sqlite_backup, RESERVE
from ops.staged_integrity import verify_staged_sqlite


class Clock:
    def __init__(self): self.now = 0
    def __call__(self): return self.now
    def sleep(self, seconds): self.now += seconds


def fixture(tmp_path):
    source = tmp_path/'source_intake.sqlite'
    with sqlite3.connect(source) as db:
        db.execute('CREATE TABLE sample(id INTEGER PRIMARY KEY, body TEXT)')
        db.executemany('INSERT INTO sample(body) VALUES (?)', [('x'*100,)]*300)
    clock = Clock()
    pending = tmp_path/'copy.pending'
    receipt = stage_sqlite_backup(source, pending, max_bytes=2**20,
        continue_check=lambda: True, clock=clock, sleep=clock.sleep,
        free_bytes=lambda: RESERVE+2**22)
    return source, pending, receipt


def verify(pending, receipt, **kw):
    clock = Clock()
    options = dict(continue_check=lambda: True, clock=clock, sleep=clock.sleep)
    options.update(kw)
    return verify_staged_sqlite(pending, receipt, **options)


def test_real_staging_quick_check_binds_digest_without_changing_files(tmp_path):
    source, pending, receipt = fixture(tmp_path)
    before = source.read_bytes(), pending.read_bytes()
    result = verify(pending, receipt)
    assert result['sqlite_quick_check_verified']
    assert not result['sqlite_full_integrity_verified']
    assert not result['deployment_backup_accepted']
    assert not result['cross_store_checkpoint_verified']
    assert result['digest_bytes_charged'] == receipt['bytes']
    assert result['paced_sleep_seconds'] >= receipt['bytes']/2**18-.001
    assert before == (source.read_bytes(), pending.read_bytes())
    assert not list(tmp_path.glob('*-journal'))


@pytest.mark.parametrize('field,value', [('sha256','a'*64), ('bytes',True),
    ('pending_path','elsewhere.pending'), ('source_write_requested',True),
    ('status','other'), ('sha256','bad')])
def test_receipt_mismatch_never_accepts(tmp_path, field, value):
    _, pending, receipt = fixture(tmp_path)
    receipt[field] = value
    with pytest.raises((ValueError, BackupStopped)):
        verify(pending, receipt)
    assert pending.exists()


@pytest.mark.parametrize('suffix', ['-wal','-shm','-journal'])
def test_sidecars_cannot_be_ignored(tmp_path, suffix):
    _, pending, receipt = fixture(tmp_path)
    sidecar = pending.with_name(pending.name+suffix)
    sidecar.write_bytes(b'committed or unfinished state')
    with pytest.raises(BackupStopped, match='sidecar'):
        verify(pending, receipt)
    assert sidecar.exists()


def test_corruption_with_matching_digest_still_fails_sqlite_check(tmp_path):
    _, pending, receipt = fixture(tmp_path)
    with pending.open('r+b') as stream: stream.write(b'not sqlite')
    receipt['sha256'] = hashlib.sha256(pending.read_bytes()).hexdigest()
    with pytest.raises(BackupStopped, match='sqlite_'):
        verify(pending, receipt)


def test_vm_work_budget_interrupts_without_certificate(tmp_path):
    _, pending, receipt = fixture(tmp_path)
    with pytest.raises(BackupStopped, match='vm_budget'):
        verify(pending, receipt, vm_budget=100)


def test_host_gate_can_stop_digest_readback(tmp_path):
    _, pending, receipt = fixture(tmp_path)
    calls = 0
    def admit():
        nonlocal calls
        calls += 1
        return calls < 8
    with pytest.raises(BackupStopped, match='host_gate'):
        verify(pending, receipt, continue_check=admit)


@pytest.mark.parametrize('key,value', [('max_bytes',2**24), ('vm_budget',True),
    ('deadline_seconds',float('nan')), ('deadline_seconds',float('inf')),
    ('deadline_seconds',True), ('rate_bytes_per_second',2**21)])
def test_invalid_resource_policy_rejected(tmp_path, key, value):
    _, pending, receipt = fixture(tmp_path)
    with pytest.raises(ValueError): verify(pending, receipt, **{key:value})


def test_changed_copy_during_pacing_rejected(tmp_path):
    _, pending, receipt = fixture(tmp_path)
    clock = Clock()
    def sleep(seconds):
        clock.sleep(seconds)
        with pending.open('ab') as stream: stream.write(b'changed')
    with pytest.raises(BackupStopped, match='changed|identity'):
        verify(pending, receipt, clock=clock, sleep=sleep)


def test_deadline_during_paced_readback_cannot_accept(tmp_path):
    _, pending, receipt = fixture(tmp_path)
    with pytest.raises(BackupStopped, match='deadline'):
        verify(pending, receipt, deadline_seconds=.001)


def test_policy_size_limit_checked_before_sqlite_open(tmp_path, monkeypatch):
    _, pending, receipt = fixture(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError('SQLite must not open over-budget input')
    monkeypatch.setattr(sqlite3, 'connect', forbidden)
    with pytest.raises(ValueError):
        verify(pending, receipt, max_bytes=1024)

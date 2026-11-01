import hashlib
import sqlite3
from pathlib import Path

import pytest
from ops.incremental_backup import stage_sqlite_backup, BackupStopped, RESERVE


class Clock:
    def __init__(self):self.now=0
    def __call__(self):return self.now
    def sleep(self,seconds):self.now+=seconds


def database(tmp_path):
    source=tmp_path/'source_intake.sqlite'
    db=sqlite3.connect(source);db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE intake(id INTEGER PRIMARY KEY, body BLOB)')
    db.executemany('INSERT INTO intake(body) VALUES (?)',[(b'x'*2000,)]*100);db.commit()
    return source,db


def options(clock):
    return dict(max_bytes=2**20,continue_check=lambda:True,pages_per_step=4,
                rate_bytes_per_second=2**18,deadline_seconds=60,
                clock=clock,sleep=clock.sleep,free_bytes=lambda:RESERVE+4*2**20)


def test_wal_backup_readback_and_fixture_recovery_are_real(tmp_path):
    source,writer=database(tmp_path);clock=Clock();pending=tmp_path/'copy.sqlite.pending'
    before=source.read_bytes()
    try:
        assert Path(str(source)+'-wal').stat().st_size>0
        result=stage_sqlite_backup(source,pending,**options(clock))
        assert hashlib.sha256(pending.read_bytes()).hexdigest()==result['sha256']
        assert result['page_batches']>1
        assert result['paced_sleep_seconds']>=result['logical_bytes_charged']/2**18-.001
        assert not result['deployment_backup_accepted']
        assert not result['sqlite_integrity_verified']
        # Integrity/recovery checked only on this tiny fixture, not asserted by
        # the staging library for arbitrary production data.
        recovered=sqlite3.connect(pending.as_uri()+'?mode=ro&immutable=1',uri=True)
        try:
            assert recovered.execute('PRAGMA quick_check').fetchone()[0]=='ok'
            assert recovered.execute('SELECT count(*) FROM intake').fetchone()[0]==100
        finally:recovered.close()
        assert source.read_bytes()==before
        writer.execute('INSERT INTO intake(body) VALUES (?)',(b'after backup',));writer.commit()
        assert writer.execute('SELECT count(*) FROM intake').fetchone()[0]==101
    finally:writer.close()


def test_host_gate_stop_retains_source_and_no_published_copy(tmp_path):
    source,writer=database(tmp_path);clock=Clock();pending=tmp_path/'copy.pending';calls=0
    def admit():
        nonlocal calls
        calls+=1
        return calls<5
    try:
        kw=options(clock);kw['continue_check']=admit
        with pytest.raises(BackupStopped,match='host_gate'):
            stage_sqlite_backup(source,pending,**kw)
        assert not pending.exists() and not list(tmp_path.glob('*.partial'))
        assert writer.execute('SELECT count(*) FROM intake').fetchone()[0]==100
    finally:writer.close()


@pytest.mark.parametrize('policy,value,reason',[
    ('deadline_seconds',.01,'deadline'),('max_bytes',4096,'size_budget'),
    ('free_bytes',lambda:RESERVE,'disk_reserve')])
def test_resource_failures_never_publish(tmp_path,policy,value,reason):
    source,writer=database(tmp_path);clock=Clock();pending=tmp_path/'copy.pending'
    try:
        kw=options(clock);kw[policy]=value
        with pytest.raises(BackupStopped,match=reason):
            stage_sqlite_backup(source,pending,**kw)
        assert not pending.exists() and not list(tmp_path.glob('*.partial'))
    finally:writer.close()


def test_existing_copy_never_overwritten(tmp_path):
    source,writer=database(tmp_path);clock=Clock();pending=tmp_path/'copy.pending'
    pending.write_bytes(b'prior private copy')
    try:
        with pytest.raises(FileExistsError):
            stage_sqlite_backup(source,pending,**options(clock))
        assert pending.read_bytes()==b'prior private copy'
    finally:writer.close()


def test_interrupt_closes_handles_and_cleans_only_owned_partial(tmp_path):
    source,writer=database(tmp_path);clock=Clock();pending=tmp_path/'copy.pending'
    unrelated=tmp_path/'unrelated.partial';unrelated.write_bytes(b'preserve')
    calls=0
    def admit():
        nonlocal calls
        calls+=1
        if calls==5:raise KeyboardInterrupt()
        return True
    try:
        kw=options(clock);kw['continue_check']=admit
        with pytest.raises(KeyboardInterrupt):stage_sqlite_backup(source,pending,**kw)
        assert unrelated.read_bytes()==b'preserve' and not pending.exists()
        assert list(tmp_path.glob('*.partial'))==[unrelated]
        writer.execute('INSERT INTO intake(body) VALUES (?)',(b'still writable',));writer.commit()
    finally:writer.close()


@pytest.mark.parametrize('field,value',[('pages_per_step',0),('pages_per_step',True),
    ('rate_bytes_per_second',2**21),('deadline_seconds',float('nan')),('max_bytes',True)])
def test_invalid_policy_rejected_before_copy(tmp_path,field,value):
    source,writer=database(tmp_path);clock=Clock();pending=tmp_path/'copy.pending'
    try:
        kw=options(clock);kw[field]=value
        with pytest.raises(ValueError):stage_sqlite_backup(source,pending,**kw)
        assert not pending.exists() and not list(tmp_path.glob('*.partial'))
    finally:writer.close()


def test_preexisting_unique_name_collision_is_not_removed(tmp_path,monkeypatch):
    from types import SimpleNamespace
    source,writer=database(tmp_path);pending=tmp_path/'copy.pending';clock=Clock()
    collision=tmp_path/'copy.pending.fixed.partial';collision.write_bytes(b'not owned')
    monkeypatch.setattr('ops.incremental_backup.uuid.uuid4',lambda:SimpleNamespace(hex='fixed'))
    try:
        with pytest.raises(FileExistsError):stage_sqlite_backup(source,pending,**options(clock))
        assert collision.read_bytes()==b'not owned' and not pending.exists()
    finally:writer.close()


def test_live_wal_write_during_steps_is_not_lost(tmp_path):
    source,writer=database(tmp_path);pending=tmp_path/'copy.pending';clock=Clock();calls=0
    def admit():
        nonlocal calls
        calls+=1
        if calls==3:
            writer.execute('INSERT INTO intake(body) VALUES (?)',(b'committed during backup',));writer.commit()
        return True
    try:
        kw=options(clock);kw['continue_check']=admit
        result=stage_sqlite_backup(source,pending,**kw)
        recovered=sqlite3.connect(pending.as_uri()+'?mode=ro&immutable=1',uri=True)
        try:assert recovered.execute('SELECT count(*) FROM intake').fetchone()[0]==101
        finally:recovered.close()
        assert result['source_write_requested'] is False
        assert result['deployment_backup_accepted'] is False
    finally:writer.close()

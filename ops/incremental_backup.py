"""Paced single-SQLite staging, never a deployable backup certificate.

SQLite online backup includes committed WAL state. Full integrity, canonical
prefix and cross-store checkpoint verification remain mandatory later gates.
Only isolated fixtures use this module until the deployment caller supplies
trusted host/CPU/task/backup admission evidence.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import sqlite3
import stat
import time
import uuid

DATABASES = frozenset(('champion_state.population.sqlite', 'free_market_discovery.sqlite',
    'frontier.sqlite', 'native_fork_probe.sqlite', 'native_pool_attribution.sqlite',
    'neural_sentiment.sqlite', 'phantom_perps.sqlite', 'public_claim_links.sqlite',
    'public_onion_frontier.sqlite', 'source_intake.sqlite'))
RESERVE = 12 * 1024**3


class BackupStopped(RuntimeError):
    """A resource/admission gate stopped staging; never retry automatically."""


class _Pacer:
    def __init__(self, rate, deadline, admit, free, clock, sleep, reserve):
        self.rate, self.limit, self.admit, self.free = rate, deadline, admit, free
        self.clock, self.sleep, self.reserve = clock, sleep, reserve
        self.started = clock()
        self.charged = 0
        self.slept = 0.0

    def guard(self, remaining=0):
        if self.clock() - self.started >= self.limit:
            raise BackupStopped('deadline')
        if self.admit() is not True:
            raise BackupStopped('host_gate')
        if self.free() - remaining < self.reserve:
            raise BackupStopped('disk_reserve')

    def charge(self, amount, remaining=0):
        self.charged += amount
        wait = self.charged / self.rate - (self.clock() - self.started)
        # Bounded sleep slices keep host/deadline/reserve gates responsive.
        while wait > 1e-9:
            self.guard(remaining)
            portion = min(wait, .1)
            self.sleep(portion); self.slept += portion
            wait = self.charged / self.rate - (self.clock() - self.started)
        self.guard(remaining)


def stage_sqlite_backup(source, pending, *, max_bytes, continue_check,
                        rate_bytes_per_second=256*1024, pages_per_step=16,
                        deadline_seconds=60, clock=time.monotonic,
                        sleep=time.sleep, free_bytes=None):
    """Create a new .pending copy and paced SHA readback under admission gates.

    Limits apply to logical batch/readback work, not physical OS IO, fsync, RAM
    or device cache. One SQLite batch can precede its pacing wait. Busy callbacks
    are counted; repeated successful steps are charged in full. No large integrity
    scan is performed here. Caller must keep the output directory private and
    serialize maintenance; no service/task/manifest action occurs.
    """
    if (type(max_bytes) is not int or max_bytes < 1
            or type(rate_bytes_per_second) is not int or not 1 <= rate_bytes_per_second <= 1024**2
            or type(pages_per_step) is not int or not 1 <= pages_per_step <= 16
            or type(deadline_seconds) not in (int, float) or not 0 < deadline_seconds <= 43200):
        raise ValueError('Bounded SQLite staging policy required')
    source, pending = Path(source).absolute(), Path(pending).absolute()
    info = source.lstat()
    if (source.name not in DATABASES or not stat.S_ISREG(info.st_mode)
            or source.is_symlink() or getattr(info, 'st_file_attributes', 0) & 0x400
            or pending.suffix != '.pending' or source == pending):
        raise ValueError('Whitelisted regular SQLite source and new pending path required')
    if pending.exists():
        raise FileExistsError('Pending target already exists')
    free = free_bytes or (lambda: shutil.disk_usage(pending.parent).free)
    pacer = _Pacer(rate_bytes_per_second, deadline_seconds, continue_check,
                   free, clock, sleep, RESERVE)
    pacer.guard()
    temporary = pending.with_name(pending.name+'.'+uuid.uuid4().hex+'.partial')
    steps = busy = 0
    final_total = None
    owned = False
    try:
        # Reserve only our unique output; never overwrite a prior copy.
        fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        owned = True
        os.close(fd)
        with sqlite3.connect(source.as_uri()+'?mode=ro', uri=True, timeout=1) as src:
            src.execute('PRAGMA query_only=ON')
            page_size = src.execute('PRAGMA page_size').fetchone()[0]
            page_count = src.execute('PRAGMA page_count').fetchone()[0]
            if page_size * page_count > max_bytes:
                raise BackupStopped('source_size_budget')
            pacer.guard(page_size * page_count)
            def progress(status, remaining, total):
                nonlocal steps, busy, final_total
                pacer.guard(remaining * page_size)
                if total * page_size > max_bytes:
                    raise BackupStopped('source_size_budget')
                if status in (sqlite3.SQLITE_OK, sqlite3.SQLITE_DONE):
                    steps += 1
                    # Charge the full quantum, including repeats after a live
                    # source change. Never infer work from net remaining delta.
                    pacer.charge(pages_per_step * page_size, remaining * page_size)
                elif status in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                    busy += 1
                else:
                    raise BackupStopped('sqlite_step_status')
                if status == sqlite3.SQLITE_DONE:
                    final_total = total
            dst = sqlite3.connect(temporary, timeout=1)
            try:
                src.backup(dst, pages=pages_per_step, progress=progress, sleep=.05)
            finally:
                dst.close()
        # sqlite3's connection context manager commits/rolls back; explicit close
        # below is necessary on Windows before publication/fixture cleanup.
        src.close()
        if final_total is None or temporary.stat().st_size != final_total * page_size:
            raise BackupStopped('completed_size_mismatch')
        size = temporary.stat().st_size
        if size > max_bytes:
            raise BackupStopped('copy_size_budget')
        # Windows fsync requires a writable file handle. No bytes are changed.
        with temporary.open('r+b') as flush_file:
            flush_file.flush(); os.fsync(flush_file.fileno())
        with temporary.open('rb') as stream:
            before = os.fstat(stream.fileno())
            digest = hashlib.sha256(); read = 0
            while True:
                pacer.guard()
                block = stream.read(65536)
                if not block:
                    break
                read += len(block); digest.update(block); pacer.charge(len(block))
            after = os.fstat(stream.fileno())
        if (read != size or (before.st_size, before.st_mtime_ns) !=
                (after.st_size, after.st_mtime_ns)):
            raise BackupStopped('readback_changed')
        pacer.guard()
        os.link(temporary, pending)  # exclusive publication, never replace
        return {'status': 'SINGLE_SQLITE_COPY_READBACK_VERIFIED_INTEGRITY_PENDING',
                'bytes': size, 'sha256': digest.hexdigest(), 'page_size': page_size,
                'page_batches': steps, 'busy_callbacks': busy,
                'logical_bytes_charged': pacer.charged, 'paced_sleep_seconds': pacer.slept,
                'elapsed_seconds': clock()-pacer.started, 'pending_path': str(pending),
                'sqlite_integrity_verified': False, 'canonical_prefix_verified': False,
                'cross_store_checkpoint_verified': False, 'deployment_backup_accepted': False,
                'physical_io_cap_claimed': False, 'source_write_requested': False}
    finally:
        # Close read-only handles even if progress aborted. Delete only the
        # uniquely owned incomplete staging name, never source/prior backups.
        if 'src' in locals():
            src.close()
        if owned and temporary.exists():
            temporary.unlink()

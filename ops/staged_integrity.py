"""Bounded read-only quick-check of a privately staged SQLite copy.

This is a single-copy gate, not a full integrity or deployment certificate.
Caller must serialize access in a private staging directory. No live database
may be passed. Current admission limit is 8 MiB; larger stores remain pending.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import sqlite3
import stat
import time
from pathlib import Path

from ops.incremental_backup import BackupStopped, _Pacer

MAX_BYTES = 8 * 1024**2


def verify_staged_sqlite(pending, receipt, *, continue_check, max_bytes=MAX_BYTES,
                         vm_budget=100000, deadline_seconds=60,
                         rate_bytes_per_second=256*1024,
                         clock=time.monotonic, sleep=time.sleep):
    """Bind a staged readback receipt to quick_check(1), failing closed.

    Digest reads are logically paced. SQLite work is limited by input size,
    progress instructions, admission and elapsed time; these are NOT physical
    IO/RAM caps or a hard interruption of an OS call. Raw SQLite errors and paths
    are never included in results. The input and receipt must be private/trusted.
    """
    if (type(max_bytes) is not int or not 1 <= max_bytes <= MAX_BYTES
            or type(vm_budget) is not int or not 100 <= vm_budget <= 1000000
            or type(deadline_seconds) not in (int, float)
            or not math.isfinite(deadline_seconds) or not 0 < deadline_seconds <= 60
            or type(rate_bytes_per_second) is not int
            or not 1 <= rate_bytes_per_second <= 1024**2):
        raise ValueError('Bounded staged verification policy required')
    path = Path(pending).absolute()
    if (not isinstance(receipt, dict)
            or receipt.get('status') != 'SINGLE_SQLITE_COPY_READBACK_VERIFIED_INTEGRITY_PENDING'
            or receipt.get('source_write_requested') is not False
            or receipt.get('deployment_backup_accepted') is not False
            or type(receipt.get('bytes')) is not int
            or not 1 <= receipt['bytes'] <= max_bytes
            or not isinstance(receipt.get('sha256'), str)
            or not re.fullmatch('[0-9a-f]{64}', receipt['sha256'])
            or Path(receipt.get('pending_path', '')).absolute() != path):
        raise ValueError('Exact pending readback receipt required')
    pacer = _Pacer(rate_bytes_per_second, deadline_seconds, continue_check,
                   lambda: 1, clock, sleep, 0)

    def identity():
        pacer.guard()
        info = path.lstat()
        if (path.suffix != '.pending' or not stat.S_ISREG(info.st_mode)
                or path.is_symlink() or getattr(info, 'st_file_attributes', 0) & 0x400
                or info.st_size != receipt['bytes']):
            raise BackupStopped('staged_file_identity')
        # Immutable readers must never silently ignore committed sidecar data.
        if any(Path(str(path)+suffix).exists() for suffix in ('-wal', '-shm', '-journal')):
            raise BackupStopped('staged_sidecar_present')
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns

    before = identity()
    instructions = 0
    stopped = None
    db = None
    try:
        db = sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1', uri=True, timeout=0)
        db.execute('PRAGMA query_only=ON')
        db.execute('PRAGMA mmap_size=0')
        db.execute('PRAGMA cache_size=-256')
        def progress():
            nonlocal instructions, stopped
            instructions += 100
            try:
                pacer.guard()
                if instructions > vm_budget:
                    raise BackupStopped('sqlite_vm_budget')
            except BackupStopped as exc:
                stopped = exc
                return 1
            return 0
        db.set_progress_handler(progress, 100)
        rows = db.execute('PRAGMA quick_check(1)').fetchmany(2)
        pacer.guard()
        if rows != [('ok',)]:
            raise BackupStopped('sqlite_quick_check_failed')
    except sqlite3.Error:
        raise stopped or BackupStopped('sqlite_verification_failed') from None
    finally:
        if db is not None:
            db.close()
    # Hash after SQLite inspection so the certificate binds checked file bytes.
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != before:
            raise BackupStopped('staged_file_changed')
        remaining = receipt['bytes']
        while remaining:
            pacer.guard()
            block = stream.read(min(65536, remaining))
            if not block:
                raise BackupStopped('staged_file_changed')
            digest.update(block); remaining -= len(block); pacer.charge(len(block))
        if stream.read(1):
            raise BackupStopped('staged_file_changed')
    if identity() != before:
        raise BackupStopped('staged_file_changed')
    if digest.hexdigest() != receipt['sha256']:
        raise BackupStopped('staged_digest_mismatch')
    pacer.guard()
    return {'status': 'SINGLE_STAGED_SQLITE_QUICK_CHECK_VERIFIED',
            'bytes': receipt['bytes'], 'sha256': digest.hexdigest(),
            'sqlite_quick_check_verified': True, 'sqlite_full_integrity_verified': False,
            'sqlite_vm_instructions_charged': instructions,
            'digest_bytes_charged': pacer.charged, 'paced_sleep_seconds': pacer.slept,
            'elapsed_seconds': clock()-pacer.started,
            'canonical_prefix_verified': False, 'cross_store_checkpoint_verified': False,
            'deployment_backup_accepted': False, 'source_write_requested': False,
            'physical_io_cap_claimed': False}

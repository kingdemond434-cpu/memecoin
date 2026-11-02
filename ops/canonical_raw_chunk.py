"""Preserve a bounded raw canonical range, including partial/oversized rows."""
import hashlib, math, os, shutil, stat, time, uuid
from pathlib import Path
from ops.incremental_backup import BackupStopped, RESERVE
from ops.worker_resource_guard import verify


def stage_raw_chunk(source, pending, *, offset, extent, continue_check,
                    max_bytes=1048576, rate_bytes_per_second=256*1024,
                    deadline_seconds=60, resource_check=verify, free_bytes=None,
                    clock=time.monotonic, sleep=time.sleep):
    """One range only; no parsing, skipping, chain cursor or recovery authority.

    Caller authenticates the append-only source, captured extent and contiguous
    range history. Trusted private ancestors are required. Cooperative pacing
    charges source reads, writes and readback, not physical IO or RAM.
    """
    if (type(offset) is not int or type(extent) is not int or not 0 <= offset < extent
            or type(max_bytes) is not int or not 1 <= max_bytes <= 1048576
            or type(rate_bytes_per_second) is not int or not 1 <= rate_bytes_per_second <= 1048576
            or type(deadline_seconds) not in (int, float) or not math.isfinite(deadline_seconds)
            or not 0 < deadline_seconds <= 60):
        raise ValueError('Exact bounded raw range policy required')
    cpu=resource_check()
    if cpu.get('status') != 'VERIFIED' or cpu.get('cpu_rate') != 200:
        raise RuntimeError('Development CPU proof required')
    source=Path(source).absolute(); pending=Path(pending).absolute(); parent=pending.parent
    if (not pending.name.endswith('.raw.pending') or not parent.is_dir()
            or parent.is_symlink() or getattr(parent.lstat(),'st_file_attributes',0)&0x400
            or parent.resolve()==source.resolve().parent
            or source.resolve().parent in parent.resolve().parents):
        raise ValueError('Separate trusted raw staging directory required')
    info=source.lstat()
    if (not stat.S_ISREG(info.st_mode) or source.is_symlink()
            or getattr(info,'st_file_attributes',0)&0x400):
        raise ValueError('Regular canonical source required')
    if os.path.lexists(pending): raise FileExistsError('Pending raw chunk already exists')
    size=min(max_bytes,extent-offset)
    if free_bytes is None: free_bytes=lambda:shutil.disk_usage(parent).free
    started=clock(); charged=0; temporary=None; owned=False
    def guard():
        if (clock()-started>=deadline_seconds or continue_check() is not True
                or free_bytes()-size<RESERVE):
            raise BackupStopped('raw_chunk_resource_gate')
    def charge(amount):
        nonlocal charged
        charged+=amount
        delay=charged/rate_bytes_per_second-(clock()-started)
        while delay>1e-9:
            guard(); sleep(min(.1,delay))
            delay=charged/rate_bytes_per_second-(clock()-started)
        guard()
    def identity(a,b): return (a.st_dev,a.st_ino)==(b.st_dev,b.st_ino)
    guard()
    # Shared with verified segment staging: no competing canonical writers here.
    lock=parent/'canonical-segment.lock'
    with lock.open('x',encoding='utf-8') as stream: stream.write('exclusive raw chunk writer\n')
    try:
        guard(); temporary=parent/('raw-'+uuid.uuid4().hex+'.partial'); digest=hashlib.sha256()
        with source.open('rb') as original:
            before=os.fstat(original.fileno())
            if not identity(info,before) or before.st_size<extent:
                raise BackupStopped('raw_source_identity_or_extent')
            with temporary.open('xb') as output:
                owned=True; original.seek(offset); remaining=size
                while remaining:
                    guard(); block=original.read(min(65536,remaining))
                    if not block: raise BackupStopped('raw_source_truncated')
                    output.write(block); digest.update(block); remaining-=len(block); charge(2*len(block))
                output.flush(); os.fsync(output.fileno())
            after=os.fstat(original.fileno())
        current=source.stat()
        if (not identity(before,after) or not identity(before,current)
                or min(after.st_size,current.st_size)<extent):
            raise BackupStopped('raw_source_changed')
        readback=hashlib.sha256(); saved=temporary.stat(); read=0
        with temporary.open('rb') as stream:
            while read<size:
                guard(); block=stream.read(min(65536,size-read))
                if not block: raise BackupStopped('raw_readback_truncated')
                readback.update(block); read+=len(block); charge(len(block))
            if stream.read(1): raise BackupStopped('raw_readback_grew')
        final=temporary.stat()
        if (not identity(saved,final) or (saved.st_size,saved.st_mtime_ns)!=(final.st_size,final.st_mtime_ns)
                or final.st_size!=size or readback.digest()!=digest.digest()):
            raise BackupStopped('raw_readback_mismatch')
        guard(); os.link(temporary,pending)
        return dict(status='SINGLE_RAW_CANONICAL_RANGE_READBACK_VERIFIED',offset=offset,
                    end_offset=offset+size,captured_extent=extent,bytes=size,sha256=digest.hexdigest(),
                    pending_path=str(pending),pending_published=True,logical_bytes_charged=charged,
                    elapsed_seconds=clock()-started,chain_verified=False,record_boundary_verified=False,
                    prefix_anchor_authenticated=False,cross_store_checkpoint_verified=False,
                    recovery_verified=False,deployment_backup_accepted=False,
                    source_write_requested=False,physical_io_cap_claimed=False)
    finally:
        if owned and temporary is not None and temporary.exists(): temporary.unlink()
        lock.unlink()

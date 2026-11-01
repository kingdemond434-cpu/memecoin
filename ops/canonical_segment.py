"""Materialize one bounded verified canonical segment, never a whole backup."""
import hashlib,os,shutil,time,uuid
from pathlib import Path
from ops.canonical_chain_window import verify_chain_window
from ops.incremental_backup import BackupStopped,RESERVE
from ops.worker_resource_guard import verify


def stage_chain_segment(source,pending,*,cursor,continue_check,max_bytes=1048576,
                        max_records=256,rate_bytes_per_second=256*1024,
                        deadline_seconds=60,resource_check=verify,free_bytes=None,
                        clock=time.monotonic,sleep=time.sleep):
    """Trusted private destination/ancestors and append-only source required.

    One cooperative directory writer, exclusive publication, no automatic loop.
    Source/SQLite coherence, prior-prefix trust and recovery remain separate.
    Logical pacing and deadlines do not cap physical IO/RAM or preempt syscalls.
    """
    source=Path(source).absolute();pending=Path(pending).absolute();parent=pending.parent
    if (not pending.name.endswith('.segment.pending') or not parent.is_dir()
            or parent.is_symlink() or getattr(parent.lstat(),'st_file_attributes',0)&0x400
            or parent.resolve()==source.resolve().parent
            or source.resolve().parent in parent.resolve().parents):
        raise ValueError('Separate trusted segment staging directory required')
    if os.path.lexists(pending):raise FileExistsError('Pending segment already exists')
    if free_bytes is None:free_bytes=lambda:shutil.disk_usage(parent).free
    started=clock();charged=0;temporary=None;owned=False
    def admitted():
        return (clock()-started<deadline_seconds and continue_check() is True
                and free_bytes()-max_bytes>=RESERVE)
    def guard():
        if not admitted():raise BackupStopped('segment_resource_gate')
    def charge(amount):
        nonlocal charged
        charged+=amount;delay=charged/rate_bytes_per_second-(clock()-started)
        while delay>0:
            guard();sleep(min(.1,delay));delay=charged/rate_bytes_per_second-(clock()-started)
        guard()
    lock=parent/'canonical-segment.lock'
    with lock.open('x',encoding='utf-8') as stream:stream.write('exclusive segment writer\n')
    try:
        result=verify_chain_window(source,cursor=cursor,continue_check=admitted,
            max_bytes=max_bytes,max_records=max_records,rate_bytes_per_second=rate_bytes_per_second,
            deadline_seconds=deadline_seconds,resource_check=resource_check,clock=clock,sleep=sleep)
        charged=result['bytes_read']
        if result['status']!='BOUNDED_CHAIN_EXTENSION_VERIFIED':
            return dict(status='CHAIN_SEGMENT_NOT_STAGED',verification=result,
                        pending_published=False,deployment_backup_accepted=False)
        end=result['next_cursor']['offset'];size=end-cursor['offset'];guard()
        if not 0<size<=max_bytes:raise BackupStopped('verified_segment_size')
        temporary=parent/('segment-'+uuid.uuid4().hex+'.partial')
        digest=hashlib.sha256()
        with source.open('rb') as original,temporary.open('xb') as output:
            owned=True;before=os.fstat(original.fileno());original.seek(cursor['offset']);remaining=size
            while remaining:
                guard();block=original.read(min(65536,remaining))
                if not block:raise BackupStopped('segment_source_truncated')
                output.write(block);digest.update(block);remaining-=len(block);charge(2*len(block))
            after=os.fstat(original.fileno());output.flush();os.fsync(output.fileno())
        if ((before.st_dev,before.st_ino)!=(after.st_dev,after.st_ino)
                or min(before.st_size,after.st_size)<end
                or digest.hexdigest()!=result['verified_segment_sha256']):
            raise BackupStopped('segment_source_changed')
        readback=hashlib.sha256();read=0;info=temporary.stat()
        with temporary.open('rb') as stream:
            while read<size:
                guard();block=stream.read(min(65536,size-read))
                if not block:raise BackupStopped('segment_readback_truncated')
                read+=len(block);readback.update(block);charge(len(block))
            if stream.read(1):raise BackupStopped('segment_readback_grew')
        after=temporary.stat()
        if ((info.st_size,info.st_mtime_ns)!=(after.st_size,after.st_mtime_ns)
                or info.st_size!=size or readback.hexdigest()!=result['verified_segment_sha256']):
            raise BackupStopped('segment_readback_mismatch')
        guard();os.link(temporary,pending)
        return dict(status='SINGLE_CANONICAL_SEGMENT_READBACK_VERIFIED',bytes=size,
            sha256=readback.hexdigest(),start_cursor=dict(cursor),next_cursor=result['next_cursor'],
            verification=result,pending_path=str(pending),pending_published=True,
            logical_bytes_charged=charged,elapsed_seconds=clock()-started,
            full_prefix_chain_verified=False,cross_store_checkpoint_verified=False,
            recovery_verified=False,deployment_backup_accepted=False,source_write_requested=False,
            physical_io_cap_claimed=False)
    finally:
        if owned and temporary is not None and temporary.exists():temporary.unlink()
        lock.unlink()

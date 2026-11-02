"""Bounded allowlisted source timings from an existing private readiness file."""
import json,math,os,stat,time
from pathlib import Path
from ops.worker_resource_guard import verify
from src.runtime.source_timing import PHASES
from src.runtime.readiness_timing import PHASES as REFRESH_PHASES

SCOPE='process_local_service_time_not_source_latency'
COUNTS=('attempts','completed','interrupted')
TIMES=('total_seconds','max_seconds','last_seconds')
MAX_BYTES=1048576


def read_source_phase_snapshot(path,*,continue_check,resource_check=verify,
                               now=time.time,clock=time.monotonic,sleep=time.sleep,
                               max_age_seconds=180):
    """No HTTP, source payload, provider calls, writes or automatic loop.

    Caller owns trusted private ancestors. Pacing is logical, not an IO/RAM cap;
    30-second deadline is cooperative, not filesystem-call preemption.
    Freshness and phase times do not establish HTTP health or stall causality.
    """
    if type(max_age_seconds) not in (int,float) or not math.isfinite(max_age_seconds) or not 0<max_age_seconds<=86400:
        raise ValueError('Finite snapshot freshness policy required')
    cpu=resource_check()
    if cpu.get('status')!='VERIFIED' or cpu.get('cpu_rate')!=200:
        raise RuntimeError('Development CPU proof required')
    started=clock()
    def guard():
        if clock()-started>=30 or continue_check() is not True:raise RuntimeError('Snapshot admission stopped')
    def identity(info):return info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns
    guard();path=Path(path);before=path.lstat()
    if (path.is_symlink() or not stat.S_ISREG(before.st_mode)
            or getattr(before,'st_file_attributes',0)&0x400 or not 0<before.st_size<=MAX_BYTES):
        raise ValueError('Regular bounded readiness snapshot required')
    chunks=[];read=0
    with path.open('rb') as stream:
        if identity(os.fstat(stream.fileno()))!=identity(before):raise ValueError('Snapshot identity changed')
        while read<before.st_size:
            guard();block=stream.read(min(65536,before.st_size-read))
            if not block:raise ValueError('Snapshot truncated')
            chunks.append(block);read+=len(block)
            delay=read/262144-(clock()-started)
            while delay>0:
                guard();sleep(min(.1,delay));delay=read/262144-(clock()-started)
        if identity(os.fstat(stream.fileno()))!=identity(before):raise ValueError('Snapshot changed')
    if identity(path.lstat())!=identity(before):raise ValueError('Snapshot replaced')
    def pairs(values):
        row={}
        for key,value in values:
            if key in row:raise ValueError('Duplicate snapshot key')
            row[key]=value
        return row
    data=json.loads(b''.join(chunks),object_pairs_hook=pairs)
    if not isinstance(data,dict):raise ValueError('Snapshot object required')
    timing=data.get('source_phase_timing');phases={};invalid=[]
    available=isinstance(timing,dict) and timing.get('scope')==SCOPE and isinstance(timing.get('phases'),dict)
    if available:
        for name in PHASES:
            if name not in timing['phases']:continue
            row=timing['phases'][name]
            if (not isinstance(row,dict)
                    or any(type(row.get(k)) is not int or row[k]<0 for k in COUNTS)
                    or any(type(row.get(k)) not in (int,float) or not math.isfinite(row[k]) or row[k]<0 for k in TIMES)
                    or row['completed']+row['interrupted']!=row['attempts']):
                invalid.append(name);continue
            phases[name]={k:row[k] for k in COUNTS+TIMES}
    guard();observed=now()
    if type(observed) not in (int,float) or not math.isfinite(observed):raise ValueError('Finite observation clock required')
    age=observed-before.st_mtime
    diagnostic=data.get('diagnostic_snapshot')
    diagnostic=diagnostic if isinstance(diagnostic,dict) else {}
    captured=diagnostic.get('observed_at')
    captured=captured if type(captured) in (int,float) and math.isfinite(captured) and captured>=0 else None
    projection_age=observed-captured if captured is not None else None
    wait_captured=diagnostic.get('refresh_wait_observed_at')
    wait_captured=wait_captured if type(wait_captured) in (int,float) and math.isfinite(wait_captured) and wait_captured>=0 else None
    def elapsed(key):
        value=diagnostic.get(key)
        return value if type(value) in (int,float) and math.isfinite(value) and value>=0 else None
    return dict(status='READINESS_SOURCE_PHASE_SAMPLE',observed_at=observed,bytes_read=read,
        snapshot_age_seconds=age,snapshot_fresh=0<=age<=max_age_seconds,
        max_age_seconds=max_age_seconds,clock_skew_detected=age<0,
        live_submission_locked=data.get('live_submission_locked') if type(data.get('live_submission_locked')) is bool else None,
        projection_observed_at=captured,projection_age_seconds=projection_age,
        projection_fresh=(0<=projection_age<=max_age_seconds) if projection_age is not None else None,
        projection_refresh_in_progress=diagnostic.get('refresh_in_progress') if type(diagnostic.get('refresh_in_progress')) is bool else None,
        projection_refresh_error_present=diagnostic.get('refresh_error_type') is not None,
        refresh_phase=diagnostic.get('refresh_phase') if diagnostic.get('refresh_phase') in REFRESH_PHASES else None,
        refresh_pending_seconds=elapsed('refresh_pending_seconds'),
        refresh_phase_seconds=elapsed('refresh_phase_seconds'),
        refresh_wait_observed_at=wait_captured,
        refresh_wait_age_seconds=observed-wait_captured if wait_captured is not None else None,
        scope=SCOPE,timing_available=available,phases=phases,invalid_phases=invalid,
        http_health_verified=False,stall_cause_established=False,source_latency_verified=False,
        source_write_requested=False,physical_io_cap_claimed=False)

"""One bounded new canonical window and a pinned history attempt per call."""
import hashlib,json,math,os,shutil,stat,time,uuid
from pathlib import Path
import ops.canonical_fragment_progress as history
from ops.canonical_raw_chunk import stage_raw_chunk
from ops.incremental_backup import RESERVE
from ops.worker_resource_guard import verify


def _encoded(value):
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if len(raw)>65536:raise ValueError('Attempt metadata budget exceeded')
    return raw


def _publish(path,value,expected):
    raw=_encoded(value);temporary=path.parent/('attempt-'+uuid.uuid4().hex+'.partial')
    try:
        with temporary.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        if expected is None:
            if os.path.lexists(path):raise FileExistsError('Existing attempt preserved')
            os.link(temporary,path)
        else:
            if hashlib.sha256(history._read(path)).hexdigest()!=expected:raise ValueError('Attempt changed')
            os.replace(temporary,path)
        return hashlib.sha256(raw).hexdigest()
    finally:
        if temporary.exists():temporary.unlink()


def _regular(path):
    info=path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink() or getattr(info,'st_file_attributes',0)&0x400:
        raise ValueError('Regular private source or raw file required')
    return info


def _identity(info):return (info.st_dev,info.st_ino)
def _signature(info):return (_identity(info),info.st_size,info.st_mtime_ns)


def _adopt(source,pending,offset,size,extent,source_identity,guard,rate,clock,sleep):
    """No byte writes: compare exact source range and fsync the retained file."""
    before=_regular(pending);src=_regular(source)
    if before.st_size!=size or _identity(src)!=source_identity or src.st_size<extent:
        raise ValueError('Orphan size or source identity mismatch')
    started=clock();charged=0;left=size;original=hashlib.sha256();saved=hashlib.sha256()
    with source.open('rb') as stream,pending.open('r+b') as retained:
        if _identity(os.fstat(stream.fileno()))!=source_identity or _signature(os.fstat(retained.fileno()))!=_signature(before):
            raise ValueError('Orphan identity changed')
        stream.seek(offset)
        while left:
            guard();amount=min(65536,left);a=stream.read(amount);b=retained.read(amount)
            if len(a)!=amount or len(b)!=amount:raise ValueError('Orphan or captured source truncated')
            original.update(a);saved.update(b);left-=amount;charged+=2*amount
            delay=charged/rate-(clock()-started)
            while delay>1e-9:
                guard();sleep(min(.1,delay));delay=charged/rate-(clock()-started)
        if original.digest()!=saved.digest() or retained.read(1):raise ValueError('Orphan source readback mismatch')
        guard();retained.flush();os.fsync(retained.fileno())
        if _signature(os.fstat(retained.fileno()))!=_signature(before):raise ValueError('Orphan changed')
    if (_signature(_regular(pending))!=_signature(before) or _identity(_regular(source))!=source_identity
            or source.stat().st_size<extent):raise ValueError('Orphan or source changed')
    return dict(status='SINGLE_RAW_CANONICAL_RANGE_READBACK_VERIFIED',pending_path=str(pending),offset=offset,
        end_offset=offset+size,bytes=size,captured_extent=extent,sha256=saved.hexdigest(),
        adopted_existing=True,bytes_written=0,source_write_requested=False)


def advance_fragment_step(source,progress_path,*,anchor,anchor_evidence_sha256,target_extent,
                          expected_progress_sha256,continue_check,max_records=64,window_bytes=1048576,
                          rate_bytes_per_second=256*1024,deadline_seconds=60,
                          resource_check=verify,free_bytes=None,clock=time.monotonic,sleep=time.sleep):
    """At most one new <=1MiB raw copy; retain at most two inputs per attempt.

    Trusted append-only source/private ancestors required. No automatic loop,
    unpinned reconciliation, orphan deletion, history rollover or deployment.
    File fsync and bounded pacing do not prove full recoverability or cap RAM/IO.
    """
    anchor=history._cursor(anchor);history._hash(anchor_evidence_sha256);history._hash(expected_progress_sha256)
    if (type(target_extent) is not int or target_extent<=anchor['offset']
            or type(window_bytes) is not int or not 1<=window_bytes<=1048576
            or type(max_records) is not int or not 1<=max_records<=256
            or type(rate_bytes_per_second) is not int or not 1<=rate_bytes_per_second<=1048576
            or type(deadline_seconds) not in (int,float) or not math.isfinite(deadline_seconds)
            or not 0<deadline_seconds<=60):raise ValueError('Bounded recovery step policy required')
    cpu=resource_check()
    if cpu.get('status')!='VERIFIED' or cpu.get('cpu_rate')!=200:raise RuntimeError('Development CPU proof required')
    source=Path(source).absolute();path=Path(progress_path).absolute();root=path.parent
    if (not path.name.endswith('.fragment-progress.json') or not root.is_dir() or root.is_symlink()
            or getattr(root.lstat(),'st_file_attributes',0)&0x400):raise ValueError('Private progress directory required')
    if free_bytes is None:free_bytes=lambda:shutil.disk_usage(root).free
    started=clock();copied=0;adopted=False
    def guard():
        if (clock()-started>=deadline_seconds or continue_check() is not True
                or free_bytes()-window_bytes-65536<RESERVE):raise RuntimeError('Recovery step resource gate stopped')
    def remaining():
        guard();return deadline_seconds-(clock()-started)
    guard();lock=root/'canonical-step.lock'
    with lock.open('x',encoding='utf-8') as stream:stream.write('bounded step pid='+str(os.getpid())+'\n')
    try:
        raw=history._read(path)
        if hashlib.sha256(raw).hexdigest()!=expected_progress_sha256:raise ValueError('Progress pin mismatch; reconciliation required')
        state=json.loads(raw,object_pairs_hook=history._pairs)
        cursor=history._state(state,anchor,anchor_evidence_sha256,target_extent,root)
        common=dict(baseline_authenticated=False,full_prefix_chain_verified=False,cross_store_checkpoint_verified=False,
                    recovery_verified=False,deployment_backup_accepted=False,source_write_requested=False,
                    physical_io_cap_claimed=False)
        if cursor['offset']==target_extent:
            return dict(status='CAPTURE_TIP_REACHED_HISTORY_ONLY',next_cursor=cursor,
                        progress_sha256=expected_progress_sha256,new_source_bytes_copied=0,**common)
        if len(state['steps'])>=32:raise ValueError('History budget exhausted; no rollover or eviction')
        src=_regular(source)
        if src.st_size<target_extent:raise ValueError('Captured source extent unavailable')
        source_identity=_identity(src)
        binding=dict(version=1,progress_sha256=expected_progress_sha256,cursor=cursor,anchor=anchor,
            anchor_evidence_sha256=anchor_evidence_sha256,target_extent=target_extent,window_bytes=window_bytes,
            source_path_sha256=hashlib.sha256(str(source.resolve()).encode()).hexdigest(),source_identity=list(source_identity))
        key=hashlib.sha256(_encoded(binding)).hexdigest();journal=root/('attempt-'+key+'.fragment-attempt.json')
        if os.path.lexists(journal):
            old=history._read(journal);attempt=json.loads(old,object_pairs_hook=history._pairs)
            if (not isinstance(attempt,dict) or set(attempt)!={'version','binding','inputs'}
                    or type(attempt['version']) is not int or attempt['version']!=1
                    or _encoded(attempt['binding'])!=_encoded(binding) or type(attempt['inputs']) is not list
                    or len(attempt['inputs'])>2):raise ValueError('Attempt binding mismatch')
            journal_pin=hashlib.sha256(old).hexdigest()
        else:
            attempt=dict(version=1,binding=binding,inputs=[]);guard();journal_pin=_publish(journal,attempt,None)
        if attempt['inputs']:
            history._inputs(attempt['inputs'],root,cursor,target_extent,stored=True)
            offset=cursor['offset']
            for i,item in enumerate(attempt['inputs']):
                if (item['name']!='step-'+key+'-'+str(i)+'.raw.pending' or item['offset']!=offset
                        or item['bytes']!=min(window_bytes,target_extent-offset)):
                    raise ValueError('Attempt raw interval mismatch')
                offset=item['end_offset']
        def record():
            receipts=[dict(status='SINGLE_RAW_CANONICAL_RANGE_READBACK_VERIFIED',pending_path=str(root/i['name']),
                           **{k:v for k,v in i.items() if k!='name'}) for i in attempt['inputs']]
            return record_fragment_step(path,anchor=anchor,anchor_evidence_sha256=anchor_evidence_sha256,
                target_extent=target_extent,receipts=receipts,expected_progress_sha256=expected_progress_sha256,
                continue_check=continue_check,max_records=max_records,resource_check=resource_check,
                free_bytes=free_bytes,deadline_seconds=remaining(),clock=clock,sleep=sleep)
        def finish(result):
            status=('RECOVERY_STEP_RECORDED' if result['status']=='FRAGMENT_STEP_RECORDED' else
                    'RECOVERY_WINDOW_RETAINED' if result['verification']['status']=='NO_COMPLETE_FRAGMENTED_ROW' else
                    'RECOVERY_STEP_NOT_RECORDED')
            return dict(status=status,history_result=result,next_cursor=result['next_cursor'],
                progress_sha256=result['progress_sha256'],attempt_path=str(journal),attempt_sha256=journal_pin,
                retained_inputs=len(attempt['inputs']),new_source_bytes_copied=copied,adopted_existing=adopted,
                bounded_row_window_exhausted=status=='RECOVERY_WINDOW_RETAINED' and len(attempt['inputs'])==2,
                elapsed_seconds=clock()-started,**common)
        if attempt['inputs']:
            result=record()
            if (result['status']=='FRAGMENT_STEP_RECORDED' or result['verification']['status']!='NO_COMPLETE_FRAGMENTED_ROW'
                    or len(attempt['inputs'])==2 or attempt['inputs'][-1]['end_offset']==target_extent):
                return finish(result)
        index=len(attempt['inputs']);offset=attempt['inputs'][-1]['end_offset'] if index else cursor['offset']
        pending=root/('step-'+key+'-'+str(index)+'.raw.pending');size=min(window_bytes,target_extent-offset)
        guard()
        if os.path.lexists(pending):
            receipt=_adopt(source,pending,offset,size,target_extent,source_identity,guard,
                           rate_bytes_per_second,clock,sleep);adopted=True
        else:
            receipt=stage_raw_chunk(source,pending,offset=offset,extent=target_extent,max_bytes=window_bytes,
                continue_check=continue_check,rate_bytes_per_second=rate_bytes_per_second,deadline_seconds=remaining(),
                resource_check=resource_check,free_bytes=free_bytes,clock=clock,sleep=sleep)
            copied=receipt['bytes']
        if _identity(_regular(source))!=source_identity:raise ValueError('Source identity changed')
        projected=dict(name=pending.name,**{k:receipt[k] for k in ('offset','end_offset','bytes','captured_extent','sha256')})
        attempt['inputs'].append(projected);guard();journal_pin=_publish(journal,attempt,journal_pin)
        return finish(record())
    finally:
        lock.unlink()


# Kept as a module binding for isolated publication-fault injection tests.
record_fragment_step=history.record_fragment_step

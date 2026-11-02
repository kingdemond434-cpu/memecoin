"""Finite admitted recovery work with a durable checkpoint between each step."""
import hashlib,json,math,os,shutil,time
from pathlib import Path
import ops.canonical_fragment_progress as history
from ops.canonical_recovery_step import advance_fragment_step,_publish
from ops.incremental_backup import RESERVE
from ops.worker_resource_guard import verify


def run_fragment_tick(source,progress_path,tick_path,*,anchor,anchor_evidence_sha256,target_extent,
                      expected_progress_sha256,continue_check,max_steps=4,max_new_bytes=1048576,
                      max_records=64,deadline_seconds=60,rollover_link=None,resource_check=verify,
                      free_bytes=None,clock=time.monotonic,sleep=time.sleep):
    """1..4 calls, <=2MiB copied source, <=60s cooperative deadline. New journal only.

    Existing progress/raw data are never reset. No automatic rollover or daemon.
    Failure with a step in flight requires explicit reconciliation, not blind retry.
    Checkpoint writes may finish after admission expires; syscalls are not preempted.
    """
    anchor=history._cursor(anchor);history._hash(anchor_evidence_sha256);history._hash(expected_progress_sha256)
    if (type(max_steps) is not int or not 1<=max_steps<=4
            or type(max_new_bytes) is not int or not 1<=max_new_bytes<=2097152
            or type(max_records) is not int or not 1<=max_records<=256
            or type(target_extent) is not int or target_extent<=anchor['offset']
            or type(deadline_seconds) not in (int,float) or not math.isfinite(deadline_seconds)
            or not 0<deadline_seconds<=60):raise ValueError('Bounded recovery tick policy required')
    cpu=resource_check()
    if cpu.get('status')!='VERIFIED' or cpu.get('cpu_rate')!=200:raise RuntimeError('Development CPU proof required')
    source=Path(source).absolute();path=Path(progress_path).absolute();tick=Path(tick_path).absolute();root=path.parent
    if (not path.name.endswith('.fragment-progress.json') or not tick.name.endswith('.fragment-tick.json')
            or tick.parent.resolve()!=root.resolve() or not root.is_dir() or root.is_symlink()
            or getattr(root.lstat(),'st_file_attributes',0)&0x400):raise ValueError('Private sibling tick journal required')
    if free_bytes is None:free_bytes=lambda:shutil.disk_usage(root).free
    started=clock();copied=0;pin=expected_progress_sha256;journal_pin=None;unresolved_copy=0
    def admitted():
        return (clock()-started<deadline_seconds and continue_check() is True
                and free_bytes()-1048576-2*history.MAX_METADATA>=RESERVE)
    if not admitted():raise RuntimeError('Tick resource gate stopped')
    lock=root/'canonical-tick.lock'
    with lock.open('x',encoding='utf-8') as stream:stream.write('bounded recovery tick writer\n')
    try:
        if os.path.lexists(tick):raise FileExistsError('Existing tick preserved; explicit reconciliation required')
        if rollover_link is not None:
            from ops.canonical_fragment_rollover import load_rollover_inputs
            if Path(rollover_link).absolute()!=path.with_name(path.name+'.rollover.json'):
                raise ValueError('Exact sibling rollover link required')
            def admission_guard():
                if not admitted():raise RuntimeError('Tick resource gate stopped')
            load_rollover_inputs(path,anchor=anchor,anchor_evidence_sha256=anchor_evidence_sha256,
                target_extent=target_extent,guard=admission_guard)
        original=history._read(path)
        if hashlib.sha256(original).hexdigest()!=pin:raise ValueError('Tick progress pin mismatch')
        state=json.loads(original,object_pairs_hook=history._pairs)
        cursor=history._state(state,anchor,anchor_evidence_sha256,target_extent,root)
        manifest=dict(version=1,progress_name=path.name,source_path_sha256=hashlib.sha256(str(source.resolve()).encode()).hexdigest(),
            anchor=anchor,anchor_evidence_sha256=anchor_evidence_sha256,target_extent=target_extent,
            initial_progress_sha256=pin,limits=dict(max_steps=max_steps,max_new_bytes=max_new_bytes,
            max_records=max_records,deadline_seconds=deadline_seconds),state='STARTED',steps=[])
        journal_pin=_publish(tick,manifest,None)
        def save():return _publish(tick,manifest,journal_pin)
        def finish(status,error_type=None):
            return dict(status=status,tick_path=str(tick),tick_sha256=journal_pin,
                last_checkpointed_progress_sha256=pin,last_checkpointed_cursor=cursor,
                checkpointed_steps=len(manifest['steps']),source_copy_bytes_confirmed=copied,
                unresolved_step_copy_upper_bound=unresolved_copy,error_type=error_type,
                elapsed_seconds=clock()-started,source_write_requested=False,physical_io_cap_claimed=False,
                baseline_authenticated=False,full_prefix_chain_verified=False,cross_store_checkpoint_verified=False,
                recovery_verified=False,deployment_backup_accepted=False,automatic_continuation=False)
        try:
            for index in range(max_steps):
                if not admitted():status='ADMISSION_STOPPED';break
                if cursor['offset']==target_extent:status='CAPTURE_TIP_HISTORY_ONLY';break
                if len(state['steps'])==32:status='HISTORY_LIMIT_REACHED';break
                remaining=max_new_bytes-copied
                if remaining<=0:status='SOURCE_COPY_BUDGET_REACHED';break
                window=min(1048576,remaining)
                manifest.update(state='STEP_IN_FLIGHT',in_flight=dict(index=index,progress_sha256=pin,copy_upper_bound=window))
                journal_pin=save();unresolved_copy=window
                result=advance_fragment_step(source,path,anchor=anchor,anchor_evidence_sha256=anchor_evidence_sha256,
                    target_extent=target_extent,expected_progress_sha256=pin,continue_check=continue_check,
                    max_records=max_records,window_bytes=window,rollover_link=rollover_link,
                    deadline_seconds=deadline_seconds-(clock()-started),resource_check=resource_check,
                    free_bytes=free_bytes,clock=clock,sleep=sleep)
                next_pin=history._hash(result['progress_sha256']);new_bytes=result['new_source_bytes_copied']
                if type(new_bytes) is not int or not 0<=new_bytes<=window:raise ValueError('Step copy budget mismatch')
                current=history._read(path)
                if hashlib.sha256(current).hexdigest()!=next_pin:raise ValueError('Step progress changed before checkpoint')
                next_state=json.loads(current,object_pairs_hook=history._pairs)
                next_cursor=history._state(next_state,anchor,anchor_evidence_sha256,target_extent,root)
                if next_cursor!=result['next_cursor']:raise ValueError('Step cursor mismatch')
                rows=next_cursor['sequence']-cursor['sequence'];size=next_cursor['offset']-cursor['offset']
                if result['status']=='RECOVERY_STEP_RECORDED':
                    if not 1<=rows<=max_records or size<=0 or len(next_state['steps'])!=len(state['steps'])+1:
                        raise ValueError('Recorded step delta mismatch')
                elif (result['status'] not in ('RECOVERY_WINDOW_RETAINED','RECOVERY_STEP_NOT_RECORDED','CAPTURE_TIP_REACHED_HISTORY_ONLY')
                        or next_pin!=pin or next_cursor!=cursor):raise ValueError('Unrecorded step changed history')
                step=dict(status=result['status'],prior_progress_sha256=pin,
                    progress_sha256=next_pin,next_cursor=next_cursor,verified_records=rows,verified_bytes=size,
                    new_source_bytes_copied=new_bytes,bounded_row_window_exhausted=result.get('bounded_row_window_exhausted',False))
                candidate=dict(manifest,state='STEP_CHECKPOINTED',steps=[*manifest['steps'],step])
                candidate.pop('in_flight')
                published=_publish(tick,candidate,journal_pin)
                manifest=candidate;journal_pin=published
                pin=next_pin;cursor=next_cursor;state=next_state;copied+=new_bytes;unresolved_copy=0
                if result['status']=='RECOVERY_STEP_NOT_RECORDED':status='STEP_REFUSED';break
                if result.get('bounded_row_window_exhausted'):status='ROW_WINDOW_EXHAUSTED';break
            else:status='STEP_BUDGET_REACHED'
            manifest.update(state=status);journal_pin=save()
            return finish(status)
        except Exception as exc:
            # Do not roll back or guess whether an interrupted step published.
            manifest.update(state='RECONCILIATION_REQUIRED',error_type=type(exc).__name__)
            try:journal_pin=save()
            except Exception:pass
            return finish('RECONCILIATION_REQUIRED',type(exc).__name__)
    finally:lock.unlink()

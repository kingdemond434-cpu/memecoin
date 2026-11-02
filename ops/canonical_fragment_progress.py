"""One freshly verified fragment batch per pinned durable history update."""
import hashlib,json,math,os,re,shutil,stat,time,uuid
from pathlib import Path
from ops.canonical_fragmented_row import verify_fragmented_row
from ops.canonical_recovery_plan import _cursor
from ops.incremental_backup import RESERVE
from ops.worker_resource_guard import verify

MAX_METADATA=65536


def _hash(value):
    if not isinstance(value,str) or re.fullmatch('[0-9a-f]{64}',value) is None:
        raise ValueError('Exact SHA256 required')
    return value


def _pairs(values):
    result={}
    for key,value in values:
        if key in result:raise ValueError('Duplicate history key')
        result[key]=value
    return result


def _read(path):
    info=path.lstat()
    def signature(i):return (i.st_dev,i.st_ino,i.st_size,i.st_mtime_ns)
    if (not stat.S_ISREG(info.st_mode) or path.is_symlink()
            or getattr(info,'st_file_attributes',0)&0x400 or info.st_size>MAX_METADATA):
        raise ValueError('Regular bounded progress required')
    with path.open('rb') as stream:
        if signature(os.fstat(stream.fileno()))!=signature(info):raise ValueError('Progress identity changed')
        raw=stream.read(MAX_METADATA+1)
        if signature(os.fstat(stream.fileno()))!=signature(info):raise ValueError('Progress changed')
    if len(raw)>MAX_METADATA or signature(path.stat())!=signature(info):raise ValueError('Progress changed')
    return raw


def _inputs(items,root,cursor,extent,stored=False):
    if type(items) is not list or not 1<=len(items)<=2:raise ValueError('One or two raw receipts required')
    projected=[];names=set();expected=None
    for item in items:
        if not isinstance(item,dict):raise ValueError('Raw receipt required')
        if stored:
            if set(item)!={'name','offset','end_offset','bytes','captured_extent','sha256'}:
                raise ValueError('Exact historical input required')
            name=item['name']
        else:
            if item.get('status')!='SINGLE_RAW_CANONICAL_RANGE_READBACK_VERIFIED':raise ValueError('Raw stage receipt required')
            value=item.get('pending_path')
            if not isinstance(value,str) or len(value)>1024:raise ValueError('Private raw path required')
            path=Path(value).absolute();name=path.name
            if path.resolve().parent!=root.resolve():raise ValueError('Raw fragments must share private progress directory')
        if (not isinstance(name,str) or not 1<=len(name)<=160 or name!=Path(name).name
                or any(c in name for c in '/\\:\x00') or not name.endswith('.raw.pending')
                or name.casefold() in names):raise ValueError('Unique plain raw filename required')
        names.add(name.casefold());_hash(item.get('sha256'))
        if (any(type(item.get(k)) is not int for k in ('offset','end_offset','bytes','captured_extent'))
                or item['offset']<0 or not 1<=item['bytes']<=1048576
                or item['end_offset']!=item['offset']+item['bytes']
                or item['captured_extent']!=extent or item['end_offset']>extent):
            raise ValueError('Bounded captured range required')
        if expected is None:
            if not item['offset']<=cursor['offset']<item['end_offset']:raise ValueError('Cursor outside first input')
        elif item['offset']!=expected:raise ValueError('Noncontiguous raw ranges')
        expected=item['end_offset']
        projected.append(dict(name=name,**{k:item[k] for k in ('offset','end_offset','bytes','captured_extent','sha256')}))
    return projected


def _state(value,anchor,evidence,extent,root):
    if (not isinstance(value,dict) or set(value)!={'version','anchor','anchor_evidence_sha256','target_extent','steps'}
            or type(value['version']) is not int or value['version']!=1 or value['anchor']!=anchor
            or value['anchor_evidence_sha256']!=evidence or type(value['target_extent']) is not int
            or value['target_extent']!=extent or type(value['steps']) is not list or len(value['steps'])>32):
        raise ValueError('History binding mismatch')
    _cursor(value['anchor']);cursor=anchor
    for step in value['steps']:
        if not isinstance(step,dict) or set(step)!={'start_cursor','next_cursor','verified_records','verified_bytes','verified_sha256','inputs'}:
            raise ValueError('Exact history step required')
        first=_cursor(step['start_cursor']);end=_cursor(step['next_cursor']);_hash(step['verified_sha256'])
        inputs=_inputs(step['inputs'],root,first,extent,stored=True)
        if (first!=cursor or type(step['verified_records']) is not int or not 1<=step['verified_records']<=256
                or type(step['verified_bytes']) is not int or not 1<=step['verified_bytes']<=2097152
                or end['sequence']-first['sequence']!=step['verified_records']
                or end['offset']-first['offset']!=step['verified_bytes']
                or end['offset']>inputs[-1]['end_offset']):
            raise ValueError('Noncontiguous history step')
        cursor=end
    return cursor


def record_fragment_step(progress_path,*,anchor,anchor_evidence_sha256,target_extent,receipts,
                         expected_progress_sha256,continue_check,max_records=64,
                         resource_check=verify,free_bytes=None,deadline_seconds=60,
                         clock=time.monotonic,sleep=time.sleep):
    """Trusted private ancestors and caller-pinned provenance required.

    Historical artifact bytes are not revalidated. Exact step intervals exclude
    overread and prior prefixes; whole raw files must not be blindly appended.
    No source copying, automatic loop, baseline/coherence/recovery authority.
    File fsync/atomic replacement is not a cross-store recovery certificate.
    """
    anchor=_cursor(anchor);_hash(anchor_evidence_sha256)
    if expected_progress_sha256 is not None:_hash(expected_progress_sha256)
    if (type(target_extent) is not int or target_extent<=anchor['offset']
            or type(max_records) is not int or not 1<=max_records<=256
            or type(deadline_seconds) not in (int,float) or not math.isfinite(deadline_seconds)
            or not 0<deadline_seconds<=60):raise ValueError('Bounded history policy required')
    cpu=resource_check()
    if cpu.get('status')!='VERIFIED' or cpu.get('cpu_rate')!=200:raise RuntimeError('Development CPU proof required')
    path=Path(progress_path).absolute();root=path.parent
    if (not path.name.endswith('.fragment-progress.json') or not root.is_dir() or root.is_symlink()
            or getattr(root.lstat(),'st_file_attributes',0)&0x400):raise ValueError('Private history directory required')
    if free_bytes is None:free_bytes=lambda:shutil.disk_usage(root).free
    started=clock();temporary=None
    def guard():
        if (clock()-started>=deadline_seconds or continue_check() is not True
                or free_bytes()-MAX_METADATA<RESERVE):raise RuntimeError('History resource gate stopped')
    guard();lock=root/'canonical-segment.lock'
    with lock.open('x',encoding='utf-8') as stream:stream.write('exclusive fragment history writer\n')
    try:
        guard()
        if expected_progress_sha256 is None:
            if os.path.lexists(path):raise FileExistsError('Unpinned existing history')
            state=dict(version=1,anchor=anchor,anchor_evidence_sha256=anchor_evidence_sha256,target_extent=target_extent,steps=[])
        else:
            original=_read(path)
            if hashlib.sha256(original).hexdigest()!=expected_progress_sha256:raise ValueError('History SHA256 mismatch')
            state=json.loads(original,object_pairs_hook=_pairs)
        cursor=_state(state,anchor,anchor_evidence_sha256,target_extent,root)
        if len(state['steps'])>=32:raise ValueError('History step budget exhausted; no eviction')
        inputs=_inputs(receipts,root,cursor,target_extent)
        live=[dict(status='SINGLE_RAW_CANONICAL_RANGE_READBACK_VERIFIED',pending_path=str(root/i['name']),
                   **{k:v for k,v in i.items() if k!='name'}) for i in inputs]
        guard();remaining=deadline_seconds-(clock()-started)
        result=verify_fragmented_row(live,cursor=cursor,continue_check=continue_check,max_records=max_records,
            resource_check=resource_check,deadline_seconds=remaining,clock=clock,sleep=sleep)
        common=dict(verification=result,previous_results_revalidated=False,baseline_authenticated=False,
            full_prefix_chain_verified=False,cross_store_checkpoint_verified=False,recovery_verified=False,
            deployment_backup_accepted=False,source_write_requested=False,physical_io_cap_claimed=False)
        if result['status'] not in ('SINGLE_FRAGMENTED_CHAIN_ROW_VERIFIED','BOUNDED_FRAGMENTED_CHAIN_ROWS_VERIFIED'):
            return dict(status='FRAGMENT_STEP_NOT_RECORDED',progress_changed=False,
                        progress_sha256=expected_progress_sha256,next_cursor=cursor,**common)
        step=dict(start_cursor=cursor,next_cursor=result['next_cursor'],verified_records=result['verified_records'],
                  verified_bytes=result['verified_bytes'],verified_sha256=result['verified_sha256'],inputs=inputs)
        state['steps'].append(step);next_cursor=_state(state,anchor,anchor_evidence_sha256,target_extent,root)
        payload=json.dumps(state,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        if len(payload)>MAX_METADATA:raise ValueError('History metadata budget exceeded')
        guard();temporary=root/('fragment-progress-'+uuid.uuid4().hex+'.partial')
        with temporary.open('xb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
        guard()
        if expected_progress_sha256 is None:
            if os.path.lexists(path):raise FileExistsError('Concurrent history publication')
            os.link(temporary,path)
        else:
            if hashlib.sha256(_read(path)).hexdigest()!=expected_progress_sha256:raise ValueError('Concurrent history change')
            os.replace(temporary,path)
        return dict(status='FRAGMENT_STEP_RECORDED',progress_changed=True,progress_sha256=hashlib.sha256(payload).hexdigest(),
                    prior_progress_sha256=expected_progress_sha256,progress_path=str(path),steps=len(state['steps']),
                    next_cursor=next_cursor,elapsed_seconds=clock()-started,**common)
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()
        lock.unlink()

"""Explicit bounded parent-to-child fragment history linkage; no recovery certificate."""
import hashlib,json,math,os,shutil,time,uuid
from pathlib import Path
import ops.canonical_fragment_progress as history
from ops.incremental_backup import RESERVE
from ops.worker_resource_guard import verify


def _encode(value):
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if len(raw)>history.MAX_METADATA:raise ValueError('Rollover metadata budget exceeded')
    return raw


def _name(value):
    if (not isinstance(value,str) or not 1<=len(value)<=160 or value!=Path(value).name
            or any(c in value for c in '/\\:\x00') or not value.endswith('.fragment-progress.json')):
        raise ValueError('Plain fragment history name required')
    return value


def _suffix(state,cursor):
    for i,item in enumerate(state['steps'][-1]['inputs']):
        if item['offset']<=cursor['offset']<item['end_offset']:
            return [dict(v) for v in state['steps'][-1]['inputs'][i:]]
    return []


def load_rollover_inputs(child_path,*,anchor,anchor_evidence_sha256,target_extent,guard):
    """Caller pins link bytes; freshly validate one parent, not recursive ancestry/raw bytes."""
    child=Path(child_path).absolute();root=child.parent;_name(child.name)
    guard();raw=history._read(child.with_name(child.name+'.rollover.json'))
    if hashlib.sha256(raw).hexdigest()!=history._hash(anchor_evidence_sha256):
        raise ValueError('Rollover link pin mismatch')
    link=json.loads(raw,object_pairs_hook=history._pairs)
    if (not isinstance(link,dict) or set(link)!={'version','parent_name','parent_sha256','parent_anchor',
            'parent_anchor_evidence_sha256','target_extent','child_name','child_anchor','retained_inputs'}
            or type(link['version']) is not int or link['version']!=1
            or link['child_name']!=child.name or link['child_anchor']!=anchor
            or type(link['target_extent']) is not int or link['target_extent']!=target_extent):
        raise ValueError('Rollover binding mismatch')
    parent=root/_name(link['parent_name'])
    if parent.name.casefold()==child.name.casefold():raise ValueError('Distinct parent required')
    history._hash(link['parent_sha256']);history._hash(link['parent_anchor_evidence_sha256'])
    guard();prior=history._read(parent)
    if hashlib.sha256(prior).hexdigest()!=link['parent_sha256']:raise ValueError('Pinned parent changed')
    state=json.loads(prior,object_pairs_hook=history._pairs)
    cursor=history._state(state,history._cursor(link['parent_anchor']),link['parent_anchor_evidence_sha256'],target_extent,root)
    if len(state['steps'])!=32 or cursor!=history._cursor(anchor) or cursor['offset']>=target_extent:
        raise ValueError('Full parent ending at child anchor required')
    seeds=_suffix(state,cursor)
    if link['retained_inputs']!=seeds:raise ValueError('Retained parent inputs mismatch')
    guard();return seeds


def _publish_exact(path,payload,guard):
    if os.path.lexists(path):
        if history._read(path)!=payload:raise FileExistsError('Existing rollover artifact preserved')
        return
    temporary=path.parent/('rollover-'+uuid.uuid4().hex+'.partial')
    try:
        with temporary.open('xb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
        guard();os.link(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()


def rollover_fragment_history(parent_path,child_path,*,parent_anchor,parent_anchor_evidence_sha256,
                              target_extent,expected_parent_sha256,continue_check,
                              resource_check=verify,free_bytes=None,deadline_seconds=30,clock=time.monotonic):
    """Full32-step parent only. Exclusive child/link publication, exact orphan retry.

    Retains old history/raw artifacts. No raw reads/copy, loop, baseline authentication,
    recursive ancestry verification, cross-store snapshot, RAM/IO cap or deployment.
    Trusted private ancestors/cooperative writers; deadlines do not preempt syscalls.
    """
    parent_anchor=history._cursor(parent_anchor);history._hash(parent_anchor_evidence_sha256);history._hash(expected_parent_sha256)
    if (type(target_extent) is not int or target_extent<=parent_anchor['offset']
            or type(deadline_seconds) not in (int,float) or not math.isfinite(deadline_seconds)
            or not 0<deadline_seconds<=30):raise ValueError('Bounded rollover policy required')
    cpu=resource_check()
    if cpu.get('status')!='VERIFIED' or cpu.get('cpu_rate')!=200:raise RuntimeError('Development CPU proof required')
    parent=Path(parent_path).absolute();child=Path(child_path).absolute();root=parent.parent
    _name(parent.name);_name(child.name)
    if (parent.name.casefold()==child.name.casefold() or child.parent.resolve()!=root.resolve()
            or not root.is_dir() or root.is_symlink() or getattr(root.lstat(),'st_file_attributes',0)&0x400):
        raise ValueError('Distinct private sibling histories required')
    if free_bytes is None:free_bytes=lambda:shutil.disk_usage(root).free
    started=clock();held=[]
    def guard():
        if (clock()-started>=deadline_seconds or continue_check() is not True
                or free_bytes()-3*history.MAX_METADATA<RESERVE):raise RuntimeError('Rollover resource gate stopped')
    guard()
    try:
        for name in ('canonical-step.lock','canonical-segment.lock'):
            lock=root/name
            with lock.open('x',encoding='utf-8') as stream:stream.write('exclusive bounded rollover writer\n')
            held.append(lock)
        guard();raw=history._read(parent)
        if hashlib.sha256(raw).hexdigest()!=expected_parent_sha256:raise ValueError('Parent pin mismatch')
        state=json.loads(raw,object_pairs_hook=history._pairs)
        cursor=history._state(state,parent_anchor,parent_anchor_evidence_sha256,target_extent,root)
        if len(state['steps'])!=32 or cursor['offset']>=target_extent:
            raise ValueError('Full nonterminal32-step parent required')
        link=dict(version=1,parent_name=parent.name,parent_sha256=expected_parent_sha256,parent_anchor=parent_anchor,
            parent_anchor_evidence_sha256=parent_anchor_evidence_sha256,target_extent=target_extent,
            child_name=child.name,child_anchor=cursor,retained_inputs=_suffix(state,cursor))
        link_bytes=_encode(link);link_pin=hashlib.sha256(link_bytes).hexdigest()
        next_state=dict(version=1,anchor=cursor,anchor_evidence_sha256=link_pin,target_extent=target_extent,steps=[])
        payload=_encode(next_state);history._state(next_state,cursor,link_pin,target_extent,root)
        # Reject an already advanced/different child before leaving any new link.
        if os.path.lexists(child) and history._read(child)!=payload:
            raise FileExistsError('Existing child history preserved')
        def publish_guard():
            guard()
            if hashlib.sha256(history._read(parent)).hexdigest()!=expected_parent_sha256:
                raise ValueError('Parent changed before publication')
        link_path=child.with_name(child.name+'.rollover.json')
        publish_guard();_publish_exact(link_path,link_bytes,publish_guard)
        publish_guard();_publish_exact(child,payload,publish_guard)
        publish_guard()
        return dict(status='FRAGMENT_HISTORY_ROLLOVER_LINKED',parent_path=str(parent),parent_sha256=expected_parent_sha256,
            child_path=str(child),progress_sha256=hashlib.sha256(payload).hexdigest(),anchor=cursor,
            anchor_evidence_sha256=link_pin,rollover_link=str(link_path),target_extent=target_extent,
            retained_inputs=len(link['retained_inputs']),parent_steps=32,child_steps=0,new_verified_records=0,
            historical_raw_bytes_revalidated=False,recursive_ancestry_verified=False,baseline_authenticated=False,
            full_prefix_chain_verified=False,cross_store_checkpoint_verified=False,recovery_verified=False,
            deployment_backup_accepted=False,source_write_requested=False,physical_io_cap_claimed=False)
    finally:
        for lock in reversed(held):lock.unlink()

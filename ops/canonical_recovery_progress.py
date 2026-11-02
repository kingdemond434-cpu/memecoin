"""One verified segment per durable progress update; never deployment authority."""
import hashlib,json,os,uuid
from pathlib import Path
from ops.bounded_manifest import _read
from ops.canonical_recovery_plan import compose_recovery_plan,_cursor
from ops.canonical_segment_verify import verify_staged_segment
from ops.worker_resource_guard import verify


def verify_next_recovery_segment(anchor,receipts,progress_path,*,continue_check,
                                 resource_check=verify,verifier=verify_staged_segment):
    """Trusted private storage/ancestors and receipt provenance required.

    Existing results are historical, not revalidated. One bounded segment read;
    no baseline scan, copying, source writes or automatic continuation loop.
    """
    cpu=resource_check()
    if cpu.get('status')!='VERIFIED' or cpu.get('cpu_rate')!=200 or continue_check() is not True:
        raise RuntimeError('Development CPU and host admission required')
    plan=compose_recovery_plan(anchor,receipts)
    binding=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    path=Path(progress_path).absolute();root=path.parent
    if (not path.name.endswith('.recovery-progress.json') or not root.is_dir()
            or root.is_symlink() or getattr(root.lstat(),'st_file_attributes',0)&0x400):
        raise ValueError('Private progress directory and exact suffix required')
    if any(Path(item['path']).resolve()==path.resolve() for item in plan['segments']):
        raise ValueError('Progress aliases staged segment')
    def pairs(values):
        result={}
        for key,value in values:
            if key in result:raise ValueError('Duplicate progress key')
            result[key]=value
        return result
    def matches(result,item):
        if isinstance(result,dict):
            _cursor(result.get('start_cursor'));_cursor(result.get('next_cursor'))
        return (isinstance(result,dict) and result.get('status')=='SINGLE_STAGED_SEGMENT_CHAIN_VERIFIED'
            and type(result.get('bytes_read')) is int and result['bytes_read']==item['bytes']
            and result.get('sha256')==item['sha256']
            and result.get('start_cursor')==item['start_cursor']
            and result.get('next_cursor')==item['next_cursor']
            and type(result.get('verified_records')) is int
            and result['verified_records']==item['next_cursor']['sequence']-item['start_cursor']['sequence']
            and all(result.get(key) is False for key in ('baseline_authenticated',
                'full_prefix_chain_verified','recovery_verified','cross_store_checkpoint_verified',
                'deployment_backup_accepted','source_write_requested','physical_io_cap_claimed')))
    lock=root/'canonical-segment.lock';temporary=None
    with lock.open('x',encoding='utf-8') as stream:stream.write('exclusive recovery progress writer\n')
    try:
        state=json.loads(_read(path),object_pairs_hook=pairs) if os.path.lexists(path) else dict(version=1,plan_sha256=binding,checked=[])
        if (not isinstance(state,dict) or set(state)!={'version','plan_sha256','checked'}
                or type(state['version']) is not int or state['version']!=1
                or state['plan_sha256']!=binding or type(state['checked']) is not list
                or len(state['checked'])>len(receipts)):
            raise ValueError('Recovery progress binding mismatch')
        checked=state['checked']
        for index,result in enumerate(checked):
            if not matches(result,plan['segments'][index]):raise ValueError('Historical segment receipt mismatch')
        if len(checked)<len(receipts):
            index=len(checked)
            result=verifier(receipts[index],continue_check=continue_check,resource_check=resource_check)
            if not matches(result,plan['segments'][index]):raise ValueError('Verified segment receipt required')
            checked.append(result);payload=json.dumps(state,sort_keys=True).encode()
            if len(payload)>65536:raise ValueError('Progress metadata budget exceeded')
            if continue_check() is not True:raise RuntimeError('Publication admission stopped')
            temporary=root/('recovery-progress-'+uuid.uuid4().hex+'.tmp')
            with temporary.open('xb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,path)
        return dict(status='CANONICAL_SEGMENT_VERIFICATION_HISTORY_ONLY',checked=len(checked),
            total=len(receipts),all_segment_receipts_recorded=len(checked)==len(receipts),
            plan_sha256=binding,next_cursor=checked[-1]['next_cursor'] if checked else plan['anchor'],
            previous_results_revalidated=False,baseline_authenticated=False,
            full_prefix_chain_verified=False,cross_store_checkpoint_verified=False,
            recovery_verified=False,deployment_backup_accepted=False,source_write_requested=False)
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()
        lock.unlink()


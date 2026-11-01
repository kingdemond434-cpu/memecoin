"""One bounded staged-copy quick-check per call; no deployment certificate."""
import hashlib,json,os,re,uuid
from pathlib import Path
from ops.bounded_manifest import _read
from ops.incremental_backup import DATABASES
from ops.staged_integrity import verify_staged_sqlite,MAX_BYTES
from ops.worker_resource_guard import verify


def verify_next_staged_store(*,staging_root,continue_check,resource_check=verify,
                             verifier=verify_staged_sqlite):
    """Caller supplies trusted private serialized staging and host admission.

    Eight-MiB per-copy limit remains enforced by the existing verifier. Previous
    results are historical receipts, not current all-store revalidation.
    """
    cpu=resource_check()
    if cpu.get('status')!='VERIFIED' or cpu.get('cpu_rate')!=200 or continue_check() is not True:
        raise RuntimeError('Development CPU and host admission required')
    requested=Path(staging_root).absolute()
    if requested.is_symlink() or not requested.is_dir():
        raise ValueError('Private staging directory required')
    root=requested.resolve();lock=root/'backup-set.lock';temporary=None
    with lock.open('x',encoding='utf-8') as stream:stream.write('exclusive verification writer\n')
    try:
        raw=_read(root/'backup-set.json');staging=json.loads(raw);rows=staging.get('stores')
        if (type(staging.get('version')) is not int or staging['version']!=1
                or not isinstance(rows,dict) or set(rows)!=DATABASES
                or re.fullmatch('[0-9a-f]{64}',str(staging.get('sources_sha256',''))) is None):
            raise ValueError('Complete ten-store staging checkpoint required')
        for name,receipt in rows.items():
            if (not isinstance(receipt,dict)
                    or receipt.get('status')!='SINGLE_SQLITE_COPY_READBACK_VERIFIED_INTEGRITY_PENDING'
                    or receipt.get('pending_path')!=str(root/(name+'.pending'))
                    or receipt.get('deployment_backup_accepted') is not False
                    or type(receipt.get('bytes')) is not int or receipt['bytes']<=0
                    or re.fullmatch('[0-9a-f]{64}',str(receipt.get('sha256',''))) is None):
                raise ValueError('Exact unaccepted staged-copy receipts required')
        binding=hashlib.sha256(raw).hexdigest();progress=root/'backup-integrity.json'
        state=json.loads(_read(progress)) if os.path.lexists(progress) else {
            'version':1,'staging_checkpoint_sha256':binding,'checked':{}}
        checked=state.get('checked')
        if (type(state.get('version')) is not int or state['version']!=1
                or state.get('staging_checkpoint_sha256')!=binding
                or not isinstance(checked,dict) or not set(checked)<=DATABASES):
            raise ValueError('Verification checkpoint binding mismatch')
        for name,result in checked.items():
            if (not isinstance(result,dict)
                    or result.get('status')!='SINGLE_STAGED_SQLITE_QUICK_CHECK_VERIFIED'
                    or result.get('bytes')!=rows[name]['bytes'] or result.get('sha256')!=rows[name]['sha256']
                    or result.get('deployment_backup_accepted') is not False):
                raise ValueError('Historical verification receipt mismatch')
        remaining=sorted(DATABASES-set(checked))
        blocked=[name for name in remaining if rows[name]['bytes']>MAX_BYTES]
        eligible=[name for name in remaining if name not in blocked]
        if eligible:
            name=eligible[0]
            result=verifier(root/(name+'.pending'),rows[name],continue_check=continue_check)
            if (result.get('status')!='SINGLE_STAGED_SQLITE_QUICK_CHECK_VERIFIED'
                    or result.get('bytes')!=rows[name]['bytes'] or result.get('sha256')!=rows[name]['sha256']
                    or result.get('deployment_backup_accepted') is not False):
                raise ValueError('Bounded staged quick-check receipt required')
            checked[name]=result
            payload=json.dumps(state,sort_keys=True).encode()
            if len(payload)>65536:raise ValueError('Verification checkpoint budget')
            temporary=root/('backup-integrity-'+uuid.uuid4().hex+'.tmp')
            with temporary.open('xb') as stream:
                stream.write(payload);stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,progress)
        return dict(status='BACKUP_SET_QUICK_CHECK_RECEIPTS_ONLY',checked=len(checked),total=10,
                    all_quick_check_receipts_recorded=len(checked)==10,
                    blocked_by_copy_budget=blocked,
                    previous_results_revalidated=False,sqlite_full_integrity_verified=False,
                    canonical_prefix_verified=False,cross_store_checkpoint_verified=False,
                    recovery_verified=False,deployment_backup_accepted=False,services_changed=False)
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()
        lock.unlink()

"""One-store-at-a-time staging checkpoints; never a recovery certificate."""
import hashlib,json,os,re,stat,uuid
from pathlib import Path
from ops.incremental_backup import DATABASES,stage_sqlite_backup
from ops.worker_resource_guard import verify


def stage_next_store(*,sources,staging_root,max_store_bytes,continue_check,
                     resource_check=verify,stage=stage_sqlite_backup):
    """Caller owns a trusted private staging parent and host/IO admission.

    Cooperative exclusive lock prevents overlapping set writers. Prior receipts
    are retained without bulk revalidation; they are not current integrity proof.
    A copy left after checkpoint failure is preserved for manual reconciliation.
    """
    cpu=resource_check()
    if cpu.get('status')!='VERIFIED' or cpu.get('cpu_rate')!=200:
        raise RuntimeError('Development CPU proof required')
    if (set(sources)!=DATABASES or type(max_store_bytes) is not int
            or not 0<max_store_bytes<=1024**4 or continue_check() is not True):
        raise ValueError('Exact ten-store inventory and explicit budget/admission required')
    paths={name:Path(path).absolute() for name,path in sources.items()}
    requested=Path(staging_root).absolute()
    if requested.exists() and (requested.is_symlink() or getattr(requested.lstat(),'st_file_attributes',0)&0x400):
        raise ValueError('Reparse staging directory rejected')
    root=requested.resolve()
    if (any(p.name!=name for name,p in paths.items())
            or any(root==p.resolve().parent or p.resolve().parent in root.parents for p in paths.values())):
        raise ValueError('Separate trusted staging directory required')
    root.mkdir(exist_ok=True)
    if root.is_symlink() or getattr(root.lstat(),'st_file_attributes',0)&0x400:
        raise ValueError('Reparse staging directory rejected')
    binding=hashlib.sha256(json.dumps({k:str(v) for k,v in paths.items()},sort_keys=True).encode()).hexdigest()
    checkpoint=root/'backup-set.json';lock=root/'backup-set.lock'
    with lock.open('x',encoding='utf-8') as stream:stream.write('exclusive set writer\n')
    temporary=None
    try:
        if checkpoint.exists():
            info=checkpoint.lstat()
            if (not stat.S_ISREG(info.st_mode) or checkpoint.is_symlink()
                    or getattr(info,'st_file_attributes',0)&0x400 or info.st_size>65536):
                raise ValueError('Regular bounded backup checkpoint required')
            with checkpoint.open('rb') as stream:raw=stream.read(65537)
            if len(raw)>65536:raise ValueError('Backup checkpoint budget')
            state=json.loads(raw)
        else:state={'version':1,'sources_sha256':binding,'stores':{}}
        entries=state.get('stores')
        if (type(state.get('version')) is not int or state['version']!=1
                or state.get('sources_sha256')!=binding or not isinstance(entries,dict)
                or not set(entries)<=DATABASES):
            raise ValueError('Backup checkpoint inventory mismatch')
        for name,receipt in entries.items():
            if (not isinstance(receipt,dict)
                    or receipt.get('status')!='SINGLE_SQLITE_COPY_READBACK_VERIFIED_INTEGRITY_PENDING'
                    or receipt.get('pending_path')!=str(root/(name+'.pending'))):
                raise ValueError('Backup checkpoint receipt mismatch')
            pending=root/(name+'.pending');info=pending.lstat()
            if (pending.is_symlink() or getattr(info,'st_file_attributes',0)&0x400
                    or not stat.S_ISREG(info.st_mode)
                    or type(receipt.get('bytes')) is not int or info.st_size!=receipt['bytes']
                    or receipt.get('deployment_backup_accepted') is not False
                    or not isinstance(receipt.get('sha256'),str)
                    or re.fullmatch('[0-9a-f]{64}',receipt['sha256']) is None):
                raise ValueError('Prior staged copy missing, changed or unaccepted scope invalid')
        remaining=sorted(DATABASES-set(entries))
        if remaining:
            name=remaining[0]
            receipt=stage(paths[name],root/(name+'.pending'),max_bytes=max_store_bytes,
                          continue_check=continue_check)
            if (receipt.get('status')!='SINGLE_SQLITE_COPY_READBACK_VERIFIED_INTEGRITY_PENDING'
                    or receipt.get('deployment_backup_accepted') is not False
                    or receipt.get('pending_path')!=str(root/(name+'.pending'))):
                raise ValueError('Staging receipt is not an unaccepted copy')
            entries[name]=receipt
            temporary=root/('backup-set-'+uuid.uuid4().hex+'.tmp')
            payload=json.dumps(state,sort_keys=True).encode()
            if len(payload)>65536:raise ValueError('Backup checkpoint budget')
            with temporary.open('xb') as stream:
                stream.write(payload);stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,checkpoint)
        return dict(status='BACKUP_SET_STAGING_CHECKPOINT_ONLY',staged=len(entries),total=10,
                    all_stores_staged=len(entries)==10,previous_receipts_revalidated=False,
                    sqlite_integrity_verified=False,canonical_prefix_verified=False,
                    cross_store_checkpoint_verified=False,recovery_verified=False,
                    deployment_backup_accepted=False,services_changed=False)
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()
        lock.unlink()

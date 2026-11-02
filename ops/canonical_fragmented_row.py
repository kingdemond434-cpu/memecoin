"""Verify one canonical row across at most two preserved raw fragments."""
import hashlib,json,math,os,re,stat,time
from pathlib import Path
from ops.worker_resource_guard import verify
from src.research.trade_evidence import GENESIS,_canonical


def verify_fragmented_row(receipts,*,cursor,continue_check,resource_check=verify,
                          rate_bytes_per_second=256*1024,deadline_seconds=60,
                          clock=time.monotonic,sleep=time.sleep):
    """Read-only, <=2MiB logical input. Caller authenticates private receipts.

    No full-prefix trust, source-to-fragment attestation, copying or automatic
    continuation. An incomplete row never advances the verified chain cursor.
    JSON materialization is not a RAM cap; deadlines do not preempt syscalls.
    """
    if (not isinstance(cursor,dict) or set(cursor)!={'offset','sequence','last_hash'}
            or any(type(cursor.get(k)) is not int or cursor[k]<0 for k in ('offset','sequence'))
            or not isinstance(cursor.get('last_hash'),str)
            or re.fullmatch('[0-9a-f]{64}',cursor['last_hash']) is None
            or (cursor['offset']==0 and (cursor['sequence']!=0 or cursor['last_hash']!=GENESIS))
            or not isinstance(receipts,list) or not 1<=len(receipts)<=2
            or type(rate_bytes_per_second) is not int or not 1<=rate_bytes_per_second<=1048576
            or type(deadline_seconds) not in (int,float) or not math.isfinite(deadline_seconds)
            or not 0<deadline_seconds<=60):
        raise ValueError('Exact cursor and bounded fragment policy required')
    cpu=resource_check()
    if cpu.get('status')!='VERIFIED' or cpu.get('cpu_rate')!=200:
        raise RuntimeError('Development CPU proof required')
    result=dict(status='FRAGMENTED_ROW_REJECTED',next_cursor=dict(cursor),verified_records=0,
        bytes_read=0,prefix_anchor_authenticated=False,full_prefix_chain_verified=False,
        recovery_verified=False,deployment_backup_accepted=False,source_write_requested=False,
        physical_io_cap_claimed=False)
    started=clock();read=0;identities=[];paths=set();blocks=[];expected=cursor['offset'];extent=None
    def guard():
        if clock()-started>=deadline_seconds or continue_check() is not True:
            raise TimeoutError('fragment_resource_gate')
    def regular(p):
        info=p.lstat()
        if not stat.S_ISREG(info.st_mode) or p.is_symlink() or getattr(info,'st_file_attributes',0)&0x400:
            raise ValueError('regular private fragment required')
        return info
    def signature(info):return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns)
    def pairs(values):
        obj={}
        for k,v in values:
            if k in obj:raise ValueError('duplicate JSON key')
            obj[k]=v
        return obj
    try:
        # Validate all metadata before opening any fragment.
        for r in receipts:
            if (not isinstance(r,dict) or r.get('status')!='SINGLE_RAW_CANONICAL_RANGE_READBACK_VERIFIED'
                    or any(type(r.get(k)) is not int for k in ('offset','end_offset','bytes','captured_extent'))
                    or r['offset']!=expected or not 1<=r['bytes']<=1048576
                    or r['end_offset']!=r['offset']+r['bytes']
                    or r['captured_extent']<r['end_offset']
                    or not isinstance(r.get('sha256'),str) or re.fullmatch('[0-9a-f]{64}',r['sha256']) is None
                    or not isinstance(r.get('pending_path'),str) or not r['pending_path'].endswith('.raw.pending')):
                raise ValueError('contiguous raw receipt required')
            if extent is not None and extent!=r['captured_extent']:raise ValueError('captured extent mismatch')
            extent=r['captured_extent'];expected=r['end_offset']
            p=Path(r['pending_path']);key=str(p.absolute()).casefold()
            if key in paths:raise ValueError('duplicate fragment path')
            paths.add(key)
        for r in receipts:
            guard();p=Path(r['pending_path']);before=regular(p)
            if before.st_size!=r['bytes']:raise ValueError('fragment size mismatch')
            if any((before.st_dev,before.st_ino)==(i.st_dev,i.st_ino) for _,i in identities):
                raise ValueError('aliased fragment identity')
            digest=hashlib.sha256();remaining=r['bytes']
            with p.open('rb') as stream:
                if signature(os.fstat(stream.fileno()))!=signature(before):raise ValueError('fragment changed')
                while remaining:
                    guard();block=stream.read(min(65536,remaining))
                    if not block:raise ValueError('fragment truncated')
                    blocks.append(block);digest.update(block);remaining-=len(block);read+=len(block)
                    delay=read/rate_bytes_per_second-(clock()-started)
                    while delay>1e-9:
                        guard();sleep(min(.1,delay));delay=read/rate_bytes_per_second-(clock()-started)
                if stream.read(1) or signature(os.fstat(stream.fileno()))!=signature(before):
                    raise ValueError('fragment changed')
            if digest.hexdigest()!=r['sha256']:raise ValueError('fragment digest mismatch')
            identities.append((p,before))
        guard()
        for p,info in identities:
            if signature(regular(p))!=signature(info):raise ValueError('fragment changed')
        data=b''.join(blocks);end=data.find(b'\n')
        if end<0:
            result.update(status='NO_COMPLETE_FRAGMENTED_ROW',raw_end_offset=expected,
                          captured_extent=extent,partial_bytes=len(data))
        else:
            raw=data[:end+1];row=json.loads(raw,object_pairs_hook=pairs)
            if not isinstance(row,dict):raise ValueError('record object required')
            digest=row.pop('hash')
            if (type(row.get('sequence')) is not int or row['sequence']!=cursor['sequence']+1
                    or row.get('previous_hash')!=cursor['last_hash'] or not isinstance(digest,str)
                    or re.fullmatch('[0-9a-f]{64}',digest) is None
                    or hashlib.sha256(_canonical(row)).hexdigest()!=digest):
                raise ValueError('canonical chain mismatch')
            guard()
            for p,info in identities:
                if signature(regular(p))!=signature(info):raise ValueError('fragment changed')
            result.update(status='SINGLE_FRAGMENTED_CHAIN_ROW_VERIFIED',verified_records=1,
                next_cursor=dict(offset=cursor['offset']+len(raw),sequence=row['sequence'],last_hash=digest),
                verified_row_bytes=len(raw),verified_row_sha256=hashlib.sha256(raw).hexdigest(),
                unconsumed_bytes=len(data)-len(raw),raw_end_offset=expected,captured_extent=extent)
    except TimeoutError:result['status']='RESOURCE_BLOCKED'
    except (OSError,ValueError,TypeError,KeyError,UnicodeError,RecursionError) as e:
        result['error_type']=type(e).__name__
    result['bytes_read']=read
    return result

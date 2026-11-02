"""Verify a bounded canonical row batch across preserved raw fragments."""
import hashlib,json,math,os,re,stat,time
from pathlib import Path
from ops.worker_resource_guard import verify
from src.research.trade_evidence import GENESIS,_canonical


def verify_fragmented_row(receipts,*,cursor,continue_check,max_records=1,resource_check=verify,
                          rate_bytes_per_second=256*1024,deadline_seconds=60,
                          clock=time.monotonic,sleep=time.sleep):
    """Read-only, <=2MiB input and <=256 rows. Default preserves one-row API.

    Caller authenticates private receipts and the exact prior chain cursor.
    No full-prefix trust, source-to-fragment attestation, copying or automatic
    continuation. The cursor never advances past an incomplete trailing row;
    any rejection rolls back the whole batch to the supplied cursor.
    JSON materialization is not a RAM cap; deadlines do not preempt syscalls.
    """
    if (not isinstance(cursor,dict) or set(cursor)!={'offset','sequence','last_hash'}
            or any(type(cursor.get(k)) is not int or cursor[k]<0 for k in ('offset','sequence'))
            or not isinstance(cursor.get('last_hash'),str)
            or re.fullmatch('[0-9a-f]{64}',cursor['last_hash']) is None
            or (cursor['offset']==0 and (cursor['sequence']!=0 or cursor['last_hash']!=GENESIS))
            or not isinstance(receipts,list) or not 1<=len(receipts)<=2
            or type(max_records) is not int or not 1<=max_records<=256
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
    started=clock();read=0;checked=0;identities=[];paths=set();blocks=[];expected=None;extent=None
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
                    or r['offset']<0 or not 1<=r['bytes']<=1048576
                    or r['end_offset']!=r['offset']+r['bytes']
                    or r['captured_extent']<r['end_offset']
                    or not isinstance(r.get('sha256'),str) or re.fullmatch('[0-9a-f]{64}',r['sha256']) is None
                    or not isinstance(r.get('pending_path'),str) or not r['pending_path'].endswith('.raw.pending')):
                raise ValueError('contiguous raw receipt required')
            if expected is None:
                if not r['offset']<=cursor['offset']<r['end_offset']:raise ValueError('cursor outside first fragment')
            elif r['offset']!=expected:raise ValueError('contiguous raw receipt required')
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
        data=b''.join(blocks);skip=cursor['offset']-receipts[0]['offset']
        if skip and data[skip-1]!=10:raise ValueError('cursor not at row boundary')
        data=data[skip:];position=0;sequence=cursor['sequence'];previous=cursor['last_hash']
        while checked<max_records:
            guard();end=data.find(b'\n',position)
            if end<0:break
            raw=data[position:end+1];row=json.loads(raw,object_pairs_hook=pairs)
            if not isinstance(row,dict):raise ValueError('record object required')
            digest=row.pop('hash')
            if (type(row.get('sequence')) is not int or row['sequence']!=sequence+1
                    or row.get('previous_hash')!=previous or not isinstance(digest,str)
                    or re.fullmatch('[0-9a-f]{64}',digest) is None
                    or hashlib.sha256(_canonical(row)).hexdigest()!=digest):
                raise ValueError('canonical chain mismatch')
            checked+=1;sequence+=1;previous=digest;position=end+1
        guard()
        for p,info in identities:
            if signature(regular(p))!=signature(info):raise ValueError('fragment changed')
        if not checked:
            result.update(status='NO_COMPLETE_FRAGMENTED_ROW',raw_end_offset=expected,
                          captured_extent=extent,partial_bytes=len(data),bytes_before_cursor=skip)
        else:
            result.update(status=('SINGLE_FRAGMENTED_CHAIN_ROW_VERIFIED' if checked==1 else
                                  'BOUNDED_FRAGMENTED_CHAIN_ROWS_VERIFIED'),verified_records=checked,
                next_cursor=dict(offset=cursor['offset']+position,sequence=sequence,last_hash=previous),
                verified_bytes=position,verified_sha256=hashlib.sha256(data[:position]).hexdigest(),
                unconsumed_bytes=len(data)-position,raw_end_offset=expected,captured_extent=extent,
                bytes_before_cursor=skip,record_limit_reached=checked==max_records and position<len(data),
                partial_trailing_row=bool(data) and not data.endswith(b'\n'))
            if checked==1:
                result.update(verified_row_bytes=position,verified_row_sha256=result['verified_sha256'])
    except TimeoutError:result.update(status='RESOURCE_BLOCKED',rows_checked_before_rejection=checked)
    except (OSError,ValueError,TypeError,KeyError,UnicodeError,RecursionError) as e:
        result.update(error_type=type(e).__name__,rows_checked_before_rejection=checked)
    result['bytes_read']=read
    return result

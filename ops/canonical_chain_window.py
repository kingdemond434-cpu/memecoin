"""Bounded forward verification from a caller-authenticated canonical cursor."""
import hashlib,json,math,os,re,stat,time
from pathlib import Path
from src.research.trade_evidence import _canonical,GENESIS
from ops.worker_resource_guard import verify


class _AdmissionStopped(RuntimeError):pass


def verify_chain_window(path,*,cursor,continue_check,max_bytes=1048576,max_records=256,
                        rate_bytes_per_second=256*1024,deadline_seconds=60,
                        resource_check=verify,clock=time.monotonic,sleep=time.sleep):
    """No prefix scan/copy or automatic sweep. Caller owns prefix trust.

    Requires trusted append-only storage; concurrent malicious rewrite resistance
    and previously certified prefix revalidation are outside this window gate.
    Logical read pacing does not cap physical IO or preempt filesystem calls.
    """
    if (not isinstance(cursor,dict) or set(cursor)!={'offset','sequence','last_hash'}
            or any(type(cursor.get(k)) is not int or cursor[k]<0 for k in ('offset','sequence'))
            or not isinstance(cursor.get('last_hash'),str)
            or re.fullmatch('[0-9a-f]{64}',cursor['last_hash']) is None
            or (cursor['offset']==0 and (cursor['sequence']!=0 or cursor['last_hash']!=GENESIS))
            or type(max_bytes) is not int or not 128<=max_bytes<=1048576
            or type(max_records) is not int or not 1<=max_records<=256
            or type(rate_bytes_per_second) is not int or not 1<=rate_bytes_per_second<=1048576
            or type(deadline_seconds) not in (int,float) or not math.isfinite(deadline_seconds)
            or not 0<deadline_seconds<=60):
        raise ValueError('Bounded chain policy and exact cursor required')
    cpu=resource_check()
    if cpu.get('status')!='VERIFIED' or cpu.get('cpu_rate')!=200:
        raise RuntimeError('Development CPU proof required')
    started=clock();read=0;checked=0
    result=dict(status='NO_COMPLETE_ROWS',next_cursor=dict(cursor),verified_records=0,
                bytes_read=0,prefix_anchor_authenticated=False,full_prefix_chain_verified=False,
                deployment_backup_accepted=False,source_write_requested=False,physical_io_cap_claimed=False)
    def guard():
        if clock()-started>=deadline_seconds or continue_check() is not True:
            raise _AdmissionStopped()
    def pairs(values):
        row={}
        for key,value in values:
            if key in row:raise ValueError('duplicate key')
            row[key]=value
        return row
    try:
        guard();path=Path(path);info=path.lstat()
        if not stat.S_ISREG(info.st_mode) or path.is_symlink() or getattr(info,'st_file_attributes',0)&0x400:
            raise ValueError('regular canonical file required')
        with path.open('rb') as stream:
            before=os.fstat(stream.fileno());extent=before.st_size
            if cursor['offset']>extent:raise ValueError('cursor past captured extent')
            if cursor['offset']:
                stream.seek(cursor['offset']-1);boundary=stream.read(1);read+=1
                if boundary!=b'\n':raise ValueError('cursor is not a record boundary')
            stream.seek(cursor['offset']);remaining=min(max_bytes-read,extent-cursor['offset']);chunks=[]
            while remaining:
                guard();block=stream.read(min(65536,remaining))
                if not block:raise ValueError('captured file truncated')
                chunks.append(block);read+=len(block);remaining-=len(block)
                delay=read/rate_bytes_per_second-(clock()-started)
                while delay>0:
                    guard();sleep(min(delay,.1));delay=read/rate_bytes_per_second-(clock()-started)
            after=os.fstat(stream.fileno())
        current=path.stat()
        if ((before.st_dev,before.st_ino)!=(after.st_dev,after.st_ino)
                or (before.st_dev,before.st_ino)!=(current.st_dev,current.st_ino)
                or min(after.st_size,current.st_size)<extent):
            raise ValueError('captured identity changed or file truncated')
        data=b''.join(chunks);position=0;sequence=cursor['sequence'];previous=cursor['last_hash']
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
                raise ValueError('chain mismatch')
            sequence+=1;previous=digest;position=end+1;checked+=1
        guard()
        result.update(status='BOUNDED_CHAIN_EXTENSION_VERIFIED' if checked else 'NO_COMPLETE_ROWS',
            next_cursor=dict(offset=cursor['offset']+position,sequence=sequence,last_hash=previous),
            verified_records=checked,captured_extent=extent,unconsumed_window_bytes=len(data)-position,
            record_limit_reached=checked==max_records and position<len(data),
            partial_trailing_row=bool(data) and not data.endswith(b'\n'),
            verified_segment_sha256=hashlib.sha256(data[:position]).hexdigest())
    except _AdmissionStopped:result.update(status='RESOURCE_BLOCKED',rows_checked_before_rejection=checked)
    except (OSError,ValueError,TypeError,KeyError,UnicodeError,RecursionError) as exc:
        result.update(status='CHAIN_WINDOW_REJECTED',error_type=type(exc).__name__,rows_checked_before_rejection=checked)
    result['bytes_read']=read
    return result

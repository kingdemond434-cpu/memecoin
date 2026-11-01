"""Read and authenticate one small staged segment against its historical receipt."""
import hashlib,json,math,os,stat,time
from pathlib import Path
from ops.canonical_recovery_plan import compose_recovery_plan
from ops.worker_resource_guard import verify
from src.research.trade_evidence import _canonical


def verify_staged_segment(receipt,*,continue_check,resource_check=verify,
                          rate_bytes_per_second=262144,deadline_seconds=60,
                          clock=time.monotonic,sleep=time.sleep):
    plan=compose_recovery_plan(receipt.get('start_cursor'),[receipt])
    item=plan['segments'][0]
    if (type(rate_bytes_per_second) is not int or not 1<=rate_bytes_per_second<=1048576
            or type(deadline_seconds) not in (int,float) or not math.isfinite(deadline_seconds)
            or not 0<deadline_seconds<=60 or plan['segment_records']>256):
        raise ValueError('Bounded segment policy required')
    cpu=resource_check()
    if cpu.get('status')!='VERIFIED' or cpu.get('cpu_rate')!=200:
        raise RuntimeError('Development CPU proof required')
    started=clock()
    def guard():
        if clock()-started>=deadline_seconds or continue_check() is not True:
            raise RuntimeError('Segment admission stopped')
    def identity(info):return info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns
    def pairs(values):
        result={}
        for key,value in values:
            if key in result:raise ValueError('Duplicate canonical key')
            result[key]=value
        return result
    guard();path=Path(item['path']);before=path.lstat()
    if (path.is_symlink() or not stat.S_ISREG(before.st_mode)
            or getattr(before,'st_file_attributes',0)&0x400 or before.st_size!=item['bytes']):
        raise ValueError('Exact regular staged segment required')
    chunks=[];read=0
    with path.open('rb') as stream:
        if identity(os.fstat(stream.fileno()))!=identity(before):raise ValueError('Segment identity changed')
        while read<item['bytes']:
            guard();block=stream.read(min(65536,item['bytes']-read))
            if not block:raise ValueError('Truncated segment')
            chunks.append(block);read+=len(block)
            delay=read/rate_bytes_per_second-(clock()-started)
            while delay>0:
                guard();sleep(min(.1,delay));delay=read/rate_bytes_per_second-(clock()-started)
        if identity(os.fstat(stream.fileno()))!=identity(before):raise ValueError('Segment changed during read')
    if identity(path.lstat())!=identity(before):raise ValueError('Segment path changed')
    body=b''.join(chunks)
    if hashlib.sha256(body).hexdigest()!=item['sha256'] or not body.endswith(b'\n'):
        raise ValueError('Segment digest or record boundary mismatch')
    previous=item['start_cursor']['last_hash'];sequence=item['start_cursor']['sequence']
    rows=body.split(b'\n')[:-1]
    if len(rows)!=plan['segment_records']:raise ValueError('Segment record count mismatch')
    for line in rows:
        guard();row=json.loads(line,object_pairs_hook=pairs);sequence+=1
        digest=row.pop('hash')
        if (type(row.get('sequence')) is not int or row['sequence']!=sequence
                or row.get('previous_hash')!=previous
                or hashlib.sha256(_canonical(row)).hexdigest()!=digest):
            raise ValueError('Segment row chain mismatch')
        previous=digest
    guard()
    if previous!=item['next_cursor']['last_hash']:raise ValueError('Segment end hash mismatch')
    return dict(status='SINGLE_STAGED_SEGMENT_CHAIN_VERIFIED',bytes_read=read,
        verified_records=len(rows),sha256=item['sha256'],start_cursor=item['start_cursor'],
        next_cursor=item['next_cursor'],elapsed_seconds=clock()-started,
        baseline_authenticated=False,full_prefix_chain_verified=False,
        recovery_verified=False,cross_store_checkpoint_verified=False,
        deployment_backup_accepted=False,source_write_requested=False,physical_io_cap_claimed=False)

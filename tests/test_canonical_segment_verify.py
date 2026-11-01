import hashlib,json
import pytest
from ops.canonical_segment_verify import verify_staged_segment
from src.research.trade_evidence import TradeEvidenceLedger,_canonical


def fixture(tmp_path):
    p=tmp_path/'fixture.jsonl';w=TradeEvidenceLedger(p);w.record('fixture',{'n':1})
    start=dict(offset=p.stat().st_size,sequence=1,last_hash=w.last_hash)
    w.record('fixture',{'n':2});w.record('fixture',{'n':3})
    body=p.read_bytes()[start['offset']:];s=tmp_path/'part.segment.pending';s.write_bytes(body)
    r=dict(status='SINGLE_CANONICAL_SEGMENT_READBACK_VERIFIED',pending_published=True,
        deployment_backup_accepted=False,start_cursor=start,next_cursor=dict(offset=p.stat().st_size,
        sequence=3,last_hash=w.last_hash),bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),pending_path=str(s))
    return r,s


def run(r,**kwargs):
    return verify_staged_segment(r,continue_check=lambda:True,
        resource_check=lambda:dict(status='VERIFIED',cpu_rate=200),**kwargs)


def test_real_chain_and_source_preservation(tmp_path):
    r,s=fixture(tmp_path);body=s.read_bytes();v=run(r)
    assert v['verified_records']==2 and s.read_bytes()==body
    assert not v['recovery_verified'] and not v['deployment_backup_accepted']


@pytest.mark.parametrize('kind',['digest','row','duplicate','boolean','end','partial'])
def test_corrupt_segment_or_receipt(tmp_path,kind):
    r,s=fixture(tmp_path);body=s.read_bytes()
    if kind=='digest':body=body.replace(b'fixture',b'fixturE')
    elif kind=='end':r['next_cursor']['last_hash']='f'*64
    elif kind=='partial':body=body[:-1]
    else:
        lines=body.splitlines();row=json.loads(lines[0]);row.pop('hash')
        if kind=='row':row['previous_hash']='e'*64
        if kind=='boolean':row['sequence']=True
        row['hash']=hashlib.sha256(_canonical(row)).hexdigest();lines[0]=_canonical(row)
        if kind=='duplicate':lines[0]=lines[0][:-1]+b',"sequence":2}'
        body=b'\n'.join(lines)+b'\n'
    s.write_bytes(body)
    if kind!='digest':r['sha256']=hashlib.sha256(body).hexdigest();r['bytes']=len(body);r['next_cursor']['offset']=r['start_cursor']['offset']+len(body)
    with pytest.raises(ValueError):run(r)


def test_admission_and_deadline(tmp_path):
    r,s=fixture(tmp_path)
    with pytest.raises(RuntimeError):verify_staged_segment(r,continue_check=lambda:False,resource_check=lambda:dict(status='VERIFIED',cpu_rate=200))
    with pytest.raises(ValueError):run(r,deadline_seconds=float('nan'))
    with pytest.raises(RuntimeError):run(r,clock=iter([0,61]).__next__)


def test_size_and_cpu_rejected(tmp_path):
    r,s=fixture(tmp_path);s.write_bytes(s.read_bytes()+b'X')
    with pytest.raises(ValueError):run(r)
    with pytest.raises(RuntimeError):verify_staged_segment(r,continue_check=lambda:True,resource_check=lambda:dict(status='VERIFIED',cpu_rate=500))


def test_mutation_during_admitted_read_is_rejected(tmp_path):
    r,s=fixture(tmp_path);calls=0
    def admitted():
        nonlocal calls
        calls+=1
        if calls==2:s.write_bytes(s.read_bytes()+b'X')
        return True
    with pytest.raises(ValueError):
        verify_staged_segment(r,continue_check=admitted,
            resource_check=lambda:dict(status='VERIFIED',cpu_rate=200))

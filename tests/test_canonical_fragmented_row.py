import hashlib,json
from pathlib import Path
import pytest
from ops.canonical_fragmented_row import verify_fragmented_row
from src.research.trade_evidence import TradeEvidenceLedger,GENESIS,_canonical
from test_canonical_raw_chunk import Timer

BASE=dict(offset=0,sequence=0,last_hash=GENESIS)
CPU=lambda:dict(status='VERIFIED',cpu_rate=200)

def fragments(tmp_path,raw=None):
    if raw is None:
        source=tmp_path/'canonical.jsonl'; writer=TradeEvidenceLedger(source)
        writer.record('fixture',{'text':'x'*1100000});writer.record('fixture',{'second':True})
        raw=source.read_bytes()
    receipts=[]
    for offset in range(0,len(raw),1048576):
        body=raw[offset:offset+1048576];p=tmp_path/(str(offset)+'.raw.pending');p.write_bytes(body)
        receipts.append(dict(status='SINGLE_RAW_CANONICAL_RANGE_READBACK_VERIFIED',offset=offset,
          end_offset=offset+len(body),captured_extent=len(raw),bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),pending_path=str(p)))
    return raw,receipts

def check(receipts,cursor=BASE,**kw):
    timer=Timer()
    return verify_fragmented_row(receipts,cursor=cursor,continue_check=kw.pop('continue_check',lambda:True),
        resource_check=kw.pop('resource_check',CPU),clock=timer.clock,sleep=timer.sleep,**kw)

def test_real_oversized_row_advances_exactly_one_row_preserving_overread(tmp_path):
    raw,rs=fragments(tmp_path);r=check(rs);end=raw.find(b'\n')+1;row=json.loads(raw[:end])
    assert end>1048576 and r['status']=='SINGLE_FRAGMENTED_CHAIN_ROW_VERIFIED'
    assert r['next_cursor']==dict(offset=end,sequence=1,last_hash=row['hash'])
    assert r['verified_row_sha256']==hashlib.sha256(raw[:end]).hexdigest()
    assert r['unconsumed_bytes']==len(raw)-end and r['bytes_read']==len(raw)
    assert not r['recovery_verified'] and not r['deployment_backup_accepted']
    assert b''.join(__import__('pathlib').Path(x['pending_path']).read_bytes() for x in rs)==raw

def test_first_fragment_incomplete_never_advances(tmp_path):
    raw,rs=fragments(tmp_path);r=check(rs[:1])
    assert r['status']=='NO_COMPLETE_FRAGMENTED_ROW' and r['next_cursor']==BASE
    assert r['raw_end_offset']==1048576

@pytest.mark.parametrize('kind',['gap','overlap','extent','digest','size','path','bool'])
def test_invalid_receipts_reject_no_cursor_progress(tmp_path,kind):
    _,rs=fragments(tmp_path)
    if kind=='gap':rs[1]['offset']+=1
    elif kind=='overlap':rs[1]['offset']-=1
    elif kind=='extent':rs[1]['captured_extent']+=1
    elif kind=='digest':rs[1]['sha256']='f'*64
    elif kind=='size':rs[1]['bytes']+=1
    elif kind=='path':rs[1]['pending_path']=rs[0]['pending_path']
    else:rs[0]['offset']=False
    r=check(rs);assert r['status']=='FRAGMENTED_ROW_REJECTED' and r['next_cursor']==BASE

@pytest.mark.parametrize('kind',['payload','previous','sequence','duplicate'])
def test_bad_chain_even_with_matching_raw_digest_rejects(tmp_path,kind):
    source=tmp_path/'c.jsonl';w=TradeEvidenceLedger(source);w.record('fixture',{'value':1})
    row=json.loads(source.read_bytes())
    if kind=='payload':row['payload']={'changed':True}
    elif kind=='previous':row['previous_hash']='f'*64
    elif kind=='sequence':row['sequence']=True
    raw=_canonical(row)+b'\n'
    if kind=='duplicate':raw=b'{"sequence":1,'+raw[1:]
    _,rs=fragments(tmp_path,raw);r=check(rs)
    assert r['status']=='FRAGMENTED_ROW_REJECTED' and r['next_cursor']==BASE

def test_resource_stop_does_not_advance(tmp_path):
    _,rs=fragments(tmp_path);r=check(rs,continue_check=lambda:False)
    assert r['status']=='RESOURCE_BLOCKED' and r['bytes_read']==0 and r['next_cursor']==BASE
    with pytest.raises(RuntimeError):check(rs,resource_check=lambda:dict(status='UNKNOWN'))

def test_changed_fragment_after_other_fragment_is_opened_rejects(tmp_path,monkeypatch):
    _,rs=fragments(tmp_path);first=Path(rs[0]['pending_path']);second=Path(rs[1]['pending_path'])
    original_open=Path.open;changed=[]
    def opened(path,*args,**kwargs):
        if path==second and args and args[0]=='rb':
            first.write_bytes(b'changed');changed.append(True)
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',opened)
    r=check(rs)
    assert changed and r['status']=='FRAGMENTED_ROW_REJECTED' and r['next_cursor']==BASE


def test_hardlink_alias_is_not_a_second_range(tmp_path):
    import os
    _,rs=fragments(tmp_path,b'x'*2097152)
    first=Path(rs[0]['pending_path']);second=Path(rs[1]['pending_path'])
    second.unlink();os.link(first,second)
    r=check(rs)
    assert r['status']=='FRAGMENTED_ROW_REJECTED' and r['next_cursor']==BASE


@pytest.mark.parametrize('policy',['too_many','bool_cursor','deadline'])
def test_policy_bounds_reject_before_read(tmp_path,policy):
    _,rs=fragments(tmp_path,b'abc')
    if policy=='too_many':
        with pytest.raises(ValueError):check(rs*3)
    elif policy=='bool_cursor':
        with pytest.raises(ValueError):check(rs,dict(BASE,sequence=False))
    else:
        with pytest.raises(ValueError):check(rs,deadline_seconds=61)

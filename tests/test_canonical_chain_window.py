import hashlib,json
import pytest
from ops.canonical_chain_window import verify_chain_window
from src.research.trade_evidence import TradeEvidenceLedger,GENESIS,_canonical


BASE=dict(offset=0,sequence=0,last_hash=GENESIS)
CPU=lambda:dict(status='VERIFIED',cpu_rate=200)


def ledger(tmp_path,count=3):
    path=tmp_path/'canonical.jsonl';writer=TradeEvidenceLedger(path)
    for value in range(count):writer.record('fixture',{'value':value})
    return path


def check(path,cursor=BASE,**kwargs):
    return verify_chain_window(path,cursor=cursor,continue_check=lambda:True,
                               resource_check=CPU,rate_bytes_per_second=1048576,**kwargs)


def test_real_writer_roundtrip_record_limited_cursor_preserves_all_rows(tmp_path):
    path=ledger(tmp_path);original=path.read_bytes();cursor=BASE
    for expected in range(1,4):
        result=check(path,cursor,max_records=1)
        assert result['status']=='BOUNDED_CHAIN_EXTENSION_VERIFIED'
        cursor=result['next_cursor'];assert cursor['sequence']==expected
        assert not result['full_prefix_chain_verified'] and not result['deployment_backup_accepted']
    assert cursor['offset']==len(original) and path.read_bytes()==original
    assert check(path,cursor)['status']=='NO_COMPLETE_ROWS'


def test_incomplete_append_stays_pending_until_committed(tmp_path):
    path=ledger(tmp_path,1);first=check(path)['next_cursor']
    writer=TradeEvidenceLedger(path);writer.record('fixture',{'value':2});body=path.read_bytes()
    path.write_bytes(body[:-1]);result=check(path,first)
    assert result['next_cursor']==first and result['partial_trailing_row']
    path.write_bytes(body);assert check(path,first)['next_cursor']['sequence']==2


@pytest.mark.parametrize('change',['payload','sequence','previous_hash','duplicate'])
def test_bad_second_row_never_advances_any_cursor(tmp_path,change):
    path=ledger(tmp_path,2);lines=path.read_bytes().splitlines(keepends=True)
    row=json.loads(lines[1])
    if change=='payload':row['payload']={'changed':True}
    elif change=='sequence':row['sequence']=True
    elif change=='previous_hash':row['previous_hash']='f'*64
    else:
        lines[1]=b'{"sequence":2,'+lines[1][1:];path.write_bytes(b''.join(lines))
        assert check(path)['next_cursor']==BASE;return
    if change!='payload':
        row.pop('hash');row['hash']=hashlib.sha256(_canonical(row)).hexdigest()
    lines[1]=_canonical(row)+b'\n';path.write_bytes(b''.join(lines))
    result=check(path);assert result['status']=='CHAIN_WINDOW_REJECTED' and result['next_cursor']==BASE


def test_mid_record_cursor_and_truncation_reject(tmp_path):
    path=ledger(tmp_path);cursor=dict(offset=10,sequence=1,last_hash='a'*64)
    assert check(path,cursor)['status']=='CHAIN_WINDOW_REJECTED'
    cursor['offset']=path.stat().st_size+1
    assert check(path,cursor)['next_cursor']==cursor


def test_lost_admission_leaves_cursor_unchanged(tmp_path):
    path=ledger(tmp_path);calls=0
    def admit():
        nonlocal calls
        calls+=1;return calls<4
    result=verify_chain_window(path,cursor=BASE,continue_check=admit,resource_check=CPU)
    assert result['status']=='RESOURCE_BLOCKED' and result['next_cursor']==BASE


def test_oversized_row_does_not_skip_to_a_later_boundary(tmp_path):
    path=tmp_path/'canonical.jsonl';path.write_bytes(b'x'*2048+b'\n')
    result=check(path,max_bytes=128)
    assert result['status']=='NO_COMPLETE_ROWS' and result['bytes_read']==128
    assert result['next_cursor']==BASE and result['partial_trailing_row']


def test_deadline_before_read_never_consumes_or_advances(tmp_path):
    path=ledger(tmp_path);ticks=iter((0.0,61.0))
    result=check(path,clock=lambda:next(ticks))
    assert result['status']=='RESOURCE_BLOCKED' and result['bytes_read']==0
    assert result['next_cursor']==BASE


@pytest.mark.parametrize('max_bytes',[True,1048577])
def test_unsupported_read_budget_rejected_before_io(tmp_path,max_bytes):
    path=tmp_path/'does-not-exist'
    with pytest.raises(ValueError):check(path,max_bytes=max_bytes)
